import os
import h5py
import numpy as np
import torch
import time
from tqdm import tqdm
from memory_profiler import profile
from sklearn.metrics.cluster import rand_score, adjusted_rand_score

from deconstruct.instance_segmentation.inference.stitching import calculate_viable_chunking, stitch_segmentations, \
    check_connectedness_in_label_volume
from deconstruct.utils.general import (add_time_and_date_to_filenames,
                                      extract_time_and_date_from_filename, remap_label_image, bincount_unique,
                                      shrink_dtype_label_arr)
from deconstruct.utils.calculating_overlaps import pairwise_overlap_of_segmentations_torch, \
    nonexclustive_iou_match_segments
from deconstruct.utils.visualization.colorize_label_img import save_label_image_to_rgb_uint8
from deconstruct.utils.visualization.segmentation_comparison import (save_colorized_segmentation_mistakes,
                                                                    save_single_color_segmentation_mistakes)
from elf.segmentation import GaspFromAffinities
from skimage.segmentation import watershed
from elf.segmentation.mutex_watershed import mutex_watershed


def compare_label_image_to_gt(label_image, target_image, ignore_background=True, bg_label=0):
    """
    both inputs as numpy
    """
    assert label_image.shape == target_image.shape, "shapes of prediction and target must match"
    # compare to ground truth segmentation:
    if ignore_background:
        bg_intersection = np.bitwise_and((target_image == bg_label), (label_image == bg_label))
        target_image = target_image[~bg_intersection]
        label_image = label_image[~bg_intersection]

    rand = rand_score(target_image.ravel(), label_image.ravel())
    arand = adjusted_rand_score(target_image.ravel(), label_image.ravel())

    return rand, arand


# @profile
def load_affinities_from_h5(path_h5_affinities, sl=None, use_torch=False, verbose=False):
    """if this uses to much RAM, devide the summed affinities by counted affinities CHUNKWISE before loading."""
    start = time.time()
    if sl is None:
        print("WARNING: loading all affinities from h5 file, this may take a lot of RAM.")
        ram_usage_summed = np.prod(h5py.File(path_h5_affinities, "r")["summed_affs"].shape) * 4 / 1024 ** 3
        ram_usage_counted = np.prod(h5py.File(path_h5_affinities, "r")["counted_affs"].shape) * 1 / 1024 ** 3
        print(f"expected ram consumption", np.round(ram_usage_summed + ram_usage_counted, 2), "GB.")
    sl = (slice(None),) if sl is None else sl
    with h5py.File(path_h5_affinities, "r") as f:
        if use_torch:
            # print("using torch")
            # division is faster in torch:
            with torch.no_grad():
                affs = torch.tensor(f["summed_affs"][sl])
                counted_affs = torch.tensor(f["counted_affs"][sl])
                affs /= torch.clamp(counted_affs, 1)
                affs = affs.numpy()
        else:
            # print("using numpy")
            # maximum is a bit faster than clip in numpy:
            affs = f["summed_affs"][sl] / np.maximum(f["counted_affs"][sl], 1)

    if verbose:
        print("loading affinities took", time.time() - start, "seconds.")
    return affs


class SmallClusterRemover:

    def __init__(self, minimum_cluster_size=400, afterwards_seeded_segmentation=True):
        self.minimum_cluster_size = minimum_cluster_size
        self.afterwards_seeded_segmentation = afterwards_seeded_segmentation

    @staticmethod
    def seeded_segmentation_to_fill_unassigned_segments(segmentation, unassigned_mask, background_label):
        # new version: let also the background segment grow:
        assert background_label != 0, f"expected background label to be != 0, background_label={background_label}"
        # set background to -1, leave unassigned zero:
        # segmentation[np.bitwise_and(segmentation == 0, ~unassign_mask)] = -1
        segmentation = watershed(
            image=np.zeros_like(segmentation),  # data where lowest values are labelled first
            markers=segmentation,  # seeds (background is also a seed, 0 means not a seed)
            mask=np.bitwise_or(unassigned_mask, segmentation > 0)  # points that will be labelled
        )

        # make sure that background becomes 0 again:
        segmentation[segmentation == background_label] = 0  # set background to 0
        return segmentation

    def __call__(self, segmentation, in_place=False):
        if not in_place:
            segmentation = segmentation.copy()

        labels, counts = bincount_unique(segmentation, return_counts=True)
        unassign_colors = labels[counts < self.minimum_cluster_size]

        # now use indexing trick to unassign labels:
        mapping_arr = np.arange(labels.max() + 1)
        mapping_arr[unassign_colors] = 0  # make unassigned label 0

        if self.afterwards_seeded_segmentation:
            # map background from 0 to to max label to be used as seed as well:
            mapping_arr[0] = labels.max() + 1  # setting background to no longer be 0
            segmentation = mapping_arr[segmentation]
            unassign_mask = (segmentation == 0)
            segmentation = self.seeded_segmentation_to_fill_unassigned_segments(segmentation, unassign_mask,
                                                                                background_label=labels.max() + 1)
        else:
            # just set unassigned voxels to be background:
            segmentation = mapping_arr[segmentation]

        assert segmentation.max() < labels.max() + 1, "temporary background label found in output"
        return segmentation


class AffinitySegmentor:
    gasp_config_defaults = {}
    mws_config_defaults = {"strides": [1, 1, 1]}

    def __init__(self, offsets, gasp_config=None, mws_config=None, remove_small_clusters_config=None,
                 relabel_result=True):
        self.offsets = offsets
        if gasp_config is None:
            # use pure mws:
            print("using pure mutex watershed")
            self.gasp = None
            mws_config = {} if mws_config is None else mws_config
            updated_mws_config = self.mws_config_defaults.copy()
            updated_mws_config.update(mws_config)
            self.mws_config = updated_mws_config
        else:
            print("warning: using gasp instead of pure mws. produced segments may not be connected.")
            updated_gasp_config = self.gasp_config_defaults.copy()
            updated_gasp_config.update(gasp_config)
            self.gasp = GaspFromAffinities(self.offsets, **updated_gasp_config)

        if remove_small_clusters_config is None:
            self.small_cluster_remover = None
        else:
            self.small_cluster_remover = SmallClusterRemover(**remove_small_clusters_config)

        self.relable_result = relabel_result

    @profile
    def __call__(self, affinities, verbose=False):
        fg_mask = affinities[-1] > 0.5  # foreground has target 1.
        affs = affinities[:-1]
        # print("the affinities are taking up ", np.round(affinities.nbytes / 1024 ** 3, 2), " GB of memory.")
        # print(f"aff_shape={affinities.shape} in theory they should be taking",
        #       np.round(np.prod(affinities.shape) * 4 / 1024 ** 3, 2), " GB of memory.")
        if self.gasp is None:
            segm = mutex_watershed(affs, self.offsets, mask=fg_mask, **self.mws_config)
        else:
            # very important here to use be 1 - affs to match gasp convention!
            # invert affs inplace:
            affs *= -1
            affs += 1
            segm, runtime = self.gasp(affs, foreground_mask=fg_mask)

        segm = shrink_dtype_label_arr(segm)

        if self.small_cluster_remover is not None:
            print("removing small clusters indeed")
            segm = self.small_cluster_remover(segm, in_place=True)

        if self.relable_result:
            segm = remap_label_image(segm)
        if verbose:
            print("number of segments: ", len(np.unique(segm)))

        return segm


class SegmentationStitcher:

    def __init__(self, path_h5_segmentation, iou_threshold=0.4, run_connected_components=True,
                 gt_segmentation=None, save_path_colored="default", save_path_colored_mistakes="default",
                 compare_only_direct_neighbor_chunks=True, final_remove_small_clusters_config=None, verbose=True):
        self.path_h5_segmentation = path_h5_segmentation
        self.iou_threshold = iou_threshold
        self.run_connected_components = run_connected_components
        self.compare_only_direct_chunk_neighbors = compare_only_direct_neighbor_chunks
        self.verbose = verbose
        self.gt_segmentation = gt_segmentation

        if final_remove_small_clusters_config is None:
            self.final_small_cluster_remover = None
        else:
            self.final_small_cluster_remover = SmallClusterRemover(**final_remove_small_clusters_config)

        self.save_path_colored = self.parse_save_path_colored(save_path_colored)
        self.save_path_colored_mistakes = self.parse_save_path_colored(save_path_colored_mistakes)
        self.save_path_colored_mistakes = self.save_path_colored_mistakes.replace("segmentation_colored",
                                                                                  "mistakes_colored")

    def parse_save_path_colored(self, save_path_colored):
        if save_path_colored == "default":
            # use same folder as path to affinities:
            save_path_colored = os.path.join(os.path.dirname(self.path_h5_segmentation), "segmentation_colored.raw")
        else:
            save_path_colored = save_path_colored

        if save_path_colored is not None:
            time_date_str = extract_time_and_date_from_filename(self.path_h5_segmentation)
            save_path_colored = add_time_and_date_to_filenames([save_path_colored],
                                                               time_date_str=time_date_str)[0]
        return save_path_colored

    def __call__(self):
        final_segmentation = stitch_segmentations(self.path_h5_segmentation, self.iou_threshold,
                                                  self.run_connected_components,
                                                  self.compare_only_direct_chunk_neighbors, verbose=self.verbose)

        if self.final_small_cluster_remover is not None:
            final_segmentation = self.final_small_cluster_remover(final_segmentation, in_place=True)
            print("after small cluster removal: ", len(np.unique(final_segmentation)), " segments left.")

        # relabel the image as some labels may get lost in the "overwritting" process:
        final_segmentation = remap_label_image(final_segmentation)

        mapping_labels_to_gt_labels = None
        if self.gt_segmentation is not None:
            rand, arand = compare_label_image_to_gt(final_segmentation, self.gt_segmentation)
            print("Final rand score: ", rand)
            print("Final ARand score: ", arand)

            # create a correspondence mapping between each segment in the final segmentation and the gt segmentation
            # based on iou and cut some which have to little overlap with any.
            _, iou_dict, _, gt_segm_counts = pairwise_overlap_of_segmentations_torch(final_segmentation,
                                                                                     self.gt_segmentation)
            mapping_labels_to_gt_labels = nonexclustive_iou_match_segments(iou_dict, iou_threshold=0.)

            if self.save_path_colored_mistakes is not None:
                # save_colorized_segmentation_mistakes(seg=final_segmentation, gt_seg=self.gt_segmentation,
                #                                      save_path=self.save_path_colored_mistakes)
                save_single_color_segmentation_mistakes(seg=final_segmentation, gt_seg=self.gt_segmentation,
                                                        save_path=self.save_path_colored_mistakes)

        with h5py.File(self.path_h5_segmentation, "r+") as f:
            if final_segmentation.max() > 2 ** 16 - 1:
                print("WARNING: final segmentation has more than 2**16 - 1 labels, will be truncated to uint16.")
            if "final_segmentation" in f:
                print(f"Overwriting final segmentation in h5 file {self.path_h5_segmentation}.")
                del f["final_segmentation"]
            f.create_dataset("final_segmentation", data=final_segmentation, dtype=np.uint16, compression="gzip")
            if "label_gtlabel_iou" in f:
                del f["label_gtlabel_iou"]
            # make this a dataset cause this may be quite large:
            if self.gt_segmentation is not None:
                f.attrs["rand"] = rand
                f.attrs["arand"] = arand
                f.create_dataset("label_gtlabel_iou", data=[(label, gt_label, iou) for label, (gt_label, iou) in
                                                            mapping_labels_to_gt_labels.items()])

        if self.save_path_colored is not None:
            save_label_image_to_rgb_uint8(final_segmentation, save_path=self.save_path_colored, seed=0)

        # check that final segmentation contains only connected components:
        check_connectedness_in_label_volume(final_segmentation)

        return final_segmentation


class SegmentationPrediction:
    sigmoid = torch.nn.Sigmoid()

    def __init__(self, path_h5_affinities, segm_chunk_shape, segm_overlap_shape,
                 aff_segmentor_config, segmentation_stitcher_config, gt_segmentation=None,
                 path_h5_segmentation="default", verbose=True):
        """
        config contains all information in the inference config.
        load model and dataset
        """

        # io paths:
        if path_h5_segmentation == "default":
            # use same folder as path to affinities:
            path_h5_segmentation = os.path.join(os.path.dirname(path_h5_affinities), "segmentations.h5")

        # add time and date from h5_affinities (if path is None, it stays None):
        time_date_str = extract_time_and_date_from_filename(path_h5_affinities)
        path_h5_segmentation = add_time_and_date_to_filenames([path_h5_segmentation],
                                                              time_date_str=time_date_str)[0]

        self.verbose = verbose
        self.path_h5_affinities = path_h5_affinities
        self.path_h5_segmentation = path_h5_segmentation
        self.segm_chunk_shape = tuple(segm_chunk_shape)
        self.segm_overlap_shape = tuple(segm_overlap_shape)
        self.gt_segmentation = gt_segmentation

        # init segmentor:
        segmentation_stitcher_config["verbose"] = verbose
        with h5py.File(self.path_h5_affinities, "r") as f:
            self.run_name = f.attrs["run_name"] if "run_name" in f.attrs else "unknown"
            self.img_shape = f["summed_affs"].shape[1:]
            aff_segmentor_config["offsets"] = f.attrs["offsets"]

        self.segmentor = AffinitySegmentor(**aff_segmentor_config)
        self.stitcher = SegmentationStitcher(self.path_h5_segmentation, gt_segmentation=gt_segmentation,
                                             **segmentation_stitcher_config)

        self.list_slices, self.all_left_corners = self.create_slicing_of_volume()

    def create_slicing_of_volume(self):
        """create slicing of volume"""
        all_left_corners, _ = calculate_viable_chunking(img_shape=self.img_shape,
                                                        chunk_shape=self.segm_chunk_shape,
                                                        overlap_shape=self.segm_overlap_shape)

        # create a list of slices:
        list_slices = []
        for left_corner in all_left_corners:
            list_slices.append(tuple([slice(left_corner[i], left_corner[i] + s)
                                      for i, s in enumerate(self.segm_chunk_shape)]))

        return list_slices, all_left_corners

    def run_segmentation_on_single_chunk(self, aff_chunk):
        """predict segmentation on single chunk"""
        return self.segmentor(aff_chunk, verbose=self.verbose)

    def run_chunkwise_segmentation(self):
        """makes all segmentation predicitons and stores them in an h5 dataset"""

        # can not have all predictions in memory at once, init h5 dataset:
        with h5py.File(self.path_h5_segmentation, "w") as f:
            f.attrs['img_shape'] = self.img_shape
            f.attrs['chunk_shape'] = self.segm_chunk_shape
            f.attrs['overlap_shape'] = self.segm_overlap_shape
            f.attrs['run_name'] = self.run_name
            f.attrs['path_h5_affinities'] = self.path_h5_affinities

        rand_score_list = []
        arand_score_list = []
        for i, sl in (enumerate(tqdm(self.list_slices)) if self.verbose else enumerate(self.list_slices)):
            # get aff chunk:
            sl_affs = (slice(None),) + sl
            aff_chunk = load_affinities_from_h5(self.path_h5_affinities, sl=sl_affs, use_torch=True,
                                                verbose=self.verbose)

            segmentation = self.run_segmentation_on_single_chunk(aff_chunk)
            check_connectedness_in_label_volume(segmentation)

            # evaluate result:
            if self.gt_segmentation is not None:
                rand, arand = compare_label_image_to_gt(segmentation, self.gt_segmentation[sl])
                rand_score_list.append(rand)
                arand_score_list.append(arand)
                print("Rand score: ", rand, "ARand score: ", arand)

            # append to h5:
            with h5py.File(self.path_h5_segmentation, "r+") as f:
                if segmentation.max() > 2 ** 16 - 1:
                    print(f"WARNING: chunk {i} has more than 2**16 - 1 labels, will be truncated to uint16.")
                grp = f.create_group("chunk" + str(i))
                grp.attrs["min_corner"] = self.all_left_corners[i]
                grp.create_dataset("segmentation", data=segmentation, dtype=np.uint16, compression="gzip")

        print("Done with chunkwise segmentation, see: ", self.path_h5_segmentation)
        if self.gt_segmentation is not None:
            print("Average Rand score: ", np.mean(rand_score_list))
            print("Average ARand score: ", np.mean(arand_score_list))

    def stitch_segmentations(self):
        return self.stitcher()

    def __call__(self):
        self.run_chunkwise_segmentation()
        return self.stitch_segmentations()
