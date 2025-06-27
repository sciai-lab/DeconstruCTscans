import os
import h5py
import yaml
import numpy as np
import shutil
from scipy.ndimage import binary_dilation

from deconstruct.utils.general import create_dirs, connected_components_with_label_mapping
from deconstruct.data_generation.part_catalog import PartCatalog
from deconstruct.instance_segmentation.inference.affinity_prediction import (AffinityPrediction)
from deconstruct.utils.visualization.colorize_label_img import save_label_image_to_rgb_uint8


def threshold_clip_real_scan_segmentation(path_raw_scan, path_h5_segmentation, iso_threshold, num_dilations=20,
                                          create_copy=True, save_colored_segmentation=False):
    # first create a copy of the h5 segmentation file:
    if create_copy:
        copy_path = path_h5_segmentation.replace(".h5", "_original.h5")
        shutil.copyfile(path_h5_segmentation, copy_path)
        print("creating copy of original segmentation file at", copy_path)

    # threshold the raw scan:
    raw_scan = np.load(path_raw_scan)
    iso_mask = (raw_scan > iso_threshold)
    dilated_isomask = binary_dilation(iso_mask, iterations=num_dilations)

    print("Computing connected components of dilated iso mask.")
    component_labeled_volume, label_mapping_dict, new_label_counts_dict = connected_components_with_label_mapping(
        dilated_isomask.astype(int), background=0)
    print("Done with conneceted components computation.")

    # use largest connected component as fg mask:
    label_list = list(new_label_counts_dict.keys())
    label_list.remove(0)  # remove background label
    print("label_list", label_list, "new_label_counts_dict", new_label_counts_dict)
    counts = [new_label_counts_dict[label] for label in label_list]
    largest_label = label_list[np.argmax(counts)]

    largest_dilated_iso_segment = (component_labeled_volume == largest_label)

    # load segmentation and apply mask:
    with h5py.File(path_h5_segmentation, "r+") as f:
        f.attrs["iso_threshold"] = iso_threshold
        f.attrs["num_dilations"] = num_dilations
        final_segmentation = f["final_segmentation"][:]
        final_segmentation[~largest_dilated_iso_segment] = 0

        # have to delete and recreate the dataset:
        del f["final_segmentation"]
        f.create_dataset("final_segmentation", data=final_segmentation, dtype=np.uint16,
                         compression="gzip")
        f.create_dataset("largest_dilated_iso_segment", data=2 * largest_dilated_iso_segment, dtype=np.uint8,
                         compression="gzip")

    if save_colored_segmentation:
        save_label_image_to_rgb_uint8(final_segmentation, path_h5_segmentation.replace(".h5", "_colored.raw"))

    print("Done with threshold clipping real scan segmentation.")


class AffinityPredictionFromRaw(AffinityPrediction):

    def __init__(self, config, path_raw_input_config, path_h5_affinities="default", rescale_input=True,
                 overlap_shape=None, model_loading_kwargs=None, verbose=True, **super_kwargs):
        """
        expects path_raw_input_volume to be an npy file with a uint16 dtype volume
        which is standard for ct scans reconstructed in vg max
        super_kwargs will be ignored, but contain the "path_h5_dataset" argument which is usually provided
        """
        with open(path_raw_input_config, "r") as f:
            self.raw_input_config = yaml.load(f, yaml.SafeLoader)

        self.path_raw_input_volume = self.raw_input_config["path_raw_input_volume"]
        assert os.path.exists(self.path_raw_input_volume), f"Path {self.path_raw_input_volume} does not exist."
        assert self.path_raw_input_volume.endswith(".npy"), f"Path {self.path_raw_input_volume} does not end with .npy."

        raw_input_volume = np.load(self.path_raw_input_volume)
        if path_h5_affinities in ["default", "iwr_data", "iwr_scratch"]:
            path_h5_affinities = self.path_raw_input_volume.replace(".npy", "_affinities.h5")

        # create a pseudo h5 dataset:
        if verbose:
            print(f"Creating pseudo h5 dataset at {os.path.dirname(path_h5_affinities)}.")
        path_h5_pseudo_dataset = os.path.join(os.path.dirname(path_h5_affinities), "pseudo_dataset.h5")
        create_dirs([os.path.dirname(path_h5_pseudo_dataset)])
        with h5py.File(path_h5_pseudo_dataset, "w") as f:
            f.create_dataset("raw_input_volume", data=raw_input_volume, dtype=np.uint16)  # no compression
            f.attrs["raw_min"] = raw_input_volume.min()
            f.attrs["raw_max"] = raw_input_volume.max()

        super().__init__(config=config, path_h5_dataset=path_h5_pseudo_dataset,
                         path_h5_affinities=path_h5_affinities, rescale_input=rescale_input, gt_is_given=False,
                         overlap_shape=overlap_shape, model_loading_kwargs=model_loading_kwargs, verbose=verbose)

    def create_pseudo_part_catalog(self):
        # init stl catalog:
        print("Creating pseudo part catalog from raw input volume.")
        part_catalog = PartCatalog(**self.raw_input_config["part_catalog_config"])

        # manually set voxelization scale:
        part_catalog.voxelization_scale = self.raw_input_config["voxelization_scale"]

        # generate voxelized catalog:
        part_catalog.generate_voxelized_catalog(**self.raw_input_config["generate_voxelized_catalog_config"])
        print("Completed dataset and voxelized catalog generation.")

        part_catalog.save_part_catalog_for_later(save_path=os.path.join(os.path.dirname(self.path_raw_input_volume),
                                                                        "pseudo_part_catalog.pkl"))


path_raw_scan = "fill_in"
path_h5_segmentation = "fill_in"
if __name__ == '__main__':
    threshold_clip_real_scan_segmentation(path_raw_scan, path_h5_segmentation, iso_threshold=23570, num_dilations=25,
                                          save_colored_segmentation=True)

