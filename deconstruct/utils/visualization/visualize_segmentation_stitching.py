import h5py
import numpy as np
from tqdm import tqdm

from deconstruct.utils.visualization.colorize_label_img import save_label_image_to_rgb_uint8
from deconstruct.instance_segmentation.inference.stitching import get_chunk_group_names


def visualize_segmentation_before_stitching(path_h5_segmentations, save_mistakes=True, margin=5,
                                            save_path="colored_segmentation_before_stitching.raw"):

    list_grp_names = get_chunk_group_names(path_h5_segmentations)
    with h5py.File(path_h5_segmentations, "r") as f:
        chunk_shape = f.attrs["chunk_shape"]

        # compute overall volume shape:
        # need pattern for arranging the chunks based on min_corners:
        all_min_corners = []
        for grp_name in tqdm(list_grp_names, desc="collecting min corners"):
            min_corner = f[grp_name].attrs["min_corner"]
            all_min_corners.append(min_corner)

        # sort min corners by x, then y, then z:
        # sort_indices = np.lexsort((all_min_corners[:, 2], all_min_corners[:, 1], all_min_corners[:, 0]))
        # all_min_corners = all_min_corners[sort_indices]

        # how many chunks in each dimension:
        all_min_corners = np.asarray(all_min_corners)
        chunks_per_dim = [np.unique(all_min_corners[:, i]).shape[0] for i in range(all_min_corners.shape[1])]
        padded_chunk_shape = np.asarray(chunk_shape) + margin
        list_corner_mappings = []
        for i in range(all_min_corners.shape[1]):
            unique_corners = np.sort(np.unique(all_min_corners[:, i]))
            corner_mapping = {c: padded_chunk_shape[i] * j for j, c in enumerate(unique_corners)}
            list_corner_mappings.append(corner_mapping)

        # compute overall volume shape:
        vol_shape = np.asarray(chunks_per_dim) * np.asarray(padded_chunk_shape)
        vol = np.zeros(vol_shape, dtype=np.uint16)

        # fill the volume:
        min_label = 0
        for grp_name in tqdm(list_grp_names):
            min_corner = f[grp_name].attrs["min_corner"].copy()
            for i in range(len(min_corner)):
                # add margin as many times as needed depending on the position of the chunk:
                min_corner[i] = list_corner_mappings[i][min_corner[i]]

            # add margin to segmentation:
            segmentation = f[grp_name]["segmentation"][:]
            segmentation = np.pad(segmentation, (0, margin), mode="constant", constant_values=0)

            sl = tuple([slice(min_corner[i], min_corner[i] + padded_chunk_shape[i]) for i in range(3)])
            segmentation[segmentation > 0] += min_label
            vol[sl] = segmentation
            min_label = np.max(vol)

    # save the volume to a colored image:
    print("visualizing started.")
    save_label_image_to_rgb_uint8(vol, save_path=save_path)