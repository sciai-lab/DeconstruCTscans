import numpy as np
import networkx as nx
import h5py
from tqdm import tqdm

from deconstruct.utils.general import (connected_components_with_label_mapping, remap_label_image)
from deconstruct.utils.calculating_overlaps import bbox_overlap, overlap_to_slices, \
    pairwise_overlap_of_segmentations_torch


def get_chunk_group_names(path_h5_segmentation):
    with h5py.File(path_h5_segmentation, "r") as f:
        return [group_name for group_name in f.keys() if group_name.startswith("chunk")]


# def calculate_viable_chunking(img_shape, chunk_shape, overlap_shape, shrinking_shape=None,
#                                              smaller_last_chunk_for_exact_overlap=False):
#     """
#     Calculates how to chunk img/volume into chunks of a given shape
#     Convention: use larger overlap for the last chunk to make everything fit.
#
#     Parameters
#     ----------
#     img_shape : tuple[int]
#     chunk_shape : tuple[int]
#     overlap_shape : tuple[int]
#
#     Returns
#     -------
#     np.ndarray, list[np.ndarray]
#         chunking indices as flatted meshgrid
#         list of indices in each dimension
#     """
#     assert len(img_shape) == len(chunk_shape) == len(overlap_shape), "inputs don't match."
#
#     if (np.asarray(img_shape) < np.asarray(chunk_shape)).any():
#         chunk_shape = tuple([min(img_shape[i], s) for i, s in enumerate(chunk_shape)])
#         print(f"Changing the chunk shape to be at most the vol shape: {chunk_shape=}")
#     assert (np.asarray(overlap_shape) < np.asarray(chunk_shape)).all(), "overlap must be smaller than chunk."
#
#     dim = len(img_shape)
#     list_of_slicing_in_different_dims = []
#     shrinking_shape = (0,) * dim if shrinking_shape is None else shrinking_shape
#     effective_chunk_shape = tuple([s - 2 * p for s, p in zip(chunk_shape, shrinking_shape)])
#     assert (np.asarray(effective_chunk_shape) > 0).all(), "shrinking shape is too large for given chunk shape."
#
#     print("Using the following shapes for affinity prediction: chunkshape", chunk_shape,
#           "shrinking_shape", shrinking_shape, "effective_chunk_shape", effective_chunk_shape,
#           "overlap_shape", overlap_shape, "img_shape", img_shape)
#
#     for i in range(dim):
#         L = img_shape[i]  # size in this dimension
#         e = effective_chunk_shape[i]
#         s = chunk_shape[i]  # actual chunk size
#         o = overlap_shape[i]  # desired overlap
#         index_array = np.arange(0, L - e + 1, e - o)  # get chunks at steps s-o
#         when_too_large = (index_array >= L - s)
#         index_array = index_array[~when_too_large]
#         # add last chunk:
#         if smaller_last_chunk_for_exact_overlap:
#             # last chunk has overlap o but smaller size
#             index_array = np.append(index_array, index_array[-1] + e - o)
#         else:
#             index_array = np.append(index_array, L - s)
#         list_of_slicing_in_different_dims.append(index_array)
#
#     print("list_of_slicing_in_different_dims", list_of_slicing_in_different_dims)
#     meshgrid = np.meshgrid(*list_of_slicing_in_different_dims)
#     index_matrix = np.stack(meshgrid, axis=dim)  # e.g.(len(listx), len(listy), len(listz), dim)
#     complete_index_arr = index_matrix.reshape((-1, dim))
#
#     return complete_index_arr, list_of_slicing_in_different_dims


def calculate_viable_chunking(img_shape, chunk_shape, overlap_shape, shape_of_last_chunk=None,
                              smaller_last_chunk_for_exact_overlap=False):
    """
    Calculates how to chunk img/volume into chunks of a given shape
    Convention: use larger overlap for the last chunk to make everything fit.

    Parameters
    ----------
    img_shape : tuple[int]
    chunk_shape : tuple[int]
    overlap_shape : tuple[int]

    Returns
    -------
    np.ndarray, list[np.ndarray]
        chunking indices as flatted meshgrid
        list of indices in each dimension
    """
    assert len(img_shape) == len(chunk_shape) == len(overlap_shape), "inputs don't match."

    if (np.asarray(img_shape) < np.asarray(chunk_shape)).any():
        chunk_shape = tuple([min(img_shape[i], s) for i, s in enumerate(chunk_shape)])
        print(f"Changing the chunk shape to be at most the vol shape: {chunk_shape=}")
    assert (np.asarray(overlap_shape) < np.asarray(chunk_shape)).all(), "overlap must be smaller than chunk."

    dim = len(img_shape)
    list_of_slicing_in_different_dims = []
    shape_of_last_chunk = chunk_shape if shape_of_last_chunk is None else shape_of_last_chunk
    for i in range(dim):
        L = img_shape[i]  # size in this dimension
        s = chunk_shape[i]  # desired chunk size
        o = overlap_shape[i]  # desired overlap
        index_array = np.arange(0, L - s + 1, s - o)  # get chunks at steps s-o

        if index_array[-1] != L - s:
            if smaller_last_chunk_for_exact_overlap:
                # last chunk has overlap o but smaller size
                index_array = np.append(index_array, index_array[-1] + s - o)
            else:
                # last chunk has size s but more overlap
                index_array = np.append(index_array, L - shape_of_last_chunk[i])  # append last chunk
            # print("overlap", s - (index_array[-1] - index_array[-2]))

        list_of_slicing_in_different_dims.append(index_array)

    # print("list_of_slicing_in_different_dims", list_of_slicing_in_different_dims)
    meshgrid = np.meshgrid(*list_of_slicing_in_different_dims)
    index_matrix = np.stack(meshgrid, axis=dim)  # e.g.(len(listx), len(listy), len(listz), dim)
    complete_index_arr = index_matrix.reshape((-1, dim))

    return complete_index_arr, list_of_slicing_in_different_dims


def stitch_two_segments(vol1, vol2, min_corner1, min_corner2, overlap, iou_threshold=0.4,
                        run_connected_components=True, background_label=0):
    sl1, sl2 = overlap_to_slices(min_corner1, min_corner2, overlap, prescribed_overlap_shape=None)
    overlap1 = vol1[sl1]
    overlap2 = vol2[sl2]

    if run_connected_components:
        overlap1, label_mapping_dict1, _ = connected_components_with_label_mapping(overlap1)
        overlap2, label_mapping_dict2, _ = connected_components_with_label_mapping(overlap2)
    else:
        label_mapping_dict1 = None
        label_mapping_dict2 = None

    # intersection over union on the overlaps:
    _, iou_dict, _, _ = pairwise_overlap_of_segmentations_torch(overlap1, overlap2)

    edges_to_add = []
    for label_pair, iou in iou_dict.items():
        if iou >= iou_threshold:
            # add edge
            if run_connected_components:
                # map back to original labels:
                label_pair = (label_mapping_dict1[label_pair[0]], label_mapping_dict2[label_pair[1]])

            if label_pair[0] == background_label or label_pair[1] == background_label:
                # don't merge anything with background
                continue
            else:
                edges_to_add.append(label_pair)

    return edges_to_add


def stitch_segmentations(path_h5_segmentations, iou_threshold=0.4, run_connected_components=True,
                         compare_only_direct_chunk_neighbors=True, verbose=True):
    node_list = []
    list_chunk_group_names = get_chunk_group_names(path_h5_segmentations)
    with h5py.File(path_h5_segmentations, "r") as f:
        img_shape = f.attrs["img_shape"]
        chunk_shape = f.attrs["chunk_shape"]

        list_bboxes = []
        for i, group_name in enumerate(list_chunk_group_names):
            min_corner = f[group_name].attrs["min_corner"]
            list_bboxes.append((min_corner, min_corner + np.asarray(chunk_shape)))
            node_list += [(i, ind) for ind in range(1, np.max(f[group_name]["segmentation"][:]) + 1)]  # no background

    # first identify all pairs of segments that overlap:
    overlapping_pairs = []
    dim = len(img_shape)
    for i, bbox1 in enumerate(list_bboxes):
        for j, bbox2 in enumerate(list_bboxes[i + 1:]):
            overlap = bbox_overlap(bbox1, bbox2)
            if overlap is None:
                continue
            if compare_only_direct_chunk_neighbors:
                # check that the bbox is the same in 2 out of 3 dims:
                if np.sum((bbox1[0] - bbox2[0]) == 0) != dim - 1 or np.sum((bbox1[1] - bbox2[1]) == 0) != dim - 1:
                    continue
            overlapping_pairs.append((i, j + i + 1, overlap))

    segment_identifier_graph = nx.Graph()
    segment_identifier_graph.add_nodes_from(node_list)  # add all labels in all segments as nodes
    for pair in (tqdm(overlapping_pairs) if verbose else overlapping_pairs):
        i, j, overlap = pair

        # load the two volumes:
        with h5py.File(path_h5_segmentations, "r") as f:
            group1 = f[list_chunk_group_names[i]]
            group2 = f[list_chunk_group_names[j]]
            min_corner1 = group1.attrs["min_corner"]
            min_corner2 = group2.attrs["min_corner"]
            vol1 = group1["segmentation"][:]
            vol2 = group2["segmentation"][:]
            edges_to_add = stitch_two_segments(vol1, vol2, min_corner1, min_corner2, overlap, iou_threshold,
                                               run_connected_components)
            # add chunk labels:
            edges_to_add = [((i, edge[0]), (j, edge[1])) for edge in edges_to_add]
            segment_identifier_graph.add_edges_from(edges_to_add)

    # run connected components on the segment_identifier_graph to build global mapping dict:
    connected_label_mapping_dict = {}
    for i, comp in enumerate(nx.connected_components(segment_identifier_graph)):
        for node in comp:
            connected_label_mapping_dict[node] = i + 1  # 0 is background

    # build the final segmentation:
    final_segmentation = np.zeros(img_shape, dtype=np.uint16)
    # uncovered_mask = np.ones(img_shape, dtype=bool)  # check that all pixels are covered
    with h5py.File(path_h5_segmentations, "r") as f:
        for i, group_name in enumerate(list_chunk_group_names):
            min_corner = f[group_name].attrs["min_corner"]
            segmentation = f[group_name]["segmentation"][:]

            check_connectedness_in_label_volume(segmentation)

            # build mapping arr (indexing trick):
            label_mapping_arr = np.zeros(np.max(segmentation) + 1, dtype=int)  # + 1 bc of background
            for ind in range(1, np.max(segmentation) + 1):
                label_mapping_arr[ind] = connected_label_mapping_dict[(i, ind)]

            # fill in only where there are zeros (so that no segment will be half overwritten):
            sl = tuple([slice(min_corner[k], min_corner[k] + s) for k, s in enumerate(segmentation.shape)])
            # sl_uncovered_mask = uncovered_mask[sl]
            bg_mask = (final_segmentation[sl] == 0)
            final_segmentation[sl][bg_mask] = label_mapping_arr[segmentation][bg_mask]
            # uncovered_mask[sl] = False  # this region has now been covered

    print("final segm number of segments=", len(np.unique(final_segmentation)))
    # assert np.sum(uncovered_mask) == 0, "volume must be completely covered by the chunks."

    return final_segmentation


def check_connectedness_in_label_volume(label_volume, background_label=0, connectivity=1):
    # label_mapping_dict: {new_label: label in label_volume}
    new_labels, label_mapping_dict, new_labels_count_dict = connected_components_with_label_mapping(label_volume,
                                                                                                    background_label,
                                                                                                    connectivity)
    inverse_label_mapping_dict = {}
    for new_label, old_label in label_mapping_dict.items():
        if old_label in inverse_label_mapping_dict:
            inverse_label_mapping_dict[old_label].append(new_label)
        else:
            inverse_label_mapping_dict[old_label] = [new_label]

    all_connected = True
    for old_label in inverse_label_mapping_dict:
        if len(inverse_label_mapping_dict[old_label]) > 1:
            all_connected = False
            print(f"WARNING: label {old_label} is not connected, "
                  f"but split into {len(inverse_label_mapping_dict[old_label])} segments of sizes:"
                  f"{[new_labels_count_dict[new_label] for new_label in inverse_label_mapping_dict[old_label]]}")

    if all_connected:
        print("All segments are indeed connected.")
