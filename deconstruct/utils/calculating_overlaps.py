import numpy as np
import torch
import z5py
from tqdm import tqdm
from skimage.morphology import binary_erosion


def get_mask_and_corner_from_z5(path_z5_dataset, group_name, dict_z5_min_corners_and_masks=None):
    """if dict_z5_min_corners_and_masks is not None they are all in memory."""
    if dict_z5_min_corners_and_masks is None:
        with z5py.File(path_z5_dataset, "r") as f:
            if group_name not in f:
                return None, None
            proposal_mask = f[group_name]["foreground_mask"][:].astype(bool)
            proposal_min_corner = f[group_name]["min_corner"][:]
    else:
        if group_name not in dict_z5_min_corners_and_masks:
            return None, None
        # print("loading from memory")
        proposal_mask, proposal_min_corner = dict_z5_min_corners_and_masks[group_name]

    return proposal_mask, proposal_min_corner


import time
def calculate_pairwise_overlap_from_z5(list_group_names, path_z5_masks, perform_k_many_erosions=0,
                                       divide_by_smaller_volume=False, dict_z5_min_corners_and_masks=None):
    """
    This could be parallelized.
    path_z5_masks: path to z5 file with masks.
    """

    overlap_matrix = np.zeros((len(list_group_names),) * 2)
    all_sizes_arr = np.zeros(len(list_group_names))

    # if all is in memory do the erosion first:
    eroded_dict_z5_min_corners_and_masks = None
    if dict_z5_min_corners_and_masks is not None:
        print("performing all erosions...")
        eroded_dict_z5_min_corners_and_masks = {}
        for group_name in tqdm(list_group_names):
            mask, min_corner = dict_z5_min_corners_and_masks[group_name]
            eroded_mask = perform_erosions(mask, k=perform_k_many_erosions)
            eroded_dict_z5_min_corners_and_masks[group_name] = (eroded_mask, min_corner)

    for i, group_name1 in enumerate(tqdm(list_group_names)):
        if eroded_dict_z5_min_corners_and_masks is None:
            segment1, min_corner1 = get_mask_and_corner_from_z5(path_z5_masks, group_name1)
            segment1 = perform_erosions(segment1, k=perform_k_many_erosions)
        else:
            segment1, min_corner1 = get_mask_and_corner_from_z5(path_z5_masks, group_name1,
                                                                eroded_dict_z5_min_corners_and_masks)

        all_sizes_arr[i] = np.sum(segment1)
        if segment1 is None:
            continue

        for j, group_name2 in enumerate(list_group_names):
            if i <= j:
                continue

            if eroded_dict_z5_min_corners_and_masks is None:
                segment2, min_corner2 = get_mask_and_corner_from_z5(path_z5_masks, group_name2)
                segment2 = perform_erosions(segment2, k=perform_k_many_erosions)
            else:
                segment2, min_corner2 = get_mask_and_corner_from_z5(path_z5_masks, group_name2,
                                                                    eroded_dict_z5_min_corners_and_masks)

            if segment2 is None:
                continue

            overlap_matrix[i, j] = overlap_between_segments(segment1, segment2, min_corner1, min_corner2)

    if divide_by_smaller_volume:
        minimum_size_matrix = np.minimum(all_sizes_arr, all_sizes_arr[:, None])
        none_mask = (overlap_matrix == -1)
        overlap_matrix[~none_mask] = overlap_matrix[~none_mask] / np.maximum(minimum_size_matrix[~none_mask], 1)

    # make symmetric:
    overlap_matrix = overlap_matrix + overlap_matrix.T

    return overlap_matrix


def perform_erosions(mask, k):
    """
    this does the same as the following:
    from scipy.ndimage import binary_dilation, binary_erosion
    structure = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ]) has to be generalized for 3d

    img2 = binary_erosion(img_, iterations=k, structure=structure, border_value=1)
    """
    if k == 0:
        return mask
    eroded_mask = mask.copy()
    for _ in range(k):
        eroded_mask = binary_erosion(eroded_mask)

    return eroded_mask


def bbox_overlap(bbox1, bbox2, min_overlap_shape=None):
    """
    could be parallelized.
    """
    min_corner1, max_corner1 = bbox1
    min_corner2, max_corner2 = bbox2
    dim = len(min_corner1)
    for i in range(dim):
        if max_corner1[i] < min_corner2[i] or max_corner2[i] < min_corner1[i]:
            # segments don't overlap
            return None

    # find intersection bbox:
    min_intersect_corner = np.maximum(min_corner1, min_corner2)
    max_intersect_corner = np.minimum(max_corner1, max_corner2)

    if min_overlap_shape is not None:
        if (max_intersect_corner - min_intersect_corner - np.asarray(min_overlap_shape) < 0).any():
            print(f"overlap between bboxes {bbox1, bbox2} is smaller than {min_overlap_shape}.")
            return None

    return (min_intersect_corner, max_intersect_corner)


def overlap_to_slices(min_corner1, min_corner2, overlap, prescribed_overlap_shape=None):
    """
    For my purposes prescribed_overlap_shape must be None.
    The overlaps are chosen in the chunking such that they are accurate.
    """
    overlap = np.asarray(overlap)
    overlap_shape = overlap[1] - overlap[0]
    if prescribed_overlap_shape is not None:
        shape_diff = np.asarray(overlap_shape) - np.asarray(prescribed_overlap_shape)
        assert (shape_diff >= 0).all(), \
            f"overlap must not be smaller than prescribed {overlap_shape, prescribed_overlap_shape}"
        for i, s in enumerate(shape_diff):
            if s > 0:
                shift = int(s / 2)
                overlap[0][i] += shift
        overlap[1] = overlap[0] + prescribed_overlap_shape

    overlap_sl1 = tuple(
        [slice(overlap[0][i] - min_corner1[i], overlap[1][i] - min_corner1[i]) for i in
         range(len(min_corner1))])
    overlap_sl2 = tuple(
        [slice(overlap[0][i] - min_corner2[i], overlap[1][i] - min_corner2[i]) for i in
         range(len(min_corner2))])

    return overlap_sl1, overlap_sl2


def pairwise_overlap_of_segmentations_torch(segA, segB, ignore_label=0):
    if ignore_label is None:
        segA = torch.from_numpy(segA.astype(np.int32).ravel())
        segB = torch.from_numpy(segB.astype(np.int32).ravel())
    else:
        fg_mask = np.logical_or(segA != ignore_label, segB != ignore_label)
        segA = torch.from_numpy(segA[fg_mask].astype(np.int32).ravel())
        segB = torch.from_numpy(segB[fg_mask].astype(np.int32).ravel())

    with torch.no_grad():
        countsA, countsB = [{label.item(): count.item() for label, count in zip(*torch.unique(seg, return_counts=True))}
                            for seg in (segA, segB)]

        uniq = torch.unique(torch.stack([segA, segB], dim=-1), dim=0, return_counts=True)
        overlap_dict = {tuple(labels.tolist()): count.item() for labels, count in zip(*uniq)}

        iou_dict = {labels: count / (countsA[labels[0]] + countsB[labels[1]] - count) for
                    labels, count in overlap_dict.items()}

    return overlap_dict, iou_dict, countsA, countsB


def nonexclustive_iou_match_segments(iou_dict, iou_threshold=0.4):
    """
    iou_dict: keys like (label1, label2)
    for each segm_label find the highest iou with gt_label and assign it to it.
    """
    mapping_label1_to_label2 = {pair[0]: (-1, -1) for pair in iou_dict.keys()}  # (-1, -1) means no match
    for pair, iou in iou_dict.items():
        if iou < iou_threshold:
            continue
        if mapping_label1_to_label2[pair[0]][1] == -1:  # no match so far
            mapping_label1_to_label2[pair[0]] = (pair[1], iou)
        else:
            if iou > mapping_label1_to_label2[pair[0]][1]:
                mapping_label1_to_label2[pair[0]] = (pair[1], iou)

    return mapping_label1_to_label2


def exclusive_iou_match_segments(gt_segm_counts, iou_dict, ignore_labels=(0,)):
    """
    identify segments based on intersection over union. Loop over GT segments (from largest to smallest)
    and assign it the best fitting predicted segment.

    Parameters
    ----------
    gt_segm_counts
    iou_dict: keys like (segm_label, gt_label)

    Returns
    -------

    """

    # first sort gt_mask_dict based on segment size (largest first):
    gt_segment_size_arr = np.asarray(list(gt_segm_counts.values()))
    gt_labels = np.asarray(list(gt_segm_counts.keys()))
    # print(gt_labels.shape, gt_segment_size_arr.shape)
    sorted_gt_labels = gt_labels[np.argsort(gt_segment_size_arr)[::-1]]

    # find a mapping between segments:
    segm_to_gt_mapping_dict = {}
    gt_to_segm_mapping_dict = {}

    lookup_dict = {}
    for gt_labels in sorted_gt_labels:
        lookup_dict[gt_labels] = {"iou_list": [], "segm_label_list": []}

    for pair, iou in iou_dict.items():
        lookup_dict[pair[1]]["iou_list"].append(iou)
        lookup_dict[pair[1]]["segm_label_list"].append(pair[0])

    for gt_label in sorted_gt_labels:
        if gt_label in ignore_labels:
            continue
        # find best segment_label
        iou_list = lookup_dict[gt_label]["iou_list"]
        best_segm_label = lookup_dict[gt_label]["segm_label_list"][np.argmax(iou_list)]
        if best_segm_label not in segm_to_gt_mapping_dict:
            # only if segm is not yet mapped to a GT segm:
            gt_to_segm_mapping_dict[gt_label] = best_segm_label
            segm_to_gt_mapping_dict[best_segm_label] = gt_label

    return gt_to_segm_mapping_dict, segm_to_gt_mapping_dict


def overlap_between_segments(mask1, mask2, min_corner1, min_corner2):
    """
    output must not necessarily be int
    """
    bbox1 = (min_corner1, min_corner1 + np.asarray(mask1.shape))
    bbox2 = (min_corner2, min_corner2 + np.asarray(mask2.shape))

    # print("bboxes", bbox1, bbox2)
    overlap = bbox_overlap(bbox1, bbox2)
    # print("overlap", overlap)
    if overlap is None:
        # zero overlap
        return 0

    # construct slices for overlap; shift according to min_corner's:
    sl1 = tuple(
        [slice(overlap[0][i] - min_corner1[i], overlap[1][i] - min_corner1[i]) for i in range(len(min_corner1))])
    sl2 = tuple(
        [slice(overlap[0][i] - min_corner2[i], overlap[1][i] - min_corner2[i]) for i in range(len(min_corner2))])

    return np.sum(mask1[sl1] * mask2[sl2])
