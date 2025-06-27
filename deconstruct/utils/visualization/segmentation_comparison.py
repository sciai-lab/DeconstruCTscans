import numpy as np
import os
import h5py
from memory_profiler import profile
from deconstruct.utils.general import rotate_volume, cut_label_volume_into_bboxes
from deconstruct.utils.calculating_overlaps import pairwise_overlap_of_segmentations_torch, exclusive_iou_match_segments
from deconstruct.utils.part_catalog_utils import clip_volumes_to_foreground
from deconstruct.utils.visualization.colorize_label_img import save_rgb_uint8_for_vgmax


def rgb_colorize_mistakes_img(mistakes_img, mistakes_color_mapping_dict="default", correct_color=(0, 0, 1),
                              bg_color=(0, 0, 0), correct_label=1, bg_label=0):
    if mistakes_color_mapping_dict == "default":
        mistakes_color_mapping_dict = {
            2: (1, 0, 0),  # overseg = red
            3: (0, 1, 0)  # underseg = green
        }

    colored_mistakes_img = np.zeros(mistakes_img.shape + (3,), dtype=np.uint8)

    # correct foreground:
    colored_mistakes_img[mistakes_img == correct_label, :] = np.asarray(correct_color)

    # background:
    colored_mistakes_img[mistakes_img == bg_label, :] = np.asarray(bg_color)

    for label, color in mistakes_color_mapping_dict.items():
        colored_mistakes_img[mistakes_img == label, :] = np.asarray(color)
    return colored_mistakes_img


@profile
def save_colorized_segmentation_mistakes(seg, gt_seg, rotation_matrix=None, save_path="colored_seg_mistakes.raw",
                                         verbose=True, ignore_labels=(0,)):
    assert seg.shape == gt_seg.shape, "shapes must match."
    # mistakes_int_mapping:
    mistakes_int_mapping_dict = {
        "overseg": 2,
        "underseg": 3
    }

    # separate individual segments:
    segm_mask_dict = cut_label_volume_into_bboxes(seg)

    # build correspondences between GT and prediction based on IoU:
    _, iou_dict, _, gt_segm_counts = pairwise_overlap_of_segmentations_torch(seg, gt_seg)
    gt_to_segm_mapping_dict, segm_to_gt_mapping_dict = exclusive_iou_match_segments(gt_segm_counts=gt_segm_counts,
                                                                                    iou_dict=iou_dict,
                                                                                    ignore_labels=ignore_labels)
    # assemble the images:
    seg = (seg > 0).astype(np.uint8)  # foreground mask

    for segm_label, segm in segm_mask_dict.items():
        segm_mask = segm[0]
        min_corner = segm[1]
        sl = tuple([slice(min_corner[i], min_corner[i] + segm_mask.shape[i]) for i in range(len(min_corner))])
        if segm_label in segm_to_gt_mapping_dict:
            # check the parts where predicted mask does not match the gt_mask:
            corresponding_gt_label = segm_to_gt_mapping_dict[segm_label]
            mistake_mask = (gt_seg[sl] != corresponding_gt_label) * segm_mask
            seg[sl][mistake_mask] = mistakes_int_mapping_dict["underseg"]
        else:
            # give unmap segm a certain color (overseg), loop over them:
            seg[sl][segm_mask] = mistakes_int_mapping_dict["overseg"]

    if verbose:
        print("Found the following mistakes: ", np.unique(seg, return_counts=True), mistakes_int_mapping_dict)

    seg = (rgb_colorize_mistakes_img(seg) * 255).astype(np.uint8)

    if rotation_matrix is not None:
        seg = rotate_volume(seg, rotation_matrix, interpolation_mode="bilinear", keep_dtype=True)
        seg = clip_volumes_to_foreground([seg], foreground_threshold=0.5, channel_axis=0)[0][0]

    if save_path is not None:
        save_rgb_uint8_for_vgmax(seg, save_path=save_path)

    return seg


def iou_instance_matching(insseg, gt_insseg, bg_label=0):
    pred_to_gt_mapping = {}
    bg_mask = (insseg == bg_label)
    gt_bg_mask = (gt_insseg == bg_label)
    bg_intersect = np.logical_and(bg_mask, gt_bg_mask)
    iou_dict = pairwise_overlap_of_segmentations_torch(insseg[~bg_intersect], gt_insseg[~bg_intersect],
                                                       ignore_label=None)[1]

    # now match the instances:
    for label_pair, iou in iou_dict.items():
        pred_label, gt_label = label_pair
        if iou > 0.5:
            pred_to_gt_mapping[pred_label] = gt_label

    return pred_to_gt_mapping


def save_single_color_segmentation_mistakes(seg, gt_seg, rotation_matrix=None, save_path="colored_seg_mistakes.raw",
                                            bg_label=0, remove_bg_mistakes=True):
    assert seg.shape == gt_seg.shape, "shapes must match."

    # separate individual segments:
    segm_mask_dict = cut_label_volume_into_bboxes(seg)

    # build correspondences between GT and prediction based on IoU:
    pred_to_gt_mapping = iou_instance_matching(seg, gt_seg, bg_label=bg_label)

    # assemble the images:
    seg = (seg > 0).astype(np.uint8)  # foreground mask

    for segm_label, segm in segm_mask_dict.items():
        segm_mask, min_corner = segm
        sl = tuple([slice(min_corner[i], min_corner[i] + segm_mask.shape[i]) for i in range(len(min_corner))])
        if segm_label in pred_to_gt_mapping:
            # check the parts where predicted mask does not match the gt_mask:
            corresponding_gt_label = pred_to_gt_mapping[segm_label]
            mistake_mask = np.logical_and((gt_seg[sl] != corresponding_gt_label), segm_mask)
        else:
            # give unmap segm a certain color (overseg), loop over them:
            mistake_mask = segm_mask
        seg[sl][mistake_mask] = 3  # this influences the color in the next step (3 = green)

    if remove_bg_mistakes:
        seg[gt_seg == bg_label] = 0

    seg = (rgb_colorize_mistakes_img(seg) * 255).astype(np.uint8)

    if rotation_matrix is not None:
        seg = rotate_volume(seg, rotation_matrix, interpolation_mode="bilinear", keep_dtype=True)
        seg = clip_volumes_to_foreground([seg], foreground_threshold=0.5, channel_axis=0)[0][0]

    if save_path is not None:
        save_rgb_uint8_for_vgmax(seg, save_path=save_path)

    return seg


path_h5_segmentation = "fill_in"
path_h5_dataset = "fill_in"
if __name__ == '__main__':
    with h5py.File(path_h5_segmentation, "r") as f:
        seg = f["final_segmentation"][:]

    with h5py.File(path_h5_dataset, "r") as f:
        gt_seg = f["gt_instance_volume"][:]

    save_path = os.path.join(os.path.dirname(path_h5_segmentation), "single_colored_seg_mistakes.raw")
    save_single_color_segmentation_mistakes(seg, gt_seg, save_path=save_path, remove_bg_mistakes=True)


