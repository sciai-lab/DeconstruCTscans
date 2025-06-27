import time
import numpy as np
from tqdm import tqdm
from deconstruct.utils.calculating_overlaps import pairwise_overlap_of_segmentations_torch


class MyPanopticQuality:

    def __init__(self, classes, background_label=None):
        """treating background separately is for potential speedup."""
        self.background_label = background_label
        # make unique:
        self.classes = set(classes)
        if len(self.classes) != len(classes):
            print("WARNING: classes should be unique.")
        # check if things and stuff are disjoint:
        if self.background_label is not None:
            self.classes.add(self.background_label)

    def iou_instance_matching(self, solution_panoptic_volume, gt_panoptic_volume):
        solution_to_gt_mapping = {}
        bg_iou = 0.0
        bg_mask = None
        gt_bg_mask = None
        if self.background_label is not None:
            print("treating background separately.")
            # treat background separately:
            # check that the background is matched to background:
            bg_mask = (solution_panoptic_volume[..., 0] == self.background_label)
            gt_bg_mask = (gt_panoptic_volume[..., 0] == self.background_label)
            bg_iou = np.logical_and(bg_mask, gt_bg_mask).sum() / max(np.logical_or(bg_mask, gt_bg_mask).sum(), 1e-7)

        # now check all other instances:
        start = time.time()
        if bg_iou > 0.5:
            solution_to_gt_mapping[0] = (0, bg_iou)
            # remove background from the volume:
            intersection_bg_mask = np.logical_and(bg_mask, gt_bg_mask)
            iou_dict = pairwise_overlap_of_segmentations_torch(solution_panoptic_volume[~intersection_bg_mask, 1],
                                                               gt_panoptic_volume[~intersection_bg_mask, 1],
                                                               ignore_label=None)[1]
        else:
            iou_dict = pairwise_overlap_of_segmentations_torch(solution_panoptic_volume[..., 1],
                                                               gt_panoptic_volume[..., 1],
                                                               ignore_label=None)[1]
        # print("iou_dict", iou_dict)
        print("pure iou took:", time.time() - start)

        # now match the instances:
        for label_pair, iou in iou_dict.items():
            solution_label, gt_label = label_pair
            if solution_label not in solution_to_gt_mapping:
                solution_to_gt_mapping[solution_label] = (gt_label, iou)
            else:
                if iou > solution_to_gt_mapping[solution_label][1]:
                    solution_to_gt_mapping[solution_label] = (gt_label, iou)

        # threshold with 0.5:
        list_unmatched_solution_labels = []
        for solution_label, (gt_label, iou) in solution_to_gt_mapping.items():
            if iou < 0.5:
                list_unmatched_solution_labels.append(solution_label)

        for solution_label in list_unmatched_solution_labels:
            del solution_to_gt_mapping[solution_label]

        # symmetrize:
        gt_to_solution_mapping = {}
        for solution_label, (gt_label, iou) in solution_to_gt_mapping.items():
            gt_to_solution_mapping[gt_label] = (solution_label, iou)

        return solution_to_gt_mapping, gt_to_solution_mapping

    def check_class_to_segments(self, mapping_things_to_segments, gt_mapping_things_to_segments):
        """
        mapping_things_to_segments: dict with thing labels as keys and list of segment labels as values

        :param mapping_things_to_segments:
        :param gt_mapping_things_to_segments:
        :return:
        """
        things_present_in_class_to_segments = set(mapping_things_to_segments.keys())
        things_present_in_gt_class_to_segments = set(gt_mapping_things_to_segments.keys())
        # should be subsets of things:
        assert things_present_in_class_to_segments.issubset(self.classes)
        assert things_present_in_gt_class_to_segments.issubset(self.classes)

        # add empty list for things that are not present:
        for thing in self.classes:
            if thing not in things_present_in_class_to_segments:
                mapping_things_to_segments[thing] = []
            if thing not in things_present_in_gt_class_to_segments:
                gt_mapping_things_to_segments[thing] = []

        return mapping_things_to_segments, gt_mapping_things_to_segments

    @staticmethod
    def pq_for_one_class(relevant_segments_labels, gt_relevant_segment_labels,
                         solution_to_gt_mapping, gt_to_solution_mapping):
        # no duplicates in relevant_segments_labels and gt_relevant_segment_labels:
        assert len(relevant_segments_labels) == len(set(relevant_segments_labels))
        assert len(gt_relevant_segment_labels) == len(set(gt_relevant_segment_labels))
        list_true_positives = []
        iou_sum = 0.0
        true_positives_counter = 0
        false_matches_counter = 0
        for segment_label in relevant_segments_labels:
            if segment_label not in solution_to_gt_mapping:
                # false positive (not matched to anything):
                false_matches_counter += 1
                pass
            else:
                gt_label, iou = solution_to_gt_mapping[segment_label]
                if gt_label in gt_relevant_segment_labels:
                    # true positive:
                    list_true_positives.append((segment_label, gt_label))
                    iou_sum += iou
                    true_positives_counter += 1
                else:
                    # false positive (matched to something else):
                    false_matches_counter += 1

        for gt_segment_label in gt_relevant_segment_labels:
            if gt_segment_label not in gt_to_solution_mapping:
                # false negative (not matched to anything):
                false_matches_counter += 1
                pass
            else:
                solution_label = gt_to_solution_mapping[gt_segment_label][0]
                if solution_label not in relevant_segments_labels:
                    # false negative (matched to something else):
                    false_matches_counter += 1

        denominator = true_positives_counter + 0.5 * false_matches_counter
        if denominator > 0:
            pq = iou_sum / denominator
        else:
            pq = None

        # print("pq for one class:", pq, list_true_positives, false_matches_counter)

        return pq

    @staticmethod
    def infer_mapping_things_to_segments(panoptic_volume):
        """This could be made more efficient with unique on the instance seg and return_index=True."""
        print("inferring mapping things to segments... This is not very efficient. provide if it is known.")
        start = time.time()
        mapping_things_to_segments = {}
        all_instance_labels, indices = np.unique(panoptic_volume[..., 1], return_index=True)
        all_class_labels = panoptic_volume[..., 0].ravel()[indices]
        for instance_label, class_label in zip(all_instance_labels, all_class_labels):
            if class_label not in mapping_things_to_segments:
                mapping_things_to_segments[class_label] = []
            mapping_things_to_segments[class_label].append(instance_label)
        print("inferring mapping things to segments took:", time.time() - start)
        return mapping_things_to_segments

    def __call__(self, solution_panoptic_volume, gt_panoptic_volume,
                 mapping_things_to_segments=None, gt_mapping_things_to_segments=None):
        """first channel of last two channels is thing label, second channel is instance label."""
        if mapping_things_to_segments is None:
            mapping_things_to_segments = self.infer_mapping_things_to_segments(solution_panoptic_volume)
        if gt_mapping_things_to_segments is None:
            gt_mapping_things_to_segments = self.infer_mapping_things_to_segments(gt_panoptic_volume)
        mapping_things_to_segments, gt_mapping_things_to_segments = self.check_class_to_segments(
            mapping_things_to_segments, gt_mapping_things_to_segments)

        solution_to_gt_mapping, gt_to_solution_mapping = self.iou_instance_matching(solution_panoptic_volume,
                                                                                    gt_panoptic_volume)
        # print("solution_to_gt_mapping", solution_to_gt_mapping)
        # compute pq for each thing:
        list_pq = []
        for thing in tqdm(self.classes):
            pq = self.pq_for_one_class(mapping_things_to_segments[thing], gt_mapping_things_to_segments[thing],
                                       solution_to_gt_mapping, gt_to_solution_mapping)
            if pq is not None:
                list_pq.append(pq)

        # print("list_pq:", list_pq)

        if len(list_pq) == 0:
            return -1
        else:
            return np.mean(list_pq)
