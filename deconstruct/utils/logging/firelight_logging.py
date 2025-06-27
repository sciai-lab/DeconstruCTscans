import numpy as np
import torch
import yaml


# suppress firelight logging:
import logging

import yaml
from firelight import get_visualizer
logging.getLogger("firelight").setLevel(logging.WARNING)


def _remove_alpha(tensor, background_brightness=1):
    """
    used in tensorboard logging with fireligth.
    taken from
    https://firelight.readthedocs.io/en/latest/_modules/firelight/inferno_callback.html#get_visualization_callback
    """
    return torch.ones_like(tensor[..., :3]) * background_brightness * (1 - tensor[..., 3:4]) + \
           tensor[..., :3] * tensor[..., 3:4]


class FirelightSegmentationVisualizer:

    def __init__(self, firelight_config=None, path_firelight_config=None, pad_value=-1, pad_target=0.5):
        assert firelight_config is not None or path_firelight_config is not None, \
            "Either firelight_config or path_firelight_config must be given."
        if path_firelight_config is not None:
            with open(path_firelight_config, "r") as f:
                firelight_config = yaml.load(f, Loader=yaml.SafeLoader)
        self.visualizer = get_visualizer(firelight_config)
        self.pad_value = pad_value
        self.pad_target = pad_target

    def __call__(self, input_img, target_img, pred_img, gt_segmentation):
        """expects tensors to be detached already"""
        if self.pad_value is not None and self.pad_target is not None:
            target_img[target_img == self.pad_value] = self.pad_target

        states_dict = {
            "raw_input": (input_img.cpu(), 'BCDHW'),
            "GT_segmentation": (gt_segmentation.cpu(), 'BCDHW'),
            "predicted_affinities": (pred_img.cpu(), 'BCDHW'),
            "GT_affinities": (target_img.cpu(), 'BCDHW'),
            "affinities_diff": (pred_img.cpu() - target_img.cpu(), 'BCDHW')}

        log_img = self.visualizer(**states_dict)
        log_img = _remove_alpha(log_img).permute(2, 0, 1)  # to [Color, Height, Width]

        return log_img




