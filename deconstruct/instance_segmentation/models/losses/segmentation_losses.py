import numpy as np
import torch
import torch.nn as nn


class DiceScore(nn.Module):

    def __init__(self, eps=1e-7, sum_over_channels=False, pad_value=-1):
        super().__init__()
        self.eps = eps
        self.sum_over_channels = sum_over_channels
        self.pad_value = pad_value

    def forward(self, predicted_img, target_img, ignore_mask=None):
        """expects tensors of shape (N, C, img_shape)"""
        assert predicted_img.shape == target_img.shape, f"shapes don't match {predicted_img.shape=} {target_img.shape=}"
        num_axes = len(predicted_img.shape)
        if self.sum_over_channels:
            summed_axes = tuple(np.arange(1, num_axes))
        else:
            summed_axes = tuple(np.arange(2, num_axes))

        if ignore_mask is None and self.pad_value is not None:
            predicted_img = predicted_img.clone()
            target_img = target_img.clone()
            ignore_mask = (target_img == self.pad_value)

        if ignore_mask is not None:
            target_img[ignore_mask] = 0
            predicted_img[ignore_mask] = 0

        dice_score = (2 * torch.sum(predicted_img * target_img, axis=summed_axes)) / \
                     (torch.sum(predicted_img ** 2, axis=summed_axes) + torch.sum(target_img ** 2,
                                                                                  axis=summed_axes) + self.eps)

        return dice_score


class CustomBCELoss(nn.Module):

        def __init__(self, eps=1e-7, sum_over_channels=False, pad_value=-1):
            super().__init__()
            self.eps = eps
            self.sum_over_channels = sum_over_channels
            self.pad_value = pad_value
            self.bce = nn.BCELoss(reduction="none")

        def forward(self, predicted_img, target_img):
            assert predicted_img.shape == target_img.shape, "shapes don't match"
            num_axes = len(predicted_img.shape)
            if self.sum_over_channels:
                summed_axes = tuple(np.arange(1, num_axes))
            else:
                summed_axes = tuple(np.arange(2, num_axes))

            bce = self.bce(predicted_img, target_img)

            if self.pad_value is not None:
                ignore_mask = (target_img == self.pad_value)
                bce[ignore_mask] = 0

            # mean over non-padded voxels only:
            bce = torch.sum(bce, axis=summed_axes) / torch.sum(torch.logical_not(ignore_mask), axis=summed_axes)

            return bce


class BCEDiceLoss(nn.Module):

    sigmoid = nn.Sigmoid()

    def __init__(self, bce_fraction=0.5, eps=1e-7, include_sigmoid=False, sum_over_channels=False, pad_value=-1):
        super().__init__()
        self.bce_fraction = bce_fraction
        self.include_sigmoid = include_sigmoid

        self.bce = CustomBCELoss(eps=eps, sum_over_channels=sum_over_channels, pad_value=pad_value)
        self.dice = DiceScore(eps=eps, sum_over_channels=sum_over_channels, pad_value=pad_value)

    def forward(self, predicted_img, target_img):
        if self.include_sigmoid:
            predicted_img = self.sigmoid(predicted_img)

        dice_loss = 1 - self.dice(predicted_img, target_img)
        bce_loss = self.bce(predicted_img, target_img)

        return (1 - self.bce_fraction) * dice_loss + self.bce_fraction * bce_loss



