import numpy as np
import torch
import h5py

from deconstruct.instance_segmentation.loaders.transforms.custom_transforms import TargetToAffinities
from deconstruct.instance_segmentation.models.losses.segmentation_losses import DiceScore
from deconstruct.instance_segmentation.inference.segmentation_prediction import load_affinities_from_h5


def pad_proposal_mask_for_offsets(proposal_mask, offsets, min_corner, vol_shape):
    max_offset = np.max(np.abs(offsets))
    # print("before", proposal_mask.shape)

    max_corner = min_corner + np.asarray(proposal_mask.shape)
    new_min_corner = np.maximum(min_corner - max_offset, 0)
    new_max_corner = np.minimum(max_corner + max_offset, np.asarray(vol_shape))

    pad_width = [(min_corner[i] - new_min_corner[i], new_max_corner[i] - max_corner[i])
                 for i in range(len(vol_shape))]

    # print(pad_width, new_max_corner)
    proposal_mask = np.pad(proposal_mask, pad_width=pad_width, constant_values=(0, 0))

    return proposal_mask, new_min_corner


class AffinityScore:

    def __init__(self, path_h5_affinities, use_only_long_range_affs=True):
        self.path_h5_affinities = path_h5_affinities
        self.use_only_long_range_affs = use_only_long_range_affs

        with h5py.File(self.path_h5_affinities, "r") as f:
            self.offsets = f.attrs["offsets"]
            self.vol_shape = f["summed_affs"].shape[1:]

        self.affinity_transform = TargetToAffinities(offsets=self.offsets, pad_value=-1)
        self.dice_score = DiceScore(sum_over_channels=False, pad_value=self.affinity_transform.pad_value)

    def calculate_affinity_score(self, predicted_affs, proposal_mask):
        """
        aff convention used in here: 1 means different segment i.e. boundary evidence
        affs should be np.darray with shape (C, img_dim)
        proposal_mask must be padded based on offsets
        """

        hypothetical_gt_affs = self.affinity_transform(target_img=torch.from_numpy(proposal_mask[None, None]))[1]
        hypothetical_gt_affs = hypothetical_gt_affs[:, :-1]   # get rid of fg/bg channel
        hypothecial_boundary_aff_mask = (hypothetical_gt_affs == 1)  # boundary evidence which should fit to prediction

        # intersection over union score:
        mask_aff_score = torch.logical_or(hypothecial_boundary_aff_mask, torch.from_numpy(proposal_mask[None, None, :]))

        # check that there is no more pad value in there:
        # this may be the case if one is at the boundary of a volume.
        if (hypothecial_boundary_aff_mask[mask_aff_score] == self.affinity_transform.pad_value).any():
            print("padding in region of interest")
            mask_aff_score[mask_aff_score] = (hypothetical_gt_affs[mask_aff_score] != self.affinity_transform.pad_value)

        # dice score for each channel
        # IMPORTANT: 1 must be the scares label i.e. indication of boundary evidence
        dice = self.dice_score(hypothetical_gt_affs, torch.from_numpy(predicted_affs[:-1])[None, :],
                               ignore_mask=torch.logical_not(mask_aff_score))

        # average over channels:
        use_channels_mask = torch.ones(len(self.offsets), dtype=bool)
        if self.use_only_long_range_affs:
            use_channels_mask[:3] = False
        return (dice[:, use_channels_mask]).mean().item()

    def __call__(self, proposal_mask, min_corner):
        # pad proposal mask sufficiently to get all the boundary evidence:
        proposal_mask, min_corner = pad_proposal_mask_for_offsets(proposal_mask, self.offsets, min_corner,
                                                                  vol_shape=self.vol_shape)

        # get predicted affs:
        sl_affs = ((slice(None),) +
                   tuple([slice(min_corner[i], min_corner[i] + s) for i, s in enumerate(proposal_mask.shape)]))
        predicted_affinities = load_affinities_from_h5(self.path_h5_affinities, sl=sl_affs)

        # compute affinity score:
        return self.calculate_affinity_score(predicted_affinities, proposal_mask)