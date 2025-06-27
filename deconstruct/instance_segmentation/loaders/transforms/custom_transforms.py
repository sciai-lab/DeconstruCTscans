import numpy as np
import torch
import h5py
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

from embeddingutils.affinities import embedding_to_affinities, label_equal_similarity
from deconstruct.utils.general import build_isotropic_3dstencil


class CustomCompose(torch.nn.Module):

    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def __call__(self, input_img, target_img):
        for t in self.transforms:
            input_img, target_img = t(input_img, target_img)
        return input_img, target_img


class TargetToAffinities(torch.nn.Module):

    def __init__(self, offsets_config=None, offsets=None, invert_affinities=True, add_bg_channel=True,
                 affinity_measure=label_equal_similarity, pad_value=-1):
        """
        Transform to convert a segmentation target to affinities.

        Parameters
        ----------
        invert_affinities : bool, optional
            If True, affinities are inverted, i.e. 0 becomes 1 and 1 becomes 0.
        """
        super().__init__()
        assert (offsets_config is None) != (offsets is None), "either offsets_config or offsets must be specified"
        if offsets_config is not None:
            self.offsets = build_isotropic_3dstencil(**offsets_config)
        else:
            self.offsets = offsets
        self.invert_affinities = invert_affinities
        self.add_bg_channel = add_bg_channel
        self.affinity_measure = affinity_measure
        self.dim = self.offsets.shape[1]
        self.pad_value = pad_value

    def __call__(self, input_img=None, target_img=None):
        """expects torch tensors as inputs"""
        assert target_img is not None, "target_img must not be None"

        # convert target to affinities:
        chunk_shape = target_img.shape[-self.dim:]

        target_img = target_img.view(1, -1, *chunk_shape)
        target_img_emb = embedding_to_affinities(target_img, self.offsets, self.affinity_measure,
                                                 pad_val=self.pad_value)

        # add background channel:
        if self.add_bg_channel:
            target_img_emb = torch.cat([target_img_emb, (target_img == 0) * 1.], dim=1)

        # invert affinities:
        if self.invert_affinities:
            pad_value_mask = (target_img_emb == self.pad_value)
            target_img_emb = 1. - target_img_emb  # so that edges are 1 (sparse label) and fg is 1 (sparse label)
            target_img_emb[pad_value_mask] = self.pad_value

        return input_img, target_img_emb


class Complete3DTransforms(torch.nn.Module):

    def __init__(self, prob_crop_resize=0., min_crop_fraction=0.3):
        """
            Transform to realize all 48 transforms of rotations and reflections in 3d + random crop&resize for d
            ata augmentation.

            Parameters
            ----------
            prob_crop_resize : float
                Specifies the probability for random crop&resize.
            min_crop_fraction :

            Returns
            -------
        """
        super().__init__()
        self.prob_crop_resize = prob_crop_resize
        self.min_crop_fraction = min_crop_fraction

    def __call__(self, input_img, target_img):
        """
        input_img : torch.Tensor
        of shape (C,img_shape)
        target_img : torch.Tensor
        of shape (C,img_shape)
        """
        assert input_img.shape == target_img.shape, "input and target image must have same shape"
        img_shape = input_img[0].shape

        # combine input and target into one tensor to transform them in the same way:
        pair = torch.stack([input_img, target_img], dim=0)  # (N, C, spatial)

        # randomly permute the spatial axes:
        axes_permutation = np.random.permutation(len(img_shape))
        # print(pair.shape, tuple(axes_permutation + 2))
        pair = torch.permute(pair, (0, 1,) + tuple(axes_permutation + 2))  # leave first two axis (N,C)

        # randomly flip along each spatial axis:
        for i in range(len(img_shape)):
            if np.random.random() < 0.5:
                pair = torch.flip(pair, dims=(i + 2,))

        # potentially perform a crop and resize:
        t = random_crop_resize(pair, min_fraction=self.min_crop_fraction, p=self.prob_crop_resize)
        assert t[0].shape == input_img.shape
        assert t[1].shape == target_img.shape

        return t[0], t[1]


def random_crop_resize(img, min_fraction=0.3, p=1):
    """
    Randomly crops an image and resizes it to the original size

    Parameters
    ----------
    img : torch.Tensor
        of shape  (N,C,img_shape)
    min_fraction : float, optional
        between [0,1], specifies minimal size to which one crops. DEfault = 0.3.
    p : float, optional
        prob to crop&resize. Default = 1.

    Returns
    -------
    torch.Tensor
        of shape  (N,C,img_shape), after cropping and resizing.
    """
    if np.random.random() < 1-p:
        # do not perform transform
        return img

    dim = len(img.shape[2:])

    # take a crop:
    fraction = np.random.uniform(min_fraction, 1.)
    size = (fraction * torch.tensor(img.shape[2:])).to(int)

    # random position of left lower corner:
    max_pos = torch.tensor(img.shape[2:]) - size
    sl = (slice(None), slice(None))
    for i in range(dim):
        pos = np.random.randint(0, max_pos[i])
        sl += (slice(pos, pos+size[i]),)
    crop = img[sl]

    # upsample crop:
    upsample = nn.Upsample(size=img.shape[2:])
    upsampled_crop = upsample(crop.float())  # expects (N,C,img_dims)

    return upsampled_crop.view(img.shape)

