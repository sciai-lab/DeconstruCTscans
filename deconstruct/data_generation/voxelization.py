import numpy as np
import stltovoxel  # see https://github.com/cpederkoff/stl-to-voxel
import os
import z5py
from tqdm.auto import tqdm
import concurrent.futures  # for multiprocessing
import itertools  # to repeat constant arguments in the map of multiprocessing
import pandas as pd

from deconstruct.utils.part_catalog_utils import get_bbox_center_for_voxelization


def tight_voxelize_single_mesh(mesh, constrained_min_corner_int=None, constrained_max_corner_int=None):
    """
    constrained_min_corner_int and constrained_max_corner_int are expectected to be in xyz mesh-like convention
    slicing made much more efficient by using the bounding box of each mesh
    """
    _, bbox = get_bbox_center_for_voxelization([mesh])
    min_corner_int = np.floor(bbox[0]).astype(int)
    max_corner_int = np.ceil(bbox[1]).astype(int)

    # check if mesh is inside max and min corner:
    if constrained_min_corner_int is not None and (min_corner_int < constrained_min_corner_int).any():
        print("mesh is outside constrained_min_corner_int", min_corner_int, constrained_min_corner_int)
        return None, None
    if constrained_max_corner_int is not None and (max_corner_int > constrained_max_corner_int).any():
        print("mesh is outside constrained_max_corner_int", max_corner_int, constrained_max_corner_int)
        return None, None

    mesh = mesh - min_corner_int
    # get foreground mask:
    foreground_mask = stltovoxel.slicing.mesh_to_plane(mesh, max_corner_int - min_corner_int, parallel=False)
    return foreground_mask, min_corner_int[::-1]


def save_tight_voxelization_to_z5(path_z5_dataset, foreground_mask, min_corner, group_name):
    if foreground_mask is None:
        return
    with z5py.File(path_z5_dataset, "a") as f:
        grp = f.create_group(group_name)
        grp.create_dataset("min_corner", data=min_corner)
        grp.create_dataset("foreground_mask", data=foreground_mask.astype(np.uint8))  # z5 can't use bool


def tight_voxelization_list_meshes(list_meshes, tqdm_bool=True, tqdm_position=None,
                                   path_z5_dataset=None, list_z5_group_names=None,
                                   constrained_min_corner_int=None, constrained_max_corner_int=None):
    """constrained_min_corner_int and constrained_max_corner_int are expectected to be in xyz mesh-like convention"""
    list_mask_min_corner = []
    if tqdm_position is None or not tqdm_bool:
        for ind, mesh in (enumerate(tqdm(list_meshes)) if tqdm_bool else enumerate(list_meshes)):
            foreground_mask, min_corner_int = tight_voxelize_single_mesh(mesh,
                                                                         constrained_min_corner_int,
                                                                         constrained_max_corner_int)
            if path_z5_dataset is not None:
                z5_group_name = f"mesh_{ind}" if list_z5_group_names is None else list_z5_group_names[ind]
                save_tight_voxelization_to_z5(path_z5_dataset, foreground_mask, min_corner_int, z5_group_name)
            else:
                list_mask_min_corner.append((foreground_mask, min_corner_int))
    else:
        # set up individual tdqm bars for multiprocessing:
        # This line is the strange hack (see https://github.com/tqdm/tqdm/issues/485)
        # wick lock = multiprocessing.Manager().Lock() also seems no longer necessary
        print(' ', end='', flush=True)
        bar = tqdm(
            desc=f'Thread {tqdm_position}',
            total=len(list_meshes),
            position=tqdm_position,
            leave=False
        )

        for ind, mesh in (enumerate(tqdm(list_meshes)) if tqdm_bool else enumerate(list_meshes)):
            foreground_mask, min_corner_int = tight_voxelize_single_mesh(mesh,
                                                                         constrained_min_corner_int,
                                                                         constrained_max_corner_int)
            if path_z5_dataset is not None:
                z5_group_name = f"mesh_{ind}" if list_z5_group_names is None else list_z5_group_names[ind]
                save_tight_voxelization_to_z5(path_z5_dataset, foreground_mask, min_corner_int, z5_group_name)
            else:
                list_mask_min_corner.append((foreground_mask, min_corner_int))
            bar.update(1)

    return list_mask_min_corner


def from_tight_masks_to_label_volume(list_mask_min_corner, output_shape):
    output_vol = np.zeros(output_shape[::-1], dtype=np.uint16)  # flipped output shape for xyz -> zyx convention.
    if len(list_mask_min_corner) >= 2 ** 16:
        print("too many instances for data type uint16.")
        output_vol = output_vol.astype(np.uint32)

    # build slice:
    for ind, pair in enumerate(list_mask_min_corner):
        foreground_mask, min_corner_int = pair
        sl = tuple([slice(min_corner_int[i], min_corner_int[i] + s) for i, s in enumerate(foreground_mask.shape)])

        if (output_vol[sl][foreground_mask] > 0).any():
            # print("Careful segments are overlapping and later ones overwrite earlier ones.")
            pass
        output_vol[sl][foreground_mask] = ind + 1

    # check if there are any labels which get zero voxels:
    fg_mask = (output_vol > 0)
    zero_voxel_label = set(np.arange(1, ind + 2)) - set(pd.unique(output_vol[fg_mask]))
    if len(zero_voxel_label) > 0:
        print(f"the following labels get zero labels in masks to vol=", zero_voxel_label)
        # build the mask of zero ones:
        zero_mask = np.zeros_like(output_vol, dtype=bool)
        for label in zero_voxel_label:
            foreground_mask, min_corner_int = list_mask_min_corner[label - 1]
            sl = tuple([slice(min_corner_int[i], min_corner_int[i] + s) for i, s in enumerate(foreground_mask.shape)])
            zero_mask[sl][foreground_mask] = True
        return output_vol, zero_mask

    return output_vol, None


class MeshVoxelizer:

    def __init__(self, output_shape=None, num_threads=1, tqdm_bool=True):
        self.output_shape = output_shape
        self.num_threads = num_threads
        self.parallel = (num_threads > 1)
        self.tqdm_bool = tqdm_bool

    @staticmethod
    def check_if_meshes_fit(list_meshes, output_shape):
        # calculate overall bounding box to check if all meshes fit:
        _, bbox = get_bbox_center_for_voxelization(list_meshes)
        assert (bbox[0] > 0).all() and (np.asarray(output_shape) > bbox[1]).all(), (f"meshes do not fit in volume of "
                                                                                    f"shape {output_shape} with bbox "
                                                                                    f"{bbox}.")

    def splitup_lists_for_parallelization(self, list_of_lists):
        ref_len = len(list_of_lists[0])
        assert all([len(list) == ref_len for list in list_of_lists]), "all lists must have the same length."
        list_of_lists_split = []
        num_per_thread = int(np.ceil(ref_len / self.num_threads))
        for list in list_of_lists:
            list_split = []
            for i in range(self.num_threads):
                list_split.append(list[i * num_per_thread: (i + 1) * num_per_thread])
            list_of_lists_split.append(list_split)

        return list_of_lists_split

    def generate_tight_masks(self, list_meshes, **tight_voxelize_kwargs):
        list_mask_min_corner = []
        list_z5_group_names = tight_voxelize_kwargs.get("list_z5_group_names",
                                                        [f"mesh_{i}" for i in range(len(list_meshes))])
        if self.parallel:
            # split up list of meshes:
            list_meshes_split, list_z5_group_names_split = self.splitup_lists_for_parallelization(
                [list_meshes, list_z5_group_names])
            print("meshes lenghts:", [len(list) for list in list_meshes_split], len(list_meshes))

            # run in parallel:
            path_z5_dataset = tight_voxelize_kwargs.get("path_z5_dataset", None)
            constrained_min_corner_int = tight_voxelize_kwargs.get("constrained_min_corner_int", None)
            constrained_max_corner_int = tight_voxelize_kwargs.get("constrained_max_corner_int", None)
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = executor.map(tight_voxelization_list_meshes,
                                       list_meshes_split,
                                       itertools.repeat(self.tqdm_bool),  # tqdm_bool
                                       np.arange(self.num_threads),  # tqdm_position for individual bars
                                       itertools.repeat(path_z5_dataset),  # path_z5_dataset
                                       list_z5_group_names_split,  # list_z5_group_names
                                       itertools.repeat(constrained_min_corner_int),  # constrained_min_corner_int
                                       itertools.repeat(constrained_max_corner_int),  # constrained_max_corner_int
                                       )

                result_list = list(results)

            # combine results:
            for result in result_list:
                list_mask_min_corner.extend(result)
        else:
            for ind, mesh in (enumerate(tqdm(list_meshes)) if self.tqdm_bool else enumerate(list_meshes)):
                foreground_mask, min_corner_int = tight_voxelize_single_mesh(mesh, **tight_voxelize_kwargs)
                list_mask_min_corner.append((foreground_mask, min_corner_int))

        return list_mask_min_corner

    def generate_volume(self, list_meshes):
        assert self.output_shape is not None, "please set output_shape before calling generate_volume."
        self.check_if_meshes_fit(list_meshes, self.output_shape)
        list_mask_min_corner = self.generate_tight_masks(list_meshes)
        output_vol, _ = from_tight_masks_to_label_volume(list_mask_min_corner, self.output_shape)
        return output_vol

    def generate_indiviual_volumes(self, list_meshes, list_out_shapes):
        assert len(list_meshes) == len(list_out_shapes), "list_meshes and list_out_shapes must have the same length."
        for mesh, output_shape in zip(list_meshes, list_out_shapes):
            self.check_if_meshes_fit([mesh], output_shape=output_shape)
        list_mask_min_corner = self.generate_tight_masks(list_meshes)
        list_output_vols = []
        for i, mask_min_corner in enumerate(list_mask_min_corner):
            output_vol, _ = from_tight_masks_to_label_volume([mask_min_corner], list_out_shapes[i])
            list_output_vols.append(output_vol)

        return list_output_vols
