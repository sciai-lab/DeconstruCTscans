import h5py
import numpy as np
import trimesh as tr
import copy
from tqdm import tqdm
import time

from marching_cubes import march
from dataclasses import dataclass

from deconstruct.utils.general import rotate_volume, cut_label_volume_into_bboxes
from deconstruct.utils.calculating_overlaps import pairwise_overlap_of_segmentations_torch
from deconstruct.utils.geometries.mesh_utils import mesh_extract_largest_connected_component
from deconstruct.utils.geometries.vtk_utils import vtk_laplacian_smoothing_trmesh
from deconstruct.utils.geometries.mesh_smoothing import smooth_mesh_points_and_normals
from deconstruct.data_generation.part_catalog import MeshFromVolume


class MeshSmoother:
    laplacian_default_kwargs = {"lamb": 0.5, "iterations": 10}

    def __init__(self, method="laplacian", method_kwargs=None):
        self.method = method
        self.method_kwargs = {} if method_kwargs is None else method_kwargs
        if method == "laplacian":
            default_kwargs = self.laplacian_default_kwargs.copy()
            default_kwargs.update(self.method_kwargs)
            self.method_kwargs = default_kwargs

    def smooth(self, trmesh, inplace=False):
        if not inplace:
            trmesh = trmesh.copy()  # to prevent changes to trmesh

        if self.method == "laplacian":
            trmesh_smoothed = tr.smoothing.filter_laplacian(trmesh, **self.method_kwargs, volume_constraint=False)
            normals_smoothed = trmesh_smoothed.vertex_normals
        elif self.method == "neighborhood":
            # this option may be deprecated.
            print(f"WARNING: the option neighborhood for smoothing is deprecated.")
            trmesh_smoothed, normals_smoothed = smooth_mesh_points_and_normals(trmesh, **self.method_kwargs)
        else:
            raise NotImplementedError(f"{self.method=} not implemented.")

        return trmesh_smoothed, normals_smoothed


class MarchingSmoother:

    def __init__(self, iso_level=0.5, flip_zyx=True, use_largest_connected_component=False,
                 center_afterwards=False, smoothing_config=None, verbose=True):
        self.iso_level = iso_level
        self.flip_zyx = flip_zyx
        self.extract_largest_connected_component = use_largest_connected_component
        self.center_afterwards = center_afterwards
        self.verbose = verbose
        smoothing_config = {} if smoothing_config is None else smoothing_config
        self.mesh_smoother = MeshSmoother(**smoothing_config)

    @staticmethod
    def find_largest_connected_component(trmesh, verbose=True):
        trmesh_disconnected = None
        if trmesh.body_count > 1:
            if verbose:
                print(f"after marching cubes there are {trmesh.body_count} components!")

                # extract the main component (hoping that other components are quite small):
                print("extracting largest connected component... "
                      "this is implemented not efficiently and may take a while.")
            trmesh_disconnected = copy.deepcopy(trmesh)
            trmesh, _ = mesh_extract_largest_connected_component(trmesh, verbose=True)
        return trmesh, trmesh_disconnected

    def run_marching_cubes(self, volume):
        # make the volume binary to control that the level is used as isosurface
        binary_volume = (volume > self.iso_level) * 1
        start = time.time()
        verts, _, faces = march(binary_volume, 0)  # no smoothing here.
        # print(f"marching cubes took {time.time() - start} seconds.")

        if self.flip_zyx:
            verts = np.flip(verts, axis=1)  # to go from zyx to xyz convention

        # check that the mesh is connected:
        trmesh = tr.Trimesh(vertices=verts, faces=faces)
        trmesh_disconnected = None

        if self.extract_largest_connected_component:
            trmesh, trmesh_disconnected = self.find_largest_connected_component(trmesh, verbose=self.verbose)

        # optionally center mesh:
        if self.center_afterwards:
            shift_trafo = np.eye(4)
            shift_trafo[:3, 3] = - np.mean(trmesh.bounds, axis=0)
            trmesh.apply_transform(shift_trafo)

        # run smoothing on largest connected component:
        start = time.time()
        if self.mesh_smoother is not None:
            trmesh, normals = self.mesh_smoother.smooth(trmesh, inplace=True)
        else:
            normals = trmesh.vertex_normals
        # print(f"smoothing took {time.time() - start} seconds.")

        # check watertightness (although not terribly important):
        if not trmesh.is_watertight and self.verbose:
            print("mesh resulting from marching cubes is not watertight.")

        return trmesh, normals, trmesh_disconnected


class SegmentationCatalog:

    def __init__(self, path_h5_segmentation, path_h5_dataset, voxelization_scale,
                 marching_smoother_config=None, verbose=True, minimum_segment_size=0):
        """
        load segmentation from h5 file

        cut label volume into boxes

        use marching cubes to get mesh

        smooth mesh
        """
        self.verbose = verbose
        # load segmentation:
        self.path_h5_segmentation = path_h5_segmentation
        self.path_h5_dataset = path_h5_dataset
        self.voxelization_scale = voxelization_scale
        self.minimum_segment_size = minimum_segment_size

        # load segmentation:
        with h5py.File(path_h5_segmentation, "r") as f:
            self.segmentation = f["final_segmentation"][:]
            if "label_gtlabel_iou" in f.keys():
                mapping_labels_to_gt_labels = {int(label): (int(gt_label), iou)
                                               for (label, gt_label, iou) in f["label_gtlabel_iou"]}
            else:
                mapping_labels_to_gt_labels = None

            if "arand" in f.attrs.keys():
                print("arand score of instance segmentation:", f.attrs["arand"])
        print(f"loaded segmentation from {path_h5_segmentation}.")

        # load important info from dataset:
        if self.path_h5_dataset is None:
            # handle this part for the raw scan:
            self.clipping_min_corner = np.array([0, 0, 0])
            volume_center = np.asarray(self.segmentation.shape[::-1]) / 2  # flip cause shift is in mesh xzy space
            self.shift_to_place_meshes_in_volume = - volume_center / voxelization_scale
            self.semantic_label_list = None
        else:
            with h5py.File(self.path_h5_dataset, "r") as f:
                self.clipping_min_corner = f.attrs["clipping_min_corner"]
                self.shift_to_place_meshes_in_volume = f.attrs["shift_to_place_meshes_in_volume"]
                self.semantic_label_list = list(f.attrs["semantic_label_list"])

        if mapping_labels_to_gt_labels is not None:
            assert self.semantic_label_list is not None, "semantic_label_list must be given in dataset."

        # cut segmentation into boxes:
        self.segm_masked_volume_dict = cut_label_volume_into_bboxes(self.segmentation, background_label=0, pad_width=1,
                                                                    verbose=self.verbose)

        # create marching cubes and smoother instance:
        marching_smoother_config = {} if marching_smoother_config is None else marching_smoother_config
        marching_smoother_config["verbose"] = self.verbose
        self.marching_smoother = MarchingSmoother(**marching_smoother_config)

        self.catalog_segmentation_meshes = {}

        if self.verbose:
            print("Creating smooth segmentation mesh from individual segments...")
        for label in (tqdm(self.segm_masked_volume_dict) if self.verbose else self.segm_masked_volume_dict):
            # run smoothing:
            binary_volume, min_corner = self.segm_masked_volume_dict[label]
            if np.sum(binary_volume) < self.minimum_segment_size:
                continue
            smooth_trmesh, smooth_normals, _ = self.marching_smoother.run_marching_cubes(binary_volume)
            smooth_trmesh.vertices /= self.voxelization_scale  # rescale by voxelization_scale

            # shift to place them in scene (based on where they came from in the image):
            shift_to_scene = min_corner + self.clipping_min_corner
            if self.marching_smoother.flip_zyx:
                shift_to_scene = np.flip(shift_to_scene)
            shift_to_scene = shift_to_scene / self.voxelization_scale + self.shift_to_place_meshes_in_volume

            smooth_trmesh.apply_translation(shift_to_scene)
            if mapping_labels_to_gt_labels is None:
                iou_gt_match = ("none", -1)
            else:
                if mapping_labels_to_gt_labels[label][1] == -1:  # meaning no match found
                    iou_gt_match = ("none", -1)
                else:
                    iou_gt_match = (self.semantic_label_list[mapping_labels_to_gt_labels[label][0]],
                                    mapping_labels_to_gt_labels[label][1])
            segmentation_mesh = MeshFromVolume(
                label=label,
                binary_volume=binary_volume,
                smooth_trmesh=smooth_trmesh,
                smooth_normals=smooth_normals,
                iou_gt_match=iou_gt_match
            )
            self.catalog_segmentation_meshes[label] = segmentation_mesh


