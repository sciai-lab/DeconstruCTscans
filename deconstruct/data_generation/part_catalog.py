import copy
import numpy as np
import json
import trimesh as tr
import os
import re
import h5py
import pickle
import open3d as o3d
from tqdm.auto import tqdm
from dataclasses import dataclass

from deconstruct.utils.geometries.vtk_utils import vtk_decimate_trmesh
from deconstruct.utils.part_catalog_utils import create_streak_trafo_from_str, map_part_name_to_material, \
    export_load_watertightness_check, voxel_scale_from_config, clip_volumes_to_foreground, Rx, Ry, Rz, \
    get_bbox_center_for_voxelization
from deconstruct.utils.visualization.colorize_label_img import save_label_image_to_rgb_uint8
from deconstruct.data_generation.voxelization import MeshVoxelizer
from deconstruct.utils.general import (invert_trafo, convert_trmesh_to_mesh_for_voxelization, create_dirs,
                                      navigate_multiple_folder_out)


@dataclass
class VoxelizedPart:
    part_name: str
    trafo_to_canonical: np.ndarray
    binary_volume: np.ndarray = None


@dataclass
class PointCloudWithProperties:
    pcd: o3d.geometry.PointCloud
    global_properties: dict
    local_properties: dict
    subsampled_pcd: o3d.geometry.PointCloud = None
    subsampled_local_properties: dict = None
    subsampled_global_properties: dict = None


class MeshFromVolume:

    def __init__(self, binary_volume, smooth_trmesh, smooth_normals, label=None, part_name=None, iou_gt_match=None):
        self.binary_volume = binary_volume
        self.smooth_trmesh = smooth_trmesh
        self.smooth_normals = smooth_normals
        self.label = label
        self.part_name = part_name
        self.pcd_with_properties = self.extract_point_cloud_with_properties()
        self.iou_gt_match = iou_gt_match  # tuple of (gt_label, iou) or None

    def extract_properties(self):
        global_properties = {}
        local_properties = {}
        global_properties["volume"] = int(np.sum(self.binary_volume))
        global_properties["surface_area"] = self.smooth_trmesh.area

        return global_properties, local_properties

    def extract_point_cloud_with_properties(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(self.smooth_trmesh.vertices))
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(self.smooth_normals))
        global_properties, local_properties = self.extract_properties()
        return PointCloudWithProperties(pcd=pcd, global_properties=global_properties, local_properties=local_properties)


@dataclass
class PartMesh:
    stl_file: str
    part_name: str
    trmesh_in_canonical: tr.Trimesh
    list_trafo_from_canonical_to_scene: list[np.ndarray]
    part_material: str = None
    mesh_from_volume: MeshFromVolume = None
    voxelized_part: VoxelizedPart = None

    def create_meshes_in_scene(self):
        self.list_meshes_in_scene = []
        for trafo in self.list_trafo_from_canonical_to_scene:
            tr_mesh = copy.deepcopy(self.trmesh_in_canonical)
            tr_mesh.apply_transform(trafo)
            self.list_meshes_in_scene.append(tr_mesh)


class PartCatalog:

    def __init__(self, name, abs_path_to_data, decimate_reduction=0.2, streak_rotation_angles=None,
                 choose_materials="real_scan", verbose=True, path_part_info="default", path_windows="default",
                 path_output_folder="default", path_stl_folder="default",
                 path_replacement_folder="default", path_stls_need_fixing_folder="default"):
        """
            Loads the meshes from the stl files in "stl_catalog" of corresponding car.
            Expect meshes with ".stl" file extension.
            Every mesh is loaded only once.
            The meshes are decimated and checked for watertightness.

            Parameters
            ----------
            name : str
                Name of considered car. Defines the default paths to load the data.
            path_replacement_folder : str, optional
                Path to find fixed replacements of "broken" (e.g. non-watertight) meshes.
                Default = f"{cad2voxel_folder}/data/stls_watertight_replacements/".
            path_stls_need_fixing_folder : str, optional
                Path where "broken" (e.g. non-watertight) meshes are collected.
                Default = f"{cad2voxel_folder}/data/stls_that_need_fixing/".
            random_rotation : bool, optional
                Specify whether a random rotation should be applied to each mesh after loading it. Default = False.
            verbose : bool, optional
                Specify whether intermediate print statements should be shown. Default = True.
            decimate_reduction : float, optiona
                Lies between [0,1] and specifies how much reduction during vtk mesh decimation. Zero means no reduction.
                Default = 0.2.
            streak_rotation_angles : list[str] or None, optional
                List with entries of the form "x-20" meaning that a rotation of -20 degrees around the x-axis is applied to
                reduce streak artifacts in artist rendering. Default = ["x0", "y0", "z0"].
            check_watertight : bool, optional
                Specify if watertightness is checked. Default = True.
            choose_materials : str, optional
                Specifies how materials are assigned to part names. Choose from {"random", "real_scan"}. Default = "real_scan".
            path_to_part_info_file : str, optional
                Path to info file which contains information how to place the individual parts in the 3d scene.
                Default is os.path.join(path_to_stl_folder, "stl_catalog", f"{car_name}_info.json").
            path_stl_folder : str, optional
                Path to all find meshes. Default is f"{cad2voxel_folder}/data/{car_name}/".
            path_output_folder : str, optional
                Path where output volumes and h5 data file are saved.
                Default = f"{cad2voxel_folder}/data/output/{car_name}/{car_name}_{rotation_str_addition}/".

            Returns
            -------
            Dict[str, tr.Trimesh], list[str], list[np.ndarray]
                Dictionary of (partnames, trimeshes) containing all watertight meshes from stl_catalog (decimated and replaced),
                List of forward rotations (later on used to undo the individual random rotation trafos).
        """

        self.name = name
        self.abs_path_to_data = abs_path_to_data
        self.verbose = verbose
        self.decimate_reduction = decimate_reduction
        self.streak_trafo, self.streak_str_addition = create_streak_trafo_from_str(streak_rotation_angles)
        self.choose_materials = choose_materials

        # io paths:
        if path_part_info == "default":
            path_part_info = os.path.join(f"{self.abs_path_to_data}/{name}/",
                                          f"stl_catalog/{name}_info.json")
        if path_stl_folder == "default":
            path_stl_folder = navigate_multiple_folder_out(path_part_info, 1)
        if path_replacement_folder == "default":
            path_replacement_folder = f"{self.abs_path_to_data}/stls_watertight_replacements/"
        if path_stls_need_fixing_folder == "default":
            path_stls_need_fixing_folder = f"{self.abs_path_to_data}/stls_that_need_fixing/"

        if path_windows == "default":
            print("WARNING: default path_windows is not set.")
            path_windows = "fill_in"
        if path_output_folder == "default":
            path_output_folder = f"{self.abs_path_to_data}/output/{name}/{name}_{self.streak_str_addition}/"
        self.path_output_folder = path_output_folder
        self.artist_export_folder = os.path.join(path_stl_folder, f"{name}_for_rendering_{self.streak_str_addition}")
        self.path_windows = path_windows
        self.path_part_info = path_part_info
        self.path_stl_folder = path_stl_folder
        self.path_replacement_folder = path_replacement_folder
        self.path_stls_need_fixing_folder = path_stls_need_fixing_folder
        # create_dirs([self.path_stl_folder, self.path_replacement_folder, self.path_stls_need_fixing_folder])

        self.list_successfully_replaced = []
        self.list_failed_to_replace = []

        # more io specific attributes:
        self.path_artist_scan = None
        self.path_scan_config = None
        self.voxelization_scale = None
        self.artist_scan = None
        self.zyx_volume_shape = None
        self.shift_to_place_meshes_in_volume = None
        self.path_h5_dataset = None
        self.mapping_gt_labels_to_part_names = None

        # load the complete catalog:
        self.catalog_part_meshes, self.global_bbox_center, self.global_bbox = self.load_and_place_complete_catalog()

    @property
    def global_centered_bbox(self):
        if self.global_bbox is None or self.global_bbox_center is None:
            return None
        return self.global_bbox - self.global_bbox_center

    def load_and_check_single_stl(self, stl_file):
        """expects and absolute path to a single stl file"""
        if stl_file[-4:] != ".stl":
            return

        part_name = os.path.basename(stl_file).replace(".stl", "")

        # load triangular surface mesh from stl file:
        tr_mesh = tr.load(stl_file)

        # decimate mesh:
        # tr_mesh = tr_mesh.simplify_quadratic_decimation(face_count=decimate_face_count)
        # print("decimating mesh...", stl_file)  # uncomment this in case of segmentation fault.
        tr_mesh = vtk_decimate_trmesh(tr_mesh, reduction=self.decimate_reduction)
        # print("decimation done.")

        # check if part is watertight:
        if tr_mesh.is_watertight:
            pass
        else:
            # check for a repaired file and use that one instead:
            replacement_path = os.path.join(self.path_replacement_folder, part_name + ".stl")
            if os.path.exists(replacement_path):
                tr_mesh = tr.load(replacement_path)

                if tr_mesh.is_watertight:
                    # print(f"Non-watertight part {part_name} successfully replaced with modified part.")
                    if part_name not in self.list_successfully_replaced:
                        self.list_successfully_replaced.append(part_name)

                else:
                    print("WARNING: replacement is not working. Oh my")
                    return
            else:
                # print(f"Part {part_name} could not be replaced with watertight part.")
                if part_name not in self.list_failed_to_replace:
                    self.list_failed_to_replace.append(part_name)
                    tr.exchange.export.export_mesh(tr_mesh, os.path.join(self.path_stls_need_fixing_folder,
                                                                         f"{part_name}.stl"))

                # the part must be skipped and will not appear in the rendered volumes
                return

        return tr_mesh

    def load_and_place_complete_catalog(self):
        catalog_part_meshes = {}
        all_meshes_in_scene = []

        # load part info from json file:
        with open(self.path_part_info, "r") as f:
            # import all relevant part information (including the transformations to reassemble them):
            part_info = json.load(f)

        # load and transform all meshes to place them in the scene:
        for part in (tqdm(part_info) if self.verbose else part_info):
            part_name = part["part"]

            # load the corresponding mesh:
            stl_file = os.path.join(self.path_stl_folder, "stl_catalog", f"{part_name}.stl")

            trmesh = self.load_and_check_single_stl(stl_file)
            if trmesh is None:
                continue
            tr_mesh_canonical = copy.deepcopy(trmesh)

            # apply the transformation to translate and rotate the part to the correct pose in 3d:
            trafo = np.array(part["transformation"]).reshape((4, 4))
            trafo = self.streak_trafo @ trafo  # global streak trafo to avoid streak artifacts
            trmesh.apply_transform(trafo)

            # check watertightness again:
            trmesh = export_load_watertightness_check(trmesh, tmp_file_name=f"{self.name}_{part_name}_tmp.stl")
            if trmesh is None:
                continue

            # save the part mesh:
            if part_name in catalog_part_meshes:
                catalog_part_meshes[part_name].list_trafo_from_canonical_to_scene.append(trafo)
            else:
                catalog_part_meshes[part_name] = PartMesh(stl_file=stl_file,
                                                          part_name=part_name,
                                                          trmesh_in_canonical=tr_mesh_canonical,
                                                          list_trafo_from_canonical_to_scene=[trafo])
            all_meshes_in_scene.append(trmesh)

        # compute global bbox and bbox center:
        global_bbox_center, global_bbox = get_bbox_center_for_voxelization(all_meshes_in_scene)

        # apply global shift to center all meshes in scene:
        # and add it as additional step in the trafo:
        for part_mesh in catalog_part_meshes.values():
            for trafo in part_mesh.list_trafo_from_canonical_to_scene:
                trafo[:3, 3] -= global_bbox_center

        return catalog_part_meshes, global_bbox_center, global_bbox

    def prepare_files_for_artist(self):
        """
        Exports all meshes in scene to the artist folder and creates csv file with materials.
        """

        # create material string for csv file:
        create_dirs([os.path.join(self.artist_export_folder, "meshes_and_materials", "all_stls")])
        count = 0
        windows_path_material_string = ""
        for part_name, part_mesh in self.catalog_part_meshes.items():
            for trafo in part_mesh.list_trafo_from_canonical_to_scene:
                part_material = map_part_name_to_material([part_name],
                                                          abs_path_to_data=self.abs_path_to_data,
                                                          mode=self.choose_materials,
                                                          unknown_mode="random", verbose=False)[0]
                part_mesh.part_material = part_material
                windows_path_material_string += (self.path_windows + "\\" + f"{part_name}_{count}.stl"
                                                 + "\t" + part_material + "\n")

                # transform and export mesh:
                trmesh = copy.deepcopy(part_mesh.trmesh_in_canonical)
                trmesh.apply_transform(trafo)
                tr.exchange.export.export_mesh(trmesh, os.path.join(self.artist_export_folder, "meshes_and_materials",
                                                                    "all_stls", f"{part_name}_{count}.stl"))
                count += 1

        # save path material string:
        path_material_csv_file = os.path.join(self.artist_export_folder, "meshes_and_materials", f"{self.name}.csv")
        with open(path_material_csv_file, "w") as f:
            f.write(windows_path_material_string)

        # Create a dummy to manually configure artist setup
        # to keep it simple, just take a homogeneous box:
        bbox_arr = self.global_bbox[1] - self.global_bbox[0]
        box = tr.creation.box()
        box.vertices = box.vertices * bbox_arr[None, :]
        tr.exchange.export.export_mesh(box, os.path.join(self.artist_export_folder, f"dummy_for_{self.name}.stl"))
        print(f"Saved a dummy bounding box to gauge the geometry of the scan at "
              f"{os.path.join(self.artist_export_folder, f'dummy_for_{self.name}.stl')}")

        # finally, perform sanity check to see if "path_material_file.csv" matches the stls:
        with open(path_material_csv_file, "r") as f:
            lines = f.readlines()
        for line in lines:
            file = re.findall(r"all_stls\\.*.stl", line)[0][9:]
            assert os.path.exists(os.path.join(self.artist_export_folder, "meshes_and_materials", "all_stls", file)), \
                f"path material csv file does not match with the exported meshes in line {line}"

        print(f"Meshes for artist successfully exported to: {self.artist_export_folder}.")

    def load_artist_scan(self, path_scan_config="default", path_artist_scan="default"):
        """
            Loads raw volume and permute axes to match the zyx-convention used in the voxelization.

            Parameters
            ----------
            zyx_volume_shape : tuple
                Shape into which the raw volume will be reshaped. Has length 3.
            path_artist_scan : str
                Specify the path to the artist scan directly.

            Returns
            -------
            np.ndarray
                Raw input volume, i.e. artist scan, in the shape zyx_volume shape.
            """
        if path_scan_config == "default":
            path_scan_config = os.path.join(self.abs_path_to_data, f"artist_scans/{self.name}/"
                                                                   f"{self.name}_{self.streak_str_addition}/"
                                                                   f"configuration_00.json")

        if not os.path.exists(path_scan_config):
            print(f"Scan config not found at {path_scan_config}.")
            return None

        # manage shapes:
        scale, output_shape = voxel_scale_from_config(path_scan_config)
        output_shape.append(output_shape[0])
        zyx_volume_shape = tuple(output_shape[::-1])  # this is the shape that volumes should have in zyx convention.

        # manage shapes specific to raw file produced by vg max:
        permute_axes = (2, 0, 1)  # permute axes to match the zyx-convention used in the voxelization
        inverse_permute_axes = np.argsort(permute_axes)
        raw_shape = [zyx_volume_shape[i] for i in inverse_permute_axes]
        name_addition = ""
        for num in raw_shape[::-1]:
            name_addition += f"_{num}"

        if path_artist_scan == "default":
            path_artist_scan = os.path.join(os.path.dirname(path_scan_config), f"artist_scan_uint16{name_addition}.raw")

        if not os.path.exists(path_artist_scan):
            print(f"Artist scan not found at path: {path_artist_scan}.")
            return None

        self.zyx_volume_shape = zyx_volume_shape
        self.path_artist_scan = path_artist_scan
        self.path_scan_config = path_scan_config

        raw_input_volume = np.fromfile(path_artist_scan, dtype=np.uint16)
        assert np.prod(raw_shape) == len(raw_input_volume), f"Shape mismatch with raw_shape: {raw_shape}."
        raw_input_volume = raw_input_volume.reshape(raw_shape)
        # permute axes to match the rendered volume:
        raw_input_volume = np.transpose(raw_input_volume, axes=permute_axes)
        print("Successfully loaded raw artist scan!")
        assert raw_input_volume.shape == zyx_volume_shape, f"Shape mismatch with import of artist scan: " \
                                                           f"{raw_input_volume.shape} vs {zyx_volume_shape}."

        self.artist_scan = raw_input_volume
        return raw_input_volume

    def generate_voxel_gt_and_insseg_dataset(self, relative_scale_to_artist=1, num_threads=1,
                                             save_colored_insseg=False):
        """
            Loads all meshes in the 3d Scene and converts the meshes to representation used in the voxelization repo.
            returns: ready_meshes, shift, bbox
        """
        voxel_offset_correction = np.array([-0.5, 0., -0.5])
        meshes_offset_correction = np.array([-0.5, -0.5, -0.5])  # not clear why not the same as voxel_offset_corr.
        assert self.artist_scan is not None, "artist scan must be loaded first."

        scale, output_shape = voxel_scale_from_config(self.path_scan_config)
        output_shape.append(output_shape[0])
        # apply relative scale (to e.g. render in twice the resolution):
        if not np.isclose(relative_scale_to_artist, 1):
            output_shape = (np.asarray(output_shape) * relative_scale_to_artist).astype(int)
            scale = scale * relative_scale_to_artist
        voxelization_scale = np.mean(scale)
        self.voxelization_scale = voxelization_scale

        # create ready_meshes list by loading all stls:
        print("Preparing for voxelization...")
        volume_center = np.asarray(output_shape) / 2
        ready_meshes = []
        self.mapping_gt_labels_to_part_names = {0: "background"}
        gt_label = 1
        for part_name, part_mesh in self.catalog_part_meshes.items():
            for trafo in part_mesh.list_trafo_from_canonical_to_scene:
                trmesh = copy.deepcopy(part_mesh.trmesh_in_canonical)
                trmesh.apply_transform(trafo)
                # use the mesh representation used in the voxelization repo:
                ready_mesh = convert_trmesh_to_mesh_for_voxelization(trmesh)

                # shift them into the center of the volume and scale them:
                ready_mesh = ready_mesh * voxelization_scale + volume_center + voxel_offset_correction
                ready_meshes.append(ready_mesh)
                self.mapping_gt_labels_to_part_names[gt_label] = part_name
                gt_label += 1

        # to do the same trafo for meshes from voxelized catalog:
        self.shift_to_place_meshes_in_volume = -(volume_center + meshes_offset_correction) / voxelization_scale

        # init voxelizer and perform voxelization:
        mesh_voxelizer = MeshVoxelizer(output_shape, num_threads=num_threads, tqdm_bool=self.verbose)
        gt_instance_vol = mesh_voxelizer.generate_volume(ready_meshes)

        # clip the volumes to the foreground:
        clipped_volumes, clipping_min_corner = \
            clip_volumes_to_foreground([gt_instance_vol, self.artist_scan])
        gt_instance_vol, raw_input_volume = clipped_volumes

        # potentially save a colored instance segmentation GT volume to be loaded into VGstudio for visulization only:
        create_dirs([self.path_output_folder])
        if save_colored_insseg:
            save_label_image_to_rgb_uint8(gt_instance_vol,
                                          save_path=os.path.join(self.path_output_folder, "GT_seg_colored.raw"))

        # create h5 dataset:
        self.path_h5_dataset = os.path.join(self.path_output_folder,
                                            f'{self.name}_{self.streak_str_addition}_dataset.h5')
        with h5py.File(self.path_h5_dataset, "w") as f:
            f.attrs["name"] = self.name
            f.attrs["streak_str_addition"] = self.streak_str_addition
            f.attrs["voxelization_scale"] = self.voxelization_scale
            f.attrs["relative_scale_to_artist"] = relative_scale_to_artist
            f.attrs["raw_min"] = np.min(raw_input_volume)
            f.attrs["raw_max"] = np.max(raw_input_volume)
            f.attrs["clipping_min_corner"] = clipping_min_corner
            f.attrs["shift_to_place_meshes_in_volume"] = self.shift_to_place_meshes_in_volume
            f.attrs["semantic_label_list"] = [self.mapping_gt_labels_to_part_names[i] for i in
                                              range(len(self.mapping_gt_labels_to_part_names))]

            f.create_dataset("raw_input_volume", data=raw_input_volume, dtype=np.uint16)  # no compression
            f.create_dataset("gt_instance_volume", data=gt_instance_vol, dtype=np.uint16, compression="gzip")

        print(f"Creation of h5 dataset completed, see {os.path.abspath(self.path_h5_dataset)}.")

    def generate_voxelized_catalog(self, num_threads=1, random_rotation=True, margin=4):
        """
        For each stl in stl_catalog it does the following: load and potentially replace with watertight, then voxelize.
        Convention: To undo shifts, rotations and scales: Scaling is always the last and the first thing that is done!

        Exports the voxelized catalog as pickle file.
        """
        voxel_offset_correction = np.array([0, +0.5, 0])

        # create a unique list of meshes to voxelize:
        list_ready_meshes = []
        list_out_shapes = []
        for part_mesh in self.catalog_part_meshes.values():
            tr_mesh = copy.deepcopy(part_mesh.trmesh_in_canonical)

            # first random rotate and upscale:
            random_rot = np.eye(4)
            if random_rotation:
                th_x, th_y, th_z = np.random.random(3) * 2 * np.pi
                random_rot[:3, :3] = Rx(th_x) @ Ry(th_y) @ Rz(th_z)
            random_rot[:3, :3] = random_rot[:3, :3] * self.voxelization_scale
            tr_mesh.apply_transform(random_rot)

            out_shape = (tr_mesh.bounds[1] - tr_mesh.bounds[0]).astype(int) + margin

            # now shift into center of volume:
            shift_trafo = np.eye(4)
            shift_trafo[:3, 3] = - (tr_mesh.bounds[1] + tr_mesh.bounds[0]) / 2 + out_shape / 2
            tr_mesh.apply_transform(shift_trafo)

            ready_mesh = convert_trmesh_to_mesh_for_voxelization(tr_mesh)
            # apply voxel offset correction:
            ready_mesh = ready_mesh + voxel_offset_correction  # I checked that this is the correct offset

            list_ready_meshes.append(ready_mesh)

            voxel_part = VoxelizedPart(part_name=part_mesh.part_name,
                                       trafo_to_canonical=invert_trafo(shift_trafo @ random_rot))
            part_mesh.voxelized_part = voxel_part
            list_out_shapes.append(out_shape)

        # init voxelizer and perform individual voxelization:
        mesh_voxelizer = MeshVoxelizer(num_threads=num_threads, tqdm_bool=self.verbose)
        list_individual_volumes = mesh_voxelizer.generate_indiviual_volumes(list_ready_meshes, list_out_shapes)

        for i, voxel_part in enumerate(self.catalog_part_meshes.values()):
            voxel_part.voxelized_part.binary_volume = list_individual_volumes[i]

        print("Voxelization of catalog completed.")

    def save_part_catalog_for_later(self, save_path="default"):
        if save_path == "default":
            save_path = os.path.join(self.path_output_folder,
                                     f"{self.name}_{self.streak_str_addition}_part_catalog.pkl")

        # free up some memory before saving:
        self.artist_scan = None

        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

        # todo make this saving/loading process more robust than pickle
        # this is quite complicated since the data classes themselves are not easy to store

        print("Part catalog successfully saved at: ", save_path)
