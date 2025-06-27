import os
import pickle
import re
import trimesh as tr
import json

import numpy as np

from deconstruct.utils.general import merge_trimeshes, Rx, Ry, Rz


def get_scale_for_voxelization(detector_resolution, dist_source_obj, dist_obj_detector):
    """
    Based on CT scan setup (all inputs are floats), calculate factor by which the vertex coordinates
    of a mesh are scaled.

    Parameters
    ----------
    detector_resolution : np.ndarray[float]
        Length of pixel (in mm) in x and y direction of detector of shape (2,).
    dist_source_obj : float
        Distance between source and scanned object in CT geometry.
    dist_obj_detector : float
        Distance between source and scanned object in CT geometry.

    Returns
    -------
    float
        The scale by which vertex coordinates in meshes have to be multiplied.
    """
    return (dist_source_obj + dist_obj_detector) / (dist_source_obj * detector_resolution)


def get_bbox_center_for_voxelization(list_meshes):
    """
    Based on the overall bounding box, calculate the shift by which the vertex coordinates of a mesh have to be shifted.
    input: .

    Parameters
    ----------
    list_meshes : list[tr.Trimesh] or list[np.ndarray]
        List of trimeshes or list of meshes in representation used in the voxelization step
        (see also convert_trmesh_to_mesh_for_voxelization functon below). From the list
        of these meshes the overall bounding box is calculated.

    Returns
    -------
    np.ndarray, list[np.ndarray]
        Center of the bounding box of shape (3,),
        The minimal and maximal coordinates of the bounding box.
    """
    assert isinstance(list_meshes, list), "input must be list."
    if isinstance(list_meshes[0], tr.Trimesh):
        merged_mesh = merge_trimeshes(list_meshes)
        min_arr, max_arr = merged_mesh.bounds
        bbox_center = (min_arr + max_arr) / 2
    else:
        min_arr = np.ones(3) * np.inf
        max_arr = - np.ones(3) * np.inf
        for mesh in list_meshes:
            min_arr = np.minimum(min_arr, np.amin(mesh, axis=(0, 1)))
            max_arr = np.maximum(max_arr, np.amax(mesh, axis=(0, 1)))
        bbox_center = (max_arr + min_arr) / 2

    return bbox_center, [min_arr, max_arr]


def voxel_scale_from_config(path_scan_config):
    """
    Calculates the scale applied to meshes during (artist) rendering.

    Parameters
    ----------
    path_scan_config : str
        Relative or absolute path to the scan_config JSON file.

    Returns
    -------
    (float, dict[str, Any])
        The scale by which vertex coordinates in meshes have to be multiplied,
        A copy of the dictionary loaded from the scan config dict.
    """
    assert os.path.exists(path_scan_config), f"scan config not found under path {path_scan_config}."
    # extract scan geometry from scan config:
    with open(path_scan_config, "r") as f:
        scan_config_dict = json.load(f)

    detector_resolution = np.asarray(scan_config_dict['detectorSize']) / np.asarray(scan_config_dict['detectorPixel'])
    if abs(detector_resolution[0] - detector_resolution[1]) > 1e-5:
        print("warning: detector resolution is not the same in x and y direction.")
    voxelization_scale = get_scale_for_voxelization(detector_resolution=np.mean(detector_resolution),
                                                    dist_source_obj=scan_config_dict['distanceSourceObject'],
                                                    dist_obj_detector=scan_config_dict['distanceObjectDetector'])

    return voxelization_scale, scan_config_dict["detectorPixel"].copy()


def create_streak_trafo_from_str(streak_rotation_angles):
    """small trafo to avoid streak artifacts in artist rendering"""
    if streak_rotation_angles is None:
        streak_rotation_angles = ["x0", "y0", "z0"]

    streak_trafo = np.eye(4)
    try:
        for angle_str in streak_rotation_angles:
            if angle_str[0] == "x":
                theta = int(angle_str[1:]) / 180 * np.pi
                streak_trafo[:3, :3] = streak_trafo[:3, :3] @ Rx(theta)
            elif angle_str[0] == "y":
                theta = int(angle_str[1:]) / 180 * np.pi
                streak_trafo[:3, :3] = streak_trafo[:3, :3] @ Ry(theta)
            elif angle_str[0] == "z":
                theta = int(angle_str[1:]) / 180 * np.pi
                streak_trafo[:3, :3] = streak_trafo[:3, :3] @ Rz(theta)
    except ValueError:
        print(f"non-valid input for streak_rotation_angles: {streak_rotation_angles}. No trafo will be performed.")

    streak_str_addition = ""
    for angle_str in streak_rotation_angles:
        streak_str_addition += angle_str

    return streak_trafo, streak_str_addition


def map_part_name_to_material(list_part_names, abs_path_to_data, list_materials="default", mode="random",
                              unknown_mode=4, path_material_info="default", base_material="Al", verbose=True):
    """
    Function to map a single part_name to a part-specific material chosen from list_materials.

    Parameters
    ----------
    list_part_names : list[str]
        List of part names to map to materials. Typically, a number but in some cases there is an extension to it.
    list_materials : list[str], optional
        List of materials to chose from. Default is "default" -> list of Al with densities [50, 65, 75, 85, 95].
    mode : str, optional
        Default = "random". Alternatively based on real scan use "real_scan".
    unknown_mode : str or int, optional
        Specifies how parts are treated for which no material info is known from real scans.
        If int, all unknown parts are mapped to the material corresponding to that index.
        if unknown_mode = "random", parts are mapped randomly based on their part names.
        Default = 5.
    base_material : str, optional
        either "Al" or "ABS". Default = "Al".
    path_material_info : str, optional
        Default = f"{cad2voxel_folder}/data/part_material_info/part_to_material.pkl"

    Returns
    -------
    list[str]
        Names of material chosen for the given part names.
    """
    if list_materials == "default":
        # use the default four abs materials
        densities = [50, 65, 75, 85, 95]  # order matters here.
        list_materials = [f"{base_material}." + str(x) for x in densities]
    if verbose:
        print(f"Make sure that the following materials: {list_materials} are configured in artist.")

    if mode == "random":
        list_part_materials = []
        for part_name in list_part_names:
            # use first number in part name as random seed:
            np.random.seed(int(re.findall(r"\d+", part_name)[0]))
            material_index = int(np.random.random() * len(list_materials))
            list_part_materials.append(list_materials[material_index])
    elif mode == "real_scan":
        # load part material info:
        path_material_info = f"{abs_path_to_data}/part_material_info/part_to_material.pkl" \
            if path_material_info == "default" else path_material_info
        assert os.path.exists(path_material_info), f"pickle file with path_material_info not found " \
                                                   f"at {path_material_info}."
        with open(path_material_info, 'rb') as f:
            part_to_material_dict = pickle.load(f)
        list_part_materials = get_materials_from_real_scan(list_part_names, list_materials, part_to_material_dict,
                                                           unknown_mode=unknown_mode, verbose=verbose)
    else:
        raise NotImplementedError(f"{mode=} not implemented.")

    return list_part_materials


def get_materials_from_real_scan(list_part_names, list_materials, part_to_material_dict, unknown_mode=5, verbose=True):
    """
    Assigns a material to a given part name.

    Parameters
    ----------
    list_part_names : list[str]
        List of part names to map to materials. Typically, a number but in some cases there is an extension to it.
    list_materials : list[str]
        List of materials to choose from.
    part_to_material_dict : dict[str, int]
        Mapping dict assigning (most) part names a material ind (ranging from 1 to 5).
    unknown_mode : str or int
        Specifies how parts are treated for which no material info is known from real scans.
        If int, all unknown parts are mapped to the material corresponding to that index.
        if unknown_mode = "random", parts are mapped randomly based on their part names.
        Default = 5.

    Returns
    -------
    list[str]
        Names of material chosen for the given part names.
    """
    list_part_materials = []
    count_unknown_parts = 0
    for part_name in list_part_names:
        part_name = re.findall(r"\w+", part_name)[0]  # extract first part of part name before - or _.
        material_index = part_to_material_dict.get(part_name, None)
        if np.issubdtype(type(material_index), np.integer):
            list_part_materials.append(list_materials[material_index - 1])  # -1 to shift from [1,5] to [0,4]
        else:
            # part is not listed in part_to_material_dict:
            count_unknown_parts += 1
            if np.issubdtype(type(unknown_mode), np.integer):
                list_part_materials.append(list_materials[unknown_mode])
            elif unknown_mode == "random":
                # use first number in part name as random seed:
                np.random.seed(int(re.findall(r"\d+", part_name)[0]))
                material_index = int(np.random.random() * len(list_materials))
                list_part_materials.append(list_materials[material_index])
            else:
                raise NotImplementedError(f"{unknown_mode=} not implemented.")

    if verbose:
        print(f"Number of unknown parts: {count_unknown_parts} / {len(list_part_materials)}"
              f" = {count_unknown_parts / len(list_part_materials) : 5.4f}.")

    return list_part_materials


def export_load_watertightness_check(tr_mesh, tmp_file_name="water_test_tmp.stl"):
    # export and load again:
    # this is a necessary before checking watertightness cause finite precision used to make problems here:
    tr_mesh.export(tmp_file_name)
    tr_mesh = tr.load(tmp_file_name)
    is_watertight = tr_mesh.is_watertight
    os.remove(tmp_file_name)

    if is_watertight:
        return tr_mesh
    else:
        return None



def clip_volumes_to_foreground(list_volumes, foreground_threshold=0.5, margin=5, verbose=True, channel_axis=None):
    """
    Takes a list of volumes and clips them to the foreground based on the first volume plus a threshold.

    Parameters
    ----------
    list_volumes : list[np.ndarray]
        list of volumes. must all have the same shape. First one is used to detect what is foreground.
    foreground_threshold : float, optional
        threshold to separate foreground > threshold from background. Default = 0.5 (for integer label images)
    margin : int, optional
        How much margin is added to the minimal foreground bounding box.
    verbose : bool, optional
        whether it should be printed how much the volume size was reduced.

    Returns
    -------
    (list[np.ndarray], np.ndarray)
        list of clipped volumes (same order as in input).
        the minimum corner that the volumes have been clipped to important for shifting segments into scene later.
    """
    assert isinstance(list_volumes, list), "input must be a list"
    # get foreground slice from first volume:
    shape_before = list_volumes[0].shape
    sl, min_corner = get_foreground_slice(list_volumes[0], foreground_threshold=foreground_threshold, margin=margin,
                                            channel_axis=channel_axis)
    for i, vol in enumerate(list_volumes):
        if len(vol.shape) != len(sl):
            assert channel_axis is not None, "please specify which axis is the channel axis."
            sl_ch = sl[:channel_axis] + (slice(None),) + sl[channel_axis:]
            list_volumes[i] = list_volumes[i][sl_ch]
        else:
            list_volumes[i] = list_volumes[i][sl]

    if verbose:
        print(f"Due to clipping, volume size was reduced by a factor of: \
              {np.prod(shape_before)/np.prod(list_volumes[0].shape) : .2f}.")

    return list_volumes, min_corner


def get_foreground_slice(vol, foreground_threshold=0.5, margin=5, channel_axis=None):

    # if vol has channels:
    if channel_axis is not None:
        vol = np.max(vol, axis=channel_axis)

    dim = len(vol.shape)
    foreground_mask = (vol > foreground_threshold)

    # determine bounding box of foreground
    mask_coords = np.asarray(np.where(foreground_mask)).transpose()
    min_corner = np.clip(np.min(mask_coords, axis=0) - margin, 0, None)
    max_corner = np.max(mask_coords, axis=0) + margin

    sl = tuple([slice(min_corner[i], max_corner[i] + 1) for i in range(dim)])

    return sl, min_corner
