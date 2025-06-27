import open3d as o3d
import numpy as np
import os
import pickle
import subprocess
import collections
from typing import Union, List, Tuple, Dict, Any, Optional, Callable, Iterable, Sequence


def deep_dict_update(d, u, copy=True):
    """
    Recursively update a nested dictionary.

    :param d: dict to update
    :param u: dict to update with
    :return: updated dict
    """
    if copy:
        d = d.copy()

    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def set_defaults_recursively(config, defaults):
    """
    Updates a config dict with defaults recursively.
    Such that the dictionaries in defaults (dict of dicts) are used
    to update any dict in the config dict that has this name.

    :param config:
    :param defaults:
    :return:
    """
    assert isinstance(config, dict), "config must be a dict"
    for default_key in defaults:
        if default_key in config:
            # found the same key as in the defaults, update it:
            obj_to_update = config[default_key]
            # print("obj_to_update: ", obj_to_update, "updating with: ", defaults[default_key])
            if isinstance(obj_to_update, dict):
                # if its a dict, deep update it:
                defaults_copy = defaults[default_key].copy()
                defaults_copy = deep_dict_update(defaults_copy, obj_to_update)
                config[default_key] = defaults_copy
            elif isinstance(obj_to_update, list):
                # if its a list, update each element (which must be a dict):
                # print("detected a list, updating each element")
                for i, conf_to_update in enumerate(obj_to_update):
                    assert isinstance(conf_to_update, dict), "list elements must be dicts"
                    defaults_copy = defaults[default_key].copy()
                    defaults_copy = deep_dict_update(defaults_copy, conf_to_update)
                    config[default_key][i] = defaults_copy
            else:
                # if its neither a dict nor a list, it is hopefully a primitive type, which will not be overwritten:
                assert isinstance(obj_to_update, (int, float, str)) or obj_to_update is None, \
                    "Unknown type: " + str(type(obj_to_update))

        else:
            # have to go one level deeper:
            for key, value in config.items():
                if isinstance(value, dict):
                    set_defaults_recursively(value, defaults)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            set_defaults_recursively(item, defaults)
                        else:
                            # this is hopefully the end of the recursion, if not, we have a problem:
                            assert isinstance(item, (int, float, str)) or item is None, "Unknown type: " + str(type(item))
                else:
                    # this is hopefully the end of the recursion, if not, we have a problem:
                    assert isinstance(value, (int, float, str)) or value is None, "Unknown type: " + str(type(value))


def add_file_extension(file_name, ext):
    # remove dot from extension:
    ext = ext.replace(".", "")

    # add yml extension or replace current extension with yml:
    if "." not in file_name:
        # if file has no extension, add yml:
        file_name += f".{ext}"
    elif file_name.split(".")[-1] != ext:
        # if file has extension, replace it with yml:
        print(f"WARNING: path_to_index_file={file_name} does not have {ext} extension. Adding it.")
        file_name = file_name.split(".")[:-1] + f".{ext}"
    return file_name


def find_refernce_forder_inside_not_repo(file_name="reference_folder.here"):
    previous_set = ()
    current_folder = "."
    start_folder = os.path.abspath(current_folder)
    while True:
        current_set = set(os.listdir(current_folder))
        if file_name in current_set:
            return os.path.abspath(current_folder)

        if previous_set == current_set:
            print("Search for file: was not successful. Started from folder: ", start_folder)
            return None

        previous_set = current_set
        current_folder = os.path.join("../", current_folder)


def get_absolute_path_to_repo():
    # get the absolute path of this file:
    current_file_path = os.path.abspath(__file__)

    # now descend up to the git root folder:
    for _ in range(4):
        current_file_path = os.path.dirname(current_file_path)

    return current_file_path


def normalize_pointcloud(pcd, max_dist=1):
    """
    Shifts pcd to center and
    Normalizes it to be in sphere of radius max_dist around origin
    :param pcd: open3d.geometry.PointCloud
    :param max_dist: float
    :return: open3d.geometry.PointCloud
    """

    com = pcd.get_center()
    pcd.translate(-com)
    pcd.scale(scale=max_dist / np.max(np.linalg.norm(np.asarray(pcd.points), axis=1)), center=(0, 0, 0))

    return pcd


def shuffle_pointcloud(pcd, seed=None):
    """
    Shuffles the points in a pointcloud.
    :param pcd: open3d.geometry.PointCloud
    :return: open3d.geometry.PointCloud
    """
    # apply the same permutation to points, normals and colors:
    if seed is not None:
        np.random.seed(seed)
    perm = np.random.permutation(len(pcd.points))
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[perm])
    if pcd.has_normals():
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[perm])
    if pcd.has_colors():
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[perm])

    if seed is not None:
        np.random.seed()

    return pcd


def numpy_arr_to_o3d(arr):
    """
    Takes a numpy array and converts it to the appropriate o3d-array.

    Parameters
    ----------
    arr : np.ndarray

    Returns
    -------
    o3d.utility.Vector2iVector | o3d.utility.Vector2dVector | o3d.utility.Vector3iVector | o3d.utility.Vector3dVector
        array in o3d representation.

    """
    assert len(arr.shape) == 2, "an array of len(arr.shape) == 2 is expected."
    is_int = issubclass(arr.dtype.type, np.integer)
    # print("is_int?", is_int)
    if is_int and arr.shape[1] == 2:
        return o3d.utility.Vector2iVector(arr)
    elif is_int and arr.shape[1] == 3:
        return o3d.utility.Vector3iVector(arr)
    elif not is_int and arr.shape[1] == 2:
        return o3d.utility.Vector2dVector(arr)
    elif not is_int and arr.shape[1] == 3:
        return o3d.utility.Vector3dVector(arr)
    else:
        print(f"dtype={arr.dtype}, dim={arr.shape[1]}")
        raise NotImplementedError("not implemented conversion")


def save_results_as_pkl(results, save_path, file_name, verbose=True, overwrite=False):
    """
    Saves results in form of dictionaries as a pkl file.

    Parameters
    ----------
    results :
        results that can be pickled.
    save_path : str
        Path to where the file should be saved.
    file_name : str
        Name of file (assumes no pkl in filename). A number will be appended to avoid overwriting existing files.
    verbose : bool, optional
        Specifies if print statement is shown. Default = True.
    overwrite : bool, optional
        Specifies if existing files will be overwritten. Default = False.

    Returns
    -------

    """
    """assumes no pkl in filename"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # remove possible .pkl ending:
    if file_name.endswith(".pkl"):
        file_name = file_name[:-4]

    exists_already = True
    num = 0
    while exists_already:
        new_complete_path = os.path.join(save_path, f"{file_name}{num}.pkl")
        exists_already = os.path.exists(new_complete_path)
        num += 1

        if overwrite:
            break

    # pickle the results:
    with open(new_complete_path, 'wb') as f:
        pickle.dump(results, f)

    if verbose:
        print(f"Results saved at {os.path.abspath(new_complete_path)}.")