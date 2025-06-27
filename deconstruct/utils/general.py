import numba
import numpy as np
import torch
import torch.nn as nn
import trimesh as tr
from tqdm import tqdm
import pickle
import re
import open3d as o3d
import os
import subprocess
import skimage
import time
from contextlib import contextmanager
import sys


def shrink_dtype_label_arr(label_arr):
    unsigned = label_arr.min() >= 0
    if unsigned:
        if label_arr.max() <= 255:
            label_arr = label_arr.astype(np.uint8)
        elif label_arr.max() <= 65535:
            label_arr = label_arr.astype(np.uint16)
        else:
            label_arr = label_arr.astype(np.uint32)
    else:
        if label_arr.max() <= 127:
            label_arr = label_arr.astype(np.int8)
        elif label_arr.max() <= 32767:
            label_arr = label_arr.astype(np.int16)
        else:
            label_arr = label_arr.astype(np.int32)
    return label_arr


def bincount_unique(arr, fg_mask=None, return_counts=False):
    """
    faster than numpy unique if expected labels are not to large.
    pandas.unique is also quite fast, but does not return counts.
    """

    if fg_mask is None:
        if len(arr.shape) > 1:
            arr = arr.ravel()
    else:
        arr = arr[fg_mask]

    counts = np.bincount(arr)
    non_zero_count_mask = (counts > 0)
    nums = np.arange(len(counts))[non_zero_count_mask]
    if return_counts:
        counts = counts[non_zero_count_mask]
        return nums, counts
    else:
        return nums


def create_dirs(list_dirs, verbose=True, create_parent_dirs=True):
    if not isinstance(list_dirs, list):
        list_dirs = [list_dirs]
    for dir_path in list_dirs:
        if not os.path.exists(dir_path):
            if verbose:
                print(f"creating dir: {dir_path}")
            if create_parent_dirs:
                os.makedirs(dir_path, exist_ok=True)
            else:
                os.mkdir(dir_path)


def find_refernce_forder_inside_not_repo(file_name="reference_folder.here"):
    previous_set = ()
    current_folder = "."
    while True:
        current_set = set(os.listdir(current_folder))
        if file_name in current_set:
            return os.path.abspath(current_folder)

        if previous_set == current_set:
            print("Search for file: was not successful.")
            return None

        previous_set = current_set
        current_folder = os.path.join("../", current_folder)


def get_car_name_ext_from_dataset_file(dataset_path):
    file_name = os.path.basename(dataset_path)
    car_name_ext = None
    try:
        m = re.match(r"([\w-]+)_dataset.h5", file_name)
        car_name_ext = m.group(1)
    except Exception:
        print(f"failed to infer car_name_ext from {dataset_path}. please stick to file name convention: "
              f"car_name_ext_dataset.h5")

    return car_name_ext


def split_streak_string(streak_suffix):
    streak_angles = []
    current_part = ""
    for char in streak_suffix:
        if char in "xyz":
            if current_part:
                streak_angles.append(current_part)
            current_part = char
        else:
            current_part += char
    if current_part:
        streak_angles.append(current_part)
    return streak_angles


def mapping_insseg_to_semseg_from_semantic_label_list(semantic_label_list):
    """
    to then be used like
    GT_semantic_volume = mapping_arr_instance_to_semantic[GT_instance_volume]
    """
    if semantic_label_list[0] == "background":
        start_idx = 1
    else:
        start_idx = 0

    unique_label_list = ["background"] + list(set(semantic_label_list[start_idx:]))  # background should come first
    mapping_part_names_to_semantic_labels = {part_name: i for i, part_name in enumerate(unique_label_list)}
    # build mapping array from instance gt to semantic gt:
    mapping_instance_to_semantic = np.zeros(len(semantic_label_list), dtype=int)
    for instance_label, part_name in enumerate(semantic_label_list):
        mapping_instance_to_semantic[instance_label] = mapping_part_names_to_semantic_labels[part_name]

    # generate semantic GT volume using the indexing trick:
    return mapping_part_names_to_semantic_labels, mapping_instance_to_semantic


def get_absolute_path_to_repo_from_inside_repo():
    """from https://stackoverflow.com/questions/22081209/find-the-root-of-the-git-repository-where-the-file-lives"""
    repo_dir = subprocess.Popen(['git', 'rev-parse', '--show-toplevel'],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')
    if repo_dir == "":
        repo_dir = find_refernce_forder_inside_not_repo()
    assert repo_dir is not None, "tried to find the root path to repo and failed. " \
                                 "Please execute the script from inside the repo. " \
                                 "Or manually place a file named reference_folder.here where the git-folder would be."
    return repo_dir


def check_save_path(save_folder, file_name, extension=None, overwrite=False):
    """extension is expected to contain the dot."""
    extension = "" if extension is None else extension
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    exists_already = True
    num = 0
    while exists_already:
        new_complete_path = os.path.join(save_folder, f"{file_name}{num}{extension}")
        exists_already = os.path.exists(new_complete_path)
        num += 1

        if exists_already and overwrite:
            print(f"Since overwrite is specified as True, the existing file {new_complete_path} will be overwritten.")
            break

    return new_complete_path


def save_results_as_pkl(results_dict, save_folder, file_name, verbose=True, overwrite=False):
    """
    Saves results in form of dictionaries as a pkl file.

    Parameters
    ----------
    results_dict : dict
        dictionary containing results.
    save_folder : str
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
    if save_folder is None:
        # don't save anything
        return

    new_complete_path = check_save_path(save_folder, file_name, extension=".pkl", overwrite=overwrite)

    # pickle the results:
    with open(new_complete_path, 'wb') as f:
        pickle.dump(results_dict, f)

    if verbose:
        print(f"Results saved at {os.path.abspath(new_complete_path)}.")

    return os.path.abspath(new_complete_path)


def Rx(theta):
    """
    Rotation matrix around x-axis, which acts on arrays of shape (n,3) with entries in xyz-order.

    Parameters
    ----------
    theta : float
        Rotation angle between 0 and 2pi.

    Returns
    -------
    np.ndarray
        Rotation matrix of shape (3,3).
    """
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])

    np.concatenate


def Ry(theta):
    """
    Rotation matrix around y-axis, which acts on arrays of shape (n,3) with entries in xyz-order.

    Parameters
    ----------
    theta : float
        Rotation angle between 0 and 2pi.

    Returns
    -------
    np.ndarray
        Rotation matrix of shape (3,3).
    """
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])


def Rz(theta):
    """
    Rotation matrix around z-axis, which acts on arrays of shape (n,3) with entries in xyz-order.

    Parameters
    ----------
    theta : float
        Rotation angle between 0 and 2pi.

    Returns
    -------
    np.ndarray
        Rotation matrix of shape (3,3).
    """
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])


def invert_trafo(trafo):
    """
    Inverts a rotation-translation-trafo.

    Parameters
    ----------
    trafo : np.ndarray
        Trafo-matrix of shape (4, 4).

    Returns
    -------
    np.ndarray
        Inverse Trafo-matrix of shape (4, 4).
    """
    inverse_trafo = np.eye(4)
    inv_rot = trafo[:3, :3].T * np.linalg.det(trafo[:3, :3]) ** (-2 / 3)
    # inv_rot2 = np.linalg.inv(trafo[:3, :3])
    # assert np.allclose(inv_rot, inv_rot2), f"inverse rotation is not correct., {inv_rot}, {inv_rot2}"
    inverse_trafo[:3, :3] = inv_rot
    inverse_trafo[:3, 3] = - inv_rot.dot(trafo[:3, 3])

    return inverse_trafo


def pad_to_center_voxel(volume, voxel, pad_value=0):
    """
    Pads an image/volume such that a certain pixel is in the center of the image/volume

    Parameters
    ----------
    volume : np.ndarray
        Input image or volume of shape (H, W) or (D, H, W).
    voxel : np.ndarray[int]
        Indices (one for each direction of the input image/volume) to pick the voxel that should end up in the center.
    pad_value : float, optional
        Value by which parts out of the volume are padded. Default = 0.

    Returns
    -------

    """
    shape = volume.shape
    dim = len(shape)

    list_pad_tuple = []

    for i in range(dim):
        p = voxel[i]
        s = shape[i]
        pad_tuple = [0, 0]

        if p >= (s - 1) / 2:
            # pad on right side:
            pad_tuple[1] = 2 * p - s + 1
        else:
            # pad on left side:
            p = s - p - 1
            pad_tuple[0] = 2 * p - s + 1

        list_pad_tuple.append(tuple(pad_tuple))

    return np.pad(volume, pad_width=list_pad_tuple, mode='constant', constant_values=(pad_value, pad_value))


def merge_trimeshes(list_meshes):
    """
    Brute force merges a list of trimeshes
    adapted from: https://stackoverflow.com/questions/62317617/merging-two-3d-meshes/62555827#62555827

    Parameters
    ----------
    list_meshes : list[tr.Trimeshes]
        List of trimeshes to be merged.

    Returns
    -------
    tr.Trimesh
        Mesh resulting from merging.
    """
    vertices_list = [mesh.vertices for mesh in list_meshes]
    faces_list = [mesh.faces for mesh in list_meshes]  # stores the index tuples of vertices which from a triangle
    vertex_color_list = [mesh.visual.vertex_colors for mesh in list_meshes]

    # need to account for the offset:
    faces_offset = np.cumsum([v.shape[0] for v in vertices_list])
    faces_offset = np.insert(faces_offset, 0, 0)[:-1]

    vertices = np.vstack(vertices_list)
    faces = np.vstack([face + offset for face, offset in zip(faces_list, faces_offset)])

    merged_meshes = tr.Trimesh(vertices, faces, vertex_colors=np.vstack(vertex_color_list))

    return merged_meshes


def convert_trmesh_to_mesh_for_voxelization(tr_mesh, shift=None):
    """
    Convert tr.Trimesh to mesh representation used in the voxelization repo
    (https://github.com/cpederkoff/stl-to-voxel).

    Parameters
    ----------
    tr_mesh : tr.Trimesh
        Mesh to be converted.
    shift : np.ndarray, optional
        Shift to apply to mesh vertices of shape (3,). Default is np.zeros(3).

    Returns
    -------
    np.ndarray
        Mesh in the representation used in repo. Of shape (num of faces, 3, 3), i.e. for each face
        store all coordinates of vertices in that face.
    """

    shift = np.zeros(3) if shift is None else shift
    vertices = tr_mesh.vertices[tr_mesh.faces] + shift
    v0 = vertices[:, 0, :]
    v1 = vertices[:, 1, :]
    v2 = vertices[:, 2, :]
    return np.hstack((v0[:, np.newaxis], v1[:, np.newaxis], v2[:, np.newaxis]))


def rotate_volume(vol, rot_matrix, interpolation_mode="bilinear", keep_dtype=False):
    """
    inputs = numpy arrays, always expects first dimension to be channels
    """
    dim = len(vol.shape[1:])  # one dim is channels
    vol_float = vol.astype(float)  # is this necessary?
    # first pad the volume to have the same size in every directions:
    padded_vol = max_pad_volume(vol_float)

    theoretical_diag_scale = np.sqrt(dim)
    output_size = int(np.ceil(padded_vol.shape[1] * theoretical_diag_scale))
    diag_scale = output_size / padded_vol.shape[1]
    # print(f"{diag_scale=}", np.sqrt(dim))

    # create affine grid:
    affine_trafo = np.zeros((dim, dim + 1))
    affine_trafo[:dim, :dim] = diag_scale * rot_matrix
    affine_grid = nn.functional.affine_grid(theta=torch.from_numpy(affine_trafo[None, :]),
                                            size=((1, vol.shape[0]) + (output_size,) * dim), align_corners=False)

    # sample grid:
    # print(interpolation_mode, torch.from_numpy(padded_vol[None]).shape, affine_grid.shape)
    output_vol = nn.functional.grid_sample(torch.from_numpy(padded_vol[None]), grid=affine_grid,
                                           mode=interpolation_mode, padding_mode='zeros', align_corners=False).numpy()
    # restore the data type:
    if keep_dtype:
        output_vol = output_vol.astype(vol.dtype)
    return output_vol[0]


def max_pad_volume(vol):
    """
    expects first dimension to be the channels
    pads vol with zeros so that it has shape (max(vol.shape), ) * dim
    """
    vol_shape_diff = max(vol.shape[1:]) - np.asarray(vol.shape[1:])
    pad_width = [(int(np.floor(d / 2)), int(np.ceil(d / 2))) for d in vol_shape_diff]
    # padding in channel dim should be trivial:
    pad_width = [(0, 0)] + pad_width
    padded_vol = np.pad(vol, pad_width)
    assert (np.asarray(padded_vol.shape[1:]) - padded_vol.shape[1] == 0).all(), "must have same size in spatial dim"
    return padded_vol


def random_rgb_cmap(x_arr, digits_seed=5):
    """
    Turns float or int array in [0,1] into random rgb colors.

    Parameters
    ----------
    x_arr : np.ndarray[float or int]
        Array of numbers between [0,1] of shape (n, ).
    digits_seed : int, optional
        How many digits should be converted to seed, i.e. np.random.seed(int(x * 10 ** digits_seed)).
        For ints use digit_seed = 0.

    Returns
    -------
    np.ndarray
        RGB color array of shape (n, 3).
    """
    color_arr = np.zeros((len(x_arr), 3))
    for i, x in enumerate(x_arr):
        np.random.seed(int(x * 10 ** digits_seed))
        color_arr[i] = np.random.random(3)

    return color_arr


def create_color_arr(max_label, seed=None, background_label=0):
    if seed is not None:
        np.random.seed(seed)

    color_arr = np.random.random((max_label + 1, 3))
    # print("max_label", max_label, seed, color_arr)

    if seed is not None:
        np.random.seed()

    if background_label is not None:
        color_arr[background_label, :] = 0.

    return color_arr


def build_isotropic_3dstencil(long_neighbour_straight, long_neighbour_diag, long_neighbour_corner):
    """
    input specify repulsive connections by three lists
    output: isotropic 3dstencil
    """

    dim = 3
    # init and short range:
    straight_base = np.zeros((dim, dim), dtype=int)
    straight_base[np.arange(dim), np.arange(dim)] = -1
    stencil = np.copy(straight_base)

    # straight connections
    for d in long_neighbour_straight:
        stencil = np.concatenate([stencil, straight_base * d])

    # diag connections:
    # need to be linearly independent and half of the total number of diagonal neighbours (=12):
    diag_base = np.array([
        [-1, -1, 0],  # flip them along a non-zero axis (surely linearly indep.)
        [-1, 0, -1],
        [0, -1, -1],
        [1, -1, 0],
        [1, 0, -1],
        [0, 1, -1]
    ])

    for d in long_neighbour_diag:
        stencil = np.concatenate([stencil, diag_base * d])

    # corner connections:
    corner_base = np.array([
        [-1, -1, -1],  # flip this one along each axis
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ])

    for d in long_neighbour_corner:
        stencil = np.concatenate([stencil, corner_base * d])

    return stencil


def connected_components_with_label_mapping(label_volume, background=0, connectivity=1):
    """wrapper around skimage.measure.label; quite inefficient"""
    component_labeled_volume = skimage.measure.label(label_volume,
                                                     background=background,
                                                     connectivity=connectivity)

    # create mapping dict:
    new_labels, first_appearance_index, new_label_counts = np.unique(component_labeled_volume,
                                                           return_index=True,
                                                           return_counts=True)
    flat_label_vol = label_volume.ravel()
    label_mapping_dict = {new_labels[i]: flat_label_vol[ind] for i, ind in enumerate(first_appearance_index)}
    new_label_counts_dict = {new_labels[i]: count for i, count in enumerate(new_label_counts)}

    return component_labeled_volume, label_mapping_dict, new_label_counts_dict


def remap_label_image(label_img, dtype=np.uint16):
    """
    Remaps a label image to labels between 0 (background) and number of different labels. I.e. the zero label is kept,
    all other labels are kept in order but "gaps" are closed.

    Parameters
    ----------
    label_img : np.ndarray[int]
        Label image with non-negative integer labels.
    Returns
    -------
    np.ndarray[int]
        Label image relabelled.
    """
    assert label_img.min() >= 0, "labels should be non-negative."

    # remap for colorization:
    map_arr = np.zeros(int(np.max(label_img) + 1), dtype=dtype)
    for i, label in enumerate(np.unique(label_img)):
        map_arr[label] = i

    # label image indexing trick:
    return map_arr[label_img]


def navigate_multiple_folder_out(abs_path, num_folder_out=1):
    """
    Navigates num_folder_out folders out of abs_path.
    num_folder_out = 0 means that dirname is returned.
    """
    for i in range(num_folder_out + 1):
        abs_path = os.path.dirname(abs_path)
    return abs_path


def add_time_and_date_to_filenames(filenames, time_date_str=None):
    """
    Adds time and date to filename.
    """
    filenames = [filenames] if isinstance(filenames, str) else filenames
    time_date_str = time.strftime('%Y-%m-%d_%H-%M-%S') if time_date_str is None else time_date_str
    for i, filename in enumerate(filenames):
        if filename is None:
            continue
        filename, ext = os.path.splitext(filename)
        filenames[i] = f"{filename}_{time_date_str}{ext}"
    return filenames


def extract_time_and_date_from_filename(filename):
    """
    Extracts time and date from filename.
    """
    filename, ext = os.path.splitext(filename)
    return filename[-19:]


@numba.njit
def _get_bboxes(segmentation, labels_idx):
    """
    modified from https://github.com/hci-unihd/plant-seg/blob/master/plantseg/viewer/widget/proofreading/utils.py
    important: note that non-positive labels will be ignored
    """
    shape = segmentation.shape

    bboxes = {}
    for idx in labels_idx:
        if idx > 0:
            _x = np.array([[shape[0], shape[1], shape[2]], [0, 0, 0]])
            bboxes[idx] = _x

    for z in range(shape[0]):
        for x in range(shape[1]):
            for y in range(shape[2]):
                idx = segmentation[z, x, y]
                if idx > 0:
                    zmin, xmin, ymin = bboxes[idx][0]
                    zmax, xmax, ymax = bboxes[idx][1]

                    if z < zmin:
                        bboxes[idx][0, 0] = z
                    if x < xmin:
                        bboxes[idx][0, 1] = x
                    if y < ymin:
                        bboxes[idx][0, 2] = y

                    if z > zmax:
                        bboxes[idx][1, 0] = z
                    if x > xmax:
                        bboxes[idx][1, 1] = x
                    if y > ymax:
                        bboxes[idx][1, 2] = y
    return bboxes


def cut_label_volume_into_bboxes(segmentation, background_label=0, pad_width=0, verbose=True):
    """
    Cuts each instance in an instance segmentation into an individual binary mask.

    Parameters
    ----------
    label_volume : np.ndarray
        Image or volume from which instance masks are extracted.
    background_label : int, optional
        Label of background. Default = 0.
    pad_width : int, optional
        padding around segment with zeros. Default = 0 -> no padding.

    Returns
    -------
    dict[int, (np.ndarray, np.ndarray)]
        Labels are the keys. First entry is the masked volume, second is the lower left corner of the crop.
    """
    segmentation = segmentation.astype('int64')
    print("segmentation", segmentation.shape, segmentation.dtype, "this seems to be causing a warning here.")
    labels_idx = np.unique(segmentation)
    bboxes = _get_bboxes(segmentation, labels_idx)

    mask_dict = {}
    dim = len(segmentation.shape)
    for key, values in tqdm(bboxes.items(), disable=not verbose, desc="cut_label_volume_into_bboxes"):
        if key == background_label:
            continue
        min_corner = np.maximum(values[0] - pad_width, [0, 0, 0])
        max_corner = np.minimum(values[1] + pad_width, segmentation.shape)

        sl = tuple([slice(min_corner[i], max_corner[i] + 1) for i in range(dim)])  # + 1 is important here
        small_mask = (segmentation[sl] == key)

        # pad with zeros:
        small_mask = np.pad(small_mask, pad_width=(pad_width,), constant_values=(0, 0))
        mask_dict[key] = (small_mask, min_corner - pad_width)

    return mask_dict


@contextmanager
def silence_print():
    # Store the original sys.stdout to restore it later
    original_stdout = sys.stdout

    # Replace sys.stdout with a custom stream that discards output
    class NullIO:
        def write(self, text):
            pass

    sys.stdout = NullIO()

    try:
        yield
    finally:
        # Restore the original sys.stdout
        sys.stdout = original_stdout


