import numpy as np
import os
from fancy import colors
from memory_profiler import profile

from deconstruct.utils.part_catalog_utils import clip_volumes_to_foreground
from deconstruct.utils.general import create_color_arr, rotate_volume


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


def colorize_label_image(label_img, ignore_label=0, seed=None):
    """
    Wrapper for colorize_segmentation from https://github.com/imagirom/fancy that remaps a label image to random
    RGB colors. Background is set to [0, 0, 0]. The mapping is needed for efficiency!

    Parameters
    ----------
    label_img : np.ndarray[int]
        Label image with non-negative integer labels.
    ignore_label : int, optional
        Background label in label image.

    Returns
    -------
    label_img : np.ndarray[int]
        Colored label image of shape img.shape + (3,).
    """

    # randomize colors:
    if seed is None:
        print("Careful using the fancy implementation to colorize the label image may take a lot of RAM.")
        new_label_img = remap_label_image(label_img)
        colored_label_img = colors.colorize_segmentation(new_label_img, ignore_label=ignore_label)
    else:
        colored_label_img = label_img_to_random_rgb(label_img, seed=seed, background_label=ignore_label)

    return colored_label_img


def save_rgb_uint8_for_vgmax(rgb_vol, save_path, add_shape_to_name=True, color_axis_is_first=False):
    """
    assumes to get uint8 three color channels as last channel, i.e. shape = (img_shape, 3).
    else: set color_axis_is_first=False
    """
    if not save_path.endswith(".raw"):
        save_path += ".raw"

    # for vg max the color axis has to be in the front:
    if color_axis_is_first:
        print("Moving color axis to back to match the VG studio convention for raw rgb volumes.")
        rgb_vol = np.moveaxis(rgb_vol, 0, -1)

    save_folder = os.path.dirname(save_path)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    name_addition = ""
    for num in rgb_vol.shape[:-1][::-1]:
        name_addition += f"_{num}"

    if add_shape_to_name:
        save_path = save_path.replace(".raw", name_addition + ".raw")

    rgb_vol.tofile(save_path)

    print(f"Created colored label image at {os.path.abspath(save_path)}.")
    return rgb_vol


# @profile
def save_label_image_to_rgb_uint8(label_img, save_path="colored_label_img.raw",
                                  add_shape_to_name=True,
                                  rotation_matrix=None,
                                  background_label=0, seed=0):
    """
    Colorizes a label image and saves it to raw file of type np.uint8. Background is set to [0, 0, 0].
    This is the desired format to visualize volumes in VG studio.

    Parameters
    ----------
    label_img : np.ndarray[int]
        Label image with non-negative integer labels.
    save_path : str, optional
        Relative or abs. path where colored image should be saved.
    add_shape_to_name : bool, optional
        Specify if shape should be added to name. Default is True.
    background_label : int, optional
        Background label in label image.

    Returns
    -------

    """
    colored_label_img = (colorize_label_image(label_img, ignore_label=background_label, seed=seed)
                         * 255).astype(np.uint8)
    if rotation_matrix is not None:
        colored_label_img = rotate_volume(colored_label_img, rotation_matrix,
                                                        interpolation_mode="bilinear", keep_dtype=True)
        colored_label_img = clip_volumes_to_foreground([colored_label_img], foreground_threshold=0.5,
                                                       channel_axis=0)[0][0]

    return save_rgb_uint8_for_vgmax(colored_label_img, save_path=save_path, add_shape_to_name=add_shape_to_name)


def label_img_to_random_rgb(img, seed=None, background_label=0, dtype=np.float16):
    """
    Turns float or int array in [0,1] into random rgb colors.

    Parameters
    ----------
    x_arr : np.ndarray[int]
    seed : int, optional
        How many digits should be converted to seed, i.e. np.random.seed(int(x * 10 ** digits_seed)).
        For ints use digit_seed = 0.

    Returns
    -------
    np.ndarray
        RGB color array of shape (n, 3).
    """

    color_arr = create_color_arr(np.max(img), seed=seed, background_label=background_label)
    color_arr = color_arr.astype(dtype)
    return color_arr[img]


