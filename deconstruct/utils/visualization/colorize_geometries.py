import re
import numpy as np
import hashlib


def get_random_weak_gray(lowest_value=0.5):
    """generate a random gray value with all values above lowest_value. Used as random colors in visualization."""
    return (np.random.random() * (1 - lowest_value) + lowest_value, ) * 3


def material_to_color(material):
    match = re.match(r"ABS.([\d]+)", material)
    if match is None:
        print(f"unknown material: {material}.")
    else:
        n = int(match.group(1))
        return ((100-n)/100, ) * 3


def part_name_to_color(part_name, use_k_digits=8):
    hash_object = hashlib.sha256(part_name.encode())
    hex_digest = hash_object.hexdigest()

    # Convert the hexadecimal hash to an integer
    numeric_value = int(hex_digest, 16)
    numeric_value = int(str(numeric_value)[:use_k_digits])
    np.random.seed(numeric_value)
    color = np.random.random(3)
    np.random.seed()
    return color


def get_color_for_o3d_geometry(choose_colors="random", material=None, part_name=None):
    if choose_colors == "random":
        color = np.random.random(3)
    elif choose_colors == "weak_grays":
        color = get_random_weak_gray()
    elif choose_colors == "by_material":
        assert material is not None, "no material specified to convert to color."
        color = material_to_color(material)
    elif choose_colors == "by_part_name":
        assert part_name is not None, "no part_name needed to convert to color."
        color = part_name_to_color(part_name)
    else:
        raise NotImplementedError(f"{choose_colors=} not implemted.")
    return color
