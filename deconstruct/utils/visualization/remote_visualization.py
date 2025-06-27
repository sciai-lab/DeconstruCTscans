import pickle
import os
from deconstruct.utils.open3d_utils.general_utils import add_file_extension


def save_visualization_obj_list_to_index(visualize_obj_list, name, save_path, draw_kwargs=None):
    draw_kwargs = {} if draw_kwargs is None else draw_kwargs
    draw_kwargs.setdefault("verbose", True)
    draw_kwargs["visualize_obj_list"] = visualize_obj_list

    # pickle the dict to the index folder:
    name = add_file_extension(name, "pkl")
    with open(save_path, "wb") as f:
        pickle.dump(draw_kwargs, f)

    print("Saved visualization to index folder: ", save_path)
