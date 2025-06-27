import os
import yaml
import torch
import argparse
from deconstruct.utils.open3d_utils.general_utils import get_absolute_path_to_repo
from deconstruct.utils.general import split_streak_string
from deconstruct.data_generation.part_catalog import PartCatalog
from deconstruct.instance_segmentation.infer import insseg_inference
from deconstruct.proposal_selection.proposal_inference import proposal_inference
from deconstruct.utils.visualization.visualize_reconstruction_result import save_reconstruction_result_to_ply

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path_to_h5_data", type=str, required=True)
args = vars(parser.parse_args())
path_h5_dataset = os.path.abspath(args["path_to_h5_data"])
car_name = os.path.basename(os.path.dirname(os.path.dirname(path_h5_dataset)))
car_name_plus_streak = os.path.basename(os.path.dirname(path_h5_dataset))
streak_addition_suffix = car_name_plus_streak.split("_")[-1]
streak_angles = split_streak_string(streak_addition_suffix)
abs_path_to_data = os.path.dirname(os.path.dirname(os.path.dirname(path_h5_dataset)))

# programmatically set paths for inputs and outputs:
if not os.path.exists(path_h5_dataset):
    raise FileNotFoundError(f"The provided path to the car data does not exist: {path_h5_dataset}")
path_data_generation_config = os.path.join(get_absolute_path_to_repo(),
                                           "configs/data_generation_config.yml")
path_insseg_config = os.path.join(abs_path_to_data, "unet_model/config.yml")
path_checkpoint = os.path.join(abs_path_to_data, "unet_model/model-epoch=16684-valid_loss=0.01.ckpt")
path_reconstruction_config = os.path.join(get_absolute_path_to_repo(),
                                          "configs/reconstruction_inference_config.yml")

abs_path_output_folder = os.path.join(abs_path_to_data, "output", car_name, car_name_plus_streak)
path_part_catalog = os.path.join(abs_path_output_folder, "part_catalog.pkl")
path_h5_affinities = os.path.join(abs_path_output_folder, "affinities.h5")
path_h5_segmentation = os.path.join(abs_path_output_folder, "segmentation.h5")

print("Running this pipeline requires a GPU with at least 24GB of memory, about 10GB of disk space"
      " and 32 GB of RAM. It takes approximately 10 minutes to execute.")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")
if torch.cuda.get_device_properties(0).total_memory < 17e9:
    raise RuntimeError("CUDA device does not have enough memory.")

# PART CATALOG GENERATION:

os.makedirs(abs_path_output_folder, exist_ok=True)
# load config:
with open(path_data_generation_config, "r") as f:
    data_generation_config = yaml.load(f, yaml.SafeLoader)

data_generation_config["part_catalog_config"]["name"] = car_name
data_generation_config["part_catalog_config"]["abs_path_to_data"] = abs_path_to_data
data_generation_config["part_catalog_config"]["streak_rotation_angles"] = streak_angles

part_catalog = PartCatalog(**data_generation_config["part_catalog_config"])
part_catalog.voxelization_scale = 5.1853  # based on resolution of simulated CT scans
part_catalog.generate_voxelized_catalog(**data_generation_config["generate_voxelized_catalog_config"])
part_catalog.save_part_catalog_for_later(save_path=path_part_catalog)

# SEGMENTATION INFERENCE:

with open(path_insseg_config, "r") as f:
    insseg_config = yaml.load(f, yaml.SafeLoader)
insseg_config["inference_config"]["affinity_prediction_config"]["path_h5_dataset"] = path_h5_dataset
insseg_config["inference_config"]["affinity_prediction_config"]["path_h5_affinities"] = path_h5_affinities
insseg_config["inference_config"]["segmentation_prediction_config"]["path_h5_segmentation"] = path_h5_segmentation
path_h5_segmentation = insseg_inference(config=insseg_config, path_checkpoint=path_checkpoint, affs_only=False)[1]

# RECONSTRUCTION INFERENCE:

with open(path_reconstruction_config, "r") as f:
    reconstruction_config = yaml.load(f, yaml.SafeLoader)

reconstruction_config["proposal_generator_config"]["path_h5_segmentation"] = path_h5_segmentation
reconstruction_config["proposal_generator_config"]["path_part_catalog"] = path_part_catalog
reconstruction_config["proposal_generator_config"]["path_h5_dataset"] = path_h5_dataset

reconstruction_config["proposal_selector_config"]["results_folder"] = abs_path_output_folder
reconstruction_config["proposal_selector_config"]["path_z5_voxelization"] = os.path.join(abs_path_output_folder,
                                                                                         "voxelized_proposals.z5")

path_saved_results = proposal_inference(config=reconstruction_config)[1]

# VISUALIZATION:

save_reconstruction_result_to_ply(path_pkl_proposal_results=path_saved_results)
