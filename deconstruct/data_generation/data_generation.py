import argparse
import yaml
from deconstruct.data_generation.part_catalog import PartCatalog


def data_generation(data_generation_config):

    # init stl catalog:
    part_catalog = PartCatalog(**data_generation_config["part_catalog_config"])

    # try to load artist scan:
    artist_scan = part_catalog.load_artist_scan(**data_generation_config["load_artist_config"])
    if artist_scan is None:
        print("Artist scan not found. Generating files needed to create artist scan.")
        part_catalog.prepare_files_for_artist()
        print("Everything ready for artist")
    else:
        # everything ready to generate data:
        part_catalog.generate_voxel_gt_and_insseg_dataset(**data_generation_config["generate_dataset_config"])

        # generate voxelized catalog:
        part_catalog.generate_voxelized_catalog(**data_generation_config["generate_voxelized_catalog_config"])
        print("Completed dataset and voxelized catalog generation.")

        part_catalog.save_part_catalog_for_later()


parser = argparse.ArgumentParser(description='data generation')
parser.add_argument('-c', '--config', type=str, required=True, help='path to config file')
if __name__ == '__main__':
    args = parser.parse_args()

    # load config:
    with open(args.config, "r") as f:
        data_generation_config = yaml.load(f, yaml.SafeLoader)

    data_generation(data_generation_config)