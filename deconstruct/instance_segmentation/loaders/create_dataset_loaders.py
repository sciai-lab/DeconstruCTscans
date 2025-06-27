import torch
from torch.utils.data import DataLoader
from deconstruct.utils.open3d_utils.general_utils import deep_dict_update
import deconstruct.instance_segmentation.loaders.datasets as datasets
from deconstruct.instance_segmentation.loaders.transforms import configure_transforms


def create_dataset(dataset_config):
    """
    dataset config specifies which kind of dataset will be created
    """
    dataset_config = dataset_config.copy()
    dataset_name = dataset_config.pop("name")

    if hasattr(datasets, dataset_name):
        dataset = getattr(datasets, dataset_name)(**dataset_config)
    else:
        raise ValueError(f"Dataset {dataset_name} not found in {datasets.__name__}")

    return dataset


def create_data_loaders(loader_config):
    """
    loader config specifies which kind of dataset will be created
    """
    info_from_ds_to_model = {}
    general_dataset_config = loader_config["general_dataset_config"]
    train_dataset_config = deep_dict_update(general_dataset_config, loader_config["train_dataset_params"])
    valid_dataset_config = deep_dict_update(general_dataset_config, loader_config["valid_dataset_params"])

    train_dataset = create_dataset(train_dataset_config)
    info_from_ds_to_model.update(getattr(train_dataset, "info_from_ds_to_model", {}))
    valid_loader = None

    # potentially use custom collate:
    dummy_sample = train_dataset[0]
    collate_fn = None
    if hasattr(dummy_sample, "collate_fn"):
        collate_fn = dummy_sample.collate_fn

    if "valid_dataset_params" in loader_config:
        valid_index_files = loader_config["valid_dataset_params"].get("list_data_files", [])
        if len(valid_index_files) > 0:
            print(f"\nCreating a valid dataset from different index than train dataset, namely: {valid_index_files}. "
                  f"Make sure that the datasets are compatible (e.g. designed for the same task).")
            valid_dataset = create_dataset(valid_dataset_config)
        else:
            # split train_dataset into train and valid and concatenate with pure_valid_dataset:
            len_train_split = int(len(train_dataset) * loader_config["train_valid_split"])
            train_dataset, valid_dataset = torch.utils.data.random_split(
                train_dataset,
                [len_train_split, len(train_dataset) - len_train_split]
            )

            if "transforms_config" in valid_dataset_config:
                valid_dataset.dataset.transforms = configure_transforms(valid_dataset_config["transforms_config"])
            else:
                # use the transforms as specified in the train dataset
                pass

        valid_loader = DataLoader(valid_dataset, **loader_config["loader_params"],
                                  shuffle=False, drop_last=False, collate_fn=collate_fn)

    train_loader = DataLoader(train_dataset, **loader_config["loader_params"],
                              shuffle=True, drop_last=True, collate_fn=collate_fn)

    return train_loader, valid_loader, info_from_ds_to_model
