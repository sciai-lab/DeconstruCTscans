import argparse
import yaml
import shutil
import os
# import torch_geometric
# import open3d as o3d
# import wandb
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import deconstruct.instance_segmentation.models as models
from deconstruct.instance_segmentation.loaders.create_dataset_loaders import create_data_loaders


def create_trainer(trainer_config, full_config_path=None, overwrite_run_dir=False):
    logger_config = trainer_config.pop("logger_config")
    save_dir = logger_config["TensorBoardLogger"]["save_dir"]
    logging_dir = os.path.join(save_dir, logger_config["TensorBoardLogger"]["name"])

    if logger_config["callbacks"]["ModelCheckpoint"]["dirpath"] == "default":
        logger_config["callbacks"]["ModelCheckpoint"]["dirpath"] = logging_dir

    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    else:
        print(f"WARNING: Run directory {logging_dir} already exists.")
        if "debug" in logger_config["TensorBoardLogger"]["name"] or overwrite_run_dir:
            print("Overwriting existing run directory.")
        else:
            print("Aborting.")
            exit()

    if full_config_path is not None:
        # copy and rename config file:
        shutil.copy(full_config_path, os.path.join(logging_dir, "config.yml"))
        print(f"Copied config file to logging folder for this run: {logging_dir}.")

        with open(full_config_path, "r") as f:
            full_config = yaml.load(f, yaml.SafeLoader)
    else:
        full_config = None

    callbacks = []
    # add learning rate logging:
    if "LearningRateMonitor" in logger_config["callbacks"]:
        callbacks.append(LearningRateMonitor(**logger_config["callbacks"]["LearningRateMonitor"]))

    # add model checkpointing:
    if "ModelCheckpoint" in logger_config["callbacks"]:
        callbacks.append(ModelCheckpoint(**logger_config["callbacks"]["ModelCheckpoint"]))

    # using tensorboard logger:
    logger = pl_loggers.TensorBoardLogger(**logger_config["TensorBoardLogger"])

    trainer = pl.Trainer(logger=logger, callbacks=callbacks,
                         default_root_dir=save_dir,
                         **trainer_config)
    return trainer, logging_dir


def train_litnet(path_config=None, path_checkpoint=None):
    if path_config is None:
        assert path_checkpoint is not None, "Either config_path or checkpoint_path must be specified."
        path_config = os.path.join(os.path.dirname(path_checkpoint), "config.yml")
        assert os.path.exists(path_config), (f"Config file {path_config} based on checkpoint_path {path_checkpoint} "
                                             f"does not exist.")

    # load the config file:
    with open(path_config, "r") as f:
        config = yaml.load(f, yaml.SafeLoader)
    print("config:", config)
    train_loader, val_loader, info_from_ds_to_model = create_data_loaders(config["loader_config"])
    trainer, logging_dir = create_trainer(config["trainer_config"],
                             full_config_path=path_config if path_checkpoint is None else None,
                             overwrite_run_dir=(path_checkpoint is not None))

    model_name = config["model_config"].pop("name")
    if path_checkpoint is None:
        info_from_ds_to_model["logging_dir"] = logging_dir
        config["model_config"]["info_from_ds_to_model"] = info_from_ds_to_model
        pl_model = getattr(models, model_name)(**config["model_config"])
        print("our model:", pl_model)
    else:
        pl_model = getattr(models, model_name).load_from_checkpoint(path_checkpoint)
        print("continue training from checkpoint:", path_checkpoint, "with model:", pl_model)

    trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader)


parser = argparse.ArgumentParser(description='Trains a pytorch lightning model')
parser.add_argument('-c', '--config', help='YAML Config-file', required=False, default=None)
parser.add_argument('-C', '--checkpoint', help='Path to checkpoint', required=False, default=None)
if __name__ == '__main__':
    args = vars(parser.parse_args())
    train_litnet(args["config"], args["checkpoint"])
