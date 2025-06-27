import os
import h5py
import numpy as np
import torch
import re
from tqdm import tqdm
import torchmetrics

import deconstruct.instance_segmentation.models as models
from deconstruct.utils.general import (navigate_multiple_folder_out, create_dirs,
                                      add_time_and_date_to_filenames)
from deconstruct.instance_segmentation.loaders.transforms.custom_transforms import TargetToAffinities
from deconstruct.instance_segmentation.loaders.datasets import SingleCarGridDataset


def get_run_name_from_insseg_config(config):
    """get run name from insseg config"""
    run_name = config["trainer_config"]["logger_config"]["TensorBoardLogger"]["name"]
    return run_name


def get_best_valid_checkpoint(dir):
    """expected file name format: model-epoch=6076-valid_loss=0.10.ckpt"""

    # get all files and sort by valid loss:
    list_files = os.listdir(dir)
    # files must match valid_loss={nr}.ckpt
    list_files = [f for f in list_files if re.match(r"model-epoch=\d+-valid_loss=\d+\.\d+\.ckpt", f)]
    # sort by valid loss:
    list_files = sorted(list_files, key=
                        lambda x: float(re.match(r"model-epoch=\d+-valid_loss=(\d+\.\d+)\.ckpt", x).group(1)))

    return os.path.join(dir, list_files[0])


def load_model_from_checkpoint(config, path_checkpoint=None, use_eval_mode=True, use_model_config_from_checkpoint=True,
                               device=None, verbose=False, logger_name="TensorBoardLogger"):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = {}
    if not use_model_config_from_checkpoint:
        print("Using model config from config file, not from checkpoint.")
        model_config = config["model_config"]

    if path_checkpoint is None:
        path_checkpoint = config["inference_config"]["path_trained_model"]
        if path_checkpoint == "default":
            save_dir = config["trainer_config"]["logger_config"][logger_name]["save_dir"]
            run_dir = os.path.join(save_dir, config["trainer_config"]["logger_config"][logger_name]["name"])

            # get last checkpoint:
            # path_checkpoint = os.path.join(run_dir, "last.ckpt")

            # get best valid checkpoint:
            path_checkpoint = get_best_valid_checkpoint(run_dir)

    if verbose:
        print("Loading the following checkpoint: ", path_checkpoint)

    model_name = config["model_config"]["name"]
    pl_model = getattr(models, model_name).load_from_checkpoint(path_checkpoint, **model_config)
    pl_model.to(device)
    if use_eval_mode:
        pl_model.eval()
    if verbose:
        print("Model used for inference: ", pl_model)

    return pl_model


def adapt_overlap_shape_to_offsets(offsets, overlap_shape, margin=3):
    # compute chunking of volume:
    min_overlap = np.abs(offsets).max() + margin  # overlap must be larger than stencil
    if overlap_shape is None:
        overlap_shape = (min_overlap,) * 3
    elif (np.asarray(overlap_shape) < min_overlap).any():
        overlap_shape = np.maximum(overlap_shape, min_overlap)
        print(f"overlap increased to minimal overlap for the given stencil size: {overlap_shape}")
    return overlap_shape


def extract_name_from_h5_dataset(path_h5_dataset):
    """get name from h5 dataset:"""
    # use re to match the following structure <name>_<streak_str_addition>_dataset.h5
    # to extract <name> and <streak_str_addition>:
    m = re.match(r"(.*)_(.*)_dataset\.h5", os.path.basename(path_h5_dataset))
    if m is None:
        raise ValueError(f"Could not extract name from h5 dataset: {path_h5_dataset}")
    name = m.group(1)
    streak_str_addition = m.group(2)
    return name, streak_str_addition


def get_default_h5_affinities_path(path_h5_dataset):
        # get name from h5 dataset:
        name, streak_str_addition = extract_name_from_h5_dataset(path_h5_dataset)
        # navigate 4 folders out from h5 dataset:
        data_folder = navigate_multiple_folder_out(path_h5_dataset, 4)
        return os.path.join(data_folder, "insseg_prediction",
                                          f"{name}", f"{name}_{streak_str_addition}", "affinities.h5")


class AffinityPrediction:

    sigmoid = torch.nn.Sigmoid()

    def __init__(self, config, path_h5_dataset, path_h5_affinities="default", rescale_input=True,
                 overlap_shape=None, shrinking_shape=None, model_loading_kwargs=None, verbose=True, gt_is_given=True):
        """load model and dataset"""

        # io paths:
        if path_h5_affinities == "default":
            path_h5_affinities = get_default_h5_affinities_path(path_h5_dataset)

        # create folder if not exists:
        create_dirs([os.path.dirname(path_h5_affinities)])

        # add time and date to path_h5_affinities:
        path_h5_affinities = add_time_and_date_to_filenames(path_h5_affinities)[0]

        model_loading_kwargs = {} if model_loading_kwargs is None else model_loading_kwargs
        model_loading_kwargs.setdefault("verbose", verbose)
        self.model = load_model_from_checkpoint(config, **model_loading_kwargs)
        self.config = config
        self.verbose = verbose
        self.path_h5_affinities = path_h5_affinities
        self.path_h5_dataset = path_h5_dataset
        self.gt_is_given = gt_is_given

        # create a grid dataset for predicting the affinities:
        aff_transform_config = config["loader_config"]["general_dataset_config"]["affinity_transform_config"]
        self.affinity_transform = TargetToAffinities(**aff_transform_config)
        self.offsets = self.affinity_transform.offsets
        self.chunk_shape = tuple(config["loader_config"]["general_dataset_config"]["chunk_shape"])
        overlap_shape = adapt_overlap_shape_to_offsets(self.offsets, overlap_shape)
        self.shrinking_shape = None if shrinking_shape is None else tuple(shrinking_shape)
        self.dataset = SingleCarGridDataset(overlap_shape=overlap_shape,
                                            shrinking_shape=shrinking_shape,
                                            chunk_shape=self.chunk_shape,
                                            path_h5_dataset=path_h5_dataset,
                                            affinity_transform_config=aff_transform_config,
                                            rescale_input=rescale_input,
                                            load_all_data_in_memory=False,
                                            gt_is_given=gt_is_given)

        # self.metric = torchmetrics.Accuracy(task="binary", average="weighted")
        self.metric = torchmetrics.classification.BinaryF1Score()

    def predict_affinities_on_single_chunk(self, raw_chunk):
        """predict affinities on single chunk"""
        with torch.no_grad():
            pred = self.sigmoid(self.model.net(raw_chunk.to(self.model.device)))
        return pred.cpu().reshape(-1, *raw_chunk.shape[2:])  # to shape (aff_channel, img_shape)

    def evaluate_prediction(self, pred, target=None, ignore_mask=None):
        if target is None:
            return
        ignore_mask = torch.zeros_like(target, dtype=bool) if ignore_mask is None else ignore_mask
        acc = self.metric(pred[~ignore_mask], target[~ignore_mask])
        self.metric.reset()
        return acc

    def __call__(self, spatial_h5_chunk_size=16):
        """makes single predictions and stitches them together"""

        # can not have all predictions in memory at once, init h5 dataset:
        full_aff_shape = (len(self.offsets) + 1, *self.dataset.img_shape)
        h5_chunk_shape = (len(self.offsets) + 1, ) + (spatial_h5_chunk_size, ) * len(self.dataset.img_shape)
        with h5py.File(self.path_h5_affinities, "w") as f:
            f.attrs["run_name"] = get_run_name_from_insseg_config(self.config)
            f.attrs["offsets"] = self.offsets
            f.create_dataset("summed_affs", shape=full_aff_shape, dtype=np.float32, chunks=h5_chunk_shape)
            f.create_dataset("counted_affs", shape=full_aff_shape, dtype=np.uint8, chunks=h5_chunk_shape)

        # create dummy input to investigate stencil dependent padding:
        dummy_target = torch.zeros((1, 1,) + self.chunk_shape)
        dummy_target = self.affinity_transform(target_img=dummy_target)[1]  # this already adds the bg/fg channel
        padded_mask = (dummy_target[0] == self.affinity_transform.pad_value)
        nonpadded_mask = torch.logical_not(padded_mask).numpy().astype(np.uint8)

        list_acc = []
        for insseg_data in (tqdm(self.dataset) if self.verbose else self.dataset):
            # get chunk:
            raw_chunk = insseg_data.raw
            affinities = self.predict_affinities_on_single_chunk(raw_chunk)
            affinities[padded_mask] = 0.  # and then divide by one less

            # evaluate prediction (if patch is not completely empty):
            if self.gt_is_given:
                if insseg_data.affs[0][~padded_mask].sum() > 0:
                    acc = self.evaluate_prediction(affinities, insseg_data.affs[0], ignore_mask=padded_mask)
                    list_acc.append(acc)

            # write affs to h5:
            left_corner = insseg_data.left_corner[0]
            if self.shrinking_shape is None:
                sl_affs = (slice(None),)  # trivial slicing in channels
                sl_affs += tuple([slice(left_corner[i], left_corner[i] + s) for i, s in enumerate(self.chunk_shape)])
                shrinked_slice = slice(None)
            else:
                shrinked_slice = [[s, -s] for s in self.shrinking_shape]
                sl_affs = [[left_corner[i], left_corner[i] + s] for i, s in enumerate(self.chunk_shape)]
                # check if we are at the boundary of the volume to adapt shinking shape there:
                for i in range(len(self.dataset.img_shape)):
                    if left_corner[i] == 0:
                        shrinked_slice[i][0] = 0
                    else:
                        # don't do this inplace it changes left_corner:
                        sl_affs[i][0] = sl_affs[i][0] + self.shrinking_shape[i]
                    if left_corner[i] + self.chunk_shape[i] == self.dataset.img_shape[i]:
                        shrinked_slice[i][1] = None
                    else:
                        # don't do this inplace it changes left_corner:
                        sl_affs[i][1] = sl_affs[i][1] - self.shrinking_shape[i]
                    assert left_corner[i] + self.chunk_shape[i] <= self.dataset.img_shape[i], \
                        f"should not happen, {left_corner, self.chunk_shape, self.dataset.img_shape}"
                shrinked_slice = (slice(None), ) + tuple([slice(*s) for s in shrinked_slice])
                sl_affs = (slice(None),) + tuple([slice(*s) for s in sl_affs])

            # print("left_corner", left_corner, "sl_affs", sl_affs, "shrinked_slice", shrinked_slice)
            with h5py.File(self.path_h5_affinities, "r+") as f:
                f["summed_affs"][sl_affs] += affinities.cpu().numpy()[shrinked_slice]
                f["counted_affs"][sl_affs] += nonpadded_mask[shrinked_slice]

        print("Done with affinity prediciton on full volume, see: ", self.path_h5_affinities)
        list_acc = [acc for acc in list_acc if acc is not None]
        if len(list_acc) > 0:
            print("Mean F1 score: ", np.mean(list_acc))