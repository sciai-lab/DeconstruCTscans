import pytorch_lightning as pl
import torch
import os
import h5py
import numpy as np

from pytorch3dunet.unet3d.model import UNet3D
from elf.segmentation.mutex_watershed import mutex_watershed
from deconstruct.instance_segmentation.inference.segmentation_prediction import compare_label_image_to_gt

from deconstruct.instance_segmentation.models.losses import configure_loss
from deconstruct.utils.logging.metric_logging import CustomMetric
from deconstruct.instance_segmentation.loaders.datasets.assembly_dataset import InsSegAffinityData
from deconstruct.utils.logging.firelight_logging import FirelightSegmentationVisualizer

sigmoid = torch.nn.Sigmoid()


class LitAffinityNet(pl.LightningModule):
    """Lightning version of PointNet++ regression model."""
    train_outputs: list
    valid_outputs: list

    def __init__(self,
                 net_config,
                 loss_config,
                 optimizer_config,
                 image_logging_config=None,
                 inference_logging_config=None,
                 scheduler_config=None,
                 metric_logging_config=None,
                 info_from_ds_to_model=None
                 ):
        super().__init__()

        # log hyperparameters and store in self.hparams:
        self.save_hyperparameters()

        # setup network:
        # print("info_from_ds_to_model: ", info_from_ds_to_model)
        net_config["out_channels"] = len(info_from_ds_to_model["offsets"]) + 1  # number of affinities + fg channel
        self.net = UNet3D(**net_config)

        # setup loss function:
        self.loss = configure_loss(loss_config)

        if metric_logging_config is None:
            self.train_metric = None
            self.valid_metric = None
        else:
            self.train_metric = CustomMetric(**metric_logging_config)
            self.valid_metric = CustomMetric(**metric_logging_config)

        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.inference_logging_config = inference_logging_config
        self.image_logging_config = image_logging_config

        if image_logging_config is None:
            self.fire_light_logger = None
        else:
            self.fire_light_logger = FirelightSegmentationVisualizer(
                path_firelight_config=image_logging_config["path_firelight_config"])

        self.info_from_ds_to_model = info_from_ds_to_model

        self.train_outputs = []
        self.valid_outputs = []

    def configure_optimizers(self):
        optimizer_config = self.optimizer_config.copy()
        optimizer_name = optimizer_config.pop("name")
        optimizer = getattr(torch.optim, optimizer_name)(self.parameters(), **optimizer_config)
        if self.scheduler_config is not None:
            scheduler_config = self.scheduler_config.copy()
            scheduler_name = scheduler_config.pop("name")
            scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)(optimizer, **scheduler_config)
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train/loss"}
        else:
            return optimizer

    def forward(self, data: InsSegAffinityData):
        out = sigmoid(self.net(data.raw))
        data.pred = out
        channel_loss = self.loss(out, data.affs)
        return data, channel_loss

    def log_channel_loss_(self, channel_loss, train_or_valid="train"):
        # average overbatch size:
        channel_loss = channel_loss.mean(dim=0)
        for i, loss in enumerate(channel_loss):
            self.log(f"{train_or_valid}_aff_loss/channel_{i}", loss, on_step=False, on_epoch=True)

    def training_step(self, batch, batch_idx):
        in_out_batch, channel_loss = self.forward(batch)
        if self.train_metric is not None:
            self.train_metric.single_batch_forward(in_out_batch, pred_key="pred", target_key="affs",
                                             ignore_target_value=self.info_from_ds_to_model["pad_value"])
        batch_size = self.batch_size(batch)
        self.log_channel_loss_(channel_loss, train_or_valid="train")
        self.log('train_loss', channel_loss.mean(), batch_size=batch_size, on_step=True, on_epoch=True)

        # NOTE: not moving in and out batch to cpu here, takes quite some memory.
        out_dict = {"channel_loss": channel_loss.detach().cpu(), "in_out_batch": in_out_batch.detach().to("cpu")}
        self.train_outputs.append(out_dict)
        return channel_loss.mean()

    def validation_step(self, batch, batch_idx):
        in_out_batch, channel_loss = self.forward(batch)
        if self.valid_metric is not None:
            self.valid_metric.single_batch_forward(in_out_batch, pred_key="pred", target_key="affs",
                                             ignore_target_value=self.info_from_ds_to_model["pad_value"])

        batch_size = self.batch_size(batch)
        self.log_channel_loss_(channel_loss, train_or_valid="valid")
        self.log('valid_loss', channel_loss.mean(), batch_size=batch_size, on_step=False, on_epoch=True)

        out_dict = {"channel_loss": channel_loss.detach().cpu(), "in_out_batch": in_out_batch.detach().to("cpu")}
        self.valid_outputs.append(out_dict)
        return channel_loss.mean()

    def run_inference(self, list_outputs, log_name="arand_score", random_index=True):
        """expects that the inputs are already detached!"""
        if self.inference_logging_config is None:
            return
        if self.current_epoch % self.inference_logging_config["frequency"] == 0 and self.current_epoch > 0:
            batch_idx = torch.randint(0, len(list_outputs), (1,)).item() if random_index else -1
            out_batch = list_outputs[batch_idx]["in_out_batch"]  # Retrieve the batch
            data_sample = out_batch.get_sample(0)  # Retrieve the first sample
            pred_affinities = data_sample.pred.numpy()[0]  # (aff_channel, img_shape)
            fg_mask = pred_affinities[-1] > 0.5  # foreground has target 1.
            affs = pred_affinities[:-1]
            mutex_watershed_kwargs = self.inference_logging_config.get("mutex_watershed_kwargs", {})
            mutex_watershed_kwargs.setdefault("strides", [1, 1, 1])
            segm = mutex_watershed(affs,
                                   offsets=self.info_from_ds_to_model["offsets"],
                                   mask=fg_mask,
                                   **mutex_watershed_kwargs)
            rand, arand = compare_label_image_to_gt(segm, data_sample.gt_insseg.numpy().reshape(*segm.shape))
            self.log(log_name, arand, on_step=False, on_epoch=True)

            save_path = self.inference_logging_config.get("save_path", None)
            if save_path is not None:
                if save_path == "default":
                    save_path = os.path.join(self.info_from_ds_to_model["logging_dir"],
                                             f"{log_name}_epoch_{self.current_epoch}.h5")
                with h5py.File(save_path, "w") as f:
                    f.attrs["rand"] = rand
                    f.attrs["arand"] = arand
                    f.create_dataset("segm", data=segm, dtype=np.int32, compression="gzip")
                    f.create_dataset("gt_segm", data=data_sample.gt_insseg.numpy().reshape(*segm.shape),
                                    dtype=np.int32, compression="gzip")



    @staticmethod
    def batch_size(data_input):
        return data_input.raw.shape[0]

    def log_metric(self, metric_instance, log_name="metric", subtract_metric_from=1):
        if metric_instance is not None and self.current_epoch > 0:
            metric = metric_instance.compute_metric()
            metric = subtract_metric_from - metric if subtract_metric_from is not None else metric  # to enable log plot
            self.log(log_name, metric, on_step=False, on_epoch=True)

    def log_prediction(self, list_outputs, log_name="prediction"):
        """expects that the inputs are already detached!"""
        if self.current_epoch % self.image_logging_config["frequency"] == 0 and self.current_epoch > 0:
            out_batch = list_outputs[-1]["in_out_batch"]  # Retrieve the batch
            data_sample = out_batch.get_sample(0)  # Retrieve the first sample
            log_img = self.fire_light_logger(input_img=data_sample.raw,
                                             target_img=data_sample.affs,
                                             pred_img=data_sample.pred,
                                             gt_segmentation=data_sample.gt_insseg)
            log_dict = {"tag": log_name, "img_tensor": log_img, "global_step": self.global_step}
            self.logger.experiment.add_image(**log_dict)

    def on_train_epoch_start(self):
        self.train_outputs = []

    def on_validation_epoch_start(self):
        self.valid_outputs = []

    def on_validation_epoch_end(self):
        self.log_metric(self.valid_metric, log_name="valid_acc")
        self.log_metric(self.train_metric, log_name="train_acc")
        if self.fire_light_logger is not None:
            self.log_prediction(self.valid_outputs, log_name="valid_pred")
            self.log_prediction(self.train_outputs, log_name="train_pred")
        self.run_inference(self.valid_outputs, log_name="valid_arand_score")
        self.run_inference(self.train_outputs, log_name="train_arand_score")
