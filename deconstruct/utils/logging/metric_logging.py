import torch
import torchmetrics

from deconstruct.utils.logging import custom_metrics


def configure_activation(name, **kwargs):
    cls = getattr(torch.nn, name)
    return cls(**kwargs)


def configure_metric(name, **kwargs):
    if hasattr(custom_metrics, name):
        cls = getattr(custom_metrics, name)
    elif hasattr(torchmetrics, name):
        cls = getattr(torchmetrics, name)
    else:
        raise ValueError(f"Metric {name} not found in torchmetrics or custom_metrics")
    return cls(**kwargs)


class CustomMetric(torch.nn.Module):

    def __init__(self, metric_config, post_act=None, post_act_kwargs=None):
        super().__init__()
        metric_config = metric_config.copy()
        metric_name = metric_config.pop("name")
        self.metric = configure_metric(metric_name, **metric_config)
        self.post_act = None
        if post_act is not None:
            self.post_act = configure_activation(post_act, **post_act_kwargs)

    def single_batch_forward(self, in_out_batch, pred_key="x", target_key="y",
                             pred_batch_key=None, target_batch_key=None, ignore_target_value=None):
        needs_batch = (pred_batch_key is not None) and (target_batch_key is not None)
        pred = in_out_batch[pred_key].to(self.metric.device)
        target = in_out_batch[target_key].to(self.metric.device)

        if self.post_act is not None:
            pred = self.post_act(pred)

        if ignore_target_value is not None:
            ignore_mask = (target == ignore_target_value)
            pred = pred[~ignore_mask]
            target = target[~ignore_mask]

        if needs_batch:
            self.metric(pred, target,
                        pred_batch=in_out_batch[pred_batch_key].to(self.metric.device),
                        target_batch=in_out_batch[target_batch_key].to(self.metric.device))
        else:
            self.metric(pred, target)

    def compute_metric(self):
        acc = self.metric.compute()  # compute metric
        self.metric.reset()  # reset metric
        return acc.item() if isinstance(acc, torch.Tensor) else acc  # could be a dict of metrics

    def forward(self, list_in_out_batches, pred_key="x", target_key="y",
                pred_batch_key=None, target_batch_key=None, ignore_target_value=None):
        """expects and out_batch as input where the out attribute is the batched output"""

        for in_out_batch in list_in_out_batches:
            self.single_batch_forward(in_out_batch,
                                      pred_key=pred_key, target_key=target_key,
                                      pred_batch_key=pred_batch_key, target_batch_key=target_batch_key,
                                      ignore_target_value=ignore_target_value)

        return self.compute_metric()




