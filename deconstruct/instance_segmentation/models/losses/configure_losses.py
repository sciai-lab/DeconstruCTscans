import torch.nn as nn
import deconstruct.instance_segmentation.models.losses as custom_losses

def configure_loss(loss_config):

    loss_config = loss_config.copy()
    loss_type = loss_config.pop("name")

    # get transform class from torch_geometric.transforms or from custom_transforms.py:
    if hasattr(custom_losses, loss_type):
        loss = getattr(custom_losses, loss_type)(**loss_config)
    elif hasattr(nn, loss_type):
        loss = getattr(nn, loss_type)(**loss_config)
    else:
        raise ValueError(f"Loss {loss_type} not found in torch.nn or custom_losses.py")

        transforms.append(loss)

    return loss