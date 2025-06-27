from deconstruct.instance_segmentation.loaders.transforms import custom_transforms


def configure_transforms(transform_configs):
    assert isinstance(transform_configs, list), f"transforms_config must be a list of dicts, each specifying a " \
                                                f"transform. Got {transform_configs}"
    transforms = []
    for transform_config in transform_configs:
        transform_config = transform_config.copy()
        transform_type = transform_config.pop("name")

        # get transform class from torch_geometric.transforms or from custom_transforms.py:
        if hasattr(custom_transforms, transform_type):
            transform = getattr(custom_transforms, transform_type)(**transform_config)
        else:
            raise ValueError(f"Transform {transform_type} not found in torch_geometric.transforms or "
                             f"custom_transforms.py")

        transforms.append(transform)

    return custom_transforms.CustomCompose(transforms)
