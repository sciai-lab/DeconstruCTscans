import argparse
import yaml
import h5py
from deconstruct.instance_segmentation.inference.affinity_prediction import AffinityPrediction
from deconstruct.instance_segmentation.inference.affinity_prediction_from_raw import AffinityPredictionFromRaw
from deconstruct.instance_segmentation.inference.segmentation_prediction import (SegmentationPrediction,
                                                                                SegmentationStitcher)
from deconstruct.utils.general import build_isotropic_3dstencil


def insseg_inference(path_config=None, config=None, path_checkpoint=None, affs_only=True, path_raw_input_config=None):
    if config is None:
        assert path_config is not None, "Either config or path_config must be provided."
        with open(path_config, "r") as f:
            config = yaml.load(f, yaml.SafeLoader)

    path_computed_h5_affinities = config["inference_config"].get("path_computed_h5_affinities", None)
    path_computed_h5_segmentations = config["inference_config"].get("path_computed_h5_segmentations", None)
    segmentation_prediction_config = config["inference_config"].get("segmentation_prediction_config", {})

    # load gt segmentation:
    affinity_prediction_config = config["inference_config"].get("affinity_prediction_config", {})
    path_h5_dataset = affinity_prediction_config.get("path_h5_dataset", None)
    if path_h5_dataset is None:
        print("No gt segmentation provided.")
        gt_segmentation = None
    else:
        print("Loading gt segmentation from ", path_h5_dataset)
        with h5py.File(path_h5_dataset, "r") as f:
            gt_segmentation = f["gt_instance_volume"][:]

    if path_computed_h5_segmentations is None:
        # have to compute affinities first:
        if path_computed_h5_affinities is None:
            affinitiy_prediction_config = config["inference_config"]["affinity_prediction_config"]
            # first affinity prediction:
            if path_checkpoint is not None:
                model_loading_kwargs = affinitiy_prediction_config.get("model_loading_kwargs", {})
                model_loading_kwargs["path_checkpoint"] = path_checkpoint
                affinitiy_prediction_config["model_loading_kwargs"] = model_loading_kwargs

            # handle the case if a raw input volume is provided:
            if path_raw_input_config is None:
                aff_pred = AffinityPrediction(config=config, **affinitiy_prediction_config)
            else:
                print(f"Predicting affinities for raw input volume provided at {path_raw_input_config}.")
                aff_pred = AffinityPredictionFromRaw(config=config, path_raw_input_config=path_raw_input_config,
                                                     **affinitiy_prediction_config)
                aff_pred.create_pseudo_part_catalog()  # create a pseudo part catalog from the raw input volume
            aff_pred()  # predicts all affinities and stitches them together, then save them to h5 file

            path_computed_h5_affinities = aff_pred.path_h5_affinities
            offsets = aff_pred.offsets
        else:
            offsets = build_isotropic_3dstencil(**config["loader_config"]["general_dataset_config"]
                                                ["affinity_transform_config"]["offsets_config"])
            print("Loading already computed affinities from ", path_computed_h5_affinities)

        if affs_only:
            print("Not performing segmentation prediction because affs_only is set to True.")
            return

        # now we can do the segmentation prediction:
        if ("aff_segmentor_config" not in segmentation_prediction_config or
                segmentation_prediction_config["aff_segmentor_config"] is None):
            segmentation_prediction_config["aff_segmentor_config"] = {}

        segmentation_prediction_config["aff_segmentor_config"]["offsets"] = offsets
        segm_pred = SegmentationPrediction(path_h5_affinities=path_computed_h5_affinities,
                                           gt_segmentation=gt_segmentation,
                                           **segmentation_prediction_config)
        final_segmentation = segm_pred()  # predicts all segmentations and stitches them, saves result to h5 file
        path_h5_segmentation = segm_pred.path_h5_segmentation
    else:
        if affs_only:
            print("Not performing segmentation prediction because affs_only is set to True.")
            return

        print("Loading already computed chunkwise segmentations from ", path_computed_h5_segmentations)
        segmentation_stitcher_config = segmentation_prediction_config.get("segmentation_stitcher_config", {})
        segm_stitcher = SegmentationStitcher(path_computed_h5_segmentations,
                                             gt_segmentation=gt_segmentation, **segmentation_stitcher_config)
        final_segmentation = segm_stitcher()
        path_h5_segmentation = segm_stitcher.path_h5_segmentation

    return final_segmentation, path_h5_segmentation


parser = argparse.ArgumentParser(description='Runs instance segmentation inference using a trained lightning model')
parser.add_argument('-c', '--config', help='YAML Config-file', required=False, default=None)
parser.add_argument('-C', '--checkpoint', help='Path to checkpoint', required=False, default=None)
parser.add_argument('-a', '--affs_only', help='Only compute affinities', action='store_true')
parser.add_argument('-s', '--segs_also', dest='affs_only', action='store_false')
parser.add_argument('-r', '--path_raw', help='Path to raw input volume', required=False, default=None)
parser.set_defaults(affs_only=True)
if __name__ == '__main__':
    args = vars(parser.parse_args())
    insseg_inference(path_config=args["config"], path_checkpoint=args["checkpoint"],
                     affs_only=args["affs_only"], path_raw_input_config=args["path_raw"])
