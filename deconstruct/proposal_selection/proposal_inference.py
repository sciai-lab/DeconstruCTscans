import yaml
import argparse
from deconstruct.utils.general import (save_results_as_pkl, extract_time_and_date_from_filename,
                                      add_time_and_date_to_filenames)
from deconstruct.proposal_generation.proposal_generation import ProposalGenerator
from deconstruct.proposal_selection.proposal_selection import ProposalSelector


def proposal_inference(path_config=None, config=None):
    # load the config file:
    if config is None:
        assert path_config is not None, "Either config or path_config must be provided."
        with open(path_config, "r") as f:
            config = yaml.load(f, yaml.SafeLoader)
        print("Successfully loaded config file.")

    proposal_generator = ProposalGenerator(**config["proposal_generator_config"])
    dict_part_proposals, dict_perfect_registrations, matching_dict = proposal_generator()
    registration_scores_for_iou_matches = proposal_generator.assess_proposal_quality(dict_part_proposals)

    proposal_selector = ProposalSelector(
        part_catalog=proposal_generator.part_catalog,
        segmentation_catalog=proposal_generator.segmentation_catalog,
        dict_part_proposals=dict_part_proposals,
        dict_perfect_registrations=dict_perfect_registrations,
        **config["proposal_selector_config"]
    )
    # save previous results before running selection:
    # to make the part proposals pickleable: an easy-fix is to remove non-pickleable o3d registration results:
    for part_proposal in dict_part_proposals.values():
        part_proposal.registration_result.o3d_registration_result = None
    results = {
        "dict_part_proposals": dict_part_proposals,
        "dict_perfect_registrations": dict_perfect_registrations,
        "matching_dict": matching_dict,
    }
    time_date_str = extract_time_and_date_from_filename(proposal_generator.path_h5_segmentation)
    save_results_as_pkl(
        results_dict=results,
        save_folder=proposal_selector.results_folder,
        file_name=f"saved_part_proposals_{time_date_str}"
    )
    proposal_selector.voxelize_and_compute_mwis_weights()
    selected_proposals = proposal_selector()

    # preform evaluations and visualizations:
    proposal_selector.create_mesh_scene_from_selected_proposals(selected_proposals, combine_all=True)  # or False?
    instance_solution_volume, instance_label_proposal_dict = (
        proposal_selector.create_instance_volume_from_selected_proposals(selected_proposals=selected_proposals))

    proposal_selector.create_semantic_volume_from_selected_proposals(
        selected_proposals=selected_proposals,
        instance_solution_volume=instance_solution_volume,
        instance_label_proposal_dict=instance_label_proposal_dict
    )

    if proposal_selector.part_catalog.path_h5_dataset is not None:
        proposal_selector.compare_voxel_solution_to_panoptic_gt(
            selected_proposals=selected_proposals,
            instance_solution_volume=instance_solution_volume,
            instance_label_proposal_dict=instance_label_proposal_dict
        )
    else:
        print("No panoptic GT available for comparison.")

    # save all results:
    results = {
        "path_part_catalog": proposal_generator.path_part_catalog,
        "path_h5_segmentation": proposal_generator.path_h5_segmentation,
        "path_z5_voxelization": proposal_selector.path_z5_voxelization,
        "registration_scores_for_iou_matches": registration_scores_for_iou_matches,
        "dict_part_proposals": dict_part_proposals,
        "dict_perfect_registrations": dict_perfect_registrations,
        "matching_dict": matching_dict,
        "selected_proposal_keys": [(p.label, p.part_name) for p in selected_proposals],
    }
    path_saved_results = save_results_as_pkl(
        results_dict=results,
        save_folder=proposal_selector.results_folder,
        file_name=f"proposal_selection_{time_date_str}"
    )

    return results, path_saved_results


parser = argparse.ArgumentParser(description='Generates and selects part proposals from an input segmentation.')
parser.add_argument('-c', '--config', help='YAML Config-file', required=False, default=None)
if __name__ == '__main__':
    args = vars(parser.parse_args())
    proposal_inference(args["config"])



