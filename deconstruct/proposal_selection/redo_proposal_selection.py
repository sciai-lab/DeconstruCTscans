import os
import pickle
from dataclasses import dataclass

import h5py
import numpy as np

from deconstruct.proposal_selection.proposal_selection import ProposalSelector
from deconstruct.proposal_generation.proposal_generation import report_proposal_quality


@dataclass
class PseudoSegmentationCatalog:
    path_h5_segmentation: str


def regenerate_proposal_selector(path_pkl_proposal_results, proposal_selector_config=None, results_folder="default"):
    proposal_selector_config = {} if proposal_selector_config is None else proposal_selector_config

    # load the proposal results:
    with open(path_pkl_proposal_results, "rb") as f:
        proposal_results = pickle.load(f)

    print(proposal_results.keys())
    with open(proposal_results["path_part_catalog"], "rb") as f:
        part_catalog = pickle.load(f)

    path_z5_voxelization = proposal_results["path_z5_voxelization"]
    path_h5_segmentation = proposal_results["path_h5_segmentation"]
    results_folder = os.path.dirname(path_pkl_proposal_results) if results_folder == "default" else results_folder
    dict_part_proposals = proposal_results["dict_part_proposals"]

    if "registration_scores_for_iou_matches" in proposal_results:
        report_proposal_quality(dict_part_proposals, proposal_results["registration_scores_for_iou_matches"])

    proposal_selector = ProposalSelector(
        part_catalog=part_catalog,
        segmentation_catalog=PseudoSegmentationCatalog(path_h5_segmentation=path_h5_segmentation),
        path_z5_voxelization=path_z5_voxelization,
        results_folder=results_folder,
        dict_part_proposals=dict_part_proposals,
        **proposal_selector_config
    )

    # load the proposals at the end which already have all properties computed:
    proposal_selector.get_successfully_voxelized_proposal_keys(proposal_selector.load_all_voxelized_proposals_in_memory)

    return proposal_selector, proposal_results


def redo_proposal_selection(path_pkl_proposal_results, proposal_selector_config=None, results_folder="default"):
    proposal_selector, proposal_results = regenerate_proposal_selector(
        path_pkl_proposal_results=path_pkl_proposal_results,
        proposal_selector_config=proposal_selector_config,
        results_folder=results_folder
    )

    # check if proposals have affinity score:
    dict_part_proposals = proposal_selector.dict_part_proposals
    example_proposal = dict_part_proposals[list(dict_part_proposals.keys())[0]]
    has_affinity_score = (example_proposal.affinity_score is not None)
    if not has_affinity_score:
        # compute affinity score:
        proposal_selector.compute_affinity_scores()

    # provide the predicted foreground mask as empty array just used for the shape:
    with h5py.File(proposal_results["path_h5_segmentation"], "r") as f:
        proposal_selector.predicted_foreground_mask = np.empty_like(f["final_segmentation"][:])

    selected_proposals = proposal_selector()

    # evaluate the proposal selection:
    pq_val, arand = proposal_selector.compare_voxel_solution_to_panoptic_gt(selected_proposals=selected_proposals)

    return pq_val, arand, proposal_selector, selected_proposals