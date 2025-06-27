import os
import pickle
import h5py
from deconstruct.proposal_selection.proposal_selection import ProposalSelector
from deconstruct.proposal_selection.redo_proposal_selection import PseudoSegmentationCatalog


def save_reconstruction_result_to_ply(path_pkl_proposal_results, path_export_for_vgmax="default"):
    if path_export_for_vgmax == "default":
        path_export_for_vgmax = os.path.join(os.path.dirname(path_pkl_proposal_results), "reconstruction_result.ply")

    # load the proposal results:
    with open(path_pkl_proposal_results, "rb") as f:
        proposal_results = pickle.load(f)

    print(proposal_results.keys())
    with open(proposal_results["path_part_catalog"], "rb") as f:
        part_catalog = pickle.load(f)

    dict_part_proposals = proposal_results["dict_part_proposals"]

    dummy_path_h5_segmentation = os.path.join(os.path.dirname(path_pkl_proposal_results), "dummy_segmentation.h5")
    with h5py.File(dummy_path_h5_segmentation, "w") as f:
        f.attrs["path_h5_affinities"] = "should_not_be_used"
    proposal_selector = ProposalSelector(
        part_catalog=part_catalog,
        segmentation_catalog=PseudoSegmentationCatalog(path_h5_segmentation=dummy_path_h5_segmentation),
        results_folder=os.path.abspath(""),  # not needed
        path_z5_voxelization=os.path.abspath(""),  # not needed
        dict_part_proposals={},
    )

    selected_proposals = [dict_part_proposals[key] for key in proposal_results["selected_proposal_keys"]]

    proposal_selector.create_mesh_scene_from_selected_proposals(
        selected_proposals=selected_proposals,
        include_gt_assembly=False,
        include_segmentation_assembly=False,
        combine_all=True,
        path_export_for_vgmax=path_export_for_vgmax,
        voxelization_scale=proposal_selector.part_catalog.voxelization_scale,
        save_path=os.path.join(os.path.dirname(path_pkl_proposal_results), "reconstructed_scene.pkl")
    )

    # delete dummy segmentation:
    os.remove(dummy_path_h5_segmentation)

