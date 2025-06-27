import os

import numpy as np
import copy
import h5py
import torch
import z5py
import time
import networkx as nx
from tqdm import tqdm
from skimage.segmentation import watershed
from torchmetrics.detection import PanopticQuality
from deconstruct.proposal_selection.panoptic_quality import MyPanopticQuality
from deconstruct.proposal_selection.mwis_solver import MWISSolver
from deconstruct.utils.calculating_overlaps import (perform_erosions, overlap_between_segments,
                                                   get_mask_and_corner_from_z5, calculate_pairwise_overlap_from_z5)
from deconstruct.utils.general import (convert_trmesh_to_mesh_for_voxelization, navigate_multiple_folder_out, extract_time_and_date_from_filename,
                                      add_time_and_date_to_filenames, create_dirs,
                                      mapping_insseg_to_semseg_from_semantic_label_list)
from deconstruct.data_generation.voxelization import MeshVoxelizer
from deconstruct.proposal_selection.affinity_score import AffinityScore
from deconstruct.utils.visualization.colorize_label_img import save_label_image_to_rgb_uint8, save_rgb_uint8_for_vgmax
from deconstruct.utils.visualization.visualize_assembly import (assemble_meshes_in_scene,
                                                               assemble_part_catalog_meshes_in_scene,
                                                               assemble_segmentation_meshes_in_scene)
from deconstruct.utils.visualization.remote_visualization import save_visualization_obj_list_to_index
from deconstruct.utils.visualization.colorize_geometries import get_color_for_o3d_geometry
from deconstruct.instance_segmentation.inference.segmentation_prediction import compare_label_image_to_gt


def compute_panoptic_quality(solution_panoptic_volume, gt_panoptic_volume, pq_config):
    """NOTE: The input tensor is assumed to have dimension ordering (B, spatial_dim0, ..., spatial_dim_N, 2)"""
    pq = PanopticQuality(**pq_config)
    pq(solution_panoptic_volume, gt_panoptic_volume)
    pq_val = pq.compute()
    pq.reset()
    return pq_val


def create_mapping_things_to_segments_from_mapping_arr(mapping_arr):
    mapping_things_to_segments = {}
    for instance_label, semantic_label in enumerate(mapping_arr):
        mapping_things_to_segments.setdefault(semantic_label, []).append(instance_label)
    return mapping_things_to_segments


class ProposalSelector:

    def __init__(self, part_catalog, segmentation_catalog, dict_part_proposals, dict_perfect_registrations=None,
                 verbose=True, path_z5_voxelization="default", results_folder="default", affinity_score_kwargs=None,
                 greedy_selection_kwargs=None, greedy_selection_only=False, mwis_selection_kwargs=None,
                 mwis_weight_kwargs=None, load_all_voxelized_proposals_in_memory=True, num_threads=8):
        self.predicted_foreground_mask = None
        self.foreground_bbox_zyx = None
        self.dict_z5_proposal_masks_and_min_corners = None
        self.load_all_voxelized_proposals_in_memory = load_all_voxelized_proposals_in_memory
        self.affinity_score_kwargs = {} if affinity_score_kwargs is None else affinity_score_kwargs
        self.greedy_selection_kwargs = {} if greedy_selection_kwargs is None else greedy_selection_kwargs
        self.mwis_selection_kwargs = {} if mwis_selection_kwargs is None else mwis_selection_kwargs
        self.mwis_weight_kwargs = {} if mwis_weight_kwargs is None else mwis_weight_kwargs
        self.greedy_selection_only = greedy_selection_only
        self.verbose = verbose
        self.part_catalog = part_catalog
        self.segmentation_catalog = segmentation_catalog
        self.dict_part_proposals = dict_part_proposals
        self.dict_perfect_registrations = dict_perfect_registrations
        self.num_threads = num_threads

        self.path_z5_voxelization = path_z5_voxelization
        self.time_date_str = extract_time_and_date_from_filename(self.segmentation_catalog.path_h5_segmentation)
        if self.path_z5_voxelization == "default":
            self.path_z5_voxelization = self.get_default_path_z5_voxelization()
        self.results_folder = results_folder
        if self.results_folder == "default":
            self.results_folder = os.path.dirname(self.get_default_path_z5_voxelization())
            self.results_folder.replace("voxelized_proposals", "proposal_selection_results")
        create_dirs([os.path.dirname(self.path_z5_voxelization)])
        print("Voxelization of part proposals will be saved to", self.path_z5_voxelization)
        print("Proposal selection results will be saved to the following folder:", self.results_folder)
        # print("time_date", self.time_date_str)

        with h5py.File(self.segmentation_catalog.path_h5_segmentation, "r") as f:
            self.path_h5_affinities = f.attrs["path_h5_affinities"]

    def voxelize_and_compute_mwis_weights(self, voxelized_already=False):
        # voxelize all part proposals:
        print("Voxelizing part proposals...")

        self.predicted_foreground_mask, self.foreground_bbox_zyx = self.get_foreground_and_foreground_bbox()
        if not voxelized_already:
            if os.path.exists(self.path_z5_voxelization):
                print("-" * 100)
                print("Be careful, the voxelization file already exists and will be overwritten."
                      "Only using the latest saved results will be reasonable to redo the proposal selection.")
                print("-" * 100)
            with z5py.File(self.path_z5_voxelization, "w") as f:
                # z5 can't use bool
                f.create_dataset("predicted_foreground_img", data=self.predicted_foreground_mask.astype(np.uint8))
            self.voxelize_proposals(num_threads=self.num_threads)
        self.get_successfully_voxelized_proposal_keys(self.load_all_voxelized_proposals_in_memory)
        self.get_proposal_foreground_overlaps()

        print("Computing mwis weights...")
        self.compute_mwis_weights(**self.mwis_weight_kwargs)

    def get_default_path_z5_voxelization(self):
        path_general_data = navigate_multiple_folder_out(self.segmentation_catalog.path_h5_segmentation, 3)
        car_name, car_name_ext = self.segmentation_catalog.path_h5_segmentation.split("/")[-3:-1]
        path_z5_voxelization = os.path.join(path_general_data, "proposal_selection", "voxelized_proposals",
                                            car_name, car_name_ext, "voxelized_proposals.z5")
        return add_time_and_date_to_filenames(path_z5_voxelization, self.time_date_str)[0]

    @staticmethod
    def compute_mwis_weight(part_proposal, score="affinity", multiply_with_foreground_overlap=True):
        """compute selection score for a single proposal"""
        if score == "affinity":
            part_proposal.mwis_weight = part_proposal.affinity_score
        elif score == "registration":
            part_proposal.mwis_weight = part_proposal.registration_result.registration_score
        elif score == "foreground":
            part_proposal.mwis_weight = part_proposal.abs_fg_overlap
        elif score == "foreground_diff":
            part_proposal.mwis_weight = 2 * part_proposal.abs_fg_overlap - part_proposal.abs_volume
        else:
            raise ValueError(f"score {score} not supported.")

        if multiply_with_foreground_overlap:
            part_proposal.mwis_weight *= part_proposal.abs_fg_overlap

    def compute_mwis_weights(self, score="affinity", multiply_with_foreground_overlap=True):
        """compute selection score for all proposals"""
        if score == "affinity":
            print("Computing affinity scores...")
            self.compute_affinity_scores()

        for part_proposal in tqdm(self.dict_part_proposals.values(), disable=not self.verbose, desc="mwis weights"):
            self.compute_mwis_weight(part_proposal, score=score,
                                     multiply_with_foreground_overlap=multiply_with_foreground_overlap)

    def get_foreground_and_foreground_bbox(self):
        # get foreground mask from segmentation result:
        print("Loading segmentation result for foreground mask from", self.segmentation_catalog.path_h5_segmentation)
        with h5py.File(self.segmentation_catalog.path_h5_segmentation, "r") as f:
            foreground_mask = (f["final_segmentation"][:] > 0.5)

        # since we are using the clipped volume as reference volume in voxelization:
        foreground_bbox_zyx = ((0, 0, 0), foreground_mask.shape)

        return foreground_mask, foreground_bbox_zyx

    def get_proposal_foreground_overlaps(self):
        """get foreground overlap for each proposal"""
        for part_proposal in tqdm(self.dict_part_proposals.values(), disable=not self.verbose, desc="fg overlap"):
            proposal_mask, proposal_min_corner = self.get_proposal_mask(part_proposal)
            fg_overlap = overlap_between_segments(proposal_mask, self.predicted_foreground_mask,
                                                  proposal_min_corner, (0, 0, 0))
            part_proposal.abs_fg_overlap = fg_overlap
            part_proposal.abs_volume = np.sum(proposal_mask)

    @staticmethod
    def get_z5_group_name(part_proposal):
        return f"proposal_{part_proposal.label}_{part_proposal.part_name}"

    def get_proposal_mask(self, part_proposal, try_to_use_memory=True):
        # load proposal mask:
        z5_group_name = self.get_z5_group_name(part_proposal)
        if try_to_use_memory:
            dict_z5_proposal_masks_and_min_corners = getattr(self, "dict_z5_proposal_masks_and_min_corners", None)
        else:
            dict_z5_proposal_masks_and_min_corners = None
        return get_mask_and_corner_from_z5(self.path_z5_voxelization, z5_group_name,
                                           dict_z5_proposal_masks_and_min_corners)

    def get_successfully_voxelized_proposal_keys(self, load_all_proposals_in_memory=True):
        self.dict_z5_proposal_masks_and_min_corners = {} if load_all_proposals_in_memory else None
        total_memory_size = 0  # in GB
        print("Checking which proposals were successfully voxelized...")
        if load_all_proposals_in_memory:
            print("Loading all proposals into memory.")
        list_proposal_keys_to_remove = []
        for key, part_proposal in tqdm(self.dict_part_proposals.items(), disable=not self.verbose):
            proposal_mask, proposal_min_corner = self.get_proposal_mask(part_proposal, try_to_use_memory=False)
            if proposal_mask is None:
                list_proposal_keys_to_remove.append(key)
            else:
                if load_all_proposals_in_memory:
                    z5_group_name = self.get_z5_group_name(self.dict_part_proposals[key])
                    self.dict_z5_proposal_masks_and_min_corners[z5_group_name] = (proposal_mask, proposal_min_corner)
                    total_memory_size += (proposal_mask.nbytes + proposal_min_corner.nbytes) / 1024 ** 3

        if load_all_proposals_in_memory:
            print(f"Loading completed. Total memory size of loaded proposals: {total_memory_size} GB.")

        # remove proposals which were not voxelized:
        for key in list_proposal_keys_to_remove:
            del self.dict_part_proposals[key]

        print(f"Out of {len(self.dict_part_proposals) + len(list_proposal_keys_to_remove)} proposals "
              f"{len(self.dict_part_proposals)} were successfully voxelized.")

    def compute_affinity_scores(self):
        """compute affinity score for each proposal"""
        affinity_score = AffinityScore(self.path_h5_affinities, **self.affinity_score_kwargs)
        for part_proposal in tqdm(self.dict_part_proposals.values(), disable=not self.verbose, desc="aff scores"):

            # load proposal mask:
            proposal_mask, proposal_min_corner = self.get_proposal_mask(part_proposal)
            assert proposal_mask is not None, "proposal mask is None. This should not happen."
            part_proposal.affinity_score = affinity_score(proposal_mask, proposal_min_corner)

    def voxelize_proposals(self, num_threads=8):
        """use voxelization as performed in part_catalog"""
        # exactly as in data generation:
        voxel_offset_correction = np.array([-0.5, 0., -0.5])

        mesh_voxelizer = MeshVoxelizer(num_threads=num_threads, tqdm_bool=self.verbose)

        list_ready_meshes = []
        list_z5_group_names = []
        print("Placing the meshes in the scene...")
        for part_proposal in tqdm(self.dict_part_proposals.values(), disable=not self.verbose):
            trmesh = copy.deepcopy(self.part_catalog.catalog_part_meshes[part_proposal.part_name].trmesh_in_canonical)
            trafo = part_proposal.registration_result.transformation
            trmesh.apply_transform(trafo)
            ready_mesh = convert_trmesh_to_mesh_for_voxelization(trmesh)

            # shift and scale to fit into volume:
            scale = self.segmentation_catalog.voxelization_scale
            volume_center = np.asarray(self.predicted_foreground_mask.shape[::-1]) / 2  # convert to xzy convention
            ready_mesh = ready_mesh * scale + volume_center + voxel_offset_correction
            list_ready_meshes.append(ready_mesh)
            list_z5_group_names.append(f"proposal_{part_proposal.label}_{part_proposal.part_name}")

        # start voxelization to z5 file:
        mesh_voxelizer.generate_tight_masks(
            list_meshes=list_ready_meshes,
            path_z5_dataset=self.path_z5_voxelization,
            list_z5_group_names=list_z5_group_names,
            constrained_min_corner_int=self.foreground_bbox_zyx[0][::-1],  # convert to xzy convention
            constrained_max_corner_int=self.foreground_bbox_zyx[1][::-1],  # convert to xzy convention
        )

        print("Voxelization of part proposals completed.")

    def preselect_proposals_based_on_fg_overlap(self, list_proposal_keys, minimum_rel_fg_overlap):
        list_proposal_keys = list(self.dict_part_proposals.keys()) if list_proposal_keys is None \
            else list_proposal_keys

        print(f"Filtering out all proposals with less than {minimum_rel_fg_overlap} fg overlap.")
        list_preselected_proposal_keys = []
        for proposal_key in list_proposal_keys:
            part_proposal = self.dict_part_proposals[proposal_key]
            rel_fg_overlap = part_proposal.abs_fg_overlap / part_proposal.abs_volume
            if rel_fg_overlap > minimum_rel_fg_overlap:
                list_preselected_proposal_keys.append(proposal_key)

        return list_preselected_proposal_keys

    def select_proposals_greedily(self, list_proposal_keys=None, relative_overlap_threshold=0.,
                                  perform_k_erosions=2, minimum_rel_fg_overlap=0.51):
        """greedy selection of proposals"""
        list_proposal_keys = self.preselect_proposals_based_on_fg_overlap(list_proposal_keys, minimum_rel_fg_overlap)
        # sort proposals by mwis weight (highest first):
        list_proposal_keys = sorted(list_proposal_keys, key=lambda x: self.dict_part_proposals[x].mwis_weight,
                                    reverse=True)

        # select proposals greedily:
        selected_proposals = []
        selected_foreground_mask = np.zeros_like(self.predicted_foreground_mask, dtype=bool)
        for proposal_key in tqdm(list_proposal_keys, disable=not self.verbose):
            part_proposal = self.dict_part_proposals[proposal_key]
            proposal_mask, proposal_min_corner = self.get_proposal_mask(part_proposal)
            if proposal_mask is None:
                print("proposal mask of proposal with key {proposal_key} is None. This should not happen.")

            proposal_mask = perform_erosions(proposal_mask, perform_k_erosions)
            eroded_volume = np.sum(proposal_mask)
            if eroded_volume == 0:
                print(f"For part proposal with key {proposal_key} all voxels were eroded away at the given resolution. "
                      "Consider using less erosions.")

            # check overlap with already accepted proposals:
            overlap = overlap_between_segments(proposal_mask, selected_foreground_mask, proposal_min_corner, (0, 0, 0))
            relative_overlap = overlap / max(eroded_volume, 1)

            if relative_overlap <= relative_overlap_threshold:
                selected_proposals.append(part_proposal)
                sl = tuple([slice(proposal_min_corner[i], proposal_min_corner[i] + s)
                            for i, s in enumerate(proposal_mask.shape)])
                selected_foreground_mask[sl] = np.logical_or(selected_foreground_mask[sl], proposal_mask)

        return selected_proposals

    def select_proposals_using_mwis(self, list_proposal_keys=None, relative_overlap_threshold=0.1,
                                    perform_k_erosions=0, minimum_rel_fg_overlap=0.51, mwis_solver_config=None):
        list_proposal_keys = self.preselect_proposals_based_on_fg_overlap(list_proposal_keys, minimum_rel_fg_overlap)
        list_z5_group_names = [self.get_z5_group_name(self.dict_part_proposals[key]) for key in list_proposal_keys]

        # compute overlap matrix between all proposals:
        print("Calculating pairwise overlaps...")
        overlap_matrix = calculate_pairwise_overlap_from_z5(
            list_z5_group_names, self.path_z5_voxelization,
            perform_k_many_erosions=perform_k_erosions,
            divide_by_smaller_volume=True,  # to make relative
            dict_z5_min_corners_and_masks=self.dict_z5_proposal_masks_and_min_corners
        )

        # threshold and build graph:
        connected_pairs = np.asarray(np.where(overlap_matrix > relative_overlap_threshold)).transpose()

        # build edge list:
        edge_list = []
        for i, j in connected_pairs:
            node1 = list_proposal_keys[i]
            node2 = list_proposal_keys[j]
            assert node1 != node2, "self-loops are not allowed"
            edge_list.append((node1, node2))

        # build weight_dict:
        weight_dict = {key: self.dict_part_proposals[key].mwis_weight for key in list_proposal_keys}
        mwis_graph = nx.Graph()
        mwis_graph.add_nodes_from(weight_dict.keys())   # proposal keys as node labels
        nx.set_node_attributes(mwis_graph, weight_dict, name="weight")
        mwis_graph.add_edges_from(edge_list)

        # mapping from nodes to numbers for mwis solvers:
        node_mapping_dict = {node: i for i, node in enumerate(mwis_graph.nodes())}
        inverse_node_mapping_dict = {i: node for i, node in enumerate(mwis_graph.nodes())}
        nx.relabel_nodes(mwis_graph, mapping=node_mapping_dict, copy=False)

        # compute mwis:
        start = time.time()
        mwis_solver_config = {} if mwis_solver_config is None else mwis_solver_config
        mwis_solver = MWISSolver(**mwis_solver_config)
        weight_sum, selected_indices = mwis_solver(mwis_graph)
        selected_proposal_keys = [inverse_node_mapping_dict[i] for i in selected_indices]
        print("MWIS solver completed in", time.time() - start, "seconds.")

        return [self.dict_part_proposals[key] for key in selected_proposal_keys]

    def __call__(self, list_proposal_keys=None):
        print("Starting with proposal selection...")
        start = time.time()
        if self.greedy_selection_only:
            selected_proposals = self.select_proposals_greedily(list_proposal_keys, **self.greedy_selection_kwargs)
        else:
            print("mwis_selection_kwargs", self.mwis_selection_kwargs)
            selected_proposals = self.select_proposals_using_mwis(list_proposal_keys, **self.mwis_selection_kwargs)
        print("Proposal selection completed in", time.time() - start, "seconds.")

        return selected_proposals

    def create_instance_volume_from_selected_proposals(self, selected_proposals, save_path="default",
                                                       overlap_rule="seeded_watershed"):
        """create voxel volume from selected proposals"""
        instance_solution_volume = np.zeros_like(self.predicted_foreground_mask, dtype=np.uint16)
        overlap_volume = np.zeros_like(self.predicted_foreground_mask, dtype=bool)

        if overlap_rule == "like_artist":
            # sort proposals based on part_name:
            # this relies on the fact that by now python dicts are ordered
            part_names_in_order = list(self.part_catalog.catalog_part_meshes.keys())
            selected_proposals = sorted(selected_proposals, key=lambda x: part_names_in_order.index(x.part_name))
        elif overlap_rule == "seeded_watershed":
            print("Using seeded watershed to resolve any overlaps.")
        else:
            raise ValueError(f"overlap_rule {overlap_rule} not supported.")

        print("Creating voxel volume from selected proposals...")
        instance_label_proposal_dict = {}
        for i, part_proposal in enumerate(selected_proposals):
            proposal_mask, proposal_min_corner = self.get_proposal_mask(part_proposal)
            sl = tuple([slice(proposal_min_corner[k], proposal_min_corner[k] + s)
                        for k, s in enumerate(proposal_mask.shape)])

            overlap_mask = np.logical_and(proposal_mask, instance_solution_volume[sl] != 0)
            instance_solution_volume[sl][proposal_mask] = i + 1  # 0 is background
            overlap_volume[sl][overlap_mask] = True

            instance_label_proposal_dict[(part_proposal.label, part_proposal.part_name)] = i + 1

        if overlap_rule == "seeded_watershed":
            # run seeded segmentation to fill in overlaps:
            instance_solution_volume[overlap_volume] = 0
            instance_solution_volume = watershed(
                image=np.zeros_like(instance_solution_volume),  # data where lowest values are labelled first
                markers=instance_solution_volume,  # seeds (background is also a seed, 0 means not a seed)
                mask=np.bitwise_or(overlap_volume, instance_solution_volume > 0)  # points that will be labelled
            )

        if save_path is not None:
            if save_path == "default":
                save_path = os.path.join(self.results_folder, f"instance_solution_volume_{self.time_date_str}.raw")
            save_label_image_to_rgb_uint8(instance_solution_volume, save_path=save_path)
            # save_label_image_to_rgb_uint8(1*overlap_volume,
            #                               save_path=save_path.replace(".raw", "_overlap.raw"))

        return instance_solution_volume, instance_label_proposal_dict

    def create_mesh_scene_from_selected_proposals(self, selected_proposals, include_gt_assembly=True,
                                                  include_segmentation_assembly=False, save_path=None,
                                                  **assemble_meshes_kwargs):
        """create mesh scene from selected proposals"""
        meshes = []
        names = []
        trafos = []
        colors = []
        for part_proposal in selected_proposals:
            part_mesh = self.part_catalog.catalog_part_meshes[part_proposal.part_name]
            meshes.append(part_mesh.trmesh_in_canonical)
            names.append(f"proposal_{part_proposal.label}_{part_proposal.part_name}")
            trafos.append(part_proposal.registration_result.transformation)
            colors.append(get_color_for_o3d_geometry("by_part_name",
                                                     part_mesh.part_material, part_mesh.part_name))

        assemble_meshes_kwargs.setdefault("combine_all", False)
        visualize_obj_list = assemble_meshes_in_scene(meshes, trafos=trafos, names=names, colors=colors,
                                                      **assemble_meshes_kwargs)

        if include_gt_assembly:
            assemble_meshes_kwargs = assemble_meshes_kwargs.copy()
            assemble_meshes_kwargs["combine_all"] = True
            visualize_obj_list += assemble_part_catalog_meshes_in_scene(self.part_catalog, **assemble_meshes_kwargs)
            visualize_obj_list[-1].is_visible = False

        if include_segmentation_assembly:
            print("Assembling segmentation meshes...")
            assemble_meshes_kwargs = assemble_meshes_kwargs.copy()
            assemble_meshes_kwargs["combine_all"] = True
            visualize_obj_list += assemble_segmentation_meshes_in_scene(self.segmentation_catalog,
                                                                        **assemble_meshes_kwargs)
            visualize_obj_list[-1].is_visible = False

        if save_path is not None:
            save_visualization_obj_list_to_index(visualize_obj_list, f"solution_scene_{self.time_date_str}",
                                                 save_path=save_path)

    def compare_voxel_solution_to_foreground(self, selected_proposals, instance_solution_vol=None,
                                             save_path="default", **voxel_solution_kwargs):
        """
        NOTE: this difference volume may be used to further iterate if there is to much red in it, i.e.
        too much of what is predicted as foreground is not explained by the proposals.
        It is very likely, that the foreground prediction is correct.
        So the proposals are not good enough or some are missing.
        blue is true foreground
        red is false foreground
        green is false background
        """
        if instance_solution_vol is None:
            voxel_solution_kwargs.setdefault("save_path", None)
            instance_solution_vol, _ = self.create_instance_volume_from_selected_proposals(selected_proposals,
                                                                                           **voxel_solution_kwargs)

        diff_volume = np.zeros(instance_solution_vol.shape + (3,), dtype=np.uint8)
        # true foreground:
        diff_volume[np.logical_and(instance_solution_vol > 0, self.predicted_foreground_mask)] = np.array([0, 0, 255])
        # false foreground:
        diff_volume[np.logical_and(instance_solution_vol > 0, ~self.predicted_foreground_mask)] = np.array([0, 255, 0])
        # false background:
        diff_volume[np.logical_and(instance_solution_vol == 0, self.predicted_foreground_mask)] = np.array([255, 0, 0])

        if save_path is not None:
            if save_path == "default":
                save_path = os.path.join(self.results_folder, f"solution_fg_diff_volume_{self.time_date_str}.raw")
            save_rgb_uint8_for_vgmax(diff_volume, save_path=save_path, color_axis_is_first=False)

        return diff_volume

    def create_semantic_volume_from_selected_proposals(self, selected_proposals, instance_solution_volume=None,
                                                       instance_label_proposal_dict=None,
                                                       save_path="default", **voxel_solution_kwargs):

        if self.part_catalog.path_h5_dataset is None:
            # no gt available:
            # randomly assign semantic labels:
            all_part_names = list(self.part_catalog.catalog_part_meshes.keys())
            mapping_part_names_to_semantic_labels = {part_name: i + 1 for i, part_name in enumerate(all_part_names)}
        else:
            # gt available:
            # get semantic gt:
            with h5py.File(self.part_catalog.path_h5_dataset) as f:
                semantic_label_list = list(f.attrs["semantic_label_list"])

            mapping_part_names_to_semantic_labels, gt_mapping_instance_to_semantic = (
                mapping_insseg_to_semseg_from_semantic_label_list(semantic_label_list))

        if instance_solution_volume is None or instance_label_proposal_dict is None:
            voxel_solution_kwargs.setdefault("save_path", None)
            instance_solution_volume, instance_label_proposal_dict = (
                self.create_instance_volume_from_selected_proposals(selected_proposals, **voxel_solution_kwargs))

        # create semantic solution volume:
        mapping_instance_to_semantic = np.zeros(len(selected_proposals) + 1, dtype=int)
        mapping_things_to_segments = {}
        for part_proposal in selected_proposals:
            instance_label = instance_label_proposal_dict[(part_proposal.label, part_proposal.part_name)]
            semantic_label = mapping_part_names_to_semantic_labels[part_proposal.part_name]
            mapping_instance_to_semantic[instance_label] = semantic_label
            if semantic_label not in mapping_things_to_segments:
                mapping_things_to_segments[semantic_label] = [instance_label]
            else:
                mapping_things_to_segments[semantic_label].append(instance_label)

        semantic_solution_volume = mapping_instance_to_semantic[instance_solution_volume]

        if save_path is not None:
            if save_path == "default":
                save_path = os.path.join(self.results_folder, f"semantic_solution_volume_{self.time_date_str}.raw")
            save_label_image_to_rgb_uint8(semantic_solution_volume, save_path=save_path)

        return semantic_solution_volume, mapping_things_to_segments, mapping_instance_to_semantic


    def compare_voxel_solution_to_panoptic_gt(self, selected_proposals,
                                              instance_solution_volume=None, instance_label_proposal_dict=None,
                                              background_label=0, infer_mapping_things_to_segments=False,
                                              **voxel_solution_kwargs):

        # create instance vol:
        voxel_solution_kwargs.setdefault("save_path", None)
        if instance_solution_volume is None or instance_label_proposal_dict is None:
            instance_solution_volume, instance_label_proposal_dict = (
                self.create_instance_volume_from_selected_proposals(selected_proposals, **voxel_solution_kwargs))

        # create semantic vol:
        semantic_solution_volume, mapping_things_to_segments, mapping_instance_to_semantic = (
            self.create_semantic_volume_from_selected_proposals(selected_proposals,
                                                                instance_solution_volume,
                                                                instance_label_proposal_dict,
                                                                **voxel_solution_kwargs))

        # create semantic gt:
        with h5py.File(self.part_catalog.path_h5_dataset) as f:
            gt_instance_volume = f["gt_instance_volume"][:]
            semantic_label_list = list(f.attrs["semantic_label_list"])
        print(f"Loaded gt_instance_volume from {self.part_catalog.path_h5_dataset}.")

        mapping_part_names_to_semantic_labels, gt_mapping_instance_to_semantic = (
            mapping_insseg_to_semseg_from_semantic_label_list(semantic_label_list))

        semantic_gt_volume = gt_mapping_instance_to_semantic[gt_instance_volume]

        # todo make this more ram efficient
        gt_panoptic_volume = np.stack([semantic_gt_volume, gt_instance_volume], axis=-1)
        solution_panoptic_volume = np.stack([semantic_solution_volume, instance_solution_volume], axis=-1)

        # check how long my instance intersection over union takes:
        my_panoptic_quality = MyPanopticQuality(classes=set(mapping_part_names_to_semantic_labels.values()),
                                                background_label=background_label)
        start = time.time()
        mapping_things_to_segments = None if infer_mapping_things_to_segments else (
            create_mapping_things_to_segments_from_mapping_arr(mapping_instance_to_semantic))
        gt_mapping_things_to_segments = None if infer_mapping_things_to_segments else (
            create_mapping_things_to_segments_from_mapping_arr(gt_mapping_instance_to_semantic))

        pq_val = my_panoptic_quality(
            solution_panoptic_volume,
            gt_panoptic_volume,
            mapping_things_to_segments,
            gt_mapping_things_to_segments,
        )
        print("my panoptic quality:", pq_val, "took", time.time() - start)

        # purely compare the instance segmentations:
        rand, arand = compare_label_image_to_gt(instance_solution_volume, gt_instance_volume)
        print("Instance quality: rand", rand, "arand", arand)

        # things = set(mapping_part_names_to_semantic_labels.values())
        # # things.remove(0)  # background should be treated as stuff
        # stuffs = set()
        # print("computing panoptic quality...")
        # pq_val = compute_panoptic_quality(torch.from_numpy(solution_panoptic_volume[None, :]),  # add batch dim
        #                                   torch.from_numpy(gt_panoptic_volume[None, :]),  # add batch dim
        #                                   pq_config={"stuffs": stuffs, "things": things})
        #
        # print("Panoptic quality:", pq_val)

        return pq_val, arand















