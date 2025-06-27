import pickle
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from deconstruct.proposal_generation.segmentation_catalog import SegmentationCatalog
from deconstruct.data_generation.part_catalog import MeshFromVolume
from deconstruct.proposal_generation.pointcloud_matching import PointCloudMatcher, BagOfVisualWords
from deconstruct.proposal_generation.pointcloud_registration import (CatalogSegmentationRegistration,
                                                                    PartProposal, RegistrationResult)
from deconstruct.utils.geometries.pcd_utils import o3d_fpfh
import argparse
import yaml


class ProposalGenerator:
    """
    load part catalog

    use data path etc. from there and path to segmentation result to create segmentation catalog
    create point cloud for each segmentation with properties global fpfh, average curvature, volume of mask, etc.

    need a from segment to canonical pcd + properties for catalog
    and a from segment to pcd in scene + properties for segmentation

    use the voxelized catalog and the matching and smoothing from the segmentation catalog to create point clouds
    and the same properties for each part proposal
    """

    def __init__(self, path_part_catalog, path_h5_segmentation, feature_config, matching_kwargs, registration_config,
                 marching_smoother_config=None, verbose=True, use_cheat_matching=False,
                 segmentation_catalog_kwargs=None, path_h5_dataset=None):
        self.verbose = verbose
        self.matching_kwargs = matching_kwargs

        # load part catalog:
        self.use_cheat_matching = use_cheat_matching
        self.path_part_catalog = path_part_catalog
        self.path_h5_segmentation = path_h5_segmentation
        with open(path_part_catalog, "rb") as f:
            self.part_catalog = pickle.load(f)
        num_parts = len(self.part_catalog.catalog_part_meshes)
        num_placed_parts = len([t for p in self.part_catalog.catalog_part_meshes.values() for t in
                                p.list_trafo_from_canonical_to_scene])
        print(f"Successfully loaded part catalog from {path_part_catalog} with "
              f"{num_parts} catalog parts and {num_placed_parts} placed parts.")
        # printing the part sizes relevant for the seeded segmentation during insseg:
        list_part_sizes = [p.voxelized_part.binary_volume.sum() for p in self.part_catalog.catalog_part_meshes.values()]
        print("part size statistics: mean, std, min, max", np.mean(list_part_sizes), np.std(list_part_sizes),
              np.min(list_part_sizes), np.max(list_part_sizes))

        # create segmentation catalog:
        print("Creating segmentation catalog...")
        segmentation_catalog_kwargs = {} if segmentation_catalog_kwargs is None else segmentation_catalog_kwargs
        if path_h5_dataset is not None:
            self.part_catalog.path_h5_dataset = path_h5_dataset
        self.segmentation_catalog = SegmentationCatalog(
            path_h5_segmentation=path_h5_segmentation,
            path_h5_dataset=self.part_catalog.path_h5_dataset,
            voxelization_scale=self.part_catalog.voxelization_scale,
            marching_smoother_config=marching_smoother_config,
            **segmentation_catalog_kwargs
        )

        # create smooth meshes from the voxelized catalog:
        print("Use voxelized catalog to create smooth meshes for each part...")
        for part_mesh in (tqdm(self.part_catalog.catalog_part_meshes.values()) if self.verbose
                                else self.part_catalog.catalog_part_meshes.values()):

            # get corresponding voxelized part:
            voxelized_part = part_mesh.voxelized_part

            smooth_trmesh, smooth_normals, _ = (
                self.segmentation_catalog.marching_smoother.run_marching_cubes(voxelized_part.binary_volume))

            # bring smooth mesh into canonical orientation (also do rescaling here by voxelization scale):
            smooth_trmesh.apply_transform(voxelized_part.trafo_to_canonical)

            # apply only the rotation to the normals:
            trafo = voxelized_part.trafo_to_canonical
            rotation = trafo[:3, :3] * np.linalg.det(trafo[:3, :3]) ** (-1 / 3)
            smooth_normals = smooth_normals @ rotation.T

            part_mesh.mesh_from_volume = MeshFromVolume(
                part_name=part_mesh.part_name,
                binary_volume=voxelized_part.binary_volume,
                smooth_trmesh=smooth_trmesh,
                smooth_normals=smooth_normals,
            )

        # create point cloud registration:
        self.cata2seg_registration = CatalogSegmentationRegistration(self.part_catalog, self.segmentation_catalog,
                                                                     **registration_config)

        # compute additional local and global features for matching:
        self.compute_features_for_matching(**feature_config)

    def perform_point_cloud_matching(self):
        property_key = self.matching_kwargs["matcher_config"]["property_key"]
        use_subsampled = self.matching_kwargs["matcher_config"]["use_subsampled"]

        if property_key == "bovw":
            self.matching_kwargs.setdefault("compute_bovw_kwargs", {})["use_subsampled"] = use_subsampled
            self.compute_bag_of_visual_words(**self.matching_kwargs["compute_bovw_kwargs"])
        elif property_key == "feature_centers":
            self.matching_kwargs.setdefault("compute_feature_centers_kwargs", {})["use_subsampled"] = use_subsampled
            self.compute_feature_centers(**self.matching_kwargs["compute_feature_centers_kwargs"])

        # create matcher:
        point_cloud_matcher = PointCloudMatcher(**self.matching_kwargs["matcher_config"])
        matching_dict = point_cloud_matcher(self.part_catalog, self.segmentation_catalog)
        return matching_dict

    @staticmethod
    def get_features_for_bovw(pcd_with_properties, use_subsampled=True, feature_key="fpfh_float32"):
        if use_subsampled:
            local_properties_attr = "subsampled_local_properties"
        else:
            local_properties_attr = "local_properties"
        features = getattr(pcd_with_properties, local_properties_attr)[feature_key]
        return features

    def compute_feature_centers(self, num_centers=50, feature_key="fpfh_float32", use_subsampled=True,
                                append_mass=True):
        print("Computing feature centers for matching...")
        all_meshes_from_volume = ([pm.mesh_from_volume for pm in self.part_catalog.catalog_part_meshes.values()] +
                                  list(self.segmentation_catalog.catalog_segmentation_meshes.values()))

        for mesh_from_volume in tqdm(all_meshes_from_volume, disable=not self.verbose):
            pcd_with_properties = mesh_from_volume.pcd_with_properties
            if use_subsampled:
                local_properties_attr = "subsampled_local_properties"
                global_properties_attr = "subsampled_global_properties"
            else:
                local_properties_attr = "local_properties"
                global_properties_attr = "global_properties"
            features = getattr(pcd_with_properties, local_properties_attr)[feature_key]

            # get feature centers by k-means:
            kmeans = KMeans(n_clusters=num_centers, n_init="auto").fit(features)
            feature_centers = kmeans.cluster_centers_

            if append_mass:
                # append mass to feature centers:
                one_hot_labels = np.eye(num_centers)[kmeans.labels_]
                mass = one_hot_labels.sum(axis=0)
                feature_centers = np.concatenate([feature_centers, mass[:, None]], axis=-1, dtype=np.float32)

            getattr(pcd_with_properties, global_properties_attr)["feature_centers"] = feature_centers

        print("feature_centers.shape", feature_centers.shape)
        print("Done with computing feature centers.")

    def compute_bag_of_visual_words(self, num_features_per_object=2000, num_centers=256, use_subsampled=True,
                                    use_all_to_fit=True):
        self.bag_of_visual_words = BagOfVisualWords(num_centers=num_centers)

        if use_subsampled:
            global_properties_attr = "subsampled_global_properties"
        else:
            global_properties_attr = "global_properties"

        if use_all_to_fit:
            list_reference_meshes = ([pm.mesh_from_volume for pm in self.part_catalog.catalog_part_meshes.values()] +
                                      list(self.segmentation_catalog.catalog_segmentation_meshes.values()))
        else:
            list_reference_meshes = [pm.mesh_from_volume for pm in self.part_catalog.catalog_part_meshes.values()]
        len_part_catalog = len(self.part_catalog.catalog_part_meshes)

        # fit bag of visual words:
        print("Fitting bag of visual words...")
        reference_features = []
        for mesh_from_volume in tqdm(list_reference_meshes, disable=not self.verbose):
            pcd_with_properties = mesh_from_volume.pcd_with_properties
            features = self.get_features_for_bovw(pcd_with_properties, use_subsampled=use_subsampled)
            random_indices = np.random.randint(0, len(features), num_features_per_object)
            reference_features.append(features[random_indices])
        reference_features = np.stack(reference_features, axis=0)
        # print("reference_features.shape", reference_features.shape)
        reference_hist = self.bag_of_visual_words.fit(reference_features)
        # print("reference_hist.shape", reference_hist.shape)

        for i, mesh_from_volume in tqdm(enumerate(list_reference_meshes[:len_part_catalog]), disable=not self.verbose):
            pcd_with_properties = mesh_from_volume.pcd_with_properties
            getattr(pcd_with_properties, global_properties_attr)["bovw"] = reference_hist[i]

        for i, mesh_from_volume in tqdm(enumerate(list_reference_meshes[len_part_catalog:]), disable=not self.verbose):
            pcd_with_properties = mesh_from_volume.pcd_with_properties
            if use_all_to_fit:
                getattr(pcd_with_properties, global_properties_attr)["bovw"] = reference_hist[i + len_part_catalog]
            else:
                features = self.get_features_for_bovw(pcd_with_properties, use_subsampled=use_subsampled)
                random_indices = np.random.randint(0, len(features), num_features_per_object)
                getattr(pcd_with_properties, global_properties_attr)["bovw"] = (
                    self.bag_of_visual_words.predict(features[random_indices][None, :]).ravel())

    def compute_features_for_matching(self, fpfh_radius, fpfh_max_nn, subsampled_only=True):
        """so far only fpfh, one could also use learned features like FCGF"""

        # compute fpfh features for all pcds:
        print("Computing fpfh features for all pcds...")
        all_meshes_from_volume = ([pm.mesh_from_volume for pm in self.part_catalog.catalog_part_meshes.values()] +
                                  list(self.segmentation_catalog.catalog_segmentation_meshes.values()))
        for mesh_from_volume in tqdm(all_meshes_from_volume, disable=not self.verbose):
            pcd_with_properties = mesh_from_volume.pcd_with_properties
            if not subsampled_only:
                fpfh = o3d_fpfh(pcd_with_properties.pcd, radius=fpfh_radius, max_nn=fpfh_max_nn,
                                        return_float32_arr=True)
                pcd_with_properties.local_properties["fpfh"] = fpfh[0]
                pcd_with_properties.local_properties["fpfh_float32"] = fpfh[1]
                pcd_with_properties.global_properties["average_fpfh"] = fpfh[1].mean(axis=0)  # (33, )

            # and the subsampled pcds:
            if pcd_with_properties.subsampled_pcd is not None:
                subsampled_fpfh = o3d_fpfh(pcd_with_properties.subsampled_pcd, radius=fpfh_radius, max_nn=fpfh_max_nn,
                                           return_float32_arr=True)
                pcd_with_properties.subsampled_local_properties["fpfh"] = subsampled_fpfh[0]
                pcd_with_properties.subsampled_local_properties["fpfh_float32"] = subsampled_fpfh[1]
                pcd_with_properties.subsampled_global_properties["average_fpfh"] = subsampled_fpfh[1].mean(axis=0)

    def get_perfect_proposals_for_debugging(self):
        dict_part_proposals = {}
        count = 0
        for part_mesh in self.part_catalog.catalog_part_meshes.values():
            for trafo in part_mesh.list_trafo_from_canonical_to_scene:
                dict_part_proposals[(count, part_mesh.part_name)] = PartProposal(
                    label=count,
                    part_name=part_mesh.part_name,
                    registration_result=RegistrationResult(transformation=trafo, registration_score=np.inf),
                )
                count += 1
        return dict_part_proposals

    def assess_proposal_quality(self, dict_part_proposals, iou_threshold=0.4):
        print("assessing proposal quality...")
        registration_scores_for_iou_matches = []

        # first check how many segments have a sufficient iou match:
        count_sufficient_iou_matches = 0
        for mesh_from_volume in self.segmentation_catalog.catalog_segmentation_meshes.values():
            iou_gt_match = mesh_from_volume.iou_gt_match
            if iou_gt_match[1] >= iou_threshold:
                count_sufficient_iou_matches += 1

        print(f"fraction of segments with iou matching over {iou_threshold}: "
              f"{count_sufficient_iou_matches / len(self.segmentation_catalog.catalog_segmentation_meshes)}"
              f" ({count_sufficient_iou_matches} out "
              f"of {len(self.segmentation_catalog.catalog_segmentation_meshes)})")

        # now check how many proposals are the correct iou match:
        for part_proposal in dict_part_proposals.values():
            # get best class based on iou matching between segments and GT:
            iou_gt_match = self.segmentation_catalog.catalog_segmentation_meshes[part_proposal.label].iou_gt_match
            if part_proposal.part_name == iou_gt_match[0] and iou_gt_match[1] >= iou_threshold:
                registration_scores_for_iou_matches.append(part_proposal.registration_result.registration_score)

        report_proposal_quality(dict_part_proposals, registration_scores_for_iou_matches)

        return registration_scores_for_iou_matches

    def cheat_matching(self, matching_dict):
        """adds to the matching dict the iou match for each segment"""
        print("-" * 80)
        print("WARNING: using cheat matching!")
        print("-" * 80)
        for label, mesh in self.segmentation_catalog.catalog_segmentation_meshes.items():
            # append to the front:
            if mesh.iou_gt_match[0] != "none":
                matching_dict[label].insert(0, (mesh.iou_gt_match[0], -1))  # (part_name, dist = -1)
        return matching_dict

    def __call__(self):
        """
        perform pairwise registration
        """
        matching_dict = self.perform_point_cloud_matching()
        if self.use_cheat_matching:
            matching_dict = self.cheat_matching(matching_dict)
        dict_part_proposals, dict_perfect_registrations = self.cata2seg_registration(matching_dict)

        return dict_part_proposals, dict_perfect_registrations, matching_dict


def report_proposal_quality(dict_part_proposals, registration_scores_for_iou_matches):
    unique_labels = set([p.label for p in dict_part_proposals.values()])
    if len(unique_labels) == 0:
        print("No proposals found.")
        return

    fraction_tried_to_register_iou_match = len(registration_scores_for_iou_matches) / len(unique_labels)
    print("-" * 80)
    print("fraction_tried_to_register_iou_match", fraction_tried_to_register_iou_match,
          f", {len(registration_scores_for_iou_matches)} out of {len(unique_labels)}")
    if len(registration_scores_for_iou_matches) == 0:
        print("No proposals which are the correct iou match were tried to register.")
    else:
        print("registration_scores_for_iou_matches: mean, std, min, max",
              np.mean(registration_scores_for_iou_matches), np.std(registration_scores_for_iou_matches),
              np.min(registration_scores_for_iou_matches), np.max(registration_scores_for_iou_matches))
    print("-" * 80)


parser = argparse.ArgumentParser(description='Create part proposals by registering catalog to segmentation.')
parser.add_argument('-c', '--config', help='YAML Config-file', required=False, default=None)
if __name__ == '__main__':
    args = vars(parser.parse_args())
    with open(args["config"], "r") as f:
        config = yaml.load(f, yaml.SafeLoader)
    proposal_generator = ProposalGenerator(**config)
    proposal_generator()


