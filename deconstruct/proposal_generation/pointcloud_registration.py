import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from deconstruct.utils.registration.pcd_registration_utils import (PCDRegistration, RegistrationResult, PCARegistration)
from deconstruct.utils.geometries.pcd_utils import PCDSubsampler, voxel_presampling, random_presampling


@dataclass
class PartProposal:
    """Class for part proposal as result of registering catalog parts to segments."""
    label: int
    part_name: str
    registration_result: RegistrationResult
    affinity_score: float = None
    mwis_weight: float = None
    abs_fg_overlap: float = None
    abs_volume: int = None


def print_down_sampling_statistics(all_meshes_from_volume):
    # move this to a separate function:
    pcd_size_original = []
    pcd_size_subsampled = []
    for mesh_from_volume in all_meshes_from_volume:
        pcd_size_original.append(np.asarray(mesh_from_volume.pcd_with_properties.pcd.points).shape[0])
        pcd_size_subsampled.append(
            np.asarray(mesh_from_volume.pcd_with_properties.subsampled_pcd.points).shape[0]
        )
    print("pcd size original (mean, std, min, max):", np.mean(pcd_size_original), np.std(pcd_size_original),
          np.min(pcd_size_original), np.max(pcd_size_original))
    print("pcd size subsampled (mean, std, min, max):", np.mean(pcd_size_subsampled), np.std(pcd_size_subsampled),
          np.min(pcd_size_subsampled), np.max(pcd_size_subsampled))


class CatalogSegmentationRegistration:
    """
    pcd registration method
    pairs in order
    do the nested loops. threshold for selection of catalog parts. threshold
    """

    def __init__(self, part_catalog, segmentation_catalog, list_registration_configs,
                 pcd_subsampler_config=None, presample_config=None, max_num_parts_per_segment=10,
                 empirical_factor_perfect_registration=0.3, verbose=True, disgard_pca_if_not_perferct=True):
        """
        the success threshold must be rather tough actually. Otherwise we get too many false positives.
        """
        self.verbose = verbose
        self.empirical_factor_perfect_registration = empirical_factor_perfect_registration
        self.max_num_parts_per_segment = max_num_parts_per_segment
        self.list_registration_configs = list_registration_configs
        self.part_catalog = part_catalog
        self.segmentation_catalog = segmentation_catalog
        self.disgard_pca_if_not_perferct = disgard_pca_if_not_perferct

        self.list_registration_methods = []
        for registration_config in list_registration_configs:
            registration_method = PCDRegistration(**registration_config)
            self.list_registration_methods.append(registration_method)

        if pcd_subsampler_config is None:
            self.pcd_subsampler = None
        else:
            self.pcd_subsampler = PCDSubsampler(**pcd_subsampler_config)

            # first subsample the catalog and segmentation pcds:
            all_meshes_from_volume = (
                        [pm.mesh_from_volume for pm in self.part_catalog.catalog_part_meshes.values()] +
                        list(self.segmentation_catalog.catalog_segmentation_meshes.values()))

            print("Subsampling all pcds... using presampling:", presample_config is not None)
            presample_num_points = []
            presample_ratios = []
            for mesh_from_volume in tqdm(all_meshes_from_volume, disable=not self.verbose):
                pcd_with_properties = mesh_from_volume.pcd_with_properties
                if presample_config is None:
                    subsampled_pcd, subsampled_local_properties = self.pcd_subsampler(
                        pcd=pcd_with_properties.pcd,
                        local_properties=pcd_with_properties.local_properties
                    )
                else:
                    subsampled_pcd, subsampled_local_properties = random_presampling(
                        pcd=pcd_with_properties.pcd,
                        local_properties=pcd_with_properties.local_properties,
                        **presample_config
                    )
                    presample_num_points.append(len(subsampled_pcd.points))
                    presample_ratios.append(len(subsampled_pcd.points) / len(pcd_with_properties.pcd.points))
                    subsampled_pcd, subsampled_local_properties = self.pcd_subsampler(
                        pcd=subsampled_pcd,
                        local_properties=subsampled_local_properties
                    )

                pcd_with_properties.subsampled_pcd = subsampled_pcd
                pcd_with_properties.subsampled_local_properties = subsampled_local_properties
                pcd_with_properties.subsampled_global_properties = {}

            # get some statistics about the sizes:
            print("presample_num_points (mean, std, min, max):", np.mean(presample_num_points),
                  np.std(presample_num_points), np.min(presample_num_points), np.max(presample_num_points))
            print_down_sampling_statistics(all_meshes_from_volume)

        # estimate average point distance for all pcds:
        self.estimate_average_point_distance_for_all_pcds()

    def estimate_average_point_distance_for_all_pcds(self):
        # combine pcds from catalog and segmentation and use random subset:
        all_pcds_with_properties = []
        print("Estimating average point distance for every point cloud...")
        for i, part_mesh in enumerate(self.part_catalog.catalog_part_meshes.values()):
            all_pcds_with_properties.append(part_mesh.mesh_from_volume.pcd_with_properties)
        for i, segment_mesh in enumerate(self.segmentation_catalog.catalog_segmentation_meshes.values()):
            all_pcds_with_properties.append(segment_mesh.pcd_with_properties)

        for pcd_with_properties in tqdm(all_pcds_with_properties, disable=not self.verbose):
            pcd_with_properties.global_properties["average_point_distance"] = (
                np.mean(pcd_with_properties.pcd.compute_nearest_neighbor_distance()))

            # and the subsampled pcds:
            if pcd_with_properties.subsampled_pcd is not None:
                pcd_with_properties.subsampled_global_properties["average_point_distance"] = (
                    np.mean(pcd_with_properties.subsampled_pcd.compute_nearest_neighbor_distance()))

    def check_if_registration_was_perfect(self, src, dst, registration_result):
        # get the average point distance for both point clouds:
        avg_dist_src = src.global_properties["average_point_distance"]
        avg_dist_dst = dst.global_properties["average_point_distance"]

        # the optimal expected distance is 1/4 of the average distance:
        perfect_registration = True
        if registration_result.dist_dst_to_src * self.empirical_factor_perfect_registration > avg_dist_src / 4:
            perfect_registration = False
        if registration_result.dist_src_to_dst * self.empirical_factor_perfect_registration > avg_dist_dst / 4:
            perfect_registration = False
        return perfect_registration

    def __call__(self, matching_dict):
        dict_part_proposals = {}
        dict_perfect_registrations = {}

        print("Perform registration for each segment...")
        for label in tqdm(matching_dict, disable=not self.verbose):
            # print("\n\nStarting with segment:", label)
            dst = self.segmentation_catalog.catalog_segmentation_meshes[label].pcd_with_properties
            for registration_method in self.list_registration_methods:
                # print("Starting with method:", registration_method)
                for part_name, _ in matching_dict[label][:self.max_num_parts_per_segment]:

                    src = self.part_catalog.catalog_part_meshes[part_name].mesh_from_volume.pcd_with_properties
                    # print("label:", label, "part_name:", part_name)
                    # print("src:", src, "dst:", dst)
                    registration_result = registration_method(src, dst)

                    part_proposal = PartProposal(
                        label=label,
                        part_name=part_name,
                        registration_result=registration_result,
                    )

                    # check if there already is a part proposal for this pair:
                    if (label, part_name) in dict_part_proposals:
                        # check if the new proposal is better than the old one:
                        old_result = dict_part_proposals[(label, part_name)].registration_result
                        if registration_result.registration_score > old_result.registration_score:
                            dict_part_proposals[(label, part_name)] = part_proposal
                    else:
                        dict_part_proposals[(label, part_name)] = part_proposal

                    if self.check_if_registration_was_perfect(src, dst, registration_result):
                        dict_perfect_registrations[label] = part_proposal
                        # print("Found perfect registration for segment:", label)
                        break
                    else:
                        if (isinstance(registration_method.registration_method, PCARegistration) and
                                self.disgard_pca_if_not_perferct):
                            # print("Disgard PCA registration for segment:", label, "and part:", part_name)
                            dict_part_proposals.pop((label, part_name))

                # check if we have a perfect registration for this segment:
                if label in dict_perfect_registrations:
                    # move on to next segment
                    break

        return dict_part_proposals, dict_perfect_registrations



