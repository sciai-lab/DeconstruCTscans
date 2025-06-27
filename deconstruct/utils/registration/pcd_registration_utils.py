import copy
from dataclasses import dataclass

import numpy as np
import open3d as o3d
import time
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

from deconstruct.utils.geometries.neighbor_search import pcd_get_neighbors
from deconstruct.utils.registration.feature_correspondences import FeatureCorrespondenceFinder


@dataclass
class PartProposal:
    """Class for part proposal as result of registering catalog parts to segments."""
    instance_label: int
    catalog_part_name: str
    found_trafo_cata2seg: np.ndarray
    registration_score: float
    affinity_score: float = None
    mwis_weight: float = None
    abs_fg_overlap: float = None
    abs_volume: int = None


@dataclass
class RegistrationResult:
    """Data class to store result from teaser registration. Important attribute "transformation" = found trafo."""
    transformation: np.ndarray
    runtime: float = None
    registration_score: float = None
    dist_src_to_dst: float = None
    dist_dst_to_src: float = None
    o3d_registration_result: o3d.pipelines.registration.RegistrationResult = None


def get_pcd_for_registration(pcd_with_properties, create_copy=True, use_subsampled=False):
    if use_subsampled:
        pcd = pcd_with_properties.subsampled_pcd
        local_properties = pcd_with_properties.subsampled_local_properties
    else:
        pcd = pcd_with_properties.pcd
        local_properties = pcd_with_properties.local_properties
    if create_copy:
        pcd = copy.deepcopy(pcd)
    return pcd, local_properties


# def estimate_average_point_distance(pcd, return_std=False):
#     """
#     There is a function in open3d for this and it is a bit faster. pcd.compute_nearest_neighbor_distance()
#     Estimate average distance and standard deviation of a point in pcd to its nearest neighbor.
#
#     Parameters
#     ----------
#     pcd : o3d.geometry.PointCloud
#
#     Returns
#     -------
#     float, float
#         mean,
#         standard deviation
#     """
#     neighbors_dict, _ = pcd_get_neighbors(pcd, knn=2)
#
#     list_neighbor_dist = []
#     for d in neighbors_dict.values():
#         list_neighbor_dist.append(d["neighbor_dist"][1])
#
#     if return_std:
#         return np.mean(list_neighbor_dist), np.std(list_neighbor_dist)
#     else:
#         return np.mean(list_neighbor_dist)


def calc_registration_score(src, dst, trafo=None, eps=1e-7):
    """
    Determines registration score (inverse of average dist from one pcd to the other and vice versa)

    Parameters
    ----------
    src : o3d.geometry.PointCloud
        Rotate and shift this one onto dst.
    dst : o3d.geometry.PointCloud
        Target of trafo.
    trafo : np.ndarray
        of shape (4,4,). Affine trafo.
    eps : float, optional
        To avoid dividing by zero.

    Returns
    -------
    flaot
        registration score
    """
    if trafo is None:
        transformed_src = src
    else:
        transformed_src = copy.deepcopy(src)
        transformed_src.transform(trafo)
    dst_to_src_dist = np.mean(np.asarray(dst.compute_point_cloud_distance(target=transformed_src)))
    src_to_dst_dist = np.mean(np.asarray(transformed_src.compute_point_cloud_distance(target=dst)))
    return 1 / (dst_to_src_dist + src_to_dst_dist + eps), dst_to_src_dist, src_to_dst_dist


class PCARegistration:

    def __init__(self):
        pass

    @staticmethod
    def get_pca_rotations(pcd):
        """
        For a o3d.geometry.PointCloud get the first 3 pca components using PCA from sklearn.decomposition.
        Assumes that pcd is already centered.
        """

        pca = PCA(n_components=3)
        points = np.asarray(pcd.points)
        pca.fit(points)
        pca_vectors = pca.components_

        # create the 4 possible rotation matrices to align the pca vectors with the canonical axes:
        list_rotation_matrices = []
        for a in [-1, 1]:
            for b in [-1, 1]:
                rot_matr = pca_vectors.copy()
                rot_matr[0] *= a
                rot_matr[1] *= b
                if not np.isclose(np.linalg.det(rot_matr), 1):
                    rot_matr[2] *= -1

                list_rotation_matrices.append(rot_matr)

        return list_rotation_matrices, pca.explained_variance_ratio_

    def __call__(self, src, dst, use_subsampled=False, **kwargs):
        """
        Find Trafo (rot + shift) to best align to point clouds based on their PCAs and center of mass.
        Note: this method struggles with symmetries.

        Parameters
        ----------
        src : o3d.geometry.PointCloud
            Rotate and shift this one onto dst.
        dst : o3d.geometry.PointCloud
            Target of trafo.
        Returns
        -------
        np.ndarray
            of shape (4,4). Affine trafo.
        """
        src_pcd, _ = get_pcd_for_registration(src, use_subsampled=use_subsampled)
        dst_pcd, _ = get_pcd_for_registration(dst, use_subsampled=use_subsampled, create_copy=False)

        # center src:
        shift_src_to_origin = np.eye(4)
        shift_src_to_origin[:3, 3] = -np.mean(np.asarray(src_pcd.points), axis=0)

        # shift src to COM of dst
        shift_origin_to_dst = np.eye(4)
        shift_origin_to_dst[:3, 3] = np.mean(np.asarray(dst_pcd.points), axis=0)

        # rotate src to best align pca's with dst:
        list_rot1, var_ratio1 = self.get_pca_rotations(dst_pcd)
        list_rot2, var_ratio2 = self.get_pca_rotations(src_pcd)

        list_registration_results = []
        trafo1 = list_rot1[0]  # can just fix one of the 4 possible rotations
        for trafo2 in list_rot2:
            rotation_at_origin = np.eye(4)
            rotation_at_origin[:3, :3] = trafo1.T @ trafo2  # align pcas

            # apply it and compute score:
            trafo = shift_origin_to_dst @ rotation_at_origin @ shift_src_to_origin
            score, dst_to_src_dist, src_to_dst_dist = calc_registration_score(src_pcd, dst_pcd, trafo)

            list_registration_results.append(RegistrationResult(
                transformation=trafo,
                registration_score=score,
                dist_src_to_dst=src_to_dst_dist,
                dist_dst_to_src=dst_to_src_dist)
            )

        # get the best trafo:
        best_result = max(list_registration_results, key=lambda x: x.registration_score)

        if use_subsampled:
            # recompute score with full pcds:
            best_trafo = best_result.transformation
            score, dst_to_src_dist, src_to_dst_dist = calc_registration_score(src.pcd, dst.pcd, best_trafo)
            best_result = RegistrationResult(
                transformation=best_trafo,
                registration_score=score,
                dist_src_to_dst=src_to_dst_dist,
                dist_dst_to_src=dst_to_src_dist
            )

        return best_result


class ICPRegistration:
    default_params = {}

    def __init__(self, max_correspondence_distance_factor=2, params=None):
        self.max_correspondence_distance_factor = max_correspondence_distance_factor
        self.params = self.default_params.copy()
        if params is not None:
            self.params.update(params)

    def __call__(self, src, dst, expected_point_distance, init_trafo=None, use_subsampled=False):
        """using generalized ICP because it is more robust and symmetric in src and dst"""
        src_pcd, _ = get_pcd_for_registration(src, use_subsampled=use_subsampled)
        dst_pcd, _ = get_pcd_for_registration(dst, use_subsampled=use_subsampled)
        init_trafo = np.eye(4) if init_trafo is None else init_trafo

        result = o3d.pipelines.registration.registration_generalized_icp(
            source=src_pcd,
            target=dst_pcd,
            init=init_trafo,
            max_correspondence_distance=self.max_correspondence_distance_factor * expected_point_distance,
            **self.params
        )

        return result


class TensorICPRegistration:
    default_params = {}
    # see here for how to implement multistage ICP:
    # http://www.open3d.org/docs/release/tutorial/t_pipelines/t_icp_registration.html#Multi-Scale-ICP-on-CUDA-device-Example

    def __init__(self, max_correspondence_distance_factor=2, params=None, use_cuda=True):
        self.max_correspondence_distance_factor = max_correspondence_distance_factor
        self.params = self.default_params.copy()
        if params is not None:
            self.params.update(params)
        if use_cuda:
            assert torch.cuda.is_available(), "cuda not available"
            self.device = o3d.core.Device("CUDA:1")
        else:
            self.device = o3d.core.Device("CPU:0")

    def __call__(self, src, dst, expected_point_distance, init_trafo=None, use_subsampled=False):
        """using generalized ICP because it is more robust and symmetric in src and dst"""
        src_pcd, _ = get_pcd_for_registration(src, use_subsampled=use_subsampled)
        dst_pcd, _ = get_pcd_for_registration(dst, use_subsampled=use_subsampled)
        init_trafo = np.eye(4) if init_trafo is None else init_trafo

        tensor_src_pcd = o3d.t.geometry.PointCloud.from_legacy(src_pcd, device=self.device)
        tensor_dst_pcd = o3d.t.geometry.PointCloud.from_legacy(dst_pcd, device=self.device)
        tensor_init_trafo = o3d.core.Tensor(init_trafo)

        result = o3d.t.pipelines.registration.icp(
            source=tensor_src_pcd,
            target=tensor_dst_pcd,
            init_source_to_target=tensor_init_trafo,
            max_correspondence_distance=self.max_correspondence_distance_factor * expected_point_distance,
            **self.params
        )

        return result


class RANSACRegistration:

    ransac_default_params = {}
    ransac_default_params_feature_matching = {}

    def __init__(self, params=None, params_corres=None):
        self.params = self.ransac_default_params.copy()
        if params is not None:
            self.params.update(params)
        self.params_features = self.ransac_default_params_feature_matching.copy()
        if params_corres is not None:
            self.params_features.update(params_corres)

    def __call__(self, src, dst, expected_point_distance, correspondences=None, use_subsampled=False):
        src_pcd, src_local_properties = get_pcd_for_registration(src, use_subsampled=use_subsampled)
        dst_pcd, dst_local_properties = get_pcd_for_registration(dst, use_subsampled=use_subsampled)
        if correspondences is None:
            result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source=src_pcd,
                target=dst_pcd,
                source_feature=src_local_properties["fpfh"],
                target_feature=dst_local_properties["fpfh"],
                max_correspondence_distance=1.5 * expected_point_distance,
                checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(2 * expected_point_distance)],
                **self.params, **self.params_features
            )
        else:
            result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
                source=src_pcd,
                target=dst_pcd,
                corres=o3d.utility.Vector2iVector(correspondences),
                max_correspondence_distance=1.5 * expected_point_distance,
                checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(2 * expected_point_distance)],
                **self.params
            )
        return result


class FGRegistration:

    fgr_default_params = {"decrease_mu": True, "tuple_test": False}
    # set tuple test to false here, as I will perform it seperately
    # see here which defaults are set:
    # https://github.com/isl-org/Open3D/blob/master/cpp/open3d/pipelines/registration/FastGlobalRegistration.h#L50

    def __init__(self, params=None):
        self.params = self.fgr_default_params.copy()
        if params is not None:
            self.params.update(params)

    def __call__(self, src, dst, expected_point_distance, correspondences=None, use_subsampled=False):
        src_pcd, src_local_properties = get_pcd_for_registration(src, use_subsampled=use_subsampled)
        dst_pcd, dst_local_properties = get_pcd_for_registration(dst, use_subsampled=use_subsampled)
        if correspondences is None:
            result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                source=src_pcd,
                target=dst_pcd,
                source_feature=src_local_properties["fpfh"],
                target_feature=dst_local_properties["fpfh"],
                option=o3d.pipelines.registration.FastGlobalRegistrationOption(
                    maximum_correspondence_distance=0.5 * expected_point_distance,
                    **self.params,
                )
            )
        else:
            result = o3d.pipelines.registration.registration_fgr_based_on_correspondence(
                source=src_pcd,
                target=dst_pcd,
                corres=o3d.utility.Vector2iVector(correspondences),
                option=o3d.pipelines.registration.FastGlobalRegistrationOption(
                    maximum_correspondence_distance=0.5 * expected_point_distance,
                    **self.params
                )
            )
        return result


class PCDRegistration:
    """
    core method
    ICP refinement
    same result format

    """

    def __init__(self, method="pca", method_params=None, expected_point_distance=None, icp_refinement=True,
                 icp_refinement_params=None, use_subsampled=False,
                 correspondence_finder_config=None, use_my_correspondences=None, min_num_correspondences=10):
        method_params = {} if method_params is None else method_params
        self.expected_point_distance = expected_point_distance
        self.method = method
        self.use_subsampled = use_subsampled
        if method == "pca":
            self.registration_method = PCARegistration(**method_params)
        elif method == "ransac":
            self.registration_method = RANSACRegistration(**method_params)
        elif method == "fgr":
            self.registration_method = FGRegistration(**method_params)
        else:
            raise NotImplementedError(f"{method=} not implemented.")

        # init correspondence finder:
        if use_my_correspondences is None:
            use_my_correspondences = (self.method in ["ransac", "fgr"])  # otherwise no need for correspondences
        if use_my_correspondences:
            correspondence_finder_config = {} if correspondence_finder_config is None else correspondence_finder_config
            self.correspondence_finder = FeatureCorrespondenceFinder(**correspondence_finder_config)
            self.min_num_correspondences = min_num_correspondences
        else:
            self.correspondence_finder = None
            self.min_num_correspondences = None

        self.icp_refinement = None
        if icp_refinement:
            icp_refinement_params = {} if icp_refinement_params is None else icp_refinement_params
            self.icp_refinement = ICPRegistration(**icp_refinement_params)

    def __str__(self):
        return f"PCDRegistration using method {self.method}"

    def find_correspondences(self, src, dst):
        """potentially make this more general to use other features as well"""
        if self.use_subsampled:
            src_features = src.subsampled_local_properties["fpfh_float32"]
            dst_features = dst.subsampled_local_properties["fpfh_float32"]
        else:
            src_features = src.local_properties["fpfh_float32"]
            dst_features = dst.local_properties["fpfh_float32"]
        # print("Finding correspondences...")
        correspondences = self.correspondence_finder(src_features, dst_features)

        return correspondences

    def get_expected_point_distance(self, src, dst):
        if self.use_subsampled:
            expected_point_distance = max(src.subsampled_global_properties["average_point_distance"],
                                            dst.subsampled_global_properties["average_point_distance"])
        else:
            expected_point_distance = max(src.global_properties["average_point_distance"],
                                            dst.global_properties["average_point_distance"])
        return expected_point_distance

    def __call__(self, src, dst, correspondences=None, expected_point_distance=None):

        if correspondences is None and self.correspondence_finder is not None:
            start = time.time()
            correspondences = self.find_correspondences(src, dst)
            # print(f"Finding correspondences took: {time.time() - start:.4f} seconds.")
            if correspondences.shape[0] < self.min_num_correspondences:
                # print("too few correspondences, returning identity trafo.")
                return RegistrationResult(transformation=np.eye(4), registration_score=-1,
                                          dist_src_to_dst=np.inf, dist_dst_to_src=np.inf)

        # get expected point distance if not already specified:
        expected_point_distance = self.expected_point_distance if expected_point_distance is None \
            else expected_point_distance
        if expected_point_distance is None:
            expected_point_distance = self.get_expected_point_distance(src, dst)

        start = time.time()
        result = self.registration_method(src, dst, correspondences=correspondences, use_subsampled=self.use_subsampled,
                                          expected_point_distance=expected_point_distance)
        # print(f"Registration with method {self} took: {time.time() - start:.4f} seconds.")

        if np.isnan(result.transformation).any():
            print("WARNING: found nan in trafo, returning identity trafo.")
            print(f"{src=}", f"{dst=}", correspondences.shape)
            return RegistrationResult(transformation=np.eye(4), registration_score=-1,
                                      dist_src_to_dst=np.inf, dist_dst_to_src=np.inf)

        start = time.time()
        if self.icp_refinement is not None:
            result = self.icp_refinement(src, dst, expected_point_distance=expected_point_distance,
                                         init_trafo=result.transformation, use_subsampled=False)
        # print("ICP refinement took:", time.time() - start)

        if isinstance(result, RegistrationResult) and hasattr(result, "registration_score"):
            pass
        else:
            # compute registration score:
            score, dst_to_src_dist, src_to_dst_dist = calc_registration_score(src.pcd, dst.pcd, result.transformation)
            result = RegistrationResult(transformation=result.transformation,
                                        o3d_registration_result=result,
                                        registration_score=score,
                                        dist_src_to_dst=src_to_dst_dist,
                                        dist_dst_to_src=dst_to_src_dist)

        return result


# def estimate_success_threshold(part_catalog, segmentation_catalog, num_pcds_used=20, empirical_threshold=0.7):
#     # combine pcds from catalog and segmentation and use random subset:
#     all_pcds = []
#     print("Estimating success threshold...")
#     for i, part_mesh in enumerate(part_catalog.catalog_part_meshes.values()):
#         all_pcds.append(part_mesh.mesh_from_volume.pcd_with_properties.pcd)
#     for i, segment_mesh in enumerate(segmentation_catalog.catalog_segmentation_meshes.values()):
#         all_pcds.append(segment_mesh.pcd_with_properties.pcd)
#
#     np.random.shuffle(all_pcds)
#     list_estimated_dists = []
#     for pcd in tqdm(all_pcds[:num_pcds_used]):
#         list_estimated_dists.append(estimate_average_point_distance(pcd))
#
#     # heuristically choose a successful registration threshold:
#     # registration score is 1 / (average dist src target + average dist target src)
#     expected_point_distance = np.mean(list_estimated_dists)
#     success_registration_score = empirical_threshold / expected_point_distance
#
#     return success_registration_score, expected_point_distance
