import numba
import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d
import time
from tqdm import tqdm
import torch

from deconstruct.utils.geometries.neighbor_search import faiss_nn


class FeatureCorrespondenceFinder:

    def __init__(self, directed=False, mutual_check=True, tuple_check=True, tuple_check_kwargs=None, num_workers=4,
                 try_to_use_gpu=True):
        assert not (directed and mutual_check), "cannot be both directed and mutual check"
        self.directed = directed
        self.mutual_check = mutual_check
        self.tuple_check = tuple_check
        self.tuple_check_kwargs = {} if tuple_check_kwargs is None else tuple_check_kwargs
        self.num_workers = num_workers
        self.use_gpu = True if try_to_use_gpu and torch.cuda.is_available() else False
        if try_to_use_gpu and not torch.cuda.is_available():
            print("IMPORTANT: gpu not available for feature neighbor search, using cpu implementation instead.")
        if self.use_gpu:
            print("using fast gpu implementation. make sure that features are of type float32")

    def __call__(self, feature1, feature2):
        # start = time.time()
        if self.directed:
            corres = find_directed_correspondences(feature1, feature2, num_workers=self.num_workers,
                                                   use_gpu=self.use_gpu)
        else:
            corres = find_symmetric_correspondences(feature1, feature2, num_workers=self.num_workers,
                                                    mutual_check=self.mutual_check, use_gpu=self.use_gpu)
        # print(f"found {corres.shape[0]} correspondences in {time.time() - start:.2f} s")

        if self.tuple_check:
            # start = time.time()
            corres = tuple_check_correspondences(feature1, feature2, corres, **self.tuple_check_kwargs)
            # print(f"after tuple check there are {corres.shape[0]} correspondences, in {time.time() - start:.2f} s")
        return corres


def find_directed_correspondences(feature1, feature2, num_workers=4, use_o3d_kdtree=False, use_gpu=True):
    """
    use cKDTree to find nearest neighbor correspondences
    for each feat in feature1 find the nearest neighbor in feature2
    features are expected to be of shape (num_points, num_features_per_point)
    """
    if feature1.shape[0] > 30000:
        print(f"warning: large number of features ({feature1.shape[0]}), "
              f"this might take a while, consider downsampling")

    if feature2.shape[0] < 1000 and use_o3d_kdtree:
        print("small number of features, using exact cDTree instead of approximate o3d KDTreeFlann.")
        use_o3d_kdtree = False

    if use_o3d_kdtree:
        corres = []
        feature_tree = o3d.geometry.KDTreeFlann(feature2.T)
        for feat in tqdm(feature1):
            corres += feature_tree.search_knn_vector_xd(feat, 1)[1]
        corres = np.asarray(corres)
    elif use_gpu:
        corres = faiss_nn(feature2, feature1, k=1).ravel()
    else:
        feature_tree = cKDTree(feature2)
        _, corres = feature_tree.query(feature1, k=1, workers=num_workers)

    return corres


def find_symmetric_correspondences(feature1, feature2, num_workers=4, mutual_check=True, use_gpu=True):
    """for now only works with neighbors_per_feat=1"""
    if mutual_check:
        corres = find_mutual_correspondences_efficiently(feature1, feature2, num_workers=num_workers, use_gpu=use_gpu)
    else:
        closest_to_feat1 = find_directed_correspondences(feature1, feature2, num_workers=num_workers, use_gpu=use_gpu)
        closest_to_feat2 = find_directed_correspondences(feature2, feature1, num_workers=num_workers, use_gpu=use_gpu)

        corres = []
        for i, j in enumerate(closest_to_feat1):
            corres.append([i, j])
        for j, i in enumerate(closest_to_feat2):
            corres.append([i, j])

        corres = np.asarray(corres)

    return corres


def find_mutual_correspondences_efficiently(feature1, feature2, num_workers=4, use_gpu=True):
    # identify smaller and larger feature set:
    swap = False
    if len(feature1) > len(feature2):
        swap = True
        feature1, feature2 = feature2, feature1  # feature1 is smaller

    # find correspondences:
    closest_to_feat1 = find_directed_correspondences(feature1, feature2, num_workers=num_workers, use_gpu=use_gpu)

    closest_to_feat2 = np.ones(len(feature2), dtype=int) * -1
    unique_closest_to_feat1 = np.unique(closest_to_feat1)
    closest_to_feat2[unique_closest_to_feat1] = find_directed_correspondences(feature2[unique_closest_to_feat1],
                                                                              feature1,
                                                                              num_workers=num_workers, use_gpu=use_gpu)

    mutual_corres = []
    for i, j in enumerate(closest_to_feat1):
        if closest_to_feat2[j] == i:
            mutual_corres.append([i, j])

    mutual_corres = np.asarray(mutual_corres)

    if swap:
        mutual_corres = mutual_corres[:, ::-1]

    return mutual_corres


@numba.njit
def numba_tuple_check_correspondences(points1, points2, corres, relative_num_checks=100, tolerance=0.95,
                                      max_num_tuples=2000):
    """
    follwing
    https://github.com/isl-org/Open3D/blob/master/cpp/open3d/pipelines/registration/FastGlobalRegistration.cpp#L60
    """
    # randomly select 3 correspondences:
    eps = 1e-7
    num_checks = corres.shape[0] * relative_num_checks

    # generate random 3-tuples:
    tuple_indices = np.random.randint(0, len(corres), size=(num_checks, 3))
    tuple_corres = corres[tuple_indices.ravel()].reshape((num_checks, 3, 2))   # (num_checks, 3, 2)

    accepted_corres = []
    count = 0
    for triple_corres in tuple_corres:
        # get the points from 1 and from 2:
        points1_triple = points1[triple_corres[:, 0]]  # (3, 3)
        points2_triple = points2[triple_corres[:, 1]]  # (3, 3)

        # do the 3 checks:
        check = True
        for pair in [[0, 1], [0, 2], [1, 2]]:
            norm1 = np.linalg.norm(points1_triple[pair[0]] - points1_triple[pair[1]])
            norm2 = np.linalg.norm(points2_triple[pair[0]] - points2_triple[pair[1]])
            if tolerance < norm1 / (norm2 + eps) < 1 / tolerance:
                pass
            else:
                check = False
                break

        if check:
            accepted_corres.append(triple_corres)
            count += 1
            if count >= max_num_tuples:
                break

    return accepted_corres


def tuple_check_correspondences(points1, points2, corres, relative_num_checks=100, tolerance=0.95, max_num_tuples=2000,
                                make_unique=True):
    accepted_corres = numba_tuple_check_correspondences(points1, points2, corres,
                                                        relative_num_checks=relative_num_checks,
                                                        max_num_tuples=max_num_tuples,
                                                        tolerance=tolerance)

    if len(accepted_corres) == 0:
        return np.empty((0, 2), dtype=int)

    accepted_corres = np.concatenate(accepted_corres, axis=0)

    if make_unique:
        accepted_corres_set = set(tuple(corr) for corr in accepted_corres)
        accepted_corres = np.asarray(list(accepted_corres_set))

    return accepted_corres


def tuple_check_correspondences_parallel(points1, points2, corres, relative_num_checks=1, tolerance=0.95,
                                         max_num_tuples=None, make_unique=False):
    """this is not working so well, because the max_num_tuple stop can not be included efficiently"""

    # randomly select 3 correspondences:
    eps = 1e-7
    num_checks = corres.shape[0] * relative_num_checks

    # generate random 3-tuples:
    tuple_indices = np.random.randint(0, len(corres), size=num_checks * 3)
    tuple_corres = corres[tuple_indices]  # (num_checks * 3, 2)

    tuple_points1 = points1[tuple_corres[:, 0]].reshape(num_checks, 3, 3)  # (num_checks, 3, 3)
    tuple_points2 = points2[tuple_corres[:, 1]].reshape(num_checks, 3, 3)  # (num_checks, 3, 3)

    # calculate norms for 3 pairs:
    norms1a = np.linalg.norm(tuple_points1[:, 0, :] - tuple_points1[:, 1, :], axis=-1)
    norms2a = np.linalg.norm(tuple_points2[:, 0, :] - tuple_points2[:, 1, :], axis=-1)
    norms1b = np.linalg.norm(tuple_points1[:, 0, :] - tuple_points1[:, 2, :], axis=-1)
    norms2b = np.linalg.norm(tuple_points2[:, 0, :] - tuple_points2[:, 2, :], axis=-1)
    norms1c = np.linalg.norm(tuple_points1[:, 1, :] - tuple_points1[:, 2, :], axis=-1)
    norms2c = np.linalg.norm(tuple_points2[:, 1, :] - tuple_points2[:, 2, :], axis=-1)

    # do the 3 checks:
    checks = np.empty((num_checks, 3), dtype=bool)
    checks[:, 1] = (tolerance < norms1a / (norms2a + eps)) & (norms1a / (norms2a + eps) < 1 / tolerance)
    checks[:, 2] = (tolerance < norms1b / (norms2b + eps)) & (norms1b / (norms2b + eps) < 1 / tolerance)
    checks[:, 3] = (tolerance < norms1c / (norms2c + eps)) & (norms1c / (norms2c + eps) < 1 / tolerance)

    checks = np.all(checks, axis=-1)
    accepted_corres = tuple_corres.reshape(num_checks, -1, 2)[checks].reshape(-1, 2)

    if make_unique:
        accepted_corres_set = set(tuple(corr) for corr in accepted_corres)
        accepted_corres = np.asarray(list(accepted_corres_set))

    return accepted_corres[:max_num_tuples]


