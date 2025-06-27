import numpy as np
import open3d as o3d
import sklearn.neighbors
from scipy.spatial import cKDTree


def find_mutual_feature_correspondences(src_features, dst_features, use_scipy=True):
    if not use_scipy:
        print("Warning: it was found that despite being faster the o3d KDTreeFlann is not as accurate as the scipy."
              "In particular, it seems to not be equivariant under permutation of points.")

    if use_scipy:
        src_feature_tree = cKDTree(src_features)
        dst_feature_tree = cKDTree(dst_features)
    else:
        src_feature_tree = o3d.geometry.KDTreeFlann(src_features.T)
        dst_feature_tree = o3d.geometry.KDTreeFlann(dst_features.T)

    corres_ij = {}
    corres_ji = {}
    mutual_corres = []

    for j, feat in enumerate(dst_features):
        if use_scipy:
            i = src_feature_tree.query(feat, k=1)[1]
        else:
            i = src_feature_tree.search_knn_vector_xd(feat, 1)[1][0]
        corres_ji[j] = i

        if i not in corres_ij:
            if use_scipy:
                corres_ij[i] = dst_feature_tree.query(src_features[i], k=1)[1]
            else:
                corres_ij[i] = dst_feature_tree.search_knn_vector_xd(src_features[i], 1)[1][0]

    for i, j in corres_ij.items():
        if corres_ji[j] == i:
            mutual_corres.append([i, j])

    return np.asarray(mutual_corres)


def find_feature_based_correspondences(src_features, dst_features, num_correspondences_per_key_point=1,
                                       key_indices=None, feature_metric="eucl_dist"):
    """
    Finds correspondences based on nearest neighbor search in feature space.

    Parameters
    ----------
    src_features : np.ndarray
        Features of all points in source pcd.
    dst_features : np.ndarray
        Features of all points in target pcd.
    num_correspondences_per_key_point : int
        Number of correspondences which are formed per key point.
    key_indices : np.ndarray[int], optional
        Used to specify which points in src pcd are key points.
    feature_metric : str, optional
        Metric used to measure distances in feature space. Default = "eucl_dist" (i.e. Euclidean distance).

    Returns
    -------
    np.ndarray(int), np.ndarray(float)
        The correspondences as index pairs. of shape (num key points * num_correspondences_per_key_point, 2)
        Convention: they should be sorted such that the closest ones come first,
        feature distance of correspondences of shape (num key points, num_correspondences_per_key_point).
    """
    if key_indices is None:
        key_indices = np.arange(len(src_features))
        key_src_features = src_features
    else:
        key_src_features = src_features[key_indices]

    if feature_metric == "eucl_dist":
        sklearn_kNN = sklearn.neighbors.NearestNeighbors(n_neighbors=num_correspondences_per_key_point)
        sklearn_kNN.fit(dst_features)

        # find closest neighbors to src_key_points in features space (for now fixed number and kNN serach):
        feature_pair_dist, knn_indices = sklearn_kNN.kneighbors(key_src_features)  # out_shape: (num key points, k)

        assert (feature_pair_dist == np.sort(feature_pair_dist)).all(), "correspondences should be sorted " \
                                                                        "by feature distance"

        # create correspondences:
        key_correspondences = np.zeros((len(key_src_features) * num_correspondences_per_key_point, 2), dtype=int)
        key_correspondences[:, 0] = key_indices.repeat(num_correspondences_per_key_point)
        key_correspondences[:, 1] = knn_indices.flatten()

    else:
        raise NotImplementedError(f"{feature_metric=} is not implemented.")

    return key_correspondences, feature_pair_dist
