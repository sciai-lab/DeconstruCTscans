import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
import time
import ot
from deconstruct.utils.geometries.neighbor_search import faiss_nn


class BagOfVisualWords:
    def __init__(self, num_centers):
        self.num_centers = num_centers

    def fit(self, reference_features):
        """are expected to be of shape (num_reference_objects, num_features_per_object, feature_dim)"""
        assert len(reference_features.shape) == 3, ("reference_features must be of shape "
                                                    "(num_reference_objects, num_features_per_object, feature_dim)")
        self.feature_dim = reference_features.shape[-1]
        num_reference_objects = reference_features.shape[0]
        num_features_per_object = reference_features.shape[1]
        start = time.time()
        self.kmeans = KMeans(n_clusters=self.num_centers, n_init="auto").fit(
            reference_features.reshape(-1, self.feature_dim))
        print(f"KMeans took {time.time() - start} seconds.")
        reference_labels_one_hot = np.zeros((len(self.kmeans.labels_), self.num_centers), dtype=int)
        reference_labels_one_hot[np.arange(len(self.kmeans.labels_)), self.kmeans.labels_] = 1
        reference_labels_one_hot = reference_labels_one_hot.reshape(num_reference_objects, num_features_per_object,
                                                                    self.num_centers)
        return reference_labels_one_hot.sum(axis=1)

    def predict(self, features):
        """are expected to be of shape (num_query_objects, num_features, feature_dim)"""
        assert len(features.shape) == 3, "features must be of shape (num_query_objects, num_features, feature_dim)"
        assert features.shape[-1] == self.feature_dim, "feature dim does not match the one used for fitting"
        labels = self.kmeans.predict(features.reshape(-1, features.shape[-1]))
        labels_one_hot = np.zeros((len(labels), self.num_centers), dtype=int)
        labels_one_hot[np.arange(len(labels)), labels] = 1
        return labels_one_hot.reshape(features.shape[0], features.shape[1], self.num_centers).sum(axis=1)


def pcd_dist(points1, points2, combine="min", last_channel_is_mass=True, ignore_worst_percentile=0.0):
    """distance from points1 to points2"""
    assert points1.shape[1] == points2.shape[1], "points must have same dimension"

    if last_channel_is_mass:
        points1 = np.ascontiguousarray(points1[:, :-1])
        points2 = np.ascontiguousarray(points2[:, :-1])  # contiguous array is required for faiss

    corres1, distance1 = faiss_nn(points1, points2, k=1, return_distances=True)
    corres2, distance2 = faiss_nn(points2, points1, k=1, return_distances=True)

    if ignore_worst_percentile > 0.0:
        num_to_ignore = int(ignore_worst_percentile * len(distance1))
        # print("ignoring worst percentile", num_to_ignore, ignore_worst_percentile)
        if num_to_ignore > 0:
            # sort distances:
            distance1 = np.sort(distance1)
            distance2 = np.sort(distance2)
            # ignore worst percentile:
            distance1 = distance1[:-num_to_ignore]
            distance2 = distance2[:-num_to_ignore]

    if combine == "min":
        return min(distance1.mean(), distance2.mean())
    elif combine == "mean":
        return (distance1.mean() + distance2.mean()) / 2
    else:
        raise NotImplementedError(f"{combine=} not implemented.")


def ot_pcd_dist(points1, points2, normalize=True, last_channel_is_mass=True):
    """distance from points1 to points2"""
    assert points1.shape[1] == points2.shape[1], "points must have same dimension"

    if last_channel_is_mass:
        mass1 = points1[:, -1]
        mass2 = points2[:, -1]
        points1 = points1[:, :-1]
        points2 = points2[:, :-1]
    else:
        mass1 = np.ones(points1.shape[0])/points1.shape[0]
        mass2 = np.ones(points2.shape[0])/points2.shape[0]

    if normalize:
        mass1 = mass1/mass1.sum()
        mass2 = mass2/mass2.sum()

    # use OT package to solve optimal transport
    M = ot.dist(points1, points2, 'euclidean')
    X = ot.emd(mass1, mass2, M)  # solve optimal transport # X[i,j] is amount of mass which source i sends to sink j
    distance = np.sum(X * M)
    return distance


class PointCloudMatcher:

    auto_distance_choise = {
        "volume": "abs_diff",
        "average_fpfh": "euclidean",
        "bovw": "euclidean",
        "feature_centers": "pcd_dist"
    }

    def __init__(self, property_key="average_fpfh", distance_metric="euclidean", verbose=True, use_subsampled=True,
                 use_parallel=False, distance_metric_kwargs=None):
        self.verbose = verbose
        self.use_subsampled = use_subsampled
        self.property_key = property_key
        self.use_parallel = use_parallel
        self.distance_metric_kwargs = {} if distance_metric_kwargs is None else distance_metric_kwargs

        if distance_metric == "auto":
            distance_metric = self.auto_distance_choise[self.property_key]

        if distance_metric == "abs_diff":
            self.distance_metric = lambda x, y: np.abs(x-y)
        elif distance_metric == "euclidean":
            self.distance_metric = lambda x, y: np.linalg.norm(x-y, axis=-1)
        elif distance_metric == "pcd_dist":
            self.distance_metric = pcd_dist
        elif distance_metric == "ot_pcd_dist":
            self.distance_metric = ot_pcd_dist
        else:
            raise NotImplementedError(f"{distance_metric=} not implemented.")

    def non_parallel_call(self, part_catalog, segmentation_catalog):
        matching_dict = {}
        list_part_names = list(part_catalog.catalog_part_meshes.keys())

        for label, mesh_from_volume in tqdm(segmentation_catalog.catalog_segmentation_meshes.items(), disable=not self.verbose):
            pcd_with_properties = mesh_from_volume.pcd_with_properties
            global_properties = pcd_with_properties.subsampled_global_properties if self.use_subsampled \
                else pcd_with_properties.global_properties
            segm_feature = global_properties[self.property_key]
            feature_diff = []
            for part_name in list_part_names:
                pcd_with_properties = part_catalog.catalog_part_meshes[part_name].mesh_from_volume.pcd_with_properties
                global_properties = pcd_with_properties.subsampled_global_properties if self.use_subsampled \
                    else pcd_with_properties.global_properties
                part_feature = global_properties[self.property_key]
                feature_diff.append(self.distance_metric(segm_feature, part_feature, **self.distance_metric_kwargs))

            sorted_indices = np.argsort(feature_diff)
            matching_dict[label] = [(list_part_names[ind], feature_diff[ind]) for ind in sorted_indices]

        return matching_dict

    def __call__(self, part_catalog, segmentation_catalog):
        """
        match pcds from catalog to segmentation, must be such that best matches come first
        """
        if not self.use_parallel:
            print("Using non-parallel matching version.")
            return self.non_parallel_call(part_catalog, segmentation_catalog)

        matching_dict = {}

        # build a feature vector for the catalog:
        list_catalog_features = []
        list_part_names = []
        for part_name, part_mesh in part_catalog.catalog_part_meshes.items():
            pcd_with_properties = part_mesh.mesh_from_volume.pcd_with_properties
            global_properties = pcd_with_properties.subsampled_global_properties if self.use_subsampled \
                else pcd_with_properties.global_properties
            feature = global_properties[self.property_key]
            list_catalog_features.append(feature)
            list_part_names.append(part_name)

        # check if feature is a number or an array:
        if isinstance(feature, (int, float)):
            catalog_features = np.array(list_catalog_features)
        else:
            catalog_features = np.stack(list_catalog_features, axis=0)

        print("Catalog features have shape:", catalog_features.shape)

        print("Perform matching of catalog parts to segmentation...")
        for label, mesh_from_volume in tqdm(segmentation_catalog.catalog_segmentation_meshes.items()):
            pcd_with_properties = mesh_from_volume.pcd_with_properties
            global_properties = pcd_with_properties.subsampled_global_properties if self.use_subsampled \
                else pcd_with_properties.global_properties
            feature = global_properties[self.property_key]

            # compare against all catalog features and rank catalog parts based on that:
            # print("feature.shape:", feature.shape)
            feature_diff = self.distance_metric(catalog_features, feature)
            # print("feature_diff.shape:", feature_diff.shape)
            sorted_indices = np.argsort(feature_diff)
            matching_dict[label] = [(list_part_names[ind], feature_diff[ind]) for ind in sorted_indices]

        return matching_dict


