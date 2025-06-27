import open3d as o3d
import numpy as np
import torch
import torch_geometric as tg
from deconstruct.utils.geometries.neighbor_search import pcd_get_neighbors


def o3d_fpfh(pcd, radius, max_nn, return_float32_arr=False):
    """
    Wrapper around o3d implementation to compute fpfh-feature for point clouds

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
    radius : float
        radius in neighbor search
    max_nn : int
        maximum number of neighbors considered.

    Returns
    -------
    o3d.pipelines.registration.Feature
        FPFH feature vectors one per point.
    """
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))
    if return_float32_arr:
        return fpfh, np.asarray(fpfh.data, dtype=np.float32).T
    else:
        return fpfh


def get_closest_local_properties(pcd, subsampled_pcd, local_properties):
    """this only works if local properties are of the format (num_points, num_features)"""
    subsampled_local_properties = {}
    # find nearest neighbors for each point and take local properties from them:
    neighbors_dict, _ = pcd_get_neighbors(subsampled_pcd, pcd, knn=1)
    selected_indices = [neighbor_dict["neighbor_indices"][0] for neighbor_dict in neighbors_dict.values()]
    for key, value in local_properties.items():
        subsampled_local_properties[key] = value[selected_indices]


def random_subsample_pcd(pcd, num_points=None, fraction=None):
    assert (num_points is None) != (fraction is None), "Either num_points or fraction must be given."
    if fraction is not None:
        num_points = int(fraction * len(pcd.points))

    random_indices = np.random.choice(len(pcd.points), num_points, replace=False)
    return pcd.select_by_index(random_indices)


def subsample_pcd_by_indices(pcd, indices):
    sampled_pcd = o3d.geometry.PointCloud()
    sampled_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[indices])
    if pcd.has_colors():
        sampled_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[indices])
    if pcd.has_normals():
        sampled_pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[indices])

    return sampled_pcd


def voxel_down_sample(pcd, voxel_size, drop_zero_normal_points=False):
    subsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # check if normals are normalized:
    normal_lenghts = np.linalg.norm(np.asarray(subsampled_pcd.normals), axis=1)
    if not np.allclose(normal_lenghts, 1.):
        # print("normals after voxel downsampling are not normalized, normalizing them now...")
        # check if some have zero length:
        zero_length_mask = (normal_lenghts < 1e-5)
        if np.any(zero_length_mask):
            # print(f"{np.sum(zero_length_mask)} normals have zero length, setting them to zero vector.")
            pass

        normals = np.asarray(subsampled_pcd.normals)
        normals[zero_length_mask, :] = 0.
        normals[~zero_length_mask, :] /= normal_lenghts[~zero_length_mask, None]
        subsampled_pcd.normals = o3d.utility.Vector3dVector(normals)

        if drop_zero_normal_points:
            if zero_length_mask.sum() / len(zero_length_mask) > 0.1:
                print("Warning: more than 10% of points have zero length normals.")
            subsampled_pcd = subsample_pcd_by_indices(subsampled_pcd, ~zero_length_mask)
            assert all(np.linalg.norm(np.asarray(subsampled_pcd.normals), axis=1) > 0.9)

    return subsampled_pcd


class FPSampler(torch.nn.Module):
    def __init__(self, ratio=None, num_points=None, random_start=True, use_cuda=True):
        super().__init__()
        assert (ratio is None) != (num_points is None), "Either ratio or num_points must be given."
        self.ratio = ratio
        self.num_points = num_points
        self.random_start = random_start
        self.use_cuda = use_cuda

    def forward(self, pos, batch=None):
        # get ratio:
        ratio = self.ratio if self.ratio is not None else self.num_points / pos.shape[0]
        if ratio == 1:
            return torch.arange(pos.shape[0], device=pos.device)
        pos = pos.to("cuda") if self.use_cuda else pos
        return tg.nn.fps(x=pos, batch=batch, ratio=ratio, random_start=self.random_start).cpu().numpy()


def voxel_presampling(pcd, desired_num_points, voxelization_scale, local_properties=None, empirical_correction=0.85):
    expected_point_distance = 1 / voxelization_scale
    # sqrt in case of surface meshes.
    voxel_size = (len(pcd.points)/desired_num_points)**(1/2) * expected_point_distance * empirical_correction
    subsampler = PCDSubsampler(method="voxel", num_points_threshold=desired_num_points,
                               method_kwargs={"voxel_size": voxel_size, "drop_zero_normal_points": True})
    return subsampler(pcd, local_properties=local_properties)


def random_presampling(pcd, desired_num_points, local_properties=None):
    subsampler = PCDSubsampler(method="random", num_points_threshold=desired_num_points,
                               method_kwargs={"num_points": desired_num_points})
    return subsampler(pcd, local_properties=local_properties)


class PCDSubsampler:

    def __init__(self, method="voxel", num_points_threshold=1000, method_kwargs=None):
        self.method = method
        self.method_kwargs = {} if method_kwargs is None else method_kwargs
        self.num_points_threshold = num_points_threshold

    def __call__(self, pcd, local_properties=None):
        if len(pcd.points) <= self.num_points_threshold:
            # do not subsample
            return pcd, local_properties

        if self.method == "voxel":
            sampled_pcd = voxel_down_sample(pcd, **self.method_kwargs)
        elif self.method == "random":
            sampled_pcd = random_subsample_pcd(pcd, **self.method_kwargs)
        elif self.method == "fps":
            # fp_sampler = FPSampler(**self.method_kwargs)
            # sampled_indices = fp_sampler(torch.from_numpy(np.asarray(pcd.points)))
            # sampled_pcd = pcd.select_by_index(list(sampled_indices))

            # note that there is also a open3d function for this which is quite fast:
            sampled_pcd = pcd.farthest_point_down_sample(num_samples=self.method_kwargs["num_points"])
        else:
            raise NotImplementedError(f"{self.method=} not implemented.")

        # todo implement subsampling for local properties.
        # so far local properties are ignored.
        sampled_local_properties = {}

        # check that colors and normals where preserved:
        assert (sampled_pcd.has_colors() == pcd.has_colors()) and (sampled_pcd.has_normals() == pcd.has_normals()), \
            "color and normals should be preserved during subsampling."

        return sampled_pcd, sampled_local_properties


