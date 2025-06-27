import numpy as np

from deconstruct.utils.geometries.neighbor_search import compute_mesh_neighbors

# this is currently not in use and may be deprecated.


def mesh_normal_smoothing(trmesh, neighbors_dict, kernel="flat", normal_length_clip=0.1, iterations=1, lamb=0.5):
    """
    Smooths the vertex normals of a mesh.

    Parameters
    ----------
    trmesh : tr.Trimesh
        Mesh to be smoothed.
    neighbors_dict : dict[int, dict]
        Neighbor dictionary: The vertex indices are the keys. Each vertex dictionary has the structure:
        {"neighbor_indices" : list of selected indices,
         "neighbor_dist": np.ndarray with distance to selceted points},
    kernel : str, optional
        How to smooth normals. Default is "flat".
    normal_length_clip : float, optional
        Set normals to np.zeros(3) if their length is smaller than this threshold. Default = 0.1.
    iterations : int, optional
        How often to smooth. Default = 1.
    lamb : float, optional
        Between [0,1]. Normals are updated like: new = old + lamb * smooth_normals

    Returns
    -------
    np.ndarray, np.ndarray
        Smoothed normals (normalized to length 1 or 0) of shape (num vertices, 3),
        Mask where smoothed normals where too short of shape 8num vertices, ).
    """
    normals = trmesh.vertex_normals.copy()
    for i in range(iterations):
        smooth_normals = np.zeros_like(normals)
        for point_idx in neighbors_dict:
            neighbors = neighbors_dict[point_idx]["neighbor_indices"]
            if kernel == "flat":
                smooth_normals[point_idx] = np.mean(normals[neighbors], axis=0)
            else:
                raise NotImplementedError(f"{kernel=} not implemented.")

        # check the length and normalize:
        normals = normals + lamb * (smooth_normals - normals)
        lengths_arr = np.sqrt(np.sum(normals ** 2, axis=1))
        too_short_mask = lengths_arr < normal_length_clip
        lengths_arr[too_short_mask] = np.inf
        normals = normals / lengths_arr[:, None]

    return normals, too_short_mask


def mesh_point_smoothing(trmesh, neighbors_dict, kernel="flat", iterations=1, lamb=0.5):
    """
    Smooths the vertex coordinates of a mesh.

    Parameters
    ----------
    trmesh : tr.Trimesh
            Mesh to be smoothed.
    neighbors_dict : dict[int, dict]
        Neighbor dictionary: The vertex indices are the keys. Each vertex dictionary has the structure:
        {"neighbor_indices" : list of selected indices,
         "neighbor_dist": np.ndarray with distance to selceted points},
    kernel : str, optional
        How to smooth normals. Default is "flat".
    iterations : int, optional
        How often to smooth. Default = 1.
    lamb : float, optional
        Between [0,1]. Points are updated like: new = old + lamb * smooth_points. Default = 0.5.

    Returns
    -------
    tr.Trimesh
        Mesh with smoothed vertices.
    """

    points = trmesh.vertices.copy()
    for i in range(iterations):
        smooth_points = np.zeros_like(points)
        for point_idx in neighbors_dict:
            neighbors = neighbors_dict[point_idx]["neighbor_indices"]
            if kernel == "flat":
                smooth_points[point_idx] = np.mean(points[neighbors], axis=0)
            else:
                print("other kernels not yet implemented.")

        points = points + lamb * (smooth_points - points)

    trmesh.vertices = points
    return trmesh


def smooth_mesh_points_and_normals(trmesh, kwargs_for_points_neighbor_search, kwargs_for_normal_neighbor_search=None,
                                   lamb_points=0.8,
                                   lamb_normals=1.0,
                                   iter_points=1, iter_normals=1,
                                   kernel_points="flat", kernel_normals="flat"):
    """
    Smooths the vertex coordinates and the normals of a mesh.

    Parameters
    ----------
    trmesh : tr.Trimesh
        Mesh to be smoothed.
    kwargs_for_points_neighbor_search : dict
        Kwargs for compute_mesh_neighbors.
        method_specific_kwargs : dict
            Args and Kwargs which are specific for the chosen method.
            "adjacency":
                max_hops : int
                Number of hops (i.e. multiplication of adjacency matrix).
                search_dist : float, optional.
                    Threshold for pose-selecting the neighbors. Default is no post-thresholding.
                sort_by_dist : bool, optional
                    Specifies whether for each point the selected neighbors are sorted based on the Euclidean distance.
                    Default is False.
                info_acceptance_ratio : bool, optional
                    Whether a print statement should be shown to increase of decrease the number of hops. Default = False.
            "dijkstra":
                search_dist : float
                    Cut off distance for Dijkstra search.
            "pcd":
                knn : int, optional
                    Needed if search radius is None. Find up to this many neighbors.
                search_radius : float, optional
                    Needed if knn is None. Search neighbors up to this radius.
                keep_only_connected_neighbors : bool, optional
                    Specifies if disconnected selected neighbors should be discarded. Connection is checked on the mesh.
                    Default = False.
                tqdm_bool : bool, optional
                    Specifies if progress bar is shown.
        method : str, optional
            One of the following: "adjacency", "dijkstra", "pcd". Default = "adjacency".
        minimum_num_neighbors : int, optional
            Print a warning if a vertex gets less than this many neighbors. Default = 5.
    kwargs_for_normal_neighbor_search : dict, optional
        By default use the same as for points.
    lamb_points : float, optional
        Between [0,1]. Normals are updated like: new = old + lamb_points * smooth_points
    lamb_normals : float, optional
        Between [0,1]. Normals are updated like: new = old + lamb_normals * smooth_normals
    iter_points : int, optional
        How often to smooth. Default = 1.
    iter_normals : int, optional
        How often to smooth. Default = 1.
    kernel_points : str, optional
        How to smooth normals. Default is "flat".
    kernel_normals : str, optional
        How to smooth normals. Default is "flat".

    Returns
    -------
    tr.Trimesh, np.ndarray
        Smoothed mesh, Array of smoothed vertex normals of shape (num vertices, 3).
    """

    points_neighbors_dict = compute_mesh_neighbors(trmesh, **kwargs_for_points_neighbor_search)

    # neighbors for normal smoothing:
    if kwargs_for_normal_neighbor_search is None:
        normals_neighbors_dict = points_neighbors_dict
    else:
        normals_neighbors_dict = compute_mesh_neighbors(trmesh, **kwargs_for_normal_neighbor_search)

    # smooth the points:
    trmesh_smoothed = mesh_point_smoothing(trmesh, points_neighbors_dict, kernel=kernel_points,
                                           iterations=iter_points, lamb=lamb_points)

    # smooth the normals:
    normals_smoothed, too_short_mask = mesh_normal_smoothing(trmesh_smoothed, normals_neighbors_dict,
                                                             iterations=iter_normals,
                                                             lamb=lamb_normals, kernel=kernel_normals,
                                                             normal_length_clip=0.05)
    if np.sum(1 - too_short_mask) > 0:
        print(f"{np.sum(1 - too_short_mask)} many normals were too short.")

    return trmesh_smoothed, normals_smoothed
