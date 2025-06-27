import networkx as nx
import numpy as np
import open3d as o3d
import scipy
from tqdm.auto import tqdm
from scipy.spatial import cKDTree
import faiss

faiss_res = faiss.StandardGpuResources()  # Initialize the GPU resources

def faiss_build_index(xb, mode='GpuIndexFlatL2'):
    d = xb.shape[1]  # dimension

    if mode == 'GpuIndexFlatL2':  # brute force on the GPU
        index = faiss.GpuIndexFlatL2(faiss_res, d)
    elif mode == 'IndexFlatL2':  # brute force on the CPU
        index = faiss.IndexFlatL2(d)
    elif mode == 'GpuIndexIVFPQ':  # fancy approximate method on the GPU: building index takes time, querying it is fast
        nlist = 100
        m = d  # 1 # d must be divisible by m. Also, I found that only some values work, e.g. powers of 2.
        nbits = 8
        quantizer = faiss.IndexFlatL2(d)  # the quantizer
        index_cpu = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
        index = faiss.index_cpu_to_gpu(faiss_res, 0, index_cpu)  # convert to GPU index
        index.train(xb)
    else:
        raise ValueError(f'Unknown mode: "{mode}"')

    index.add(xb)

    return index


def faiss_nn(xb, xq, k=1, mode='GpuIndexFlatL2', return_distances=False):
    assert xb.dtype == np.float32, "xb must be float32"
    assert xq.dtype == np.float32, "xq must be float32"
    index = faiss_build_index(xb, mode)

    # Performing a nearest neighbor search on the GPU
    distances, indices = index.search(xq, k)

    if return_distances:
        return indices, np.sqrt(distances)
    else:
        return indices


def pcd_get_neighbors(pcd1, pcd2=None, knn=None, search_radius=None, return_distances=False, num_workers=4):
    """Find the closest points in pcd2 for each point in pcd 1. Either knn many or threshold by a serach radius."""
    assert (knn is None) != (search_radius is None), "Either knn or search_radius must be not None!"
    pcd2 = pcd1 if pcd2 is None else pcd2
    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)

    # get closest neighbours (only by Euclidean distance):
    kdtree = cKDTree(points2)
    if search_radius is None:
        distances, indices = kdtree.query(points1, k=knn, workers=num_workers)
        k_arr = np.ones(len(pcd1.points), dtype=int) * knn
        neighbors_dict = {}
        for i, ind in enumerate(indices):
            neighbors_dict[i] = {"neighbor_indices": ind,
                                 "neighbor_dist": distances[i]}
    else:
        indices = kdtree.query_ball_point(points1, r=search_radius)
        k_arr = np.asarray([len(ind) for ind in indices])
        neighbors_dict = {}
        for i, ind in enumerate(indices):
            neighbors_dict[i] = {"neighbor_indices": ind}
        if return_distances:
            # create all pairs:
            pairs = np.asarray([[ind1, ind2] for ind1, ind2_list in enumerate(indices) for ind2 in ind2_list])
            distances = np.linalg.norm(points1[pairs[:, 0]] - points2[pairs[:, 1]], axis=1)
            start_ind = 0
            for i, ind in enumerate(indices):
                neighbors_dict[i]["neighbor_dist"] = distances[start_ind: start_ind + len(ind)]
                start_ind += len(ind)

    return neighbors_dict, k_arr


def pcd_get_neighbors_old(pcd1, pcd2=None, knn=None, search_radius=None):
    """
    Find the closest points in pcd2 for each point in pcd 1. Either knn many or threshold by a serach radius.

    Parameters
    ----------
    pcd1 : o3d.geometry.PointCloud
        For each point in here find the closest points in pcd2.
    pcd2 : o3d.geometry.PointCloud, optional
        In here find the "neighbors". Default = pcd1.
    knn : int, optional
        Needed if search radius is None. Find up to this many neighbors.
    search_radius : float, optional
        Needed if knn is None. Search neighbors up to this radius.

    Returns
    -------
    (dict[int, dict], np.ndarray)
        Neighbor dictionary: The vertex indices are the keys. Each vertex dictionary has the structure:
        {"neighbor_indices" : list of selected indices,
         "neighbor_dist": np.ndarray with distance to selceted points},
        Array with number of selected points.
    """
    if knn is None and search_radius is None:
        print("Either knn or search_radius must be not None!")
        return

    if pcd2 is None:
        pcd2 = pcd1

    # get closest neighbours (only by Euclidean distance):
    pcd_tree = o3d.geometry.KDTreeFlann(pcd2)
    k_arr = np.zeros(len(pcd1.points), dtype=int)
    neighbors_dict = {}
    for i, point in enumerate(pcd1.points):
        if search_radius is not None:
            k, NN_indices, squared_dist_arr = pcd_tree.search_radius_vector_3d(point, radius=search_radius)
        elif knn is not None:
            k, NN_indices, squared_dist_arr = pcd_tree.search_knn_vector_3d(point, knn=knn)
        neighbors_dict[i] = {"neighbor_indices": list(NN_indices),
                             "neighbor_dist": np.sqrt(np.asarray(squared_dist_arr))}
        k_arr[i] = k

    return neighbors_dict, k_arr


def mesh_get_neighbors(trmesh, knn=None, search_radius=None, keep_only_connected_neighbors=False, tqdm_bool=False):
    """
    Find closest neighbors either up to a number k or upto a search radius.

    Convention: The node itself is not counted as neighbor!

    Parameters
    ----------
    trmesh : tr.Trimesh
    knn : int, optional
        Needed if search radius is None. Find up to this many neighbors.
    search_radius : float, optional
        Needed if knn is None. Search neighbors up to this radius.
    keep_only_connected_neighbors : bool, optional
        Specifies if disconnected selected neighbors should be discarded. Connection is checked on the mesh.
        Default = False.
    tqdm_bool : bool, optional
        Specifies if progress bar is shown.

    Returns
    -------
    (dict[int, dict], np.ndarray)
        Neighbor dictionary: The vertex indices are the keys. Each vertex dictionary has the structure:
        {"neighbor_indices" : list of selected indices,
         "neighbor_dist": np.ndarray with distance to selceted points},
        Array with number of selected points.
    """

    # check that nodes have the correct labelling
    assert (np.max(trmesh.faces) - np.min(trmesh.faces)) == len(
        trmesh.vertices) - 1, "mesh labelling of vertices is not correct"

    # construct mesh graph in networkx:
    mesh_graph = nx.Graph()
    mesh_graph.add_edges_from(trmesh.edges_unique)
    nx.set_edge_attributes(mesh_graph, values=trmesh.edges_unique_length, name='length')

    # construct pcd and get NNS (only by Euclidean distance):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(trmesh.vertices)
    neighbors_dict, k_arr = pcd_get_neighbors(pcd, knn=knn, search_radius=search_radius)

    if keep_only_connected_neighbors:
        # only select connected neighbors:
        list_num_NNs = []

        for node in (tqdm(neighbors_dict) if tqdm_bool else neighbors_dict):
            NNs = neighbors_dict[node]["neighbor_indices"]
            # invert this:
            NNs_lookup_dict = dict([(y, x) for x, y in enumerate(NNs)])

            # restrict to subgraph and find connected component that contains node:
            sub_mesh_graph = mesh_graph.subgraph(NNs)
            connected_neighbors = list(nx.node_connected_component(sub_mesh_graph, node))
            connected_neighbors.remove(node)
            neighbors_dict[node]["neighbor_indices"] = connected_neighbors
            connected_neighbors_lookup = [NNs_lookup_dict[n] for n in connected_neighbors]
            neighbors_dict[node]["neighbor_dist"] = neighbors_dict[node]["neighbor_dist"][connected_neighbors_lookup]
            list_num_NNs.append(len(neighbors_dict[node]["neighbor_indices"]))

        k_arr = np.asarray(list_num_NNs)

    return neighbors_dict, k_arr


def mesh_dijktra_with_cutoff(trmesh, search_dist):
    """
    Finds closest neighbors on a mesh upto a specified shortest path distance.

    Convention: The node itself is not counted as neighbor!


    Parameters
    ----------
    trmesh : tr.Trimesh
        Mesh to find the neighbors for each point.
    search_dist : float
        Cut off distance for Dijkstra search.

    Returns
    -------
    (dict[int, dict], np.ndarray)
            Neighbor dictionary: The vertex indices are the keys. Each vertex dictionary has the structure:
            {"neighbor_indices" : list of selected indices,
             "neighbor_dist": np.ndarray with distance to selceted points},
            Array with number of selected points.
    """
    # construct mesh graph in networkx:
    mesh_graph = nx.Graph()
    mesh_graph.add_nodes_from(np.arange(len(trmesh.vertices)))
    mesh_graph.add_edges_from(trmesh.edges_unique)

    edge_length_list = [[(trmesh.edges_unique[i][0], trmesh.edges_unique[i][1]), trmesh.edges_unique_length[i]]
                        for i in range(len(trmesh.edges_unique))]
    nx.set_edge_attributes(mesh_graph, values=dict(edge_length_list), name="length")

    # Dijkstra with cutoff:
    APSP = dict(nx.all_pairs_dijkstra_path_length(mesh_graph, cutoff=search_dist, weight="length"));

    neighbors_dict = {}
    k_arr = np.zeros(mesh_graph.number_of_nodes())
    for i in APSP:
        neighbor_indices = list(APSP[i].keys())
        neighbor_indices.remove(i)
        neighbors_dict[i] = {"neighbor_indices": neighbor_indices,
                             "neighbor_dist": np.asarray(list(APSP[i].values()))}
        k_arr[i] = len(APSP[i]) - 1

    return neighbors_dict, k_arr


def mesh_adjacency_with_cutoff(trmesh, max_hops, search_dist=None, sort_by_dist=False, info_acceptance_ratio=False):
    """
    Find mesh neighbors by hop distance which his done by squaring the adjacency matrix.
    Then the neighbors may be thresholded by a Euclidean search distance.

    Convention: The node itself is not counted as neighbor!

    Parameters
    ----------
    trmesh : tr.Trimesh
        Mesh to find the neighbors for each points.
    max_hops : int
        Number of hops (i.e. multiplication of adjacency matrix).
    search_dist : float, optional.
        Threshold for pose-selecting the neighbors. Default is no post-thresholding.
    sort_by_dist : bool, optional
        Specifies whether for each point the selected neighbors are sorted based on the Euclidean distance.
        Default is False.
    info_acceptance_ratio : bool, optional
        Whether a print statement should be shown to increase of decrease the number of hops. Default = False.
    Returns
    -------
    (dict[int, dict], np.ndarray)
                Neighbor dictionary: The vertex indices are the keys. Each vertex dictionary has the structure:
                {"neighbor_indices" : list of selected indices,
                 "neighbor_dist": np.ndarray with distance to selceted points},
                Array with number of selected points.
    """

    # construct mesh graph in networkx:
    mesh_graph = nx.Graph()
    mesh_graph.add_nodes_from(np.arange(len(trmesh.vertices)))
    mesh_graph.add_edges_from(trmesh.edges_unique)

    # squaring the sparse adjacency matrix:
    A = nx.to_scipy_sparse_array(mesh_graph)
    A = scipy.sparse.csr_matrix(A)
    assert scipy.sparse.isspmatrix_csr(A), "adjacency matrix must be sparse matrix not array " \
                                           "so that multiplication works correctly"
    B = A.copy()
    hop_dist_matrix = A.copy()
    # multiply is elementwise, but * must be proper matrix multiplication
    assert (A.multiply(A) - A * A).nnz != 0, "matrix multiplication is not working."
    for i in range(2, max_hops + 1):
        B = B * A  # A ** i  # (A ** i).tolil()
        B[B > 0] = 1
        # only fill in i*B[B > 0] where there is not something yet written:
        hop_dist_matrix = hop_dist_matrix + i * (B - B * (hop_dist_matrix > 0))
    hop_dist_matrix = hop_dist_matrix.tolil()
    hop_dist_matrix.setdiag(0)  # to exclude the node itself from the neighbor list
    hop_dist_matrix = hop_dist_matrix.tocsr()

    # get all pair distances:
    ind_arr1, ind_arr2 = hop_dist_matrix.nonzero()  # non-zero indices
    all_pairs_hop_dist = np.array(hop_dist_matrix[ind_arr1, ind_arr2])[0]
    all_pairs_dist = np.sqrt(np.sum((trmesh.vertices[ind_arr1] - trmesh.vertices[ind_arr2]) ** 2, axis=1))
    search_dist_mask = np.ones(len(ind_arr1)).astype(bool)
    if search_dist is not None:
        search_dist_mask = (all_pairs_dist < search_dist)

        acceptance_ratio = np.sum(search_dist_mask) / len(search_dist_mask)
        if info_acceptance_ratio:
            if acceptance_ratio > 0.9:
                print("acceptance_ratio > 0.9 -> Increase max_hops.")
            elif acceptance_ratio < 0.5:
                print("acceptance_ratio < 0.5 -> Decrease max_hops.")
            else:
                print("ratio is alright=", acceptance_ratio)

    # split the index array according to the following partition:
    partition_index_arrays = [i for i, x in enumerate(ind_arr1) if x != ind_arr1[i - 1]] + [len(ind_arr1)]

    neighbors_dict = {}
    k_arr = np.zeros(mesh_graph.number_of_nodes(), dtype=int)
    for node in range(A.shape[0]):
        start = partition_index_arrays[node]
        stop = partition_index_arrays[node + 1]
        mask_partition = search_dist_mask[start:stop]
        nn_indices = ind_arr2[start:stop][mask_partition]
        nn_beyond_indices = ind_arr2[start:stop][~mask_partition]
        nn_dist = all_pairs_dist[start:stop][mask_partition]
        nn_hop_dist = all_pairs_hop_dist[start:stop][mask_partition]
        k_arr[node] = np.sum(mask_partition)

        if sort_by_dist:
            # sort based on dist:
            ind_sort = np.argsort(nn_dist)
            nn_indices = nn_indices[ind_sort]
            nn_dist = nn_dist[ind_sort]
            nn_hop_dist = nn_hop_dist[ind_sort]

        neighbors_dict[node] = {"neighbor_indices": nn_indices,
                                "neighbor_dist": nn_dist,
                                "neighbor_hop_dist": nn_hop_dist,
                                "neighbor_indices_beyond_search_dist": nn_beyond_indices
                                }

    return neighbors_dict, k_arr


def compute_mesh_neighbors(trmesh, method_specific_kwargs, method="adjacency", minimum_num_neighbors=5):
    """
    Finds neighbors for every vertex in a mesh based on different methods.
    By convention, the point itself is not included in the neighbors. The fastest method is "adjacency".

    Parameters
    ----------
    trmesh : tr.Trimesh
        Mesh to find the neighbors for each points.
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

    Returns
    -------
    (dict[int, dict], np.ndarray)
                Neighbor dictionary: The vertex indices are the keys. Each vertex dictionary has the structure:
                {"neighbor_indices" : list of selected indices,
                 "neighbor_dist": np.ndarray with distance to selceted points},
                Array with number of selected points.
    """
    if method_specific_kwargs is None:
        method_specific_kwargs = {"sort_by_distance": False}

    if method == "adjacency":
        neighbors_dict, k_arr = mesh_adjacency_with_cutoff(trmesh, **method_specific_kwargs)
    elif method == "dijkstra":
        neighbors_dict, k_arr = mesh_dijktra_with_cutoff(trmesh, **method_specific_kwargs)
    elif method == "pcd":
        neighbors_dict, k_arr, _ = mesh_get_neighbors(trmesh, **method_specific_kwargs)
    else:
        print(f"method not implemented: {method}")

    if k_arr.min() < minimum_num_neighbors:
        print(f"Careful. Only very few nearest neighbors ({k_arr.min()} < {minimum_num_neighbors}) found.")
    return neighbors_dict, k_arr
