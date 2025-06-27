import networkx as nx
import numpy as np
import trimesh as tr
import open3d as o3d


def mesh_extract_largest_connected_component(trmesh, verbose=True):
    """
    Extracts the largest connected component of a mesh.

    Unfortunately, I have only seen after I had implemented it that o3d has this implemented, see:
    http://www.open3d.org/docs/release/tutorial/geometry/mesh.html#Connected-components.

    Parameters
    ----------
    trmesh : tr.Trimesh
        Mesh with potentially multiple connected components.
    verbose : bool, optional


    Returns
    -------
    tr.Trimesh
        Largest connected component of input mesh.
    """
    if trmesh.body_count == 1:
        return trmesh

    # turn mesh into nx.Graph()
    mesh_graph = nx.Graph()
    mesh_graph.add_nodes_from(np.arange(len(trmesh.vertices)))
    mesh_graph.add_edges_from(trmesh.edges_unique)

    # get nodes from smaller components:
    disconnected_nodes = []
    size_sorted_components = sorted(nx.connected_components(mesh_graph), key=len, reverse=True)

    if verbose:
        print(f"connected_component_sizes={[len(c) for c in size_sorted_components]}")

    for i, comp in enumerate(size_sorted_components):
        if i == 0:
            # skip largest component:
            continue

        disconnected_nodes += list(comp)
    connected_nodes_indices = [ind for ind in range(len(trmesh.vertices)) if ind not in disconnected_nodes]

    # remove vertices and corresponding faces
    disconnected_faces = []
    for node in disconnected_nodes:
        l = list(set(trmesh.vertex_faces[node, :]))
        if -1 in l:
            l.remove(-1)  # -1 is used as placeholder in vertex_faces
        disconnected_faces += l
    disconnected_faces = set(disconnected_faces)  # to make entries unique
    connected_faces_indices = [ind for ind in range(len(trmesh.faces)) if ind not in disconnected_faces]

    # use indexing trick to rename faces
    faces_mapping_arr = np.zeros(len(trmesh.vertices))
    faces_mapping_arr[connected_nodes_indices] = np.arange(len(connected_nodes_indices))

    # build main component mesh
    main_trmesh = tr.Trimesh(trmesh.vertices[connected_nodes_indices],
                             faces_mapping_arr[trmesh.faces[connected_faces_indices]])

    # sanity checks the number of components:
    assert len(main_trmesh.vertices) == len(size_sorted_components[0]), "something went wrong"
    if main_trmesh.body_count != 1:
        print(f"something went wrong amin trmesh has {main_trmesh.body_count} many components.")

    return main_trmesh, connected_nodes_indices


def convert_trmesh_to_o3d(trmesh):
    """
    Converts a trimesh mesh to an open3d mesh.

    Parameters
    ----------
    trmesh : tr.Trimesh
        Mesh to be converted.

    Returns
    -------
    o3d.geometry.TriangleMesh
        Converted mesh.
    """
    return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(trmesh.vertices),
                                     o3d.utility.Vector3iVector(trmesh.faces))


