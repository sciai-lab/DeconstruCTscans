import os.path
import open3d as o3d

from deconstruct.utils.open3d_utils.o3d_draw import wrapped_o3d_draw, VisualizePoints, VisualizeMesh


def view_pcds(list_pcds, list_names=None, save_path=None, suppress_viewer=True, wrapped_o3d_draw_kwargs=None,
              **visualize_points_kwargs):
    """
    Function to view a list of point clouds together using o3d.visualization.draw.

    Parameters
    ----------
    list_pcds : o3d.geometry.PointCloud or list[o3d.geometry.PointCloud]
        List of pcds to visualize.
    list_names : list[str], optional
        Names as shown in viewer. Default : [f"unnamed_{i}" for i,_ in enumerate(list_pcds)]
    save_path : str, optional
        Path to export first pcd in list. Default = None -> no export.
    suppress_viewer : bool, optional
        Specify if viewer should be shown. For export only. Default = False.
    wrapped_o3d_draw_kwargs : dict, optional
        Arguments that wrapped_o3d_draw takes.
        Amongst others also:
        draw_param_dict : dict, optional
            Arguments that o3d.visualization.draw takes:
            http://www.open3d.org/docs/release/python_api/open3d.visualization.draw.html.
            Default = {"width": 2560, "height": 1440, "show_skybox": False, "show_ui": False, "raw_mode": False}.
    visualize_points_kwargs : dict, optional
        point_size : float, optional
            Specifies point size. Note this is a global property of the viewer. The maximum is chosen.
        show_normals : bool, optional
            Specifies if normals are toggled on or off (if normals are available).
        normal_width : float, optional
            Line width of normals.
        normal_length : float, optional
            Length of normals.
        normal_color : , optional
            Color of normals.
        is_visible : bool, optional
            Specify if geometry should be toggled on or off in viewer. Default = True.

    Returns
    -------
    list[VisualizeObject]
        list of objects visualized.
    """
    if isinstance(list_pcds, o3d.geometry.PointCloud):
        list_pcds = [list_pcds]

    list_names = [None, ] * len(list_pcds) if list_names is None else list_names
    assert list_names is None or len(list_names) == len(list_pcds), "names don't match pcds"

    visualize_obj_list = []
    for i, pcd in enumerate(list_pcds):
        visualize_obj_list.append(VisualizePoints(geometry=pcd, name=list_names[i], **visualize_points_kwargs))

    if not suppress_viewer:
        wrapped_o3d_draw_kwargs = {} if wrapped_o3d_draw_kwargs is None else wrapped_o3d_draw_kwargs
        wrapped_o3d_draw(visualize_obj_list, **wrapped_o3d_draw_kwargs)

    if save_path is not None:
        # assumes that only one pcd is given as input:
        if len(list_pcds) > 1:
            print(f"careful. only the first of the list_pcds is saved to {os.path.abspath(save_path)}.")
        o3d.io.write_point_cloud(save_path, list_pcds[0])

    return visualize_obj_list


def view_meshes(list_meshes, list_names=None, save_path=None,
                suppress_viewer=True, draw_param_dict=None, **visualize_mesh_kwargs):
    """
    Function to view a list of meshes together using o3d.visualization.draw.

    Parameters
    ----------
    list_meshes : o3d.geometry.TriangleMesh or list[o3d.geometry.TriangleMesh]
        List of meshes to visualize.
    list_names : list[str], optional
        Names as shown in viewer. Default : [f"unnamed_{i}" for i,_ in enumerate(list_trmeshes)]
    save_path : str, optional
        Path to export first pcd in list. Default = None -> no export.
    suppress_viewer : bool, optional
        Specify if viewer should be shown. For export only. Default = False.
    draw_param_dict : dict, optional
        Arguments that o3d.visualization.draw takes:
        http://www.open3d.org/docs/release/python_api/open3d.visualization.draw.html.
        Default = {"width": 2560, "height": 1440, "show_skybox": False, "show_ui": False, "raw_mode": False}.
    visualize_mesh_kwargs
        show_wire_frame : bool, optional
            Specifies if mesh wire frame is toggled on or off. Default = True.
        wire_frame_width : float, optional
            Line width of wire frame. Default = 5
        wire_frame_color : tuple or np.ndarray or str, optional
            Color of wire frame. Default =  np.zeros(3)
        light_reflections : bool, optional
            Specify if light reflections are present. Default = True.
        is_visible : bool, optional
            Specify if geometry should be toggled on or off in viewer. Default = True.
    Returns
    -------

    """
    if isinstance(list_meshes, o3d.geometry.TriangleMesh):
        list_meshes = [list_meshes]

    assert list_names is None or len(list_names) == len(list_meshes), "names don't match meshes"
    list_names = [None, ] * len(list_meshes) if list_names is None else list_names

    visualize_obj_list = []
    for i, mesh in enumerate(list_meshes):
        visualize_obj_list.append(VisualizeMesh(geometry=mesh, name=list_names[i], **visualize_mesh_kwargs))

    if not suppress_viewer:
        wrapped_o3d_draw(visualize_obj_list, draw_param_dict=draw_param_dict)
        # this was the old way:
        # my_o3d_geometry_viewer(geometries=list_o3d_meshes, mesh_show_wireframe=mesh_show_wireframe)

    if save_path is not None:
        # assumes that only one mesh is given as input:
        if len(list_meshes) > 1:
            print("careful. only the first of the list_trmeshes is stored.")
        o3d.io.write_triangle_mesh(save_path, list_meshes[0])

    return visualize_obj_list