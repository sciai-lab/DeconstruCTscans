import copy
import trimesh as tr

import numpy as np
import open3d as o3d

from deconstruct.utils.visualization.colorize_geometries import get_color_for_o3d_geometry
from deconstruct.utils.geometries.vtk_utils import vtk_decimate_trmesh
from deconstruct.utils.geometries.mesh_utils import convert_trmesh_to_o3d
from deconstruct.utils.open3d_utils.visualization import VisualizeMesh


def export_mesh_for_vgmax(mesh, path_export_for_vgmax, voxelization_scale=1, flip_zyx=True):
    mesh = copy.deepcopy(mesh)
    # flip xzy:
    vertices = np.asarray(mesh.vertices)
    if flip_zyx:
        vertices = np.flip(vertices, axis=1)
    vertices *= voxelization_scale
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d.io.write_triangle_mesh(path_export_for_vgmax, mesh)
    print(f"Exported mesh to {path_export_for_vgmax}.")


def assemble_meshes_in_scene(meshes, trafos=None, colors=None, names=None, combine_all=False, alpha=1.,
                             light_reflections=True, flip_xyz=True, **export_vgmax_kwargs):
    if combine_all and names is not None:
        print("Warning: combine_all=True and names is not None. Names will be ignored.")

    if trafos is None:
        trafos = [np.eye(4) for _ in range(len(meshes))]
    if names is None:
        names = [f"mesh_{i}" for i in range(len(meshes))]

    assert len(meshes) == len(trafos), (f"Number of meshes and transformations must be equal: "
                                        f"{len(meshes)} != {len(trafos)}.")
    assert len(meshes) == len(names), f"Number of meshes and names must be equal: {len(meshes)} != {len(names)}."
    if colors is not None:
        assert len(meshes) == len(colors), f"Number of meshes and colors must be equal: {len(meshes)} != {len(colors)}."

    combined_geom = o3d.geometry.TriangleMesh()
    visualize_obj_list = []

    for i, mesh_ in enumerate(meshes):
        mesh = copy.deepcopy(mesh_)
        if isinstance(mesh, o3d.geometry.TriangleMesh):
            pass
        elif isinstance(mesh, tr.Trimesh):
            mesh = convert_trmesh_to_o3d(mesh)
        else:
            raise NotImplementedError(f"mesh type {type(mesh)} not implemented.")

        mesh.transform(trafos[i])
        if colors is not None:
            mesh.paint_uniform_color(colors[i])
        else:
            mesh.paint_uniform_color(get_color_for_o3d_geometry("random"))

        combined_geom += mesh
        if not combine_all:
            visualize_obj_list.append(VisualizeMesh(geometry=mesh, name=names[i],
                                                    is_visible=True, show_wire_frame=False,
                                                    light_reflections=light_reflections,
                                                    flip_xyz=flip_xyz))

    if combine_all:
        visualize_obj_list = [VisualizeMesh(geometry=combined_geom, name=f"assembly",
                                            is_visible=True, show_wire_frame=False, alpha=alpha,
                                            light_reflections=light_reflections, flip_xyz=flip_xyz)]

    if "path_export_for_vgmax" in export_vgmax_kwargs:
        export_mesh_for_vgmax(combined_geom, **export_vgmax_kwargs)

    return visualize_obj_list


def assemble_part_catalog_meshes_in_scene(part_catalog, choose_colors="by_part_name",
                                          combine_all=True, alpha=1., light_reflections=True,
                                          use_original_nonsmoothed_catalog=True, flip_xyz=True,
                                          comine_all_name="part_catalog_assembly"):
    """
    Assembles all smoothed meshes of the catalog parts in their ground truth pose in 3d scene.

    Parameters
    ----------
    catalog_parts_dict : dict[str, CatalogPart]
        Dict of all parts in catalog. Key = part name.
    choose_colors : str, optional
        Specifies colormap. Choose from {"random", "weak_grays", "by_material", "by_label"}. Default = "random" RGB.
    suppress_viewer : bool, optional
        To not show viewer but only return visualizer list, default = False.
    combine_all : bool, optional
        To combine all meshes into a single geometry. Default = True.
    alpha : float, optional
        Transparency in visualizer between [0,1]. Default = 1.
    light_reflections : bool, optional
        Specifies if light reflections are switched on. Default = true.
    use_original_nonsmoothed_catalog : bool, optional
        Use original catalog. Default=True.

    Returns
    -------
    list[VisualizeObject]
        list which, if passed to wrapped_o3d_draw(visualize_obj_list), shows the geometries in scene.
    """

    meshes = []
    names = []
    trafos = []
    colors = []
    for k, part_mesh in enumerate(part_catalog.catalog_part_meshes.values()):
        part_color = get_color_for_o3d_geometry(choose_colors, part_mesh.part_material, part_mesh.part_name)
        for i, trafo in enumerate(part_mesh.list_trafo_from_canonical_to_scene):
            if use_original_nonsmoothed_catalog:
                meshes.append(part_mesh.trmesh_in_canonical)
            else:
                meshes.append(part_mesh.mesh_from_volume.smooth_trmesh)
            names.append(f"{part_mesh.part_name}_{i}")
            trafos.append(trafo)
            colors.append(part_color)

    visualize_obj_list = assemble_meshes_in_scene(meshes,
                                                  trafos=trafos, colors=colors, names=names,
                                                  combine_all=combine_all, alpha=alpha,
                                                  light_reflections=light_reflections, flip_xyz=flip_xyz)

    if combine_all:
        visualize_obj_list[0].name = comine_all_name

    return visualize_obj_list


def assemble_segmentation_meshes_in_scene(segmentation_catalog,
                                          combine_all=True, alpha=1., light_reflections=True, flip_xyz=True,
                                          assemble_catalog_kwargs=None,
                                          combine_all_name="segmentation_catalog_assembly",
                                          decimation_factor=0.3):
    """
    Assembles all smoothed meshes of segmentation in their ground truth pose in 3d scene.

    Parameters
    ----------
    segmentation_parts_dict : dict[str, SegmentationPart]
        Dict of all parts in catalog. Key = part name.
    choose_colors : str, optional
        Specifies colormap. Choose from {"random", "weak_grays"}. Default = "random" RGB.
    suppress_viewer : bool, optional
        To not show viewer but only return visualizer list, default = False.
    combine_all : bool, optional
        To combine all meshes into a single geometry. Default = True.
    alpha : float, optional
        Transparency in visualizer between [0,1]. Default = 1.
    light_reflections : bool, optional
        Specifies if light reflections are switched on. Default = true.
    assemble_catalog_kwargs : dict, optional
        Inputs of assemble_gt_catalog_meshes_in_scene. Default = None -> no underlying gt is shown.

    Returns
    -------
    list[VisualizeObject]
        list which, if passed to wrapped_o3d_draw(visualize_obj_list), shows the geometries in scene.
    """

    meshes = []
    names = []
    for mesh_from_volume in segmentation_catalog.catalog_segmentation_meshes.values():
        smooth_trmesh = mesh_from_volume.smooth_trmesh.copy()
        smooth_trmesh = vtk_decimate_trmesh(smooth_trmesh, reduction=decimation_factor)
        meshes.append(smooth_trmesh)
        names.append(f"segm_{mesh_from_volume.label}")

    visualize_obj_list = assemble_meshes_in_scene(meshes,
                                                  names=names,
                                                  combine_all=combine_all, alpha=alpha,
                                                  light_reflections=light_reflections, flip_xyz=flip_xyz)

    if combine_all:
        visualize_obj_list[0].name = combine_all_name

    if assemble_catalog_kwargs is not None:
        assemble_catalog_default_kwargs = {
            "suppress_viewer": True,
            "alpha": 0.5,
            "choose_colors": "by_material"
        }
        assemble_catalog_default_kwargs.update(assemble_catalog_kwargs)
        visualize_obj_list += assemble_part_catalog_meshes_in_scene(**assemble_catalog_default_kwargs)

    return visualize_obj_list