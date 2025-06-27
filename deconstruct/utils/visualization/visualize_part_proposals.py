from deconstruct.utils.open3d_utils.visualization import view_pcds
import numpy as np
import copy

from deconstruct.utils.visualization.remote_visualization import save_visualization_obj_list_to_index


def shift_pcds_to_show_side_by_side(pcd1, pcd2, shift_dim=0, center_other_dim=True):
    """
    shift pcd2 to the right of pcd1, pcd1 stays where it is.
    """

    pcd2 = copy.deepcopy(pcd2)
    bbox1 = pcd1.get_axis_aligned_bounding_box()
    bbox2 = pcd2.get_axis_aligned_bounding_box()

    # shift pcd2 to the right of pcd1:
    shift_arr = np.zeros(3)
    shift_arr[shift_dim] = bbox1.max_bound[shift_dim] - bbox2.min_bound[shift_dim]
    pcd2.translate(shift_arr)

    # center pcd2 in other dimensions:
    if center_other_dim:
        for dim in range(3):
            if dim != shift_dim:
                shift_arr = np.zeros(3)
                shift_arr[dim] = ((bbox1.max_bound[dim] - bbox1.min_bound[dim]) / 2 -
                                  (bbox2.max_bound[dim] - bbox2.min_bound[dim]) / 2)
                pcd2.translate(shift_arr)

    return pcd1, pcd2


def visualize_part_proposal(part_proposal, part_catalog, segmentation_catalog):

    # get the point clouds from the part catalog and segmentation catalog:
    src_pcd = part_catalog.catalog_part_meshes[part_proposal.part_name].mesh_from_volume.pcd_with_properties.pcd
    src_pcd = copy.deepcopy(src_pcd)
    dst_pcd = segmentation_catalog.catalog_segmentation_meshes[part_proposal.label].pcd_with_properties.pcd

    # get the registration result:
    registration_result = part_proposal.registration_result
    registration_trafo = registration_result.transformation
    print("registration_score:", registration_result.registration_score)

    # show both pointclouds side by side, and then the registration result:
    # shift src_pcd to show them side by side:
    # dst_pcd, src_pcd_on_side = shift_pcds_to_show_side_by_side(dst_pcd, src_pcd, shift_dim=0, center_other_dim=True)

    src_pcd.transform(registration_trafo)
    src_pcd.paint_uniform_color([0, 0, 1])

    # additionally show bboxes:
    # bbox_on_side = src_pcd.get_axis_aligned_bounding_box()
    # bbox_dst = dst_pcd.get_axis_aligned_bounding_box()

    list_visualization_objs = view_pcds([src_pcd, dst_pcd],
                                        list_names=["src_pcd", "dst_pcd"],
                                        suppress_viewer=True, show_normals=False)

    return list_visualization_objs


