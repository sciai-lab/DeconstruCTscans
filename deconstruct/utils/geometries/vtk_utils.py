import open3d as o3d
import trimesh as tr
import numpy as np
from vtk import vtkPolyData, vtkCellArray, vtkPoints, vtkPolygon
from vtk import vtkPLYWriter, vtkDecimatePro, vtkSmoothPolyDataFilter, vtkPolyDataNormals
from vtk.util.numpy_support import vtk_to_numpy

"""
adapted from https://github.com/hci-unihd/plant-seg-tools/blob/main/plantsegtools/meshes/vtkutils.py
"""


def vtk_to_trmesh(vtk_poly):
    return tr.Trimesh(vertices=vtk_to_numpy(vtk_poly.GetPoints().GetData()),
                      faces=vtk_to_numpy(vtk_poly.GetPolys().GetData()).reshape((-1, 4))[:, 1:])


def ndarray2vtkMesh(inVertexArray, inFacesArray):
    ''' Code inspired by https://github.com/selaux/numpy2vtk '''
    # Handle the points & vertices:
    z_index = 0
    vtk_points = vtkPoints()
    for p in inVertexArray:
        z_value = p[2] if inVertexArray.shape[1] == 3 else z_index
        vtk_points.InsertNextPoint([p[0], p[1], z_value])
    number_of_points = vtk_points.GetNumberOfPoints()

    indices = np.array(range(number_of_points), dtype=int)
    vtk_vertices = vtkCellArray()
    for v in indices:
        vtk_vertices.InsertNextCell(1)
        vtk_vertices.InsertCellPoint(v)

    # Handle faces
    number_of_polygons = inFacesArray.shape[0]
    poly_shape = inFacesArray.shape[1]
    vtk_polygons = vtkCellArray()
    for j in range(0, number_of_polygons):
        polygon = vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(poly_shape)
        for i in range(0, poly_shape):
            polygon.GetPointIds().SetId(i, inFacesArray[j, i])
        vtk_polygons.InsertNextCell(polygon)

    # Assemble the vtkPolyData from the points, vertices and faces
    poly_data = vtkPolyData()
    poly_data.SetPoints(vtk_points)
    poly_data.SetVerts(vtk_vertices)
    poly_data.SetPolys(vtk_polygons)

    return poly_data


def smooth(vtkPoly, iterations=10, relaxation=0.5, edgesmoothing=True):
    # Smooth mesh with Laplacian Smoothing
    smooth = vtkSmoothPolyDataFilter()
    smooth.SetInputData(vtkPoly)
    smooth.SetRelaxationFactor(relaxation)
    smooth.SetNumberOfIterations(iterations)
    if edgesmoothing:
        smooth.FeatureEdgeSmoothingOn()
    else:
        smooth.FeatureEdgeSmoothingOff()
    smooth.BoundarySmoothingOn()
    smooth.Update()

    smoothPoly = vtkPolyData()
    smoothPoly.ShallowCopy(smooth.GetOutput())

    # Find mesh normals (Not sure why)
    normal = vtkPolyDataNormals()
    normal.SetInputData(smoothPoly)
    normal.ComputePointNormalsOn()
    normal.ComputeCellNormalsOn()
    normal.Update()

    normalPoly = vtkPolyData()
    normalPoly.ShallowCopy(normal.GetOutput())

    return normalPoly


def vtk_laplacian_smoothing_trmesh(trmesh, iterations=10, relaxation=0.5, edgesmoothing=True):
    poly_data = ndarray2vtkMesh(inVertexArray=np.asarray(trmesh.vertices), inFacesArray=np.asarray(trmesh.faces))
    poly_data_smoothed = smooth(poly_data, iterations=iterations, relaxation=relaxation, edgesmoothing=edgesmoothing)
    return vtk_to_trmesh(poly_data_smoothed)


def decimation(vtkPoly, reduction):
    # see documentation here: https://www.geologie.uni-freiburg.de/root/manuals/vtkman/vtkDecimatePro.html#method5
    # official docu: https://kitware.github.io/vtk-examples/site/Python/Meshes/Decimation/

    # decimate and copy data
    decimate = vtkDecimatePro()
    decimate.SetInputData(vtkPoly)
    decimate.SetTargetReduction(reduction)  # (float) set 0 for no reduction and 1 for 100% reduction
    # decimate.SetMaximumError(20000.)
    # decimate.PreserveTopologyOff()
    # decimate.BoundaryVertexDeletionOn()
    # decimate.SplittingOn()
    # decimate.SetPreserveTopology(0)
    # decimate.SetBoundaryVertexDeletion(1)
    # decimate.SetSplitting(1)
    # decimate.SetSplitAngle(75)
    # decimate.SetInflectionPointRatio(10)
    # decimate.SetDegree(25)
    # decimate.SetAccumulateError(0)
    # decimate.SetFeatureAngle(15)
    decimate.Update()

    decimatedPoly = vtkPolyData()
    decimatedPoly.ShallowCopy(decimate.GetOutput())

    return decimatedPoly


def vtk_decimate_trmesh(trmesh, reduction):
    poly_data = ndarray2vtkMesh(inVertexArray=np.asarray(trmesh.vertices), inFacesArray=np.asarray(trmesh.faces))
    poly_data_decimated = decimation(poly_data, reduction)
    return vtk_to_trmesh(poly_data_decimated)





