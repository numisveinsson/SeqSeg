# Built on top of code from Martin Pfaller

import os
import vtk

import numpy as np
from collections import defaultdict

from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import get_vtk_array_type


def calc_caps(polyData):
    """
    Calculate the center of mass of the caps of a surface mesh
    Parameters
    ----------
    polyData : vtkPolyData
        Surface mesh to calculate the caps from.
    Returns
    -------
    caps_locs : list of tuples
        List of tuples with the coordinates of the caps of the surface mesh.
    """
    # Now extract feature edges
    boundaryEdges = vtk.vtkFeatureEdges()
    boundaryEdges.SetInputData(polyData)
    boundaryEdges.BoundaryEdgesOn()
    boundaryEdges.FeatureEdgesOff()
    boundaryEdges.NonManifoldEdgesOff()
    boundaryEdges.ManifoldEdgesOff()
    boundaryEdges.Update()
    output = boundaryEdges.GetOutput()

    conn = connectivity_all(output)
    data = get_points_cells(conn.GetOutput())
    connects = v2n(conn.GetOutput().GetPointData().GetArray(2))

    caps_locs = []
    for i in range(int(connects.max())+1):

        locs = data[0][connects == i]
        center = np.mean(locs, axis=0)
        caps_locs.append(center)

    return caps_locs


class ClosestPoints:
    """
    Find closest points within a geometry
    """
    def __init__(self, inp):
        if isinstance(inp, str):
            geo = read_geo(inp)
            inp = geo.GetOutput()
        dataset = vtk.vtkPolyData()
        dataset.SetPoints(inp.GetPoints())

        locator = vtk.vtkPointLocator()
        locator.Initialize()
        locator.SetDataSet(dataset)
        locator.BuildLocator()

        self.locator = locator

    def search(self, points, radius=None):
        """
        Get ids of points in geometry closest to input points
        Args:
            points: list of points to be searched
            radius: optional, search radius
        Returns:
            Id list
        """
        ids = []
        for p in points:
            if radius is not None:
                result = vtk.vtkIdList()
                self.locator.FindPointsWithinRadius(radius, p, result)
                ids += [result.GetId(k)
                        for k in range(result.GetNumberOfIds())]
            else:
                ids += [self.locator.FindClosestPoint(p)]
        return ids


def collect_arrays(output):
    res = {}
    for i in range(output.GetNumberOfArrays()):
        name = output.GetArrayName(i)
        data = output.GetArray(i)
        res[name] = v2n(data)
    return res


def get_all_arrays(geo):
    # collect all arrays
    cell_data = collect_arrays(geo.GetCellData())
    point_data = collect_arrays(geo.GetPointData())

    return point_data, cell_data


def read_geo(fname):
    """
    Read geometry from file, chose corresponding vtk reader
    Args:
        fname: vtp surface or vtu volume mesh file

    Returns:
        vtk reader, point data, cell data
    """
    _, ext = os.path.splitext(fname)
    if ext == '.vtp':
        reader = vtk.vtkXMLPolyDataReader()
    elif ext == '.vtu':
        reader = vtk.vtkXMLUnstructuredGridReader()
    else:
        raise ValueError('File extension ' + ext + ' unknown.')
    reader.SetFileName(fname)
    reader.Update()

    return reader


def read_img(fname):
    """
    Read image from file, chose corresponding vtk reader
    Args:
        fname: vti image

    Returns:
        vtk reader
    """
    _, ext = os.path.splitext(fname)
    if ext == '.vti':
        reader = vtk.vtkXMLImageDataReader()
    else:
        raise ValueError('File extension ' + ext + ' unknown.')
    reader.SetFileName(fname)
    reader.Update()

    return reader


def write_img(fname, input):
    """
    Write image to file
    Args:
        fname: file name
    """
    _, ext = os.path.splitext(fname)
    if ext == '.mha':
        writer = vtk.vtkXMLPolyDataWriter()
    elif ext == '.vti':
        writer = vtk.vtkXMLImageDataWriter()
    else:
        raise ValueError('File extension ' + ext + ' unknown.')
    writer.SetFileName(fname)
    writer.SetInputData(input)
    writer.Update()
    writer.Write()


def change_vti_vtk(fname):
    """
    Change image file from vti to vtk
    Args:
        fname: file name
    """
    # Read in the VTI file
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(fname)
    reader.Update()

    # Write out the VTK file
    writer = vtk.vtkDataSetWriter()
    writer.SetFileName(fname.replace('.vti', '.vtk'))
    writer.SetInputConnection(reader.GetOutputPort())
    writer.Write()


def write_geo(fname, input):
    """
    Write geometry to file
    Args:
        fname: file name
    """
    _, ext = os.path.splitext(fname)
    if ext == '.vtp':
        writer = vtk.vtkXMLPolyDataWriter()
    elif ext == '.vtu':
        writer = vtk.vtkXMLUnstructuredGridWriter()
    else:
        raise ValueError('File extension ' + ext + ' unknown.')
    writer.SetFileName(fname)
    writer.SetInputData(input)
    writer.Update()
    writer.Write()


def threshold(inp, t, name):
    """
    Threshold according to cell array
    Args:
        inp: InputConnection
        t: BC_FaceID
        name: name in cell data used for thresholding
    Returns:
        reader, point data
    """
    thresh = vtk.vtkThreshold()
    thresh.SetInputData(inp)
    thresh.SetInputArrayToProcess(0, 0, 0, 1, name)
    thresh.ThresholdBetween(t, t)
    thresh.Update()
    return thresh


def calculator(inp, function, inp_arrays, out_array):
    """
    Function to add vtk calculator
    Args:
        inp: InputConnection
        function: string with function expression
        inp_arrays: list of input point data arrays
        out_array: name of output array
    Returns:
        calc: calculator object
    """
    calc = vtk.vtkArrayCalculator()
    for a in inp_arrays:
        calc.AddVectorArrayName(a)
    calc.SetInputData(inp.GetOutput())
    if hasattr(calc, 'SetAttributeModeToUsePointData'):
        calc.SetAttributeModeToUsePointData()
    else:
        calc.SetAttributeTypeToPointData()
    calc.SetFunction(function)
    calc.SetResultArrayName(out_array)
    calc.Update()
    return calc


def cut_plane(inp, origin, normal):
    """
    Cuts geometry at a plane
    Args:
        inp: InputConnection
        origin: cutting plane origin
        normal: cutting plane normal
    Returns:
        cut: cutter object
    """
    # define cutting plane
    plane = vtk.vtkPlane()
    plane.SetOrigin(origin[0], origin[1], origin[2])
    plane.SetNormal(normal[0], normal[1], normal[2])

    # define cutter
    cut = vtk.vtkCutter()
    cut.SetInputData(inp)
    cut.SetCutFunction(plane)
    cut.Update()
    return cut


def get_points_cells(inp):
    cells = []
    for i in range(inp.GetNumberOfCells()):
        cell_points = []
        for j in range(inp.GetCell(i).GetNumberOfPoints()):
            cell_points += [inp.GetCell(i).GetPointId(j)]
        cells += [cell_points]
    return v2n(inp.GetPoints().GetData()), cells


def connectivity(inp, origin):
    """
    If there are more than one unconnected geometries, extract the closest one
    Args:
        inp: InputConnection
        origin: region closest to this point will be extracted
    Returns:
        con: connectivity object
    """
    con = vtk.vtkConnectivityFilter()
    con.SetInputData(inp)  # .GetOutput())
    con.SetExtractionModeToClosestPointRegion()
    con.SetClosestPoint(origin[0], origin[1], origin[2])
    con.Update()
    return con


def connectivity_all(inp):
    """
    Color regions according to connectivity
    Args:
        inp: InputConnection
    Returns:
        con: connectivity object
    """
    con = vtk.vtkConnectivityFilter()
    con.SetInputData(inp)
    con.SetExtractionModeToAllRegions()
    con.ColorRegionsOn()
    con.Update()
    assert con.GetNumberOfExtractedRegions() > 0, 'empty geometry'
    return con


def extract_surface(inp):
    """
    Extract surface from 3D geometry
    Args:
        inp: InputConnection
    Returns:
        extr: vtkExtractSurface object
    """
    extr = vtk.vtkDataSetSurfaceFilter()
    extr.SetInputData(inp)
    extr.Update()
    return extr.GetOutput()


def clean(inp):
    """
    Merge duplicate Points
    """
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(inp)
    # cleaner.SetTolerance(1.0e-3)
    cleaner.PointMergingOn()
    cleaner.Update()
    return cleaner.GetOutput()


def scalar_array(length, name, fill):
    """
    Create vtkIdTypeArray array with given name and constant value
    """
    ids = vtk.vtkIdTypeArray()
    ids.SetNumberOfValues(length)
    ids.SetName(name)
    ids.Fill(fill)
    return ids


def add_scalars(inp, name, fill):
    """
    Add constant value array to point and cell data
    """
    inp.GetOutput().GetCellData().AddArray(scalar_array(inp.GetOutput()
                                                        .GetNumberOfCells(),
                                                        name, fill))
    inp.GetOutput().GetPointData().AddArray(scalar_array(inp.GetOutput()
                                                         .GetNumberOfPoints(),
                                                         name, fill))


def rename(inp, old, new):
    if inp.GetOutput().GetCellData().HasArray(new):
        inp.GetOutput().GetCellData().RemoveArray(new)
    if inp.GetOutput().GetPointData().HasArray(new):
        inp.GetOutput().GetPointData().RemoveArray(new)
    inp.GetOutput().GetCellData().GetArray(old).SetName(new)
    inp.GetOutput().GetPointData().GetArray(old).SetName(new)


def replace(inp, name, array):
    arr = n2v(array)
    arr.SetName(name)
    inp.GetOutput().GetCellData().RemoveArray(name)
    inp.GetOutput().GetCellData().AddArray(arr)


def geo(inp):
    poly = vtk.vtkGeometryFilter()
    poly.SetInputData(inp)
    poly.Update()
    return poly.GetOutput()


def region_grow(geo, seed_points, seed_ids, n_max=99):
    # initialize output arrays
    array_dist = -1 * np.ones(geo.GetNumberOfPoints(), dtype=int)
    array_ids = -1 * np.ones(geo.GetNumberOfPoints(), dtype=int)
    array_ids[seed_points] = seed_ids

    # initialize ids
    cids_all = set()
    pids_all = set(seed_points.tolist())
    pids_new = set(seed_points.tolist())

    # surf = extract_surface(geo)
    # pids_surf = set(v2n(surf
    #                     .GetPointData().GetArray('GlobalNodeID')).tolist())

    # loop until region stops growing or reaches maximum number of iterations
    i = 0
    while len(pids_new) > 0 and i < n_max:
        # count grow iterations
        i += 1

        # update
        pids_old = pids_new

        # print progress
        print_str = 'Iteration ' + str(i)
        print_str += '\tNew points ' + str(len(pids_old)) + '     '
        print_str += '\tTotal points ' + str(len(pids_all))
        print(print_str)

        # grow region one step
        pids_new = grow(geo, array_ids, pids_old, pids_all, cids_all)

        # convert to array
        pids_old_arr = list(pids_old)

        # create point locator with old wave front
        points = vtk.vtkPoints()
        points.Initialize()
        for i_old in pids_old:
            points.InsertNextPoint(geo.GetPoint(i_old))

        dataset = vtk.vtkPolyData()
        dataset.SetPoints(points)

        locator = vtk.vtkPointLocator()
        locator.Initialize()
        locator.SetDataSet(dataset)
        locator.BuildLocator()

        # find closest point in new wave front
        for i_new in pids_new:
            array_ids[i_new] = array_ids[pids_old_arr[locator
                                                      .FindClosestPoint(
                                                          geo.GetPoint(i_new))
                                                      ]]
            array_dist[i_new] = i

    return array_ids, array_dist + 1


def appendPolyData(poly_list):
    """
    Combine two VTK PolyData objects together
    Args:
        poly_list: list of polydata
    Return:
        poly: combined PolyData
    """
    appendFilter = vtk.vtkAppendPolyData()
    for poly in poly_list:
        appendFilter.AddInputData(poly)
    appendFilter.Update()
    out = appendFilter.GetOutput()
    return out


def grow(geo, array, pids_in, pids_all, cids_all):
    # ids of propagating wave-front
    pids_out = set()

    # loop all points in wave-front
    for pi_old in pids_in:
        cids = vtk.vtkIdList()
        geo.GetPointCells(pi_old, cids)

        # get all connected cells in wave-front
        for j in range(cids.GetNumberOfIds()):
            # get cell id
            ci = cids.GetId(j)

            # skip cells that are already in region
            if ci in cids_all:
                continue
            else:
                cids_all.add(ci)

            pids = vtk.vtkIdList()
            geo.GetCellPoints(ci, pids)

            # loop all points in cell
            for k in range(pids.GetNumberOfIds()):
                # get point id
                pi_new = pids.GetId(k)

                # add point only if it's new
                # and doesn't fullfill stopping criterion
                if array[pi_new] == -1 and pi_new not in pids_in:
                    pids_out.add(pi_new)
                    pids_all.add(pi_new)

    return pids_out


def cell_connectivity(geo):
    """
    Extract the point connectivity from vtk and return a dictionary
    that can be used in meshio
    """
    vtk_to_meshio = {3: 'line', 5: 'triangle', 10: 'tetra'}

    cells = defaultdict(list)
    for i in range(geo.GetNumberOfCells()):
        cell_type_vtk = geo.GetCellType(i)
        if cell_type_vtk in vtk_to_meshio:
            cell_type = vtk_to_meshio[cell_type_vtk]
        else:
            raise ValueError('vtkCellType '
                             + str(cell_type_vtk) + ' not supported')

        points = geo.GetCell(i).GetPointIds()
        point_ids = []
        for j in range(points.GetNumberOfIds()):
            point_ids += [points.GetId(j)]
        cells[cell_type] += [point_ids]

    for t, c in cells.items():
        cells[t] = np.array(c)

    return cells


def get_location_cells(surface):
    """
    Compute centers of cells and return their surface_locations
    Args:
        vtk polydata, e.g. surface
    Returns:
        np.array with centroid surface_locations
    """
    ecCentroidFilter = vtk.vtkCellCenters()
    ecCentroidFilter.VertexCellsOn()
    ecCentroidFilter.SetInputData(surface)
    ecCentroidFilter.Update()
    ecCentroids = ecCentroidFilter.GetOutput()

    surface_locations = v2n(ecCentroids.GetPoints().GetData())
    return surface_locations


def exportSitk2VTK(sitkIm, spacing=None):
    """
    This function creates a vtk image from a simple itk image
    Args:
        sitkIm: simple itk image
    Returns:
        imageData: vtk image
import SimpleITK as sitk
    """
    if not spacing:
        spacing = sitkIm.GetSpacing()
    import SimpleITK as sitk
    import vtk
    img = sitk.GetArrayFromImage(sitkIm).transpose(2, 1, 0)
    vtkArray = exportPython2VTK(img)
    imageData = vtk.vtkImageData()
    imageData.SetDimensions(sitkIm.GetSize())
    imageData.GetPointData().SetScalars(vtkArray)
    imageData.SetOrigin([0., 0., 0.])
    imageData.SetSpacing(spacing)
    matrix = build_transform_matrix(sitkIm)
    space_matrix = np.diag(list(spacing)+[1.])
    matrix = np.matmul(matrix, np.linalg.inv(space_matrix))
    matrix = np.linalg.inv(matrix)
    vtkmatrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vtkmatrix.SetElement(i, j, matrix[i, j])
    reslice = vtk.vtkImageReslice()
    reslice.SetInputData(imageData)
    reslice.SetResliceAxes(vtkmatrix)
    reslice.SetInterpolationModeToNearestNeighbor()
    reslice.Update()
    imageData = reslice.GetOutput()
    # imageData.SetDirectionMatrix(sitkIm.GetDirection())

    return imageData, vtkmatrix


def exportVTK2Sitk(vtkIm):
    """
    This function creates a simple itk image from a vtk image
    Args:
        vtkIm: vtk image
    Returns:
        sitkIm: simple itk image
    """
    import SimpleITK as sitk
    # vtkIm = vtkIm.GetOutput()
    vtkIm.GetPointData().GetScalars().SetName('Scalars_')
    vtkArray = v2n(vtkIm.GetPointData().GetScalars())
    vtkArray = np.reshape(vtkArray, vtkIm.GetDimensions(), order='F')
    vtkArray = np.transpose(vtkArray, (2, 1, 0))
    sitkIm = sitk.GetImageFromArray(vtkArray)
    sitkIm.SetSpacing(vtkIm.GetSpacing())
    sitkIm.SetOrigin(vtkIm.GetOrigin())
    return sitkIm


def build_transform_matrix(image):
    matrix = np.eye(4)
    matrix[:-1, :-1] = np.matmul(np.reshape(image.GetDirection(),
                                            (3, 3)),
                                 np.diag(image.GetSpacing()))
    matrix[:-1, -1] = np.array(image.GetOrigin())
    return matrix


def exportPython2VTK(img):
    """
    This function creates a vtk image from a python array
    Args:
        img: python ndarray of the image
    Returns:
        imageData: vtk image
    """
    vtkArray = n2v(num_array=img.flatten('F'),
                   deep=True,
                   array_type=get_vtk_array_type(img.dtype))
    # vtkArray = n2v(img.flatten())
    return vtkArray


def bound_polydata_by_image(image, poly, threshold):
    """
    Function to cut polydata to be bounded
    by image volume
    """
    bound = vtk.vtkBox()
    image.ComputeBounds()
    b_bound = image.GetBounds()
    b_bound = [b+threshold if (i % 2) == 0 else b-threshold
               for i, b in enumerate(b_bound)]
    # print("Bounding box: ", b_bound)
    bound.SetBounds(b_bound)
    clipper = vtk.vtkClipPolyData()
    clipper.SetClipFunction(bound)
    clipper.SetInputData(poly)
    clipper.InsideOutOn()
    clipper.Update()
    return clipper.GetOutput()


def write_vtk_polydata(poly, fn):
    """
    This function writes a vtk polydata to disk
    Args:
        poly: vtk polydata
        fn: file name
    Returns:
        None
    """
    # print('Writing vtp with name:', fn)
    if (fn == ''):
        return 0

    _, extension = os.path.splitext(fn)

    if extension == '.vtk':
        writer = vtk.vtkPolyDataWriter()
    elif extension == '.vtp':
        writer = vtk.vtkXMLPolyDataWriter()
    else:
        raise ValueError("Incorrect extension"+extension)
    writer.SetInputData(poly)
    writer.SetFileName(fn)
    writer.Update()
    writer.Write()
    return


def vtk_marching_cube(vtkLabel, bg_id, seg_id, smooth=None):
    """
    Use the VTK marching cube to create isosrufaces
    for all classes excluding the background
    Args:
        labels: vtk image contraining the label map
        bg_id: id number of background class
        smooth: smoothing iteration
    Returns:
        mesh: vtk PolyData of the surface mesh
    """
    contour = vtk.vtkMarchingCubes()
    contour.SetInputData(vtkLabel)
    contour.SetValue(0, seg_id)
    contour.Update()
    mesh = contour.GetOutput()

    return mesh


def voi_contain_caps(voi_min, voi_max, caps_locations):
    """
    See if model caps are enclosed in volume
    Args:
        voi_min: min bounding values of volume
        voi_max: max bounding values of volume
    Returns:
        contain: boolean if a cap point was found within volume
    """
    larger = caps_locations > voi_min
    smaller = caps_locations < voi_max

    contain = np.any(np.logical_and(smaller.all(axis=1), larger.all(axis=1)))
    return contain


def points2polydata(xyz, attribute_float=None):
    import vtk
    points = vtk.vtkPoints()
    # Create the topology of the point (a vertex)
    vertices = vtk.vtkCellArray()
    if attribute_float is not None:
        # Create a vtkFloatArray to hold the attribute values
        attr = vtk.vtkFloatArray()
        attr.SetName("Radius")
        attr.SetNumberOfComponents(1)
        attr.SetNumberOfTuples(len(xyz))
        for i, val in enumerate(attribute_float):
            attr.SetValue(i, val)
    # Add points
    for i in range(0, len(xyz)):
        try:
            p = xyz.loc[i].values.tolist()
        except AttributeError:
            p = xyz[i]

        point_id = points.InsertNextPoint(p)
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(point_id)
    # Create a poly data object
    polydata = vtk.vtkPolyData()
    # Set the points and vertices we created as the geometry
    # and topology of the polydata
    polydata.SetPoints(points)
    polydata.SetVerts(vertices)
    # If we have an attribute, add it to the polydata
    if attribute_float is not None:
        polydata.GetPointData().AddArray(attr)
        polydata.GetPointData().SetActiveScalars("Radius")
    polydata.Modified()

    return polydata


def smooth_polydata(poly, iteration=25, boundary=False,
                    feature=False, smoothingFactor=0.):  # .1
    """
    This function smooths a vtk polydata
    Original settings: 25, False, False, 0.
    Args:
        poly: vtk polydata to smooth
        boundary: boundary smooth bool
    Returns:
        smoothed: smoothed vtk polydata
    """
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(poly)
    smoother.SetPassBand(pow(10., -4. * smoothingFactor))  # -3
    smoother.SetBoundarySmoothing(boundary)
    smoother.SetFeatureEdgeSmoothing(feature)
    smoother.SetNumberOfIterations(iteration)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()

    smoothed = smoother.GetOutput()

    return smoothed


def decimation(poly, rate):
    """
    Simplifies a VTK PolyData
    Args:
        poly: vtk PolyData
        rate: target rate reduction
    """
    decimate = vtk.vtkQuadricDecimation()
    decimate.SetInputData(poly)
    decimate.AttributeErrorMetricOn()
    decimate.ScalarsAttributeOn()
    decimate.SetTargetReduction(rate)
    decimate.VolumePreservationOff()
    decimate.Update()
    output = decimate.GetOutput()
    # output = cleanPolyData(output, 0.)
    return output


def vtkImageResample(image, spacing, opt):
    """
    Resamples the vtk image to the given dimenstion
    Args:
        image: vtk Image data
        spacing: image new spacing
        opt: interpolation option: linear, NN, cubic
    Returns:
        image: resampled vtk image data
    """
    reslicer = vtk.vtkImageReslice()
    reslicer.SetInputData(image)
    if opt == 'linear':
        reslicer.SetInterpolationModeToLinear()
    elif opt == 'NN':
        reslicer.SetInterpolationModeToNearestNeighbor()
    elif opt == 'cubic':
        reslicer.SetInterpolationModeToCubic()
    else:
        raise ValueError("interpolation option not recognized")

    # size = np.array(image.GetSpacing())*np.array(image.GetDimensions())
    # new_spacing = size/np.array(dims)

    reslicer.SetOutputSpacing(*spacing)
    reslicer.Update()

    return reslicer.GetOutput()


def evaluate_surface(ref_im, value=1):

    ref_im, M = exportSitk2VTK(ref_im)
    r_s = vtk_marching_cube(ref_im, 0, value)

    return r_s


def smooth_surface(polydata, smoothingIterations):

    if smoothingIterations == 0:
        return polydata

    passBand = 0.01  # 0.001
    # featureAngle = 120.0
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(polydata)
    smoother.SetNumberOfIterations(smoothingIterations)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    # smoother.SetFeatureAngle(featureAngle)
    smoother.SetPassBand(passBand)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()

    return smoother.GetOutput()


def get_largest_connected_polydata(poly):

    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(poly)
    connectivity.SetExtractionModeToLargestRegion()
    connectivity.Update()
    poly = connectivity.GetOutput()

    return poly


def is_point_in_surface(surface, point):
    """
    Function to check if point is enclosed
    by vtk polydata surface
    """

    pnt = vtk.vtkPoints()
    pnt.InsertNextPoint(point)
    pd = vtk.vtkPolyData()
    pd.SetPoints(pnt)
    enclosed = vtk.vtkSelectEnclosedPoints()
    enclosed.SetSurfaceData(surface)
    enclosed.CheckSurfaceOn()
    enclosed.SetInputData(pd)
    enclosed.Update()

    is_inside = enclosed.IsInside(0)
    # print(str(is_inside))

    if is_inside == 1:
        return True
    elif is_inside == 0:
        return False
    else:
        print("Error, output from VTK enclosed surface")
        return 0


def process_cardiac_mesh(mesh_file, scale=1):

    mesh = read_geo(mesh_file).GetOutput()

    if mesh.GetPointData().GetNormals() is None:
        # calculate normals
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(mesh)
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOff()
        normals.Update()
        mesh = normals.GetOutput()

    mesh_data = collect_arrays(mesh.GetPointData())
    # point locations as numpy array
    point_loc = v2n(mesh.GetPoints().GetData())
    region_id = mesh_data['RegionId']  # Region ID as numpy array
    # get ids of points in region 8
    region_8 = np.where(region_id == 8)[0]
    region_3 = np.where(region_id == 3)[0]
    # get coordinates of points in region 8 and 3
    region_8_coords = point_loc[region_8, :] * scale
    region_3_coords = point_loc[region_3, :] * scale
    # calculate the center of mass of region 8
    region_8_center = np.mean(region_8_coords, axis=0)
    region_3_center = np.mean(region_3_coords, axis=0)
    # calculate the normal vector of region 8
    region_8_normal = np.mean(mesh_data['Normals'][region_8, :], axis=0)
    # check if the normal vector is pointing
    # towards the center of mass of region 3
    if np.dot(region_8_normal, region_3_center - region_8_center) > 0:
        region_8_normal = -region_8_normal

    return region_8_center, region_8_normal, region_3_center


def write_normals_centers(mesh_dir, region_8_center,
                          region_8_normal, region_3_center):

    # write the normal vector as vtk starting
    # from the center of mass of region 8

    # to the center of mass of region 3
    # create a vtkPolyData object
    line = vtk.vtkPolyData()
    # create a vtkPoints object and add the points to it
    points = vtk.vtkPoints()
    points.InsertNextPoint(region_8_center)
    points.InsertNextPoint(region_8_center + 3*region_8_normal)
    points.InsertNextPoint(region_3_center)
    # add the points to the vtkPolyData object
    line.SetPoints(points)
    # create a vtkCellArray object and add the cells to it
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(2)
    cells.InsertCellPoint(0)
    cells.InsertCellPoint(1)
    cells.InsertNextCell(2)
    cells.InsertCellPoint(1)
    cells.InsertCellPoint(2)
    # add the cells to the vtkPolyData object
    line.SetLines(cells)
    # write the vtkPolyData object to file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(os.path.join(mesh_dir, 'normal.vtp'))
    writer.SetInputData(line)
    writer.Write()


def convertPolyDataToImageData(poly, ref_im):
    """
    Convert the vtk polydata to imagedata
    Args:
        poly: vtkPolyData
        ref_im: reference vtkImage to match the polydata with
    Returns:
        output: resulted vtkImageData
    """

    ref_im.GetPointData().SetScalars(
        n2v(np.zeros(v2n(ref_im.GetPointData().GetScalars()).shape)))
    ply2im = vtk.vtkPolyDataToImageStencil()
    ply2im.SetTolerance(0.05)
    ply2im.SetInputData(poly)
    ply2im.SetOutputSpacing(ref_im.GetSpacing())
    ply2im.SetInformationInput(ref_im)
    ply2im.Update()

    stencil = vtk.vtkImageStencil()
    stencil.SetInputData(ref_im)
    stencil.ReverseStencilOn()
    stencil.SetStencilData(ply2im.GetOutput())
    stencil.Update()
    output = stencil.GetOutput()

    return output