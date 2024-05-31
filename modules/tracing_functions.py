import sys
import numpy as np
from .vtk_functions import (collect_arrays, get_points_cells, exportSitk2VTK,
                            vtkImageResample, vtk_marching_cube,
                            appendPolyData, bound_polydata_by_image, read_geo)
from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n


sys.stdout.flush()


def get_smoothing_params(radius, scale_unit, mega_sub = False, already_seg = False):

    if not mega_sub:
        if not already_seg:
            num_iterations = 6
            if radius > 1 * scale_unit: num_iterations = 36
            elif radius > 0.5 * scale_unit:
                print("Small radius; less smoothing")
                num_iterations = 24
        else:
            num_iterations = 8
            if radius > 1 * scale_unit: num_iterations = 8
            elif radius > 0.5 * scale_unit:
                print("Small radius; less smoothing")
                num_iterations = 8
    else:
        num_iterations = 3
        # less smoothing for bigger volumes
        if radius > 1 * scale_unit: num_iterations = 7
        elif radius > 0.5 * scale_unit:
            print("Small radius; less smoothing")
            num_iterations = 5

    return num_iterations


class SkipThisStepError(Exception):
    pass


def get_next_points(centerline_poly, current_point, old_point, old_radius, post_proc = False, magn_radius = 1, min_radius = 0, mega_sub = False):
    """
    Get the next point along the centerline
    Args:
        centerline_poly: vtk polydata of centerline
        current_point: current location
        old_point: previous location
        old_radius: previous radius
    Returns:
        next_point: next location
        next_radius: next radius
    """
    if not mega_sub:    angle_allow = 135
    else:               angle_allow = 165 # allow more for bigger

    point_ids_list = get_point_ids(centerline_poly, post_proc = post_proc)

    #print('Calculating next steps')
    cent_data = collect_arrays(centerline_poly.GetPointData())           # point locations as numpy array
    radii = cent_data['MaximumInscribedSphereRadius']   # Max Inscribed Sphere Radius as numpy array
    points_in_cells = get_points_cells(centerline_poly)

    old_vector = current_point-old_point
    old_vector = old_vector/np.linalg.norm(old_vector)

    points = []
    vessel_r = []
    angles = []
    for ip in range(len(point_ids_list)):

        point_ids = point_ids_list[ip]

        locs = points_in_cells[0][point_ids] ## Locations of the points on line
        rads = radii[point_ids]
        if len(locs) < 4:
            print("ERROR: too few points for ip: " +str(ip))
            continue

        if not mega_sub:
            id_along_cent = len(locs) -1 #*9//10 used to not want to use last point for vector
            id_along_cent_save = len(locs)*3//4
        else:
            id_along_cent = len(locs) -1 #*9//10 used to not want to use last point for vector
            id_along_cent_save = len(locs) -1 # we have more trust in the end

        vector = (locs[id_along_cent]-current_point)/np.linalg.norm(locs[id_along_cent]-current_point)

        if np.dot(old_vector, vector) < 0:
            print(f"dot product between vectors is negative")
        #     print("Flipping for ip: " +str(ip))
        #     locs = np.flip(locs, 0)
        #     rads = np.flip(rads)
        #     vector = (locs[id_along_cent]-current_point)/np.linalg.norm(locs[id_along_cent]-current_point)

        angle = 360/(2*np.pi)*np.arccos(np.dot(old_vector, vector))
        print("The angle is: " + str(angle))

        if angle < angle_allow:
            point_to_check = locs[-1] + 3*rads[id_along_cent_save] * vector
            #pfn = '/Users/numisveinsson/Downloads/point.vtp'
            #polydata_point = points2polydata([point_to_check.tolist()])
            #write_geo(pfn, polydata_point)

            #if not next((True for elem in points if np.array_equal(elem, locs[id_along_cent])), False):
            #print("Saving for ip: " +str(ip))
            
            radius_to_save = rads[id_along_cent_save]

            if old_radius > radius_to_save:
                radius_to_save = (1/2*radius_to_save + 1/2*old_radius) ## Have old radius carry into new
            if radius_to_save < min_radius:
                print(f"Radius too small, saving mininum radius")
                radius_to_save = min_radius
            vessel_r.append( radius_to_save)
            angles.append(angle)

            if np.linalg.norm(current_point - locs[id_along_cent_save]) > 1/4*rads[id_along_cent_save] :
                points.append(  locs[id_along_cent_save]) # current_point + rads[id_along_cent_save] * vector) #
            else:

                print("\nERROR: Point is too close to old point so adding vector\n")
                points.append(  current_point + 1/2*rads[id_along_cent_save] * vector)
                # polydata_point = points2polydata([current_point.tolist(), locs[id_along_cent].tolist()])
                # pfn = '/Users/numisveinsson/Downloads/vector.vtp'
                # write_geo(pfn, polydata_point)

        else:
            print("Angle not within limit, returning None for ip=" + str(ip))

    arr_rad = np.array(vessel_r)*magn_radius
    arr_pt = np.array(points)
    arr_angl = np.array(angles)

    if not mega_sub:
        sort_index = np.flip(np.argsort(arr_rad)) ## Sort from largest to smallest
    else:
        sort_index = np.argsort(arr_angl) ## Sort from smallest angle to largest

    return arr_pt[sort_index], arr_rad[sort_index], arr_angl[sort_index]

def get_point_ids_post_proc(centerline_poly):

    cent = centerline_poly
    num_points = cent.GetNumberOfPoints() # number of points in centerline
    cent_data = collect_arrays(cent.GetPointData())           # point locations as numpy array
    radii = cent_data['MaximumInscribedSphereRadius']   # Max Inscribed Sphere Radius as numpy array

    # cell_data = collect_arrays(cent.GetCellData())
    # points_in_cells = get_points_cells_pd(cent)

    cent_id = cent_data['CenterlineId']
    # num_cent = max(cent_id)+1 # number of centerlines (one is assembled of multiple)
    try:
        num_cent = len(cent_id[0]) # number of centerlines (one is assembled of multiple)
    except:
        num_cent = 1 # in the case of only one centerline
    
    point_ids_list = []
    for ip in range(num_cent):
        try:
            ids = [i for i in range(num_points) if cent_id[i,ip]==1]    # ids of points belonging to centerline ip
        except:
            ids = [i for i in range(num_points)]
        point_ids_list.append(ids)

    return point_ids_list

def get_point_ids_no_post_proc(centerline_poly):
    """
    For this case, the polydata does not have CenterlineIds,
    so we need to find the centerline ids manually based on the
    connectivity of the points
    Args:
        centerline_poly: vtk polydata of centerline
    Returns:
        point_ids: point ids of centerline (list of lists)
    """
    # the centerline is composed of vtk lines
    # Get the lines from the polydata
    point_ids_list = []
    # Iterate through cells and extract lines
    for i in range(centerline_poly.GetNumberOfCells()):
        cell = centerline_poly.GetCell(i)
        if cell.GetCellType() == 4:
            point_ids = []
            for j in range(cell.GetNumberOfPoints()):
                point_id = cell.GetPointId(j)
                # point = centerline_poly.GetPoint(point_id)
                point_ids.append(point_id)
            point_ids_list.append(point_ids)

    return point_ids_list


def get_point_ids(centerline_poly, post_proc = True):
    """
    Get the point ids of the centerline
    Args:
        centerline_poly: vtk polydata of centerline
        post_proc: boolean if post processing was used
    Returns:
        point_ids: point ids of centerline
    """
    if post_proc:
        point_ids = get_point_ids_post_proc(centerline_poly)
    else:
        point_ids = get_point_ids_no_post_proc(centerline_poly)

    return point_ids

def orient_caps(caps, current_point, old_point, direction):
    """
    Note: direction is local, old_point can be from previous subvolume
    Return a sorted list of cap ids based on angle compared to direction

    Args:
        caps: list of cap points
        current_point: current location
        old_point: previous location
        direction: direction of exploration
    Returns:
        sort_index: sorted list of cap ids
        source_id: id of source cap
    """
    
    source_dist = 100000
    target = []
    poly = []
    angles = []
    for i in range(len(caps)):
        target = target + caps[i].tolist()
        poly.append(caps[i].tolist())
        # calculate vector
        vector = caps[i] - current_point
        vector = vector/np.linalg.norm(vector)
        # angle between direction and vector to cap
        angle = 360/(2*np.pi)*np.arccos(np.dot(direction, vector))
        print(f"Angle to cap {i}: {angle}")
        angles.append(angle)

    for i in range(len(caps)):
        cap_dist = np.linalg.norm(caps[i] - old_point)
        print(f"Cap dist {i}: {cap_dist}")
        if  cap_dist < source_dist:
            source_id = i
            source_dist = cap_dist
            sourcee = caps[i].tolist()
    target[source_id*3:source_id*3+3] = []

    # sort the caps based on angle
    s = np.array(angles)
    sort_index = np.argsort(s).tolist()
    # remove source_id from sort_index
    sort_index.remove(source_id)
    print(f"Sorted cap ids: {sort_index}")

    #polydata_point = points2polydata(poly)
    #pfn = '/Users/numisveinsson/Downloads/points.vtp'
    #write_geo(pfn, polydata_point)

    return sort_index, source_id

def get_seed(cent_fn, centerline_num, point_on_cent):
    """
    Get a location and radius at a point along centerline
    Args:
        cent_fn: file directory for centerline
        centerline_num: starting from 0, which sub centerline do you wish to sample from
        point_on_cent: starting from 0, how far along the sub centerline you wish to sample
    Returns:
        location coords, radius at the specific point
    """

    # Centerline
    cent = read_geo(cent_fn).GetOutput()
    # Sort centerline into data of interest
    num_points, c_loc, radii, cent_ids, bifurc_id, num_cent = sort_centerline(cent)
    # Pick the branch of interest
    cent_id_wanted = cent_ids[centerline_num]
    # Pick the point along the branch
    id_point = cent_id_wanted[point_on_cent]

    return c_loc[id_point], radii[id_point]

def sort_centerline(centerline):
    """
    Function to sort the centerline data
    """

    num_points = centerline.GetNumberOfPoints()               # number of points in centerline
    cent_data = collect_arrays(centerline.GetPointData())
    c_loc = v2n(centerline.GetPoints().GetData())             # point locations as numpy array
    radii = cent_data['MaximumInscribedSphereRadius']   # Max Inscribed Sphere Radius as numpy array
    
    # get cent_ids, a list of lists
    # each list is the ids of the points belonging to a centerline
    try:
        cent_ids = get_point_ids_post_proc(centerline)
        bifurc_id = cent_data['BifurcationIdTmp']
    except:
        # centerline hasnt been processed
        cent_ids = get_point_ids_no_post_proc(centerline)
        bifurc_id = np.zeros(num_points)
        print(f"\nCenterline has not been processed, no known bifurcations\n")
    
    # check if there are duplicate points
    if np.unique(c_loc, axis=0).shape[0] != c_loc.shape[0]:
        # remove duplicate points
        print(f"\nCenterline has duplicate points, removing them\n")
        _, unique_ids = np.unique(c_loc, axis=0, return_index=True)
        # same for cent_ids, but keep same order
        cent_ids_new = []
        for i in range(len(cent_ids)):
            cent_ids_new.append([])
            for j in range(len(cent_ids[i])):
                if cent_ids[i][j] in unique_ids:
                    cent_ids_new[i].append(cent_ids[i][j])
        cent_ids = cent_ids_new

    # pdb.set_trace()
    num_cent = len(cent_ids)
    
    return num_points, c_loc, radii, cent_ids, bifurc_id, num_cent

def convert_seg_to_surfs(seg, target_node_num=100, bound=False, new_spacing=[1.,1.,1.], mega_sub = False, ref_min_dims = None):
    
    import SimpleITK as sitk

    py_seg = sitk.GetArrayFromImage(seg)
    # py_seg = eraseBoundary(py_seg, 1, 0)
    labels = np.unique(py_seg)
    for i, l in enumerate(labels):
        py_seg[py_seg==l] = i
    seg2 = sitk.GetImageFromArray(py_seg)
    seg2.CopyInformation(seg)

    seg_vtk,_ = exportSitk2VTK(seg2)

    # choose new spacing based on resolution
    # divide spacing if min dimension is less than 10 pixels
    spacing = np.array(seg.GetSpacing())
    dims = np.array(seg.GetSize())

    # if not mega_sub:    min_dim = 10
    # # need larger res for mega
    # else:               min_dim = 20

    if not mega_sub:    ref_min_dim = np.min(dims)
    # need to refer to current local sub if we doing mega
    else:               ref_min_dim = np.min(ref_min_dims)

    if ref_min_dim < 20:
        print(f"Upsampling segmentation because low resolution may cause problems")
        new_spacing = (spacing/3.).tolist()
        seg_vtk = vtkImageResample(seg_vtk,new_spacing,'cubic')

    poly_l = []
    for i, _ in enumerate(labels):
        if i==0:
            continue
        p = vtk_marching_cube(seg_vtk, 0, i)
        # p = smooth_polydata(p, iteration=10)
        # rate = max(0., 1. - float(target_node_num)/float(p.GetNumberOfPoints()))
        # p = decimation(p, rate)
        # p = smooth_polydata(p, iteration=20)
        arr = np.ones(p.GetNumberOfPoints())*i
        arr_vtk = n2v(arr)
        arr_vtk.SetName('RegionId')
        p.GetPointData().AddArray(arr_vtk)
        poly_l.append(p)
    poly = appendPolyData(poly_l)
    if bound:
        poly = bound_polydata_by_image(seg_vtk, poly, 1.5)
    return poly

    def calc_centerline_vmtk(surface_smooth, global_config, source_id, sorted_targets, i, caps, step_seg, length):

        centerline_poly = calc_centerline(  surface_smooth,
                                        global_config['TYPE_CENT'],
                                        var_source=[source_id],
                                        var_target = sorted_targets,
                                        number = i,
                                        caps = caps,
                                        point = step_seg['point'])

        centerline_poly = resample_centerline(centerline_poly)
        # if write_samples:
        #     write_centerline(centerline_poly, cfn.replace('.vtp', 'resampled.vtp'))
        centerline_poly = smooth_centerline(centerline_poly)
        # if write_samples:
        #     write_centerline(centerline_poly, cfn.replace('.vtp', 'smooth.vtp'))

        if not centerline_poly or centerline_poly.GetNumberOfPoints() < 5:
            print("\n Attempting with more smoothing \n")
            surface_smooth1 = smooth_surface(surface, 15)
            surface_smooth1 = bound_polydata_by_image(vtkimage[0], surface_smooth1, length*1/40)
            centerline_poly1 = calc_centerline(surface_smooth1,
                                                global_config['TYPE_CENT'],
                                                var_source=[source_id],
                                                var_target = sorted_targets,
                                                number = i,
                                                caps = caps,
                                                point = step_seg['point'])
            if centerline_poly1.GetNumberOfPoints() > 5:
                sfn = output_folder +'surfaces/surf_'+case+'_'+str(i)+'_1.vtp'
                surface_smooth = surface_smooth1
                cfn = output_folder +'centerlines/cent_'+case+'_'+str(i)+'_1.vtp'
                centerline_poly = centerline_poly1
                success = True
            else: success = False
        else: success = True

        return centerline_poly, success