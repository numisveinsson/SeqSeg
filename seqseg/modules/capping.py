import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy as v2n
from modules.vtk_functions import (get_largest_connected_polydata,
                                   get_points_cells, points2polydata,
                                   write_geo, exportSitk2VTK, exportVTK2Sitk,
                                   convertPolyDataToImageData)


def bryan_get_clipping_parameters(clpd):
    """ get all three parameters """
    points = v2n(clpd.GetPoints().GetData())

    if clpd.GetPointData().GetArray('CenterlineId') is not None:
        CenterlineID_for_each_point = v2n(clpd.GetPointData().GetArray(
                                        'CenterlineId'))
    else:
        raise ValueError("CenterlineId array not found in the input polydata")

    radii = v2n(clpd.GetPointData().GetArray('MaximumInscribedSphereRadius'))
    n_keys = len(CenterlineID_for_each_point[0])
    line_dict = {}
    # create dict with keys line0, line1, line2, etc
    for i in range(n_keys):
        key = f"line{i}"
        line_dict[key] = []

    for i in range(len(points)):
        for j in range(n_keys):
            if CenterlineID_for_each_point[i][j] == 1:
                key = f"line{j}"
                line_dict[key].append(points[i])

    for i in range(n_keys):
        key = f"line{i}"
        line_dict[key] = np.array(line_dict[key])
    # Done with spliting centerliens into dictioanry

    # find the end points of each line
    lst_of_end_pts = []
    # append the very first point
    lst_of_end_pts.append(line_dict["line0"][0])
    # append the rest of the end points
    for i in range(n_keys):
        key = f"line{i}"
        lst_of_end_pts.append(line_dict[key][-1])
    nplst_of_endpts = np.array(lst_of_end_pts)  # convert to numpy array

    # find the radii at the end points
    radii_at_caps = []
    for i in lst_of_end_pts:
        for j in range(len(points)):
            if np.array_equal(i, points[j]):
                radii_at_caps.append(radii[j])
    nplst_radii_at_caps = np.array(radii_at_caps)  # convert to numpy array

    # find the unit tangent vectors at the end points
    unit_tangent_vectors = []
    # compute the unit tangent vector of the first point of the first line
    key = "line0"
    line = line_dict[key]
    tangent_vector = line[0] - line[1]
    unit_tangent_vector = tangent_vector / np.linalg.norm(tangent_vector)
    unit_tangent_vectors.append(unit_tangent_vector)
    # compute the unit tangent vector of the last point of each line
    for i in range(len(line_dict)):
        key = f"line{i}"
        line = line_dict[key]
        tangent_vector = line[-1] - line[-2]
        unit_tangent_vector = tangent_vector / np.linalg.norm(tangent_vector)
        unit_tangent_vectors.append(unit_tangent_vector)

    return nplst_of_endpts, nplst_radii_at_caps, unit_tangent_vectors


def get_clipping_parameters_cells(clpd, targets, outdir):
    """
    Loop over the cells of the input polydata and find the end points of
    the centerlines that are closest to the target points.
    Assumes number of cells = number of centerlines.
    """
    # Get the points and cells of the input polydata
    # cells is a list of lists, where each sublist contains the indices of the points
    # pts is a np array of shape (n_points, 3)
    pts, cells = get_points_cells(clpd)
    # check first and last point of each cell, which is the closest to the target points is the end point
    endpts = []
    radii = []
    unit_vecs = []
    for i in range(len(cells)):
        # If less than 6 points in the centerline, skip it
        if len(cells[i]) < 6:
            continue
        target = targets[i]
        # Get the first and last point of the centerline
        first_pt = pts[cells[i][0]]
        last_pt = pts[cells[i][-1]]
        # Compute the distance between the first and last point of the centerline and the target points
        dist_first = np.linalg.norm(first_pt - target)
        dist_last = np.linalg.norm(last_pt - target)
        # Append the end point of the centerline that is closest to the target points
        if dist_first < dist_last:
            endpts.append(first_pt)
            radii.append(clpd.GetPointData().GetArray('MaximumInscribedSphereRadius').GetValue(cells[i][5]))
            # Compute the unit tangent vector of the centerline
            second_first_pt = pts[cells[i][1]]
            tangent_vector = first_pt - second_first_pt
            unit_vecs.append(tangent_vector / np.linalg.norm(tangent_vector))
        else:
            endpts.append(last_pt)
            radii.append(clpd.GetPointData().GetArray('MaximumInscribedSphereRadius').GetValue(cells[i][-5]))
            # Compute the unit tangent vector of the centerline
            second_last_pt = pts[cells[i][-2]]
            tangent_vector = last_pt - second_last_pt
            unit_vecs.append(tangent_vector / np.linalg.norm(tangent_vector))

    # write the end points to a .geo file
    endpts_pd = points2polydata(endpts)
    write_geo(outdir+'endpts.vtp', endpts_pd)

    return endpts, radii, unit_vecs


def bryan_generate_oriented_boxes(endpts, unit_tan_vectors, radius,
                                  output_file, outdir, box_scale=3):

    box_surfaces = vtk.vtkAppendPolyData()
    # Convert the input center_points to a list, in case it is a NumPy array
    endpts = np.array(endpts).tolist()
    centerpts = []
    pd_lst = []
    for i in range(len(endpts)):
        compute_x = endpts[i][0]+0.5*box_scale*radius[i]*unit_tan_vectors[i][0]
        compute_y = endpts[i][1]+0.5*box_scale*radius[i]*unit_tan_vectors[i][1]
        compute_z = endpts[i][2]+0.5*box_scale*radius[i]*unit_tan_vectors[i][2]
        centerpts.append([compute_x, compute_y, compute_z])

    box_surfaces = vtk.vtkAppendPolyData()

    for i in range(len(centerpts)):
        # Create an initial vtkCubeSource for the box
        box = vtk.vtkCubeSource()
        box.SetXLength(box_scale*radius[i])
        box.SetYLength(box_scale*radius[i])
        box.SetZLength(box_scale*radius[i])
        box.Update()

        # Compute the rotation axis by taking the cross product of the unit_vector and the z-axis
        rotation_axis = np.cross(np.array([0, 0, 1]), unit_tan_vectors[i])

        # Compute the rotation angle in degrees between the unit_vector and the z-axis
        rotation_angle = np.degrees(np.arccos(np.dot(unit_tan_vectors[i],
                                                     np.array([0, 0, 1]))))
        transform = vtk.vtkTransform()
        transform.Translate(centerpts[i])
        transform.RotateWXYZ(rotation_angle, rotation_axis)

        # Apply the transform to the box
        box_transform = vtk.vtkTransformPolyDataFilter()
        box_transform.SetInputConnection(box.GetOutputPort())
        box_transform.SetTransform(transform)
        box_transform.Update()

        pd_lst.append(box_transform.GetOutput())
        box_surfaces.AddInputData(box_transform.GetOutput())

    box_surfaces.Update()

    # Write the oriented box to a .vtp file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(outdir+output_file+'.vtp')
    writer.SetInputData(box_surfaces.GetOutput())
    writer.Write()
    return box_surfaces.GetOutput(), pd_lst


def bryan_clip_surface(surf1, surf2):
    # Create an implicit function from surf2
    implicit_function = vtk.vtkImplicitPolyDataDistance()
    implicit_function.SetInput(surf2)

    # Create a vtkClipPolyData filter and set the input and implicit function
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(surf1)
    clipper.SetClipFunction(implicit_function)
    clipper.InsideOutOff()  # keep the part of surf1 outside of surf2
    clipper.Update()

    # Get the output polyData with the part enclosed by surf2 clipped away
    clipped_surf1 = clipper.GetOutput()

    return clipped_surf1


def cap_surface(pred_surface, centerline, pred_seg, file_name, outdir,
                targets=None):

    if targets is None:
        endpts, radii, unit_vecs = bryan_get_clipping_parameters(centerline)
    else:

        endpts, radii, unit_vecs = get_clipping_parameters_cells(centerline,
                                                                 targets,
                                                                 outdir)
    boxpd, boxpdlst = bryan_generate_oriented_boxes(endpts, unit_vecs, radii,
                                                    file_name+'_boxclips',
                                                    outdir, 4)
    clippedpd = bryan_clip_surface(pred_surface, boxpd)
    largest = get_largest_connected_polydata(clippedpd)

    img_vtk = exportSitk2VTK(pred_seg)[0]
    seg_vtk = convertPolyDataToImageData(largest, img_vtk)
    capped_seg = exportVTK2Sitk(seg_vtk)

    return clippedpd, capped_seg
