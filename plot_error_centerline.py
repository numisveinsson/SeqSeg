import faulthandler

faulthandler.enable()

import time
start_time = time.time()

import os

import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

from modules import sitk_functions as sf
from modules import vtk_functions as vf
from modules import vmtk_functions as vmtkfs

def calc_distance(path, cent):

    """
    - Calculate distance between centerlines, GT and predicted (path)
    - Goes through each GT point and locates closest path point and
      calculates the distance between
    """
    # get number of points in centerline
    n_point_c = cent.GetNumberOfPoints()

	# get points: Note, maybe useless
    c_points = cent.GetPoints()
    p_points = path.GetPoints()
    dataset = vtk.vtkPolyData()
    dataset.SetPoints(p_points)
    locator = vtk.vtkPointLocator()
    locator.Initialize()
    locator.SetDataSet(dataset)
    locator.BuildLocator()
    path.locator = locator

    c_loc = v2n(cent.GetPoints().GetData())
    p_loc = v2n(path.GetPoints().GetData())

    p_closest_ids = []
    for i in c_loc:
        p_closest_ids += [path.locator.FindClosestPoint(i)]

    a = c_loc
    b = p_loc[p_closest_ids]
    dist = np.ones((n_point_c), dtype = float)

    for i in range(n_point_c):
        dist[i] = np.linalg.norm(a[i]-b[i])

    return dist

def calc_radius_error(path, cent):

    path_data = vf.collect_arrays(path.GetPointData())
    cent_data = vf.collect_arrays(cent.GetPointData())

    # get number of points in centerline
    n_point_c = cent.GetNumberOfPoints()

	# get points
    p_points = path.GetPoints()
    dataset = vtk.vtkPolyData()
    dataset.SetPoints(p_points)
    locator = vtk.vtkPointLocator()
    locator.Initialize()
    locator.SetDataSet(dataset)
    locator.BuildLocator()
    path.locator = locator

    c_loc = v2n(cent.GetPoints().GetData())
    p_loc = v2n(path.GetPoints().GetData())

    rads_closest_ids = []
    for i in range(len(c_loc)):
        id = path.locator.FindClosestPoint(c_loc[i])
        dist = np.linalg.norm(c_loc[i]-p_loc[id])
        if dist < cent_data['MaximumInscribedSphereRadius'][i]:
            rads_closest_ids += [path_data['MaximumInscribedSphereRadius'][id]]
        else:
            rads_closest_ids += [0]

    radius_error = np.abs(cent_data['MaximumInscribedSphereRadius'] - np.array(rads_closest_ids))
    relative_radius_error = radius_error/cent_data['MaximumInscribedSphereRadius']

    return radius_error, relative_radius_error

def count_branches_caught(path, cent):

    # get number of points in centerline
    n_point_c = cent.GetNumberOfPoints()
    cent_data = vf.collect_arrays(cent.GetPointData())

	# get points: Note, maybe useless
    c_points = cent.GetPoints()
    p_points = path.GetPoints()
    dataset = vtk.vtkPolyData()
    dataset.SetPoints(p_points)
    locator = vtk.vtkPointLocator()
    locator.Initialize()
    locator.SetDataSet(dataset)
    locator.BuildLocator()
    path.locator = locator

    c_loc = v2n(cent.GetPoints().GetData())
    p_loc = v2n(path.GetPoints().GetData())

    missed_branches = 0
    branchids_checked = []
    for i in range(n_point_c):
        p_closest_id = path.locator.FindClosestPoint(c_loc[i])
        dist = np.linalg.norm(c_loc[i]-p_loc[p_closest_id])

        if dist > cent_data['MaximumInscribedSphereRadius'][i] and cent_data['BranchId'][i] not in branchids_checked:
            missed_branches += 1
            branchids_checked += [cent_data['BranchId'][i]]

    percent_branches_caught = (1 - missed_branches/(np.max(cent_data['BranchId'])+1))*100

    return percent_branches_caught

def calc_errors(fname_path, fname_cent):

    path = vf.read_geo(fname_path).GetOutput()
    cent = vf.read_geo(fname_cent).GetOutput()

    dist = calc_distance(path, cent)
    print('Ave distance: '+str(dist.mean()))
    rad_error,relative_rad_error = calc_radius_error(path, cent)
    print('Ave radius error: '+str(rad_error.mean()))
    print('Ave relative radius error: '+str(relative_rad_error.mean()))

    percent_branches_caught = count_branches_caught(path, cent)
    print('Percent Branches Caught: '+ str(percent_branches_caught))

    out_array = n2v(dist)
    out_array.SetName('Distance from Closest Pathline Point')
    cent.GetPointData().AddArray(out_array)

    out_array = n2v(rad_error)
    out_array.SetName('Difference from Predicted Radius')
    cent.GetPointData().AddArray(out_array)

    out_array = n2v(relative_rad_error)
    out_array.SetName('Relative Difference from Predicted Radius')
    cent.GetPointData().AddArray(out_array)

    return cent, percent_branches_caught

if __name__=='__main__':

    fname_est = '/Users/numisveinsson/Documents/Berkeley/Research/Automatic_Centerline_ML/output_test27_0146_1001_0/final_assembly0146_1001_test27_0_centerline_manual_smooth.vtp'
    fname_true = '/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/vascular_data_3d/centerlines/0146_1001.vtp'

    cent, percent_branches_caught = calc_errors(fname_est, fname_true)

	# write geometry to file
    f_out = fname_est.replace('centerline_manual_smooth', 'centerline_manual_smooth_error')
    vf.write_geo(f_out, cent)
