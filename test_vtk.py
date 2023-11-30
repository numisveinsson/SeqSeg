import faulthandler

faulthandler.enable()

import time
start_time = time.time()

import os

import numpy as np
import SimpleITK as sitk

from modules import sitk_functions as sf
from modules import vtk_functions as vf
import vtk
# from modules import vmtk_functions as vmtkfs

def organize_polydata(polydata):
    # Get the lines from the polydata
    lines = []
    # Iterate through cells and extract lines
    for i in range(polydata.GetNumberOfCells()):
        cell = polydata.GetCell(i)
        if cell.GetCellType() == 4:
            line = []
            for j in range(cell.GetNumberOfPoints()):
                point_id = cell.GetPointId(j)
                point = polydata.GetPoint(point_id)
                line.append(point)
            lines.append(line)

    return lines

def calc_distance_between_points(point1, point2):
    # Calculate the distance between two points
    distance = np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))
    return distance

def calc_dist_lines(lines):
    # Calculate the distance along the centerline between each pair of points in line
    # Input: lines - list of lines, each line is a list of points
    dists = []
    for i in range(len(lines)):
        line = lines[i]
        dist = []
        for j in range(len(line)-1):
            dist.append(calc_distance_between_points(line[j], line[j+1]))
        dists.append(dist)
    return dists


if __name__=='__main__':

    # fname = '/Users/numisveinsson/Documents/Berkeley/Research/Automatic_Centerline_ML/output_test28_0002_0001/surfaces/surf_0002_0001_27.vtp'
    # surf = vf.read_geo(fname).GetOutput()
    #
    # caps = vf.calc_caps(surf)
    #
    # pfn = '/Users/numisveinsson/Downloads/point.vtp'
    # polydata_point = vf.points2polydata(caps)
    # vf.write_geo(pfn, polydata_point)
    #
    # #import pdb; pdb.set_trace()
    # centerline_poly = vmtkfs.calc_centerline(surf, "pointlist", caps[2].tolist(), caps[0].tolist() +caps[1].tolist(), number = 0)
    #
    # cfn = '/Users/numisveinsson/Downloads/centerline.vtp'
    # vmtkfs.write_centerline(centerline_poly, cfn)
    import pdb; pdb.set_trace()

    cent_file = '/Users/numisveins/Downloads/3d_fullres_0081_0001_0/centerlines/cent_0081_0001_41.vtp'

    centerline = vf.read_geo(cent_file).GetOutput()
    # get centerline lines
    lines = organize_polydata(centerline)




    from modules import vtk_functions as vf
    dir = '/Users/numisveinsson/Documents_numi/vmr_data_new/images/'
    imgs = os.listdir(dir)
    imgs = [img for img in imgs if '.vti' in img]

    for img in imgs:
        vf.change_vti_vtk(dir + img)
        print(f"Done: {img}")
    import pdb; pdb.set_trace()
    ####

    fname = '/Users/numisveinsson/Documents/Berkeley/Research/Automatic_Centerline_ML/output_test27_0146_1001_0/final_assembly0146_1001_test27_0_surface_smooth.vtp'

    surf = vf.read_geo(fname).GetOutput()
    # source = [ -0.44117078,   4.0839257,  -10.793491 ]
    # target = [ -2.44117078,   0.0839257,  -27.793491 ]
    # polydata_point = vf.points2polydata([target])
    # pfn = '/Users/numisveinsson/Downloads/point.vtp'
    # vf.write_geo(pfn, polydata_point)

    centerline_poly = vmtkfs.calc_centerline(surf, "pickpoint", number = 0)

    vf.write_vtk_polydata(centerline_poly, fname.replace('surface', 'centerline_manual'))
