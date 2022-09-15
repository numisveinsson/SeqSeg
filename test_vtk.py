import faulthandler

faulthandler.enable()

import time
start_time = time.time()

import os

import numpy as np
import SimpleITK as sitk

from modules import sitk_functions as sf
from modules import vtk_functions as vf
from modules import vmtk_functions as vmtkfs

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
    # import pdb; pdb.set_trace()

    fname = '/Users/numisveinsson/Documents/Berkeley/Research/Automatic_Centerline_ML/output_test27_0146_1001_0/final_assembly0146_1001_test27_0_surface_smooth.vtp'

    surf = vf.read_geo(fname).GetOutput()
    # source = [ -0.44117078,   4.0839257,  -10.793491 ]
    # target = [ -2.44117078,   0.0839257,  -27.793491 ]
    # polydata_point = vf.points2polydata([target])
    # pfn = '/Users/numisveinsson/Downloads/point.vtp'
    # vf.write_geo(pfn, polydata_point)

    centerline_poly = vmtkfs.calc_centerline(surf, "pickpoint", number = 0)

    vf.write_vtk_polydata(centerline_poly, fname.replace('surface', 'centerline_manual'))
