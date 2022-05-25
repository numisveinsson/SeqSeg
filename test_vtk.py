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

    fname = '/Users/numisveinsson/Documents/Berkeley/Research/Automatic_Centerline_ML/output_test28_0002_0001/surfaces/surf_0002_0001_27.vtp'
    surf = vf.read_geo(fname).GetOutput()

    caps = vf.calc_caps(surf)

    pfn = '/Users/numisveinsson/Downloads/point.vtp'
    polydata_point = vf.points2polydata(caps)
    vf.write_geo(pfn, polydata_point)

    import pdb; pdb.set_trace()
