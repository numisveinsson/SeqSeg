import faulthandler

faulthandler.enable()

#import warnings
#warnings.filterwarnings('ignore')

import time
start_time = time.time()
from datetime import datetime

import os
import sys
sys.stdout.flush()
import pickle
import argparse
import numpy as np
import SimpleITK as sitk

from modules import sitk_functions as sf
from modules import vtk_functions as vf
# from modules import vmtk_functions as vmtkfs
from modules import initialization as init
from modules.tracing import trace_centerline
from modules.datasets import get_testing_samples_json
from modules.assembly import create_step_dict
from modules.evaluation import EvaluateTracing
from modules.params import load_yaml

def create_directories(output_folder, write_samples):
    try:
        os.mkdir(output_folder)
    except Exception as e: print(e)
    try:
        os.mkdir(output_folder+'errors')
    except Exception as e: print(e)
    try:
        os.mkdir(output_folder+'assembly')
    except Exception as e: print(e)

    if write_samples:
        try:
            os.mkdir(output_folder+'volumes')
        except Exception as e: print(e)
        try:
            os.mkdir(output_folder+'predictions')
        except Exception as e: print(e)
        try:
            os.mkdir(output_folder+'centerlines')
        except Exception as e: print(e)
        try:
            os.mkdir(output_folder+'surfaces')
        except Exception as e: print(e)
        try:
            os.mkdir(output_folder+'points')
        except Exception as e: print(e)
        try:
            os.mkdir(output_folder+'animation')
        except Exception as e: print(e)


def get_testing_samples(dataset):

    directory = '/global/scratch/users/numi/vascular_data_3d/'
    if dataset == 'Dataset005_SEQAORTANDFEMOMR':
        testing_samples = [

            ['0006_0001',0,10,20,'mr'], # Aortofemoral MR
            ['0063_1001',0,10,20,'mr'], # Aortic MR
            ['0090_0001',0,10,20,'mr'], # Aortic MR
            ['0131_0000',0,10,20,'mr'], # Aortic MR
            ['0070_0001',0,10,20,'mr'], # Aortic MR
            ['KDR12_aorta',0,20,30,'mr'], # Aortic MR
            ['KDR33_aorta',3,-10,-20,'mr'], # Aortic MR

        ]

    elif dataset == 'Dataset006_SEQAORTANDFEMOCT':
        directory = '/global/scratch/users/numi/vascular_data_3d/'
        dir_json = directory + 'test.json'
        testing_samples = get_testing_samples_json(dir_json)
        
        # [

            # ['0176_0000',0,10,20,'ct'], # Aorta CT
            # ['0174_0000',0,10,20,'ct'], # Aorta CT
            # ['0139_1001',0,10,20,'ct'], # Aortofemoral CT
            # ['0141_1001',0,0,10,'ct'], # Aortofemoral CT
            # ['0146_1001',0,0,10,'ct'], # Aortofemoral CT
            # ['0188_0001_aorta',5,-10,-20,'ct'], # Aorta CT
            # ['O150323_2009_aorta',0,10,20,'ct'], # Aorta CT
            # ['O344211000_2006_aorta',0,10,20,'ct'], # Aorta CT
        # ]
    
    elif dataset == 'Dataset007_SEQPULMONARYMR':
        testing_samples = [

            ['0085_1001',0,0,10,'mr'], # Pulmonary MR
            # ['0085_1001',0,200,220,'mr'], # Pulmonary MR
            # ['0085_1001',1,200,220,'mr'], # Pulmonary MR
            ['0081_0001',0,20,30,'mr'], # Pulmonary MR
            # ['0081_0001',1,200,220,'mr'], # Pulmonary MR
            # ['0081_0001',1,2000,220,'mr'], # Pulmonary MR
        ]
    elif dataset == 'Dataset009_SEQAORTASMICCT':
        
        directory = '/global/scratch/users/numi/test_data/miccai_aortas/'
        dir_json = directory + 'test.json'
        testing_samples = get_testing_samples_json(dir_json)


    elif dataset == 'Dataset010_SEQCOROASOCACT':

        directory = '/global/scratch/users/numi/ASOCA_test/'
        dir_asoca_json = directory + 'test.json'
        testing_samples = get_testing_samples_json(dir_asoca_json)

    else:
        print('Dataset not found')
        testing_samples =  None

    #    ['0108_0001_aorta',4,-10,-20,'ct'],
    #    ['0183_1002_aorta',3,-10,-20,'ct'],
    #    ['0184_0001_aorta',3,-10,-20,'ct'],
    #    ['0188_0001_aorta',5,-10,-20,'ct'],
    #    ['0189_0001_aorta',4,-50,-60,'ct'],

    #    ['O0171SC_aorta',0,10,20,'ct'],
    #    ['O6397SC_aorta',0,10,20,'ct'],
    #    ['O8693SC_aorta',0,10,20,'ct'],
    #    ['O344211000_2006_aorta',0,10,20,'ct'],
    #    ['O11908_aorta',0,10,20,'ct'],
    #    ['O20719_2006_aorta',0,10,20,'ct'],
    #   these are left:
    #    ['O51001_2009_aorta',0,10,20,'ct'],
    #    ['O128301_2008_aorta',0,10,20,'ct'],
    #    ['O145207_aorta',0,10,20,'ct'],
    #    ['O150323_2009_aorta',0,10,20,'ct'],
    #    ['O227241_2006_aorta',0,10,20,'ct'],
    #    ['O351095_aorta',0,10,20,'ct'],
    #    ['O690801_2007_aorta',0,10,20,'ct'],

    #    ['KDR08_aorta',0,10,20,'mr'],
    #    ['KDR10_aorta',0,10,20,'mr'],
    #    ['KDR12_aorta',0,10,20,'mr'],
    #    ['KDR13_aorta',0,10,20,'mr'],
    #    ['KDR32_aorta',0,10,20,'mr'],
    #    ['KDR33_aorta',3,-10,-20,'mr'],
    #    ['KDR34_aorta',0,10,20,'mr'],
    #    ['KDR48_aorta',0,10,20,'mr'],
    #    ['KDR57_aorta',4,-10,-20,'mr'],

    #    ['0002_0001',0,150,170,'ct']  ,
    #    ['0002_0001',1,150, 170,'ct'] ,
    #    ['0001_0001',0,30,50,'ct'],
    #    ['0001_0001',7,30,50,'ct'],
    #    ['0001_0001',8,110,130,'ct'],
    #    ['0005_1001',0,300,320,'ct']  ,
    #    ['0005_1001',1,200,220,'ct']  ,

    return testing_samples, directory

if __name__=='__main__':
    """# Set up"""

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_directory',                     help='Name of the folder containing the testing data')
    parser.add_argument('--test_name',  default= '3d_fullres'   help='Name of nnUNet test to use, eg 3d_fullres/2d')
    parser.add_argument('--dataset',                            help='Name of dataset used to train nnUNet, eg Dataset010_SEQCOROASOCACT')
    parser.add_argument('--fold',       default= 'all',         help='Which fold to use for nnUNet model')
    parser.add_argument('--img_ext',                            help='Image extension, eg .nii.gz')
    parser.add_argument('--scale',      default= 1,             help='Whether to scale image data, needed if units for nnUNet model and testing data are different)
    args = parser.parse_args()

    import pdb; pdb.set_trace()

    #       [name , dataset, fold, modality, json file present, unit]
    #       note: fold is either 0,1,2,3 or 'all'
    #       note: json file present is either True or False
    tests = [
            #  ['3d_fullres','Dataset002_SEQAORTAS', 0],
            #  ['3d_fullres','Dataset005_SEQAORTANDFEMOMR', 'all', False],
            #  ['3d_fullres','Dataset006_SEQAORTANDFEMOCT', 'all', True, '.vtk', 0.1], # 0.1 here means scaling (model is cm but data is mm)
            #  ['3d_fullres','Dataset007_SEQPULMONARYMR', 'all', False],
            #  ['3d_fullres','Dataset009_SEQAORTASMICCT', 'all', True, '.nrrd', 1], # 1 here means no scaling (model and data are both mm)
             ['3d_fullres','Dataset010_SEQCOROASOCACT', 'all', True, '.nrrd', 1],
            ]

    global_config = load_yaml("./config/global.yaml")
    
    # dir_output0 = 'output_miccai_1000/'

    dir_output0 = 'output_2000_steps/'

    max_step_size  = global_config['MAX_STEPS']
    write_samples  = global_config['WRITE_STEPS']
    retrace_cent   = global_config['RETRACE']
    take_time      = global_config['TIME_ANALYSIS']
    weighted = global_config['WEIGHTED_ASSEMBLY']
    calc_global_centerline = global_config['GLOBAL_CENTERLINE']

    for test in tests:

        print('\n test is: \n', test)
        dataset = test[1]
        fold = test[2]
        json_file_present = test[3]
        img_format = test[4]
        scale = test[5]
        test_name = test[0]

        ## Weight directory
        dir_model_weights = dataset+'/nnUNetTrainer__nnUNetPlans__'+test_name

        testing_samples, directory_data = get_testing_samples(dataset)
        print(f"Testing samples about to run: {testing_samples}")
        ct_dice, mr_dice, ct_cent, mr_cent = [],[],[],[]

        final_dice_scores, final_perc_caught, final_tot_perc, final_missed_branches, final_n_steps_taken, final_ave_step_dice = [],[],[],[],[],[]
        
        for test_case in testing_samples[2:]:

            print(test_case)

            dir_output, dir_image, dir_seg, dir_cent, case, i = init.process_init(test_case, 
                                                                                            directory_data, 
                                                                                            dir_output0, 
                                                                                            img_format,
                                                                                            json_file_present, 
                                                                                            test_name)

            ## Create directories for results
            create_directories(dir_output, write_samples)

            potential_branches, initial_seeds = init.initialization(json_file_present, test_case, dir_output, dir_cent, directory_data, scale, write_samples)
            
            # print to .txt file all outputs
            sys.stdout = open(dir_output+"/out.txt", "w")
            print(test_case)
            print(f"Initial points: {potential_branches}")
            # print(f"Time is: {time.time()}")

            ## Trace centerline
            centerlines, surfaces, points, assembly_obj, vessel_tree, n_steps_taken = trace_centerline( dir_output,
                                                                                                    dir_image,
                                                                                                    case,
                                                                                                    dir_model_weights,
                                                                                                    fold,
                                                                                                    potential_branches,
                                                                                                    max_step_size,
                                                                                                    scale,
                                                                                                    dir_seg,
                                                                                                    write_samples,
                                                                                                    take_time,
                                                                                                    retrace_cent,
                                                                                                    weighted)

            print("\nTotal calculation time is: " + str((time.time() - start_time)/60) + " min\n")

            if take_time:
                vessel_tree.time_analysis()

            # End points
            if calc_global_centerline:
                end_points = vessel_tree.get_end_points()
                in_source = end_points[0].tolist()
                in_target_lists = [point.tolist() for point in end_points[1:]]
                in_target = []
                for target in in_target_lists:
                    in_target += target

            ## Assembly work
            assembly_org = assembly_obj.assembly
            #assembly_ups = assembly_obj.upsample_sitk()
            print("\nTotal calculation time is: " + str((time.time() - start_time)/60) + " min\n")
            #sitk.WriteImage(assembly_org, dir_output+'/'+case+'_assembly_' + test_name +'_'+str(i)+'.mha')

            assembly = assembly_org
            name = 'original'
            #for assembly,name in zip([assembly_ups, assembly_org],['upsampled', 'original']):
            assembly_binary = sitk.BinaryThreshold(assembly, lowerThreshold=0.5, upperThreshold=1)
            sitk.WriteImage(assembly_binary, dir_output+'/'+case+'_seg_'+ test_name +'_'+str(i)+'.mha')
            
            assembly_binary = sf.keep_component_seeds(assembly_binary, initial_seeds)
            sitk.WriteImage(assembly_binary, dir_output+'/'+case+'_seg_rem_' + test_name +'_'+str(i)+'.mha')

            assembly_surface    = vf.evaluate_surface(assembly_binary, 1)
            vf.write_vtk_polydata(assembly_surface, dir_output+'/final_assembly_'+name+'_'+case+'_'+test_name +'_'+str(i)+'_'+str(n_steps_taken)+'_'+'_surface.vtp')
            for level in [10,40]:#range(10,50,10):
                surface_smooth      = vf.smooth_surface(assembly_surface, level)
                vf.write_vtk_polydata(surface_smooth, dir_output+'/final_assembly_'+name+'_'+case+'_'+test_name +'_'+str(i)+'_'+str(n_steps_taken)+'_'+str(level)+'_surface_smooth.vtp')
                #path = vmtkfs.calc_centerline(   surface_smooth, "pointlist", var_source=in_source, var_target=in_target)
                #vf.write_vtk_polydata(path, dir_output+'/final_assembly'+case+'_'+test_name +'_'+str(i)+'_'+str(n_steps_taken)+'_'+str(level)+'_centerline_smooth.vtp')


            #path1 = vmtkfs.calc_centerline(surface_smooth, "profileidlist", number = 0)
            #vf.write_vtk_polydata(path1, dir_output+'/final_assembly'+case+'_'+test_name +'_'+str(i)+'_'+str(n_steps_taken)+'_centerline_smooth1.vtp')


            final_surface = vf.appendPolyData(surfaces)
            final_centerline = vf.appendPolyData(centerlines)
            final_points = vf.appendPolyData(points)
            vf.write_vtk_polydata(final_surface,    dir_output+'/final_'+case+'_'+test_name +'_'+str(i)+'_'+str(n_steps_taken)+'_surfaces.vtp')
            vf.write_vtk_polydata(final_centerline, dir_output+'/final_'+case+'_'+test_name +'_'+str(i)+'_'+str(n_steps_taken)+'_centerlines.vtp')
            vf.write_vtk_polydata(final_points,     dir_output+'/final_'+case+'_'+test_name +'_'+str(i)+'_'+str(n_steps_taken)+'_points.vtp')

            #print('Number of outlets: ' + str(len(final_caps[1])))


        print("\nTotal calculation time is: " + str((time.time() - start_time)/60) + " min\n")
