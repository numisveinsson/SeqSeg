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
from modules import vmtk_functions as vmtkfs
from modules.tracing import trace_centerline
from modules.datasets import vmr_directories, get_directories, get_testing_samples_json
from modules.assembly import create_step_dict
from modules.evaluation import EvaluateTracing

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
        testing_samples = [

            # ['0176_0000',0,10,20,'ct'], # Aorta CT
            # ['0174_0000',0,10,20,'ct'], # Aorta CT
            # ['0139_1001',0,10,20,'ct'], # Aortofemoral CT
            ['0141_1001',0,0,10,'ct'], # Aortofemoral CT
            # ['0146_1001',0,0,10,'ct'], # Aortofemoral CT
            # ['0188_0001_aorta',5,-10,-20,'ct'], # Aorta CT
            # ['O150323_2009_aorta',0,10,20,'ct'], # Aorta CT
            # ['O344211000_2006_aorta',0,10,20,'ct'], # Aorta CT
        ]
    
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
    parser.add_argument('--dataset',  help='Name of the folder containing the image data')
    parser.add_argument('--model',  help='Name of the folder containing the image data')
    parser.add_argument('--output',  help='Name of the output folder')
    parser.add_argument('--modality', nargs='+', help='Name of the modality, mr, ct, split by space')
    args = parser.parse_args()

    #       [name , dataset, fold, modality, json file present]
    #       note: fold is either 0,1,2,3 or 'all'
    #       note: json file present is either True or False
    tests = [
            #  ['3d_fullres','Dataset002_SEQAORTAS', 0,'ct'],
            #  ['3d_fullres','Dataset005_SEQAORTANDFEMOMR', 'all','mr', False],
            #  ['3d_fullres','Dataset006_SEQAORTANDFEMOCT', 'all','ct', False],
            #  ['3d_fullres','Dataset007_SEQPULMONARYMR', 'all','mr', False],
            #  ['3d_fullres','Dataset009_SEQAORTASMICCT', 'all','ct', True, '.nrrd'],
             ['3d_fullres','Dataset010_SEQCOROASOCACT', 'all','ct', True, '.nrrd'],
            ]

    calc_restults = False

    if calc_restults:
        global_dict             = {}
        global_dict['test']     = []
        global_dict['ct dice']  = []
        global_dict['mr dice']  = []
        global_dict['ct cent']  = []
        global_dict['mr cent']  = []

    # dir_output0 = 'output_enlarged_bb/'
    #dir_output0 = 'output_min_resolution/'
    #dir_output0 = 'output_min_resolution5/'
    dir_output0 = 'output_2000_steps/'

    dir_seg = True
    cropped_volume = False
    original = True # is this new vmr or old
    masked = False

    max_step_size  = 2000
    write_samples  = True
    retrace_cent   = False
    take_time      = False
    weighted = True
    calc_global_centerline = False

    for test in tests:

        print('\n test is: \n', test)
        dataset = test[1]
        fold = test[2]
        modality_model = test[3]
        json_file_present = test[4]
        img_format = test[5]
        test = test[0]

        ## Weight directory
        dir_model_weights = dataset+'/nnUNetTrainer__nnUNetPlans__'+test

        testing_samples, directory_data = get_testing_samples(dataset)
        print(f"Testing samples about to run: {testing_samples}")
        ct_dice, mr_dice, ct_cent, mr_cent = [],[],[],[]
        testing_samples_done = []

        final_dice_scores, final_perc_caught, final_tot_perc, final_missed_branches, final_n_steps_taken, final_ave_step_dice = [],[],[],[],[],[]
        
        for test_case in testing_samples:
            print(test_case)
            initial_seeds = []
            if json_file_present:
                ## Information
                modality = modality_model
                i = 0
                case = test_case['name']
                dir_image, dir_seg, dir_cent, dir_surf = get_directories(directory_data, case, img_format, dir_seg =False)
                dir_seg = None
                dir_output = dir_output0 +test+'_'+case+'/'
                ## Create directories for results
                create_directories(dir_output, write_samples)

                ## Seed points
                potential_branches = []

                if not test_case['seeds']:
                    print(f"No seed given, trying to get one from centerline ground truth")
                    old_seed, old_radius = vf.get_seed(dir_cent, 0, 150)
                    initial_seed, initial_radius = vf.get_seed(dir_cent, 0, 160)
                    init_step = create_step_dict(old_seed, old_radius, initial_seed, initial_radius, 0)
                    print(f"Seed found from centerline, took point nr {160}!")
                    print(f"Old seed: {old_seed}, {old_radius}")
                    print(f"Initial seed: {initial_seed}, {initial_radius} ")
                    initial_seeds.append(initial_seed)
                    potential_branches = [init_step]
                    if write_samples:
                        vf.write_geo(dir_output+ 'points/0_seed_point.vtp', vf.points2polydata([old_seed.tolist()]))
                else:
                    for seed in test_case['seeds']:
                        step  = create_step_dict(np.array(seed[0]), seed[2], np.array(seed[1]), seed[2], 0)
                        potential_branches.append(step)
                        initial_seeds.append(np.array(seed[1]))
                        if write_samples:
                            vf.write_geo(dir_output+ 'points/'+str(test_case['seeds'].index(seed))+'_oldseed_point.vtp', vf.points2polydata([seed[0]]))

            else:
                ## Information
                case = test_case[0]
                i = test_case[1]
                id_old = test_case[2]
                id_current = test_case[3]
                modality = test_case[4]

                if modality != modality_model: 
                    print(f"Modality doesnt match model and case: {modality},{modality_model} ")
                    continue
                else: testing_samples_done.append(test_case)
                print(test_case)

                dir_image, dir_seg, dir_cent, dir_surf = vmr_directories(directory_data, case, dir_seg, cropped_volume, original)
                dir_seg = None
                dir_output = dir_output0 +test+'_'+case+'_'+str(i)+'/'
                ## Create directories for results
                create_directories(dir_output, write_samples)

                ## Get inital seed point + radius
                old_seed, old_radius = vf.get_seed(dir_cent, i, id_old)
                print(old_seed)
                initial_seed, initial_radius = vf.get_seed(dir_cent, i, id_current)
                initial_seeds.append(initial_seed)
                if write_samples:
                    vf.write_geo(dir_output+ 'points/0_seed_point.vtp', vf.points2polydata([old_seed.tolist()]))
                init_step = create_step_dict(old_seed, old_radius, initial_seed, initial_radius, 0)
                potential_branches = [init_step]

            # print to .txt file all outputs
            sys.stdout = open(dir_output+"/out.txt", "w")
            print(test_case)
            print(f"Initial points: {potential_branches}")
            print(f"Time is: {time.time()}")

            ## Trace centerline
            centerlines, surfaces, points, assembly_obj, vessel_tree, n_steps_taken = trace_centerline( dir_output,
                                                                                                    dir_image,
                                                                                                    case,
                                                                                                    dir_model_weights,
                                                                                                    fold,
                                                                                                    modality,
                                                                                                    potential_branches,
                                                                                                    max_step_size,
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
            #sitk.WriteImage(assembly_org, dir_output+'/'+case+'_assembly_' + test +'_'+str(i)+'.mha')

            assembly = assembly_org
            name = 'original'
            #for assembly,name in zip([assembly_ups, assembly_org],['upsampled', 'original']):
            assembly_binary = sitk.BinaryThreshold(assembly, lowerThreshold=0.5, upperThreshold=1)
            sitk.WriteImage(assembly_binary, dir_output+'/'+case+'_seg_'+ test +'_'+str(i)+'.mha')
            
            assembly_binary = sf.keep_component_seeds(assembly_binary, initial_seeds)
            sitk.WriteImage(assembly_binary, dir_output+'/'+case+'_seg_rem_' + test +'_'+str(i)+'.mha')

            assembly_surface    = vf.evaluate_surface(assembly_binary, 1)
            vf.write_vtk_polydata(assembly_surface, dir_output+'/final_assembly_'+name+'_'+case+'_'+test +'_'+str(i)+'_'+str(n_steps_taken)+'_'+'_surface.vtp')
            for level in [10,40]:#range(10,50,10):
                surface_smooth      = vf.smooth_surface(assembly_surface, level)
                vf.write_vtk_polydata(surface_smooth, dir_output+'/final_assembly_'+name+'_'+case+'_'+test +'_'+str(i)+'_'+str(n_steps_taken)+'_'+str(level)+'_surface_smooth.vtp')
                #path = vmtkfs.calc_centerline(   surface_smooth, "pointlist", var_source=in_source, var_target=in_target)
                #vf.write_vtk_polydata(path, dir_output+'/final_assembly'+case+'_'+test +'_'+str(i)+'_'+str(n_steps_taken)+'_'+str(level)+'_centerline_smooth.vtp')


            #path1 = vmtkfs.calc_centerline(surface_smooth, "profileidlist", number = 0)
            #vf.write_vtk_polydata(path1, dir_output+'/final_assembly'+case+'_'+test +'_'+str(i)+'_'+str(n_steps_taken)+'_centerline_smooth1.vtp')


            final_surface = vf.appendPolyData(surfaces)
            final_centerline = vf.appendPolyData(centerlines)
            final_points = vf.appendPolyData(points)
            vf.write_vtk_polydata(final_surface,    dir_output+'/final_'+case+'_'+test +'_'+str(i)+'_'+str(n_steps_taken)+'_surfaces.vtp')
            vf.write_vtk_polydata(final_centerline, dir_output+'/final_'+case+'_'+test +'_'+str(i)+'_'+str(n_steps_taken)+'_centerlines.vtp')
            vf.write_vtk_polydata(final_points,     dir_output+'/final_'+case+'_'+test +'_'+str(i)+'_'+str(n_steps_taken)+'_points.vtp')

            #print('Number of outlets: ' + str(len(final_caps[1])))

            if calc_restults:
                evaluate_tracing = EvaluateTracing(case, initial_seed, dir_seg, dir_surf, dir_cent, assembly_binary, surface_smooth)
                missed_branches, perc_caught, total_perc = evaluate_tracing.count_branches()
                if dir_seg:
                    final_dice = evaluate_tracing.calc_dice_score()
                    #ave_dice = vessel_tree.calc_ave_dice()

                    if masked:
                        masked_dir = '/Users/numisveinsson/Downloads/tests_masks/test_global_masks/mask_'+case+'.vtk'
                        masked_dice = evaluate_tracing.masked_dice(masked_dir)
                        print(f"******* Masked dice: {masked_dice} **************")
                        final_dice = masked_dice


                #final_ave_step_dice.append(ave_dice)
                final_dice_scores.append(final_dice)
                final_n_steps_taken.append(n_steps_taken)
                final_perc_caught.append(perc_caught)
                final_tot_perc.append(total_perc)
                final_missed_branches.append(missed_branches)

                if modality == 'ct': ct_dice.append(final_dice)
                elif modality == 'mr': mr_dice.append(final_dice)
                if modality == 'ct': ct_cent.append(total_perc)
                elif modality == 'mr': mr_cent.append(total_perc)

        print("\nTotal calculation time is: " + str((time.time() - start_time)/60) + " min\n")
        if calc_restults:
            global_dict['test'].append(test)
            global_dict['ct dice'].append(np.array(ct_dice).mean())
            global_dict['mr dice'].append(np.array(mr_dice).mean())
            global_dict['ct cent'].append(np.array(ct_cent).mean())
            global_dict['mr cent'].append(np.array(mr_cent).mean())

            for i in range(len(testing_samples_done)):
                print(testing_samples_done[i][0])
                print('Steps taken: ', final_n_steps_taken[i])
                #print('Ave dice per step: ', final_ave_step_dice[i])
                print('Dice: ', final_dice_scores[i])
                print('Percent caught: ', final_perc_caught[i])
                print('Total centerline caught: ', final_tot_perc[i])
                print(str(final_missed_branches[i][0])+'/'+str(final_missed_branches[i][1])+' branches missed\n')

            now = datetime.now()
            dt_string = now.strftime("_%d_%m_%Y_%H_%M_%S")
            info_file_name = "info_"+test+".txt"
            f = open(dir_output0 +info_file_name,'a')
            for i in range(len(testing_samples_done)):
                f.write(testing_samples_done[i][0])
                f.write('\nSteps taken: ' + str(final_n_steps_taken[i]))
                #f.write('\nAve dice per step: ' + str(final_ave_step_dice[i]))
                f.write('\nDice: ' +str(final_dice_scores[i]))
                f.write('\nPercent caught: ' +str(final_perc_caught[i]))
                f.write('\nTotal cent caught: ' +str(final_tot_perc[i]))
                f.write('\n'+str(final_missed_branches[i][0])+'/'+str(final_missed_branches[i][1])+' branches missed\n')
            for key in global_dict.keys():
                f.write(f"\n{key}: {global_dict[key]}")
            f.close()
    if calc_restults:
        with open(dir_output0+'results.pickle', 'wb') as handle:
            pickle.dump(global_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # import pdb; pdb.set_trace()
    # with open('filename.pickle', 'rb') as handle:
    #     b = pickle.load(handle)
