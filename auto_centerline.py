import faulthandler

faulthandler.enable()

#import warnings
#warnings.filterwarnings('ignore')

import time
start_time = time.time()
from datetime import datetime

import os
import pickle
import numpy as np
import SimpleITK as sitk

from modules import sitk_functions as sf
from modules import vtk_functions as vf
from modules import vmtk_functions as vmtkfs
from modules.tracing import trace_centerline
from modules.vmr_data import vmr_directories
from modules.assembly import create_step_dict
from modules.evaluation import EvaluateTracing

# sys.path.append('/Users/numisveinsson/SimVascular/Python/site-packages/sv_1d_simulation')
# from sv_1d_simulation import centerlines
        # cl = centerlines.Centerlines()
        # cl.extract_center_lines(params)
        # cl.extract_branches(params)
        # cl.write_outlet_face_names(params)

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


if __name__=='__main__':
    #       [name    , global_scale]
    tests = [
             ['test55',True, 'ct'],
             ['test53',True, 'mr'],
            #['test54',True],

            #['test56',False],
            # ['test57',False, 'ct'],
            # ['test58',False, 'mr']
            ]# 'test49', 'test27']

    calc_restults = True

    if calc_restults:
        global_dict             = {}
        global_dict['test']     = []
        global_dict['ct dice']  = []
        global_dict['mr dice']  = []
        global_dict['ct cent']  = []
        global_dict['mr cent']  = []

    dir_output0    = '//Users/numisveinsson/Documents_numi/Automatic_Centerline_Data/outputs_new/'
    directory_data = '/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/vascular_data_3d/'
    directory_data = '/Users/numisveinsson/Documents_numi/vmr_data_new/'
    dir_seg = True
    cropped_volume = False
    original = False # is this new vmr or old
    masked = False

    max_step_size  = 20
    nn_input_shape = [64, 64, 64] # Input shape for NN
    threshold      = 0.5 # Threshold for binarization of prediction
    write_samples  = True
    retrace_cent   = False
    take_time      = False
    weighted = True

    for test in tests:

        global_scale = test[1]
        modality_model = test[2]
        test         = test[0]
        print('\n test is: \n', test)
        ## Weight directory
        dir_model_weights = '/Users/numisveinsson/Documents_numi/Automatic_Centerline_Data/weights/' + test + '/'

        testing_samples = [#['0002_0001',0,150,170,'ct']  ,
                           # ['0002_0001',1,150, 170,'ct'] ,
                           #['0001_0001',0,30,50,'ct'],
                           #['0001_0001',7,30,50,'ct'],
                           #['0001_0001',8,110,130,'ct'],
                           # ['0005_1001',0,300,320,'ct']  ,
                           #  ['0005_1001',1,200,220,'ct']  ,
                           # ['0146_1001',0,10,20,'ct'],
                           # ['0006_0001',0,10,20,'mr'],
                           # ['0063_1001',0,10,20,'mr'],
                           # ['0176_0000',0,10,20,'ct'],
                           # ['0141_1001',0,10,20,'ct'],
                           # ['0090_0001',0,10,20,'mr'],
                           # ['0108_0001_aorta',0,10,20,'ct'],
                           ['0183_1002_aorta',3,-1,-2,'ct'],
                           # ['0184_0001_aorta',3,-1,-2,'ct'],
                           #['0188_0001_aorta',5,-1,-10,'ct'],
                           #['0189_0001_aorta',0,0,1,'ct'],

                           # ['KDR08_aorta',0,10,20,'mr'],
                           # ['KDR10_aorta',0,10,20,'mr'],
                           # ['KDR12_aorta',0,10,20,'mr'],
                           # ['KDR13_aorta',0,10,20,'mr'],
                           # ['KDR32_aorta',0,10,20,'mr'],
                           # ['KDR33_aorta',3,-10,-20,'mr'],
                           # ['KDR34_aorta',0,10,20,'mr'],
                           # ['KDR48_aorta',0,10,20,'mr'],
                           # ['KDR57_aorta',4,-10,-20,'mr'],
                           # ['O0171SC_aorta',0,10,20,'ct'],
                           # ['O6397SC_aorta',0,10,20,'ct'],
                           # ['O8693SC_aorta',0,10,20,'ct'],
                           # ['O344211000_2006_aorta',0,10,20,'ct'],
                           # ['O11908_aorta',0,10,20,'ct'],
                           # ['O20719_2006_aorta',0,10,20,'ct'],
                           # these are left:
                           # ['O51001_2009_aorta',0,10,20,'ct'],
                           # ['O128301_2008_aorta',0,10,20,'ct'],
                           # ['O145207_aorta',0,10,20,'ct'],
                           # ['O150323_2009_aorta',0,10,20,'ct'],
                           # ['O227241_2006_aorta',0,10,20,'ct'],
                           # ['O351095_aorta',0,10,20,'ct'],
                           # ['O690801_2007_aorta',0,10,20,'ct'],

                           ]

        ct_dice, mr_dice, ct_cent, mr_cent = [],[],[],[]
        testing_samples_done = []

        final_dice_scores, final_perc_caught, final_tot_perc, final_missed_branches, final_n_steps_taken, final_ave_step_dice = [],[],[],[],[],[]
        for test_case in testing_samples:

            ## Information
            case = test_case[0]
            i = test_case[1]
            id_old = test_case[2]
            id_current = test_case[3]
            modality = test_case[4]

            if modality != modality_model: continue
            else: testing_samples_done.append(test_case)
            print(test_case)

            dir_image, dir_seg, dir_cent, dir_surf = vmr_directories(directory_data, case, global_scale, dir_seg, cropped_volume, original)
            dir_output = dir_output0 +test+'_'+case+'_'+str(i)+'/'
            ## Create directories for results
            create_directories(dir_output, write_samples)

            ## Get inital seed point + radius
            old_seed, old_radius = vf.get_seed(dir_cent, i, id_old)
            print(old_seed)
            initial_seed, initial_radius = vf.get_seed(dir_cent, i, id_current)
            if write_samples:
                vf.write_geo(dir_output+ 'points/0_seed_point.vtp', vf.points2polydata([old_seed.tolist()]))
            init_step = create_step_dict(old_seed, old_radius, initial_seed, initial_radius, 0)
            potential_branches = [init_step]
            ## Trace centerline
            centerlines, surfaces, points, assembly_obj, vessel_tree, n_steps_taken = trace_centerline( dir_output,
                                                                                                    dir_image,
                                                                                                    case,
                                                                                                    dir_model_weights,
                                                                                                    modality,
                                                                                                    nn_input_shape,
                                                                                                    threshold,
                                                                                                    potential_branches,
                                                                                                    max_step_size,
                                                                                                    dir_seg,
                                                                                                    global_scale,
                                                                                                    write_samples,
                                                                                                    take_time,
                                                                                                    retrace_cent,
                                                                                                    weighted)

            print("\nTotal calculation time is: " + str((time.time() - start_time)/60) + " min\n")

            if take_time:
                vessel_tree.time_analysis()

            ## Assembly work
            assembly_org = assembly_obj.assembly
            assembly_ups = assembly_obj.upsample_sitk()
            print("\nTotal calculation time is: " + str((time.time() - start_time)/60) + " min\n")
            sitk.WriteImage(assembly_org, dir_output+'/final_assembly_'+case+'_'+test +'_'+str(i)+'.vtk')

            for assembly,name in zip([assembly_ups, assembly_org],['upsampled', 'original']):
                assembly_binary     = sitk.BinaryThreshold(assembly, lowerThreshold=0.5, upperThreshold=1)
                if name == 'original':
                    seed = assembly.TransformPhysicalPointToIndex(initial_seed.tolist())
                    assembly_binary     = sf.remove_other_vessels(assembly_binary, seed)
                assembly_surface    = vf.evaluate_surface(assembly_binary, 1)
                vf.write_vtk_polydata(assembly_surface, dir_output+'/final_assembly_'+name+'_'+case+'_'+test +'_'+str(i)+'_'+str(max_step_size)+'_'+'_surface.vtp')
                for level in [10,40]:#range(10,50,10):
                    surface_smooth      = vf.smooth_surface(assembly_surface, level)
                    vf.write_vtk_polydata(surface_smooth, dir_output+'/final_assembly_'+name+'_'+case+'_'+test +'_'+str(i)+'_'+str(max_step_size)+'_'+str(level)+'_surface_smooth.vtp')
            import pdb; pdb.set_trace()
            end_points = vessel_tree.get_end_points()
            in_source = end_points[0].tolist()
            in_target_lists = [point.tolist() for point in end_points[1:]]
            in_target = []
            for target in in_target_lists:
                in_target += target
            path = vmtkfs.calc_centerline(   surface_smooth, "pointlist", var_source=in_source, var_target=in_target)
            vf.write_vtk_polydata(path, dir_output+'/final_assembly'+case+'_'+test +'_'+str(i)+'_'+str(max_step_size)+'_centerline_smooth.vtp')
            path1 = vmtkfs.calc_centerline(surface_smooth, "profileidlist", number = 0)
            vf.write_vtk_polydata(path1, dir_output+'/final_assembly'+case+'_'+test +'_'+str(i)+'_'+str(max_step_size)+'_centerline_smooth1.vtp')




            final_surface = vf.appendPolyData(surfaces)
            final_centerline = vf.appendPolyData(centerlines)
            final_points = vf.appendPolyData(points)
            vf.write_vtk_polydata(final_surface,    dir_output+'/final_'+case+'_'+test +'_'+str(i)+'_'+str(max_step_size)+'_surfaces.vtp')
            vf.write_vtk_polydata(final_centerline, dir_output+'/final_'+case+'_'+test +'_'+str(i)+'_'+str(max_step_size)+'_centerlines.vtp')
            vf.write_vtk_polydata(final_points,     dir_output+'/final_'+case+'_'+test +'_'+str(i)+'_'+str(max_step_size)+'_points.vtp')

            #print('Number of outlets: ' + str(len(final_caps[1])))

            if calc_restults:
                evaluate_tracing = EvaluateTracing(case, initial_seed, dir_seg, dir_surf, dir_cent, assembly_binary, surface_smooth)
                missed_branches, perc_caught, total_perc = evaluate_tracing.count_branches()
                if dir_seg:
                    final_dice = evaluate_tracing.calc_dice_score()
                    ave_dice = vessel_tree.calc_ave_dice()

                    if masked:
                        masked_dir = '/Users/numisveinsson/Downloads/tests_masks/test_global_masks/mask_'+case+'.vtk'
                        masked_dice = evaluate_tracing.masked_dice(masked_dir)
                        print(f"******* Masked dice: {masked_dice} **************")
                        final_dice = masked_dice


                final_ave_step_dice.append(ave_dice)
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
                print('Ave dice per step: ', final_ave_step_dice[i])
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
                f.write('\nAve dice per step: ' + str(final_ave_step_dice[i]))
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

    import pdb; pdb.set_trace()
    with open('filename.pickle', 'rb') as handle:
        b = pickle.load(handle)
