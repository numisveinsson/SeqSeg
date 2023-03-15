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
from modules.vmr_data import vmr_directories
from modules.assembly import Segmentation, VesselTree, print_error, create_step_dict
from modules.evaluation import EvaluateTracing
from prediction import Prediction
from model import UNet3DIsensee

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



def trace_centerline(output_folder, image_file, case, model_folder, modality,
                    img_shape, threshold, potential_branches, max_step_size,
                    seg_file=None,      global_scale=False, write_samples=True,
                    take_time=False,    retrace_cent=False, weighted=False
                    ):

    allowed_steps = 10
    prevent_retracing = True
    volume_size_ratio = 5
    magnify_radius = 1
    number_chances = 2
    run_time = False
    use_buffer = True
    forceful_sidebranch = False
    forceful_sidebranch_magnify = 1.1

    if seg_file:
        reader_seg, origin_im, size_im, spacing_im = sf.import_image(seg_file)

    reader_im, origin_im, size_im, spacing_im = sf.import_image(image_file)

    init_step = potential_branches[0]
    vessel_tree   = VesselTree(case, image_file, init_step, potential_branches)
    assembly_segs = Segmentation(case, image_file, weighted)
    model         = UNet3DIsensee((img_shape[0], img_shape[1], img_shape[2], 1),
                                num_class=1)
    unet = model.build()
    model_name = os.path.realpath(model_folder) + '/weights_unet.hdf5'
    unet.load_weights(model_name)

    ## Note: make initial seed within loop
    initial_seed = assembly_segs.assembly.TransformPhysicalPointToIndex(vessel_tree.steps[0]['point'].tolist())
    branch = 0

    # Track combos of all polydata
    list_centerlines, list_surfaces, list_points = [], [], []
    dice_list, assd_list, haus_list = [], [], []
    surfaces_animation, cent_animation = [], []

    num_steps_direction = 0
    inside_branch = 0
    i = 0 # numbering chronological order
    while vessel_tree.potential_branches and i < max_step_size:

        if i in range(0, max_step_size, max_step_size//10):
            print(f"*** Step number {i} ***")

        try:

            start_time_loc = time.time()

            # Take next step
            step_seg = vessel_tree.steps[i]

            if prevent_retracing:
                if inside_branch == allowed_steps:
                    step_seg['is_inside'] = True
                    inside_branch = 0
                    #vessel_tree.remove_branch(branch)
                    #branch -= 1
                    #i = i - allowed_steps
                    print('\n \n Inside already segmented vessel!! \n \n')

                    if write_samples:
                        polydata_point = vf.points2polydata([step_seg['point'].tolist()])
                        pfn = output_folder + 'points/inside_point_'+case+'_'+str(i)+'.vtp'
                        vf.write_geo(pfn, polydata_point)

                    print(error)
                elif vf.is_point_in_image(assembly_segs.assembly, step_seg['point']): #+ step_seg['radius']*step_seg['tangent']):
                    inside_branch += 1
                else:
                    inside_branch = 0
            #print('\n The inside branch is ', inside_branch)

            # Point
            polydata_point = vf.points2polydata([step_seg['point'].tolist()])

            if write_samples:
                pfn = output_folder + 'points/point_'+case+'_'+str(i)+'.vtp'
                vf.write_geo(pfn, polydata_point)

            perc, mag = 1,1
            # while perc>0.33 and mag < 1.5:
            # Extract Volume
            size_extract, index_extract = sf.map_to_image(  step_seg['point'],
                                                            step_seg['radius']*mag,
                                                            volume_size_ratio,
                                                            origin_im,
                                                            spacing_im          )
            step_seg['img_index'] = index_extract
            step_seg['img_size'] = size_extract
            cropped_volume = sf.extract_volume(reader_im, index_extract, size_extract)
            volume_fn = output_folder +'volumes/volume_'+case+'_'+str(i)+'.vtk'

            if seg_file:
                seg_volume = sf.extract_volume(reader_seg, index_extract, size_extract)
                seg_fn = output_folder +'volumes/volume_'+case+'_'+str(i)+'_truth.vtk'
            else:
                seg_volume=None

            step_seg['img_file'] = volume_fn
            if write_samples:
                sitk.WriteImage(cropped_volume, volume_fn)
                if seg_file:
                    sitk.WriteImage(seg_volume, seg_fn)
            if take_time:
                print("\n Extracting and writing volumes: " + str(time.time() - start_time_loc) + " s\n")
            if run_time:
                step_seg['time']=[]
                step_seg['time'].append(time.time()-start_time_loc)
                start_time_loc = time.time()

            # Prediction
            predict = Prediction(   unet,
                                    model_name,
                                    modality,
                                    cropped_volume,
                                    img_shape,
                                    output_folder+'predictions',
                                    threshold,
                                    seg_volume,
                                    global_scale)
            predict.volume_prediction(1)
            predict.resample_prediction()
            if seg_file:
                d = predict.dice()
                #print(f"Local dice: {d:.4f}")
                step_seg['dice'] = d
                dice_list.append(d)

            predicted_vessel = predict.prediction
            # perc = sitk.GetArrayFromImage(predicted_vessel).mean()
            # mag = mag+0.1
            #print(f"Perc as 1: {perc:.3f}")

            step_seg['prob_predicted_vessel'] = predict.prob_prediction
            seed = np.rint(np.array(size_extract)/2).astype(int).tolist()
            predicted_vessel = sf.remove_other_vessels(predicted_vessel, seed)

            #print("Now the components are: ")
            #labels, means = sf.connected_comp_info(predicted_vessel, True)
            pd_fn = output_folder +'predictions/seg_'+case+'_'+str(i)+'.vtk'
            if take_time:
                print("\n Prediction, forward pass: " + str(time.time() - start_time_loc) + " s\n")
            if run_time:
                step_seg['time'].append(time.time()-start_time_loc)
                start_time_loc = time.time()

            # Surface

            # if seg_file:
            #     surface = vf.evaluate_surface(predict.seg_vol)
            #     if write_samples:
            #         vf.write_vtk_polydata(surface_smooth, seg_fn)

            surface = vf.evaluate_surface(predicted_vessel) # Marching cubes
            surface_smooth = vf.smooth_surface(surface, 8) # Smooth marching cubes

            vtkimage = vf.exportSitk2VTK(cropped_volume)
            length = predicted_vessel.GetSize()[0]*predicted_vessel.GetSpacing()[0]
            surface_smooth = vf.bound_polydata_by_image(vtkimage[0], surface_smooth, length*1/20)

            #surface_smooth = vf.get_largest_connected_polydata(surface_smooth)

            if take_time:
                print("\n Calc and smooth surface: " + str(time.time() - start_time_loc) + " s\n")
            if run_time:
                step_seg['time'].append(time.time()-start_time_loc)
                start_time_loc = time.time()

            sfn = output_folder +'surfaces/surf_'+case+'_'+str(i)+'.vtp'
            cfn = output_folder +'centerlines/cent_'+case+'_'+str(i)+'.vtp'
            if write_samples:
                sitk.WriteImage(predicted_vessel, pd_fn)
                vf.write_vtk_polydata(surface_smooth, sfn)
            step_seg['seg_file'] = pd_fn
            step_seg['surf_file'] = sfn
            step_seg['surface'] = surface_smooth


            if i != 0:
                prev_step = vessel_tree.get_previous_step(i)
                old_point_ref = prev_step['old point']
            elif i == 0:
                old_point_ref = step_seg['old point']
            caps = vf.calc_caps(surface_smooth)
            step_seg['caps'] = caps
            _ , source_id = vf.orient_caps(caps, old_point_ref)


            #print('Number of caps: ', len(caps))
            if len(caps) < 2: print(error)

            # polydata_point = vf.points2polydata([step_seg['caps'][0]])
            # pfn = '/Users/numisveinsson/Downloads/point.vtp'
            # vf.write_geo(pfn, polydata_point)

            # Centerline
            centerline_poly = vmtkfs.calc_centerline(   surface_smooth,
                                                            "profileidlist",
                                                            var_source=[source_id],
                                                            number = i)
            #centerline_poly = vmtkfs.calc_centerline(surface_smooth, "profileidlist", number = i)
            #centerline_poly = vf.get_largest_connected_polydata(centerline_poly)
            step_seg['cent_file'] = cfn
            if centerline_poly.GetNumberOfPoints() < 5:
                #print("\n Attempting with more smoothing and cropping \n")
                surface_smooth1 = vf.smooth_surface(surface, 12)
                surface_smooth1 = vf.bound_polydata_by_image(vtkimage[0], surface_smooth, length*1/10)
                centerline_poly1 = vmtkfs.calc_centerline(surface_smooth1, "profileidlist")
                if centerline_poly1.GetNumberOfPoints() > 5:
                    sfn = output_folder +'surfaces/surf_'+case+'_'+str(i)+'_1.vtp'
                    surface_smooth = surface_smooth1
                    cfn = output_folder +'centerlines/cent_'+case+'_'+str(i)+'_1.vtp'
                    centerline_poly = centerline_poly1

            if write_samples:
                vmtkfs.write_centerline(centerline_poly, cfn)
            if take_time:
                print("\n Calc centerline: " + str(time.time() - start_time_loc) + " s\n")
            if run_time:
                step_seg['time'].append(time.time()-start_time_loc)
                start_time_loc = time.time()

            # Save step information
            step_seg['point_pd'] = polydata_point
            step_seg['surface'] = surface_smooth
            step_seg['centerline'] = centerline_poly

            # Assembly
            N = 10
            buffer = 10
            if use_buffer:
                if len(vessel_tree.steps) % N == 0 and len(vessel_tree.steps) >= (N+buffer):
                    for j in range(1,N+1):
                        if vessel_tree.steps[-(j+buffer)]['prob_predicted_vessel']:
                            #print(f"Adding step {-(j+buffer)} and number of steps are {len(vessel_tree.steps)}")
                            assembly_segs.add_segmentation( vessel_tree.steps[-(j+buffer)]['prob_predicted_vessel'],
                                                            vessel_tree.steps[-(j+buffer)]['img_index'],
                                                            vessel_tree.steps[-(j+buffer)]['img_size'], (1/vessel_tree.steps[-(j+buffer)]['radius'])**2)
                            vessel_tree.steps[-(j+buffer)]['prob_predicted_vessel'] = None
                            vessel_tree.caps = vessel_tree.caps + vessel_tree.steps[-(j+buffer)]['caps']
                    if len(vessel_tree.steps) % (N*5) == 0 and write_samples:
                            sitk.WriteImage(assembly_segs.assembly, output_folder +'assembly/assembly_'+case+'_'+str(i)+'.vtk')
                            assembly = sitk.BinaryThreshold(assembly_segs.assembly, lowerThreshold=0.5, upperThreshold=1)
                            assembly = sf.remove_other_vessels(assembly, initial_seed)
                            surface_assembly = vf.evaluate_surface(assembly, 1)
                            vf.write_vtk_polydata(surface_assembly, output_folder +'assembly/assembly_surface_'+case+'_'+str(i)+'.vtp')

                if take_time:
                        print("\n Adding to seg volume: " + str(time.time() - start_time_loc) + " s\n")
                if run_time:
                    step_seg['time'].append(time.time()-start_time_loc)
                    start_time_loc = time.time()
            else:
                assembly_segs.add_segmentation(step_seg['prob_predicted_vessel'], step_seg['img_index'], step_seg['img_size'], step_seg['radius'])
                step_seg['prob_predicted_vessel'] = None

            # Print polydata surfaces,cents together for animation
            if write_samples:
                print('No animation')
                #surfaces_animation.append(surface_smooth)
                #surface_accum = vf.appendPolyData(surfaces_animation)
                #vf.write_vtk_polydata(surface_accum, dir_output+'animation/animation_'+str(i).zfill(3)+'.vtp')
                # cent_animation.append(centerline_poly)
                # cent_accum = vf.appendPolyData(cent_animation)
                # vf.write_vtk_polydata(cent_accum, dir_output+'animation/animation_'+str(2*i+1).zfill(3)+'.vtp')

            point_tree, radius_tree, angle_change = vf.get_next_points( centerline_poly,
                                                                        step_seg['point'],
                                                                        step_seg['old point'],
                                                                        step_seg['old radius'])

            if take_time:
                print("\n Calc next point: " + str(time.time() - start_time_loc) + " s\n")
            if run_time:
                step_seg['time'].append(time.time()-start_time_loc)
                start_time_loc = time.time()

            vessel_tree.steps[i] = step_seg

            if point_tree.size != 0:

                i += 1
                num_steps_direction += 1
                next_step = create_step_dict(   step_seg['point'],
                                                step_seg['radius'],
                                                point_tree[0],
                                                radius_tree[0]*magnify_radius,
                                                angle_change[0])
                #print("Next radius is: " + str(radius_tree[0]*magnify_radius))
                vessel_tree.add_step(i, next_step, branch)

                if len(radius_tree) > 1:
                    # print('\n _ \n')
                    # print("\n BIFURCATION BIFURCATION BIFURCATION\n")
                    # print('\n _ \n')
                    for j in range(1, len(radius_tree)):
                        dict = create_step_dict(step_seg['point'],
                                                step_seg['radius'],
                                                point_tree[j],
                                                radius_tree[j]*magnify_radius,
                                                angle_change[j])
                        dict['connection'] = [branch, i-1]
                        if forceful_sidebranch:
                            dict['point'] += dict['radius']*dict['tangent']
                            dict['radius'] = dict['radius']*forceful_sidebranch_magnify

                        vessel_tree.potential_branches.append(dict)
                    list_points = []
                    for pot in point_tree:
                        list_points.append(vf.points2polydata([pot.tolist()]))
                    final_pot = vf.appendPolyData(list_points)
                    if write_samples:
                        vf.write_vtk_polydata(  final_pot,
                                                output_folder+'/points/bifurcation_'+case+'_'+str(branch)+'_'+str(i-1)+'_points.vtp')
            else:
                print('point_tree.size is 0')
                print(error)
            # print("\n This location done: " + str(time.time() - start_time_loc) + " s\n")


        except Exception as e:
            print(e)

            if step_seg['seg_file']:
                print_error(output_folder, i, step_seg, cropped_volume, predicted_vessel)
            elif step_seg['img_file']:
                print_error(output_folder, i, step_seg, cropped_volume)

            if i == 0:
                print("Didnt work for first surface")
                #break

            if vessel_tree.steps[i]['chances'] < number_chances and not step_seg['is_inside']:
                print("Giving chance for surface: " + str(i))
                if take_time:
                    print('Radius is ', step_seg['radius'])
                vessel_tree.steps[i]['point'] = vessel_tree.steps[i]['point'] + vessel_tree.steps[i]['radius']*vessel_tree.steps[i]['tangent']
                vessel_tree.steps[i]['chances'] += 1

            else:

                #print("\n*** Error for surface: \n" + str(i))
                del vessel_tree.branches[branch][-1]
                print("\n Moving onto another branch")

                list_surf_branch, list_cent_branch, list_pts_branch = [], [], []
                for id in vessel_tree.branches[branch][1:]:
                    list_surf_branch.append(vessel_tree.steps[id]['surface'])
                    list_cent_branch.append(vessel_tree.steps[id]['centerline'])
                    list_pts_branch.append(vessel_tree.steps[id]['point_pd'])
                    del vessel_tree.steps[id]['surface']
                    del vessel_tree.steps[id]['centerline']
                    del vessel_tree.steps[id]['point_pd']
                list_centerlines.extend(list_cent_branch)
                list_surfaces.extend(list_surf_branch)
                list_points.extend(list_pts_branch)

                #print('Printing potentials')
                list_pot = []
                for pot in vessel_tree.potential_branches:
                    list_pot.append(vf.points2polydata([pot['point'].tolist()]))
                final_pot = vf.appendPolyData(list_pot)

                if take_time:
                    print("Branches are: ", vessel_tree.branches)
                if write_samples:
                    final_surface    = vf.appendPolyData(list_surf_branch)
                    final_centerline = vf.appendPolyData(list_cent_branch)
                    final_points     = vf.appendPolyData(list_pts_branch)
                    vf.write_vtk_polydata(final_pot, output_folder+'/assembly/potentials_'+case+'_'+str(branch)+'_'+str(i)+'_points.vtp')
                    vf.write_vtk_polydata(final_surface, output_folder+'/assembly/branch_'+case+'_'+str(branch)+'_'+str(i)+'_surfaces.vtp')
                    vf.write_vtk_polydata(final_centerline, output_folder+'/assembly/branch_'+case+'_'+str(branch)+'_'+str(i)+'_centerlines.vtp')
                    vf.write_vtk_polydata(final_points, output_folder+'/assembly/branch_'+case+'_'+str(branch)+'_'+str(i)+'_points.vtp')

                vessel_tree.caps = vessel_tree.caps + [step_seg['point'] + volume_size_ratio*step_seg['radius']*step_seg['tangent']]

                if retrace_cent:
                    ind = vessel_tree.branches[branch][1] # second step on this branch
                    step_to_add = vessel_tree.steps[ind]
                    step_to_add['connection'] = [-branch+1, ind] # add connection that says retrace
                    vessel_tree.potential_branches.append(step_to_add)
                # Sort the potentials by radius
                vessel_tree.sort_potential()

                if len(vessel_tree.potential_branches) == 1:
                    break
                next_step = vessel_tree.potential_branches.pop(1)
                if next_step['radius'] == vessel_tree.steps[0]['radius']:
                    next_step = vessel_tree.potential_branches.pop(1)

                branch += 1
                vessel_tree.add_branch(next_step['connection'][1], i)
                vessel_tree.steps[i] = next_step
                num_steps_direction = 0
                if take_time:
                    print("Post Branches are: ", vessel_tree.branches)
                    print("Number of steps are: ", len(vessel_tree.steps))
                    print("Connections of branches are: ", vessel_tree.bifurcations)
                    print("Number of potentials left are: ", len(vessel_tree.potential_branches))
    if len(vessel_tree.potential_branches) > 0:
        print('Printing potentials')
        list_pot = []
        for pot in vessel_tree.potential_branches:
            list_pot.append(vf.points2polydata([pot['point'].tolist()]))
            #vessel_tree.caps = vessel_tree.caps + [pot['point']+ volume_size_ratio*pot['radius']*pot['tangent']]
        final_pot = vf.appendPolyData(list_pot)
        vf.write_vtk_polydata(final_pot, output_folder+'/assembly/potentials_'+case+'_'+str(branch)+'_'+str(i)+'_points.vtp')

    # if use_buffer:
    #     if len(vessel_tree.steps) % N == 0 and len(vessel_tree.steps) >= (N+buffer):
    #         for j in range(N):
    #             if vessel_tree.steps[-(j+buffer)]['prob_predicted_vessel']:
    #                 assembly_segs.add_segmentation(vessel_tree.steps[-(j+buffer)]['prob_predicted_vessel'], vessel_tree.steps[-(j+buffer)]['img_index'], vessel_tree.steps[-(j+buffer)]['img_size'])
    #                 vessel_tree.steps[-(j+buffer)]['prob_predicted_vessel'] = None
    #
    return list_centerlines, list_surfaces, list_points, assembly_segs, vessel_tree, i

if __name__=='__main__':
    #       [name    , global_scale]
    tests = [
             ['test53',True, 'mr'],
            #['test54',True],
            ['test55',True, 'ct'],
            #['test56',False],
            # ['test57',False, 'ct'],
            # ['test58',False, 'mr']
            ]# 'test49', 'test27']

    global_dict                  = {}
    global_dict['test']          = []
    global_dict['ct dice']       = []
    global_dict['mr dice']       = []
    global_dict['ct cent']       = []
    global_dict['mr cent']       = []

    dir_output0    = '//Users/numisveinsson/Documents_numi/Automatic_Centerline_Data/outputs/'
    directory_data = '/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/vascular_data_3d/'
    max_step_size  = 700
    nn_input_shape = [64, 64, 64] # Input shape for NN
    threshold      = 0.5 # Threshold for binarization of prediction
    write_samples  = False
    retrace_cent = False
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
                           # ['0001_0001',0,30,50,'ct']
                           # ['0005_1001',0,300,320,'ct']  ,
                            #['0005_1001',1,200,220,'ct']  ,
                           ['0146_1001',0,10,20,'ct'],
                           ['0006_0001',0,10,20,'mr'],
                           ['0063_1001',0,10,20,'mr'],
                           ['0176_0000',0,10,20,'ct'],
                           ['0141_1001',0,10,20,'ct'],
                           ['0090_0001',0,10,20,'mr']
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

            dir_image, dir_seg, dir_cent, dir_surf = vmr_directories(directory_data, case, global_scale)
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
                for level in range(10,50,10):
                    surface_smooth      = vf.smooth_surface(assembly_surface, level)
                    vf.write_vtk_polydata(surface_smooth, dir_output+'/final_assembly_'+name+'_'+case+'_'+test +'_'+str(i)+'_'+str(max_step_size)+'_'+str(level)+'_surface_smooth.vtp')

            final_surface = vf.appendPolyData(surfaces)
            final_centerline = vf.appendPolyData(centerlines)
            final_points = vf.appendPolyData(points)
            vf.write_vtk_polydata(final_surface,    dir_output+'/final_'+case+'_'+test +'_'+str(i)+'_'+str(max_step_size)+'_surfaces.vtp')
            vf.write_vtk_polydata(final_centerline, dir_output+'/final_'+case+'_'+test +'_'+str(i)+'_'+str(max_step_size)+'_centerlines.vtp')
            vf.write_vtk_polydata(final_points,     dir_output+'/final_'+case+'_'+test +'_'+str(i)+'_'+str(max_step_size)+'_points.vtp')

            #print('Number of outlets: ' + str(len(final_caps[1])))

            evaluate_tracing = EvaluateTracing(case, initial_seed, dir_seg, dir_surf, dir_cent, assembly_binary, surface_smooth)
            missed_branches, perc_caught, total_perc = evaluate_tracing.count_branches()
            final_dice = evaluate_tracing.calc_dice_score()
            ave_dice = vessel_tree.calc_ave_dice()

            masked_dir = '/Users/numisveinsson/Downloads/tests_masks/test_global_masks/mask_'+case+'.vtk'
            masked_dice = evaluate_tracing.masked_dice(masked_dir)
            print(f"******* Masked dice: {masked_dice} **************")
            final_dice = masked_dice

            final_ave_step_dice.append(ave_dice)
            final_n_steps_taken.append(n_steps_taken)
            final_dice_scores.append(final_dice)
            final_perc_caught.append(perc_caught)
            final_tot_perc.append(total_perc)
            final_missed_branches.append(missed_branches)

            if modality == 'ct': ct_dice.append(final_dice)
            elif modality == 'mr': mr_dice.append(final_dice)
            if modality == 'ct': ct_cent.append(total_perc)
            elif modality == 'mr': mr_cent.append(total_perc)

            # final_centerline_poly = vmtkfs.calc_centerline(assembly_surface, "pointlist", final_caps[0], final_caps[1], number = 0)
            # vf.write_vtk_polydata(path, dir_output+'/final_assembly'+case+'_'+test +'_'+str(i)+'_'+str(max_step_size)+'_centerline_from_caps.vtp')

        global_dict['test'].append(test)
        global_dict['ct dice'].append(np.array(ct_dice).mean())
        global_dict['mr dice'].append(np.array(mr_dice).mean())
        global_dict['ct cent'].append(np.array(ct_cent).mean())
        global_dict['mr cent'].append(np.array(mr_cent).mean())

        print("\nTotal calculation time is: " + str((time.time() - start_time)/60) + " min\n")
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

    with open(dir_output0+'results.pickle', 'wb') as handle:
        pickle.dump(global_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    import pdb; pdb.set_trace()
    with open('filename.pickle', 'rb') as handle:
        b = pickle.load(handle)

    path = vmtkfs.calc_centerline(surface_smooth, "pickpoint", number = 0)
    vf.write_vtk_polydata(path, dir_output+'/final_assembly'+case+'_'+test +'_'+str(i)+'_'+str(max_step_size)+'_centerline_smooth.vtp')
