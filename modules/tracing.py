import time
import numpy as np
import SimpleITK as sitk

from .sitk_functions import *
from .vtk_functions import *
from .vmtk_functions import *
from .assembly import Segmentation, VesselTree, print_error, create_step_dict
#from .model import UNet3DIsensee
#from .prediction import Prediction

import sys
sys.path.append("/global/scratch/users/numi/SeqSeg/nnUNet/")

from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

import pdb
def trace_centerline(output_folder, image_file, case, model_folder, fold, modality,
                    potential_branches, max_step_size,
                    seg_file=None,    write_samples=True,
                    take_time=False,    retrace_cent=False, weighted=False
                    ):
    animation = True

    allowed_steps = 10
    prevent_retracing = True
    volume_size_ratio = 5
    magnify_radius = 1
    number_chances = 2
    run_time = False
    use_buffer = True
    forceful_sidebranch = False
    forceful_sidebranch_magnify = 1.1

    #Assembly params
    N = 10
    buffer = 10

    if seg_file:
        reader_seg, origin_im, size_im, spacing_im = import_image(seg_file)

    reader_im, origin_im, size_im, spacing_im = import_image(image_file)
    print(f"Image data. size: {size_im}, spacing: {spacing_im}, origin: {origin_im}")

    init_step = potential_branches[0]
    vessel_tree   = VesselTree(case, image_file, init_step, potential_branches)
    assembly_segs = Segmentation(case, image_file, weighted)

    print('About to load predictor object')
    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True,
        device=torch.device('cpu', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    print('About to load model')
    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, model_folder),
        use_folds=(fold,),
        checkpoint_name='checkpoint_best.pth',
    )
    print('Done loading model, ready to predict')

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
    while vessel_tree.potential_branches and i < (max_step_size +1):

        if i in range(0, max_step_size, max_step_size*0 +1):
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
                        polydata_point = points2polydata([step_seg['point'].tolist()])
                        pfn = output_folder + 'points/inside_point_'+case+'_'+str(i)+'.vtp'
                        write_geo(pfn, polydata_point)

                    print(error)
                elif is_point_in_image(assembly_segs.assembly, step_seg['point']): #+ step_seg['radius']*step_seg['tangent']):
                    inside_branch += 1
                else:
                    inside_branch = 0
            #print('\n The inside branch is ', inside_branch)

            # Point
            polydata_point = points2polydata([step_seg['point'].tolist()])

            if write_samples:
                pfn = output_folder + 'points/point_'+case+'_'+str(i)+'.vtp'
                write_geo(pfn, polydata_point)

            # perc = 1
            mag = 1
            # while perc>0.42 and mag < 1.3:

            # Extract Volume
            size_extract, index_extract = map_to_image(  step_seg['point'],
                                                            step_seg['radius']*mag,
                                                            volume_size_ratio,
                                                            origin_im,
                                                            spacing_im,
                                                            size_im)
            step_seg['img_index'] = index_extract
            step_seg['img_size'] = size_extract
            cropped_volume = extract_volume(reader_im, index_extract, size_extract)
            volume_fn = output_folder +'volumes/volume_'+case+'_'+str(i)+'.mha'

            if seg_file:
                seg_volume = extract_volume(reader_seg, index_extract, size_extract)
                seg_fn = output_folder +'volumes/volume_'+case+'_'+str(i)+'_truth.mha'
            else:
                seg_volume=None

            step_seg['img_file'] = volume_fn
            if write_samples:
                sitk.WriteImage(cropped_volume, volume_fn)
                # if seg_file:
                #     sitk.WriteImage(seg_volume, seg_fn)
            if take_time:
                print("\n Extracting and writing volumes: " + str(time.time() - start_time_loc) + " s\n")
            if run_time:
                step_seg['time']=[]
                step_seg['time'].append(time.time()-start_time_loc)
                start_time_loc = time.time()

            # Prediction
            spacing = spacing_im.tolist()
            spacing = spacing[::-1]
            props={}
            props['spacing'] = spacing
            img_np = sitk.GetArrayFromImage(cropped_volume)
            img_np = img_np[None]
            img_np = img_np.astype('float32')
            # prediction0 = predictor.predict_from_files([[volume_fn]],
            #                      None,
            #                      save_probabilities=False, overwrite=False,
            #                      num_processes_preprocessing=1, num_processes_segmentation_export=1,
            #                      folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

            # prediction1 = predictor.predict_from_list_of_npy_arrays([img_np],
            #                                         None,
            #                                         [props],
            #                                         None, 1, save_probabilities=False,
            #                                         num_processes_segmentation_export=2)
            
            prediction = predictor.predict_single_npy_array(img_np, props, None, None, True)

            predicted_vessel = prediction[0]
            pred_img = sitk.GetImageFromArray(predicted_vessel)
            pred_img = copy_settings(pred_img, cropped_volume)
            
            perc = predicted_vessel.mean()
            # mag = mag+0.1
            print(f"Perc as 1: {perc:.3f}")
            
            prob_prediction = sitk.GetImageFromArray(prediction[1][1])
            prob_prediction = copy_settings(prob_prediction, cropped_volume)

            step_seg['prob_predicted_vessel'] = prob_prediction
            seed = np.rint(np.array(size_extract)/2).astype(int).tolist()
            
            predicted_vessel = remove_other_vessels(pred_img, seed)

            #print("Now the components are: ")
            #labels, means = connected_comp_info(predicted_vessel, True)
            pd_fn = output_folder +'predictions/seg_'+case+'_'+str(i)+'.mha'
            if take_time:
                print("\n Prediction, forward pass: " + str(time.time() - start_time_loc) + " s\n")
            if run_time:
                step_seg['time'].append(time.time()-start_time_loc)
                start_time_loc = time.time()

            # Surface

            # if seg_file:
            #     surface = evaluate_surface(predict.seg_vol)
            #     if write_samples:
            #         write_vtk_polydata(surface_smooth, seg_fn)

            surface = evaluate_surface(predicted_vessel) # Marching cubes
            surface_smooth = smooth_surface(surface, 12) # Smooth marching cubes

            vtkimage = exportSitk2VTK(cropped_volume)
            length = predicted_vessel.GetSize()[0]*predicted_vessel.GetSpacing()[0]
            surface_smooth = bound_polydata_by_image(vtkimage[0], surface_smooth, length*1/40)

            surface_smooth = get_largest_connected_polydata(surface_smooth)

            if take_time:
                print("\n Calc and smooth surface: " + str(time.time() - start_time_loc) + " s\n")
            if run_time:
                step_seg['time'].append(time.time()-start_time_loc)
                start_time_loc = time.time()

            sfn = output_folder +'surfaces/surf_'+case+'_'+str(i)+'.vtp'
            cfn = output_folder +'centerlines/cent_'+case+'_'+str(i)+'.vtp'
            if write_samples:
                sitk.WriteImage(predicted_vessel, pd_fn)
                write_vtk_polydata(surface_smooth, sfn)
            step_seg['seg_file'] = pd_fn
            step_seg['surf_file'] = sfn
            step_seg['surface'] = surface_smooth


            if i != 0:
                prev_step = vessel_tree.get_previous_step(i)
                old_point_ref = prev_step['old point']
            elif i == 0:
                old_point_ref = step_seg['old point']
            caps = calc_caps(surface_smooth)
            step_seg['caps'] = caps
            _ , source_id = orient_caps(caps, old_point_ref)


            print('Number of caps: ', len(caps))
            if len(caps) < 2: print(error)

            # polydata_point = points2polydata([step_seg['caps'][0]])
            # pfn = '/Users/numisveinsson/Downloads/point.vtp'
            # write_geo(pfn, polydata_point)

            # Centerline
            centerline_poly = calc_centerline(   surface_smooth,
                                                            "profileidlist",
                                                            var_source=[source_id],
                                                            number = i)
            write_centerline(centerline_poly, cfn)

            centerline_poly = resample_centerline(centerline_poly)
            write_centerline(centerline_poly, cfn.replace('.vtp', 'resampled.vtp'))
            centerline_poly = smooth_centerline(centerline_poly)
            write_centerline(centerline_poly, cfn.replace('.vtp', 'smooth.vtp'))
            # centerline_poly = get_largest_connected_polydata(centerline_poly)
            # write_centerline(centerline_poly, cfn.replace('.vtp', 'largest.vtp'))
            centerline_poly = calc_branches(centerline_poly)
            
            step_seg['cent_file'] = cfn
            if not centerline_poly or centerline_poly.GetNumberOfPoints() < 5:
                print("\n Attempting with more smoothing \n")
                surface_smooth1 = smooth_surface(surface, 15)
                surface_smooth1 = bound_polydata_by_image(vtkimage[0], surface_smooth1, length*1/20)
                centerline_poly1 = calc_centerline(surface_smooth1, "profileidlist")
                centerline_poly1 = calc_branches(centerline_poly1)
                if centerline_poly1.GetNumberOfPoints() > 5:
                    sfn = output_folder +'surfaces/surf_'+case+'_'+str(i)+'_1.vtp'
                    surface_smooth = surface_smooth1
                    cfn = output_folder +'centerlines/cent_'+case+'_'+str(i)+'_1.vtp'
                    centerline_poly = centerline_poly1

            if write_samples:
                write_centerline(centerline_poly, cfn)
                write_vtk_polydata(surface_smooth, sfn)
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
            if use_buffer:
                if len(vessel_tree.steps) % N == 0 and len(vessel_tree.steps) >= (N+buffer):
                    for j in range(1,N+1):
                        if vessel_tree.steps[-(j+buffer)]['prob_predicted_vessel']:
                            print(f"Adding step {-(j+buffer)} and number of steps are {len(vessel_tree.steps)}")
                            assembly_segs.add_segmentation( vessel_tree.steps[-(j+buffer)]['prob_predicted_vessel'],
                                                            vessel_tree.steps[-(j+buffer)]['img_index'],
                                                            vessel_tree.steps[-(j+buffer)]['img_size'],
                                                            (1/vessel_tree.steps[-(j+buffer)]['radius'])**2)
                            vessel_tree.steps[-(j+buffer)]['prob_predicted_vessel'] = None
                            vessel_tree.caps = vessel_tree.caps + vessel_tree.steps[-(j+buffer)]['caps']
                    if len(vessel_tree.steps) % (N*5) == 0 and write_samples:
                            #sitk.WriteImage(assembly_segs.assembly, output_folder +'assembly/assembly_'+case+'_'+str(i)+'.mha')
                            assembly = sitk.BinaryThreshold(assembly_segs.assembly, lowerThreshold=0.5, upperThreshold=1)
                            assembly = remove_other_vessels(assembly, initial_seed)
                            surface_assembly = evaluate_surface(assembly, 1)
                            write_vtk_polydata(surface_assembly, output_folder +'assembly/assembly_surface_'+case+'_'+str(i)+'.vtp')

                if take_time:
                        print("\n Adding to seg volume: " + str(time.time() - start_time_loc) + " s\n")
                if run_time:
                    step_seg['time'].append(time.time()-start_time_loc)
                    start_time_loc = time.time()
            else:
                assembly_segs.add_segmentation(step_seg['prob_predicted_vessel'], step_seg['img_index'], step_seg['img_size'], step_seg['radius'])
                step_seg['prob_predicted_vessel'] = None

            # Print polydata surfaces,cents together for animation
            if animation and write_samples:
                # print('Animation step added')
                surfaces_animation.append(surface_smooth)
                surface_accum = appendPolyData(surfaces_animation)
                write_vtk_polydata(surface_accum, output_folder+'animation/animationsurf_'+str(i).zfill(3)+'.vtp')
                cent_animation.append(centerline_poly)
                cent_accum = appendPolyData(cent_animation)
                write_vtk_polydata(cent_accum, output_folder+'animation/animationcent_'+str(i).zfill(3)+'.vtp')

            point_tree, radius_tree, angle_change = get_next_points( centerline_poly,
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
                        list_points.append(points2polydata([pot.tolist()]))
                    final_pot = appendPolyData(list_points)
                    if write_samples:
                        write_vtk_polydata(  final_pot,
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
                if step_seg['seg_file'] and perc>0.33 and vessel_tree.steps[i]['chances'] > 0:
                    print(f'Magnifying radius, perc: {perc}')
                    vessel_tree.steps[i]['radius'] *= 1.1
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
                    list_pot.append(points2polydata([pot['point'].tolist()]))
                final_pot = appendPolyData(list_pot)

                if take_time:
                    print("Branches are: ", vessel_tree.branches)
                if write_samples:
                    final_surface    = appendPolyData(list_surf_branch)
                    final_centerline = appendPolyData(list_cent_branch)
                    final_points     = appendPolyData(list_pts_branch)
                    write_vtk_polydata(final_pot, output_folder+'/assembly/potentials_'+case+'_'+str(branch)+'_'+str(i)+'_points.vtp')
                    write_vtk_polydata(final_surface, output_folder+'/assembly/branch_'+case+'_'+str(branch)+'_'+str(i)+'_surfaces.vtp')
                    write_vtk_polydata(final_centerline, output_folder+'/assembly/branch_'+case+'_'+str(branch)+'_'+str(i)+'_centerlines.vtp')
                    write_vtk_polydata(final_points, output_folder+'/assembly/branch_'+case+'_'+str(branch)+'_'+str(i)+'_points.vtp')

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
            list_pot.append(points2polydata([pot['point'].tolist()]))
            #vessel_tree.caps = vessel_tree.caps + [pot['point']+ volume_size_ratio*pot['radius']*pot['tangent']]
        final_pot = appendPolyData(list_pot)
        write_vtk_polydata(final_pot, output_folder+'/assembly/potentials_'+case+'_'+str(branch)+'_'+str(i)+'_points.vtp')

    if use_buffer:
        print("Adding rest of segs to global")
        # Add rest of local segs to global before returning
        check = 2
        while check < len(vessel_tree.steps) and vessel_tree.steps[-check]['prob_predicted_vessel']:
            assembly_segs.add_segmentation( vessel_tree.steps[-check]['prob_predicted_vessel'],
                                            vessel_tree.steps[-check]['img_index'],
                                            vessel_tree.steps[-check]['img_size'],
                                            (1/vessel_tree.steps[-check]['radius'])**2)
            vessel_tree.steps[-check]['prob_predicted_vessel'] = None
            check += 1
    #
    return list_centerlines, list_surfaces, list_points, assembly_segs, vessel_tree, i