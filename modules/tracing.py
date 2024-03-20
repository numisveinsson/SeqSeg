import time
import numpy as np
import SimpleITK as sitk

from .sitk_functions import *
from .vtk_functions import *
from .vmtk_functions import *
from .assembly import Segmentation, VesselTree, print_error, create_step_dict, get_old_ref_point
from .local_assembly import construct_subvolume
from .tracing_functions import *

import sys

import pdb

def trace_centerline(output_folder, image_file, case, model_folder, fold,
                    potential_branches, max_step_size, global_config,
                    unit = 'cm', scale = 1, seg_file=None):

    if unit == 'cm': scale_unit = 0.1
    else:            scale_unit = 1

    # If tracing a an already segmented vasculature
    trace_seg =                     global_config['SEGMENTATION']

    # Debugging
    debug =                         global_config['DEBUG']
    debug_step =                    global_config['DEBUG_STEP']

    # Write out params
    write_samples =                 global_config['WRITE_STEPS']

    retrace_cent   =                global_config['RETRACE']
    take_time      =                global_config['TIME_ANALYSIS']

    # Animation params
    animation =                     global_config['ANIMATION']
    animation_steps =               global_config['ANIMATION_STEPS']
    
    # Tracing params
    allowed_steps =                 global_config['NR_ALLOW_RETRACE_STEPS']
    prevent_retracing =             global_config['PREVENT_RETRACE']
    sort_potentials =               global_config['SORT_NEXT']

    volume_size_ratio =             global_config['VOLUME_SIZE_RATIO']
    magnify_radius =                global_config['MAGN_RADIUS']
    number_chances =                global_config['NR_CHANCES']
    min_radius =                    global_config['MIN_RADIUS'] * scale_unit
    run_time =                      global_config['TIME_ANALYSIS']
    forceful_sidebranch =           global_config['FORCEFUL_SIDEBRANCH']
    forceful_sidebranch_magnify =   global_config['FORCEFUL_SIDEBRANCH_MAGN_RADIUS']
    stop_pre =                      global_config['STOP_PRE']
    stop_radius =                   global_config['STOP_RADIUS'] * scale_unit
    max_step_branch =               global_config['MAX_STEPS_BRANCH']

    #Assembly params
    use_buffer =                    global_config['USE_BUFFER']
    N =                             global_config['ASSEMBLY_EVERY_N']
    buffer =                        global_config['BUFFER_N']
    weighted =                      global_config['WEIGHTED_ASSEMBLY']
    weight_type =                   global_config['WEIGHT_TYPE']

    if seg_file and trace_seg:
        print(f"We are tracing a segmented vasculature! No need for prediction.")
        print(f"Reading in seg file: {seg_file}")
        reader_seg, origin_im, size_im, spacing_im = import_image(seg_file)
        print(f"Seg data. size: {size_im}, spacing: {spacing_im}, origin: {origin_im}")

    print(f"Reading in image file: {image_file}, scale: {scale}")
    reader_im, origin_im, size_im, spacing_im = import_image(image_file)
    print(f"Image data. size: {size_im}, spacing: {spacing_im}, origin: {origin_im}")

    init_step = potential_branches[0]
    vessel_tree   = VesselTree(case, image_file, init_step, potential_branches)
    assembly_segs = Segmentation(case, image_file, weighted, weight_type=weight_type)

    if not seg_file and trace_seg:

        sys.path.append("/global/scratch/users/numi/SeqSeg/nnUNet/")

        from nnunetv2.paths import nnUNet_results
        import torch
        from batchgenerators.utilities.file_and_folder_operations import join
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        print('About to load predictor object')
        # instantiate the nnUNetPredictor
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_gpu=False,
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
    else:
        print('No need to load model, we are using a given segmentation')

    ## Note: make initial seed within loop
    initial_seed = assembly_segs.assembly.TransformPhysicalPointToIndex(vessel_tree.steps[0]['point'].tolist())
    branch = 0

    # Track combos of all polydata
    list_centerlines, list_surfaces, list_points, list_inside_pts = [], [], [], []
    surfaces_animation, cent_animation = [], []

    num_steps_direction = 0
    inside_branch = 0
    i = 0 # numbering chronological order
    while vessel_tree.potential_branches and i < (max_step_size +1):

        if i in range(0, max_step_size, max_step_size*0 +1):
            print(f"\n*** Step number {i} ***")
            # print real time
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        if debug and i >= debug_step: pdb.set_trace()

        try:

            start_time_loc = time.time()

            # Take next step
            step_seg = vessel_tree.steps[i]

            # Check if end prematurely
            if stop_pre:
                # Check if radius is too small
                if step_seg['radius'] < stop_radius:
                    vessel_tree.steps[i]['chances'] = number_chances
                    raise SkipThisStepError(
                        "Radius estimate lower than allowed, stop here"
                    )
                # Check if too many steps in current branch
                if num_steps_direction > max_step_branch:
                    vessel_tree.steps[i]['chances'] = number_chances
                    raise SkipThisStepError(
                        "Too many steps in current branch, stop here"
                    )

            # Check if retracing previously traced area
            if prevent_retracing:
                if inside_branch == allowed_steps:
                    step_seg['is_inside'] = True
                    inside_branch = 0
                    list_inside_pts.append(points2polydata([step_seg['point'].tolist()]))
                    if write_samples:
                        polydata_point = points2polydata([step_seg['point'].tolist()])
                        pfn = output_folder + 'points/inside_point_'+case+'_'+str(i)+'.vtp'
                        write_geo(pfn, polydata_point)
                    # cause failure
                    raise SkipThisStepError(
                        "Inside already segmented vessel, stop here"
                    )
                    
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

            mag = 1
            perc = 1
            continue_enlarge = True
            max_mag = 1.3 # stops when reaches this
            add_mag = 0.2
            
            if step_seg['radius'] > 3 *scale_unit: mag = max_mag # if above 3mm then dont change size
            
            while perc > 0.33 and continue_enlarge:
                if mag > 1 and mag < max_mag:
                    print(f"Enlarging bounding box because percentage vessel > 0.33")
                if mag >= max_mag:
                    print(f"Keeping original size")
                    mag = 1
                    continue_enlarge = False
                # Extract Volume
                size_extract, index_extract = map_to_image(  step_seg['point'],
                                                                step_seg['radius']*mag,
                                                                volume_size_ratio,
                                                                origin_im,
                                                                spacing_im,
                                                                size_im,
                                                                global_config['MIN_RES'])
                step_seg['img_index'] = index_extract
                step_seg['img_size'] = size_extract
                cropped_volume = extract_volume(reader_im, index_extract, size_extract)
                volume_fn = output_folder +'volumes/volume_'+case+'_'+str(i)+'.mha'

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

                if not seg_file and trace_seg:
                    # Prediction
                    spacing = (spacing_im* scale).tolist()
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
                    
                    # Create probability prediction
                    prob_prediction = sitk.GetImageFromArray(prediction[1][1])
                    
                    # Create segmentation prediction (binary)
                    predicted_vessel = prediction[0]
                    pred_img = sitk.GetImageFromArray(predicted_vessel)
                    pred_img = copy_settings(pred_img, cropped_volume)
                
                else:
                    # Use the given segmentation
                    pred_img = extract_volume(reader_seg, index_extract, size_extract)
                    predicted_vessel = sitk.GetArrayFromImage(pred_img)

                    # in this case, the probability is the same as the segmentation
                    prob_prediction = pred_img
                    # seg_fn = output_folder +'volumes/volume_'+case+'_'+str(i)+'_truth.mha'
                    
                perc = predicted_vessel.mean()
                print(f"Perc as 1: {perc:.3f}, mag: {mag}")
                mag += add_mag

            # Probabilities prediction
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
            if write_samples:
                sitk.WriteImage(predicted_vessel, pd_fn)
                
            if global_config['MEGA_SUBVOLUME']:
                predicted_vessel, cropped_volume = construct_subvolume(step_seg, vessel_tree, global_config['NR_MEGA_SUB'], i, inside_branch)
                if write_samples:
                    sitk.WriteImage(cropped_volume, volume_fn.replace('.mha', '_mega'+str(time.time())+'.mha'))
                    sitk.WriteImage(predicted_vessel, pd_fn.replace('.mha', '_mega'+str(time.time())+'.mha'))

            # Surface

            # if seg_file:
            #     surface = evaluate_surface(predict.seg_vol)
            #     if write_samples:
            #         write_vtk_polydata(surface_smooth, seg_fn)

            # surface = evaluate_surface(predicted_vessel) # Marching cubes

            surface = convert_seg_to_surfs(predicted_vessel, mega_sub = global_config['MEGA_SUBVOLUME'], ref_min_dims = size_extract)

            num_iterations = 25 #get_smoothing_params(step_seg['radius'], scale_unit, mega_sub = global_config['MEGA_SUBVOLUME'], already_seg = trace_seg)

            surface_smooth = smooth_surface(surface, smoothingIterations = num_iterations) # Smooth marching cubes

            vtkimage = exportSitk2VTK(cropped_volume)
            length = predicted_vessel.GetSize()[0]*predicted_vessel.GetSpacing()[0]
            surface_smooth = bound_polydata_by_image(vtkimage[0], surface_smooth, length*1/40)

            surface_smooth = get_largest_connected_polydata(surface_smooth)

            if take_time:
                print("\n Calc and smooth surface: " + str(time.time() - start_time_loc) + " s\n")
            if run_time:
                step_seg['time'].append(time.time()-start_time_loc)
                start_time_loc = time.time()

            sfn = output_folder +'surfaces/surf_'+case+'_'+str(i)+'smooth.vtp'
            sfn_un = output_folder +'surfaces/surf_'+case+'_'+str(i)+'_unsmooth.vtp'
            cfn = output_folder +'centerlines/cent_'+case+'_'+str(i)+'.vtp'
            # cfn_un = output_folder +'centerlines/cent_'+case+'_'+str(i)+'_unsmooth.vtp'
            if write_samples:
                write_vtk_polydata(surface_smooth, sfn)
                write_vtk_polydata(surface, sfn_un)
            step_seg['seg_file'] = pd_fn
            step_seg['surf_file'] = sfn
            step_seg['surface'] = surface_smooth

            old_point_ref = get_old_ref_point(vessel_tree, step_seg, i, global_config['MEGA_SUBVOLUME'], global_config['NR_MEGA_SUB'])
            step_seg['old_point_ref'] = old_point_ref
            if write_samples:
                polydata_point = points2polydata([old_point_ref.tolist()])
                pfn = output_folder + 'points/point_'+case+'_'+str(i)+'_ref.vtp'
                write_geo(pfn, polydata_point)
                
            caps = calc_caps(surface_smooth)

            step_seg['caps'] = caps
            sorted_targets , source_id = orient_caps(caps, step_seg['point'], old_point_ref, step_seg['tangent'])
            print(f"Source id: {source_id}")

            print('Number of caps: ', len(caps))
            if len(caps) < 2 and i != 0 : 
                raise SkipThisStepError(
                    "Less than 2 caps, stop here"
                )

            # polydata_point = points2polydata([step_seg['caps'][0]])
            # pfn = '/Users/numisveinsson/Downloads/point.vtp'
            # write_geo(pfn, polydata_point)

            # Centerline
            centerline_poly = calc_centerline(  surface_smooth,
                                                global_config['TYPE_CENT'],
                                                var_source=[source_id],
                                                var_target = sorted_targets,
                                                number = i,
                                                caps = caps,
                                                point = step_seg['point'])
            if write_samples:
                write_centerline(centerline_poly, cfn)

            centerline_poly = resample_centerline(centerline_poly)
            if write_samples:
                write_centerline(centerline_poly, cfn.replace('.vtp', 'resampled.vtp'))
            centerline_poly = smooth_centerline(centerline_poly)
            if write_samples:
                write_centerline(centerline_poly, cfn.replace('.vtp', 'smooth.vtp'))
            # centerline_poly = get_largest_connected_polydata(centerline_poly)
            # write_centerline(centerline_poly, cfn.replace('.vtp', 'largest.vtp'))
            # centerline_poly = calc_branches(centerline_poly)
            
            step_seg['cent_file'] = cfn
            if not centerline_poly or centerline_poly.GetNumberOfPoints() < 5:
                print("\n Attempting with more smoothing \n")
                surface_smooth1 = smooth_surface(surface, 15)
                surface_smooth1 = bound_polydata_by_image(vtkimage[0], surface_smooth1, length*1/40)
                centerline_poly1 = calc_centerline(surface_smooth1,
                                                    global_config['TYPE_CENT'],
                                                    var_source=[source_id],
                                                    var_target = sorted_targets,
                                                    number = i,
                                                    caps = caps,
                                                    point = step_seg['point'])
                # centerline_poly1 = calc_branches(centerline_poly1)
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
                            # assembly = remove_other_vessels(assembly, initial_seed)
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
                # Add to surface and cent lists
                surfaces_animation.append(surface_smooth)
                cent_animation.append(centerline_poly)

                if i % animation_steps == 0: # only write out every 10 steps
                    # Create single polydata
                    surface_accum = appendPolyData(surfaces_animation)
                    cent_accum = appendPolyData(cent_animation)
                    # Write out polydata
                    write_vtk_polydata(surface_accum, output_folder+'animation/animationsurf_'+str(i).zfill(3)+'.vtp')
                    write_vtk_polydata(cent_accum, output_folder+'animation/animationcent_'+str(i).zfill(3)+'.vtp')

            point_tree, radius_tree, angle_change = get_next_points(    centerline_poly,
                                                                        step_seg['point'],
                                                                        step_seg['old point'],
                                                                        step_seg['old radius'],
                                                                        magn_radius = magnify_radius,
                                                                        min_radius = min_radius,
                                                                        mega_sub = global_config['MEGA_SUBVOLUME']
                                                                    )

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
                                                radius_tree[0],
                                                angle_change[0]
                                            )
                
                print("Next radius is: " + str(radius_tree[0]))
                vessel_tree.add_step(i, next_step, branch)

                if len(radius_tree) > 1:
                    print('\n _ \n')
                    print("\n BIFURCATION BIFURCATION BIFURCATION\n")
                    print('\n _ \n')
                    if global_config['MEGA_SUBVOLUME']:     start = 1   # add all bifurcation points
                    else:                                   start = 1   # don't add this one
                    for j in range(start, len(radius_tree)):
                        dict = create_step_dict(step_seg['point'],
                                                step_seg['radius'],
                                                point_tree[j],
                                                radius_tree[j],
                                                angle_change[j])
                        dict['connection'] = [branch, i-1]
                        if forceful_sidebranch:
                            dict['point'] += dict['radius']*dict['tangent']
                            dict['radius'] = dict['radius']*forceful_sidebranch_magnify

                        vessel_tree.potential_branches.append(dict)
                    list_points_pot = []
                    for pot in point_tree:
                        list_points_pot.append(points2polydata([pot.tolist()]))
                    final_pot = appendPolyData(list_points_pot)
                    if write_samples:
                        write_vtk_polydata(  final_pot, output_folder+'/points/bifurcation_'+case+'_'+str(branch)+'_'+str(i-1)+'_points.vtp')
            else:
                print('point_tree.size is 0')
                raise SkipThisStepError(
                    "No next points, stop here"
                )
            # print("\n This location done: " + str(time.time() - start_time_loc) + " s\n")


        except Exception as e:
            print(e)

            if step_seg['centerline']:
                print_error(output_folder, i, step_seg, cropped_volume, predicted_vessel, old_point_ref, centerline_poly)
            elif step_seg['seg_file']:
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
                    print(f'Magnifying radius by 1.2 because percentage vessel is above 0.33: {perc:.3f}')
                    vessel_tree.steps[i]['radius'] *= 1.2
                vessel_tree.steps[i]['point'] = vessel_tree.steps[i]['point'] + vessel_tree.steps[i]['radius']*vessel_tree.steps[i]['tangent']
                vessel_tree.steps[i]['chances'] += 1

            else:
                
                if step_seg['is_inside']:
                    # If inside, then move on to next branch and remove allowed_steps
                    # i -= allowed_steps
                    print(f"Redoing this branch, i is now {i}")
                    # vessel_tree.restart_branch(branch)
                else:
                    print("\n*** Error for surface: \n" + str(i))
                    print("\n Moving onto another branch")

                    if debug:
                        pdb.set_trace()
                    
                    del vessel_tree.branches[branch][-1]
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
                if sort_potentials:
                    vessel_tree.sort_potential_radius()
                else:
                    vessel_tree.shuffle_potential()

                if len(vessel_tree.potential_branches) == 1:
                    break
                next_step = vessel_tree.potential_branches.pop(1)
                if next_step['point'][0] == vessel_tree.steps[0]['point'][0]:
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
        write_vtk_polydata(final_pot, output_folder+'/potentials_'+case+'_'+str(i)+'_points.vtp')

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

    return list_centerlines, list_surfaces, list_points, list_inside_pts, assembly_segs, vessel_tree, i
