import time
import pdb
import numpy as np
import SimpleITK as sitk
import sys

from .sitk_functions import (import_image, is_point_in_image,
                             map_to_image, extract_volume, copy_settings,
                             remove_other_vessels, check_seg_border)

from .vtk_functions import (points2polydata, write_geo, smooth_surface,
                            exportSitk2VTK, bound_polydata_by_image,
                            get_largest_connected_polydata, write_vtk_polydata,
                            calc_caps, evaluate_surface, appendPolyData,
                            get_points_cells)

# from .vmtk_functions import write_centerline

from .assembly import (Segmentation, VesselTree, print_error,
                       create_step_dict, get_old_ref_point)

from .simvascular import create_pth

from .local_assembly import construct_subvolume

from .tracing_functions import (SkipThisStepError, convert_seg_to_surfs,
                                get_smoothing_params, orient_caps,
                                calc_centerline_vmtk, get_next_points)

from .centerline import calc_centerline_fmm

from .nnunet import initialize_predictor

sys.stdout.flush()


def trace_centerline(
    output_folder,
    image_file,
    case,
    model_folder,
    fold,
    potential_branches,
    max_step_size,
    max_n_branches,
    max_n_steps_per_branch,
    global_config,
    unit='cm',
    scale=1,
    seg_file=None,
    start_seg=None,
    write_samples=False
):

    if unit == 'cm':
        scale_unit = 0.1
    else:
        scale_unit = 1

    # If tracing a an already segmented vasculature
    trace_seg = global_config['SEGMENTATION']

    # Debugging
    debug = global_config['DEBUG']
    debug_step = global_config['DEBUG_STEP']

    retrace_cent = global_config['RETRACE']
    take_time = global_config['TIME_ANALYSIS']

    # Animation params
    animation = global_config['ANIMATION']
    animation_steps = global_config['ANIMATION_STEPS']

    # Tracing params
    allowed_steps = global_config['NR_ALLOW_RETRACE_STEPS']
    prevent_retracing = global_config['PREVENT_RETRACE']
    sort_potentials = global_config['SORT_NEXT']
    merge_potentials = global_config['MERGE_NEXT']

    volume_size_ratio = global_config['VOLUME_SIZE_RATIO']
    perc_enlarge = global_config['MAX_PERC_ENLARGE']
    magnify_radius = global_config['MAGN_RADIUS']
    number_chances = global_config['NR_CHANCES']
    min_radius = global_config['MIN_RADIUS'] * scale_unit
    add_radius = global_config['ADD_RADIUS'] * scale_unit
    run_time = global_config['TIME_ANALYSIS']
    forceful_sidebranch = global_config['FORCEFUL_SIDEBRANCH']
    forceful_sidebranch_magnify = (global_config
                                   ["FORCEFUL_SIDEBRANCH_MAGN_RADIUS"])
    stop_pre = global_config['STOP_PRE']
    stop_radius = global_config['STOP_RADIUS'] * scale_unit
    max_step_branch = max_n_steps_per_branch

    # Assembly params
    use_buffer = global_config['USE_BUFFER']
    N = global_config['ASSEMBLY_EVERY_N']
    buffer = global_config['BUFFER_N']
    weighted = global_config['WEIGHTED_ASSEMBLY']
    weight_type = global_config['WEIGHT_TYPE']

    if (seg_file and trace_seg):
        print("\nWe are tracing a segmented vasculature!")
        print("No need for prediction.")
        print(f"Reading in seg file: {seg_file}")
        reader_seg, origin_im, size_im, spacing_im = import_image(seg_file)
        print(f"""Seg data.
            size: {size_im},
            spacing: {spacing_im},
            origin: {origin_im}""")

    if not (seg_file and trace_seg):
        print(f"Reading in image file: {image_file}, scale: {scale}")
        reader_im, origin_im, size_im, spacing_im = import_image(image_file)
        print(f"""Image data. size: {size_im},\n
           spacing: {spacing_im},\n origin: {origin_im}""")
    else:
        print("No need to read in image file,")
        print("we are using a given segmentation")
        image_file = seg_file
        reader_im = reader_seg

    init_step = potential_branches[0]
    vessel_tree = VesselTree(case, image_file, init_step, potential_branches)
    assembly_segs = Segmentation(case, image_file, weighted,
                                 weight_type=weight_type,
                                 start_seg=start_seg)

    # Load model
    if not (seg_file and trace_seg):
        predictor = initialize_predictor(model_folder, fold)
    else:
        print('No need to load model, we are using a given segmentation')

    branch = 0

    # Track combos of all polydata
    list_centerlines, list_surfaces = [], []
    list_points, list_inside_pts = [], []
    surfaces_animation, cent_animation = [], []

    num_steps_direction = 0
    inside_branch = 0
    i = 0  # numbering chronological order
    while vessel_tree.potential_branches and i < (max_step_size + 1):

        if i in range(0, max_step_size, max_step_size*0 + 1):
            print(f"\n*** Step number {i} ***")
            # print real time
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        if debug and i >= debug_step:
            pdb.set_trace()

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
                    list_inside_pts.append(
                        points2polydata([step_seg['point'].tolist()]))
                    if write_samples:
                        polydata_point = points2polydata(
                                         [step_seg['point'].tolist()])
                        pfn = (output_folder +
                               'points/inside_point_'+case+'_'+str(i)+'.vtp')
                        write_geo(pfn, polydata_point)

                    # cause failure
                    raise SkipThisStepError(
                        "Inside already segmented vessel, stop here"
                    )

                elif is_point_in_image(assembly_segs.assembly,
                                       step_seg['point']
                                       # + step_seg['radius']
                                       # * step_seg['tangent']
                                       ):
                    inside_branch += 1
                else:
                    inside_branch = 0
            # print('\n The inside branch is ', inside_branch)

            # Point
            polydata_point = points2polydata([step_seg['point'].tolist()])

            if write_samples:
                pfn = output_folder + 'points/point_'+case+'_'+str(i)+'.vtp'
                write_geo(pfn, polydata_point)

            mag = 1
            perc = 1
            continue_enlarge = True
            max_mag = 1.3  # stops when reaches this
            add_mag = 0.2

            if step_seg['radius'] > 3 * scale_unit:
                mag = max_mag  # if above 3mm then dont change size

            while perc > perc_enlarge and continue_enlarge:
                if mag > 1 and mag < max_mag:
                    print("""Enlarging bounding box because
                           percentage vessel > 0.33""")
                if mag >= max_mag:
                    print("Keeping original size")
                    mag = 1
                    continue_enlarge = False
                # Extract Volume
                (size_extract,
                 index_extract,
                 border) = map_to_image(step_seg['point'],
                                        step_seg['radius']*mag,
                                        volume_size_ratio,
                                        origin_im,
                                        spacing_im,
                                        size_im,
                                        global_config['MIN_RES'])

                step_seg['img_index'] = index_extract
                step_seg['img_size'] = size_extract
                cropped_volume = extract_volume(reader_im,
                                                index_extract,
                                                size_extract)

                volume_fn = (output_folder +
                             'volumes/volume_'+case+'_'+str(i)+'.mha')

                step_seg['img_file'] = volume_fn
                if write_samples:
                    sitk.WriteImage(cropped_volume, volume_fn)
                    # if seg_file:
                    #     sitk.WriteImage(seg_volume, seg_fn)
                if take_time:
                    print("\n Extracting and writing volumes: " +
                          str(time.time() - start_time_loc) + " s\n")
                if run_time:
                    step_seg['time'] = []
                    step_seg['time'].append(time.time()-start_time_loc)
                    start_time_loc = time.time()

                if not (seg_file and trace_seg):
                    # Prediction
                    spacing = (spacing_im * scale).tolist()
                    spacing = spacing[::-1]
                    props = {}
                    props['spacing'] = spacing
                    img_np = sitk.GetArrayFromImage(cropped_volume)
                    img_np = img_np[None]
                    img_np = img_np.astype('float32')

                    start_time_pred = time.time()
                    prediction = predictor.predict_single_npy_array(img_np,
                                                                    props,
                                                                    None,
                                                                    None,
                                                                    True)
                    print(f"""Prediction time:
                          {(time.time() - start_time_pred):.3f} s""")

                    # Create probability prediction
                    prob_prediction = sitk.GetImageFromArray(prediction[1][1])

                    # Create segmentation prediction (binary)
                    predicted_vessel = prediction[0]
                    pred_img = sitk.GetImageFromArray(predicted_vessel)
                    pred_img = copy_settings(pred_img, cropped_volume)

                else:
                    # Use the given segmentation
                    pred_img = extract_volume(reader_seg,
                                              index_extract,
                                              size_extract)
                    predicted_vessel = sitk.GetArrayFromImage(pred_img)

                    # in this case, the probability is the same
                    # as the segmentation
                    prob_prediction = pred_img
                    # seg_fn = (output_folder
                    #           + 'volumes/volume_'+case+'_'
                    #           + str(i)+'_truth.mha')

                perc = predicted_vessel.mean()
                print(f"Perc as 1: {perc:.3f}, mag: {mag}")
                mag += add_mag

            # Probabilities prediction
            prob_prediction = copy_settings(prob_prediction, cropped_volume)
            step_seg['prob_predicted_vessel'] = prob_prediction

            seed = np.rint(np.array(size_extract)/2).astype(int).tolist()

            predicted_vessel = remove_other_vessels(pred_img, seed)

            # print("Now the components are: ")
            # labels, means = connected_comp_info(predicted_vessel, True)
            pd_fn = output_folder + 'predictions/seg_'+case+'_'+str(i)+'.mha'
            if take_time:
                print("\n Prediction, forward pass: " +
                      str(time.time() - start_time_loc) + " s\n")
            if run_time:
                step_seg['time'].append(time.time()-start_time_loc)
                start_time_loc = time.time()
            if write_samples:
                sitk.WriteImage(predicted_vessel, pd_fn)

            if global_config['MEGA_SUBVOLUME']:
                (predicted_vessel,
                 cropped_volume) = construct_subvolume(step_seg,
                                                       vessel_tree,
                                                       global_config
                                                       ['NR_MEGA_SUB'],
                                                       i,
                                                       inside_branch)
                if write_samples:
                    sitk.WriteImage(cropped_volume,
                                    volume_fn.replace('.mha', '_mega' +
                                                      str(time.time())+'.mha'))
                    sitk.WriteImage(predicted_vessel,
                                    pd_fn.replace('.mha', '_mega' +
                                                  str(time.time())+'.mha'))

            # Surface

            # if seg_file:
            #     surface = evaluate_surface(predict.seg_vol)
            #     if write_samples:
            #         write_vtk_polydata(surface_smooth, seg_fn)

            # surface = evaluate_surface(predicted_vessel) # Marching cubes

            surface = convert_seg_to_surfs(predicted_vessel,
                                           mega_sub=global_config
                                           ['MEGA_SUBVOLUME'],
                                           ref_min_dims=size_extract)

            num_iterations = get_smoothing_params(step_seg['radius'],
                                                  scale_unit,
                                                  mega_sub=global_config
                                                  ['MEGA_SUBVOLUME'],
                                                  already_seg=trace_seg)

            surface_smooth = smooth_surface(surface,
                                            smoothingIterations=num_iterations
                                            )  # Smooth marching cubes

            vtkimage = exportSitk2VTK(cropped_volume)
            length = (predicted_vessel.GetSize()[0]
                      * predicted_vessel.GetSpacing()[0])
            surface_smooth = bound_polydata_by_image(vtkimage[0],
                                                     surface_smooth,
                                                     length*1/40)

            surface_smooth = get_largest_connected_polydata(surface_smooth)

            if take_time:
                print("\n Calc and smooth surface: "
                      + str(time.time() - start_time_loc) + " s\n")
            if run_time:
                step_seg['time'].append(time.time()-start_time_loc)
                start_time_loc = time.time()

            sfn = output_folder + 'surfaces/surf_'+case+'_'+str(i)+'smooth.vtp'
            cfn = output_folder + 'centerlines/cent_'+case+'_'+str(i)+'.vtp'

            if write_samples:
                write_vtk_polydata(surface_smooth, sfn)
                # sfn_un = (output_folder
                #           + 'surfaces/surf_'+case+'_'+str(i)+'_unsmooth.vtp')
                # write_vtk_polydata(surface, sfn_un)

            step_seg['seg_file'] = pd_fn
            step_seg['surf_file'] = sfn
            step_seg['surface'] = surface_smooth

            old_point_ref = get_old_ref_point(vessel_tree,
                                              step_seg,
                                              i,
                                              global_config['MEGA_SUBVOLUME'],
                                              global_config['NR_MEGA_SUB'])
            step_seg['old_point_ref'] = old_point_ref
            if write_samples:
                polydata_point = points2polydata([old_point_ref.tolist()])
                pfn = (output_folder + 'points/point_'
                       + case + '_'+str(i)+'_ref.vtp')
                write_geo(pfn, polydata_point)

            caps = calc_caps(surface_smooth)

            step_seg['caps'] = caps
            sorted_targets, source_id = orient_caps(caps,
                                                    step_seg['point'],
                                                    old_point_ref,
                                                    step_seg['tangent'])
            print(f"Source id: {source_id}")

            print('Number of caps: ', len(caps))
            if len(caps) < 2 and i > 1:
                raise SkipThisStepError(
                    "Less than 2 caps, stop here"
                )

            # polydata_point = points2polydata([step_seg['caps'][0]])
            # pfn = '/Users/numisveinsson/Downloads/point.vtp'
            # write_geo(pfn, polydata_point)

            # Centerline
            if not global_config['CENTERLINE_EXTRACTION_VMTK']:
                print("Calculating centerline using FMM + Gradient Stepping")
                # if first steps and only one cap, use point as source
                if i <= 1 and len(caps) == 1:
                    seed = step_seg['point']
                    targets = caps  # [cap for ind, cap in enumerate(caps)
                    #  if ind != source_id] #caps
                # else use info from orientation
                else:
                    source = source_id
                    seed = caps[source]
                    targets = [cap for ind, cap in enumerate(caps)
                               if ind != source]
                # calculate centerline
                (centerline_poly,
                 success) = calc_centerline_fmm(predicted_vessel,
                                                seed,
                                                targets,
                                                min_res=40)

            else:
                print("Calculating centerline using VMTK")
                (centerline_poly,
                 success) = calc_centerline_vmtk(surface_smooth,
                                                 vtkimage,
                                                 global_config,
                                                 source_id,
                                                 sorted_targets,
                                                 i,
                                                 caps,
                                                 step_seg,
                                                 length)

            if write_samples:
                step_seg['cent_file'] = cfn
                # write_centerline(centerline_poly, cfn)
                write_vtk_polydata(centerline_poly, cfn)
                write_vtk_polydata(surface_smooth, sfn)
            if take_time:
                print("\n Calc centerline: "
                      + str(time.time() - start_time_loc) + " s\n")
            if run_time:
                step_seg['time'].append(time.time()-start_time_loc)
                start_time_loc = time.time()

            # Save step information
            step_seg['point_pd'] = polydata_point
            step_seg['surface'] = surface_smooth
            step_seg['centerline'] = centerline_poly

            if not success:
                raise SkipThisStepError(
                    "Centerline calculation failed, stop here"
                )

            # Assembly
            if use_buffer:
                if (len(vessel_tree.steps) % N == 0
                   and len(vessel_tree.steps) >= (N+buffer)):

                    for j in range(1, N+1):
                        if (vessel_tree.steps[-(j+buffer)]
                           ['prob_predicted_vessel']):
                            print(f"""Adding step {-(j+buffer)} and number of
                                  steps are {len(vessel_tree.steps)}""")
                            assembly_segs.add_segmentation(
                                (vessel_tree.steps[-(j+buffer)]
                                 ['prob_predicted_vessel']),
                                vessel_tree.steps[-(j+buffer)]['img_index'],
                                vessel_tree.steps[-(j+buffer)]['img_size'],
                                (1/vessel_tree.steps[-(j+buffer)]
                                 ['radius'])**2
                                 )
                            (vessel_tree.steps[-(j+buffer)]
                             ['prob_predicted_vessel']) = None
                            vessel_tree.caps = (vessel_tree.caps
                                                + vessel_tree.steps
                                                [-(j+buffer)]['caps'])

                    if len(vessel_tree.steps) % (N*5) == 0 and write_samples:
                        # sitk.WriteImage(assembly_segs.assembly,
                        #                 (output_folder + 'assembly/assembly_'
                        #                  + case + '_'+str(i)+'.mha'))
                        assembly = sitk.BinaryThreshold(assembly_segs.assembly,
                                                        lowerThreshold=0.5,
                                                        upperThreshold=1)
                        # assembly = remove_other_vessels(assembly,
                        #                                 initial_seed)
                        surface_assembly = evaluate_surface(assembly, 1)
                        write_vtk_polydata(surface_assembly,
                                           output_folder
                                           + 'assembly/assembly_surface_'+case
                                           + '_'+str(i)+'.vtp')

                if take_time:
                    print("\n Adding to seg volume: "
                          + str(time.time() - start_time_loc) + " s\n")
                if run_time:
                    step_seg['time'].append(time.time()-start_time_loc)
                    start_time_loc = time.time()
            else:
                assembly_segs.add_segmentation(
                    step_seg['prob_predicted_vessel'],
                    step_seg['img_index'],
                    step_seg['img_size'],
                    step_seg['radius'])
                step_seg['prob_predicted_vessel'] = None

            # Print polydata surfaces,cents together for animation
            if animation and write_samples:
                # print('Animation step added')
                # Add to surface and cent lists
                surfaces_animation.append(surface_smooth)
                cent_animation.append(centerline_poly)

                if i % animation_steps == 0:  # only write out every 10 steps
                    # Create single polydata
                    surface_accum = appendPolyData(surfaces_animation)
                    cent_accum = appendPolyData(cent_animation)
                    # Write out polydata
                    write_vtk_polydata(surface_accum,
                                       (output_folder +
                                        'animation/animationsurf_' +
                                        str(i).zfill(3)+'.vtp'))
                    write_vtk_polydata(cent_accum,
                                       (output_folder +
                                        'animation/animationcent_' +
                                        str(i).zfill(3)+'.vtp'))

            (point_tree,
             radius_tree,
             angle_change) = get_next_points(centerline_poly,
                                             step_seg['point'],
                                             step_seg['old point'],
                                             step_seg['old radius'],
                                             magn_radius=magnify_radius,
                                             min_radius=min_radius,
                                             add_radius=add_radius,
                                             mega_sub=(global_config
                                                       ['MEGA_SUBVOLUME']))

            if take_time:
                print("\n Calc next point: " +
                      str(time.time() - start_time_loc) + " s\n")
            if run_time:
                step_seg['time'].append(time.time()-start_time_loc)
                start_time_loc = time.time()

            vessel_tree.steps[i] = step_seg

            # end if on border and segmentation is on border
            if border:
                print("Checking if segmentation has reached the border")
                if check_seg_border(size_extract,
                                    index_extract,
                                    predicted_vessel,
                                    size_im):
                    raise SkipThisStepError(
                        "Segmentation has reached border, stop here"
                    )

            if point_tree.size != 0:

                i += 1
                num_steps_direction += 1
                next_step = create_step_dict(step_seg['point'],
                                             step_seg['radius'],
                                             point_tree[0],
                                             radius_tree[0],
                                             angle_change[0])

                if len(radius_tree) > 1 and forceful_sidebranch:
                    print('Forceful sidebranch -  adding vector')
                    next_step['point'] += (next_step['radius']
                                           * next_step['tangent'])
                    next_step['radius'] = (next_step['radius']
                                           * forceful_sidebranch_magnify)

                print(f"Next radius is {radius_tree[0]:.3f}")
                vessel_tree.add_step(i, next_step, branch)

                if len(radius_tree) > 1:
                    print('\n _ \n')
                    print(f"\n BIFURCATION - {len(radius_tree)} BRANCHES \n")
                    print('\n _ \n')

                    for j in range(1, len(radius_tree)):
                        dict = create_step_dict(step_seg['point'],
                                                step_seg['radius'],
                                                point_tree[j],
                                                radius_tree[j],
                                                angle_change[j])
                        dict['connection'] = [branch, i-1]
                        if forceful_sidebranch:
                            dict['point'] += dict['radius']*dict['tangent']
                            dict['radius'] = (dict['radius']
                                              * forceful_sidebranch_magnify)

                        vessel_tree.potential_branches.append(dict)
                    list_points_pot = []
                    for pot in point_tree:
                        list_points_pot.append(points2polydata([pot.tolist()]))
                    final_pot = appendPolyData(list_points_pot)
                    if write_samples:
                        write_vtk_polydata(final_pot,
                                           (output_folder
                                            + '/points/bifurcation_'+case+'_'
                                            + str(branch)+'_'+str(i-1)
                                            + '_points.vtp'))
            else:
                print('point_tree.size is 0')
                raise SkipThisStepError(
                    "No next points, stop here"
                )
            print("\n   This location done: "
                  + f"{time.time() - start_time_loc:.3f} s\n")

        # This step failed for some reason
        except Exception as e:
            print(e)

            # Print error
            if step_seg['centerline']:
                print_error(output_folder, i, step_seg, cropped_volume,
                            predicted_vessel, old_point_ref, centerline_poly)
            elif step_seg['seg_file']:
                print_error(output_folder, i, step_seg, cropped_volume,
                            predicted_vessel)
            elif step_seg['img_file']:
                print_error(output_folder, i, step_seg, cropped_volume)

            if i == 0:
                print("Didnt work for first surface")
                # break

            # Allow for another chance, if not already given
            if (vessel_tree.steps[i]['chances'] < number_chances
               and not step_seg['is_inside']):
                print("Giving chance for surface: " + str(i))
                if take_time:
                    print('Radius is ', step_seg['radius'])
                if (step_seg['seg_file'] and perc > 0.33
                   and vessel_tree.steps[i]['chances'] > 0):
                    print(f"""Magnifying radius by 1.2 because percentage
                          vessel is above 0.33: {perc:.3f}""")
                    vessel_tree.steps[i]['radius'] *= 1.2
                (vessel_tree.steps[i]['point']) = ((vessel_tree.steps[i]
                                                    ['point'])
                                                   + (vessel_tree.steps[i]
                                                      ['radius'])
                                                   * (vessel_tree.steps[i]
                                                      ['tangent']))
                vessel_tree.steps[i]['chances'] += 1

            # If already given chance, then move on
            else:

                print("\n*** Error for step: \n" + str(i))
                print("Length of branch: ", len(vessel_tree.branches[branch]))
                print("\n Moving onto another branch\n")

                if len(vessel_tree.branches) <= 1 and step_seg['is_inside']:
                    print("\n\nWARNING: First branch is inside, so stopping\n\n")

                # If inside, then restart branch and i is not 0
                if i != 0 and len(vessel_tree.branches) > 1 and ((step_seg['is_inside'] and global_config['RESTART_BRANCH'])
                   or len(vessel_tree.branches[branch]) <= 2):
                    # If inside, then move on to next branch
                    # and remove allowed_steps

                    if len(vessel_tree.branches[branch]) <= 2:
                        print("Branch length is 1 or 0, so restarting")
                    elif (step_seg['is_inside']
                          and global_config['RESTART_BRANCH']):
                        print('Branch inside, restarting branch')

                    # If this branch is inside, then restart branch
                    # if len(vessel_tree.branches[branch]) <= allowed_steps+2:
                    vessel_tree.restart_branch(branch)
                    branch -= 1
                    # If not the whole branch is inside
                    # then remove allowed_steps
                    # else:
                    #     print('Part inside, removing last n steps')
                    #     vessel_tree.remove_previous_n(branch,
                    #                                   n=allowed_steps-1)
                    i = vessel_tree.branches[-1][-1] + 1
                    print(f"i is now {i}")

                else:

                    if debug:
                        print("Debugging")

                    del vessel_tree.branches[branch][-1]
                    (list_surf_branch, list_cent_branch,
                     list_pts_branch) = [], [], []
                    for id in vessel_tree.branches[branch][1:]:
                        list_surf_branch.append((vessel_tree.steps[id]
                                                 ['surface']))
                        list_cent_branch.append((vessel_tree.steps[id]
                                                 ['centerline']))
                        list_pts_branch.append((vessel_tree.steps[id]
                                                ['point_pd']))
                        vessel_tree.steps[id]['surface'] = None
                        vessel_tree.steps[id]['centerline'] = None
                        vessel_tree.steps[id]['point_pd'] = None
                    list_centerlines.extend(list_cent_branch)
                    list_surfaces.extend(list_surf_branch)
                    list_points.extend(list_pts_branch)

                    # print('Printing potentials')
                    list_pot = []
                    for pot in vessel_tree.potential_branches:
                        list_pot.append(points2polydata(
                            [pot['point'].tolist()],
                            attribute_float=[pot['radius']]))
                    final_pot = appendPolyData(list_pot)

                    if take_time:
                        print("Branches are: ", vessel_tree.branches)
                    # if write_samples:
                    final_surface = appendPolyData(list_surf_branch)
                    final_centerline = appendPolyData(list_cent_branch)
                    final_points = appendPolyData(list_pts_branch)
                    write_vtk_polydata(final_pot,
                                       output_folder
                                       + '/assembly/potentials_'+case+'_'
                                       + str(branch)+'_'+str(i)
                                       + '_points.vtp')
                    write_vtk_polydata(final_surface,
                                       output_folder
                                       + '/assembly/branch_'+case+'_'
                                       + str(branch)+'_'+str(i)
                                       + '_surfaces.vtp')
                    write_vtk_polydata(final_centerline,
                                       output_folder
                                       + '/assembly/branch_'+case+'_'
                                       + str(branch)+'_'+str(i)
                                       + '_centerlines.vtp')
                    
                    # Create SimVascular .pth file for this branch
                    try:
                        centerline_points, _ = get_points_cells(
                            final_points)
                        if len(centerline_points) > 0:
                            # Convert points to list of tuples for create_pth
                            pth_points = [tuple(point)
                                          for point in centerline_points]
                            # remove identical points
                            pth_points = list(dict.fromkeys(pth_points))
                            pth_points = [list(point) for point in pth_points]
                            pth_output_path = (
                                output_folder
                                + '/simvascular/Paths/branch_'+case+'_'
                                + str(branch)+'_'+str(i)
                                + '.pth')
                            create_pth(pth_points, pth_output_path)
                    except Exception as e:
                        print(f"Warning: Could not create .pth file for "
                              f"branch {branch}: {e}")
                    
                    write_vtk_polydata(final_points,
                                       output_folder
                                       + '/assembly/branch_'+case+'_'
                                       + str(branch)+'_'+str(i)
                                       + '_points.vtp')

                    vessel_tree.caps = (vessel_tree.caps
                                        + [step_seg['point']
                                           + volume_size_ratio
                                           * step_seg['radius']
                                           * step_seg['tangent']])

                    if retrace_cent:
                        # second step on this branch
                        ind = vessel_tree.branches[branch][1]
                        step_to_add = vessel_tree.steps[ind]
                        # add connection that says retrace
                        step_to_add['connection'] = [-branch+1, ind]
                        vessel_tree.potential_branches.append(step_to_add)

                # Merge potentials
                if merge_potentials:
                    vessel_tree.merge_pots_radius()

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

                print("Post Branches are: ")
                for print_branch in vessel_tree.branches:
                    print(print_branch)
                print("Number of steps are: ", len(vessel_tree.steps))
                print("Connections of branches are: ",
                      vessel_tree.bifurcations)
                print("Number of potentials left are: ",
                      len(vessel_tree.potential_branches))

                # Check if end prematurely
                if len(vessel_tree.branches) > max_n_branches:
                    print(f"Max number of branches reached: {max_n_branches}")
                    break

    if len(vessel_tree.potential_branches) > 0:
        print('Printing potentials')
        list_pot = []
        for pot in vessel_tree.potential_branches:
            list_pot.append(points2polydata([pot['point'].tolist()],
                                            attribute_float=[pot['radius']]))
        final_pot = appendPolyData(list_pot)
        write_vtk_polydata(final_pot,
                           output_folder+'/potentials_'+case+'_'+str(i)
                           + '_points.vtp')

    if use_buffer:
        print("Adding rest of segs to global")
        # Add rest of local segs to global before returning
        check = 2
        while (check <= len(vessel_tree.steps)
               and vessel_tree.steps[-check]['prob_predicted_vessel']):
            assembly_segs.add_segmentation((vessel_tree.steps[-check]
                                            ['prob_predicted_vessel']),
                                           (vessel_tree.steps[-check]
                                            ['img_index']),
                                           (vessel_tree.steps[-check]
                                            ['img_size']),
                                           (1/vessel_tree.steps[-check]
                                            ['radius'])**2)
            vessel_tree.steps[-check]['prob_predicted_vessel'] = None
            check += 1

    return (list_centerlines,
            list_surfaces,
            list_points,
            list_inside_pts,
            assembly_segs,
            vessel_tree,
            i)
