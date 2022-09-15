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
from modules.vmr_data import vmr_directories
from modules.assembly import Segmentation, VesselTree, print_error, create_step_dict
from prediction import Prediction
from model import UNet3DIsensee

# sys.path.append('/Users/numisveinsson/SimVascular/Python/site-packages/sv_1d_simulation')
# from sv_1d_simulation import centerlines
        # cl = centerlines.Centerlines()
        # cl.extract_center_lines(params)
        # cl.extract_branches(params)
        # cl.write_outlet_face_names(params)

def create_directories(output_folder):
    try:
        os.mkdir(output_folder)
    except Exception as e: print(e)
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
        os.mkdir(output_folder+'assembly')
    except Exception as e: print(e)


def trace_centerline(output_folder, image_file, case, model_folder, modality, img_shape, threshold, stepsize, potential_branches, max_step_size, seg_file=None, write_samples=True):

    take_time = False
    prevent_retracing = True
    magnify_radius = 1
    number_chances = 1
    run_time = False
    use_buffer = True
    forceful_sidebranch = False

    if seg_file:
        reader_seg, origin_im, size_im, spacing_im = sf.import_image(seg_file)

    reader_im, origin_im, size_im, spacing_im = sf.import_image(image_file)

    init_step = potential_branches[0]
    vessel_tree = VesselTree(case, image_file, init_step, potential_branches)
    assembly_segs = Segmentation(case, image_file)

    model = UNet3DIsensee((img_shape[0], img_shape[1], img_shape[2], 1), num_class=1)
    unet = model.build()
    model_name = os.path.realpath(model_folder) + '/weights_unet.hdf5'
    unet.load_weights(model_name)

    ## Note: make initial seed within loop
    initial_seed = assembly_segs.assembly.TransformPhysicalPointToIndex(vessel_tree.steps[0]['point'].tolist())
    branch = 0

    # Track combos of all polydata
    list_centerlines, list_surfaces, list_points = [], [], []
    dice_list, assd_list, haus_list = [], [], []

    num_steps_direction = 0
    inside_branch = 0
    i = 0 # numbering chronological order
    while vessel_tree.potential_branches and i < max_step_size:

        print("\n** i is: " + str(i) + "**")
        try:

            start_time_loc = time.time()

            # Take next step
            step_seg = vessel_tree.steps[i]

            if prevent_retracing:
                allowed_steps = 5
                if inside_branch == allowed_steps:
                    step_seg['is_inside'] = True
                    inside_branch = 0
                    #vessel_tree.remove_branch(branch)
                    #branch -= 1
                    #i = i - allowed_steps
                    print('\n \n Inside already segmented vessel!! \n \n')

                    polydata_point = vf.points2polydata([step_seg['point'].tolist()])
                    pfn = output_folder + 'inside_point_'+case+'_'+str(i)+'.vtp'
                    vf.write_geo(pfn, polydata_point)

                    print(error)
                elif vf.is_point_in_image(assembly_segs.assembly, step_seg['point']): #+ step_seg['radius']*step_seg['tangent']):
                    inside_branch += 1
                else:
                    inside_branch = 0
            #print('\n The inside branch is ', inside_branch)

            # Point
            polydata_point = vf.points2polydata([step_seg['point'].tolist()])
            pfn = output_folder + 'points/point_'+case+'_'+str(i)+'.vtp'
            if write_samples:
                vf.write_geo(pfn, polydata_point)

            # Extract Volume
            volume_size = 5
            size_extract, index_extract = sf.map_to_image(step_seg['point'], step_seg['radius'], volume_size, origin_im, spacing_im)
            step_seg['img_index'] = index_extract
            step_seg['img_size'] = size_extract
            cropped_volume = sf.extract_volume(reader_im, index_extract, size_extract)
            volume_fn = output_folder +'volumes/volume_'+case+'_'+str(i)+'.vtk'

            if seg_file:
                seg_volume = sf.extract_volume(reader_seg, index_extract, size_extract)
                seg_fn = output_folder +'volumes/volume_'+case+'_'+str(i)+'_truth.vtk'
            else:
                seg_volume=None
            if write_samples:
                sitk.WriteImage(cropped_volume, volume_fn)
                step_seg['img_file'] = volume_fn
                if seg_file:
                    sitk.WriteImage(seg_volume, seg_fn)
            if take_time:
                print("\n Extracting and writing volumes: " + str(time.time() - start_time_loc) + " s\n")
            if run_time:
                step_seg['time']=[]
                step_seg['time'].append(time.time()-start_time_loc)
                start_time_loc = time.time()

            # Prediction
            predict = Prediction(unet, model_name, modality, cropped_volume, img_shape, output_folder+'predictions', threshold, seg_volume)
            predict.volume_prediction(1)
            predict.resample_prediction()
            if seg_file:
                d = predict.dice()
                print(str(i)+" , dice score: " +str(d))
                step_seg['dice'] = d
                dice_list.append(d)

            predicted_vessel = predict.prediction
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
            caps = vf.calc_caps(surface)
            vtkimage = vf.exportSitk2VTK(cropped_volume)
            length = predicted_vessel.GetSize()[0]*predicted_vessel.GetSpacing()[0]
            surface_smooth = vf.bound_polydata_by_image(vtkimage[0], surface_smooth, length*1/20)
            if take_time:
                print("\n Calc and smooth surface: " + str(time.time() - start_time_loc) + " s\n")
            if run_time:
                step_seg['time'].append(time.time()-start_time_loc)
                start_time_loc = time.time()

            sfn = output_folder +'surfaces/surf_'+case+'_'+str(i)+'.vtp'
            cfn = output_folder +'centerlines/cent_'+case+'_'+str(i)+'.vtp'
            if write_samples:
                sitk.WriteImage(predicted_vessel, pd_fn)
                step_seg['seg_file'] = pd_fn
                vf.write_vtk_polydata(surface_smooth, sfn)
                step_seg['surf_file'] = sfn
                step_seg['surface'] = surface_smooth

            # Centerline
            centerline_poly = vmtkfs.calc_centerline(surface_smooth, "profileidlist", number = i)
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

            N = 5
            buffer = 5
            if use_buffer:
                if len(vessel_tree.steps) % N == 0 and len(vessel_tree.steps) >= (N+buffer):
                    for j in range(N):
                        if vessel_tree.steps[-(j+buffer)]['prob_predicted_vessel']:
                            assembly_segs.add_segmentation(vessel_tree.steps[-(j+buffer)]['prob_predicted_vessel'], vessel_tree.steps[-(j+buffer)]['img_index'], vessel_tree.steps[-(j+buffer)]['img_size'])
                            vessel_tree.steps[-(j+buffer)]['prob_predicted_vessel'] = None
                    if len(vessel_tree.steps) % (N*10) == 0:
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
                assembly_segs.add_segmentation(step_seg['prob_predicted_vessel'], step_seg['img_index'], step_seg['img_size'])
                step_seg['prob_predicted_vessel'] = None

            point_tree, radius_tree, angle_change = vf.get_next_points(centerline_poly, step_seg['point'], step_seg['old point'], step_seg['old radius'])

            if take_time:
                print("\n Calc next point: " + str(time.time() - start_time_loc) + " s\n")
            if run_time:
                step_seg['time'].append(time.time()-start_time_loc)
                start_time_loc = time.time()

            vessel_tree.steps[i] = step_seg

            if point_tree.size != 0:

                i += 1
                num_steps_direction += 1
                next_step = create_step_dict(step_seg['point'], step_seg['radius'], point_tree[0], radius_tree[0]*magnify_radius, angle_change[0])
                #print("Next radius is: " + str(radius_tree[0]*magnify_radius))
                vessel_tree.add_step(i, next_step, branch)


                if len(radius_tree) > 1:
                    # print('\n _ \n')
                    # print("\n BIFURCATION BIFURCATION BIFURCATION\n")
                    # print('\n _ \n')
                    for j in range(1, len(radius_tree)):
                        dict = create_step_dict(step_seg['point'], step_seg['radius'], point_tree[j], radius_tree[j]*magnify_radius, angle_change[j])
                        dict['connection'] = [branch, i]
                        if forceful_sidebranch:
                            dict['point'] += dict['radius']*dict['tangent']
                        vessel_tree.potential_branches.append(dict)
            else:
                print(this)
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
                if take_time:
                    print("Giving chance for surface: " + str(i))
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
                vf.write_vtk_polydata(final_pot, output_folder+'/potentials_'+case+'_'+str(branch)+'_'+str(i)+'_points.vtp')

                if take_time:
                    print("Branches are: ", vessel_tree.branches)
                if write_samples:
                    final_surface = vf.appendPolyData(list_surf_branch)
                    vf.write_vtk_polydata(final_surface, output_folder+'/branch_'+case+'_'+str(branch)+'_'+str(i)+'_surfaces.vtp')

                    final_centerline = vf.appendPolyData(list_cent_branch)
                    vf.write_vtk_polydata(final_centerline, output_folder+'/branch_'+case+'_'+str(branch)+'_'+str(i)+'_centerlines.vtp')

                    final_points = vf.appendPolyData(list_pts_branch)
                    vf.write_vtk_polydata(final_points, output_folder+'/branch_'+case+'_'+str(branch)+'_'+str(i)+'_points.vtp')

                vessel_tree.sort_potential()
                if len(vessel_tree.potential_branches) == 1:
                    break
                next_step = vessel_tree.potential_branches.pop(-1)
                if next_step['radius'] == vessel_tree.steps[0]['radius']:
                    next_step = vessel_tree.potential_branches.pop(-1)

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
        final_pot = vf.appendPolyData(list_pot)
        vf.write_vtk_polydata(final_pot, output_folder+'/potentials_'+case+'_'+str(branch)+'_'+str(i)+'_points.vtp')

    return list_centerlines, list_surfaces, list_points, assembly_segs.assembly, vessel_tree

if __name__=='__main__':

    test = 'test28'
    print('\n test is: \n', test)
    ## Directories
    dir_output = '/Users/numisveinsson/Documents/Berkeley/Research/Automatic_Centerline_ML/'
    directory_data = directory_data = '/Users/numisveins/Library/Mobile Documents/com~apple~CloudDocs/Documents/Side_SV_projects/SV_ML_Training/vascular_data_3d/'
    dir_model_weights = '/Users/numisveinsson/Documents/Berkeley/Research/BloodVessel_UNet3D/output/' + test + '/'
    write_samples=True

    ## Information
    case = '0002_0001'
    max_step_size = 500
    modality = 'ct'
    nn_input_shape = [64, 64, 64] # Input shape for NN
    threshold = 0.5 # Threshold for binarization of prediction
    stepsize = 1 # Step size along centerline (proportional to radius at the point)
    dir_image, dir_seg, dir_cent, dir_surf = vmr_directories(directory_data, case)

    dir_seg=None
    #dir_output = dir_output + 'output_'+test+'_'+case+'/'
    dir_output = '/Users/numisveins/Documents/Automatic_Tracing_ML/output/scratch'
    dir_output = dir_output + 'output_'+test+'_'+case+'/'

    ## Create directories for results
    create_directories(dir_output)

    ## Get inital seed point + radius
    i = 1
    old_seed, old_radius = vf.get_seed(dir_cent, i, 120)
    initial_seed, initial_radius = vf.get_seed(dir_cent, i, 140)
    vf.write_geo(dir_output+ 'points/0_seed_point.vtp', vf.points2polydata([old_seed.tolist()]))
    init_step = create_step_dict(old_seed, old_radius, initial_seed, initial_radius, 0)
    potential_branches = [init_step]
    ## Trace centerline
    centerlines, surfaces, points, assembly, vessel_tree = trace_centerline(dir_output, dir_image, case, dir_model_weights, modality, nn_input_shape, threshold, stepsize, potential_branches, max_step_size, dir_seg, write_samples)

    print("\nTotal calculation time is: " + str((time.time() - start_time)/60) + " min\n")
    sitk.WriteImage(assembly, dir_output+'/final_assembly_'+case+'_'+test +'_'+str(i)+'.vtk')

    names = ['extraction     ',
             'prediction     ',
             'surface        ',
             'centerline     ',
             'global assembly',
             'next point     ']
    time_sum = np.zeros(len(names))
    counter = 0
    for step in vessel_tree.steps:
        if step['time']:
            time_arr = np.zeros(len(names))
            for i in range(len(step['time'])):
                time_arr[i] = step['time'][i]
            time_sum += time_arr
            counter += 1

    for i in range(len(names)):
        print('Average time for ' + names[i]+ ' : ', time_sum[i]/counter)
    print(np.array(time_sum/counter).tolist())
    import pdb; pdb.set_trace()


    final_surface = vf.appendPolyData(surfaces)
    vf.write_vtk_polydata(final_surface, dir_output+'/final_'+case+'_'+test +'_'+str(i)+'_surfaces.vtp')

    final_centerline = vf.appendPolyData(centerlines)
    vf.write_vtk_polydata(final_centerline, dir_output+'/final_'+case+'_'+test +'_'+str(i)+'_centerlines.vtp')

    final_points = vf.appendPolyData(points)
    vf.write_vtk_polydata(final_points, dir_output+'/final_'+case+'_'+test +'_'+str(i)+'_points.vtp')



    ## Assembly work
    assembly = sitk.BinaryThreshold(assembly, lowerThreshold=0.5, upperThreshold=1)
    seed = assembly.TransformPhysicalPointToIndex(initial_seed.tolist())
    assembly = sf.remove_other_vessels(assembly, seed)
    assembly_surface = vf.evaluate_surface(assembly, 1)
    vf.write_vtk_polydata(assembly_surface, dir_output+'/final_assembly'+case+'_'+test +'_'+str(i)+'_surface.vtp')
    surface_smooth = vf.smooth_surface(assembly_surface, 10)
    vf.write_vtk_polydata(surface_smooth, dir_output+'/final_assembly'+case+'_'+test +'_'+str(i)+'_surface_smooth.vtp')

    total_time, total_dice = 0, 0
    count1, count2 = 0, 0
    for i in range(1,len(vessel_tree.steps)):
        if vessel_tree.steps[i]['time']:
            count1 += 1
            total_time += vessel_tree.steps[i]['time']
        if vessel_tree.steps[i]['dice']:
            if not vessel_tree.steps[i]['dice'] == 0:
                count2 += 1
                total_dice += vessel_tree.steps[i]['dice']
    print('Average time was :' + str(total_time/count1))
    print('Average dice was :' + str(total_dice/count2))

    import pdb; pdb.set_trace()
