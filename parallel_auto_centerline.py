import faulthandler
faulthandler.enable()

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED

N_cores = mp.cpu_count()
print("\n Number of cores are: ", N_cores)

import time
start_time = time.time()

import os

import numpy as np
import SimpleITK as sitk

from modules import sitk_functions as sf
from modules import vtk_functions as vf
from modules import vmtk_functions as vmtkfs
from modules.vmr_data import vmr_directories
from modules.assembly import Segmentation, VesselTreeParallel, Branch
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

def create_step_dict(old_point, old_radius, new_point, new_radius, connection=None, angle_change=None):

    step_dict = {}
    step_dict['old point'] = old_point
    step_dict['point'] = new_point
    step_dict['old radius'] = old_radius
    step_dict['tangent'] = (new_point - old_point)/np.linalg.norm(new_point - old_point)
    step_dict['radius'] = new_radius
    step_dict['connection'] = connection
    step_dict['chances'] = 0
    step_dict['seg_file'] = None
    step_dict['img_file'] = None
    step_dict['surf_file'] = None
    step_dict['cent_file'] = None
    step_dict['prob_predicted_vessel'] = None
    step_dict['point_pd'] = None
    step_dict['surface'] = None
    step_dict['centerline'] = None
    step_dict['is_inside'] = False
    step_dict['time'] = None
    if angle_change:
        step_dict['angle change'] = angle_change

    return step_dict

def print_error(output_folder, i, step_seg, image=None, predicted_vessel=None):

    directory = output_folder + str(i) + '_error_'

    if step_seg['img_file']:
        sitk.WriteImage(image, directory + 'img.vtk')

        if step_seg['seg_file']:
            sitk.WriteImage(predicted_vessel, directory + 'seg.vtk')

            if step_seg['surf_file']:
                vf.write_vtk_polydata(step_seg['surface'], directory + 'surf.vtp')

def trace_branch(input_queue, init_step, branch_number, image_file, output_folder, model_folder, modality, img_shape, threshold, initial_seed, seg_file = None):

    prevent_retracing = True
    forceful_sidebranch = True
    take_time = False
    write_samples = False
    inside_branch = 0
    magnify_radius = 1.1
    number_chances = 1

    model = UNet3DIsensee((img_shape[0], img_shape[1], img_shape[2], 1), num_class=1)
    unet = model.build()
    model_name = os.path.realpath(model_folder) + '/weights_unet.hdf5'
    unet.load_weights(model_name)

    if seg_file:
        reader_seg, origin_im, size_im, spacing_im = sf.import_image(seg_file)

    reader_im, origin_im, size_im, spacing_im = sf.import_image(image_file)

    branch = Branch(init_step)
    i = 0
    next_step_worked = True

    while next_step_worked:
        print("\n** i is: " + str(i) + "**")
        try:

            start_time_loc = time.time()

            # Take next step
            step_seg = branch.steps[i]

            if prevent_retracing:
                allowed_steps = 5
                if inside_branch == allowed_steps:
                    step_seg['is_inside'] = True
                    inside_branch = 0
                    #vessel_tree.remove_branch(branch)
                    #branch -= 1
                    #i = i - allowed_steps
                    print('\n \n Inside already segmented vessel!! \n \n')
                    print(vessel_tree.steps[i][i])
                elif vf.is_point_in_image(assembly_segs.assembly, step_seg['point']): #+ step_seg['radius']*step_seg['tangent']):
                    inside_branch += 1
                else:
                    inside_branch = 0
            print('\n The inside branch is ', inside_branch)
            # Point
            polydata_point = vf.points2polydata([step_seg['point'].tolist()])
            pfn = output_folder + 'points/point_'+case+'_branch'+str(branch_number)+'_'+str(i)+'.vtp'
            vf.write_geo(pfn, polydata_point)

            # Extract Volume
            volume_size = 5
            size_extract, index_extract = sf.map_to_image(step_seg['point'], step_seg['radius'], volume_size, origin_im, spacing_im)
            step_seg['img_index'] = index_extract
            step_seg['img_size'] = size_extract
            cropped_volume = sf.extract_volume(reader_im, index_extract, size_extract)
            volume_fn = output_folder +'volumes/volume_'+case+'_branch'+str(branch_number)+'_'+str(i)+'.vtk'

            if seg_file:
                seg_volume = sf.extract_volume(reader_seg, index_extract, size_extract)
                seg_fn = output_folder +'volumes/volume_'+case+'_branch'+str(branch_number)+'_'+str(i)+'_truth.vtk'

            if write_samples:
                sitk.WriteImage(cropped_volume, volume_fn)
                step_seg['img_file'] = volume_fn
                if seg_file:
                    sitk.WriteImage(seg_volume, seg_fn)
            if take_time:
                print("\n Extracting and writing volumes: " + str(time.time() - start_time_loc) + " s\n")

            # Prediction
            predict = Prediction(unet, model_name, modality, cropped_volume, img_shape, output_folder+'predictions', threshold, seg_volume)
            predict.volume_prediction(1)
            predict.resample_prediction()
            d = predict.dice()
            print(str(i)+" , dice score: " +str(d))
            step_seg['dice'] = d

            predicted_vessel = predict.prediction
            step_seg['prob_predicted_vessel'] = predict.prob_prediction
            seed = np.rint(np.array(size_extract)/2).astype(int).tolist()
            predicted_vessel = sf.remove_other_vessels(predicted_vessel, seed)
            #print("Now the components are: ")
            #labels, means = sf.connected_comp_info(predicted_vessel, True)
            pd_fn = output_folder +'predictions/seg_'+case+'_branch'+str(branch_number)+'_'+str(i)+'.vtk'
            if take_time:
                print("\n Prediction, forward pass: " + str(time.time() - start_time_loc) + " s\n")
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
            if take_time:
                print("\n Calc and smooth surface: " + str(time.time() - start_time_loc) + " s\n")

            sfn = output_folder +'surfaces/surf_'+case+'_branch'+str(branch_number)+'_'+str(i)+'.vtk'
            cfn = output_folder +'centerlines/cent_'+case+'_branch'+str(branch_number)+'_'+str(i)+'.vtk'
            if write_samples:
                sitk.WriteImage(predicted_vessel, pd_fn)
                step_seg['seg_file'] = pd_fn
                vf.write_vtk_polydata(surface_smooth, sfn)
                step_seg['surf_file'] = sfn
                step_seg['surface'] = surface_smooth

            # Assembly

            # N = 5
            # buffer = 5
            # if len(branch.steps) % N == 0 and len(branch.steps) >= (N+buffer):
            #     for j in range(N):
            #         if branch.steps[-(j+buffer)]['prob_predicted_vessel']:
            #             assembly_segs.add_segmentation(branch.steps[-(j+buffer)]['prob_predicted_vessel'], branch.steps[-(j+buffer)]['img_index'], branch.steps[-(j+buffer)]['img_size'])
            #             branch.steps[-(j+buffer)]['prob_predicted_vessel'] = None
            #
            #     sitk.WriteImage(assembly_segs.assembly, output_folder +'assembly/assembly_'+case+'_branch'+str(branch_number)+'_'+str(i)+'.vtk')
            #     assembly = sitk.BinaryThreshold(assembly_segs.assembly, lowerThreshold=0.5, upperThreshold=1)
            #     assembly = sf.remove_other_vessels(assembly, initial_seed)
            #     surface_assembly = vf.evaluate_surface(assembly, 1)
            #     vf.write_vtk_polydata(surface_assembly, output_folder +'assembly/assembly_surface_'+case+'_branch'+str(branch_number)+'_'+str(i)+'.vtk')
            #
            #     if take_time:
            #         print("\n Adding to seg volume: " + str(time.time() - start_time_loc) + " s\n")

            # Centerline

            centerline_poly = vmtkfs.calc_centerline(surface_smooth, "profileidlist", number = i)
            #centerline_poly = vf.get_largest_connected_polydata(centerline_poly)
            step_seg['cent_file'] = cfn
            if centerline_poly.GetNumberOfPoints() < 5:
                print("\n Attempting with more smoothing and cropping \n")
                surface_smooth1 = vf.smooth_surface(surface, 12)
                surface_smooth1 = vf.bound_polydata_by_image(vtkimage[0], surface_smooth, length*1/10)
                centerline_poly1 = vmtkfs.calc_centerline(surface_smooth1, "profileidlist")
                if centerline_poly1.GetNumberOfPoints() > 5:
                    sfn = output_folder +'surfaces/surf_'+case+'_branch'+str(branch_number)+'_'+str(i)+'_1.vtk'
                    surface_smooth = surface_smooth1
                    cfn = output_folder +'centerlines/cent_'+case+'_branch'+str(branch_number)+'_'+str(i)+'_1.vtk'
                    centerline_poly = centerline_poly1

            if write_samples:
                vmtkfs.write_centerline(centerline_poly, cfn)

            # Save step information
            step_seg['point_pd'] = polydata_point
            step_seg['surface'] = surface_smooth
            step_seg['centerline'] = centerline_poly

            # Next points

            point_tree, radius_tree, angle_change = vf.get_next_points(centerline_poly, step_seg['point'], step_seg['old point'], step_seg['old radius'], assembly_segs.assembly)
            if take_time:
                print("\n Calc next point: " + str(time.time() - start_time_loc) + " s\n")

            step_seg['time'] = time.time() - start_time_loc
            branch.steps[i] = step_seg

            if point_tree.size != 0:
                i += 1
                next_step = create_step_dict(step_seg['point'], step_seg['radius'], point_tree[0], radius_tree[0]*magnify_radius, angle_change[0])
                print("Next radius is: " + str(radius_tree[0]*magnify_radius))
                branch.add_step(next_step)

                if len(radius_tree) > 1:

                    for j in range(1, len(radius_tree)):
                        dict = create_step_dict(step_seg['point'], step_seg['radius'], point_tree[j], radius_tree[j]*magnify_radius, angle_change[j])
                        dict['connection'] = [branch_number, i]
                        if forceful_sidebranch:
                            dict['point'] += dict['radius']*dict['tangent']
                        branch.add_child(dict)
            else:
                print(this)
            print("\n This location done: " + str(time.time() - start_time_loc) + " s\n")


        except:

            if step_seg['seg_file']:
                print_error(output_folder, i, step_seg, cropped_volume, predicted_vessel)
            elif step_seg['img_file']:
                print_error(output_folder, i, step_seg, cropped_volume)

            if i == 0:
                "Didnt work for first surface"
                break

            if branch.steps[i]['chances'] < number_chances and not step_seg['is_inside']:

                print("\n Giving chance for surface: " + str(i))
                print('Radius is ', step_seg['radius'])
                branch.steps[i]['point'] = branch.steps[i]['point'] + branch.steps[i]['radius']*branch.steps[i]['tangent']
                branch.steps[i]['chances'] += 1

            else:
                # for step in branch.steps:
                #     if step['prob_predicted_vessel']:
                #         assembly_segs.add_segmentation(step['prob_predicted_vessel'], step['img_index'], step['img_size'])
                #         step['prob_predicted_vessel'] = None
                sitk.WriteImage(assembly_segs.assembly, output_folder +'assembly/assembly_'+case+'_branch'+str(branch_number)+'_'+str(i)+'.vtk')
                assembly = sitk.BinaryThreshold(assembly_segs.assembly, lowerThreshold=0.5, upperThreshold=1)
                assembly = sf.remove_other_vessels(assembly, initial_seed)
                surface_assembly = vf.evaluate_surface(assembly, 1)
                vf.write_vtk_polydata(surface_assembly, output_folder +'assembly/assembly_surface_'+case+'_branch'+str(branch_number)+'_'+str(i)+'.vtk')

                print("\n*** Error for surface: \n" + str(i))
                print("\n Moving onto another branch")
                next_step_worked = False

                list_surf_branch, list_cent_branch, list_pts_branch = [], [], []
                for id in branch.steps:
                    list_surf_branch.append(id['surface'])
                    list_cent_branch.append(id['centerline'])
                    list_pts_branch.append(id['point_pd'])
                    del id['surface']
                    del id['centerline']
                    del id['point_pd']
                # list_centerlines.extend(list_cent_branch)
                # list_surfaces.extend(list_surf_branch)
                # list_points.extend(list_pts_branch)

                final_surface = vf.appendPolyData(list_surf_branch)
                vf.write_vtk_polydata(final_surface, output_folder+'/branch_'+case+'_'+str(branch_number)+'_branch'+str(branch_number)+'_'+str(i)+'_surfaces.vtp')

                final_centerline = vf.appendPolyData(list_cent_branch)
                vf.write_vtk_polydata(final_centerline, output_folder+'/branch_'+case+'_'+str(branch_number)+'_branch'+str(branch_number)+'_'+str(i)+'_centerlines.vtp')

                final_points = vf.appendPolyData(list_pts_branch)
                vf.write_vtk_polydata(final_points, output_folder+'/branch_'+case+'_'+str(branch_number)+'_branch'+str(branch_number)+'_'+str(i)+'_points.vtp')


    return branch

WAIT_SLEEP = 0.1 # second to adjust based on our application

class BufferedIter(object):
    def __init__(self):
        pass
        # self.queue = input_queue
        # print(self.queue.get())

    def nextN(self, input_queue, n):
        vals = []
        for _ in range(n):
            if input_queue.empty():
                continue
            vals.append(input_queue.get(timeout=0))
        return vals

def queue_processor(input_queue, task, num_workers):
    futures = dict()
    buffer = BufferedIter()

    with ProcessPoolExecutor(num_workers) as executor:
        while True:
            idle_workers = num_workers - len(futures)
            # print(idle_workers)
            points = buffer.nextN(input_queue, idle_workers)
            for point in points:
                futures[executor.submit(task, input_queue, point)] = point
            done, _ = wait(futures, timeout=WAIT_SLEEP, return_when=ALL_COMPLETED)

            for f in done:
                data = futures[f]
                try:
                    ret = f.result(timeout=0)
                    print(ret)
                except Exception as exc:
                    print(f'future encountered an exception {data, exc}')
                del futures[f]

            if input_queue.empty() and len(futures)==0:
                break

def trace_centerline(output_folder, image_file, case, model_folder, modality, img_shape, threshold, stepsize, potential_branches, seg_file=None, write_samples=True):

    init_step = potential_branches[0]
    vessel_tree = VesselTreeParallel(case, image_file, potential_branches)
    assembly_segs = Segmentation(case, image_file)
    import pdb; pdb.set_trace()


    ## Note: make initial seed within loop
    initial_seed = assembly_segs.assembly.TransformPhysicalPointToIndex(vessel_tree.potential_branches[0]['point'].tolist())

    # Track combos of all polydata
    list_centerlines, list_surfaces, list_points = [], [], []

    m = mp.Manager()
    input_queue = m.Queue()
    for i in vessel_tree.potential_branches:
        input_queue.put(i)

    num_workers = mp.cpu_count()
    futures = dict()
    buffer = BufferedIter()

    import pdb; pdb.set_trace()
    #shared_arr = mp.Array(ctypes.c_double, N)
    branch_number = 0
    with ProcessPoolExecutor(num_workers) as executor:


        while True: # vessel_tree.potential_branches:
            idle_workers = num_workers - len(futures)
            next_pot_branches = buffer.nextN(input_queue, idle_workers)

            for pot_branch in next_pot_branches: #vessel_tree.potential_branches:

                #vessel_tree.remove_potential(pot_branch)
                futures[executor.submit(trace_branch, input_queue, pot_branch, branch_number, image_file, output_folder, model_folder, modality, img_shape, threshold, initial_seed, seg_file)] = pot_branch
                #branch, assembly_segs = trace_branch(input_queue, pot_branch, branch_number, origin_im, spacing_im, reader_im, assembly_segs, output_folder, unet, model_name, modality, img_shape, threshold, initial_seed, reader_seg)
                #vessel_tree.add_branch(branch)
                # print('Branch done: ', branch_number)
                # print("Number of steps were: ", len(branch.steps))
                # print("Branches are: ", vessel_tree.branches)
                branch_number += 1

                # print('Printing potentials')
                # list_pot = []
                # for pot in vessel_tree.potential_branches:
                #     list_pot.append(vf.points2polydata([pot['point'].tolist()]))
                # final_pot = vf.appendPolyData(list_pot)
                # vf.write_vtk_polydata(final_pot, output_folder+'/potentials_'+case+'_'+str(branch_number)+'_'+str(i)+'_points.vtp')

                #vessel_tree.sort_potential()

                #print("Number of potentials left are: ", len(vessel_tree.potential_branches))
            done, _ = wait(futures, timeout=WAIT_SLEEP, return_when=ALL_COMPLETED)
            for f in done:
                data = futures[f]
                try:
                    branch = f.result(timeout=0)

                    for step in branch.steps:
                        if step['prob_predicted_vessel']:
                            assembly_segs.add_segmentation(step['prob_predicted_vessel'], step['img_index'], step['img_size'])
                            step['prob_predicted_vessel'] = None
                    vessel_tree.add_branch(branch)

                except Exception as exc:
                    print(f'future encountered an exception {data, exc}')
                del futures[f]
            import pdb; pdb.set_trace()
            if input_queue.empty() and len(futures)==0:
                break

    return list_centerlines, list_surfaces, list_points, assembly_segs.assembly, vessel_tree

if __name__=='__main__':

    ## Directories
    dir_output = '/Users/numisveinsson/Documents/Berkeley/Research/Automatic_Centerline_ML/scratch_output/'
    directory_data = '/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/vascular_data_3d/'
    dir_model_weights = '/Users/numisveinsson/Documents/Berkeley/Research/BloodVessel_UNet3D/output/test14/'

    ## Information
    case = '0176_0000'
    modality = 'ct'
    nn_input_shape = [64, 64, 64] # Input shape for NN
    threshold = 0.5 # Threshold for binarization of prediction
    stepsize = 1 # Step size along centerline (proportional to radius at the point)
    dir_image, dir_seg, dir_cent, dir_surf = vmr_directories(directory_data, case)

    ## Create directories for results
    create_directories(dir_output)

    ## Get inital seed point + radius
    i = 0
    old_seed, old_radius = vf.get_seed(dir_cent, i, 20)
    initial_seed, initial_radius = vf.get_seed(dir_cent, i, 40)
    vf.write_geo(dir_output+ 'points/0_seed_point.vtp', vf.points2polydata([old_seed.tolist()]))
    init_step = create_step_dict(old_seed, old_radius, initial_seed, initial_radius, [0,0], 0)
    potential_branches = [init_step]
    ## Trace centerline
    centerlines, surfaces, points, assembly, vessel_tree = trace_centerline(dir_output, dir_image, case, dir_model_weights, modality, nn_input_shape, threshold, stepsize, potential_branches, dir_seg, write_samples=True)

    print("\nTotal calculation time is: " + str((time.time() - start_time)/60) + " min\n")
    import pdb; pdb.set_trace()
    total_time = 0
    count = 0
    for i in range(1,len(vessel_tree.steps)):
        if vessel_tree.steps[i]['time']:
            count += 1
            total_time += vessel_tree.steps[i]['time']
    print('Average time was :' + str(total_time/count))

    final_surface = vf.appendPolyData(surfaces)
    vf.write_vtk_polydata(final_surface, dir_output+'/final_'+case+'_'+str(i)+'_surfaces.vtp')

    final_centerline = vf.appendPolyData(centerlines)
    vf.write_vtk_polydata(final_centerline, dir_output+'/final_'+case+'_'+str(i)+'_centerlines.vtp')

    final_points = vf.appendPolyData(points)
    vf.write_vtk_polydata(final_points, dir_output+'/final_'+case+'_'+str(i)+'_points.vtp')

    sitk.WriteImage(assembly, dir_output+'/final_assembly_'+case+'_'+str(i)+'.vtk')

    ## Assembly work
    assembly = sitk.BinaryThreshold(assembly, lowerThreshold=0.5, upperThreshold=1)
    seed = assembly.TransformPhysicalPointToIndex(initial_seed.tolist())
    assembly = sf.remove_other_vessels(assembly, seed)
    assembly_surface = vf.evaluate_surface(assembly, 1)
    vf.write_vtk_polydata(assembly_surface, dir_output+'/final_assembly'+case+'_'+str(i)+'_surface.vtp')
    surface_smooth = vf.smooth_surface(assembly_surface, 10)
    vf.write_vtk_polydata(surface_smooth, dir_output+'/final_assembly'+case+'_'+str(i)+'_surface_smooth.vtp')

    import pdb; pdb.set_trace()
