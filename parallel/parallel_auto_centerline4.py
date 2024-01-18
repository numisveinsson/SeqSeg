import faulthandler
faulthandler.enable()

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait, ALL_COMPLETED
from functools import partial
from itertools import product

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
from modules.datasets import vmr_directories
from modules.assembly import Segmentation, VesselTreeParallel, Branch
from prediction import Prediction
from model import UNet3DIsensee

# sys.path.append('/Users/numisveinsson/SimVascular/Python/site-packages/sv_1d_simulation')
# from sv_1d_simulation import centerlines
        # cl = centerlines.Centerlines()
        # cl.extract_center_lines(params)
        # cl.extract_branches(params)
        # cl.write_outlet_face_names(params)

#import cProfile
#cProfile.run('main()')

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

def chunks(l, n):
    """Yield n number of sequential chunks from l."""
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
        yield l[si:si+(d+1 if i < r else d)]

def trace_branch(input_queue, init_step, branch_number, image_file, output_folder, modality, threshold, initial_seed, seg_file = None):

    prevent_retracing = False
    forceful_sidebranch = True
    take_time = False
    write_samples = False
    inside_branch = 0
    magnify_radius = 1.1
    number_chances = 1
    run_time = True

    if seg_file:
        reader_seg, origin_im, size_im, spacing_im = sf.import_image(seg_file)

    reader_im, origin_im, size_im, spacing_im = sf.import_image(image_file)

    branch = Branch(init_step, branch_number)
    i = 0
    next_step_worked = True
    print("\n** branch is: " + str(branch_number) + "**")

    # lock = mp.Lock()
    workers = 2

    def write_to_global(fl_ids_chunks, lock, init_id):
        sub_array = np.array([index_extract[0], index_extract[1], index_extract[2]])
        len_in_z = len(fl_ids_chunks[0])
        # print('length of z is ', len_in_z)
        for i, fl_list in enumerate(fl_ids_chunks):
            lock.acquire()
            local_index = idx_arr[:, init_id + i*len_in_z] - sub_array
            values = np.array(assembly_array[fl_list[0]:fl_list[-1]+1]) + np_prob_array[local_index[0], local_index[1], :]
            assembly_array[fl_list[0]:fl_list[-1]+1] = values
            values = np.array(number_updates_array[fl_list[0]:fl_list[-1]+1]) + 1
            number_updates_array[fl_list[0]:fl_list[-1]+1] = values
            lock.release()
        return

    def get_from_global_image(fl_ids_chunks, lock, init_id):
        # np_cropped is global outside loop
        sub_array = np.array([index_extract[0], index_extract[1], index_extract[2]])
        len_in_z = len(fl_ids_chunks[0])
        for i, fl_list in enumerate(fl_ids_chunks):
            # lock.acquire()
            local_index = idx_arr[:, init_id + i*len_in_z] - sub_array
            np_cropped[local_index[0], local_index[1], :] = image_array[fl_list[0]:fl_list[-1]+1]
            #print(np_cropped[local_index[0], local_index[1], :])
            #print(image_array[fl_list[0]:fl_list[-1]+1])
            # lock.release()
        return


    while next_step_worked:
        #print("\n** branch is: " + str(branch_number) + "**")
        #print("\n** i is: " + str(i) + "**")
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
                    #print('\n \n Inside already segmented vessel!! \n \n')
                    print(vessel_tree.steps[i][i])
                elif vf.is_point_in_image(assembly_segs.assembly, step_seg['point']): #+ step_seg['radius']*step_seg['tangent']):
                    inside_branch += 1
                else:
                    inside_branch = 0
            #print('\n The inside branch is ', inside_branch)

            # Point
            polydata_point = vf.points2polydata([step_seg['point'].tolist()])
            pfn = output_folder + 'points/point_'+case+'_branch'+str(branch_number)+'_'+str(i)+'.vtp'
            vf.write_geo(pfn, polydata_point)

            # Extract Volume
            volume_size = 5
            size_extract, index_extract = sf.map_to_image(step_seg['point'], step_seg['radius'], volume_size, origin_im, spacing_im)
            step_seg['img_index'] = index_extract
            step_seg['img_size'] = size_extract
            #cropped_volume1 = sf.extract_volume(reader_im, index_extract, size_extract)
            # print(cropped_volume.GetSize())

            ####################################################
            # Calculate boundaries
            edges = np.array(index_extract) + np.array(size_extract)
            vals = product(range(index_extract[0],edges[0]), range(index_extract[1],edges[1]), range(index_extract[2],edges[2]))
            listx, listy, listz = zip(*vals)
            idx_arr = np.array([listx, listy, listz])
            fl_ids = np.ravel_multi_index(idx_arr, size_im)
            # getting chunks to be assigned to each thread
            size_slice = edges[2]-index_extract[2]
            number_mini_chunk = (edges[0]-index_extract[0])*(edges[1]-index_extract[1])
            mini_ids = list(chunks(fl_ids, number_mini_chunk))

            start_time_global = time.time()
            list_ids_slices = list(chunks(mini_ids, workers))
            size_workers = np.array([len(c) for c in list_ids_slices])
            cumu = np.cumsum(size_workers)
            init_ids = np.zeros(workers).astype(int)

            init_ids[1:] += cumu[0:-1]*size_slice
            # create np_cropped which will be updated in the threads
            np_cropped = np.zeros((edges[0]-index_extract[0], edges[1]-index_extract[1], edges[2]-index_extract[2])).astype('int')

            with ThreadPoolExecutor(max_workers=workers) as executor_thread:
                fut = [executor_thread.submit(get_from_global_image, list_ids_slices[i_thread], mp.Lock(), init_id) for i_thread,init_id in enumerate(init_ids)]
            done, _ = wait(fut, return_when=ALL_COMPLETED)
            for future in done:
                try:
                    data = future.result()
                except Exception as e: print(e)

            # print(cropped_volume1.GetPixelID())
            cropped_volume = sitk.GetImageFromArray(np_cropped.transpose(2, 1, 0).astype('int16'))
            new_origin = np.array(origin_im) + np.array(index_extract)*np.array(spacing_im)
            cropped_volume.SetOrigin(tuple(new_origin))
            cropped_volume.SetSpacing(spacing_im)
            cropped_volume.SetDirection(reader_im.GetDirection())
            # print(cropped_volume.GetOrigin())
            # print(cropped_volume1.GetOrigin())

            #cropped_volume = sf.numpy_to_sitk(np_cropped.transpose(2,1,0),reader_im)
            # print(sitk.GetArrayFromImage(cropped_volume))
            # print(sitk.GetArrayFromImage(cropped_volume1))
            # print(cropped_volume.GetSize())
            # print(cropped_volume1.GetSize())
            # print(cropped_volume)
            # print(cropped_volume1)
            # print((sitk.GetArrayFromImage(cropped_volume) == sitk.GetArrayFromImage(cropped_volume1)).all())
            ####################################################

            volume_fn = output_folder +'volumes/volume_'+case+'_branch'+str(branch_number)+'_'+str(i)+'.vtk'

            if seg_file:
                seg_volume = sf.extract_volume(reader_seg, index_extract, size_extract)
                seg_fn = output_folder +'volumes/volume_'+case+'_branch'+str(branch_number)+'_'+str(i)+'_truth.vtk'
            else:
                seg_volume = None
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
            predict = Prediction(model, model_name, modality, cropped_volume, img_shape, output_folder+'predictions', threshold, seg_volume=None)
            predict.volume_prediction(1)
            predict.resample_prediction()
            if seg_file:
                d = predict.dice()
                print(str(i)+" , dice score: " +str(d))
                step_seg['dice'] = d

            predicted_vessel = predict.prediction
            # print(predict.prob_prediction)
            #step_seg['prob_predicted_vessel'] = predict.prob_prediction

            seed = np.rint(np.array(size_extract)/2).astype(int).tolist()
            predicted_vessel = sf.remove_other_vessels(predicted_vessel, seed)

            pd_fn = output_folder +'predictions/seg_'+case+'_branch'+str(branch_number)+'_'+str(i)+'.vtk'
            if take_time:
                print("\n Prediction, forward pass: " + str(time.time() - start_time_loc) + " s\n")

            if run_time:
                step_seg['time'].append(time.time()-start_time_loc)
                #print('Prediction took: ', time.time()-start_time_loc)
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
            if take_time:
                print("\n Calc and smooth surface: " + str(time.time() - start_time_loc) + " s\n")

            if run_time:
                step_seg['time'].append(time.time()-start_time_loc)
                start_time_loc = time.time()

            sfn = output_folder +'surfaces/surf_'+case+'_branch'+str(branch_number)+'_'+str(i)+'.vtk'
            cfn = output_folder +'centerlines/cent_'+case+'_branch'+str(branch_number)+'_'+str(i)+'.vtk'
            if write_samples:
                sitk.WriteImage(predicted_vessel, pd_fn)
                step_seg['seg_file'] = pd_fn
                vf.write_vtk_polydata(surface_smooth, sfn)
                step_seg['surf_file'] = sfn
                #step_seg['surface'] = surface_smooth

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
                    sfn = output_folder +'surfaces/surf_'+case+'_branch'+str(branch_number)+'_'+str(i)+'_1.vtk'
                    surface_smooth = surface_smooth1
                    cfn = output_folder +'centerlines/cent_'+case+'_branch'+str(branch_number)+'_'+str(i)+'_1.vtk'
                    centerline_poly = centerline_poly1
            if take_time:
                print("\n Calc centerline: " + str(time.time() - start_time_loc) + " s\n")
            if write_samples:
                vmtkfs.write_centerline(centerline_poly, cfn)

            # Save step information
            #step_seg['point_pd'] = polydata_point
            #step_seg['surface'] = surface_smooth
            #step_seg['centerline'] = centerline_poly
            if take_time:
                print("\n Write centerline: " + str(time.time() - start_time_loc) + " s\n")
            if run_time:
                step_seg['time'].append(time.time()-start_time_loc)
                start_time_loc = time.time()

            # Assembly
            cut = 1
            edges = np.array(index_extract) + np.array(size_extract) - cut
            index_extract = np.array(index_extract) + cut

            np_arr_add = sitk.GetArrayFromImage(predict.prob_prediction).transpose(2,1,0)
            np_prob_array = np_arr_add[cut:size_extract[0]-cut, cut:size_extract[1]-cut, cut:size_extract[2]-cut]

            # Calculate boundaries
            vals = product(range(index_extract[0],edges[0]), range(index_extract[1],edges[1]), range(index_extract[2],edges[2]))
            listx, listy, listz = zip(*vals)
            idx_arr = np.array([listx, listy, listz])
            fl_ids = np.ravel_multi_index(idx_arr, size_im)
            # getting chunks to be assigned to each thread
            size_slice = edges[2]-index_extract[2]
            number_mini_chunk = (edges[0]-index_extract[0])*(edges[1]-index_extract[1])
            mini_ids = list(chunks(fl_ids, number_mini_chunk))

            start_time_global = time.time()
            list_ids_slices = list(chunks(mini_ids, workers))
            size_workers = np.array([len(c) for c in list_ids_slices])
            cumu = np.cumsum(size_workers)
            init_ids = np.zeros(workers).astype(int)

            init_ids[1:] += cumu[0:-1]*size_slice

            # print('about to enter thread')
            with ThreadPoolExecutor(max_workers=workers) as executor_thread:
                fut = [executor_thread.submit(write_to_global, list_ids_slices[i_thread], mp.Lock(), init_id) for i_thread,init_id in enumerate(init_ids)]
            done, _ = wait(fut, timeout=None, return_when=ALL_COMPLETED)
            for future in done:
                try:
                    data = future.result()
                except Exception as e: print(e)
            # for index in range(len(fl_ids)):
            #     lock.acquire()
            #     #local_assembly = assembly_array[ids]
            #     #print(assembly_array[0:5])
            #     assembly_array[fl_ids[index]] += np_prob_array[tuple(idx_arr[:,index] - np.array([index_extract[0], index_extract[1], index_extract[2]])) ]
            #     number_updates_array[fl_ids[index]] += 1
            #     lock.release()
            if take_time:
                print('writing to global done: ', time.time()-start_time_global)
            if run_time:
                step_seg['time'].append(time.time()-start_time_loc)
                start_time_loc = time.time()
            # Next points

            point_tree, radius_tree, angle_change = vf.get_next_points(centerline_poly, step_seg['point'], step_seg['old point'], step_seg['old radius'])
            if take_time:
                print("\n Calc next point: " + str(time.time() - start_time_loc) + " s\n")
            if run_time:
                step_seg['time'].append(time.time()-start_time_loc)
                start_time_loc = time.time()
            #step_seg['time_total'] = time.time() - start_time_loc
            branch.steps[i] = step_seg

            if point_tree.size != 0:
                i += 1
                next_step = create_step_dict(step_seg['point'], step_seg['radius'], point_tree[0], radius_tree[0]*magnify_radius, angle_change[0])
                #print("Next radius is: " + str(radius_tree[0]*magnify_radius))
                branch.add_step(next_step)

                if len(radius_tree) > 1:
                    # print('\n _ \n')
                    # print("\n BIFURCATION BIFURCATION BIFURCATION\n")
                    # print('\n _ \n')
                    for j in range(1, len(radius_tree)):
                        dict = create_step_dict(step_seg['point'], step_seg['radius'], point_tree[j], radius_tree[j]*magnify_radius, angle_change[j])
                        dict['connection'] = [branch_number, i]
                        if forceful_sidebranch:
                            dict['point'] += dict['radius']*dict['tangent']
                        branch.add_child(dict)
                        input_queue.put(dict)
            else:
                print(this)
            #print("\n This location done: " + str(time.time() - start_time_loc) + " s\n")

        except Exception as e:
            #print(e)

            print('Failed step')
            if step_seg['seg_file']:
                print_error(output_folder, i, step_seg, cropped_volume, predicted_vessel)
            elif step_seg['img_file']:
                print_error(output_folder, i, step_seg, cropped_volume)

            if i == 0:
                #print("Didnt work for first surface")
                break

            if branch.steps[i]['chances'] < number_chances and not step_seg['is_inside']:

                #print("\n Giving chance for surface: " + str(i))
                #print('Radius is ', step_seg['radius'])
                branch.steps[i]['point'] = branch.steps[i]['point'] + branch.steps[i]['radius']*branch.steps[i]['tangent']
                branch.steps[i]['chances'] += 1

            else:
                # for step in branch.steps:
                #     if step['prob_predicted_vessel']:
                #         assembly_segs.add_segmentation(step['prob_predicted_vessel'], step['img_index'], step['img_size'])
                #         step['prob_predicted_vessel'] = None
                # sitk.WriteImage(assembly_segs.assembly, output_folder +'assembly/assembly_'+case+'_branch'+str(branch_number)+'_'+str(i)+'.vtk')
                # assembly = sitk.BinaryThreshold(assembly_segs.assembly, lowerThreshold=0.5, upperThreshold=1)
                # assembly = sf.remove_other_vessels(assembly, initial_seed)
                # surface_assembly = vf.evaluate_surface(assembly, 1)
                # vf.write_vtk_polydata(surface_assembly, output_folder +'assembly/assembly_surface_'+case+'_branch'+str(branch_number)+'_'+str(i)+'.vtk')

                #print("\n*** Error for surface: \n" + str(i))
                #print('Branch number is: ', branch_number)
                print("\n Moving onto another branch")
                next_step_worked = False

                # list_surf_branch, list_cent_branch, list_pts_branch = [], [], []
                # for id in branch.steps:
                #     list_surf_branch.append(id['surface'])
                #     list_cent_branch.append(id['centerline'])
                #     list_pts_branch.append(id['point_pd'])
                #     del id['surface']
                #     del id['centerline']
                #     del id['point_pd']
                # list_centerlines.extend(list_cent_branch)
                # list_surfaces.extend(list_surf_branch)
                # list_points.extend(list_pts_branch)
                if write_samples:
                    final_surface = vf.appendPolyData(list_surf_branch)
                    vf.write_vtk_polydata(final_surface, output_folder+'/branch_'+case+'_'+str(branch_number)+'_branch'+str(branch_number)+'_'+str(i)+'_surfaces.vtp')

                    final_centerline = vf.appendPolyData(list_cent_branch)
                    vf.write_vtk_polydata(final_centerline, output_folder+'/branch_'+case+'_'+str(branch_number)+'_branch'+str(branch_number)+'_'+str(i)+'_centerlines.vtp')

                    final_points = vf.appendPolyData(list_pts_branch)
                    vf.write_vtk_polydata(final_points, output_folder+'/branch_'+case+'_'+str(branch_number)+'_branch'+str(branch_number)+'_'+str(i)+'_points.vtp')
    print('Branch done: ', branch_number)
    print('had steps: ', i)
    return branch

WAIT_SLEEP = 1 # second to adjust based on our application

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

def do_init(img_shape1, model_folder):

    global model
    global model_name
    global img_shape

    model = UNet3DIsensee((img_shape1[0], img_shape1[1], img_shape1[2], 1), num_class=1)
    model = model.build()
    model_name = os.path.realpath(model_folder) + '/weights_unet.hdf5'
    model.load_weights(model_name)
    img_shape = img_shape1

    print('created model')

    return

inited =  False
initresult = None
def initwrapper(do_init, initargs, x):
    # This will be called in the child. inited
    # Will be False the first time its called, but then
    # remain True every other time its called in a given
    # worker process.
    global inited, initresult
    if not inited:
        inited = True
        initresult = do_init(*initargs)
    return initresult

def initmap(executor, do_init, initargs, it):
    return executor.map(partial(initwrapper, do_init, initargs),it)

def trace_centerline(output_folder, image_file, case, model_folder, modality, img_shape, threshold, stepsize, potential_branches, seg_file=None, write_samples=True):

    max_number_steps = 300

    init_step = potential_branches[0]
    vessel_tree = VesselTreeParallel(case, image_file, potential_branches)
    assembly_segs = Segmentation(case, image_file)

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

    #shared_arr = mp.Array(ctypes.c_double, N)
    branch_number = 0

    with ProcessPoolExecutor(num_workers) as executor:

        out = initmap(executor, do_init, (img_shape, model_folder), range(num_workers))
        for o in out:
            print('')
        print('Initialization Done')

        counter = 0
        while True: # vessel_tree.potential_branches:

            idle_workers = num_workers - len(futures)
            next_pot_branches = buffer.nextN(input_queue, idle_workers)
            # print('\n _ \n')
            # print("\n NEW BRANCHES FOR PROC \n")
            # print('\n _ \n')
            #print('next pot branches are: ', next_pot_branches)
            for pot_branch in next_pot_branches: #vessel_tree.potential_branches:

                #vessel_tree.remove_potential(pot_branch)
                futures[executor.submit(trace_branch, input_queue, pot_branch, branch_number, image_file, output_folder, modality, threshold, initial_seed, seg_file)] = pot_branch
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
            # print('\n _ \n')
            # print("\n CHECK \n")
            # print('\n _ \n')
            done, _ = wait(futures, timeout=WAIT_SLEEP, return_when=ALL_COMPLETED)
            for f in done:
                data = futures[f]
                try:
                    branch = f.result(timeout=0)
                    counter += len(branch.steps)
                    print('\n _ \n')
                    print("\n Counter is \n", counter)
                    print('\n _ \n')
                    # for step in branch.steps:
                    #     if step['prob_predicted_vessel']:
                    #         assembly_segs.add_segmentation(step['prob_predicted_vessel'], step['img_index'], step['img_size'])
                    #         step['prob_predicted_vessel'] = None
                    vessel_tree.add_branch(branch, )

                except Exception as exc:
                    print(f'future encountered an exception {data, exc}')
                del futures[f]
            if input_queue.empty() and len(futures)==0:
                break
            if counter > max_number_steps:
                break

    start_time_assembly = time.time()
    output_array = np.array(assembly_array[:])/(np.array(number_updates_array[:])+1e-6)
    reader_img, _, size_im, spacing_im = sf.import_image(image_file)
    assembly_segs.assembly = sitk.GetImageFromArray(output_array.reshape(size_im).transpose(2,1,0))
    assembly_segs.assembly = sf.copy_settings(assembly_segs.assembly, reader_img)
    print('Creating final 3D volume :', time.time() - start_time_assembly)
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
    dir_seg = None
    write_samples = False
    ## Create directories for results
    create_directories(dir_output)

    ## Get inital seed point + radius
    i = 0
    old_seed, old_radius = vf.get_seed(dir_cent, i, 20)
    initial_seed, initial_radius = vf.get_seed(dir_cent, i, 40)
    vf.write_geo(dir_output+ 'points/0_seed_point.vtp', vf.points2polydata([old_seed.tolist()]))
    init_step = create_step_dict(old_seed, old_radius, initial_seed, initial_radius, [0,0], 0)
    potential_branches = [init_step]

    ## Create Global Shared Arrays
    img = sf.read_image(dir_image)
    sizes = img.GetSize()
    start_time = time.time()
    total_size = sizes[0]*sizes[1]*sizes[2]
    assembly_array = mp.Array("f", total_size, lock=False)
    number_updates_array = mp.Array("i", total_size, lock=False)
    print('done created array: ', (time.time() - start_time))

    np_array_img = sitk.GetArrayFromImage(sitk.ReadImage(dir_image)).transpose(2,1,0)
    #print(np_array_img)
    image_array = mp.Array("f", total_size, lock=False)
    image_array[:] = np_array_img.flatten()
    print('done created array: ', (time.time() - start_time))

    ## Call function
    centerlines, surfaces, points, assembly, vessel_tree = trace_centerline(dir_output, dir_image, case, dir_model_weights, modality, nn_input_shape, threshold, stepsize, potential_branches, dir_seg, write_samples)

    print("\nTotal calculation time is: " + str((time.time() - start_time)/60) + " min\n")

    sitk.WriteImage(assembly, dir_output+'/final_assembly_'+case+'_'+'.vtk')

    names = ['extraction     ',
             'prediction     ',
             'surface        ',
             'centerline     ',
             'global assembly',
             'next point     ']
    time_sum = np.zeros(len(names))
    counter = 0
    for branch in vessel_tree.branches:
        for step in branch.steps:
            if step['time']:
                time_arr = np.zeros(len(names))
                for i in range(len(step['time'])):
                    time_arr[i] = step['time'][i]
                time_sum += time_arr
                counter += 1

    for i in range(len(names)):
        print('Average time for ' + names[i]+ ' : ', time_sum[i]/counter)
        print('Total time for '+ names[i]+ ' : ', time_sum[i])

    print(np.array(time_sum/counter).tolist())
    import pdb; pdb.set_trace()
    # total_time = 0
    # count = 0
    # for i in range(1,len(vessel_tree.steps)):
    #     if vessel_tree.steps[i]['time']:
    #         count += 1
    #         total_time += vessel_tree.steps[i]['time']
    # print('Average time was :' + str(total_time/count))
    #
    # final_surface = vf.appendPolyData(surfaces)
    # vf.write_vtk_polydata(final_surface, dir_output+'/final_'+case+'_'+str(i)+'_surfaces.vtp')
    #
    # final_centerline = vf.appendPolyData(centerlines)
    # vf.write_vtk_polydata(final_centerline, dir_output+'/final_'+case+'_'+str(i)+'_centerlines.vtp')
    #
    # final_points = vf.appendPolyData(points)
    # vf.write_vtk_polydata(final_points, dir_output+'/final_'+case+'_'+str(i)+'_points.vtp')
    #
    # ## Assembly work
    # assembly = sitk.BinaryThreshold(assembly, lowerThreshold=0.5, upperThreshold=1)
    # seed = assembly.TransformPhysicalPointToIndex(initial_seed.tolist())
    # assembly = sf.remove_other_vessels(assembly, seed)
    # assembly_surface = vf.evaluate_surface(assembly, 1)
    # vf.write_vtk_polydata(assembly_surface, dir_output+'/final_assembly'+case+'_'+str(i)+'_surface.vtp')
    # surface_smooth = vf.smooth_surface(assembly_surface, 10)
    # vf.write_vtk_polydata(surface_smooth, dir_output+'/final_assembly'+case+'_'+str(i)+'_surface_smooth.vtp')
    #
    # import pdb; pdb.set_trace()
