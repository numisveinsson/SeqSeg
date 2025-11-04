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
from modules.datasets import vmr_directories
from modules.assembly import Segmentation
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


def trace_centerline(output_folder, image_file, case, model_folder, modality, img_shape, threshold, stepsize, old_point, old_radius, point, radius, seg_file=None, write_samples=True):

    take_time = False

    if seg_file:
        reader_seg, origin_im, size_im, spacing_im = sf.import_image(seg_file)

    reader_im, origin_im, size_im, spacing_im = sf.import_image(image_file)

    assembly_segs = Segmentation(case, image_file)

    model = UNet3DIsensee((img_shape[0], img_shape[1], img_shape[2], 1), num_class=1)
    unet = model.build()
    model_name = os.path.realpath(model_folder) + '/weights_unet.hdf5'
    unet.load_weights(model_name)

    initial_seed = assembly_segs.assembly.TransformPhysicalPointToIndex(point.tolist())
    storage = [{'ID':0, 'OLD': old_point, 'CURRENT': point, 'RADIUS': radius}]

    potential_branch = []

    list_surfaces = []
    list_centerlines = []
    list_points = []

    list_surf_branch = []
    list_cent_branch = []
    list_pts_branch = []
    chances = 0
    for i in range(500):

        print("\n** i is: " + str(i) + "**")
        try:
            start_time_loc = time.time()

            polydata_point = vf.points2polydata([point.tolist()])
            pfn = output_folder + 'points/point_'+case+'_'+str(i)+'.vtp'
            vf.write_geo(pfn, polydata_point)
            if take_time:
                print("\n Writing point: " + str(time.time() - start_time_loc) + " s\n")

            volume_size = 5
            size_extract, index_extract = sf.map_to_image(point, radius, volume_size, origin_im, spacing_im)

            cropped_volume = sf.extract_volume(reader_im, index_extract, size_extract)
            volume_fn = output_folder +'volumes/volume_'+case+'_'+str(i)+'.vtk'

            if seg_file:
                seg_volume = sf.extract_volume(reader_seg, index_extract, size_extract)
                seg_fn = output_folder +'volumes/volume_'+case+'_'+str(i)+'_truth.vtk'

            if write_samples:
                sitk.WriteImage(cropped_volume, volume_fn)
                sitk.WriteImage(seg_volume, seg_fn)
            if take_time:
                print("\n Extracting and writing volumes: " + str(time.time() - start_time_loc) + " s\n")

            predict = Prediction(unet, model_name, modality, volume_fn, seg_fn, img_shape, out_fn=output_folder+'predictions')
            predict.volume_prediction(1, threshold)
            predict.resample_prediction()
            predicted_vessel = predict.pred_label
            if take_time:
                print("\n Prediction, forward pass: " + str(time.time() - start_time_loc) + " s\n")

            seed = np.rint(np.array(size_extract)/2).astype(int).tolist()

            predicted_vessel = sf.remove_other_vessels(predicted_vessel, seed)
            #print("Now the components are: ")
            #labels, means = sf.connected_comp_info(predicted_vessel, True)

            predict.prediction = predict.prob_prediction
            predict.resample_prediction()
            prob_predicted_vessel = predict.pred_label
            pd_fn = output_folder +'predictions/seg_'+case+'_'+str(i)+'.vtk'


            surface = vf.evaluate_surface(predicted_vessel) # Marching cubes
            surface_smooth = vf.smooth_surface(surface, 8) # Smooth marching cubes

            vtkimage = vf.exportSitk2VTK(cropped_volume)
            length = predicted_vessel.GetSize()[0]*predicted_vessel.GetSpacing()[0]
            surface_smooth = vf.bound_polydata_by_image(vtkimage[0], surface_smooth, length*1/20)
            if take_time:
                print("\n Calc and smooth surface: " + str(time.time() - start_time_loc) + " s\n")

            sfn = output_folder +'surfaces/surf_'+case+'_'+str(i)+'.vtk'
            cfn = output_folder +'centerlines/cent_'+case+'_'+str(i)+'.vtk'
            if write_samples:
                sitk.WriteImage(predicted_vessel, pd_fn)
                vf.write_vtk_polydata(surface_smooth, sfn)

            centerline_poly = vmtkfs.calc_centerline(surface_smooth, "profileidlist", number = i)
            #centerline_poly = vf.get_largest_connected_polydata(centerline_poly)

            if take_time:
                print("\n Calc centerline: " + str(time.time() - start_time_loc) + " s\n")

            if centerline_poly.GetNumberOfPoints() < 5:
                print("\n Attempting with more smoothing and cropping \n")
                surface_smooth1 = vf.smooth_surface(surface, 12)
                surface_smooth1 = vf.bound_polydata_by_image(vtkimage[0], surface_smooth, length*1/10)
                centerline_poly1 = vmtkfs.calc_centerline(surface_smooth1, "profileidlist")
                if centerline_poly1.GetNumberOfPoints() > 5:
                    sfn = output_folder +'surfaces/surf_'+case+'_'+str(i)+'_1.vtk'
                    surface_smooth = surface_smooth1
                    cfn = output_folder +'centerlines/cent_'+case+'_'+str(i)+'_1.vtk'
                    centerline_poly = centerline_poly1

            if write_samples:
                vmtkfs.write_centerline(centerline_poly, cfn)
                vf.write_vtk_polydata(surface_smooth, sfn)
            if take_time:
                print("\n Writing pred, surf, cent: " + str(time.time() - start_time_loc) + " s\n")

            # # enter global surface post threshold and connected filter
            # # see if point is aiming inside, if so dont save
            # assembly = sitk.BinaryThreshold(assembly_segs.assembly, lowerThreshold=0.5, upperThreshold=1)
            # assembly = sf.remove_other_vessels(assembly, initial_seed)
            # #assembly_surface = vf.evaluate_surface(assembly, 1)

            point_tree, radius_tree, angle_change = vf.get_next_points(centerline_poly, point, old_point, old_radius, assembly_segs.assembly)
            if take_time:
                print("\n Calc next point: " + str(time.time() - start_time_loc) + " s\n")

            assembly_segs.add_segmentation(prob_predicted_vessel, index_extract, size_extract)
            if take_time:
                print("\n Adding to seg volume: " + str(time.time() - start_time_loc) + " s\n")

            if i % 20 == 0:
                sitk.WriteImage(assembly_segs.assembly, output_folder +'assembly/assembly_'+case+'_'+str(i)+'.vtk')
                assembly = sitk.BinaryThreshold(assembly_segs.assembly, lowerThreshold=0.5, upperThreshold=1)
                assembly = sf.remove_other_vessels(assembly, initial_seed)
                surface_assembly = vf.evaluate_surface(assembly, 1)
                vf.write_vtk_polydata(surface_assembly, output_folder +'assembly/assembly_surface_'+case+'_'+str(i)+'.vtk')

            if point_tree.size != 0:
                old_point = point
                old_radius = radius

            angle = angle_change[0]
            point = point_tree[0]
            radius = radius_tree[0]*1.1
            print("Next radius is: " + str(radius))

            list_pts_branch.append(polydata_point)
            list_surf_branch.append(surface_smooth)
            list_cent_branch.append(centerline_poly)
            chances = 0
            chance_enlarge = False

            if len(radius_tree) > 1:

                for i in range(1, len(radius_tree)):
                    dict = {}
                    dict["old point"] = old_point
                    dict["current point"] = point_tree[i]
                    dict["old radius"] = old_radius
                    dict["radius"] = radius_tree[i]
                    dict["angle change"] = angle_change[i]
                    potential_branch.append(dict)
            print("\n This location done: " + str(time.time() - start_time_loc) + " s\n")

        except:

            if i == 0:
                "Didnt work for first surface"
                break

            #import pdb; pdb.set_trace()
            if chances < 3 and angle < 45:
                save_point = point
                save_oldpoint = old_point
                save_radius = radius
                save_oldradius = old_radius

                print("Giving chance for surface: " + str(i))
                old_vector = point-old_point
                old_vector = old_vector/np.linalg.norm(old_vector)
                old_point = point
                old_radius = radius

                point = old_point + old_radius*old_vector
                radius = old_radius

                chances = chances + 1

            elif chances == 3 and not chance_enlarge:

                print("Let's backtrack for surface: " + str(i) + ", and enlarge the volume")
                point = save_point
                radius = save_radius*1.5
                old_point = save_oldpoint
                old_radius = save_oldradius
                chance_enlarge = True


            else:


                print("\n*** Error for surface: \n" + str(i))
                print("\n Moving onto another branch")

                if list_surf_branch:
                    final_surface = vf.appendPolyData(list_surf_branch)
                    vf.write_vtk_polydata(final_surface, output_folder+'/branch_'+case+'_'+str(i)+'_surfaces.vtp')

                    final_centerline = vf.appendPolyData(list_cent_branch)
                    vf.write_vtk_polydata(final_centerline, output_folder+'/branch_'+case+'_'+str(i)+'_centerlines.vtp')

                    final_points = vf.appendPolyData(list_pts_branch)
                    vf.write_vtk_polydata(final_points, output_folder+'/branch_'+case+'_'+str(i)+'_points.vtp')

                if not potential_branch:
                    break

                new_branch = potential_branch.pop(0)
                old_point = new_branch["old point"]
                old_radius = new_branch["old radius"]
                point = new_branch["current point"]
                radius = new_branch["radius"]
                angle = new_branch["angle change"]

                list_centerlines.extend(list_cent_branch)
                list_surfaces.extend(list_surf_branch)
                list_points.extend(list_pts_branch)

                list_surf_branch = []
                list_cent_branch = []
                list_pts_branch = []

    return list_centerlines, list_surfaces, list_points, assembly_segs.assembly

if __name__=='__main__':

    ## Directories
    dir_output = '/Users/numisveinsson/Documents/Berkeley/Research/Automatic_Centerline_ML/output2_test3/'
    directory_data = '/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/vascular_data_3d/'
    dir_model_weights = '/Users/numisveinsson/Documents/Berkeley/Research/BloodVessel_UNet3D/output/test18/'

    ## Information
    case = '0146_1001'
    modality = 'ct'
    nn_input_shape = [64, 64, 64] # Input shape for NN
    threshold = 0.5 # Threshold for binarization of prediction
    stepsize = 1 # Step size along centerline (proportional to radius at the point)
    dir_image, dir_seg, dir_cent, dir_surf = vmr_directories(directory_data, case)

    ## Create directories for results
    create_directories(dir_output)

    ## Get inital seed point + radius
    i = 0
    old_seed, old_radius = vf.get_seed(dir_cent, i, 50)
    initial_seed, initial_radius = vf.get_seed(dir_cent, i, 70)
    vf.write_geo(dir_output+ 'points/0_seed_point.vtp', vf.points2polydata([old_seed.tolist()]))

    ## Trace centerline
    centerlines, surfaces, points, assembly = trace_centerline(dir_output, dir_image, case, dir_model_weights, modality, nn_input_shape, threshold, stepsize, old_seed, old_radius, initial_seed, initial_radius, dir_seg, write_samples=True)

    print("\nTotal calculation time is: " + str((time.time() - start_time)/60) + " min\n")
    import pdb; pdb.set_trace()
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
    surface_smooth = vf.smooth_surface(assembly_surface, 8)
    vf.write_vtk_polydata(assembly_surface, dir_output+'/final_assembly'+case+'_'+str(i)+'_surface_smooth.vtp')

    import pdb; pdb.set_trace()
    assembly_centerline = vmtkfs.calc_centerline(assembly_surface, "carotidprofiles")
    vf.write_vtk_polydata(assembly_centerline, dir_output+'/final_assembly'+case+'_'+str(i)+'_centerline.vtp')
