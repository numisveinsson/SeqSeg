import faulthandler
import time

import os
import sys
import argparse
import SimpleITK as sitk

from seqseg.modules import sitk_functions as sf
from seqseg.modules import vtk_functions as vf
from seqseg.modules import initialization as init
from seqseg.modules.assembly import calc_centerline_global
from seqseg.modules.tracing import trace_centerline
from seqseg.modules.datasets import get_testing_samples
from seqseg.modules.params import load_yaml
from seqseg.modules.capping import cap_surface
from importlib import resources
import yaml


sys.stdout.flush()
start_time = time.time()
faulthandler.enable()


def load_yaml_config(config_name):
    with resources.files('seqseg.config').joinpath(f'{config_name}.yaml').open('r') as f:
        return yaml.safe_load(f)


def create_directories(output_folder, write_samples):
    try:
        os.mkdir(output_folder)
    except Exception as e:
        print(e)
    try:
        os.mkdir(output_folder+'errors')
    except Exception as e:
        print(e)
    try:
        os.mkdir(output_folder+'assembly')
    except Exception as e:
        print(e)
    try:
        os.mkdir(output_folder+'simvascular')
    except Exception as e:
        print(e)
    # and sub-directories for SimVascular: Images, Paths, Models
    try:
        os.mkdir(output_folder+'simvascular/Images')
    except Exception as e:
        print(e)
    try:
        os.mkdir(output_folder+'simvascular/Paths')
    except Exception as e:
        print(e)
    try:
        os.mkdir(output_folder+'simvascular/Models')
    except Exception as e:
        print(e)

    if write_samples:
        try:
            os.mkdir(output_folder+'volumes')
        except Exception as e:
            print(e)
        try:
            os.mkdir(output_folder+'predictions')
        except Exception as e:
            print(e)
        try:
            os.mkdir(output_folder+'centerlines')
        except Exception as e:
            print(e)
        try:
            os.mkdir(output_folder+'surfaces')
        except Exception as e:
            print(e)
        try:
            os.mkdir(output_folder+'points')
        except Exception as e:
            print(e)
        try:
            os.mkdir(output_folder+'animation')
        except Exception as e:
            print(e)


def main():
    """ Set up"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', '--data_directory',
                        type=str,
                        help='Name of the folder containing the testing data')
    parser.add_argument('-nnunet_results_path', '--nnunet_results_path',
                        type=str,
                        help='Path to nnUNet results folder')
    parser.add_argument('-nnunet_type', '--nnunet_type',
                        choices=['3d_fullres', '2d'],
                        default='3d_fullres',
                        type=str,
                        help='Type of nnUNet model to use, eg 3d_fullres/2d')
    parser.add_argument('-train_dataset', '--train_dataset',
                        type=str,
                        default='Dataset010_SEQCOROASOCACT',
                        help="""Name of dataset used to train nnUNet
                             , eg Dataset010_SEQCOROASOCACT""")
    parser.add_argument('-fold', '--fold',
                        default='all',
                        type=str,
                        help='Which fold to use for nnUNet model')
    parser.add_argument('-img_ext', '--img_ext',
                        type=str,
                        help='Image extension, eg .nii.gz')
    parser.add_argument('-outdir', '--outdir',
                        type=str,
                        help='Output directory')
    parser.add_argument('-scale', '--scale',
                        default=1,
                        type=float,
                        help="""Whether to scale image data,
                             needed if units for nnUNet model
                             and testing data are different, eg 0.1
                             if model is in cm and data is in mm""")
    parser.add_argument('-start', '--start',
                        default=0,
                        type=int,
                        help='In the list of testing samples, where to start')
    parser.add_argument('-stop', '--stop',
                        default=-1,
                        type=int,
                        help='In the list of testing samples, where to stop')
    parser.add_argument('-max_n_steps', '--max_n_steps',
                        default=1000,
                        type=int,
                        help='Max number of steps to take')
    parser.add_argument('-max_n_steps_per_branch', '--max_n_steps_per_branch',
                        default=100,
                        type=int,
                        help='Max number of steps to take per branch')
    parser.add_argument('-max_n_branches', '--max_n_branches',
                        default=100,
                        type=int,
                        help='Max number of branches to take')
    parser.add_argument('-unit',
                        '--unit',
                        default='cm',
                        type=str,
                        help='Unit of medical image')
    parser.add_argument('-config_name', '--config_name',
                        default='global',
                        type=str,
                        help='Name of configuration file')
    parser.add_argument('-pt_centerline', '--pt_centerline',
                        default=50,
                        type=int,
                        help='Use point centerline')
    parser.add_argument('-num_seeds_centerline', '--num_seeds_centerline',
                        default=1,
                        type=int,
                        help='Number of seeds for centerline')
    parser.add_argument('-write_steps', '--write_steps',
                        default=0,
                        type=int,
                        help='Whether to write all intermediate steps')
    parser.add_argument('-extract_global_centerline', '--extract_global_centerline',
                        default=0,
                        type=int,
                        help='Whether to extract global centerline after segmentation')
    parser.add_argument('-cap_surface_cent', '--cap_surface_cent',
                        default=0,
                        type=int,
                        help='Whether to cap surface centerline')
    args = parser.parse_args()

    # print(args)

    # Load configuration file
    global_config = load_yaml_config(args.config_name)
    print(f"\nUsing config file: {args.config_name}")

    dir_output0 = args.outdir
    data_dir = args.data_directory

    # Make sure the output directory exists
    try:
        os.mkdir(dir_output0)
    except Exception as e:
        print(e)

    unit = args.unit
    max_step_size = args.max_n_steps
    max_n_branches = args.max_n_branches
    max_n_steps_per_branch = args.max_n_steps_per_branch
    write_samples = args.write_steps
    take_time = global_config['TIME_ANALYSIS']
    calc_global_centerline = args.extract_global_centerline
    cap_surface_cent = args.cap_surface_cent

    dataset = args.train_dataset
    fold = args.fold
    img_format = args.img_ext
    scale = args.scale
    test_name = args.nnunet_type
    pt_centerline = args.pt_centerline
    num_seeds = args.num_seeds_centerline

    # Weight directory
    dir_model_weights = dataset+'/nnUNetTrainer__nnUNetPlans__'+test_name
    if args.nnunet_results_path is not None:
        weight_dir_nnunet = args.nnunet_results_path
        dir_model_weights = os.path.join(weight_dir_nnunet, dir_model_weights,)

    testing_samples, directory_data = get_testing_samples(dataset, data_dir)
    print("Testing samples about to run:")
    for sample in testing_samples:
        print(sample)

    # Choose testing samples
    if args.stop == -1:
        args.stop = len(testing_samples)

    print(f"Running from {args.start} to {args.stop} of {len(testing_samples)} samples")

    for i, test_case in enumerate(testing_samples[args.start: args.stop]):

        print(f"\nProcessing test case {i+args.start+1} of {len(testing_samples)}: {test_case}")
        # print(f'\n{test_case}\n')

        (dir_output, dir_image, dir_seg, dir_cent,
         case, i, json_file_present) = init.process_init(test_case,
                                                         directory_data,
                                                         dir_output0,
                                                         img_format,
                                                         test_name)

        # Create directories for results
        create_directories(dir_output, write_samples)

        (potential_branches,
         initial_seeds) = init.initialization(json_file_present,
                                              test_case, dir_output, dir_cent,
                                              directory_data, unit,
                                              pt_centerline, num_seeds,
                                              write_samples)

        # print to .txt file all outputs
        if not global_config['DEBUG']:
            # write to file
            sys.stdout = open(dir_output+"/out.txt", "w")
        else:
            print("\nStart tracking with debug mode on")
            # import pdb
            # pdb.set_trace()
        if json_file_present:
            print("\nWe got seed point from json file")
        else:
            print("\nWe did not get seed point from json file")
        print(test_case)
        print(f"Number of initial points: {len(potential_branches)}")
        print(f"Time is: {time.time() - start_time:.2f} sec")

        # Trace centerline
        (centerlines, surfaces, points, inside_pts, assembly_obj,
         vessel_tree, n_steps_taken) = trace_centerline(
            dir_output,
            dir_image,
            case,
            dir_model_weights,
            fold,
            potential_branches,
            max_step_size,
            max_n_branches,
            max_n_steps_per_branch,
            global_config,
            unit,
            scale,
            dir_seg,
            write_samples=write_samples
        )

        print("\nTotal calculation time is:"
              + f"{((time.time() - start_time)/60):.2f} min\n")

        if take_time:
            vessel_tree.time_analysis()

        # End points
        # if calc_global_centerline:
        #     end_points = vessel_tree.get_end_points()
        #     in_source = end_points[0].tolist()
        #     in_target_lists = [point.tolist() for point in end_points[1:]]
        #     in_target = []
        #     for target in in_target_lists:
        #         in_target += target

        # Plot tree info
        if global_config['TREE_ANALYSIS']:
            # vessel_tree.create_tree_graph(dir_output)
            # vessel_tree.create_tree_graph_smaller(dir_output)
            vessel_tree.create_tree_polydata_v1(dir_output)
            vessel_tree.create_tree_polydata_v2(dir_output)
            vessel_tree.plot_radius_distribution(dir_output)

        # Assembly work
        assembly_org = assembly_obj.assembly
        # assembly_ups = assembly_obj.upsample_sitk()
        # sitk.WriteImage(assembly_org, dir_output+'/'+case+'_assembly_'
        #                 + test_name + '_'+str(i)+'.mha')

        assembly = assembly_org

        assembly_binary = sitk.BinaryThreshold(assembly, lowerThreshold=0.5,
                                               upperThreshold=1)
        sitk.WriteImage(assembly_binary, dir_output+'/'+case+'_binary_seg_'
                        + test_name + '_' + str(i) + '.mha')
        sitk.WriteImage(assembly, dir_output+'/'+case+'_prob_seg_'
                        + test_name + '_' + str(i) + '.mha')

        assembly_binary = sf.keep_component_seeds(assembly_binary,
                                                  initial_seeds)
        sitk.WriteImage(assembly_binary, dir_output0+'/'+case
                        + '_seg_containing_seeds_'
                        + str(n_steps_taken) + '_steps' + '.mha')

        assembly_surface = vf.evaluate_surface(assembly_binary, 1)
        vf.write_vtk_polydata(assembly_surface, dir_output + '/' + case
                              + '_surface_mesh_nonsmooth_' + str(n_steps_taken)
                              + '_steps' + '.vtp')
        surface_smooth = vf.smooth_polydata(assembly_surface, iteration=75, smoothingFactor=0.1)
        vf.write_vtk_polydata(surface_smooth, dir_output0 + '/' + case
                              + '_surface_mesh_smooth_' + str(n_steps_taken)
                              + '_steps' + '.vtp')

        if len(centerlines) > 0:
            final_surface = vf.appendPolyData(surfaces)
            final_centerline = vf.appendPolyData(centerlines)
            final_points = vf.appendPolyData(points)

            vf.write_vtk_polydata(final_surface,    dir_output+'/all_'+case+'_'
                                  + test_name + '_'+str(i)+'_'
                                  + str(n_steps_taken)
                                  + '_surfaces.vtp')
            vf.write_vtk_polydata(final_centerline, dir_output+'/all_'+case+'_'
                                  + test_name + '_'+str(i)+'_'
                                  + str(n_steps_taken)
                                  + '_centerlines.vtp')
            vf.write_vtk_polydata(final_points,     dir_output+'/all_'+case+'_'
                                  + test_name + '_'+str(i)+'_'
                                  + str(n_steps_taken)
                                  + '_points.vtp')
        if calc_global_centerline:
            # Calculate global centerline
            global_centerline, targets, success = calc_centerline_global(
                assembly_binary,
                initial_seeds)
            # if centerline is not None
            if success or len(targets) > 0:
                vf.write_vtk_polydata(global_centerline, dir_output0 + '/'
                                      + case + '_centerline_'
                                      + str(n_steps_taken)
                                      + '_steps' + '.vtp')
                # write targets
                targets_pd = vf.points2polydata([target.tolist()
                                                for target in targets])
                vf.write_vtk_polydata(targets_pd, dir_output+'/'
                                      + case + '_' + test_name + '_'+str(i)+'_'
                                      + str(n_steps_taken)
                                      + '_targets.vtp')

                if cap_surface_cent:
                    capped_surface, capped_seg = cap_surface(
                        pred_surface=assembly_surface,
                        centerline=global_centerline,
                        pred_seg=assembly_binary,
                        file_name=case,
                        outdir=dir_output,
                        targets=targets)
                    vf.write_vtk_polydata(capped_surface, dir_output+'/'
                                        + case + '_' + test_name + '_'+str(i)+'_'
                                        + str(n_steps_taken)
                                        + '_capped_surface.vtp')
                    sitk.WriteImage(capped_seg, dir_output+'/'
                                    + case + '_' + str(n_steps_taken)
                                    + '_capped_seg.mha')

        if global_config['PREVENT_RETRACE']:
            if len(inside_pts) > 0:
                final_inside_pts = vf.appendPolyData(inside_pts)
                vf.write_vtk_polydata(final_inside_pts, dir_output + '/final_'
                                      + case + '_' + test_name + '_'+str(i)+'_'
                                      + str(n_steps_taken)
                                      + '_inside_points.vtp')

        if not global_config['DEBUG']:
            # close the file
            sys.stdout.close()
            sys.stdout = sys.__stdout__

    print("\nTotal calculation time for all cases is: ")
    print(f"{((time.time() - start_time)/60):.2f} min\n")


if __name__ == '__main__':
    main()
