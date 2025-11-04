import faulthandler
import time

import os
import sys
import argparse
import SimpleITK as sitk

from modules import sitk_functions as sf
from modules import vtk_functions as vf
from modules import initialization as init
from modules.assembly import calc_centerline_global
from modules.tracing import trace_centerline
from modules.datasets import get_testing_samples
from modules.params import load_yaml
from modules.capping import cap_surface
from modules.sweep import run_global_segmentation

sys.stdout.flush()
start_time = time.time()
faulthandler.enable()


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
    parser.add_argument('-seqseg_test_name', '--seqseg_test_name',
                        default='3d_fullres',
                        type=str,
                        help='Name of nnUNet test to use, eg 3d_fullres/2d')
    parser.add_argument('-seqseg_train_dataset', '--seqseg_train_dataset',
                        type=str,
                        default='Dataset010_SEQCOROASOCACT',
                        help="""Name of dataset used to train nnUNet
                             , eg Dataset010_SEQCOROASOCACT""")
    parser.add_argument('-seqseg_fold', '--seqseg_fold',
                        default='all',
                        type=str,
                        help='Which fold to use for nnUNet model')
    parser.add_argument('-img_ext', '--img_ext',
                        type=str,
                        help='Image extension, eg .nii.gz')
    parser.add_argument('-outdir', '--outdir',
                        type=str,
                        help='Output directory')
    parser.add_argument('-seqseg_scale', '--seqseg_scale',
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
                        default=10000,
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
    parser.add_argument('-global_test_name', '--global_test_name',
                        default='3d_fullres',
                        type=str,
                        help='Name of nnUNet test to use for global segmentation')
    parser.add_argument('-global_fold', '--global_fold',
                        default='all',
                        type=str,
                        help='Which fold to use for nnUNet model for global segmentation')
    parser.add_argument('-global_scale', '--global_scale',
                        default=1,
                        type=float,
                        help="""Whether to scale image data for global segmentation,
                             needed if units for nnUNet model and testing data are different,
                             eg 0.1 if model is in cm and data is in mm""")
    parser.add_argument('-global_train_dataset', '--global_train_dataset',
                        type=str,
                        default='Dataset012_COROASOCACT',
                        help="""Name of dataset used to train nnUNet
                             , eg Dataset012_COROASOCACT""")
    args = parser.parse_args()

    print(args)

    global_config = load_yaml("./config/"+args.config_name+".yaml")
    print(f"Using config file: {args.config_name}")

    dir_output0 = args.outdir
    data_dir = args.data_directory

    # Make sure the output directory exists
    try:
        os.mkdir(dir_output0)
    except Exception as e:
        print(e)

    img_format = args.img_ext
    unit = args.unit
    max_step_size = args.max_n_steps
    max_n_branches = args.max_n_branches
    write_samples = global_config['WRITE_STEPS']
    take_time = global_config['TIME_ANALYSIS']
    calc_global_centerline = global_config['GLOBAL_CENTERLINE']

    seqseg_dataset = args.seqseg_train_dataset
    seqseg_fold = args.seqseg_fold
    seqseg_test_name = args.seqseg_test_name
    seqseg_scale = args.seqseg_scale

    global_dataset = args.global_train_dataset
    global_fold = args.global_fold
    global_test_name = args.global_test_name
    global_scale = args.global_scale

    pt_centerline = args.pt_centerline
    num_seeds = args.num_seeds_centerline

    # Weight directory
    dir_model_weights_seqseg = (seqseg_dataset
                                + '/nnUNetTrainer__nnUNetPlans__'
                                + seqseg_test_name)
    dir_model_weights_global = (global_dataset
                                + '/nnUNetTrainer__nnUNetPlans__'
                                + global_test_name)
    weight_dir_nnunet = args.nnunet_results_path
    dir_model_weights_seqseg = os.path.join(weight_dir_nnunet, dir_model_weights_seqseg,)
    dir_model_weights_global = os.path.join(weight_dir_nnunet, dir_model_weights_global,)
    print("Using nnUNet model weights from:")
    print(dir_model_weights_seqseg)
    print("Using nnUNet model weights for global segmentation from:")
    print(dir_model_weights_global)

    testing_samples, directory_data = get_testing_samples(seqseg_dataset,
                                                          data_dir)
    print("Testing samples about to run:")
    for sample in testing_samples:
        print(sample)

    # Choose testing samples
    if args.stop == -1:
        args.stop = len(testing_samples)

    for test_case in testing_samples[args.start: args.stop]:

        print(f'\n{test_case}\n')

        (dir_output, dir_image, dir_seg, dir_cent,
         case, i, json_file_present) = init.process_init(test_case,
                                                         directory_data,
                                                         dir_output0,
                                                         img_format,
                                                         seqseg_test_name)

        # Get sweep segmentation
        pred_sweep, prob_pred_sweep = run_global_segmentation(
            dir_image=dir_image,
            model_folder=dir_model_weights_global,
            fold=global_fold,
            scale=global_scale
        )
        # Write sweep segmentation
        sitk.WriteImage(pred_sweep, dir_output0+'/'+case+'_sweep_seg.mha')

        # Create directories for results
        create_directories(dir_output, write_samples)

        (potential_branches,
         initial_seeds) = init.initialize_from_seg(pred_sweep, dir_output)

        # print to .txt file all outputs
        if not global_config['DEBUG']:
            # write to file
            sys.stdout = open(dir_output+"/out.txt", "w")
        else:
            print("Start tracing with debug mode on")

        print(test_case)
        print(f"Initial points: {potential_branches}")
        print(f"Time is: {time.time()}")

        # Trace centerline
        (centerlines, surfaces, points, inside_pts, assembly_obj,
         vessel_tree, n_steps_taken) = trace_centerline(
            dir_output,
            dir_image,
            case,
            dir_model_weights_seqseg,
            seqseg_fold,
            potential_branches,
            max_step_size,
            max_n_branches,
            global_config,
            unit,
            seqseg_scale,
            dir_seg,
            start_seg=pred_sweep  # Note this is binary (should be prob)
        )

        print("\nTotal calculation time is:"
              + str((time.time() - start_time)/60) + " min\n")

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
        print("""\nTotal calculation time is: "
              + str((time.time() - start_time)/60) + " min\n""")
        # sitk.WriteImage(assembly_org, dir_output+'/'+case+'_assembly_'
        #                 + test_name + '_'+str(i)+'.mha')

        assembly = assembly_org

        assembly_binary = sitk.BinaryThreshold(assembly, lowerThreshold=0.5,
                                               upperThreshold=1)
        sitk.WriteImage(assembly_binary, dir_output+'/'+case+'_raw_seg_'
                        + seqseg_test_name + '_' + str(i) + '.mha')

        assembly_binary = sf.keep_component_seeds(assembly_binary,
                                                  initial_seeds)
        sitk.WriteImage(assembly_binary, dir_output0+'/'+case+'_segmentation_'
                        + str(n_steps_taken) + '_steps' + '.mha')

        assembly_surface = vf.evaluate_surface(assembly_binary, 1)
        vf.write_vtk_polydata(assembly_surface, dir_output + '/' + case
                              + '_surface_' + str(n_steps_taken)
                              + '_steps' + '.vtp')
        surface_smooth = vf.smooth_polydata(assembly_surface)
        vf.write_vtk_polydata(surface_smooth, dir_output0 + '/' + case
                              + '_surface_' + str(n_steps_taken)
                              + '_steps' + '.vtp')

        final_surface = vf.appendPolyData(surfaces)
        final_centerline = vf.appendPolyData(centerlines)
        final_points = vf.appendPolyData(points)

        vf.write_vtk_polydata(
            final_surface, dir_output+'/all_'+case+'_'
            + seqseg_test_name + '_'+str(i)+'_'+str(n_steps_taken)
            + '_surfaces.vtp')
        vf.write_vtk_polydata(
            final_centerline, dir_output+'/all_'+case+'_'
            + seqseg_test_name + '_'+str(i)+'_'+str(n_steps_taken)
            + '_centerlines.vtp')
        vf.write_vtk_polydata(
            final_points, dir_output+'/all_'+case+'_'
            + seqseg_test_name + '_'+str(i)+'_'+str(n_steps_taken)
            + '_points.vtp')
        if calc_global_centerline:
            # Calculate global centerline
            global_centerline, targets, success = calc_centerline_global(
                assembly_binary,
                initial_seeds)
            # if centerline is not None
            if success:
                vf.write_vtk_polydata(
                    global_centerline, dir_output0 + '/'
                    + case + '_centerline_' + str(n_steps_taken)
                    + '_steps' + '.vtp')
                # write targets
                targets_pd = vf.points2polydata([target.tolist()
                                                for target in targets])
                vf.write_vtk_polydata(
                    targets_pd, dir_output+'/'
                    + case + '_' + seqseg_test_name + '_'+str(i)+'_'
                    + str(n_steps_taken)
                    + '_targets.vtp')

                capped_surface, capped_seg = cap_surface(
                    pred_surface=assembly_surface,
                    centerline=global_centerline,
                    pred_seg=assembly_binary,
                    file_name=case,
                    outdir=dir_output,
                    targets=targets)
                vf.write_vtk_polydata(
                    capped_surface, dir_output+'/'
                    + case + '_' + seqseg_test_name + '_'+str(i)+'_'
                    + str(n_steps_taken)+'_capped_surface.vtp')
                sitk.WriteImage(capped_seg, dir_output+'/'
                                + case + '_' + str(n_steps_taken)
                                + '_capped_seg.mha')

        if global_config['PREVENT_RETRACE']:
            final_inside_pts = vf.appendPolyData(inside_pts)
            vf.write_vtk_polydata(
                final_inside_pts, dir_output + '/final_'
                + case + '_' + seqseg_test_name + '_'+str(i)+'_'
                + str(n_steps_taken)+'_inside_points.vtp')

        if not global_config['DEBUG']:
            # close the file
            sys.stdout.close()
            sys.stdout = sys.__stdout__

    print("\nTotal calculation time is: ")
    print(f"{((time.time() - start_time)/60):.2f} min\n")


if __name__ == '__main__':
    main()
