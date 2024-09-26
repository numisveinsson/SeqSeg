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


if __name__ == '__main__':
    """ Set up"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', '--data_directory',
                        type=str,
                        help='Name of the folder containing the testing data')
    parser.add_argument('-test_name', '--test_name',
                        default='3d_fullres',
                        type=str,
                        help='Name of nnUNet test to use, eg 3d_fullres/2d')
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
                             and testing data are different""")
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
    parser.add_argument('-gpu',
                        '--gpu',
                        default=False,
                        type=bool,
                        help='Use GPU for nn')
    args = parser.parse_args()

    print(args)

    #       [name , dataset, fold, modality, json file present, unit]
    #       note: fold is either 0,1,2,3 or 'all'
    #       note: json file present is either True or False
    # tests = [['3d_fullres', 'Dataset002_SEQAORTAS', 0],
    #          ['3d_fullres', 'Dataset005_SEQAORTANDFEMOMR', 'all', False],
    #          ['3d_fullres', 'Dataset006_SEQAORTANDFEMOCT', 'all', True,
    #           '.vtk', 0.1],
    #          # 0.1 here means scaling (model is cm but data is mm)
    #          ['3d_fullres', 'Dataset007_SEQPULMONARYMR', 'all', False],
    #          ['3d_fullres', 'Dataset009_SEQAORTASMICCT', 'all', True,
    #           '.nrrd', 1],
    #          # 1 here means no scaling (model and data are both mm)
    #          ['3d_fullres', 'Dataset010_SEQCOROASOCACT', 'all', True,
    #           '.nrrd', 1]
    #          ]

    global_config = load_yaml("./config/"+args.config_name+".yaml")
    print(f"Using config file: {args.config_name}")

    dir_output0 = args.outdir
    data_dir = args.data_directory

    unit = args.unit
    max_step_size = args.max_n_steps
    max_n_branches = args.max_n_branches
    gpu_avail = args.gpu
    write_samples = global_config['WRITE_STEPS']
    take_time = global_config['TIME_ANALYSIS']
    calc_global_centerline = global_config['GLOBAL_CENTERLINE']

    dataset = args.train_dataset 
    fold = args.fold            
    img_format = args.img_ext   
    scale = args.scale          
    test_name = args.test_name  

    # Weight directory
    dir_model_weights = dataset+'/nnUNetTrainer__nnUNetPlans__'+test_name

    testing_samples, directory_data = get_testing_samples(dataset, data_dir)
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
                                                         test_name)

        # Create directories for results
        create_directories(dir_output, write_samples)

        (potential_branches,
         initial_seeds) = init.initialization(json_file_present,
                                              test_case, dir_output, dir_cent,
                                              directory_data, unit,
                                              write_samples)

        # print to .txt file all outputs
        if not global_config['DEBUG']:
            # write to file
            sys.stdout = open(dir_output+"/out.txt", "w")
        else:
            print("Start tracing with debug mode on")
            # import pdb
            # pdb.set_trace()

        print(test_case)
        print(f"Initial points: {potential_branches}")
        print(f"Time is: {time.time()}")

        # Trace centerline
        (centerlines, surfaces, points, inside_pts, assembly_obj,
         vessel_tree, n_steps_taken) = trace_centerline(dir_output,
                                                        dir_image,
                                                        case,
                                                        dir_model_weights,
                                                        fold,
                                                        potential_branches,
                                                        max_step_size,
                                                        max_n_branches,
                                                        global_config,
                                                        unit,
                                                        scale,
                                                        dir_seg,
                                                        gpu_avail)

        print("\nTotal calculation time is:"
              + str((time.time() - start_time)/60) + " min\n")

        if take_time:
            vessel_tree.time_analysis()

        # End points
        if calc_global_centerline:
            end_points = vessel_tree.get_end_points()
            in_source = end_points[0].tolist()
            in_target_lists = [point.tolist() for point in end_points[1:]]
            in_target = []
            for target in in_target_lists:
                in_target += target

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
        name = 'original'
        # for assembly,name in zip([assembly_ups, assembly_org],
        #                          ['upsampled', 'original']):
        assembly_binary = sitk.BinaryThreshold(assembly, lowerThreshold=0.5,
                                               upperThreshold=1)
        sitk.WriteImage(assembly_binary, dir_output+'/'+case+'_seg_'
                        + test_name + '_' + str(i) + '.mha')

        assembly_binary = sf.keep_component_seeds(assembly_binary,
                                                  initial_seeds)
        sitk.WriteImage(assembly_binary, dir_output0+'/'+case+'_seg_rem_'
                        + test_name + '_' + str(i) + '.mha')

        assembly_surface = vf.evaluate_surface(assembly_binary, 1)
        vf.write_vtk_polydata(assembly_surface, dir_output0+'/'+case
                              + '_surface_' + test_name + '_'
                              + str(n_steps_taken) + '.vtp')
        for level in [10, 40]:  # range(10,50,10):
            surface_smooth = vf.smooth_surface(assembly_surface, level)
            vf.write_vtk_polydata(surface_smooth, dir_output+'/final_assembly_'
                                  + name + '_'+case+'_'+test_name + '_'+str(i)
                                  + '_'+str(n_steps_taken)+'_'+str(level)
                                  + '_surface_smooth.vtp')

        final_surface = vf.appendPolyData(surfaces)
        final_centerline = vf.appendPolyData(centerlines)
        final_points = vf.appendPolyData(points)

        vf.write_vtk_polydata(final_surface,    dir_output+'/final_'+case+'_'
                              + test_name + '_'+str(i)+'_'+str(n_steps_taken)
                              + '_surfaces.vtp')
        vf.write_vtk_polydata(final_centerline, dir_output+'/final_'+case+'_'
                              + test_name + '_'+str(i)+'_'+str(n_steps_taken)
                              + '_centerlines.vtp')
        vf.write_vtk_polydata(final_points,     dir_output+'/final_'+case+'_'
                              + test_name + '_'+str(i)+'_'+str(n_steps_taken)
                              + '_points.vtp')
        if calc_global_centerline:
            # Calculate global centerline
            global_centerline = calc_centerline_global(assembly_binary,
                                                       initial_seeds)
            vf.write_vtk_polydata(global_centerline, dir_output+'/final_'
                                  + case + '_' + test_name + '_'+str(i)+'_'
                                  + str(n_steps_taken)
                                  + '_global_centerline.vtp')

        if global_config['PREVENT_RETRACE']:
            final_inside_pts = vf.appendPolyData(inside_pts)
            vf.write_vtk_polydata(final_inside_pts, dir_output + '/final_'
                                  + case + '_' + test_name + '_'+str(i)+'_'
                                  + str(n_steps_taken)+'_inside_points.vtp')

        if not global_config['DEBUG']:
            # close the file
            sys.stdout.close()
            sys.stdout = sys.__stdout__

    print("\nTotal calculation time is: ")
    print(f"{((time.time() - start_time)/60):.2f} min\n")
