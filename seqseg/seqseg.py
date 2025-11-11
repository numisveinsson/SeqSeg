"""
SeqSeg: Sequential Vessel Segmentation and Centerline Tracing

Main entry point for the SeqSeg vessel tracing pipeline that combines:
- nnU-Net deep learning segmentation
- Sequential volume extraction and processing
- Centerline tracing with bifurcation detection
- Global assembly and post-processing

This script processes medical images to extract vessel centerlines and create
segmentations suitable for computational fluid dynamics and medical analysis.
"""

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

# Initialize global settings for error handling and timing
sys.stdout.flush()
start_time = time.time()
faulthandler.enable()  # Enable traceback on segfaults for debugging


def load_yaml_config(config_name):
    """
    Load YAML configuration file with fallback mechanism.
    
    Attempts to load configuration using importlib.resources (package-aware),
    then falls back to direct file path for development environments.
    
    Args:
        config_name (str): Name of the config file without .yaml extension
        
    Returns:
        dict: Parsed YAML configuration
        
    Raises:
        FileNotFoundError: If configuration file cannot be found
    """
    try:
        # Try to use importlib.resources first (preferred method for installed packages)
        with resources.files('seqseg.config').joinpath(f'{config_name}.yaml').open('r') as f:
            return yaml.safe_load(f)
    except (ImportError, FileNotFoundError, ModuleNotFoundError):
        # Fallback to direct file path for development/editable installs
        config_dir = os.path.join(os.path.dirname(__file__), 'config')
        config_path = os.path.join(config_dir, f'{config_name}.yaml')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"Configuration file '{config_name}.yaml' not found in {config_dir}")


def create_directories(output_folder, write_samples):
    """
    Create directory structure for SeqSeg output files.
    
    Sets up organized folders for different types of outputs including
    final results, debug information, and optional intermediate files.
    
    Args:
        output_folder (str): Base output directory path
        write_samples (bool): Whether to create directories for intermediate files
    """
    # Core directories for all runs
    directories = [
        output_folder,                    # Main output folder
        output_folder + 'errors',         # Error debugging files
        output_folder + 'assembly',       # Assembly process outputs
        output_folder + 'simvascular',    # SimVascular-compatible outputs
        output_folder + 'simvascular/Images',  # Medical images
        output_folder + 'simvascular/Paths',   # Centerline paths (.pth files)
        output_folder + 'simvascular/Models'   # 3D models
    ]
    
    # Additional directories for detailed intermediate outputs
    if write_samples:
        directories.extend([
            output_folder + 'volumes',        # Extracted volume samples
            output_folder + 'predictions',    # nnU-Net prediction outputs
            output_folder + 'centerlines',    # Step-by-step centerlines
            output_folder + 'surfaces',       # Generated surface meshes
            output_folder + 'points',         # Point clouds and markers
            output_folder + 'animation'       # Animation/visualization files
        ])
    
    # Create each directory, silently continuing if it already exists
    for directory in directories:
        try:
            os.mkdir(directory)
        except Exception as e:
            print(e)  # Print error but continue execution


def main():
    """
    Main function for SeqSeg vessel tracing pipeline.
    
    Processes command line arguments, initializes nnU-Net models, and runs
    sequential vessel segmentation and centerline tracing on medical images.
    Outputs include vessel segmentations, centerlines, and 3D surface meshes.
    """
    # Configure command line argument parser
    parser = argparse.ArgumentParser(
        description='SeqSeg: Sequential vessel segmentation and centerline tracing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
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

    # Load algorithm configuration from YAML file
    global_config = load_yaml_config(args.config_name)
    print(f"\nUsing config file: {args.config_name}")

    # Set up directory paths
    dir_output0 = args.outdir
    data_dir = args.data_directory

    # Ensure the main output directory exists
    try:
        os.mkdir(dir_output0)
    except Exception as e:
        print(e)

    # Extract and organize configuration parameters
    # Tracing control parameters
    unit = args.unit                                    # Physical units (cm/mm)
    max_step_size = args.max_n_steps                   # Total iteration limit
    max_n_branches = args.max_n_branches               # Maximum vessel branches
    max_n_steps_per_branch = args.max_n_steps_per_branch  # Steps per branch
    write_samples = args.write_steps                   # Save intermediate files
    take_time = global_config['TIME_ANALYSIS']         # Performance timing
    calc_global_centerline = args.extract_global_centerline  # Post-process centerline
    cap_surface_cent = args.cap_surface_cent           # Cap surface ends

    # nnU-Net model configuration
    dataset = args.train_dataset                       # Training dataset name
    fold = args.fold                                   # Cross-validation fold
    img_format = args.img_ext                          # Image file extension
    scale = args.scale                                 # Unit scaling factor
    test_name = args.nnunet_type                       # Model type (3d_fullres/2d)
    pt_centerline = args.pt_centerline                 # Centerline point spacing
    num_seeds = args.num_seeds_centerline              # Number of seed points

    # Construct nnU-Net model weights directory path
    dir_model_weights = dataset + '/nnUNetTrainer__nnUNetPlans__' + test_name
    if args.nnunet_results_path is not None:
        weight_dir_nnunet = args.nnunet_results_path
        dir_model_weights = os.path.join(weight_dir_nnunet, dir_model_weights)

    # Load testing dataset and determine which samples to process
    testing_samples, directory_data = get_testing_samples(dataset, data_dir)
    print("Testing samples about to run:")
    for sample in testing_samples:
        print(sample)

    # Determine sample range for batch processing
    if args.stop == -1:
        args.stop = len(testing_samples)  # Process all samples if no stop specified

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

        # Configure output logging (redirect to file unless debugging)
        if not global_config['DEBUG']:
            sys.stdout = open(dir_output+"/out.txt", "w")
        else:
            print("\nStart tracking with debug mode on")
            # Debugger can be enabled here for step-through debugging
        
        # Log initialization status and parameters
        if json_file_present:
            print("\nWe got seed point from json file")
        else:
            print("\nWe did not get seed point from json file")
        print(test_case)
        print(f"Number of initial points: {len(potential_branches)}")
        print(f"Time is: {time.time() - start_time:.2f} sec")

        # Execute main vessel tracing algorithm
        # This is the core SeqSeg pipeline that performs sequential segmentation
        (centerlines, surfaces, points, inside_pts, assembly_obj,
         vessel_tree, n_steps_taken) = trace_centerline(
            dir_output,              # Output directory for this case
            dir_image,               # Input medical image path
            case,                    # Case identifier/name
            dir_model_weights,       # nnU-Net model weights directory
            fold,                    # Cross-validation fold
            potential_branches,      # Initial seed points for tracing
            max_step_size,           # Maximum total steps
            max_n_branches,          # Maximum vessel branches
            max_n_steps_per_branch,  # Maximum steps per branch
            global_config,           # Algorithm configuration parameters
            unit,                    # Physical units
            scale,                   # Image scaling factor
            dir_seg,                 # Segmentation reference (if available)
            write_samples=write_samples  # Save intermediate outputs
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

        # Process and save final segmentation results
        # Global assembly contains accumulated segmentations from all tracing steps
        assembly_org = assembly_obj.assembly
        assembly = assembly_org

        # Create binary segmentation by thresholding probability map
        assembly_binary = sitk.BinaryThreshold(assembly, lowerThreshold=0.5,
                                               upperThreshold=1)
        
        # Save intermediate segmentation results
        sitk.WriteImage(assembly_binary, dir_output+'/'+case+'_binary_seg_'
                        + test_name + '_' + str(i) + '.mha')
        sitk.WriteImage(assembly, dir_output+'/'+case+'_prob_seg_'
                        + test_name + '_' + str(i) + '.mha')

        # Filter segmentation to keep only components connected to seed points
        # This removes disconnected vessel segments that may be artifacts
        assembly_binary = sf.keep_component_seeds(assembly_binary,
                                                  initial_seeds)
        
        # Save final cleaned segmentation
        sitk.WriteImage(assembly_binary, dir_output0+'/'+case
                        + '_segmentation_' + test_name + '_'
                        + str(n_steps_taken) + '_steps' + '.mha')

        # Generate 3D surface meshes from binary segmentation
        assembly_surface = vf.evaluate_surface(assembly_binary, 1)
        
        # Save raw surface mesh (may have irregular geometry)
        vf.write_vtk_polydata(assembly_surface, dir_output + '/' + case
                              + '_surface_mesh_nonsmooth_' + test_name + '_'
                              + str(n_steps_taken) + '_steps' + '.vtp')
        
        # Apply smoothing to create high-quality surface for CFD/analysis
        surface_smooth = vf.smooth_polydata(assembly_surface, iteration=75, smoothingFactor=0.1)
        vf.write_vtk_polydata(surface_smooth, dir_output0 + '/' + case
                              + '_surface_mesh_' + test_name + '_'
                              + str(n_steps_taken) + '_steps' + '.vtp')

        # Combine and save tracing step results if any centerlines were found
        if len(centerlines) > 0:
            # Merge all individual step results into single datasets
            final_surface = vf.appendPolyData(surfaces)       # All surface patches
            final_centerline = vf.appendPolyData(centerlines) # All centerline segments
            final_points = vf.appendPolyData(points)          # All tracing points

            # Save combined step-by-step results for analysis
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
        # Optional: Extract global centerline using distance-based methods
        # This provides a single continuous centerline through the entire vessel tree
        if calc_global_centerline:
            global_centerline, targets, success = calc_centerline_global(
                assembly_binary,
                initial_seeds)
            
            if success or len(targets) > 0:
                # Save main centerline for CFD inlet/outlet definition
                vf.write_vtk_polydata(global_centerline, dir_output0 + '/'
                                      + case + '_centerline_' + test_name + '_'
                                      + str(n_steps_taken)
                                      + '_steps' + '.vtp')
                
                # Save target points (vessel endpoints) for boundary condition setup
                targets_pd = vf.points2polydata([target.tolist()
                                                for target in targets])
                vf.write_vtk_polydata(targets_pd, dir_output+'/'
                                      + case + '_' + test_name + '_'+str(i)+'_'
                                      + str(n_steps_taken)
                                      + '_targets.vtp')

                # Optional: Cap vessel ends for CFD-ready geometry
                if cap_surface_cent:
                    capped_surface, capped_seg = cap_surface(
                        pred_surface=assembly_surface,
                        centerline=global_centerline,
                        pred_seg=assembly_binary,
                        file_name=case,
                        outdir=dir_output,
                        targets=targets)
                    
                    # Save capped surface (ready for mesh generation)
                    vf.write_vtk_polydata(capped_surface, dir_output+'/'
                                        + case + '_' + test_name + '_'+str(i)+'_'
                                        + str(n_steps_taken)
                                        + '_capped_surface.vtp')
                    
                    # Save capped segmentation (binary image with caps)
                    sitk.WriteImage(capped_seg, dir_output+'/'
                                    + case + '_' + test_name + '_'
                                    + str(n_steps_taken) + '_capped_seg.mha')

        # Save points detected inside existing vessels (for debugging retrace prevention)
        if global_config['PREVENT_RETRACE']:
            if len(inside_pts) > 0:
                final_inside_pts = vf.appendPolyData(inside_pts)
                vf.write_vtk_polydata(final_inside_pts, dir_output + '/final_'
                                      + case + '_' + test_name + '_'+str(i)+'_'
                                      + str(n_steps_taken)
                                      + '_inside_points.vtp')

        # Restore stdout from file redirection
        if not global_config['DEBUG']:
            sys.stdout.close()
            sys.stdout = sys.__stdout__

    print("\nTotal calculation time for all cases is: ")
    print(f"{((time.time() - start_time)/60):.2f} min\n")


if __name__ == '__main__':
    main()
