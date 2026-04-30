import sys
import os
import numpy as np
import SimpleITK as sitk

# sys.path.insert(0, './')
from seqseg.modules.vtk_functions import write_geo, points2polydata
from seqseg.modules.tracing_functions import (get_seed,
                                              get_largest_radius_seed,
                                              get_equally_spaced_radius_seeds)
from seqseg.modules.assembly import create_step_dict
from seqseg.modules.datasets import get_directories
from seqseg.modules.centerline import calc_multi_component_centerlines


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


def process_init(test_case, directory_data, dir_output0, img_format, test):

    path = directory_data + 'seeds.json'
    json_file_present = os.path.isfile(path)

    dir_seg = os.path.isdir(directory_data + 'truths')

    if json_file_present:
        # Information
        i = 0
        case = test_case['name']
        dir_image, dir_seg, dir_cent, _ = get_directories(directory_data,
                                                          case,
                                                          img_format,
                                                          dir_seg)
        dir_output = dir_output0 + test + '_' + case+'/'

    else:
        # Information
        case = test_case[0]
        i = test_case[1]

        print(test_case)

        # dir_image, dir_seg, dir_cent, _ = vmr_directories(directory_data,
        #                                                   case,
        #                                                   dir_seg)
        dir_image, dir_seg, dir_cent, _ = get_directories(directory_data,
                                                          case,
                                                          img_format,
                                                          dir_seg)
        dir_output = dir_output0 + test + '_'+case+'_'+str(i)+'/'

    return dir_output, dir_image, dir_seg, dir_cent, case, i, json_file_present


def initialization(json_file_present, test_case, dir_output, dir_cent,
                   dir_data, unit='mm', pt_centerline=50, num_seeds=1,
                   write_samples=False):

    if json_file_present:
        potential_branches, initial_seeds = initialize_json(test_case,
                                                            dir_output,
                                                            dir_cent,
                                                            dir_data,
                                                            unit,
                                                            write_samples)
    else:
        if_largest_radius=True
        potential_branches, initial_seeds = initialize_cent(test_case,
                                                            dir_output,
                                                            dir_cent,
                                                            if_largest_radius=if_largest_radius,
                                                            pt_centerline=pt_centerline,
                                                            num_seeds=num_seeds,
                                                            write_samples=write_samples)

    return potential_branches, initial_seeds


def initialize_json(test_case, dir_output, dir_cent, dir_data, unit,
                    write_samples=False):
    """
    Initialize the potential branches and initial seeds for the
    segmentation process, based on the json file
    """
    initial_seeds = []
    # Seed points
    potential_branches = []

    if not test_case['seeds']:
        if test_case['cardiac_mesh']:
            print("Cardiac mesh given, getting seed from aortic root")
            (old_seed,
             old_radius,
             initial_seed,
             initial_radius) = get_seeds_cardiac_mesh(dir_data,
                                                      test_case['name'],
                                                      unit)
        else:
            print("""No seed given,
                  trying to get one from centerline ground truth""")
            old_seed, old_radius = get_seed(dir_cent, 0, 150)
            initial_seed, initial_radius = get_seed(dir_cent, 0, 160)
            print(f"Seed found from centerline, took point nr {160}!")

        init_step = create_step_dict(old_seed, old_radius, initial_seed,
                                     initial_radius, 0)
        init_step['connection'] = [0, 0]
        print(f"Old seed: {old_seed}, {old_radius}")
        print(f"Initial seed: {initial_seed}, {initial_radius} ")
        initial_seeds.append(initial_seed)
        potential_branches = [init_step]
        if write_samples:
            write_geo(dir_output + 'points/0_seed_point.vtp',
                      points2polydata([old_seed.tolist()]))
            write_geo(dir_output + 'points/1_seed_point.vtp',
                      points2polydata([initial_seed.tolist()]))
    else:
        for seed in test_case['seeds']:
            step = create_step_dict(np.array(seed[0]),
                                    seed[2],
                                    np.array(seed[1]),
                                    seed[2], 0)
            step['connection'] = [0, 0]
            potential_branches.append(step)
            initial_seeds.append(np.array(seed[1]))
            if write_samples:
                write_geo(dir_output +
                          'points/'+str(test_case['seeds'].index(seed)) +
                          '_oldseed_point.vtp', points2polydata([seed[0]]))

    return potential_branches, initial_seeds


def initialize_cent(test_case, dir_output, dir_cent, if_largest_radius=True,
                    pt_centerline=0, num_seeds=1, write_samples=False):

    # Information
    if if_largest_radius:
        print("Getting seed from longest centerline where the radius is largest")
        (old_seeds, old_radiuss,
         initial_seeds, initial_radiuss) = get_largest_radius_seed(
                dir_cent, pt_centerline, num_seeds)
    else:
        i = test_case[1]
        id_old = test_case[2]
        id_current = test_case[3]

        # Get inital seed point + radius
        old_seed, old_radius = get_seed(dir_cent, i, id_old)
        print(old_seed)
        initial_seed, initial_radius = get_seed(dir_cent, i, id_current)
        initial_seeds = [initial_seed]
        old_seeds = [old_seed]
        old_radiuss = [old_radius]
        initial_radiuss = [initial_radius]

    if write_samples:
        for i in range(len(old_seeds)):
            old_seed = old_seeds[i]
            initial_seed = initial_seeds[i]
            write_geo(dir_output + 'points/0_seed_point.vtp',
                      points2polydata([old_seed.tolist()]))
            write_geo(dir_output + 'points/1_seed_point.vtp',
                      points2polydata([initial_seed.tolist()]))

    potential_branches = create_pots(len(old_seeds), old_seeds, old_radiuss,
                                     initial_seeds, initial_radiuss)

    return potential_branches, initial_seeds


def create_pots(num_seeds, old_seeds, old_radiuss,
                initial_seeds, initial_radiuss):

    potential_branches = []
    for i in range(num_seeds):
        init_step = create_step_dict(old_seeds[i], old_radiuss[i],
                                     initial_seeds[i], initial_radiuss[i], 0)
        init_step['connection'] = [0, 0]
        potential_branches.append(init_step)

    return potential_branches


def get_seeds_cardiac_mesh(mesh_dir, name, unit):
    """
    Get testing samples from cardiac meshes
    """
    import os
    from .vtk_functions import process_cardiac_mesh, write_normals_centers

    # radius estimate
    if unit == 'mm':
        radius_est = 13
    elif unit == 'cm':
        radius_est = 1.3

    # list_meshes = os.listdir(mesh_dir)
    mesh_dir += '/cardiac_meshes/'

    # Scale is 1 (assume same unit for image and mesh)
    (region_8_center,
     region_8_normal,
     region_3_center) = process_cardiac_mesh(
         os.path.join(mesh_dir, name+'.vtp'), scale=1
         )

    write_normals_centers(mesh_dir, region_8_center,
                          region_8_normal, region_3_center)

    # old_seed, old_radius, initial_seed, initial_radius

    return (region_8_center,
            radius_est,
            region_8_center+2*radius_est*region_8_normal,
            radius_est)


def initialize_from_seg(segmentation,
                        dir_output,
                        num_seeds=3,
                        write_samples=False,
                        return_centerline=False):
    """
    Initialize the potential branches and initial seeds for the
    segmentation process, based on the segmentation

    Parameters
    ----------
    segmentation : sitk.Image
        Segmentation image, binary
    dir_output : str
        Output directory
    num_seeds : int
        Number of seeds
    write_samples : bool
        Write samples

    Returns
    -------
    potential_branches : list
        List of potential branches
    initial_seeds : list
        List of initial seeds
    """
    # Seed points
    initial_seeds = []
    potential_branches = []
    extracted_centerline = None

    # Use multi-component centerline extraction over the three largest bodies.
    centerline, success_info = calc_multi_component_centerlines(
        segmentation,
        nr_seeds=3,
        min_res=300,
        out_dir=dir_output,
        write_files=write_samples,
        move_target_if_fail=False,
        relax_factor=1,
        verbose=True,
        return_failed=False,
    )
    success = bool(success_info.get('overall_success', False))
    if success and centerline.GetNumberOfPoints() > 0:
        extracted_centerline = centerline
        write_geo(dir_output + 'centerline.vtp', centerline)
        # num_seeds controls how many equally spaced seed pairs are placed
        # per branch; use all branches available in the centerline.
        n_positions = max(1, num_seeds)
        for position_idx in range(n_positions):
            (old_seeds, old_radiuss,
             component_initial_seeds, initial_radiuss) = get_equally_spaced_radius_seeds(
                dir_cent=dir_output+'centerline.vtp',
                num_positions=n_positions,
                position_idx=position_idx,
                num_branches=None
                )
            if len(component_initial_seeds) == 0:
                continue
            potential_branches += create_pots(
                len(component_initial_seeds),
                old_seeds,
                old_radiuss,
                component_initial_seeds,
                initial_radiuss
            )
            initial_seeds.extend(component_initial_seeds)
    else:
        print("Centerline calculation failed! - No seeds found!")

    if return_centerline:
        if extracted_centerline is not None:
            return (potential_branches,
                    initial_seeds,
                    extracted_centerline)
        return potential_branches, initial_seeds, None

    return potential_branches, initial_seeds


if __name__ == '__main__':
    "Test the initialization"
    out_dir = '/Users/numisveins/Documents/repositories/SeqSeg/tests/test_init/'
    # Read in segmentation
    seg = sitk.ReadImage('/Users/numisveins/Documents/datasets/ASOCA_dataset/Results_Predictions/output_2d_coroasocact/new_format_mha/coroasocact_001.mha')
    # Initialize
    potential_branches, initial_seeds = initialize_from_seg(seg,
                                                            out_dir,
                                                            num_seeds=1,
                                                            write_samples=True)