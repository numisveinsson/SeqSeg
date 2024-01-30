from .vtk_functions import *
from .assembly import create_step_dict
from .datasets import get_directories, vmr_directories

def process_init(test_case, directory_data, dir_output0, img_format, modality_model, json_file_present, test):
    if json_file_present:
        ## Information
        modality = modality_model
        i = 0
        case = test_case['name']
        dir_image, dir_seg, dir_cent, dir_surf = get_directories(directory_data, case, img_format, dir_seg =False)
        dir_seg = None
        dir_output = dir_output0 +test+'_'+case+'/'

    else:
        ## Information
        case = test_case[0]
        i = test_case[1]
        modality = test_case[4]

        if modality != modality_model:
            print(f"Modality doesnt match model and case: {modality},{modality_model} ")
        print(test_case)

        dir_image, dir_seg, dir_cent, dir_surf = vmr_directories(directory_data, case, dir_seg)
        dir_seg = None
        dir_output = dir_output0 +test+'_'+case+'_'+str(i)+'/'

    return dir_output, dir_image, dir_seg, dir_cent, modality, case, i
    

def initialization(json_file_present, test_case, dir_output, dir_cent, dir_data, write_samples=False):

    if json_file_present:
        potential_branches, initial_seeds = initialize_json(test_case, dir_output, dir_cent, dir_data, write_samples)
    else:
        potential_branches, initial_seeds = initialize_dict(test_case, dir_output, dir_cent, write_samples)

    return potential_branches, initial_seeds

def initialize_json(test_case, dir_output, dir_cent, dir_data, write_samples=False):
    """

    """
    initial_seeds = []
    ## Seed points
    potential_branches = []

    if not test_case['seeds']:
        if test_case['cardiac_mesh']:
            print(f"Cardiac mesh given, getting seed from aortic root")
            old_seed, old_radius, initial_seed, initial_radius = get_seeds_cardiac_mesh(dir_data, test_case['name'])
        
        else:
            print(f"No seed given, trying to get one from centerline ground truth")
            old_seed, old_radius = get_seed(dir_cent, 0, 150)
            initial_seed, initial_radius = get_seed(dir_cent, 0, 160)
            print(f"Seed found from centerline, took point nr {160}!")

        init_step = create_step_dict(old_seed, old_radius, initial_seed, initial_radius, 0)
        print(f"Old seed: {old_seed}, {old_radius}")
        print(f"Initial seed: {initial_seed}, {initial_radius} ")
        initial_seeds.append(initial_seed)
        potential_branches = [init_step]
        if write_samples:
            write_geo(dir_output+ 'points/0_seed_point.vtp', points2polydata([old_seed.tolist()]))
            write_geo(dir_output+ 'points/1_seed_point.vtp', points2polydata([initial_seed.tolist()]))
    else:
        for seed in test_case['seeds']:
            step  = create_step_dict(np.array(seed[0]), seed[2], np.array(seed[1]), seed[2], 0)
            potential_branches.append(step)
            initial_seeds.append(np.array(seed[1]))
            if write_samples:
                write_geo(dir_output+ 'points/'+str(test_case['seeds'].index(seed))+'_oldseed_point.vtp', points2polydata([seed[0]]))

    return potential_branches, initial_seeds

def initialize_dict(test_case, dir_output, dir_cent, write_samples=False):
    """

    """
    ## Information
    case = test_case[0]
    i = test_case[1]
    id_old = test_case[2]
    id_current = test_case[3]

    ## Get inital seed point + radius
    old_seed, old_radius = get_seed(dir_cent, i, id_old)
    print(old_seed)
    initial_seed, initial_radius = get_seed(dir_cent, i, id_current)
    initial_seeds = [initial_seed]
    if write_samples:
        write_geo(dir_output+ 'points/0_seed_point.vtp', points2polydata([old_seed.tolist()]))
    init_step = create_step_dict(old_seed, old_radius, initial_seed, initial_radius, 0)
    potential_branches = [init_step]

    return potential_branches, initial_seeds

def get_seeds_cardiac_mesh(mesh_dir, name):
    """
    Get testing samples from cardiac meshes
    """
    import os
    from .vtk_functions import process_cardiac_mesh, write_normals_centers

    # radius estimate 
    radius_est = 1.4

    # list_meshes = os.listdir(mesh_dir)
    mesh_dir += '/cardiac_meshes/'
    
    region_8_center, region_8_normal, region_3_center = process_cardiac_mesh(os.path.join(mesh_dir, name+'.vtp'))

    write_normals_centers(mesh_dir, region_8_center, region_8_normal, region_3_center)

    # old_seed, old_radius, initial_seed, initial_radius
    
    return region_8_center, radius_est, region_8_center+radius_est*region_8_normal, radius_est