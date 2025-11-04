from .sitk_functions import *
from .assembly import *
import SimpleITK as sitk

def construct_subvolume(step_seg, vessel_tree, N_steps = 5, N_curr = None, inside_branch = 0):
    """
    Function to create a 'mega' subvolume, assembled from a few
    Args:
        step_seg: current step dict
        vessel_tree: vessel tree object containing all previous steps
        N_steps: number of steps to include
    Returns:
        assembled subvolume
    
    Note: This function is dependent on only being used while tracing (not retracing)
    """
    if N_steps > 1 and inside_branch > 0:
        N_steps = 1
    
    branch = len(vessel_tree.branches) - 1
    prev_n = vessel_tree.get_previous_n(branch, N_steps)
    print(f"\nPrev n used for mega subvolume: {prev_n}")
    # First we calculate the bounds for the N_steps previous steps
    size_extract = [0,0,0]
    index = [10000,10000,10000]
    for n in prev_n:
        step = vessel_tree.steps[n]
        bounds = get_bounds(step['img_index'], step['img_size'])
        for i in range(3):
            if bounds[i][0] < index[i]:
                index[i] = bounds[i][0]
            if bounds[i][1] > size_extract[i]:
                size_extract[i] = bounds[i][1]
    
    for i in range(3):
        size_extract[i] -= index[i]

    # Then we extract this subvolume from the global
    img_reader = read_image(vessel_tree.image)
    # seg_reader = create_new(img_reader)
    subvolume_img = extract_volume(img_reader, index, size_extract)
    subvolume_seg = create_new(subvolume_img, 1)
    # print(f"Global image size: {img_reader.GetSize()}")
    print(f"Mega index: {index}, size: {size_extract}")

    # Then we loop over previous subvolumes and average them together
    Assembly = Segmentation(    image = subvolume_seg, 
                                weighted = False
                            )
    # Add the current step
    # import pdb; pdb.set_trace()
    Assembly.add_segmentation(step_seg['prob_predicted_vessel'], get_local_ind(index, step_seg['img_index']), step_seg['img_size'])
    prev_n.remove(N_curr)
    # Add the previous steps
    for n in prev_n:
        step = vessel_tree.steps[n]
        # print(f"Global local index: {step['img_index']}, size: {step['img_size']}")
        local_index = get_local_ind(index, step['img_index'])
        local_size = step['img_size']
        # print(f"Local index: {local_index}, size: {local_size}")
        Assembly.add_segmentation(      step['prob_predicted_vessel'], 
                                        local_index,
                                        local_size
        )
    
    # return a binary image
    assembly_binary = sitk.BinaryThreshold(Assembly.assembly, lowerThreshold=0.5, upperThreshold=1)

    print(f"Mega subvolume constructed.")
    return assembly_binary, subvolume_img

def get_bounds(index, size):
    """
    Function to calculate the bounds of a subvolume from its index and size
    """

    bounds = []
    for i in range(3):
        bounds.append([index[i], index[i] + size[i]]) 

    return bounds

def get_local_ind(global_index, index):
    """
    Function to calculate the local index and size of a subvolume from its global index and size
    Args:
        global_index: index of the global volume in global coordinates
        index: index of the subvolume in the global volume in global coordinates
    """
    local_index = []
    for i in range(3):
        local_index.append(index[i] - global_index[i])

    return local_index