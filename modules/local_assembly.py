from .sitk_functions import *
from .assembly import * 

def construct_subvolume(step_seg, vessel_tree, N_steps = 5):
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
    branch = len(vessel_tree.branches) - 1
    prev_n = vessel_tree.get_previous_n(branch, N_steps)
    # First we calculate the bounds for the N_steps previous steps
    index = [0,0,0]
    size_extract = [10000,10000,10000]
    for n in prev_n:
        step = vessel_tree.steps[n]
        bounds = get_bounds(step['img_index'], step['img_size'])
        for i in range(3):
            if bounds[i][0] > index[i]:
                index[i] = bounds[i][0]
            if bounds[i][1] < size_extract[i]:
                size_extract[i] = bounds[i][1]
    
    for i in range(3):
        size_extract[i] -= index[i]

    # Then we extract this subvolume from the global
    img_reader = read_image(vessel_tree.image)
    # seg_reader = create_new(img_reader)
    subvolume_img = extract_volume(img_reader, index, size_extract)
    subvolume_seg = create_new(subvolume_img, 1)

    # Convert to numpy arrays
    # index = np.array([0,0,0])
    # size_extract = np.array([10000,10000,10000])

    # for n in prev_n:
    #     step = vessel_tree.step[n]
    #     bounds = np.array(get_bounds(step['img_index'], step['img_size']))
    #     index = np.maximum(index, bounds[:, 0])
    #     size_extract = np.minimum(size_extract, bounds[:, 1])

    # size_extract -= index

    # # Then we extract this subvolume from the global
    # img_reader = read_image(vessel_tree.image)
    # seg_reader = create_new(img_reader)
    # subvolume_img = extract_volume(img_reader, index.tolist(), size_extract.tolist())
    # subvolume_seg = extract_volume(seg_reader, index.tolist(), size_extract.tolist())
    
    # Then we loop over previous subvolumes and average them together
    Assembly = Segmentation(    image = subvolume_seg, 
                                weighted = False
                            )
    # Add the current step
    Assembly.add_segmentation(step_seg['prob_predicted_vessel'], get_local_ind(index, step_seg['img_index']), step_seg['img_size'])
    import pdb; pdb.set_trace()
    # Add the previous steps
    for n in prev_n:
        step = vessel_tree.steps[n]
        local_index = get_local_ind(index, step['img_index'])
        local_size = step['img_size']
        
        Assembly.add_segmentation(      step['prob_predicted_vessel'], 
                                        local_index,
                                        local_size
        )

    return Assembly.assembly, subvolume_img

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