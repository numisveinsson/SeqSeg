import modules.sitk_functions as sf
import numpy as np

class Segmentation:

    def __init__(self, case, image_file):
        self.name = case
        self.image_reader = sf.read_image(image_file)

        new_img = sf.create_new(self.image_reader)
        self.assembly = new_img

        self.number_updates = np.zeros(sf.sitk_to_numpy(self.assembly).shape)

    def add_segmentation(self, volume_seg, index_extract, size_extract):

        # Load the volumes
        np_arr = sf.sitk_to_numpy(self.assembly).astype(float)
        np_arr_add = sf.sitk_to_numpy(volume_seg).astype(float)

        # Calculate boundaries
        cut = 1
        edges = np.array(index_extract) + np.array(size_extract) - cut
        index_extract = np.array(index_extract) + cut

        # Keep track of number of updates
        curr_n = self.number_updates[index_extract[2]:edges[2], index_extract[1]:edges[1], index_extract[0]:edges[0]]

        # Isolate current subvolume of interest
        curr_sub_section = np_arr[index_extract[2]:edges[2], index_extract[1]:edges[1], index_extract[0]:edges[0]]
        np_arr_add = np_arr_add[cut:size_extract[2]-cut, cut:size_extract[1]-cut, cut:size_extract[0]-cut]

        # Find indexes where we need to average predictions
        ind = curr_n > 0

        # Update those values, calculating an average
        curr_sub_section[ind] = 1/(curr_n[ind])*( np_arr_add[ind] + (curr_n[ind] - 1)*curr_sub_section[ind] )

        # Where this is the first update, copy directly
        curr_sub_section[curr_n == 0] = np_arr_add[curr_n == 0]

        # Update the global volume
        np_arr[index_extract[2]:edges[2], index_extract[1]:edges[1], index_extract[0]:edges[0]] = curr_sub_section
        self.assembly = sf.numpy_to_sitk(np_arr, self.image_reader)

        # Add to update counter for these voxels
        self.number_updates[index_extract[2]:edges[2], index_extract[1]:edges[1], index_extract[0]:edges[0]] += 1

class VesselTree:

    def __init__(self, case, image_file, init_step):
        self.name = case
        self.image = image_file
        self.bifurcations = []
        self.branches = []
        self.steps = [init_step]

    def add_step(self, i, step, branch):
        self.steps.append(step)
        self.branches[branch].append(i)

    def add_branch(self, i, step):
        self.branches.append([i])
        self.bifurcations.append(i)

    def get_previous_n(self, i, branch, n):
        branch0 = self.branches[branch]
        if len(branch0) > n:
            previous_n = branch0[-n:]
        else:
            previous_n = branch0
            res = n-len(branch0)+2
            conn = self.bifurcations[branch]
            for i in range(1,res):
                previous_n.append(conn+i)
                previous_n.append(conn-i)
