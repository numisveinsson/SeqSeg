import modules.sitk_functions as sf
from modules import vtk_functions as vf
import numpy as np
import SimpleITK as sitk

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

    def __init__(self, case, image_file, init_step, pot_branches):
        self.name = case
        self.image = image_file
        self.bifurcations = [0]
        self.branches = [[0]]
        self.steps = [init_step]
        self.potential_branches = pot_branches
        self.caps = []

    def add_step(self, i, step, branch):
        self.steps.append(step)
        self.branches[branch].append(i)

    def remove_step(self, i, step, branch):
        self.steps.remove(step)
        self.branches[branch].remove(i)

    def add_branch(self, connector, i):
        self.branches.append([connector, i])
        self.bifurcations.append(connector)

    def remove_branch(self, branch):
        start = self.branches[branch][1]
        end = self.branches[branch][-1]
        del self.steps[start:end]
        self.branches[branch] = []
        del self.bifurcations[branch]

    def sort_potential(self):
        import operator
        self.potential_branches.sort(key=operator.itemgetter('radius'))

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

class VesselTreeParallel:

    def __init__(self, case, image_file, pot_branches):
        self.name = case
        self.image = image_file
        self.potential_branches = pot_branches
        self.branches = []

    def add_branch(self, branch):
        self.branches.append(branch)
        self.potential_branches.extend(branch.children)

    def remove_potential(self, pot_branch):
        self.potential_branches = [i for i in self.potential_branches if not (i['radius'] == pot_branch['radius'])]

    def sort_potential(self):
        import operator
        self.potential_branches.sort(key=operator.itemgetter('radius'))


class Branch:

    def __init__(self, init_step, branch_number):
        self.steps = [init_step]
        self.parent = init_step['connection']
        self.children = []
        self.branch_number = branch_number

    def add_step(self, step):
        self.steps.append(step)

    def add_child(self, step):
        self.children.append(step)


def print_error(output_folder, i, step_seg, image=None, predicted_vessel=None):

    directory = output_folder + 'errors/'+str(i) + '_error_'

    if step_seg['img_file']:
        sitk.WriteImage(image, directory + 'img.vtk')

        if step_seg['seg_file']:
            sitk.WriteImage(predicted_vessel, directory + 'seg.vtk')

            if step_seg['surf_file']:
                vf.write_vtk_polydata(step_seg['surface'], directory + 'surf.vtp')


def create_step_dict(old_point, old_radius, new_point, new_radius, angle_change=None):

    step_dict = {}
    step_dict['old point'] = old_point
    step_dict['point'] = new_point
    step_dict['old radius'] = old_radius
    step_dict['tangent'] = (new_point - old_point)/np.linalg.norm(new_point - old_point)
    step_dict['radius'] = new_radius
    step_dict['chances'] = 0
    step_dict['seg_file'] = None
    step_dict['img_file'] = None
    step_dict['surf_file'] = None
    step_dict['cent_file'] = None
    step_dict['prob_predicted_vessel'] = None
    step_dict['point_pd'] = None
    step_dict['surface'] = None
    step_dict['centerline'] = None
    step_dict['is_inside'] = False
    step_dict['time'] = None
    step_dict['dice'] = None
    if angle_change:
        step_dict['angle change'] = angle_change

    return step_dict
