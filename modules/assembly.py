import pdb
import modules.sitk_functions as sf
from modules import vtk_functions as vf
import numpy as np
import SimpleITK as sitk
import operator
import time
from datetime import datetime

class Segmentation:

    def __init__(self, case, image_file, weighted = False):
        self.name = case
        self.image_reader = sf.read_image(image_file)

        new_img = sf.create_new(self.image_reader)
        self.assembly = new_img

        self.number_updates = np.zeros(sf.sitk_to_numpy(self.assembly).shape)

        self.weighted = weighted

        if weighted:
            # also keep track of how many updates to pixels
            self.n_updates = np.zeros(sf.sitk_to_numpy(self.assembly).shape)
            #print("Creating weighted segmentation")

    def add_segmentation(self, volume_seg, index_extract, size_extract, weight=None):

        # Load the volumes
        np_arr = sf.sitk_to_numpy(self.assembly).astype(float)
        np_arr_add = sf.sitk_to_numpy(volume_seg).astype(float)

        # Calculate boundaries
        cut = 0
        edges = np.array(index_extract) + np.array(size_extract) - cut
        index_extract = np.array(index_extract) + cut

        # Keep track of number of updates
        curr_n = self.number_updates[index_extract[2]:edges[2], index_extract[1]:edges[1], index_extract[0]:edges[0]]

        # Isolate current subvolume of interest
        curr_sub_section = np_arr[index_extract[2]:edges[2], index_extract[1]:edges[1], index_extract[0]:edges[0]]
        np_arr_add = np_arr_add[cut:size_extract[2]-cut, cut:size_extract[1]-cut, cut:size_extract[0]-cut]

        # Find indexes where we need to average predictions
        ind = curr_n > 0
        # Where this is the first update, copy directly
        curr_sub_section[curr_n == 0] = np_arr_add[curr_n == 0]

        if not self.weighted: # Then we do plain average
            # Update those values, calculating an average
            curr_sub_section[ind] = 1/(curr_n[ind]+1)*( np_arr_add[ind] + (curr_n[ind])*curr_sub_section[ind] )
            # Add to update counter for these voxels
            self.number_updates[index_extract[2]:edges[2], index_extract[1]:edges[1], index_extract[0]:edges[0]] += 1

        else:
            curr_sub_section[ind] = 1/(curr_n[ind]+weight)*( weight*np_arr_add[ind] + (curr_n[ind])*curr_sub_section[ind] )
            # Add to update weight sum for these voxels
            self.number_updates[index_extract[2]:edges[2], index_extract[1]:edges[1], index_extract[0]:edges[0]] += weight
            self.n_updates[index_extract[2]:edges[2], index_extract[1]:edges[1], index_extract[0]:edges[0]] += 1

        # Update the global volume
        np_arr[index_extract[2]:edges[2], index_extract[1]:edges[1], index_extract[0]:edges[0]] = curr_sub_section
        self.assembly = sf.numpy_to_sitk(np_arr, self.image_reader)

    def create_mask(self):
        "Function to create a global image mask of areas that were segmented"
        mask = (self.number_updates > 0).astype(int)
        mask = sf.numpy_to_sitk(mask,self.image_reader)
        # import pdb; pdb.set_trace()
        self.mask = mask

        return mask
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
        self.potential_branches.sort(key=operator.itemgetter('radius'), reverse = True)

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

    def calc_ave_dice(self):
        total_dice, count = 0,0
        for step in self.steps[1:]:
            if step['dice']:
                if not step['dice'] == 0:
                    count += 1
                    total_dice += step['dice']
        ave_dice = total_dice/count
        if count > 1:
            print(f"Average dice per step was : {ave_dice}")
        return ave_dice

    def calc_ave_time(self):
        total_time, count = 0,0
        for step in self.steps[1:]:
            if step['time']:
                count += 1
                total_time += step['time']
        ave_time = total_time/count
        if count > 1:
            print(f"Average time was : {ave_time}")
        return ave_time

    def time_analysis(self):
        names = ['extraction     ',
                 'prediction     ',
                 'surface        ',
                 'centerline     ',
                 'global assembly',
                 'next point     ']
        time_sum = np.zeros(len(names))
        counter = 0
        for step in self.steps:
            if step['time']:
                time_arr = np.zeros(len(names))
                for j in range(len(step['time'])):
                    time_arr[j] = step['time'][j]
                time_sum += time_arr
                counter += 1

        for j in range(len(names)):
            print('Average time for ' + names[j]+ ' : ', time_sum[j]/counter)
        print(np.array(time_sum/counter).tolist())

    def calc_caps(self):
        'Temp try at calculating global caps'
        final_caps = []
        for point in vessel_tree.caps:
            if not vf.is_point_in_image(assembly, point):
                final_caps.append(point)
            #else:
                #print('Inside \n')

        print('Number of outlets: ' + str(len(final_caps)))
        #final_caps = vf.orient_caps(final_caps, init_step)
        return final_caps

    def write_csv(self):
        """
        Function to write tree graph as a csv file
        Each line will have: Node 1, Node 2 that are connected
        and any attributes associated with the edge: Radius, Angle, etc
        """
        import csv


    def plot_graph(self):
        import networkx
        import matplotlib.pyplot as plt

        G = nx.DiGraph()
        G.add_edges_from(
            [('A', 'B'), ('A', 'C'), ('D', 'B'), ('E', 'C'), ('E', 'F'),
             ('B', 'H'), ('B', 'G'), ('B', 'F'), ('C', 'G')])

        nx.draw(G)

        G = nx.Graph()
        G.add_edge(1, 2, color='r' ,weight=3)
        G.add_edge(2, 3, color='b', weight=5)
        G.add_edge(3, 4, color='g', weight=7)

        pos = nx.circular_layout(G)

        colors = nx.get_edge_attributes(G,'color').values()
        weights = nx.get_edge_attributes(G,'weight').values()

        nx.draw(G, pos, edge_color=colors, width=list(weights))

        plt.show()


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
        self.potential_branches.sort(key=operator.itemgetter('old radius'))



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

    now = datetime.now()
    dt_string = now.strftime("_%d_%m_%Y_%H_%M_%S")
    directory = output_folder + 'errors/'+str(i) + '_error_'+dt_string

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
