import pdb
from .sitk_functions import *
from .vtk_functions import is_point_in_image, write_vtk_polydata, points2polydata
import numpy as np
import SimpleITK as sitk
import operator
import time
from datetime import datetime

class Segmentation:
    """
    Class to keep track of a global segmentation, and update it with new segmentations
    """

    def __init__(self, case = None, image_file = None, weighted = False, weight_type = None, image = None):
        """
        Args:
            case: name of the case
            image_file: image file to create the global segmentation in the same space
            weighted: whether to use a weighted average for the segmentation
            weight_type: type of weight to use for the weighted average
            image: image object to create the global segmentation
        """
        if case:
            self.name = case
        if image_file:
            self.image_reader = read_image(image_file)

            new_img = create_new(self.image_reader)
            self.assembly = new_img

        elif image:

            self.image_reader = image
            self.assembly = image
        
        else:
            print("Please provide either an image file or an image object")

        self.number_updates = np.zeros(sitk_to_numpy(self.assembly).shape)

        self.weighted = weighted

        if weighted:
            # also keep track of how many updates to pixels
            self.n_updates = np.zeros(sitk_to_numpy(self.assembly).shape)
            #print("Creating weighted segmentation")
            assert weight_type, "Please provide a weight type"
            assert weight_type in ['radius', 'gaussian'], "Weight type not recognized"
            self.weight_type = weight_type

    def add_segmentation(self, volume_seg, index_extract, size_extract, weight=None):
        """
        Function to add a new segmentation to the global assembly
        Args:
            volume_seg: the new segmentation to add
            index_extract: index for sitk volume extraction
            size_extract: number of voxels to extract in each dim
            weight: weight for the weighted average
        """
        # Load the volumes
        np_arr = sitk_to_numpy(self.assembly).astype(float)
        np_arr_add = sitk_to_numpy(volume_seg).astype(float)

        # Calculate boundaries
        cut = 0
        edges = np.array(index_extract) + np.array(size_extract) - cut
        index_extract = np.array(index_extract) + cut

        # Keep track of number of updates
        curr_n = self.number_updates[index_extract[2]:edges[2], index_extract[1]:edges[1], index_extract[0]:edges[0]]

        # Isolate current subvolume of interest
        curr_sub_section = np_arr[index_extract[2]:edges[2], index_extract[1]:edges[1], index_extract[0]:edges[0]]
        np_arr_add = np_arr_add[cut:size_extract[2]-cut, cut:size_extract[1]-cut, cut:size_extract[0]-cut]
        # import pdb; pdb.set_trace()
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
            if self.weight_type == 'radius':
                curr_sub_section[ind] = 1/(curr_n[ind]+weight)*( weight*np_arr_add[ind] + (curr_n[ind])*curr_sub_section[ind] )
                # Add to update weight sum for these voxels
                self.number_updates[index_extract[2]:edges[2], index_extract[1]:edges[1], index_extract[0]:edges[0]] += weight
                self.n_updates[index_extract[2]:edges[2], index_extract[1]:edges[1], index_extract[0]:edges[0]] += 1
            elif self.weight_type == 'gaussian':
                # now the weight varies with the distance to the center of the volume, and the distance to the border
                weight_array = self.calc_weight_array_gaussian(size_extract)
                # print(f"weight array size: {weight_array.shape}, ind size: {ind.shape}")
                # Update those values, calculating an average
                curr_sub_section[ind] = 1/(curr_n[ind]+weight_array[ind])*( weight_array[ind]*np_arr_add[ind] + (curr_n[ind])*curr_sub_section[ind] )
                # Add to update weight sum for these voxels
                self.number_updates[index_extract[2]:edges[2], index_extract[1]:edges[1], index_extract[0]:edges[0]] += weight_array
                self.n_updates[index_extract[2]:edges[2], index_extract[1]:edges[1], index_extract[0]:edges[0]] += 1

        # Update the global volume
        np_arr[index_extract[2]:edges[2], index_extract[1]:edges[1], index_extract[0]:edges[0]] = curr_sub_section
        self.assembly = numpy_to_sitk(np_arr, self.image_reader)

    def calc_weight_array_gaussian(self, size_extract):
        "Function to calculate the weight array for a gaussian weighted segmentation"
        # have std so the weight is 0.1 at the border of the volume
        std = 0.1*np.array(size_extract)
        # create a grid of distances to the center of the volume
        x = np.linspace(-size_extract[0]/2, size_extract[0]/2, size_extract[0])
        y = np.linspace(-size_extract[1]/2, size_extract[1]/2, size_extract[1])
        z = np.linspace(-size_extract[2]/2, size_extract[2]/2, size_extract[2])
        x, y, z = np.meshgrid(z, y, x)
        # now transpose
        x = x.transpose(1,0,2)
        y = y.transpose(1,0,2)
        z = z.transpose(1,0,2)
        # calculate the weight array
        weight_array = np.exp(-0.5*(x**2/std[0]**2 + y**2/std[1]**2 + z**2/std[2]**2))
        return weight_array

    def create_mask(self):
        "Function to create a global image mask of areas that were segmented"
        mask = (self.number_updates > 0).astype(int)
        mask = numpy_to_sitk(mask,self.image_reader)
        # import pdb; pdb.set_trace()
        self.mask = mask

        return mask
    def upsample(self, template_size=[1000,1000,1000]):
        from .prediction import centering
        import pdb; pdb.set_trace()
        or_im = sitk.GetImageFromArray(self.assembly)
        or_im.SetSpacing(self.image_resampled.GetSpacing())
        or_im.SetOrigin(self.image_resampled.GetOrigin())
        or_im.SetDirection(self.image_resampled.GetDirection())
        target_im = create_new(or_im)
        target_im.SetSize(template_size)
        new_spacing = (np.array(or_im.GetSize())*np.array(or_im.GetSpacing()))/np.array(template_size)
        target_im.SetSpacing(new_spacing)
        import pdb; pdb.set_trace()

        resampled = centering(or_im, target_im, order=0)
        return resampled

    def upsample_sitk(self, template_size=[1000,1000,1000]):

        from .prediction import resample_spacing
        resampled, ref_im = resample_spacing(self.assembly, template_size=template_size)
        return resampled
47
class VesselTree:

    def __init__(self, case, image_file, init_step, pot_branches):
        self.name = case
        self.image = image_file
        self.bifurcations = [0]
        self.branches = [[0]]
        self.steps = [init_step]
        if len(pot_branches) > 1:
            for i in range(len(pot_branches)-1):
                pot_branches[i+1]['connection'] = [0,0]
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

    def get_previous_n(self, branch, n):

        branch0 = self.branches[branch]
        if len(branch0) > n:
            previous_n = branch0[-n:]
        else:
            # previous_n = branch0
            previous_n = [bra for bra in branch0] #branch0
            # previous_n = previous_n[:]
            # conn = self.bifurcations[branch]
            # if conn != 0:
            if len(previous_n) > 1:
                previous_n = previous_n[1:]

            #     res = n-len(branch0)+2
            #     for i in range(1,res):
            #         previous_n.append(conn+i)
            #         previous_n.append(conn-i)
        # remove 0 from the list
        if 0 in previous_n and len(previous_n) > 1:
            previous_n.remove(0)
            
        return previous_n

    def get_previous_step(self,step_number):

        inds = [(i, colour.index(step_number)) for i, colour in enumerate(self.branches) if step_number in colour]

        for ind in inds:
            if ind[1] == 0: continue
            else:
                ind_prev = self.branches[ind[0]][ind[1]-1]
        return self.steps[ind_prev]

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

    def calc_caps(self, global_assembly):
        'Temp try at calculating global caps'
        final_caps = []
        for point in self.caps:
            if not is_point_in_image(global_assembly, point):
                final_caps.append(point)
            #else:
                #print('Inside \n')

        print('Number of outlets: ' + str(len(final_caps)))
        #final_caps = orient_caps(final_caps, init_step)
        return final_caps

    def get_end_points(self):
        points = [self.steps[0]['point'] - self.steps[0]['tangent'] * 2 * self.steps[0]['radius']]
        for branch in self.branches:
            id = branch[-1]
            points.append(self.steps[id]['point'] + self.steps[id]['tangent'] * 1 * self.steps[id]['radius'])
        return points

    def write_csv(self):
        """
        Function to write tree graph as a csv file
        Each line will have: Node 1, Node 2 that are connected
        and any attributes associated with the edge: Radius, Angle, etc
        """
        import csv


    def plot_graph(self):
        import networkx as nx
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


def print_error(output_folder, i, step_seg, image=None, predicted_vessel=None, old_point_ref = None, centerline_poly=None):

    now = datetime.now()
    dt_string = now.strftime("_%d_%m_%Y_%H_%M_%S")
    directory = output_folder + 'errors/'+str(i) + '_error_'+dt_string

    polydata_point = points2polydata([step_seg['point'].tolist()])
    write_vtk_polydata(polydata_point, directory + 'point.vtp')

    if step_seg['img_file'] and not step_seg['is_inside']:
        sitk.WriteImage(image, directory + 'img.vtk')

        if step_seg['seg_file']:
            sitk.WriteImage(predicted_vessel, directory + 'seg.vtk')

            if step_seg['surf_file']:
                write_vtk_polydata(step_seg['surface'], directory + 'surf.vtp')

                if step_seg['centerline']:
                    polydata_point = points2polydata([step_seg['old_point_ref'].tolist()])
                    write_vtk_polydata(polydata_point, directory + 'old_point_ref.vtp')
                
                    write_vtk_polydata(centerline_poly, directory + 'cent.vtp')


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

def get_old_ref_point(vessel_tree, step_seg, i, mega_sub = False, mega_sub_N = 0):

    if not mega_sub:
        # return the old point
        if i != 0:
            prev_step = vessel_tree.get_previous_step(i)
            old_point_ref = prev_step['old point']
        elif i == 0:
            old_point_ref = step_seg['old point']
    else:
        # return the old of the oldest step used for mega subvolume
        branch = len(vessel_tree.branches) - 1
        prev_n = vessel_tree.get_previous_n(branch, mega_sub_N)

        step = vessel_tree.steps[prev_n[0]]
        old_point_ref = step['old point']

    return old_point_ref