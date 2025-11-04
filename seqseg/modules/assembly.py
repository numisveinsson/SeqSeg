from .sitk_functions import (read_image, create_new, sitk_to_numpy,
                             numpy_to_sitk, keep_component_seeds,
                             is_point_in_image)
from .centerline import calc_centerline_fmm
from .vtk_functions import (write_vtk_polydata,
                            points2polydata, appendPolyData)
import numpy as np
import SimpleITK as sitk
import operator
from datetime import datetime
import sys
sys.stdout.flush()


class Segmentation:
    """
    Class to keep track of a global segmentation,
    and update it with new segmentations
    """

    def __init__(self,
                 case=None,
                 image_file=None,
                 weighted=False,
                 weight_type=None,
                 image=None,
                 start_seg=None):
        """
        Args:
            case: name of the case
            image_file: image file to create the global segmentation
            in the same space
            weighted: whether to use a weighted average for the segmentation
            weight_type: type of weight to use for the weighted average
            image: image object to create the global segmentation
            start_seg: initial segmentation to start with
        """
        if case:
            self.name = case
        if image_file:
            self.image_reader = read_image(image_file)

            if start_seg is not None:
                self.assembly = start_seg
            else:
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
            if start_seg is None:
                self.n_updates = np.zeros(sitk_to_numpy(self.assembly).shape)
            else:
                self.n_updates = sitk_to_numpy(self.assembly)
            # print("Creating weighted segmentation")
            assert weight_type, "Please provide a weight type"
            assert weight_type in ['radius', 'gaussian'], """Weight type
            not recognized"""
            self.weight_type = weight_type

    def add_segmentation(self,
                         volume_seg,
                         index_extract,
                         size_extract,
                         weight=None):
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
        curr_n = self.number_updates[index_extract[2]:edges[2],
                                     index_extract[1]:edges[1],
                                     index_extract[0]:edges[0]]

        # Isolate current subvolume of interest
        curr_sub_section = np_arr[index_extract[2]:edges[2],
                                  index_extract[1]:edges[1],
                                  index_extract[0]:edges[0]]
        np_arr_add = np_arr_add[cut:size_extract[2]-cut,
                                cut:size_extract[1]-cut,
                                cut:size_extract[0]-cut]
        # Find indexes where we need to average predictions
        ind = curr_n > 0
        # Where this is the first update, copy directly
        curr_sub_section[curr_n == 0] = np_arr_add[curr_n == 0]

        if not self.weighted:  # Then we do plain average
            # Update those values, calculating an average
            curr_sub_section[ind] = 1/(curr_n[ind]+1) * (
                np_arr_add[ind] + (curr_n[ind])*curr_sub_section[ind])
            # Add to update counter for these voxels
            self.number_updates[index_extract[2]:edges[2],
                                index_extract[1]:edges[1],
                                index_extract[0]:edges[0]] += 1

        else:
            if self.weight_type == 'radius':
                curr_sub_section[ind] = 1/(curr_n[ind]+weight)*(
                    weight*np_arr_add[ind] + (
                        curr_n[ind])*curr_sub_section[ind])
                # Add to update weight sum for these voxels
                self.number_updates[index_extract[2]:edges[2],
                                    index_extract[1]:edges[1],
                                    index_extract[0]:edges[0]] += weight
                self.n_updates[index_extract[2]:edges[2],
                               index_extract[1]:edges[1],
                               index_extract[0]:edges[0]] += 1

            elif self.weight_type == 'gaussian':
                # now the weight varies with the distance to the center of
                # the volume, and the distance to the border
                weight_array = self.calc_weight_array_gaussian(size_extract)
                # print(f"weight array size: {weight_array.shape},
                # ind size: {ind.shape}")
                # Update those values, calculating an average
                curr_sub_section[ind] = 1/(
                    curr_n[ind]+weight_array[ind])*(
                        weight_array[ind]*np_arr_add[ind]
                        + (curr_n[ind])*curr_sub_section[ind])
                # Add to update weight sum for these voxels
                self.number_updates[index_extract[2]:edges[2],
                                    index_extract[1]:edges[1],
                                    index_extract[0]:edges[0]] += weight_array
                self.n_updates[index_extract[2]:edges[2],
                               index_extract[1]:edges[1],
                               index_extract[0]:edges[0]] += 1

        # Update the global volume
        np_arr[index_extract[2]:edges[2],
               index_extract[1]:edges[1],
               index_extract[0]:edges[0]] = curr_sub_section

        self.assembly = numpy_to_sitk(np_arr, self.image_reader)

    def calc_weight_array_gaussian(self, size_extract):
        """Function to calculate the weight array for
        a gaussian weighted segmentation"""
        # define std as 10% of the size of the volume
        std = 0.5*np.ones_like(size_extract)  # 0.1
        # create a grid of distances to the center of the volume
        x = np.linspace(-size_extract[0]/2, size_extract[0]/2, size_extract[0])
        y = np.linspace(-size_extract[1]/2, size_extract[1]/2, size_extract[1])
        z = np.linspace(-size_extract[2]/2, size_extract[2]/2, size_extract[2])
        # normalize
        x = x/(size_extract[0]/2)
        y = y/(size_extract[1]/2)
        z = z/(size_extract[2]/2)
        # create a meshgrid
        x, y, z = np.meshgrid(z, y, x)
        # now transpose
        x = x.transpose(1, 0, 2)
        y = y.transpose(1, 0, 2)
        z = z.transpose(1, 0, 2)
        # calculate the weight array
        weight_array = np.exp(-0.5*(x**2/std[0]**2
                                    + y**2/std[1]**2
                                    + z**2/std[2]**2))
        # print(f"Max weight: {np.max(weight_array)}")
        # print(f"Min weight: {np.min(weight_array)}")
        return weight_array

    def create_mask(self):
        "Function to create a global image mask of areas that were segmented"
        mask = (self.number_updates > 0).astype(int)
        mask = numpy_to_sitk(mask, self.image_reader)
        self.mask = mask

        return mask

    def upsample(self, template_size=[1000, 1000, 1000]):
        from .prediction import centering
        or_im = sitk.GetImageFromArray(self.assembly)
        or_im.SetSpacing(self.image_resampled.GetSpacing())
        or_im.SetOrigin(self.image_resampled.GetOrigin())
        or_im.SetDirection(self.image_resampled.GetDirection())
        target_im = create_new(or_im)
        target_im.SetSize(template_size)
        new_spacing = (
            np.array(or_im.GetSize())*np.array(
                or_im.GetSpacing()))/np.array(template_size)
        target_im.SetSpacing(new_spacing)

        resampled = centering(or_im, target_im, order=0)
        return resampled

    def upsample_sitk(self, template_size=[1000, 1000, 1000]):

        from .prediction import resample_spacing
        resampled, _ = resample_spacing(self.assembly,
                                        template_size=template_size)
        return resampled


class VesselTree:

    def __init__(self, case, image_file, init_step, pot_branches):
        self.name = case
        self.image = image_file
        self.bifurcations = [0]
        self.branches = [[0]]
        self.steps = [init_step]
        if len(pot_branches) > 1:
            for i in range(len(pot_branches)-1):
                pot_branches[i+1]['connection'] = [0, 0]
        self.potential_branches = pot_branches
        self.caps = []

    def add_step(self, i, step, branch):
        self.steps.append(step)
        self.branches[branch].append(i)

    def remove_step(self, i, step, branch):
        self.steps.remove(step)
        self.branches[branch].remove(i)

    def remove_previous_n(self, branch, n):
        print(f'Removing last {n} steps on branch {branch}')
        print(f'Branches are: {self.branches} before')
        print(f'Nr steps before: {len(self.steps)}')
        for i in range(n-1):
            self.steps.pop(-1)
            self.branches[branch].pop(-1)
        self.branches[branch].pop(-1)  # once more for branches
        print(f'Nr steps after: {len(self.steps)}')
        print(f'Branches are: {self.branches} after')

    def add_branch(self, connector, i):
        self.branches.append([connector, i])
        self.bifurcations.append(connector)

    def remove_branch(self, branch):
        start = self.branches[branch][1]
        end = self.branches[branch][-1]
        del self.steps[start:end]
        self.branches[branch] = []
        del self.bifurcations[-1]

    def sort_potential_radius(self):
        self.potential_branches.sort(key=operator.itemgetter('radius'),
                                     reverse=True)

    def shuffle_potential(self):
        np.random.shuffle(self.potential_branches)

    def merge_pots_radius(self):
        """
        Function to loop over potential branches 
        and remove ones within 1 radius of each other
        """
        print(f"Number of pots before merge: {len(self.potential_branches)}")
        i = 0
        while i < len(self.potential_branches):
            j = i + 1
            while j < len(self.potential_branches):
                mult_r = 2  # within 2 radii
                location_close = (np.linalg.norm(
                    np.array(self.potential_branches[i]['point']) - np.array(
                        self.potential_branches[j]['point']))
                        < mult_r * self.potential_branches[i]['radius'])
                angle_range = 0.5  # 30 degrees
                tangent_i = self.potential_branches[i]['tangent']
                tangent_j = self.potential_branches[j]['tangent']
                angle_close = (np.arccos(
                    np.dot(tangent_i, tangent_j))/(np.linalg.norm(tangent_i)
                                                   * np.linalg.norm(tangent_j))
                    < angle_range)
                if location_close and angle_close:
                    del self.potential_branches[j]
                else:
                    j += 1
            i += 1
        print(f"Number of pots after merge: {len(self.potential_branches)}")

    def get_previous_n(self, branch, n):

        branch0 = self.branches[branch]
        if len(branch0) > n:
            previous_n = branch0[-n:]
        else:
            # previous_n = branch0
            previous_n = [bra for bra in branch0]  # branch0
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

    def get_previous_step(self, step_number):

        inds = [(i, colour.index(step_number))
                for i, colour in enumerate(self.branches)
                if step_number in colour]

        for ind in inds:
            if ind[1] == 0:
                continue
            else:
                ind_prev = self.branches[ind[0]][ind[1]-1]

        return self.steps[ind_prev]

    def restart_branch(self, branch):
        "Function to restart a branch from beginning"

        # change potential branches that originated in the branch
        for pot_branch in self.potential_branches:
            if pot_branch['connection'][0] == branch:
                # change the connection to first step in branch
                first_step = self.branches[branch][0]
                # find which branch the first step is in
                for j, br in enumerate(self.branches):
                    if first_step in br:
                        pot_branch['connection'] = [j, first_step]
                        break

        # if len(self.branches[branch]) > 1:
        start = self.branches[branch][1]
        end = self.branches[branch][-1]
        print(f"Removing steps {start+1} - {end}")
        del self.steps[start+1:]
        # remove the list at self.branches[branch]
        self.branches.pop(branch)
        del self.bifurcations[-1]

        # if last branch is empty, remove it - TODO
        # if not self.branches[-1]:
        #     self.branches.pop(-1)
        #     del self.bifurcations[-1]

        print(f"Restarted branch {branch}")

    def calc_ave_dice(self):
        total_dice, count = 0, 0
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
        total_time, count = 0, 0
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
            print('Average time for ' + names[j] + ' : ', time_sum[j]/counter)
        print(np.array(time_sum/counter).tolist())

    def calc_caps(self, global_assembly):
        'Temp try at calculating global caps'
        final_caps = []
        for point in self.caps:
            if not is_point_in_image(global_assembly, point):
                final_caps.append(point)
            # else:
                # print('Inside \n')

        print('Number of outlets: ' + str(len(final_caps)))
        # final_caps = orient_caps(final_caps, init_step)
        return final_caps

    def get_end_points(self):
        points = [self.steps[0]['point'] - self.steps[0]['tangent']
                  * 2 * self.steps[0]['radius']]
        for branch in self.branches:
            id = branch[-1]
            points.append(self.steps[id]['point'] + self.steps[id]['tangent']
                          * 1 * self.steps[id]['radius'])
        return points

    def plot_graph(self):
        import networkx as nx

        G = nx.DiGraph()
        G.add_edges_from(
            [('A', 'B'), ('A', 'C'), ('D', 'B'), ('E', 'C'), ('E', 'F'),
             ('B', 'H'), ('B', 'G'), ('B', 'F'), ('C', 'G')])

        nx.draw(G)

        G = nx.Graph()
        G.add_edge(1, 2, color='r', weight=3)
        G.add_edge(2, 3, color='b', weight=5)
        G.add_edge(3, 4, color='g', weight=7)

        pos = nx.circular_layout(G)

        colors = nx.get_edge_attributes(G, 'color').values()
        weights = nx.get_edge_attributes(G, 'weight').values()

        nx.draw(G, pos, edge_color=colors, width=list(weights))

        # plt.show()

    def create_tree_graph(self, dir_output):
        """
        Function to create a graph of the tree
        Each step is a node, all nodes in a branch are connected
        Branches are connected by 'connection' attribute
        The size of the node is proportional to the radius
        """
        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.DiGraph()
        for i, step in enumerate(self.steps):
            G.add_node(i, radius=step['radius'])
        for branch in self.branches:
            for i in range(len(branch)-1):
                G.add_edge(branch[i], branch[i+1])

        # pos = nx.spring_layout(G)
        pos = nx.planar_layout(G)
        # pos = nx.kamada_kawai_layout(G)
        # pos = nx.spectral_layout(G)
        # pos = nx.multipartite_layout(G)
        # pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

        # nx.draw(G, pos, with_labels=True, node_size=100,
        #         node_color='lightblue', font_weight='bold',
        #         font_size=1, edge_color='grey')
        nx.draw(G, pos, with_labels=True,
                node_color='lightblue', font_weight='bold')

        # save the graph
        plt.savefig(dir_output+'/tree_graph.png')
        plt.close()

        # plt.show()

    def create_tree_graph_smaller(self, dir_output):
        """
        Function to create a graph of the tree
        Now only keep start and end of branches and bifurcations
        Bifurcations are connected by 'connection' attribute
        Bifurcations are between start and end of branches

        Args:
            dir_output: directory to save the graph
        """
        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.DiGraph()

        # add all the nodes
        for i, step in enumerate(self.steps):
            G.add_node(i, radius=step['radius'])
        # add all the edges
        for branch in self.branches:
            for i in range(len(branch)-1):
                G.add_edge(branch[i], branch[i+1])

        # now remove nodes with one parent and one child and connect them
        for i, step in enumerate(self.steps):
            # if i is node
            if i in G.nodes:
                if (len(list(G.successors(i))) == 1
                   and len(list(G.predecessors(i))) == 1):
                    pred = list(G.predecessors(i))[0]
                    succ = list(G.successors(i))[0]
                    G.add_edge(pred, succ)
                    G.remove_node(i)

        pos = nx.planar_layout(G)
        # pos = nx.kamada_kawai_layout(G)
        # pos = nx.spectral_layout(G)
        # pos = nx.multipartite_layout(G)
        # pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

        # nx.draw(G, pos, with_labels=True, node_size=100,
        #         node_color='lightblue', font_weight='bold',
        #         font_size=1, edge_color='grey')
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                font_weight='bold')

        # save the graph
        plt.savefig(dir_output+'/tree_graph_smaller.png')
        plt.close()

    def create_tree_polydata_v1(self, dir_output):
        """
        Function to create a polydata of the steps in the tree
        The function uses self.branches to know the connections between steps
        The actual points are in self.steps
        An example of a branch is [0, 1, 2, 3]
        where 0 is the start of the branch
        The next branch is [1, 4, 5, 6] where 1 is where
        the branch connects to another (previous) branch
        4, 5, 6 are the steps in the branch
        We use vtk.lines to connect the points in the branches so that:
            0 - 1 - 2 - 3 are connected
            1 - 4 - 5 - 6 are connected
        Such that 1 is connected to 0, 2 and 4

        Args:
            dir_output: directory to save the graph
        """
        import vtk

        print("\nCreating polydata of tree")
        print("Branches are:")
        for branch in self.branches:
            print(branch)

        # Create the polydata
        polydata = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()

        # add the points
        for step in self.steps:
            points.InsertNextPoint(step['point'])
        polydata.SetPoints(points)

        # add the lines
        for branch in self.branches:
            line = vtk.vtkLine()
            for i in range(len(branch)-1):
                line.GetPointIds().SetId(0, branch[i])
                line.GetPointIds().SetId(1, branch[i+1])
                lines.InsertNextCell(line)
        polydata.SetLines(lines)

        # save the polydata
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(dir_output + '/tree_polydata_v1.vtp')
        writer.SetInputData(polydata)
        writer.Write()

    def create_tree_polydata_v2(self, dir_output):
        """
        Function to create a polydata of the steps in the tree
        The function uses self.branches to know the connections between steps
        The actual points are in self.steps
        An example of a branch is [0, 1, 2, 3]
        where 0 is the start of the branch
        The next branch is [1, 4, 5, 6] where 1 is where
        the branch connects to another (previous) branch
        4, 5, 6 are the steps in the branch
        We use vtk.lines to connect the points in the branches so that:
            0 - 1 - 2 - 3 are connected
            1 - 4 - 5 - 6 are connected
        Such that 1 is connected to 0, 2 and 4

        Args:
            dir_output: directory to save the graph
        """
        import vtk

        # first create point array
        points = []
        for step in self.steps:
            points.append(step['point'])
        points = np.array(points)

        # create the connection array
        connections = np.zeros((len(self.steps), len(self.steps)))
        # add connection within branches
        for branch in self.branches:
            for i in range(len(branch)-1):
                connections[branch[i], branch[i+1]] = 1
        # add connection between branches
        for branch in self.branches:
            if len(branch) > 1:
                step_in_other_branch = branch[0]
                first_step_this_branch = branch[1]
                connections[step_in_other_branch, first_step_this_branch] = 1

        # create the polydata of the mesh
        polydata = vtk.vtkPolyData()
        points_vtk = vtk.vtkPoints()
        lines = vtk.vtkCellArray()

        # add the points
        for point in points:
            points_vtk.InsertNextPoint(point)
        polydata.SetPoints(points_vtk)

        # add multiple attributes to the points
        # radius
        radius = np.array([step['radius'] for step in self.steps])
        # ids
        ids = np.arange(len(self.steps))
        # branch number
        branch_number = np.zeros(len(self.steps))
        for i, branch in enumerate(self.branches):
            for step in branch:
                branch_number[step] = int(i)
        branch_number = branch_number.astype(int)
        # bifurcation
        bifurcation = np.zeros(len(self.steps))
        for bif in self.bifurcations:
            bifurcation[bif] = 1
        bifurcation = bifurcation.astype(int)
        # create the arrays
        radius_vtk = vtk.vtkDoubleArray()
        ids_vtk = vtk.vtkIntArray()
        branch_number_vtk = vtk.vtkIntArray()
        bifurcation_vtk = vtk.vtkIntArray()

        # add the values
        for i in range(len(self.steps)):
            radius_vtk.InsertNextValue(radius[i])
            ids_vtk.InsertNextValue(ids[i])
            branch_number_vtk.InsertNextValue(branch_number[i])
            bifurcation_vtk.InsertNextValue(bifurcation[i])

        # add the arrays to the polydata
        polydata.GetPointData().AddArray(radius_vtk)
        polydata.GetPointData().AddArray(ids_vtk)
        polydata.GetPointData().AddArray(branch_number_vtk)
        polydata.GetPointData().AddArray(bifurcation_vtk)

        # add the names of the arrays
        polydata.GetPointData().GetArray(0).SetName('Radius')
        polydata.GetPointData().GetArray(1).SetName('ID')
        polydata.GetPointData().GetArray(2).SetName('BranchID')
        polydata.GetPointData().GetArray(3).SetName('BifLabel')

        # add the lines
        for i in range(len(connections)):
            for j in range(len(connections)):
                if connections[i, j] == 1:
                    line = vtk.vtkLine()
                    line.GetPointIds().SetId(0, i)
                    line.GetPointIds().SetId(1, j)
                    lines.InsertNextCell(line)
        polydata.SetLines(lines)

        # save the polydata
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(dir_output + '/tree_polydata_v2.vtp')
        writer.SetInputData(polydata)
        writer.Write()

    def plot_radius_distribution(self, dir_output):
        """
        Function to plot the radius distribution of the tree
        """
        import matplotlib.pyplot as plt
        radii = [step['radius'] for step in self.steps]
        n_step = len(radii)
        plt.hist(radii, bins=20)
        # plt.show()
        # save the graph
        plt.savefig(dir_output + '/radius_distribution.png')
        plt.close()

        # plot the radius across steps and have space betwwen steps on x axis
        plt.figure(figsize=(25, 5))
        plt.plot(range(n_step), radii)
        plt.xlabel('Step')
        plt.ylabel('Radius')
        plt.title('Radius Change')
        # plt.xticks(np.arange(min(range(n_step)),
        #                      max(range(n_step))+1,
        #                      max(range(n_step))//20))
        # plt.tick_params(axis='x', which='major', labelsize=3)
        # plt.show()
        # save the graph
        plt.savefig(dir_output + '/radius_evolution.png')
        # add red vertical line for bifurcations
        # add blue horizontal line for end of branches
        for branch in self.branches:
            plt.axvline(x=branch[-1], color='b', linestyle='--')
        plt.savefig(dir_output + '/radius_evolution_branches.png')
        for bif in self.bifurcations:
            plt.axvline(x=bif, color='r', linestyle='--')
        plt.savefig(dir_output + '/radius_evolution_branches_bif.png')
        plt.close()


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
        self.potential_branches = [i for i in self.potential_branches
                                   if not (i['radius'] == pot_branch['radius'])
                                   ]

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


def print_error(output_folder,
                i,
                step_seg,
                image=None,
                predicted_vessel=None,
                old_point_ref=None,
                centerline_poly=None):

    now = datetime.now()
    dt_string = now.strftime("_%d_%m_%Y_%H_%M_%S")
    directory = output_folder + 'errors/'+str(i) + '_error_'+dt_string

    polydata_point = points2polydata([step_seg['point'].tolist()])
    write_vtk_polydata(polydata_point, directory + 'point.vtp')

    try:
        if step_seg['img_file'] and not step_seg['is_inside']:
            sitk.WriteImage(image, directory + 'img.vtk')

            if step_seg['seg_file']:
                sitk.WriteImage(predicted_vessel, directory + 'seg.vtk')

                if step_seg['surf_file']:
                    write_vtk_polydata(step_seg['surface'], directory + 'surf.vtp')

                    if step_seg['centerline']:
                        polydata_point = points2polydata(
                            [step_seg['old_point_ref'].tolist()])
                        write_vtk_polydata(polydata_point,
                                        directory + 'old_point_ref.vtp')
                        write_vtk_polydata(centerline_poly,
                                        directory + 'cent.vtp')
    except Exception as e:
        print('Didnt work to save error')
        print(e)

def create_step_dict(old_point,
                     old_radius,
                     new_point,
                     new_radius,
                     angle_change=None):

    step_dict = {}
    step_dict['old point'] = old_point
    step_dict['point'] = new_point
    step_dict['old radius'] = old_radius
    step_dict['tangent'] = (new_point - old_point)/np.linalg.norm(
                                                    new_point - old_point)
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


def get_old_ref_point(vessel_tree,
                      step_seg,
                      i,
                      mega_sub=False,
                      mega_sub_N=0):

    if not mega_sub:
        # return the old point
        if i != 0:
            # return the old of two steps before
            branch = len(vessel_tree.branches) - 1
            prev_n = vessel_tree.get_previous_n(branch, 3)
            print(f"Prev n: {prev_n}")
            step = vessel_tree.steps[prev_n[0]]
            old_point_ref = step['old point']
            # prev_step = vessel_tree.get_previous_step(i)
            # old_point_ref = prev_step['old point']
        elif i == 0:
            old_point_ref = step_seg['old point']
    else:
        # return the old of the oldest step used for mega subvolume
        branch = len(vessel_tree.branches) - 1
        prev_n = vessel_tree.get_previous_n(branch, mega_sub_N)

        step = vessel_tree.steps[prev_n[0]]
        old_point_ref = step['old point']

    return old_point_ref


def calc_centerline_global(predicted_vessels, initial_seeds):
    """
    Function to loop over inital seeds and construct global centerline(s)

    Targets are not defined here, but in the calc_centerline_fmm function

    Parameters:
    -----------
        predicted_vessels: SITK image
            the predicted vessel segmentation
        initial_seeds: list of np arrays
            the initial seeds for the global centerline
    """
    print(f"""Calculating global centerline
          with {len(initial_seeds)} initial seeds""")
    # create a list for centerline polydata
    centerline_poly = []
    # create a list for targets
    targets_list = []
    # create a success flag
    success_list = []
    # loop over the initial seeds
    for seed in initial_seeds:
        # keep the component with the seed
        predicted_vessel = keep_component_seeds(predicted_vessels, [seed])
        # calculate the centerline
        cent, success, targets = calc_centerline_fmm(predicted_vessel,
                                                     seed=seed,
                                                     min_res=700,
                                                     relax_factor=3,
                                                     return_target_all=True,
                                                     verbose=True,
                                                     return_failed=True)
        centerline_poly.append(cent)
        targets_list.append(targets)
        success_list.append(success)

    # append the centerline polydata list to a single polydata
    centerline_poly = appendPolyData(centerline_poly)
    # make single list
    targets_list = [item for sublist in targets_list for item in sublist]
    # success if any of the centerlines were successful
    success_list = any(success_list)

    return centerline_poly, targets_list, success_list
