from .vtk_functions import read_geo, collect_arrays, is_point_in_image
from .prediction import dice_score
from vtk.util.numpy_support import vtk_to_numpy as v2n
import numpy as np
import SimpleITK as sitk

class EvaluateTracing:

    def __init__(self, case, seed, dir_seg_truth, dir_surf_vtp_truth, dir_cent_vtp_truth, seg_pred, surf_vtp_prediction):
        self.name = case
        self.seed = seed
        if dir_seg_truth:
            seg_truth = sitk.ReadImage(dir_seg_truth)
            seg_truth = sitk.GetArrayFromImage(seg_truth).astype('int')
            if seg_truth.max() > 1:
                seg_truth = seg_truth/(seg_truth.max())
            self.seg_truth = seg_truth
        self.seg_pred = seg_pred
        self.surf_truth = dir_surf_vtp_truth
        self.surf_pred = surf_vtp_prediction
        self.cent_truth = dir_cent_vtp_truth

    def count_branches(self):
        """
        Function to count how many branches in grountruth were captured in tracing
        """
        move_distance_radius = 0.1

        ## Centerline
        cent = read_geo(self.cent_truth).GetOutput()  # read in geometry
        num_points = cent.GetNumberOfPoints()   # number of points in centerline
        cent_data = collect_arrays(cent.GetPointData())
        c_loc = v2n(cent.GetPoints().GetData())             # point locations as numpy array
        radii = cent_data['MaximumInscribedSphereRadius']   # Max Inscribed Sphere Radius as numpy array
        cent_id = cent_data['CenterlineId']
        bifurc_id = cent_data['BifurcationIdTmp']

        try:
            num_cent = len(cent_id[0]) # number of centerlines (one is assembled of multiple)
        except:
            num_cent = 1 # in the case of only one centerline

        missed_branches = 0
        percent_caught = []
        total_lengths = []
        ids_total = []
        for ip in range(num_cent):
            try:
                ids = [i for i in range(num_points) if cent_id[i,ip]==1]    # ids of points belonging to centerline ip
            except:
                ids = [i for i in range(num_points)]
            locs = c_loc[ids] # locations of those points
            rads = radii[ids] # radii at those locations
            bifurc = bifurc_id[ids]

            if self.seed in locs:
                #print(f"Initial seed on centerline: {ip}")
                ind = np.where(locs==self.seed)[0][0]
                locs = np.delete(locs, np.s_[0:ind], 0)
                rads = np.delete(rads, np.s_[0:ind], 0)
                bifurc = np.delete(bifurc, np.s_[0:ind], 0)

            on_cent = True
            count = 0 # the point along centerline
            first_count = True
            lengths = [0]
            lengths_prev = [0]
            #print("\n ** Ip is " + str(ip)+"\n")
            while on_cent:
                if not (ids[count] in ids_total):
                    if first_count:
                        min_count = count
                        first_count = False
                        total_length = np.cumsum(np.insert(np.linalg.norm(np.diff(locs[min_count:], axis=0), axis=1), 0, 0))[-1]
                        #print(f"Min count is: {min_count}")
                    # Do something at this location
                    if not is_point_in_image(self.seg_pred, locs[count]): #+ step_seg['radius']*step_seg['tangent']):
                        on_cent = False
                        missed_branches += 1
                        #print(f"Total length: {total_length}")
                        #print(f"Prev lengths: {lengths_prev}")
                        perc = round(lengths_prev[-1]/total_length,3)
                        #if perc == 0: import pdb; pdb.set_trace()
                if not first_count:
                    lengths_prev = np.cumsum(np.insert(np.linalg.norm(np.diff(locs[min_count:count], axis=0), axis=1), 0, 0))
                lengths = np.cumsum(np.insert(np.linalg.norm(np.diff(locs[count:], axis=0), axis=1), 0, 0))
                move = 1
                count = count+1
                if count == len(locs):
                    on_cent = False
                    perc=1
                    break
                move_distance = move_distance_radius*rads[count]
                while lengths[move] < move_distance :
                    count = count+1
                    move = move+1
                    if count == len(locs):
                        on_cent = False
                        perc=1
                        break
            percent_caught.append(perc)
            total_lengths.append(total_length)
            ids_total.extend(ids)

        print(str(missed_branches)+'/'+str(num_cent)+' branches missed\n')
        print(percent_caught)
        print(total_lengths)
        print(num_cent)
        total_perc = (np.array(percent_caught)*np.array(total_lengths)/np.array(total_lengths).sum()).sum()
        print(f"Total caught {total_perc}")
        #total_perc = np.array(percent_caught).sum()/len(percent_caught)

        return [missed_branches, num_cent], percent_caught, total_perc

    def calc_dice_score(self):

        dice = dice_score(sitk.GetArrayFromImage(self.seg_pred), self.seg_truth)[0]

        print('Global dice score: ', dice)
        return dice

    def masked_dice(self, masked_dir):
        mask = sitk.ReadImage(masked_dir)
        filter = sitk.GetArrayFromImage(mask)
        pred = sitk.GetArrayFromImage(self.seg_pred)
        pred[filter == 0] = 0

        dice = dice_score(pred, self.seg_truth)[0]

        return dice



    def calc_sens_spec(self):

        return sensitivity_specificity(self.seg_pred, self.seg_truth)

def sensitivity_specificity(pred, truth):
        C = (((pred==1)*2 + (truth==1)).reshape(-1,1) == range(4)).sum(0)
        sensitivity, specificity = C[3]/C[1::2].sum(), C[0]/C[::2].sum()

        return sensitivity, specificity
