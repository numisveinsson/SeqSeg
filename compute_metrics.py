import os
import SimpleITK as sitk
import numpy as np
from modules import vtk_functions as vf
from vtk.util.numpy_support import vtk_to_numpy as v2n

from scipy.stats import ttest_ind, ttest_rel

def dice(pred, truth):

    if not isinstance(pred, np.ndarray):
        pred = sitk.GetArrayFromImage(pred)
        truth = sitk.GetArrayFromImage(truth)
    pred = pred.astype(np.int32)
    true = truth.astype(np.int32)
    if true.max() > 1:
        true = true // true.max()

    num_class = np.unique(true)

    #change to one hot
    dice_out = [None]*len(num_class)
    for i in range(1, len(num_class)):
        pred_c = pred == num_class[i]
        true_c = true == num_class[i]
        dice_out[i] = np.sum(pred_c*true_c)*2.0 / (np.sum(pred_c) + np.sum(true_c))

    mask =( pred > 0 )+ (true > 0)
    dice_out[0] = np.sum((pred==true)[mask]) * 2. / (np.sum(pred>0) + np.sum(true>0))
    
    return dice_out[0]

def masked_dice(self, masked_dir):
        mask = sitk.ReadImage(masked_dir)
        filter = sitk.GetArrayFromImage(mask)
        pred = sitk.GetArrayFromImage(self.seg_pred)
        pred[filter == 0] = 0

        dice = dice_score(pred, self.seg_truth)[0]

        return dice

def hausdorff(pred, truth):

    # check pixel type
    if truth.GetPixelID() != sitk.sitkUInt8:
        truth = sitk.Cast(truth, sitk.sitkUInt8)
    if pred.GetPixelID() != sitk.sitkUInt8:
        pred = sitk.Cast(pred, sitk.sitkUInt8)

    # check that image origin is the same
    if truth.GetOrigin() != pred.GetOrigin():
        pred.SetOrigin(truth.GetOrigin())

    haus_filter = sitk.HausdorffDistanceImageFilter()
    haus_filter.Execute(pred, truth)

    return haus_filter.GetAverageHausdorffDistance()

def percent_centerline_length(pred, cent_truth):

    # check if cent_truth is string or sitk image
    if isinstance(cent_truth, str):
        cent_truth = vf.read_geo(cent_truth).GetOutput()
    # now cent_truth is vtkPolyData
    num_cells = cent_truth.GetNumberOfCells()   # number of cells in centerline

    centerline_length = 0
    cent_length_init = 0
    # check if cells are within pred
    for i in range(num_cells):
    
            cell = cent_truth.GetCell(i)
            num_points = cell.GetNumberOfPoints()
            points = cell.GetPoints()
            # calulate length of cell
            length = 0
            for j in range(num_points-1):
                p1 = points.GetPoint(j)
                p2 = points.GetPoint(j+1)
                length += np.linalg.norm(np.array(p1)-np.array(p2))
            centerline_length += length
            # check if points are within pred
            loc1 = points.GetPoint(0)
            loc2 = points.GetPoint(num_points-1)
            index1 = pred.TransformPhysicalPointToIndex(loc1)
            index2 = pred.TransformPhysicalPointToIndex(loc2)
            # check if location is in pred
            try:
                if pred[index1] == 1 and pred[index2] == 1:
                    cent_length_init += length
            except:
                # print('Index out of bounds')
                # if location is not within boundary, remove point
                centerline_length -= length
    
    return cent_length_init/centerline_length



def percent_centerline_points(pred, cent_truth):

    # check if cent_truth is string or sitk image
    if isinstance(cent_truth, str):
        cent_truth = vf.read_geo(cent_truth).GetOutput()

    num_points = cent_truth.GetNumberOfPoints()   # number of points in centerline
    cent_data = vf.collect_arrays(cent_truth.GetPointData())
    c_loc = v2n(cent_truth.GetPoints().GetData())             # point locations as numpy array    

    num_points_init = num_points

    # check if points are within pred
    for i in range(num_points):

        location = c_loc[i]
        index = pred.TransformPhysicalPointToIndex(location.tolist())

        # check if location is in pred
        try:
            if pred[index] == 0:
                num_points_init -= 1

        except:
            # if location is not within boundary, remove point
            num_points -= 1
            num_points_init -= 1

    return num_points_init/num_points

def only_keep_mask(pred0, mask):
    
    filter_mask = sitk.GetArrayFromImage(mask)
    pred = sitk.GetArrayFromImage(pred0)
    pred[filter_mask == 0] = 0

    from modules import sitk_functions as sf
    pred = sf.numpy_to_sitk(pred, pred0)

    return pred

def process_case_name(case_name):

    'Change this function if naming convention changes'

    case_name = case_name[:-17]

    if 'seg' in case_name:
        name = case_name[13:]
        name_split = case_name.split('_')
        # add until 'seg'
        name = ''
        for n in name_split:
            if n == 'seg':
                break
            name += n
            name += '_'
        name = name[:-1]
    else:
        name = case_name[15:]
    
    return name

def read_truth(case, truth_folder):

    try:
        truth = sitk.ReadImage(truth_folder+case+'.vtk')
    except:
        try:
            truth = sitk.ReadImage(truth_folder+case+'.mha')
        except:
            truth = sitk.ReadImage(truth_folder+case+'.nii.gz')

    return truth

def read_seg(pred_folder, seg):

    pred = sitk.ReadImage(pred_folder+seg)

    return pred

def from_prob_to_binary(pred):

    # convert to binary
    pred_binary = sitk.BinaryThreshold(pred, lowerThreshold=0.5, upperThreshold=1)

    return pred_binary

def keep_largest_label(pred_binary, num_labels=1):

    # get connected components
    ccimage = sitk.ConnectedComponent(pred_binary)
    # check how many labels there are
    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(ccimage)
    labels = label_stats.GetLabels()
    # print(f"Number of labels: {len(labels)}")
    # check which label is largest
    label_sizes = [label_stats.GetPhysicalSize(l) for l in labels]
    # print(f"Label sizes: {label_sizes}")
    # sort labels by size
    labels = [x for _,x in sorted(zip(label_sizes,labels), reverse=True)]
    # print(f"Sorted labels: {labels}")
    # keep only largest label
    if num_labels == 1:
        label = labels[0]
    else:
        label = labels[:num_labels]
    # print(f"Keeping label {label}")
    if num_labels == 1:
        labelImage = sitk.BinaryThreshold(ccimage, lowerThreshold=label, upperThreshold=label)
        return labelImage
    else:
        labelImage = sitk.BinaryThreshold(ccimage, lowerThreshold=label[0], upperThreshold=label[0])
        for l in label[1:]:
            labelImage += sitk.BinaryThreshold(ccimage, lowerThreshold=l, upperThreshold=l)
        return labelImage

def pre_process(pred, write_postprocessed):
    
    if write_postprocessed:
        # marching cubes
        from modules import vtk_functions as vf
        surface = vf.evaluate_surface(pred, 0.5) # Marching cubes
        # surface_smooth = vf.smooth_surface(surface, 12) # Smooth marching cubes
        vf.write_geo(pred_folder+folder+'/postprocessed/'+case+'.vtp', surface)

    # only change to binary if prediction is probability map
    if pred.GetPixelID() != sitk.sitkUInt8:

        pred  = from_prob_to_binary(pred)
    #else:
        #print("Prediction is already binary")

    labelImage = keep_largest_label(pred)

    if write_postprocessed:
        sitk.WriteImage(labelImage, pred_folder+folder+'/postprocessed/'+case+'.mha')

    # print np array of label image
    # labelImageArray = sitk.GetArrayFromImage(labelImage)
    # print(f"Label image array: {labelImageArray}")

    return labelImage

def get_case_names(folder, pred_folder):
    
    segs = os.listdir(pred_folder+folder)
    print(f"Segs: {segs}")
    #only keep segmentation files and ignore hidden files
    segs = [seg for seg in segs if '.' not in seg[0]]
    # only keep files not folders
    segs = [seg for seg in segs if '.' in seg]
    # sort
    segs.sort()

    segs = [process_case_name(seg) for seg in segs]
    
    return segs



def calc_metric(metric, pred, truth, mask=None, centerline=None):

    if mask is not None:
        pred = only_keep_mask(pred, mask)

    if metric == 'dice':

        score = dice(pred, truth)

    elif metric == 'hausdorff':

        score = hausdorff(pred, truth)

    elif metric == 'centerline overlap':

        score = percent_centerline_length(pred, centerline)

    return score

def get_names_folders(list_folders):

    names = []

    for i,folder in enumerate(list_folders):

        if 'pred_seqseg' in folder: 

            names.append('SeqSeg')

        elif 'pred_benchmark_2d' in folder:

            names.append('2D Global\nnnU-Net')

        elif 'pred_benchmark_3d' in folder:

            names.append('3D Global\nnnU-Net')

        else:
            names.append(folder)

    return names

def get_metric_name(metric):
    if metric == 'dice':
        return 'Dice Score'
    elif metric == 'hausdorff':
        return 'Avg. Hausdorff Distance'
    elif metric == 'centerline overlap':
        return 'Centerline Overlap'
    else:
        return metric

if __name__=='__main__':

    name_graph = 'Comparison'
    save_name = 'test_keep'
    preprocess_pred = True
    masked = True
    write_postprocessed = True

    print_case_names = True

    keep_largest_label_benchmark = True

    # input folder of segmentation results
    pred_folder = '/Users/numisveins/Documents/data_seqseg_paper/pred_aortas_june24_3/'
    pred_folders = os.listdir(pred_folder)
    # only keep folders and ignore hidden files
    pred_folders = [folder for folder in pred_folders if '.' not in folder and 'old' not in folder and 'gt' not in folder]
    pred_folders.sort()
    print(f"Prediction folders: {pred_folders}")
    # pred_folders = [folder for folder in pred_folders if '3d' not in folder]

    truth_folder = '/Users/numisveins/Documents/vascular_data_3d/truths/'
    cent_folder = '/Users/numisveins/Documents/vascular_data_3d/centerlines/'
    mask_folder = '/Users/numisveins/Documents/vascular_data_3d/masks_around_truth/masks_4r/'

    # output folder for plots
    output_folder = ''

    # modalities
    modalities = ['mr', 'ct']
    # metrics
    metrics = ['dice', 'hausdorff', 'centerline overlap']#  'dice mask', 'hausdorff mask']

    for modality in modalities:

        for metric in metrics:

            print(f"\nCalculating {metric} for {modality} data...")
                # keep track of scores to plot
            scores = {}

            folders_mod = [folder for folder in pred_folders if modality in folder]
            case_names = get_case_names([fo for fo in folders_mod if 'seqseg' in fo][0], pred_folder)
            print(f"Case names: {case_names}")
            for folder in folders_mod:
                
                print(f"\n{folder}:")

                scores[folder] = []

                segs = os.listdir(pred_folder+folder)
                #only keep segmentation files and ignore hidden files
                segs = [seg for seg in segs if '.' not in seg[0]]
                # only keep files not folders
                segs = [seg for seg in segs if '.' in seg]
                # sort
                segs.sort()

                if write_postprocessed:
                    os.makedirs(pred_folder+folder+'/'+'postprocessed', exist_ok=True)


                for i,seg in enumerate(segs):

                    if 'seqseg' in folder:
                        case = process_case_name(seg)
                    else:
                        # remove file extension
                        case = case_names[i]

                    pred = read_seg(pred_folder+folder+'/', seg)
                    truth = read_truth(case, truth_folder)

                    if preprocess_pred and 'segseg' in folder:
                        pred = pre_process(pred, write_postprocessed)
                    elif preprocess_pred and keep_largest_label_benchmark:
                        pred = pre_process(pred, write_postprocessed)

                    if write_postprocessed:
                        # marching cubes
                        from modules import vtk_functions as vf
                        surface = vf.evaluate_surface(pred, 0.5) # Marching cubes
                        # surface_smooth = vf.smooth_surface(surface, 12) # Smooth marching cubes
                        vf.write_geo(pred_folder+folder+'/postprocessed/'+case+'.vtp', surface)
                    
                    if masked and 'seqseg' in folder:
                        mask = read_truth('mask_'+case, mask_folder)
                    else: mask = None

                    if metric == 'centerline overlap':
                        centerline = vf.read_geo(cent_folder+'/'+case+'.vtp').GetOutput()
                    else: centerline = None

                    score = calc_metric(metric, pred, truth, mask, centerline)

                    scores[folder].append(score)
                    if print_case_names:
                        print(f"{case}: {score:.3f}")
                    else:
                        print(f"{score:.3f}")
                if print_case_names:
                    print(f"Average {metric}: {np.mean(scores[folder]):.3f}")
                else:
                    print(f"{np.mean(scores[folder]):.3f}")

            # t-test to compare score from folder with 'seqseg' in name to score from folder without 'seqseg' in name
            if len(scores.keys()) > 1:
                for i in range(len(scores.keys())-1):
                    for j in range(i+1, len(scores.keys())):
                        t, p = ttest_ind(scores[list(scores.keys())[i]], scores[list(scores.keys())[j]])
                        # print(f"unpaired p-value between {list(scores.keys())[i]} and {list(scores.keys())[j]}: {p}")
                        t, p = ttest_rel(scores[list(scores.keys())[i]], scores[list(scores.keys())[j]])
                        print(f"\n {p}: paired p-value between {list(scores.keys())[i]} and {list(scores.keys())[j]}")
                        # if p < 0.05, then print
                        if p < 0.05:
                            print(f"***DING DING*** Significant difference between {list(scores.keys())[i]} and {list(scores.keys())[j]}\n")
                        else:
                            print(f"No significant difference between {list(scores.keys())[i]} and {list(scores.keys())[j]}\n")

            # Make box plot for modality
            import matplotlib.pyplot as plt
            plt.figure()
            # set font to Times New Roman
            plt.rcParams["font.family"] = "Times New Roman"
            # set font size to 14
            plt.rcParams.update({'font.size': 20})
            
            # create boxplot with colors
            colors = ['pink', 'lightblue', 'lightgreen', 'orange', 'purple', 'yellow', 'brown', 'black', 'grey']
            boxplot = plt.boxplot(scores.values(), patch_artist=True)
            for patch, color in zip(boxplot['boxes'], colors):
                patch.set_facecolor(color)
            # add legend
            # plt.legend(boxplot['boxes'], scores.keys())

            # add x axis labels
            plt.xticks(range(1, len(scores.keys())+1), get_names_folders(scores.keys()))
            #plt.title(f'{modality.upper()}')
            # set y axis lower limit to 0
            if 'hausdorff' in metric:
                plt.ylim(bottom=0)
                plt.ylim(top=0.5)
            plt.ylabel(f'{get_metric_name(metric)}')
            # plt.xlabel('Method')
            if 'dice' in metric:
                plt.ylim(top=1)
                plt.ylim(bottom=0.5)
            if 'centerline' in metric:
                plt.ylim(top=1)
                plt.ylim(bottom=0)

            # add grid on y axis
            plt.grid(axis='y')
            plt.tight_layout() # make sure labels are not cut off
            # add two horizontal lines at means
            means = [np.mean(scores[folder]) for folder in scores.keys()]

            # add p-values from t-test
            if len(scores.keys()) > 1:
                for i in range(len(scores.keys())-1):
                    for j in range(i+1, len(scores.keys())):
                        t, p = ttest_ind(scores[list(scores.keys())[i]], scores[list(scores.keys())[j]])
                        if p < 0.05:
                            plt.text(i+1.5, 0.5, '*', fontsize=20, horizontalalignment='center', verticalalignment='center')
                        if p < 0.01:
                            plt.text(i+1.5, 0.4, '**', fontsize=20, horizontalalignment='center', verticalalignment='center')
                        if p < 0.001:
                            plt.text(i+1.5, 0.3, '***', fontsize=20, horizontalalignment='center', verticalalignment='center')

            # add horizontal lines at means with same color as boxplot
            # for mean, color in zip(means, colors):
            #     plt.hlines(mean, 0.5, len(scores.keys())+0.5, colors=color, linestyles='dashed', label='mean', linewidth=1)

            # plt.hlines(means, 0.5, len(scores.keys())+0.5, colors='r', linestyles='dashed', label='mean', linewidth=1)
            # save plot
            plt.savefig(output_folder + f'{save_name}_{metric}_{modality}_scores.png',bbox_inches="tight")
            # plt.savefig(output_folder + f'{save_name}_{metric}_{modality}_scores.svg',bbox_inches="tight")
            plt.close()

