import os
import SimpleITK as sitk
import vtk
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')

from modules import vtk_functions as vf
from modules.capping import (bryan_get_clipping_parameters,
                             bryan_generate_oriented_boxes,
                             bryan_clip_surface)
from vtk.util.numpy_support import vtk_to_numpy as v2n

from scipy.stats import ttest_ind, ttest_rel, wilcoxon


def hausdorf_95(pred, truth):
    from SimpleITK import GetArrayViewFromImage as ArrayView
    from functools import partial

    prediction = pred
    gold = truth
    perc = 100

    distance_map = partial(sitk.SignedMaurerDistanceMap, squaredDistance=False, useImageSpacing=True)

    num_labels = 1
    for label in range(1, num_labels + 1):
        gold_surface = sitk.LabelContour(gold == label, False)
        prediction_surface = sitk.LabelContour(prediction == label, False)

        # Get distance map for contours (the distance map computes the minimum distances)
        prediction_distance_map = sitk.Abs(distance_map(prediction_surface))
        gold_distance_map = sitk.Abs(distance_map(gold_surface))

        # Find the distances to surface points of the contour.  Calculate in both directions
        gold_to_prediction = ArrayView(prediction_distance_map)[ArrayView(gold_surface) == 1]
        prediction_to_gold = ArrayView(gold_distance_map)[ArrayView(prediction_surface) == 1]

        # Find the 95% Distance for each direction and average
        print((np.percentile(prediction_to_gold, perc) + np.percentile(gold_to_prediction, perc)) / 2.0)

        return (np.percentile(prediction_to_gold, perc) + np.percentile(gold_to_prediction, perc)) / 2.0


def mcnemar_test(list_1, list_2):
    """
    Perform McNemar test for paired data
    Parameters
    ----------
    list_1 : list
        List of scores for method 1
    list_2 : list
        List of scores for method 2

    Returns
    -------
    p : float
        p-value
    """
    import numpy as np
    from scipy.stats import binom
    # check if lists have the same length
    if len(list_1) != len(list_2):
        print("Lists must have the same length")
        return
    # number of discordant pairs
    discordant = 0
    for i in range(len(list_1)):
        if list_1[i] > 0.5 and list_2[i] < 0.5:
            discordant += 1
        elif list_1[i] < 0.5 and list_2[i] > 0.5:
            discordant += 1
    # perform McNemar test
    p = 1 - 2 * binom.cdf(discordant, len(list_1), 0.5)
    return p


def dice(pred, truth):

    if not isinstance(pred, np.ndarray):
        pred = sitk.GetArrayFromImage(pred)
        truth = sitk.GetArrayFromImage(truth)
    pred = pred.astype(np.int32)
    true = truth.astype(np.int32)
    if true.max() > 1:
        true = true // true.max()

    num_class = np.unique(true)
    # change to one hot
    dice_out = [None]*len(num_class)
    for i in range(1, len(num_class)):
        pred_c = pred == num_class[i]
        true_c = true == num_class[i]
        dice_out[i] = np.sum(pred_c*true_c)*2.0 / (np.sum(pred_c)
                                                   + np.sum(true_c))

    mask = (pred > 0) + (true > 0)
    dice_out[0] = np.sum((pred == true)[mask]) * 2. / (np.sum(pred > 0)
                                                       + np.sum(true > 0))

    return dice_out[0]


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

    # Get 95th percentile Hausdorff distance

    return haus_filter.GetAverageHausdorffDistance()


def percent_centerline_length(pred, cent_truth):

    # check if cent_truth is string or sitk image
    if isinstance(cent_truth, str):
        cent_truth = vf.read_geo(cent_truth).GetOutput()
    # now cent_truth is vtkPolyData
    num_cells = cent_truth.GetNumberOfCells()   # number of cells in centerline

    centerline_length = 0
    cent_length_init = 0
    points_checked = []
    # check if cells are within pred
    for i in range(num_cells):

        cell = cent_truth.GetCell(i)
        num_points = cell.GetNumberOfPoints()
        points = cell.GetPoints()
        # calulate length of cell
        length = 0
        length_in = 0
        for j in range(num_points-1):
            p1 = points.GetPoint(j)
            p2 = points.GetPoint(j+1)
            if p2 in points_checked:
                # print("Point already checked", end=' ')
                continue
            length += np.linalg.norm(np.array(p1)-np.array(p2))
            # check if points are within pred
            index1 = pred.TransformPhysicalPointToIndex(p1)
            index2 = pred.TransformPhysicalPointToIndex(p2)
            # check if location is in pred
            try:
                if pred[index1] == 1 and pred[index2] == 1:
                    length_in += np.linalg.norm(np.array(p1)-np.array(p2))
            except Exception as e:
                print(f"Error: {e}")
                # print('Index out of bounds')
                # if location is not within boundary, remove point
                length -= np.linalg.norm(np.array(p1)-np.array(p2))
            points_checked.append(p1)
        centerline_length += length
        cent_length_init += length_in
        # check if points are within pred
        # loc1 = points.GetPoint(0)
        # loc2 = points.GetPoint(num_points-1)
        # index1 = pred.TransformPhysicalPointToIndex(loc1)
        # index2 = pred.TransformPhysicalPointToIndex(loc2)
        # # check if location is in pred
        # try:
        #     if pred[index1] == 1 and pred[index2] == 1:
        #         cent_length_init += length
        # except Exception as e:
        #     print(f"Error: {e}")
        #     print('Index out of bounds')
        #     # if location is not within boundary, remove point
        #     centerline_length -= length
    # print(f"Centerline length: {centerline_length}")
    # print(f"In centerline length: {cent_length_init}")
    return cent_length_init/centerline_length


def percent_centerline_points(pred, cent_truth):

    # check if cent_truth is string or sitk image
    if isinstance(cent_truth, str):
        cent_truth = vf.read_geo(cent_truth).GetOutput()
    # number of points in centerline
    num_points = cent_truth.GetNumberOfPoints()
    # cent_data = vf.collect_arrays(cent_truth.GetPointData())
    # point locations as numpy array
    c_loc = v2n(cent_truth.GetPoints().GetData())

    num_points_init = num_points

    # check if points are within pred
    for i in range(num_points):

        location = c_loc[i]
        index = pred.TransformPhysicalPointToIndex(location.tolist())

        # check if location is in pred
        try:
            if pred[index] == 0:
                num_points_init -= 1

        except Exception as e:
            print(f"Error: {e}")
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

    print(f"Case name: {case_name}")

    if 'seg' in case_name:
        name = case_name[13:]
        name_split = case_name.split('_')
        # add until 'seg'
        name = ''
        for n in name_split:
            if 'seg' in n:
                break
            name += n
            name += '_'
        name = name[:-1]
    else:
        name = case_name[15:]

    print(f"New name: {name}")
    return name


def read_truth(case, truth_folder):

    try:
        truth = sitk.ReadImage(truth_folder+case+'.vtk')
    except:
        try:
            truth = sitk.ReadImage(truth_folder+case+'.mha')
        except:
            try:
                truth = sitk.ReadImage(truth_folder+case+'.nii.gz')
            except:
                truth = sitk.ReadImage(truth_folder+case+'.nrrd')
    # make 0 and 1
    truth = sitk.BinaryThreshold(truth, lowerThreshold=1)

    # make uint8
    if truth.GetPixelID() != sitk.sitkUInt8:
        truth = sitk.Cast(truth, sitk.sitkUInt8)

    return truth


def read_seg(pred_folder, seg):

    pred = sitk.ReadImage(pred_folder+seg)

    return pred


def from_prob_to_binary(pred):

    # convert to binary
    pred_binary = sitk.BinaryThreshold(pred, lowerThreshold=0.5,
                                       upperThreshold=1)

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
    label_sizes = [label_stats.GetPhysicalSize(l)
                   for l in labels]
    # print(f"Label sizes: {label_sizes}")
    # sort labels by size
    labels = [x for _, x in sorted(zip(label_sizes, labels), reverse=True)]
    # print(f"Sorted labels: {labels}")
    # keep only largest label
    if num_labels == 1:
        label = labels[0]
    else:
        label = labels[:num_labels]
    # print(f"Keeping label {label}")
    if num_labels == 1:
        labelImage = sitk.BinaryThreshold(ccimage, lowerThreshold=label,
                                          upperThreshold=label)
        return labelImage
    else:
        labelImage = sitk.BinaryThreshold(ccimage, lowerThreshold=label[0],
                                          upperThreshold=label[0])
        for l in label[1:]:
            labelImage += sitk.BinaryThreshold(ccimage, lowerThreshold=l,
                                               upperThreshold=l)
        return labelImage


def pre_process(pred, folder, case, write_postprocessed,
                cap=False, centerline=None, i_metric=0):

    if write_postprocessed and i_metric == 0:
        # marching cubes
        surface = vf.evaluate_surface(pred, 0.5)
        # Smooth marching cubes
        surface_smooth = vf.smooth_surface(surface, 18)
        vf.write_geo(pred_folder+folder+'/postprocessed/'+case+'.vtp', surface)
        vf.write_geo(pred_folder+folder+'/postprocessed/'+case+'_smooth.vtp',
                     surface_smooth)

    # only change to binary if prediction is probability map
    if pred.GetPixelID() != sitk.sitkUInt8:

        pred = from_prob_to_binary(pred)
    # else:
        # print("Prediction is already binary")

    if not cap:
        labelImage = keep_largest_label(pred)

    elif cap and centerline is not None:
        labelImage = cap_and_keep_largest(pred, centerline, case,
                                          pred_folder+folder+'/postprocessed/')
        if write_postprocessed and i_metric == 0:
            surface = vf.evaluate_surface(labelImage, 0.5)
            vf.write_geo(pred_folder+folder+'/postprocessed/'
                         + case + '_clipped.vtp',
                         surface)
    else:
        print("No centerline provided for clipping")

    # if write_postprocessed and i_metric == 0:
    #     sitk.WriteImage(labelImage,
    #                     pred_folder+folder+'/postprocessed/'+case+'.mha')

    # print np array of label image
    # labelImageArray = sitk.GetArrayFromImage(labelImage)
    # print(f"Label image array: {labelImageArray}")

    return labelImage


def get_case_names(folder, pred_folder):

    segs = os.listdir(pred_folder+folder)
    # only keep segmentation files and ignore hidden files
    segs = [seg for seg in segs if '.' not in seg[0]]
    # only keep files not folders
    segs = [seg for seg in segs if '.' in seg]
    # only keep files with .nii.gz or .mha extension
    segs = [seg for seg in segs if '.nii.gz' in seg or '.mha' in seg]
    # sort
    import natsort
    segs = natsort.natsorted(segs)
    # segs.sort()

    segs = [process_case_name(seg) for seg in segs]
    print(f"Segs: {segs}")

    return segs


def get_case_names_basic(folder, pred_folder):

    segs = os.listdir(pred_folder+folder)
    # only keep segmentation files and ignore hidden files
    segs = [seg for seg in segs if '.' not in seg[0]]
    # only keep files not folders
    segs = [seg for seg in segs if '.' in seg]
    # remove image extension
    segs = [seg.replace('.nii.gz', '') for seg in segs]
    # sort
    import natsort
    segs = natsort.natsorted(segs)
    # segs.sort()

    print(f"Segs: {segs}")

    return segs


def calc_metric(metric, pred, truth, mask=None, centerline=None):

    if mask is not None:
        pred = only_keep_mask(pred, mask)

    if metric == 'dice':

        score = dice(pred, truth)

    elif metric == 'hausdorff':

        # score = hausdorff(pred, truth)
        score = hausdorf_95(pred, truth)

    elif metric == 'centerline overlap':

        score = percent_centerline_length(pred, centerline)

    return score


def get_names_folders(list_folders):

    names = []

    for i, folder in enumerate(list_folders):

        if 'seqseg' in folder:

            names.append('SeqSeg')

        elif 'pred_benchmark_2d' in folder:
            
            if 'largest' in folder:
                names.append('2D Global\nnnU-Net\nLargest Connected')
            else:
                names.append('2D Global\nnnU-Net')

        elif 'pred_benchmark_3d' in folder:

            if 'largest' in folder:
                names.append('3D Global\nnnU-Net\nLargest Connected')
            else:
                names.append('3D Global\nnnU-Net')

        else:
            # keep first 6 characters
            names.append(folder[:6]+'...')

    return names


def get_metric_name(metric):
    if metric == 'dice':
        return 'Dice Score'
    elif metric == 'hausdorff':
        return 'Hausdorff Distance'
    elif metric == 'centerline overlap':
        return 'Centerline Overlap'
    else:
        return metric


def keep_largest_surface(polyData):
    # Create a connectivity filter to label the regions
    connectivity = vtk.vtkConnectivityFilter()
    connectivity.SetInputData(polyData)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.ColorRegionsOn()
    connectivity.Update()

    # Get the output of the connectivity filter
    connectedPolyData = connectivity.GetOutput()

    # Get the region labels
    regionLabels = connectedPolyData.GetPointData().GetArray('RegionId')

    # Convert region labels to numpy array
    regionLabels_np = vtk.util.numpy_support.vtk_to_numpy(regionLabels)

    # Find the unique region labels and their counts
    uniqueLabels, counts = np.unique(regionLabels_np, return_counts=True)

    # Find the label of the largest region
    largestRegionLabel = uniqueLabels[np.argmax(counts)]

    # Create a threshold filter to extract the largest region
    threshold = vtk.vtkThreshold()
    threshold.SetInputData(connectedPolyData)
    threshold.ThresholdBetween(largestRegionLabel, largestRegionLabel)
    threshold.SetInputArrayToProcess(0, 0, 0,
                                     vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
                                     'RegionId')
    threshold.Update()

    # Convert the output of the threshold filter to vtkPolyData
    largestRegionPolyData = vtk.vtkGeometryFilter()
    largestRegionPolyData.SetInputData(threshold.GetOutput())
    largestRegionPolyData.Update()

    return largestRegionPolyData.GetOutput()


def cap_and_keep_largest(pred, centerline, file_name, outdir):
    from modules.vtk_functions import convertPolyDataToImageData

    predpd = vf.evaluate_surface(pred, 0.5)
    endpts, radii, unit_vecs = bryan_get_clipping_parameters(centerline)
    boxpd, boxpdlst = bryan_generate_oriented_boxes(endpts, unit_vecs, radii,
                                                    file_name+'_boxclips',
                                                    outdir, 4)
    clippedpd = bryan_clip_surface(predpd, boxpd)
    largest = keep_largest_surface(clippedpd)
    # write clipped surface
    vf.write_geo(outdir+file_name+'_clippeddd.vtp', largest)
    # convert to sitk image
    img_vtk = vf.exportSitk2VTK(pred)[0]
    seg_vtk = convertPolyDataToImageData(largest, img_vtk)
    seg = vf.exportVTK2Sitk(seg_vtk)

    return seg


def calc_metrics_folders(pred_folders, pred_folder, truth_folder, cent_folder,
                         mask_folder, output_folder, modalities, metrics,
                         save_name, preprocess_pred, masked,
                         write_postprocessed, print_case_names,
                         keep_largest_label_benchmark, cap=False,
                         paired_ttest=True, mecnemar=False,
                         wilcoxon_bool=False,
                         process_names=False):
    """
    Calculate metrics for all folders in pred_folders

    Parameters
    ----------
    pred_folders : list
        List of folders with segmentation results
    pred_folder : string
        Path to folder with segmentation results
    truth_folder : string
        Path to folder with ground truth segmentations
    cent_folder : string
        Path to folder with centerlines
    mask_folder : string
        Path to folder with masks
    output_folder : string
        Path to folder where plots will be saved
    modalities : list
        List of modalities
    metrics : list
        List of metrics
    save_name : string
        Name of the plot
    preprocess_pred : bool
        Preprocess predictions
    masked : bool
        Mask predictions
    write_postprocessed : bool
        Write postprocessed predictions
    print_case_names : bool
        Print case names
    keep_largest_label_benchmark : bool
        Keep only largest label for benchmark
    """
    for modality in modalities:

        for i_metric, metric in enumerate(metrics):

            print(f"\nCalculating {metric} for {modality} data...")
            # keep track of scores to plot
            scores = {}

            folders_mod = [folder for folder in pred_folders
                           if modality in folder]
            
            # only keep if 'seqseg' in name
            # folders_mod = [folder for folder in folders_mod if 'seqseg' in folder]

            if process_names:
                case_names = get_case_names([fo for fo in folders_mod
                                            if 'seqseg' in fo][0], pred_folder)
            else:
                case_names = get_case_names_basic(folders_mod[0], pred_folder)
            print(f"Case names: {case_names}")
            for folder in folders_mod:

                print(f"\n{folder}:")

                scores[folder] = []

                segs = os.listdir(pred_folder+folder)
                # only keep segmentation files and ignore hidden files
                segs = [seg for seg in segs if '.' not in seg[0]]
                # only keep files not folders
                segs = [seg for seg in segs if '.' in seg]
                # only keep files with .nii.gz or .mha extension
                segs = [seg for seg in segs if '.nii.gz' in seg or '.mha' in seg]
                # sort
                import natsort
                segs = natsort.natsorted(segs)
                # segs.sort()

                if write_postprocessed:
                    os.makedirs(pred_folder+folder+'/'+'postprocessed',
                                exist_ok=True)

                for i, seg in enumerate(segs):
                    print(f" Seg: {seg}")
                    if modality == 'mr' and i == 2:
                        # skip first case for MR
                        continue
                    # if i == 0:
                    #     continue

                    if 'seqseg' in folder:
                        case = process_case_name(seg)
                    else:
                        # remove file extension
                        case = case_names[i]

                    pred = read_seg(pred_folder+folder+'/', seg)
                    truth = read_truth(case, truth_folder)

                    # check origin and make same
                    if pred.GetOrigin() != truth.GetOrigin():
                        pred.SetOrigin(truth.GetOrigin())

                        # if write_postprocessed:
                        #     sitk.WriteImage(pred, pred_folder+folder+'/postprocessed/'
                        #                     + case + '_origin.mha')
                        #     sitk.WriteImage(truth, pred_folder+folder+'/postprocessed/'
                        #                     + case + '_origin_truth.mha')
                    
                    # check transform matrix and make same
                    if pred.GetDirection() != truth.GetDirection():
                        pred.SetDirection(truth.GetDirection())

                        # if write_postprocessed:
                        #     sitk.WriteImage(pred, pred_folder+folder+'/postprocessed/'
                        #                     + case + '_direction.mha')
                        #     sitk.WriteImage(truth, pred_folder+folder+'/postprocessed/'
                        #                     + case + '_direction_truth.mha')

                    if metric == 'centerline overlap' or cap:
                        from modules import vtk_functions as vf
                        centerline = vf.read_geo(
                                cent_folder+'/' + case + '.vtp').GetOutput()
                    else:
                        centerline = None

                    if preprocess_pred and 'seqseg' in folder:
                        pred = pre_process(pred, folder, case,
                                           write_postprocessed, cap,
                                           centerline, i_metric)
                    elif preprocess_pred and keep_largest_label_benchmark:
                        pred = pre_process(pred, folder, case,
                                           write_postprocessed, cap,
                                           centerline, i_metric)

                    if write_postprocessed and i_metric == 0:
                        # marching cubes
                        from modules import vtk_functions as vf
                        # Marching cubes
                        surface = vf.evaluate_surface(pred, 0.5)
                        # Smooth marching cubes
                        # surface_smooth = vf.smooth_surface(surface, 12)
                        vf.write_geo(pred_folder+folder+'/postprocessed/'
                                     + case + '.vtp', surface)
                        # sitk.WriteImage(pred, pred_folder+folder+'/postprocessed/'
                                        # + case + '.mha')

                    if masked:  # and ('seqseg' in folder or 'combined' in folder):
                        mask = read_truth('mask_'+case, mask_folder)
                    else:
                        mask = None

                    score = calc_metric(metric, pred, truth,
                                        mask, centerline)

                    scores[folder].append(score)
                    if print_case_names:
                        print(f"{case}: {score:.3f}")
                    else:
                        print(f"{score:.3f}")
                if print_case_names:
                    print(f"Average {metric}: {np.mean(scores[folder]):.3f}")
                else:
                    print(f"{np.mean(scores[folder]):.3f}")

            # t-test to compare score from folder with 'seqseg' in name
            # to score from folder without 'seqseg' in name
            if len(scores.keys()) > 1:
                for i in range(len(scores.keys())-1):
                    for j in range(i+1, len(scores.keys())):
                        if wilcoxon_bool:
                            t, p = wilcoxon(scores[list(scores.keys())[i]],
                                            scores[list(scores.keys())[j]])
                            print(f"\n {p}: wilcoxon p-value between {list(scores.keys())[i]} and {list(scores.keys())[j]}")
                        elif mecnemar:
                            p = mcnemar_test(scores[list(scores.keys())[i]],
                                             scores[list(scores.keys())[j]])
                            print(f"\n {p}: mcnemar p-value between {list(scores.keys())[i]} and {list(scores.keys())[j]}")
                        elif not paired_ttest:
                            t, p = ttest_ind(scores[list(scores.keys())[i]],
                                             scores[list(scores.keys())[j]])
                            print(f"unpaired p-value between {list(scores.keys())[i]} and {list(scores.keys())[j]}: {p}")
                        else:
                            t, p = ttest_rel(scores[list(scores.keys())[i]],
                                             scores[list(scores.keys())[j]])
                            print(f"\n {p}: paired p-value between {list(scores.keys())[i]} and {list(scores.keys())[j]}")
                        # if p < 0.05, then print
                        if p < 0.05:
                            print(f"***DING DING*** Significant difference between {list(scores.keys())[i]} and {list(scores.keys())[j]}\n")
                        else:
                            print(f"No significant difference between {list(scores.keys())[i]} and {list(scores.keys())[j]}\n")

            # Make box plot for modality
            plt.figure(figsize=(3, 5))
            # set font to Times New Roman
            plt.rcParams["font.family"] = "Times New Roman"
            # set font size to 14
            plt.rcParams.update({'font.size': 14})

            # create boxplot with colors
            colors = ['pink', 'lightblue', 'lightgreen', 'orange',
                      'yellow', 'lightbrown', 'purple', 'brown',
                      'black', 'grey']
            boxplot = plt.boxplot(scores.values(), patch_artist=True, widths=0.3)
            for patch, color in zip(boxplot['boxes'], colors):
                patch.set_facecolor(color)
            # add legend
            # plt.legend(boxplot['boxes'], scores.keys())

            # add x axis labels
            plt.xticks(range(1, len(scores.keys())+1),
                       get_names_folders(scores.keys()))
            # plt.title(f'{modality.upper()}')
            # set y axis lower limit to 0
            # plt.style.use('ggplot')
            if 'hausdorff' in metric:
                plt.ylim(bottom=0)
                # plt.ylim(top=0.5)
            plt.ylabel(f'{get_metric_name(metric)}')
            # plt.xlabel('Method')
            if 'dice' in metric:
                plt.ylim(top=1.1+(0.05*(len(scores.keys()) - 1)))
                # plt.ylim(top=1)
                plt.ylim(bottom=0)
            if 'centerline' in metric:
                plt.ylim(top=1.1+(0.05*(len(scores.keys()) - 1)))
                plt.ylim(bottom=0)
            if 'dice' in metric or 'centerline' in metric:
                # add horizontal line at 1
                plt.axhline(y=1, color='r', linestyle='--')
                # add text at 1, 'max', on left side in red
                plt.text(0.58, 1.01, 'max', color='r')

            # add grid on y axis
            plt.grid(axis='y')
            plt.tight_layout()  # make sure labels are not cut off
            # add two horizontal lines at means
            means = [np.mean(scores[folder]) for folder in scores.keys()]
            max_height = max([max(scores[folder]) for folder in scores.keys()])
            height = get_heights(scores)
            center = [i for i in range(1, len(scores.keys())+1)]

            i_list, j_list = get_i_j_order(len(scores.keys()))
            i_list, j_list = get_i_j_order_only_0(len(scores.keys()))
            # add p-values from t-test
            if len(scores.keys()) > 1:
                # for i in i_list:  # range(len(scores.keys())-1):
                #     for j in j_list:  # range(i+1, len(scores.keys())):
                count = 0
                for i, j in zip(i_list, j_list):
                    if wilcoxon_bool:
                        t, p = wilcoxon(scores[list(scores.keys())[i]],
                                        scores[list(scores.keys())[j]])
                    elif mecnemar:
                        p = mcnemar_test(scores[list(scores.keys())[i]],
                                            scores[list(scores.keys())[j]])
                    elif not paired_ttest:
                        t, p = ttest_ind(scores[list(scores.keys())[i]],
                                            scores[list(scores.keys())[j]])
                    else:
                        t, p = ttest_rel(scores[list(scores.keys())[i]],
                                            scores[list(scores.keys())[j]])

                    barplot_annotate_brackets(i, j,
                                              f"p = {p:.3f}",
                                              center, height,
                                              dh=.01+((count)*0.05),
                                              barh=.03,
                                              fs=10)
                    count += 1

            # set ylim based on height of plot
            if metric == 'hausdorff':
                plt.ylim(top=max_height+(len(scores.keys())) * 0.05*max_height)
            # save plot
            plt.savefig(output_folder
                        + f'{save_name}_{metric}_{modality}_scores.png',
                        bbox_inches="tight")
            # plt.savefig(output_folder + f'{save_name}_{metric}_{modality}_scores.svg',bbox_inches="tight")
            plt.close()

            # write to txt file
            write_txt_file(scores, output_folder, save_name, metric, modality)

        # combine txt files
        combine_txt_files(output_folder, save_name, metrics, modality)


def combine_txt_files(output_folder, save_name, metrics, modality):
    """
    Combine txt files for each metric into one txt file

    For each line we append the corresponding line from each metric txt file

    So the final txt file will have the same number of lines as the
    metric txt files and each line will have the scores for all metrics

    Parameters
    ----------
    output_folder : string
        Path to folder where txt files are saved
    save_name : string
        Name of the txt files
    metrics : list
        List of metrics
    modality : string
        Name of the modality
    """
    for metric in metrics:
        with open(output_folder + f'{save_name}_{metric}_{modality}_scores.txt', 'r') as f:
            lines = f.readlines()
            if metric == metrics[0]:
                all_lines = lines
                # add number to start of each line
                all_lines = [f"& {i+1} &  & {line.strip()}" for i, line in enumerate(all_lines)]
            else:
                for i, line in enumerate(lines):
                    all_lines[i] = all_lines[i].strip() + ' ' + line.strip()
    # add '\\' to end of each line
    all_lines = [line + ' \\\\' for line in all_lines]

    with open(output_folder + f'{save_name}_{modality}_scores.txt', 'w') as f:
        for line in all_lines:
            f.write(line + '\n')


def write_txt_file(scores, output_folder, save_name, metric, modality):
    """
    Write scores to txt file in the following format:

    case1:  method1 score &  method2 score &  method3 score & ...
    case2:  method1 score &  method2 score &  method3 score & ...
    ...

    Parameters
    ----------
    scores : dict
        Dictionary with scores
    output_folder : string
        Path to folder where txt file will be saved
    save_name : string
        Name of the txt file
    metric : string
        Name of the metric
    modality : string
        Name of the modality
    """
    with open(output_folder + f'{save_name}_{metric}_{modality}_scores.txt', 'w') as f:
        for case in range(len(list(scores.values())[0])):
            # f.write(f"{case}: ")
            for folder in scores.keys():
                # max 3 significant digits
                f.write(f"{scores[folder][case]:.3g} & ")
            f.write("\n")


def get_heights(scores):

    heights = [max(scores[folder]) for folder in scores.keys()]
    # loop through and if previous height is higher, keep it
    for i in range(1, len(heights)):
        if heights[i] < heights[i-1]:
            heights[i] = heights[i-1]

    return heights


def get_i_j_order(n):
    """
    Get the combinations of i and j for n groups
    so that we do closest comparisons first
    and the furthest comparisons last
    first 0-1, 1-2, 2-3, 3-4 etc.
    then 0-2, 1-3, 2-4, 3-5 etc.
    then 0-3, 1-4 etc.
    then 0-4

    Example:
    n = 4
    we get
    i_list = [0, 1, 2, 0, 1, 0]
    j_list = [1, 2, 3, 2, 3, 3]

    Parameters
    ----------
    n : int
        Number of groups

    Returns
    -------
    i_list : list
        List of i values
    j_list : list
        List of j values
    """
    i_list = []
    j_list = []
    for sub in range(1, n):
        for i in range(n-sub):
            i_list.append(i)
            j_list.append(i+sub)

    print(f"i_list: {i_list}")
    print(f"j_list: {j_list}")
    return i_list, j_list


def get_i_j_order_only_0(n):
    """
    Get the combinations of i and j for n groups
    but only comparing all to the first, 0
    """
    i_list = []
    j_list = []
    for i in range(1, n):
        i_list.append(0)
        j_list.append(i)

    print(f"i_list: {i_list}")
    print(f"j_list: {j_list}")
    return i_list, j_list


def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None,
                              dh=.05, barh=.05, fs=None, maxasterix=None):
    """
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """
    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    plt.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)


def combine_segs(pred_folder,
                 modalities=['ct', 'mr'],
                 combine_names=['seqseg', 'benchmark_2d']):
    """
    Combine segmentations from different methods

    We loop through all cases and combine segmentations from different methods
    by adding them together. We then save the combined segmentation in a new
    folder called 'combined_ct' or 'combined_mr'.

    Parameters
    ----------
    pred_folder : string
        Path to folder with segmentation results
    combine_names : list
        List of names of methods to combine
    """
    # create new folder for combined segmentations
    for modality in modalities:
        os.makedirs(pred_folder+'combined_'+modality, exist_ok=True)

    folders_all = os.listdir(pred_folder)
    # only keep folders and ignore hidden files
    folders_all = [folder for folder in folders_all if '.' not in folder]
    folders_all.sort()

    for modality in modalities:

        folders = [folder for folder in folders_all if modality in folder]
        folders = [folder for folder in folders if 'combined' not in folder]
        # only keep folders with names in combine_names
        folders = [folder for folder in folders if any(name in folder
                                                       for name in combine_names)]
        print(f"Folders: {folders}")

        # get case names
        case_names = get_case_names([fo for fo in folders
                                    if 'seqseg' in fo][0], pred_folder)

        case_names_first = os.listdir(pred_folder+folders[0]+'/')
        case_names_first = [case for case in case_names_first if '.' in case]
        case_names_first = [case for case in case_names_first if case[0] != '.']
        case_names_first.sort()

        for i, case in enumerate(case_names):
            # get the first segmentation
            print(f"Combining segmentations for {case}")
            print(f"Reading in pred {case_names_first[i]} from {folders[0]}")
            pred = read_seg(pred_folder+folders[0]+'/', case_names_first[i])
            # add all other segmentations
            for folder in folders[1:]:
                case_names_folder = os.listdir(pred_folder+folder+'/')
                case_names_folder = [case for case in case_names_folder if '.' in case]
                case_names_folder = [case for case in case_names_folder if case[0] != '.']
                case_names_folder.sort()
                print(f"Reading in pred {case_names_folder[i]} from {folder}")
                pred = add_seg(pred, read_seg(pred_folder+folder+'/', case_names_folder[i]))

            pred = average_seg(pred, len(folders))
            # save combined segmentation
            sitk.WriteImage(pred, pred_folder+'combined_'+modality+'/'+case+'.mha')


def add_seg(pred1_img, pred2_img):

    pred1 = sitk.GetArrayFromImage(pred1_img)
    pred2 = sitk.GetArrayFromImage(pred2_img)
    pred = np.add(pred1, pred2)

    pred = sitk.GetImageFromArray(pred)
    pred.CopyInformation(pred1_img)

    return pred


def average_seg(pred_img, num_segs):

    print(f"Number of segmentations: {num_segs}")
    pred = sitk.GetArrayFromImage(pred_img)
    pred = pred / num_segs

    # threshold to 0.5
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    pred = sitk.GetImageFromArray(pred)
    pred.CopyInformation(pred_img)

    return pred


if __name__ == '__main__':

    name_graph = 'Comparison'
    # save_name = 'test_keep_noclip_unpaired_ttest_skipfirst'
    save_name = 'test_seqseg_cap_keep_largest'
    preprocess_pred = True  # if cap or keep largest label
    masked = False
    write_postprocessed = True

    print_case_names = True
    process_names = True  # if we want to process case names, like seqseg paper

    cap = True
    keep_largest_label_benchmark = True
    paired_ttest = False
    mecnemar = False
    wilcoxon_bool = False

    # input folder of segmentation results
    pred_folder = '/Users/numisveins/Documents/data_seqseg_paper/pred_mic_aortas_june24/results/'
    truth_folder = '/Users/numisveins/Documents/MICCAI_Challenge23_Aorta_Tree_Data/truths/'
    cent_folder = '/Users/numisveins/Documents/MICCAI_Challenge23_Aorta_Tree_Data/centerlines/'
    mask_folder = '/Users/numisveins/Documents/MICCAI_Challenge23_Aorta_Tree_Data/global_masks/'
    output_folder = '/Users/numisveins/Documents/data_seqseg_paper/pred_mic_aortas_june24/graphs/'

    pred_folder = '//Users/numisveins/Documents/data_papers/data_seqseg_paper/dataset_size_study/'
    truth_folder = '/Users/numisveins/Documents/datasets/vmr/truths/'
    cent_folder = '/Users/numisveins/Documents/datasets/vmr/centerlines/'
    # mask_folder = '/Users/numisveins/Documents/vascular_data_3d/masks_around_truth/masks_4r/'
    output_folder = '//Users/numisveins/Documents/data_papers/data_seqseg_paper/dataset_size_study/out/'

    # Gala data
    pred_folder = '/Users/numisveins/Documents/data_papers/data_gala_aaas/'
    truth_folder = '/Users/numisveins/Documents/datasets/vmr/truths/'
    cent_folder = '/Users/numisveins/Documents/datasets/vmr/centerlines/'
    output_folder = '/Users/numisveins/Documents/data_papers/data_gala_aaas/out/'


    # # vascular data
    # pred_folder = '/Users/numisveins/Documents/data_combo_paper/ct_data/vascular_segs/vascular_segs_mha/pred_seqseg_ct/new_format/'
    # output_folder = '/Users/numisveins/Documents/data_combo_paper/ct_data/graphs/'

    # # cardiac data
    # pred_folder = '/Users/numisveins/Documents/data_combo_paper/ct_data/meshes/'
    # truth_folder = '/Users/numisveins/Documents/data_combo_paper/outct_data/Ground truth cardiac segmentations/'

    # get all folders in pred_folder
    pred_folders = os.listdir(pred_folder)
    # only keep folders and ignore hidden files
    pred_folders = [folder for folder in pred_folders if '.' not in folder and 'old' not in folder and 'gt' not in folder]
    pred_folders.sort()
    print(f"Prediction folders: {pred_folders}")
    # pred_folders = [folder for folder in pred_folders if '3d' not in folder]

    # modalities
    modalities = ['ct']  # , 'mr']
    # metrics
    metrics = ['centerline overlap', 'dice', 'hausdorff', ]

    calc_metrics_folders(pred_folders, pred_folder, truth_folder, cent_folder,
                         mask_folder, output_folder, modalities, metrics,
                         save_name, preprocess_pred, masked,
                         write_postprocessed, print_case_names,
                         keep_largest_label_benchmark, cap, paired_ttest,
                         mecnemar, wilcoxon_bool, process_names)

    import pdb; pdb.set_trace()
    # combine segmentations
    combine_segs(pred_folder, modalities)