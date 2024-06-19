import os
import SimpleITK as sitk
import vtk
import numpy as np
from modules import vtk_functions as vf
from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

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

    # change to one hot
    dice_out = [None]*len(num_class)
    for i in range(1, len(num_class)):
        pred_c = pred == num_class[i]
        true_c = true == num_class[i]
        dice_out[i] = np.sum(pred_c*true_c)*2.0 / (np.sum(pred_c) + np.sum(true_c))

    mask = (pred > 0) + (true > 0)
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
            except Exception as e:
                print(f"Error: {e}")
                # print('Index out of bounds')
                # if location is not within boundary, remove point
                centerline_length -= length

    return cent_length_init/centerline_length


def percent_centerline_points(pred, cent_truth):

    # check if cent_truth is string or sitk image
    if isinstance(cent_truth, str):
        cent_truth = vf.read_geo(cent_truth).GetOutput()

    num_points = cent_truth.GetNumberOfPoints()   # number of points in centerline
    # cent_data = vf.collect_arrays(cent_truth.GetPointData())
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


def pre_process(pred, folder, case, write_postprocessed, cap=False, centerline=None):

    if write_postprocessed:
        # marching cubes
        surface = vf.evaluate_surface(pred, 0.5) # Marching cubes
        # surface_smooth = vf.smooth_surface(surface, 12) # Smooth marching cubes
        vf.write_geo(pred_folder+folder+'/postprocessed/'+case+'.vtp', surface)

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
    else:
        print("No centerline provided for clipping")

    if write_postprocessed:
        sitk.WriteImage(labelImage, pred_folder+folder+'/postprocessed/'+case+'.mha')
        surface = vf.evaluate_surface(labelImage, 0.5)
        vf.write_geo(pred_folder+folder+'/postprocessed/'+case+'_clipped.vtp', surface)

    # print np array of label image
    # labelImageArray = sitk.GetArrayFromImage(labelImage)
    # print(f"Label image array: {labelImageArray}")

    return labelImage


def get_case_names(folder, pred_folder):

    segs = os.listdir(pred_folder+folder)
    print(f"Segs: {segs}")
    # only keep segmentation files and ignore hidden files
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

    for i, folder in enumerate(list_folders):

        if 'pred_seqseg' in folder: 

            names.append('SeqSeg')

        elif 'pred_benchmark_2d' in folder:

            names.append('2D Global\nnnU-Net')

        elif 'pred_benchmark_3d' in folder:

            names.append('3D Global\nnnU-Net')

        else:
            # keep first 6 characters
            names.append(folder[:6]+'...')

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


def bryan_get_clipping_parameters(clpd):
    """ get all three parameters """
    points = v2n(clpd.GetPoints().GetData())
    CenterlineID_for_each_point = v2n(clpd.GetPointData().GetArray('CenterlineId'))
    radii = v2n(clpd.GetPointData().GetArray('MaximumInscribedSphereRadius'))
    n_keys = len(CenterlineID_for_each_point[0])
    line_dict = {}

    # create dict with keys line0, line1, line2, etc
    for i in range(n_keys):
        key = f"line{i}"  
        line_dict[key] = []

    for i in range(len(points)):
        for j in range(n_keys):
            if CenterlineID_for_each_point[i][j] == 1:
                key = f"line{j}"
                line_dict[key].append(points[i])

    for i in range(n_keys):
        key = f"line{i}"  
        line_dict[key] = np.array(line_dict[key])
    # Done with spliting centerliens into dictioanry

    # find the end points of each line
    lst_of_end_pts = []
    # append the very first point
    lst_of_end_pts.append(line_dict["line0"][0])
    # append the rest of the end points
    for i in range(n_keys):
        key = f"line{i}"
        lst_of_end_pts.append(line_dict[key][-1])
    nplst_of_endpts = np.array(lst_of_end_pts)  # convert to numpy array

    # find the radii at the end points
    radii_at_caps = []
    for i in lst_of_end_pts:
        for j in range(len(points)):
            if np.array_equal(i, points[j]):
                radii_at_caps.append(radii[j])
    nplst_radii_at_caps = np.array(radii_at_caps)  # convert to numpy array

    # find the unit tangent vectors at the end points
    unit_tangent_vectors = []
    # compute the unit tangent vector of the first point of the first line
    key = "line0"
    line = line_dict[key]
    tangent_vector = line[0] - line[1]
    unit_tangent_vector = tangent_vector / np.linalg.norm(tangent_vector)
    unit_tangent_vectors.append(unit_tangent_vector)
    # compute the unit tangent vector of the last point of each line
    for i in range(len(line_dict)):
        key = f"line{i}"
        line = line_dict[key]
        tangent_vector = line[-1] - line[-2]
        unit_tangent_vector = tangent_vector / np.linalg.norm(tangent_vector)
        unit_tangent_vectors.append(unit_tangent_vector)

    return nplst_of_endpts, nplst_radii_at_caps, unit_tangent_vectors


def bryan_generate_oriented_boxes(endpts, unit_tan_vectors, radius,
                                  output_file, outdir, box_scale=3):

    box_surfaces = vtk.vtkAppendPolyData()
    # Convert the input center_points to a list, in case it is a NumPy array
    endpts = np.array(endpts).tolist()
    centerpts = []
    pd_lst = []
    for i in range(len(endpts)):
        compute_x = endpts[i][0]+0.5*box_scale*radius[i]*unit_tan_vectors[i][0]
        compute_y = endpts[i][1]+0.5*box_scale*radius[i]*unit_tan_vectors[i][1]
        compute_z = endpts[i][2]+0.5*box_scale*radius[i]*unit_tan_vectors[i][2] 
        centerpts.append([compute_x, compute_y, compute_z])

    box_surfaces = vtk.vtkAppendPolyData()

    for i in range(len(centerpts)):
        # Create an initial vtkCubeSource for the box
        box = vtk.vtkCubeSource()
        box.SetXLength(box_scale*radius[i])
        box.SetYLength(box_scale*radius[i])
        box.SetZLength(box_scale*radius[i])
        box.Update()

        # Compute the rotation axis by taking the cross product of the unit_vector and the z-axis
        rotation_axis = np.cross(np.array([0, 0, 1]), unit_tan_vectors[i])

        # Compute the rotation angle in degrees between the unit_vector and the z-axis
        rotation_angle = np.degrees(np.arccos(np.dot(unit_tan_vectors[i], np.array([0, 0, 1]))))
        transform = vtk.vtkTransform()
        transform.Translate(centerpts[i])
        transform.RotateWXYZ(rotation_angle, rotation_axis)

        # Apply the transform to the box
        box_transform = vtk.vtkTransformPolyDataFilter()
        box_transform.SetInputConnection(box.GetOutputPort())
        box_transform.SetTransform(transform)
        box_transform.Update()

        pd_lst.append(box_transform.GetOutput())
        box_surfaces.AddInputData(box_transform.GetOutput())

    box_surfaces.Update()

    # Write the oriented box to a .vtp file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(outdir+output_file+'.vtp')
    writer.SetInputData(box_surfaces.GetOutput())
    writer.Write()
    return box_surfaces.GetOutput(), pd_lst


def bryan_clip_surface(surf1, surf2):
    # Create an implicit function from surf2
    implicit_function = vtk.vtkImplicitPolyDataDistance()
    implicit_function.SetInput(surf2)

    # Create a vtkClipPolyData filter and set the input and implicit function
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(surf1)
    clipper.SetClipFunction(implicit_function)
    clipper.InsideOutOff()  # keep the part of surf1 outside of surf2
    clipper.Update()

    # Get the output polyData with the part enclosed by surf2 clipped away
    clipped_surf1 = clipper.GetOutput()

    return clipped_surf1


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
    threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, 'RegionId')
    threshold.Update()

    # Convert the output of the threshold filter to vtkPolyData
    largestRegionPolyData = vtk.vtkGeometryFilter()
    largestRegionPolyData.SetInputData(threshold.GetOutput())
    largestRegionPolyData.Update()

    return largestRegionPolyData.GetOutput()


def convertPolyDataToImageData(poly, ref_im):
    """
    Convert the vtk polydata to imagedata
    Args:
        poly: vtkPolyData
        ref_im: reference vtkImage to match the polydata with
    Returns:
        output: resulted vtkImageData
    """

    ref_im.GetPointData().SetScalars(n2v(np.zeros(v2n(ref_im.GetPointData().GetScalars()).shape)))
    ply2im = vtk.vtkPolyDataToImageStencil()
    ply2im.SetTolerance(0.05)
    ply2im.SetInputData(poly)
    ply2im.SetOutputSpacing(ref_im.GetSpacing())
    ply2im.SetInformationInput(ref_im)
    ply2im.Update()

    stencil = vtk.vtkImageStencil()
    stencil.SetInputData(ref_im)
    stencil.ReverseStencilOn()
    stencil.SetStencilData(ply2im.GetOutput())
    stencil.Update()
    output = stencil.GetOutput()

    return output


def cap_and_keep_largest(pred, centerline, file_name, outdir):

    predpd = vf.evaluate_surface(pred, 0.5)
    endpts, radii, unit_vecs = bryan_get_clipping_parameters(centerline)
    boxpd, boxpdlst = bryan_generate_oriented_boxes(endpts, unit_vecs, radii,
                                                    file_name+'_boxclips',
                                                    outdir, 4)
    clippedpd = bryan_clip_surface(predpd, boxpd)
    largest = keep_largest_surface(clippedpd)
    # convert to sitk image
    img_vtk = vf.exportSitk2VTK(pred)[0]
    seg_vtk = convertPolyDataToImageData(largest, img_vtk)
    seg = vf.exportVTK2Sitk(seg_vtk)

    return seg


def calc_metrics_folders(pred_folders, pred_folder, truth_folder, cent_folder,
                         mask_folder, output_folder, modalities, metrics,
                         save_name, preprocess_pred, masked,
                         write_postprocessed, print_case_names,
                         keep_largest_label_benchmark, cap=False):
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

        for metric in metrics:

            print(f"\nCalculating {metric} for {modality} data...")
            # keep track of scores to plot
            scores = {}

            folders_mod = [folder for folder in pred_folders
                           if modality in folder]
            case_names = get_case_names([fo for fo in folders_mod
                                         if 'seqseg' in fo][0], pred_folder)
            print(f"Case names: {case_names}")
            for folder in folders_mod:

                print(f"\n{folder}:")

                scores[folder] = []

                segs = os.listdir(pred_folder+folder)
                # only keep segmentation files and ignore hidden files
                segs = [seg for seg in segs if '.' not in seg[0]]
                # only keep files not folders
                segs = [seg for seg in segs if '.' in seg]
                # sort
                segs.sort()

                if write_postprocessed:
                    os.makedirs(pred_folder+folder+'/'+'postprocessed',
                                exist_ok=True)

                for i, seg in enumerate(segs):

                    # if modality == 'mr' and i == 2:
                    #     # skip first case for MR
                    #     continue

                    if 'seqseg' in folder:
                        case = process_case_name(seg)
                    else:
                        # remove file extension
                        case = case_names[i]

                    pred = read_seg(pred_folder+folder+'/', seg)
                    truth = read_truth(case, truth_folder)

                    if metric == 'centerline overlap' or cap:
                        from modules import vtk_functions as vf
                        centerline = vf.read_geo(cent_folder+'/' + case + '.vtp').GetOutput()
                    else:
                        centerline = None

                    if preprocess_pred and 'seqseg' in folder:
                        pred = pre_process(pred, folder, case,
                                           write_postprocessed, cap,
                                           centerline)
                    elif preprocess_pred and keep_largest_label_benchmark:
                        pred = pre_process(pred, folder, case,
                                           write_postprocessed, cap,
                                           centerline)

                    if write_postprocessed:
                        # marching cubes
                        from modules import vtk_functions as vf
                        # Marching cubes
                        surface = vf.evaluate_surface(pred, 0.5)
                        # Smooth marching cubes
                        # surface_smooth = vf.smooth_surface(surface, 12)
                        vf.write_geo(pred_folder+folder+'/postprocessed/'
                                     + case + '.vtp', surface)

                    if masked and ('seqseg' in folder or 'combined' in folder):
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
                        t, p = ttest_ind(scores[list(scores.keys())[i]],
                                         scores[list(scores.keys())[j]])
                        # print(f"unpaired p-value between {list(scores.keys())[i]} and {list(scores.keys())[j]}: {p}")
                        t, p = ttest_rel(scores[list(scores.keys())[i]],
                                         scores[list(scores.keys())[j]])
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
            plt.rcParams.update({'font.size': 10})

            # create boxplot with colors
            colors = ['pink', 'lightblue', 'lightgreen', 'orange', 'purple',
                      'yellow', 'brown', 'black', 'grey']
            boxplot = plt.boxplot(scores.values(), patch_artist=True)
            for patch, color in zip(boxplot['boxes'], colors):
                patch.set_facecolor(color)
            # add legend
            # plt.legend(boxplot['boxes'], scores.keys())

            # add x axis labels
            plt.xticks(range(1, len(scores.keys())+1),
                       get_names_folders(scores.keys()))
            # plt.title(f'{modality.upper()}')
            # set y axis lower limit to 0
            if 'hausdorff' in metric:
                plt.ylim(bottom=0)
                # plt.ylim(top=0.5)
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
                        t, p = ttest_rel(scores[list(scores.keys())[i]],
                                         scores[list(scores.keys())[j]])
                        if p < 0.05:
                            # add horizontal line that connects means
                            # of two groups with significant difference
                            plt.plot([i+1, j+1], [means[i], means[j]], 'k-')
                            # add p-value
                            plt.text((i+1+j+1)/2, max(means[i], means[j])+0.05,
                                     f'p={p:.3f}', ha='center')
                        elif p < 0.1:
                            # add horizontal dashed line that connects means
                            # of two groups with significant difference
                            plt.plot([i+1, j+1], [means[i], means[j]], 'k--')
                            # add p-value
                            plt.text((i+1+j+1)/2, max(means[i], means[j])+0.05,
                                     f'p={p:.3f}', ha='center')

            # add horizontal lines at means with same color as boxplot
            # for mean, color in zip(means, colors):
            #     plt.hlines(mean, 0.5, len(scores.keys())+0.5, colors=color, linestyles='dashed', label='mean', linewidth=1)

            # plt.hlines(means, 0.5, len(scores.keys())+0.5, colors='r', linestyles='dashed', label='mean', linewidth=1)
            # save plot
            plt.savefig(output_folder
                        + f'{save_name}_{metric}_{modality}_scores.png',
                        bbox_inches="tight")
            # plt.savefig(output_folder + f'{save_name}_{metric}_{modality}_scores.svg',bbox_inches="tight")
            plt.close()


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
                # import pdb; pdb.set_trace()
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

if __name__=='__main__':

    name_graph = 'Comparison'
    save_name = 'test_keep_clip'
    preprocess_pred = True
    masked = True
    write_postprocessed = True

    print_case_names = True

    cap = True
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
    output_folder = '/Users/numisveins/Documents/data_seqseg_paper/fresh_graphs/'

    # modalities
    modalities = ['mr', 'ct']
    # metrics
    metrics = ['dice', 'hausdorff', 'centerline overlap']

    calc_metrics_folders(pred_folders, pred_folder, truth_folder, cent_folder,
                         mask_folder, output_folder, modalities, metrics,
                         save_name, preprocess_pred, masked,
                         write_postprocessed, print_case_names,
                         keep_largest_label_benchmark, cap)

    import pdb; pdb.set_trace()
    # combine segmentations
    combine_segs(pred_folder, modalities)