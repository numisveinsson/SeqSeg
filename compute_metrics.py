import os
import SimpleITK as sitk
import numpy as np

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

    haus_filter = sitk.HausdorffDistanceImageFilter()
    haus_filter.Execute(pred, truth)

    return haus_filter.GetHausdorffDistance()

def only_keep_mask(pred, mask):
    
    filter = sitk.GetArrayFromImage(mask)
    pred = sitk.GetArrayFromImage(pred)
    pred[filter == 0] = 0

    return sitk.GetImageFromArray(masked_pred)

def process_case_name(case_name):
    'Change this function if naming convention changes'
    return case_name[15:24]

def read_truth(case, truth_folder):

    try:
        truth = sitk.ReadImage(truth_folder+case+'.vtk')
    except:
        truth = sitk.ReadImage(truth_folder+case+'.mha')

    return truth

def read_seg(pred_folder, seg):

    pred = sitk.ReadImage(pred_folder+seg)

    return pred

def pre_process(pred):
    
    pred_binary = sitk.BinaryThreshold(pred, lowerThreshold=0.5, upperThreshold=1)

    return pred_binary

def calc_metric(metric, pred, truth, mask):

    if metric == 'dice':

        score = dice(pred, truth)

    elif metric == 'hausdorff':

        score = hausdorff(pred, truth)

    elif metric == 'dice mask':

        masked_pred = only_keep_mask(pred, mask)
        score = dice(masked_pred, truth)

    elif metric == 'hausdorff mask':

        masked_pred = only_keep_mask(pred, mask)
        score = hausdorff(masked_pred, truth)
    
    return score

if __name__=='__main__':

    preprocess_pred = True

    #input folder of segmentation results
    pred_folder = '/Users/numisveins/Downloads/preds/'
    truth_folder = '/Users/numisveins/Downloads/truths/'
    mask_folder = ''

    # output folder for plots
    output_folder = '/Users/numisveins/Downloads/'

    #metrics
    metrics = ['hausdorff', 'dice']#, 'dice mask', 'hausdorff mask']

    for metric in metrics:

        print(f"\nCalculating {metric}...")
        scores = []

        for seg in os.listdir(pred_folder):
            
            case = process_case_name(seg)

            pred = read_seg(pred_folder, seg)
            truth = read_truth(case, truth_folder)

            if preprocess_pred:
                pred = pre_process(pred)

            if 'mask' in metric:
                mask = read_truth(case, mask_folder)
            else: mask = None

            score = calc_metric(metric, pred, truth, mask)

            scores.append(score)

            print(f"{case}: {score}")

        print(f"Average {metric}: {np.mean(scores)}")

        # Make box plot
        import matplotlib.pyplot as plt
        plt.figure()
        plt.boxplot(scores)
        plt.title(f'{metric} scores')
        # set y axis lower limit to 0
        plt.ylim(bottom=0)
        plt.ylabel(f'{metric} score')
        if metric.includes('dice'):
            plt.ylim(top=1)
        plt.savefig(output_folder + f'{metric}_scores.png')
        plt.close()

