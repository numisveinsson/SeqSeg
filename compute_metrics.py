import os

def dice(pred, truth):

    if not isinstance(pred, np.ndarray):
        pred = sitk.GetArrayFromImage(pred)
        truth = sitk.GetArrayFromImage(truth)
    
    pred = pred.astype(np.int)
    true = truth.astype(np.int)
    num_class = np.unique(true)

    #change to one hot
    dice_out = [None]*len(num_class)
    for i in range(1, len(num_class)):
        pred_c = pred == num_class[i]
        true_c = true == num_class[i]
        dice_out[i] = np.sum(pred_c*true_c)*2.0 / (np.sum(pred_c) + np.sum(true_c))

    mask =( pred > 0 )+ (true > 0)
    dice_out[0] = np.sum((pred==true)[mask]) * 2. / (np.sum(pred>0) + np.sum(true>0))
    return dice_out

def masked_dice(self, masked_dir):
        mask = sitk.ReadImage(masked_dir)
        filter = sitk.GetArrayFromImage(mask)
        pred = sitk.GetArrayFromImage(self.seg_pred)
        pred[filter == 0] = 0

        dice = dice_score(pred, self.seg_truth)[0]

        return dice

def hausdorff(pred, truth):

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
    return case_name[4:12]

def read_in(case, pred_folder, truth_folder):
    import SimpleITK as sitk

    pred = sitk.ReadImage(pred_folder+case+'.vtk')
    truth = sitk.ReadImage(truth_folder+case+'.vtk')
    
    return pred, truth

def pre_process(pred):
    
    pred_binary = sitk.BinaryThreshold(pred, lowerThreshold=0.5, upperThreshold=1)

    return pred_binary

def calc_metric(metric, pred, truth, mask):

    if metric = 'dice':

        score = dice(pred, truth)

    elif metric = 'hausdorff':

        score = hausdorff(pred, truth)

    elif metric = 'dice mask':

        masked_pred = only_keep_mask(pred, mask)
        score = dice(masked_pred, truth)

    elif metric = 'hausdorff mask':

        masked_pred = only_keep_mask(pred, mask)
        score = hausdorff(masked_pred, truth)
    
    return score

if __name__=='__main__':

    preprocess_pred = True

    #input folder of segmentation results
    pred_folder = ''
    truth_folder = ''
    mask_folder = ''

    # output folder for plots
    output_folder = ''

    #metrics
    metrics = ['dice', 'hausdorff']#, 'dice mask', 'hausdorff mask']

    for metric in metrics:

        scores = []

        for seg in os.listdir(pred_folder):
            
            case = process_case_name(seg)

            pred, truth = read_in(case, pred_folder, truth_folder)

            if preprocess_pred:
                pred = pre_process(pred)

            if 'mask' in metric:
                mask = read_in(case, mask_folder, truth_folder)
            else: mask = None

            score = calc_metric(metric, pred, truth, mask)