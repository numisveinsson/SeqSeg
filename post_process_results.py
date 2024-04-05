from modules import vtk_functions as vf
import SimpleITK as sitk
import os
from compute_metrics import keep_largest_label, from_prob_to_binary

def post_process(pred, num_regions=1):
    # only change to binary if prediction is probability map
    if pred.GetPixelID() != sitk.sitkUInt8:

        pred  = from_prob_to_binary(pred)
    #else:
        #print("Prediction is already binary")

    labelImage = keep_largest_label(pred, num_regions)

    return labelImage


if __name__=='__main__':

    folder = '/Users/numisveins/Documents/ASOCA_dataset/Results_Predictions/output_2d_coroasocact/'
    ext = '.nii.gz'

    num_regions = 2

    if not os.path.exists(folder+'/postprocessed_largest_label'):
        os.mkdir(folder+'/postprocessed_largest_label')
    else:
        print("Folder already exists")

    pred_cases = os.listdir(folder)
    pred_cases = [case for case in pred_cases if case.endswith(ext)]

    for pred in pred_cases:

        img = sitk.ReadImage(folder+pred)

        write_postprocessed = folder+'/postprocessed/'+pred
        img = post_process(img, num_regions)
        sitk.WriteImage(img, write_postprocessed)

        surface = vf.evaluate_surface(img, 0.5) # Marching cubes
        # surface_smooth = vf.smooth_surface(surface, 12) # Smooth marching cubes
        vf.write_geo(folder+'/postprocessed/'+pred.replace(ext, '.vtp'), surface)

