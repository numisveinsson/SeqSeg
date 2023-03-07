import numpy as np
import pdb
import os
import pandas
from ast import literal_eval

from modules import vtk_functions as vf
from modules import sitk_functions as sf
from modules.assembly import Segmentation
from modules.vmr_data import vmr_directories
from modules.evaluation import sensitivity_specificity
from prediction import Prediction, dice_score
from model import UNet3DIsensee

import SimpleITK as sitk

from datetime import datetime
now = datetime.now()
dt_string = now.strftime("_%d_%m_%Y_%H_%M_%S")

def create_directories(output_folder):
    try:
        os.mkdir(output_folder)
    except Exception as e: print(e)
    try:
        os.mkdir(output_folder+'volumes')
    except Exception as e: print(e)
    try:
        os.mkdir(output_folder+'predictions')
    except Exception as e: print(e)
    # try:
    #     os.mkdir(output_folder+'centerlines')
    # except Exception as e: print(e)
    try:
        os.mkdir(output_folder+'truth')
    except Exception as e: print(e)

def list_samples_cases(directory):

    samples = os.listdir(directory)
    samples = [f for f in samples if ('mask' not in f and 'vtk' not in f)]

    cases = []
    samples_cases = {}
    for f in samples:
        case_name = f[:9]
        if case_name not in cases:
            cases.append(case_name)
            samples_cases[str(case_name)] = []
        samples_cases[str(case_name)].append(f)
    #samples = [directory + f for f in samples]

    return samples_cases, cases

def predict(output_folder, data_folder, model_folder, dir_data_3d, modality, img_shape, threshold, weighted, seg_file=False, write_samples=True):

    #if seg_file:
    #    reader_seg, origin_im, size_im, spacing_im = sf.import_image(seg_file)
    #reader_im, origin_im, size_im, spacing_im = sf.import_image(image_file)
    #assembly_segs = Segmentation(case, image_file)

    exponent = 3

    csv_file = modality+"_test_Sample_stats.csv"
    csv_list = pandas.read_csv(data_folder+csv_file)
    keep_values = ['NAME','INDEX', 'SIZE_EXTRACT', 'BIFURCATION', 'RADIUS']

    data_folder = data_folder+modality+'_test/'
    samples_cases, cases = list_samples_cases(data_folder)

    model = UNet3DIsensee((img_shape[0], img_shape[1], img_shape[2], 1), num_class=1)
    unet = model.build()
    model_name = os.path.realpath(model_folder) + '/weights_unet.hdf5'
    unet.load_weights(model_name)

    predictions = []
    final_dice_scores = {}

    for case in cases:
        print('\nNext case: ', case)
        samples = samples_cases[case]
        N_tot = len(samples)
        #print('\nNumber of samples: ', N_tot)
        dice_list = []

        samples_names = [f.replace('.nii.gz','') for f in samples]
        csv_array = csv_list.loc[csv_list['NAME'].isin(samples_names)]
        csv_list_small = csv_array[keep_values]

        dir_image, dir_seg, dir_cent, dir_surf = vmr_directories(dir_data_3d, case)
        assembly_segs = Segmentation(case, dir_image, weighted)

        for sample in samples:#[0:N_tot:20]:

            if len(samples) > 10:
                if sample in samples[1:N_tot:N_tot//10]: print('*', end=" ", flush=True)
            # Read in
            volume = sitk.ReadImage(data_folder+sample)
            vol_np = sitk.GetArrayFromImage(volume)
            print(f"Min: {vol_np.min():.2f}, Max: {vol_np.max():.2f}")
            seg_volume = sitk.ReadImage(data_folder.replace('_test', '_test_masks') +sample)

            sample = sample.replace('.nii.gz','.vtk')

            true_fn = output_folder +'truth/'+sample
            sitk.WriteImage(seg_volume, true_fn)
            img_fn = output_folder +'volumes/'+sample
            sitk.WriteImage(volume, img_fn)

            # Prediction
            predict = Prediction(unet, model_name, modality, volume, img_shape, output_folder+'predictions', threshold, seg_volume)
            predict.volume_prediction(1)
            #pred64 = predict.volume_prediction(1)
            predict.resample_prediction()

            if seg_file:
                d = predict.dice()
                #print(sample.replace('.nii.gz', '')+" , dice score: " +str(d))
                dice_list.append(d)

            stat_pd = csv_list_small.loc[csv_list_small['NAME'] == sample.replace('.vtk','')]
            index = literal_eval(stat_pd['INDEX'].values[0])
            size_extract = literal_eval(stat_pd['SIZE_EXTRACT'].values[0])
            radius = stat_pd['RADIUS'].values[0]
            #print('\n radius: ', radius)
            weight_size = (1/radius)**exponent
            #print('weight :', weight_size)

            assembly_segs.add_segmentation(predict.prob_prediction, index, size_extract, weight_size)


            # pd_fn64 = output_folder +'predictions/64_'+sample
            # pred_raw = sitk.GetImageFromArray(pred64.transpose(2,1,0))
            # pred_raw.SetOrigin(volume.GetOrigin())
            # pred_raw.SetDirection(volume.GetDirection())
            # new_space = ((np.array(volume.GetSize())-1) * np.array(volume.GetSpacing())) / np.array(img_shape)
            # pred_raw.SetSpacing(new_space.tolist())
            # sitk.WriteImage(pred_raw, pd_fn64)
            #print('Max value is: ', predict.prediction.max())
            #print('Min value is: ', predict.prediction.min())
            # if predict.prediction.min() < 0:
            #     predict.prediction[predict.prediction < 0 ] = 0
            #     print('New min is: ', predict.prediction.min())
            
            # seg_np = sitk.GetArrayFromImage(predict.seg_vol)

            # if seg_np.max()>1:
            #     seg_np = seg_np/seg_np.max()

            pd_fn = output_folder +'predictions/'+sample
            predict.write_prediction(pd_fn)

            predictions.append(predict.prediction)

        print('ALL done for case: ', case)
        print('Average dice: ', np.array(dice_list).mean())
        sitk.WriteImage(assembly_segs.assembly, output_folder +'assembly_'+case+'.vtk')

        if assembly_segs.weighted:
            sitk_n_upd = sf.numpy_to_sitk(assembly_segs.n_updates, assembly_segs.image_reader)
        else:
            sitk_n_upd = sf.numpy_to_sitk(assembly_segs.number_updates)
        sitk.WriteImage(sitk_n_upd, output_folder +'assembly_N_updates_'+case+'.vtk')

        global_seg_truth = sitk.ReadImage(dir_seg)
        global_seg_truth_np = sitk.GetArrayFromImage(global_seg_truth)
        global_seg_truth_np = global_seg_truth_np/global_seg_truth_np.max()

        for i in range(1,20):
            threshold = i*0.05
            assembly = sitk.BinaryThreshold(assembly_segs.assembly, lowerThreshold=threshold, upperThreshold=1)
            assembly_np = sitk.GetArrayFromImage(assembly)
            final_dice = dice_score(assembly_np, global_seg_truth_np)[0]
            final_dice_scores[case] = final_dice
            #print('\nDice for threshold '+str(threshold)[:4]+': ', final_dice)
            sens, spec = sensitivity_specificity(assembly_np, global_seg_truth_np)
            print(f"Sensitivity, Specificity, DICE for threshold {threshold:.2f}: {sens:.3f}, {spec:.3f}, {final_dice:.3f}")
            #print(f"Specificity for threshold {threshold}: {spec}")
        surface_assembly = vf.evaluate_surface(assembly, 1)
        vf.write_vtk_polydata(surface_assembly, output_folder +'assembly_surface_'+case+'.vtp')


    return final_dice_scores

if __name__=='__main__':

    test = 'test100'
    print('\nTest is: ', test)

    data_folder = '/Users/numisveins/Documents/Automatic_Tracing_Data/train_global_aortas/'
    dir_model_weights = '/Users/numisveins/Documents/Automatic_Tracing_ML/weights/' + test + '/'
    write_samples = True
    dir_seg = True
    weighted_assembly = False

    dir_data_3d = '/Users/numisveins/Library/Mobile Documents/com~apple~CloudDocs/Documents/Side_SV_projects/SV_ML_Training/vascular_data_3d/'
    dir_output = '/Users/numisveins/Documents/Automatic_Tracing_Data/output_global/'+test+'/'
    ## Create directories for results
    create_directories(dir_output)

    modality = 'both'
    nn_input_shape = [64, 64, 64] # Input shape for NN
    threshold = 0.5

    dice_list = predict(dir_output, data_folder, dir_model_weights, dir_data_3d, modality, nn_input_shape, threshold, weighted_assembly, dir_seg, write_samples)

    info_file_name = "info"+'_'+modality+dt_string+".txt"
    
    f = open(dir_output +info_file_name,'a')
    for key in dice_list:
        f.write(f"\nCase: {key} , Dice: {dice_list[key]:.4f}")
    f.close()