import numpy as np
import pdb
import os
import pandas
import pickle
from ast import literal_eval

from modules import vtk_functions as vf
from modules import sitk_functions as sf
from modules.assembly import Segmentation
from modules.datasets import vmr_directories
from modules.evaluation import sensitivity_specificity, EvaluateTracing
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

def predict(output_folder, data_folder, model_folder, 
            dir_data_3d, modality, img_shape, threshold, 
            weighted, seg_file=False, write_samples=True, 
            global_scale=False, cropped=False, eval_threshold=False):

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
    final_dice_scores = []
    final_cent_scores = []

    for case in cases:
        print('\nNext case: ', case)
        samples = samples_cases[case]
        N_tot = len(samples)
        #print('\nNumber of samples: ', N_tot)
        dice_list = []

        samples_names = [f.replace('.nii.gz','') for f in samples]
        csv_array = csv_list.loc[csv_list['NAME'].isin(samples_names)]
        csv_list_small = csv_array[keep_values]

        dir_image, dir_seg, dir_cent, dir_surf = vmr_directories(dir_data_3d, 
                                                                 case, 
                                                                 global_scale,
                                                                 cropped)
        assembly_segs = Segmentation(case, dir_image, weighted)

        for sample in samples:#[0:N_tot:20]:

            if len(samples) > 10:
                if sample in samples[1:N_tot:N_tot//10]: print('*', end=" ", flush=True)
            # Read in
            volume = sitk.ReadImage(data_folder+sample)
            vol_np = sitk.GetArrayFromImage(volume)
            #print(f"Min: {vol_np.min():.2f}, Max: {vol_np.max():.2f}")
            seg_volume = sitk.ReadImage(data_folder.replace('_test', '_test_masks') +sample)

            sample = sample.replace('.nii.gz','.vtk')

            true_fn = output_folder +'truth/'+sample
            sitk.WriteImage(seg_volume, true_fn)
            img_fn = output_folder +'volumes/'+sample
            sitk.WriteImage(volume, img_fn)

            # Prediction
            predict = Prediction(unet, model_name, modality, volume, 
                                 img_shape, output_folder+'predictions', 
                                 threshold, seg_volume, global_scale)
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

        assembly_segs.create_mask()
        sitk.WriteImage(assembly_segs.mask, output_folder +'mask_'+case+'.vtk')

        if eval_threshold:
            for i in range(1,20):
                threshold = i*0.05
                assembly = sitk.BinaryThreshold(assembly_segs.assembly, lowerThreshold=threshold, upperThreshold=1)
                assembly_np = sitk.GetArrayFromImage(assembly)
                final_dice = dice_score(assembly_np, global_seg_truth_np)[0]
                #print('\nDice for threshold '+str(threshold)[:4]+': ', final_dice)
                sens, spec = sensitivity_specificity(assembly_np, global_seg_truth_np)
                print(f"Sensitivity, Specificity, DICE for threshold {threshold:.2f}: {sens:.3f}, {spec:.3f}, {final_dice:.3f}")
                #print(f"Specificity for threshold {threshold}: {spec}")
        else:
            assembly = sitk.BinaryThreshold(assembly_segs.assembly, lowerThreshold=threshold, upperThreshold=1)
        
        surface_assembly = vf.evaluate_surface(assembly, 1)
        surface_assembly = vf.smooth_surface(surface_assembly, 15)
        vf.write_vtk_polydata(surface_assembly, output_folder +'assembly_surface_'+case+'.vtp')

        seed = np.zeros(3)
        eval_tracing = EvaluateTracing(case, seed, dir_seg, dir_surf, dir_cent, assembly, surface_assembly)
        missed_branches, perc_caught, total_perc = eval_tracing.count_branches()
        final_dice = eval_tracing.calc_dice_score()
        final_dice_scores.append(final_dice)
        final_cent_scores.append(total_perc)


    return cases, final_dice_scores, final_cent_scores

if __name__=='__main__':
    # [test, global_scale]
    tests = [['test122', False],
             ['test123', False],
            #  ['test115', False],
            #  ['test113', False],
            #  ['test118', False],
            #  ['test119', False],
             ]
    
    data_folder = '/Users/numisveins/Documents/Automatic_Tracing_Data/train_global_aortas/'
    dir_data_3d = '/Users/numisveins/Library/Mobile Documents/com~apple~CloudDocs/Documents/Side_SV_projects/SV_ML_Training/vascular_data_3d/'
    dir_output0 = '/Users/numisveins/Documents/Automatic_Tracing_Data/output_global/'    
    cropped=True
    
    write_samples = True
    dir_seg = True
    weighted_assembly = False
    eval_threshold = False
    nn_input_shape = [128, 128, 128] # Input shape for NN
    threshold = 0.5

    global_dict = {}
    global_dict['test'] = []
    global_dict['ct dice'] = []
    global_dict['mr dice'] = []
    global_dict['ct cent'] = []
    global_dict['mr cent'] = []

    for test in tests:
        
        if test == 'test101': nn_input_shape = [128,128,128]
        global_scale = test[1]
        test = test[0]
        print('\nTest is: ', test)
        dir_model_weights = '/Users/numisveins/Documents/Automatic_Tracing_ML/weights/' + test + '/'
        dir_output = dir_output0+test+'/'
        ## Create directories for results
        create_directories(dir_output)

        global_dict['test'].append(test)

        for modality in ['ct','mr']:

            cases, dice_scores, cent_scores = predict(dir_output, data_folder, dir_model_weights, 
                                dir_data_3d, modality, nn_input_shape, threshold, 
                                weighted_assembly, dir_seg, write_samples, 
                                global_scale, cropped, eval_threshold)

            info_file_name = "info"+'_'+modality+".txt"
            
            f = open(dir_output +info_file_name,'a')
            for i, case in enumerate(cases):
                f.write(f"\nCase: {case} , Dice: {dice_scores[i]:.4f}, Cent: {cent_scores[i]:.4f}")
            f.close()

            global_dict[modality+' dice'].append(np.array(dice_scores).mean())
            global_dict[modality+' cent'].append(np.array(cent_scores).mean())

    #import pdb; pdb.set_trace()
    with open(dir_output0+'results.pickle', 'wb') as handle:
        pickle.dump(global_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    pdb.set_trace()