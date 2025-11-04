import sys
sys.path.append("/global/scratch/users/numi/SeqSeg/nnUNet/")

from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

if __name__=='__main__':

    # predict_from_raw_data('/media/fabian/data/nnUNet_raw/Dataset003_Liver/imagesTs',
    #                       '/media/fabian/data/nnUNet_raw/Dataset003_Liver/imagesTs_predlowres',
    #                       join(nnUNet_results, 'Dataset002_SEQAORTAS/nnUNetTrainer__nnUNetPlans__2d'),
    #                       (0,),
    #                       0.5,
    #                       use_gaussian=True,
    #                       use_mirroring=False,
    #                       perform_everything_on_gpu=True,
    #                       verbose=True,
    #                       save_probabilities=False,
    #                       overwrite=False,
    #                       checkpoint_name='checkpoint_final.pth',
    #                       num_processes_preprocessing=3,
    #                       num_processes_segmentation_export=3)

    print('About to load predictor object')
    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True,
        device=torch.device('cpu', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    print('About to load model')
    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, 'Dataset002_SEQAORTAS/nnUNetTrainer__nnUNetPlans__3d_fullres'),
        use_folds=(0,),
        checkpoint_name='checkpoint_best.pth',
    )
    print('Done loading model, about to predict')

    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    # predict a single numpy array
    img, props = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset002_SEQAORTAS/imagesTs/ct_test/0141_1001_6_1.nii.gz')])
    import pdb; pdb.set_trace()

    ret = predictor.predict_single_npy_array(img, props, None, None, True)
    print('Prediction done')
    
    print(type(img))
    print(type(ret[0]))
    print(type(ret[1]))
    print(img)
    print(ret[0])
    print(ret[1])
    print(ret[1].mean())
    print(ret[1].min())
    print(ret[1].max())
    print(props)