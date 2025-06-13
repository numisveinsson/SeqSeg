def initialize_predictor(model_folder, fold):

    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    # check if GPU is available
    if torch.cuda.is_available():
        print('GPU available, using GPU')
        device_use = torch.device('cuda', 0)
    # check if mps is available (for Apple Silicon)
    elif torch.backends.mps.is_available() and not '3d' in model_folder:
        print('Using MPS backend for Apple Silicon, only available for 2D models')
        device_use = torch.device('mps')
    else:
        print('GPU not available, using CPU')
        device_use = torch.device('cpu', 0)
    print('About to load predictor object')

    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        device=device_use,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    print('About to load model')
    print('Model folder:', model_folder)
    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(fold,),
        checkpoint_name='checkpoint_best.pth',
    )
    print('Done loading model, ready to predict')

    return predictor
