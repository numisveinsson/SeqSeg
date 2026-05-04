import os


def _normalize_fold(fold):
    s = str(fold).strip().lower()
    if s == 'all':
        return 'all'
    return int(fold)


def _list_available_folds(model_folder):
    """Fold ids (int or 'all') for each fold_* directory under model_folder."""
    if not os.path.isdir(model_folder):
        return []
    folds = []
    for name in os.listdir(model_folder):
        if not name.startswith('fold_') or not os.path.isdir(
            os.path.join(model_folder, name)
        ):
            continue
        suffix = name[len('fold_'):]
        if suffix == 'all':
            folds.append('all')
        else:
            try:
                folds.append(int(suffix))
            except ValueError:
                continue
    return folds


def _fold_try_order(requested, available):
    """Try requested first, then numeric folds (sorted), then 'all'."""
    out = []
    seen = set()

    def push(f):
        if f not in seen:
            seen.add(f)
            out.append(f)

    push(requested)
    for f in sorted(x for x in available if isinstance(x, int)):
        if f != requested:
            push(f)
    if 'all' in available and requested != 'all':
        push('all')
    return out


# Prefer final (default nnU-Net export), then best, then latest.
_CHECKPOINT_CANDIDATES = (
    'checkpoint_final.pth',
    'checkpoint_best.pth',
    'checkpoint_latest.pth',
)


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

    requested = _normalize_fold(fold)
    available = _list_available_folds(model_folder)
    fold_order = _fold_try_order(requested, available)

    last_error = None
    for fold_id in fold_order:
        for checkpoint_name in _CHECKPOINT_CANDIDATES:
            try:
                predictor.initialize_from_trained_model_folder(
                    model_folder,
                    use_folds=(fold_id,),
                    checkpoint_name=checkpoint_name,
                )
                print(f'Loaded checkpoint: {checkpoint_name} (fold {fold_id})')
                print('Done loading model, ready to predict')
                return predictor
            except FileNotFoundError as error:
                last_error = error
                print(
                    f'Missing: fold {fold_id!r} with {checkpoint_name} '
                    f'({error.filename if getattr(error, "filename", None) else error})'
                )

    tried_folds = ', '.join(repr(f) for f in fold_order)
    tried_chk = ', '.join(_CHECKPOINT_CANDIDATES)
    raise FileNotFoundError(
        f'Could not load any checkpoint under {model_folder}. '
        f'Requested fold {requested!r}; tried folds [{tried_folds}] '
        f'with checkpoints [{tried_chk}].'
    ) from last_error
