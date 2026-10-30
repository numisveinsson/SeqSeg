# Functions to bind SITK functionality

import json
import os

import SimpleITK as sitk
import numpy as np


def read_image(file_dir_image):
    """
    Read image from file
    Args:
        file_dir_image: image directory
    Returns:
        SITK image reader
    """
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(file_dir_image)
    file_reader.ReadImageInformation()
    return file_reader


def create_new(file_reader, PixelID=None):
    """
    Create new SITK image with same formating as another
    Args:
        file_reader: reader from another image, or image object
    Returns:
        SITK image
    """
    if PixelID:
        pix_id = PixelID
    else:
        pix_id = file_reader.GetPixelID()

    # check if reader or image object
    if isinstance(file_reader, sitk.ImageFileReader):
        result_img = sitk.Image(file_reader.GetSize(), pix_id,
                                file_reader.GetNumberOfComponents())
    else:
        result_img = sitk.Image(file_reader.GetSize(), pix_id)
    result_img.SetSpacing(file_reader.GetSpacing())
    result_img.SetOrigin(file_reader.GetOrigin())
    result_img.SetDirection(file_reader.GetDirection())

    return result_img


def copy_settings(img, ref_img):

    img.SetSpacing(ref_img.GetSpacing())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())

    return img


# Cap assembly upsampling (e.g. CENT_MAX_SPACING in mm applied with ``-unit cm``)
# before SimpleITK tries to allocate a multi-billion-voxel grid and the process
# is SIGKILL'd (exit 137). 500M float32 voxels is ~2 GB for one volume.
DEFAULT_MAX_RESAMPLE_VOXELS = 500_000_000


def resampled_grid_size(orig_size, orig_spacing, new_spacing):
    """Voxel size that :func:`resample_to_spacing` would produce."""
    orig_size = np.asarray(orig_size, dtype=np.int64)
    orig_spacing = np.asarray(orig_spacing, dtype=np.float64)
    target_spacing = np.asarray(new_spacing, dtype=np.float64)
    if np.any(target_spacing <= 0):
        raise ValueError(f"new_spacing must be > 0, got {target_spacing.tolist()}")
    return np.maximum(
        1,
        np.rint(orig_size * (orig_spacing / target_spacing)).astype(np.int64),
    )


def fit_spacing_to_max_voxels(
    orig_size,
    orig_spacing,
    target_spacing,
    max_voxels=DEFAULT_MAX_RESAMPLE_VOXELS,
):
    """
    Coarsen ``target_spacing`` so the resampled grid has at most ``max_voxels``.

    Axes that are already at ``orig_spacing`` are left unchanged. Remaining
    (finer) axes are scaled uniformly. Never coarsens past ``orig_spacing``.

    Returns
    -------
    list[float] | None
        Spacing to resample to, or ``None`` if no axis can be refined without
        exceeding ``max_voxels`` (caller should skip resampling).
    """
    if max_voxels is None:
        return [float(s) for s in target_spacing]

    orig_size = np.asarray(orig_size, dtype=np.float64).reshape(3)
    orig_spacing = np.asarray(orig_spacing, dtype=np.float64).reshape(3)
    target = np.asarray(target_spacing, dtype=np.float64).reshape(3)
    max_voxels = int(max_voxels)
    if max_voxels <= 0:
        raise ValueError("max_voxels must be > 0")

    def n_vox_for(spacing):
        return int(np.prod(resampled_grid_size(orig_size, orig_spacing, spacing)))

    n_vox = n_vox_for(target)
    if n_vox <= max_voxels:
        return target.tolist()

    for _ in range(32):
        finer = target < orig_spacing * (1.0 - 1e-12)
        n_fine = int(np.count_nonzero(finer))
        if n_fine == 0:
            return None
        k = (n_vox / float(max_voxels)) ** (1.0 / n_fine)
        new_target = target.copy()
        new_target[finer] = np.minimum(orig_spacing[finer], target[finer] * k)
        if np.allclose(new_target, target):
            new_target[finer] = np.minimum(
                orig_spacing[finer], target[finer] * 1.001
            )
            if np.allclose(new_target, target):
                return None
        target = new_target
        n_vox = n_vox_for(target)
        if n_vox <= max_voxels:
            if not np.any(target < orig_spacing):
                return None
            return target.tolist()
    return None


def resample_to_spacing(image, new_spacing, is_label=False,
                        max_voxels=DEFAULT_MAX_RESAMPLE_VOXELS):
    """
    Resample a SITK image to a target spacing while preserving physical extent.

    Args:
        image (sitk.Image): Input image.
        new_spacing (iterable): Target spacing [sx, sy, sz].
        is_label (bool): Use nearest-neighbor interpolation for labels.
        max_voxels (int | None): If set, raise ``ValueError`` when the
            output grid would exceed this many voxels instead of allocating.
    """
    orig_size = np.array(image.GetSize(), dtype=np.int64)
    orig_spacing = np.array(image.GetSpacing(), dtype=np.float64)
    target_spacing = np.array(new_spacing, dtype=np.float64)

    new_size_arr = resampled_grid_size(orig_size, orig_spacing, target_spacing)
    n_vox = int(np.prod(new_size_arr))
    if max_voxels is not None and n_vox > int(max_voxels):
        raise ValueError(
            f"Resampling to spacing {target_spacing.tolist()} would create "
            f"{tuple(int(s) for s in new_size_arr)} voxels ({n_vox:,}), "
            f"exceeding max_voxels={int(max_voxels):,}. Check CENT_MAX_SPACING "
            f"and -unit (mm vs cm) before refining a full-volume assembly."
        )
    new_size = new_size_arr.tolist()

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing.tolist())
    resample.SetSize([int(s) for s in new_size])
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputDirection(image.GetDirection())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)
    return resample.Execute(image)


def max_spacing_mm_to_image_units(max_spacing_mm, unit):
    """
    Convert a max spacing specified in mm to image coordinate units.

    Parameters
    ----------
    max_spacing_mm : float
        Maximum spacing in millimeters.
    unit : str
        Image units, ``'cm'`` or ``'mm'``.
    """
    unit_norm = str(unit).strip().lower()
    max_spacing_mm = float(max_spacing_mm)
    if max_spacing_mm <= 0:
        raise ValueError("max_spacing_mm must be > 0")
    if unit_norm == "cm":
        return max_spacing_mm * 0.1
    if unit_norm == "mm":
        return max_spacing_mm
    raise ValueError(f"Unsupported unit '{unit}'; expected 'cm' or 'mm'")


def capped_target_spacing(spacing, max_spacing_mm, unit):
    """
    Cap each spacing component to at most ``max_spacing_mm``.

    ``max_spacing_mm`` is in millimeters and converted to image units via
    ``unit``. Dimensions already finer than the cap are left unchanged.

    Returns
    -------
    list[float] | None
        Target spacing if any dimension was capped, otherwise ``None``.
    """
    max_sp = max_spacing_mm_to_image_units(max_spacing_mm, unit)
    orig = [float(s) for s in spacing]
    target = [min(s, max_sp) for s in orig]
    if any(t < s for t, s in zip(target, orig)):
        return target
    return None


def parse_resample_spacing_arg(spacing_arg):
    """
    Convert CLI ``--resample_spacing`` values to an (sx, sy, sz) tuple.

    Parameters
    ----------
    spacing_arg : list[float] | None
        From argparse ``nargs='+'``. None or empty means disabled.

    Returns
    -------
    tuple[float, float, float] | None
        SimpleITK spacing order, or None when the flag is omitted.
        Omitted means each axis uses the finer of the native spacing and
        the model spacing.
    """
    if not spacing_arg:
        return None
    if len(spacing_arg) == 1:
        s = float(spacing_arg[0])
        return (s, s, s)
    if len(spacing_arg) == 3:
        return tuple(float(x) for x in spacing_arg)
    raise ValueError(
        '--resample_spacing expects 1 value (isotropic) or 3 values '
        f'(sx, sy, sz in SimpleITK order); got {len(spacing_arg)} values.'
    )


def _configuration_name(model_folder):
    """Last path component's nnU-Net configuration, e.g. ``3d_fullres``."""
    folder = os.path.basename(os.path.normpath(str(model_folder)))
    return folder.split("__")[-1]


def load_nnunet_plans(model_folder):
    """Read ``plans.json`` from an nnU-Net trainer folder."""
    folder = os.path.abspath(os.path.expanduser(str(model_folder)))
    path = os.path.join(folder, "plans.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"nnU-Net plans.json not found in {folder}. "
            "The model folder should be a trainer directory such as "
            "nnUNetTrainer__nnUNetPlans__3d_fullres."
        )
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def plan_spacing_to_image_units(plan_spacing, transpose_forward, image_spacing, scale):
    """
    Convert an nnU-Net configuration spacing to SimpleITK image-header units.

    ``plan_spacing`` is in the transposed axis order stored in ``plans.json``
    and in the units the network was trained with. ``scale`` is the factor
    applied to image spacing before it is passed to nnU-Net, so dividing by
    ``scale`` returns image-header units. Order is SimpleITK ``(x, y, z)``.

    A 2D plan lists only the two in-plane spacings. The through-plane axis
    (the first axis after ``transpose_forward``) keeps the image spacing.
    """
    scale = float(scale)
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")
    image_spacing = [float(s) for s in image_spacing]
    if len(image_spacing) != 3:
        raise ValueError(
            f"image_spacing must have 3 values, got {image_spacing}"
        )
    # props order matches what tracing passes to nnU-Net: reversed SimpleITK.
    props = [s * scale for s in image_spacing[::-1]]
    transpose_forward = [int(i) for i in transpose_forward]
    if sorted(transpose_forward) != [0, 1, 2]:
        raise ValueError(
            "transpose_forward must be a permutation of 0, 1, 2; "
            f"got {transpose_forward}"
        )
    transposed_native = [props[i] for i in transpose_forward]
    plan_spacing = [float(s) for s in plan_spacing]
    if len(plan_spacing) == 3:
        target_transposed = plan_spacing
    elif len(plan_spacing) == 2:
        target_transposed = [transposed_native[0]] + plan_spacing
    else:
        raise ValueError(
            f"plans spacing must have 2 or 3 values, got {plan_spacing}"
        )
    target_props = [None] * 3
    for new_i, old_i in enumerate(transpose_forward):
        target_props[old_i] = target_transposed[new_i]
    return [s / scale for s in target_props[::-1]]


def model_spacing_in_image_units(model_folder, image_spacing, scale):
    """
    Model target spacing in SimpleITK order and image-header units.

    Read from ``plans.json`` for the configuration named by the trainer
    folder (``3d_fullres``, ``2d``, ...).
    """
    plans = load_nnunet_plans(model_folder)
    configuration = _configuration_name(model_folder)
    configs = plans.get("configurations") or {}
    if configuration not in configs:
        available = ", ".join(repr(k) for k in sorted(configs)) or "(none)"
        raise ValueError(
            f"plans.json in {model_folder} has no configuration "
            f"{configuration!r}. Available: {available}"
        )
    plan_spacing = configs[configuration].get("spacing")
    if not plan_spacing:
        raise ValueError(
            f"configuration {configuration!r} in {model_folder} has no spacing"
        )
    transpose = plans.get("transpose_forward", [0, 1, 2])
    return plan_spacing_to_image_units(
        plan_spacing, transpose, image_spacing, scale
    )


def finer_axis_spacing(native, *candidates, atol=1e-8):
    """
    Per-axis minimum of ``native`` and each candidate spacing.

    Returns the target spacing, or ``None`` when no axis is finer than
    ``native``.
    """
    target = [float(s) for s in native]
    if len(target) != 3:
        raise ValueError(f"native spacing must have 3 values, got {target}")
    for cand in candidates:
        cand = [float(s) for s in cand]
        if len(cand) != 3:
            raise ValueError(f"candidate spacing must have 3 values, got {cand}")
        target = [min(a, b) for a, b in zip(target, cand)]
    native_f = [float(s) for s in native]
    if not any(t < n - atol for t, n in zip(target, native_f)):
        return None
    return target


def automatic_resample_spacing(
    image_spacing,
    image_size,
    model_spacings,
    max_voxels=DEFAULT_MAX_RESAMPLE_VOXELS,
):
    """
    Spacing to resample an input to, before tracing.

    Each axis is the finer of the native spacing and every model spacing
    (SimpleITK order, image-header units). The target is then coarsened so
    the grid stays within ``max_voxels``, and never coarser than native.

    Returns
    -------
    tuple[list[float] | None, list[float] | None]
        ``(target, ideal)``. ``target`` is what to resample to, or ``None``
        to keep the native grid. ``ideal`` is the uncapped finer spacing, or
        ``None`` when the native grid is already as fine as the models.
    """
    ideal = finer_axis_spacing(image_spacing, *model_spacings)
    if ideal is None:
        return None, None
    fitted = fit_spacing_to_max_voxels(
        image_size, image_spacing, ideal, max_voxels=max_voxels
    )
    if fitted is None or finer_axis_spacing(image_spacing, fitted) is None:
        return None, ideal
    return fitted, ideal


def resample_inputs_for_case(
    dir_image,
    dir_seg,
    out_dir,
    stem,
    img_ext,
    model_specs,
    explicit_spacing=None,
):
    """
    Resample a case image (and truth) before tracing.

    ``explicit_spacing`` is used as given (SimpleITK order, image-header
    units). When it is omitted, each axis uses the finer of the native
    spacing and the target spacing of each ``(model_folder, scale)`` pair.
    An empty ``model_specs`` list keeps the native grid: pre-segmented
    tracing (``SEGMENTATION: True``) never loads nnU-Net, so there is no
    ``plans.json`` spacing to match.

    Returns the image and segmentation paths to trace. Original dataset
    files are not modified.
    """
    if explicit_spacing is not None:
        os.makedirs(out_dir, exist_ok=True)
        return maybe_resample_volume_paths(
            dir_image,
            dir_seg,
            explicit_spacing,
            out_dir,
            stem,
            img_ext,
        )

    if not model_specs:
        print("Keeping native image spacing (no nnU-Net model spacing applies)")
        return dir_image, dir_seg

    reader = read_image(dir_image)
    native = [float(s) for s in reader.GetSpacing()]
    model_spacings = []
    for folder, scale in model_specs:
        spacing = model_spacing_in_image_units(folder, native, scale)
        name = os.path.basename(os.path.normpath(str(folder)))
        print(
            f"Model target spacing ({name}) in image units (x, y, z): "
            f"{spacing}"
        )
        model_spacings.append(spacing)

    target, ideal = automatic_resample_spacing(
        native, reader.GetSize(), model_spacings
    )
    if target is None:
        if ideal is None:
            print(f"Keeping native image spacing (x, y, z): {native}")
        else:
            print(
                "Keeping native image spacing "
                f"{native}: resampling to {ideal} would exceed "
                f"{DEFAULT_MAX_RESAMPLE_VOXELS:,} voxels"
            )
        return dir_image, dir_seg

    if any(abs(a - b) > 1e-12 for a, b in zip(target, ideal)):
        print(
            "Relaxed input spacing from "
            f"{ideal} to {target} to stay <= "
            f"{DEFAULT_MAX_RESAMPLE_VOXELS:,} voxels"
        )
    print(f"Resampling input from {native} to {target} (x, y, z)")
    os.makedirs(out_dir, exist_ok=True)
    return maybe_resample_volume_paths(
        dir_image,
        dir_seg,
        target,
        out_dir,
        stem,
        img_ext,
    )


def maybe_resample_volume_paths(
    dir_image,
    dir_seg,
    target_spacing,
    out_dir,
    stem,
    img_ext,
):
    """
    Optionally resample the case image (and truth seg) to ``target_spacing``.

    Writes under ``out_dir`` only; original dataset paths are not modified.

    Parameters
    ----------
    dir_image : str
        Path to the input image volume.
    dir_seg : str | None
        Optional path to a label/truth volume aligned with the image.
    target_spacing : tuple[float, float, float] | None
        If None, returns ``(dir_image, dir_seg)`` unchanged.
    out_dir : str
        Existing directory for resampled outputs.
    stem : str
        Case stem used in output filenames.
    img_ext : str
        File extension including dot (e.g. ``.nii.gz``).

    Returns
    -------
    tuple[str, str | None]
        ``(dir_image_out, dir_seg_out)`` paths to use for the pipeline.
    """
    if target_spacing is None:
        return dir_image, dir_seg

    if any(s <= 0 for s in target_spacing):
        raise ValueError(
            f'target_spacing must be positive in all dimensions; got {target_spacing}'
        )

    img_in = sitk.ReadImage(dir_image)
    orig_sp = img_in.GetSpacing()
    print(
        f'Resampling input image to spacing {target_spacing} '
        f'(original spacing {orig_sp})'
    )
    img_out = resample_to_spacing(img_in, target_spacing, is_label=False)
    out_image_path = os.path.join(out_dir, f'{stem}_resampled{img_ext}')
    write_image(img_out, out_image_path)

    out_seg_path = dir_seg
    if dir_seg and isinstance(dir_seg, str) and os.path.isfile(dir_seg):
        seg_in = sitk.ReadImage(dir_seg)
        print(
            f'Resampling truth segmentation to spacing {target_spacing} '
            f'(original spacing {seg_in.GetSpacing()})'
        )
        seg_out = resample_to_spacing(seg_in, target_spacing, is_label=True)
        out_seg_path = os.path.join(out_dir, f'{stem}_truth_resampled{img_ext}')
        write_image(seg_out, out_seg_path)

    return out_image_path, out_seg_path


def geometry_matches(image, reference, atol=1e-6):
    """Return True if ``image`` and ``reference`` share size, spacing, origin, direction."""
    if tuple(image.GetSize()) != tuple(reference.GetSize()):
        return False
    if not np.allclose(image.GetSpacing(), reference.GetSpacing(), atol=atol):
        return False
    if not np.allclose(image.GetOrigin(), reference.GetOrigin(), atol=atol):
        return False
    if not np.allclose(image.GetDirection(), reference.GetDirection(), atol=atol):
        return False
    return True


def as_probability_image(seg):
    """
    Cast a segmentation to float32 in ``[0, 1]``.

    Integer labels (e.g. 0/1 or 0/255) are divided by their max so they can
    be used as an assembly prior.
    """
    arr = sitk.GetArrayFromImage(seg).astype(np.float32, copy=False)
    vmax = float(np.max(arr)) if arr.size else 0.0
    if vmax > 1.0:
        arr = arr / vmax
    arr = np.clip(arr, 0.0, 1.0)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(seg)
    return out


def resample_to_reference(image, reference, is_label=True):
    """Resample ``image`` onto ``reference``'s grid, or return a copy if already aligned."""
    if geometry_matches(image, reference):
        aligned = sitk.Image(image)
        aligned.CopyInformation(reference)
        return aligned
    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    return sitk.Resample(
        image,
        reference,
        sitk.Transform(),
        interpolator,
        0.0,
        image.GetPixelID(),
    )


def resolve_existing_sweep_seg(case_dir, case, legacy_dir=None):
    """Find a global sweep mask a previous plus run already wrote.

    ``{case}_sweep_seg.mha`` is checked in ``case_dir`` first, then in
    ``legacy_dir`` (older runs wrote it next to the case folders).
    ``{case}_sweep_probability.mha`` beside that mask is included when
    the file is present.

    Returns ``(binary_path, probability_path)`` or ``None``.
    ``probability_path`` may be ``None``.
    """
    binary_name = f"{case}_sweep_seg.mha"
    probability_name = f"{case}_sweep_probability.mha"
    roots = []
    for directory in (case_dir, legacy_dir):
        if not directory:
            continue
        root = os.path.abspath(os.path.expanduser(str(directory)))
        if root not in roots:
            roots.append(root)
    for root in roots:
        binary = os.path.join(root, binary_name)
        if os.path.isfile(binary):
            probability = os.path.join(root, probability_name)
            if not os.path.isfile(probability):
                probability = None
            return binary, probability
    return None


def resolve_global_seg_files(seg_dir, case, img_ext):
    """Find the global mask for one case.

    Search order:

    1. ``{case}_segmentation.mha``, plus ``{case}_probability.mha`` when present
    2. ``{case}{img_ext}``
    3. ``{case}_probability.mha`` alone

    Returns ``(binary_path, probability_path)``. Either path may be ``None``,
    but not both. Raises ``FileNotFoundError`` listing the paths that were tried.
    """
    root = os.path.abspath(os.path.expanduser(str(seg_dir)))
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"Global segmentation directory not found: {root}"
        )
    segmentation = os.path.join(root, f"{case}_segmentation.mha")
    probability = os.path.join(root, f"{case}_probability.mha")
    named = os.path.join(root, f"{case}{img_ext}") if img_ext else None

    if os.path.isfile(segmentation):
        prob = probability if os.path.isfile(probability) else None
        return segmentation, prob
    if named is not None and os.path.isfile(named):
        return named, None
    if os.path.isfile(probability):
        return None, probability

    tried = [segmentation]
    if named is not None:
        tried.append(named)
    tried.append(probability)
    listing = "\n  ".join(tried)
    raise FileNotFoundError(
        f"No global segmentation for case {case!r} in {root}. Tried:\n  {listing}"
    )


def _clip_probability(image, reference_image):
    """Float32 image in ``[0, 1]`` on ``reference_image``'s grid."""
    arr = np.clip(
        sitk.GetArrayFromImage(image).astype(np.float32, copy=False),
        0.0,
        1.0,
    )
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(reference_image)
    return out


def load_global_segmentation(binary_path, probability_path, reference_image):
    """Load a global mask and optional probability onto ``reference_image``.

    A label mask is resampled with nearest neighbor. Foreground is any voxel
    greater than 0, stored as a 0/1 ``UInt8`` image. When no probability file
    is given, that mask is the assembly prior. A probability file is resampled
    linearly and clipped to ``[0, 1]``; it is the assembly prior and is not
    replaced by the binary mask. A probability file alone is thresholded at
    0.5 to make the binary mask.

    Returns ``(binary, probability)``.
    """
    if binary_path is None and probability_path is None:
        raise ValueError("A binary mask or a probability image is required")

    binary = None
    probability = None
    if binary_path is not None:
        print(f"Loading global segmentation: {binary_path}")
        label = sitk.ReadImage(binary_path)
        if not geometry_matches(label, reference_image):
            print(
                "Global segmentation grid does not match the image; "
                "resampling onto the image grid"
            )
            label = resample_to_reference(
                label, reference_image, is_label=True
            )
        mask = (sitk.GetArrayFromImage(label) > 0).astype(np.uint8)
        binary = sitk.GetImageFromArray(mask)
        binary.CopyInformation(reference_image)

    if probability_path is not None:
        print(f"Loading global probability: {probability_path}")
        prob = sitk.ReadImage(probability_path)
        if not geometry_matches(prob, reference_image):
            print(
                "Global probability grid does not match the image; "
                "resampling onto the image grid"
            )
            prob = resample_to_reference(
                prob, reference_image, is_label=False
            )
        probability = _clip_probability(prob, reference_image)

    if binary is None:
        binary_img = sitk.BinaryThreshold(
            probability,
            lowerThreshold=0.5,
            upperThreshold=1.0,
        )
        binary = sitk.Cast(binary_img, sitk.sitkUInt8)
        binary.CopyInformation(reference_image)
    if probability is None:
        probability = as_probability_image(binary)
    return binary, probability


def load_start_segmentation(path, reference_image):
    """
    Load a starting segmentation and place it on ``reference_image``'s grid.

    Values are normalized to probability ``[0, 1]`` so they can seed the
    global assembly and be merged with SeqSeg's output.
    """
    path = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Starting segmentation not found: {path}")
    print(f"Loading starting segmentation: {path}")
    seg = sitk.ReadImage(path)
    if not geometry_matches(seg, reference_image):
        print(
            "Starting segmentation grid does not match the image; "
            "resampling onto the image grid"
        )
        seg = resample_to_reference(seg, reference_image, is_label=True)
    return as_probability_image(seg)


def merge_binary_with_start(binary, start_prob, threshold=0.5):
    """Union a SeqSeg binary mask with a starting probability / label image."""
    if not geometry_matches(start_prob, binary):
        start_prob = resample_to_reference(start_prob, binary, is_label=True)
    start_bin = sitk.BinaryThreshold(
        start_prob,
        lowerThreshold=float(threshold),
        upperThreshold=1.0,
    )
    merged = sitk.Or(
        sitk.Cast(binary, sitk.sitkUInt8),
        sitk.Cast(start_bin, sitk.sitkUInt8),
    )
    merged.CopyInformation(binary)
    return merged


def merge_probability_with_start(prob, start_prob):
    """Voxel-wise maximum of SeqSeg probabilities and a starting segmentation."""
    if not geometry_matches(start_prob, prob):
        start_prob = resample_to_reference(start_prob, prob, is_label=False)
    merged = sitk.Maximum(
        sitk.Cast(prob, sitk.sitkFloat32),
        sitk.Cast(start_prob, sitk.sitkFloat32),
    )
    merged.CopyInformation(prob)
    return merged


def write_image(image, outputImageFileName):
    """
    Write image to file
    Args:
        SITK image, filename
    Returns:
        image file
    """
    writer = sitk.ImageFileWriter()
    writer.SetFileName(outputImageFileName)
    writer.Execute(image)


def validate_potential_branches_image_bounds(image, potential_branches,
                                             case=None, image_path=None):
    """
    Ensure all tracing seed points (old and initial) lie inside the image grid.

    Uses continuous index vs image size so physical points are rejected when
    they map outside [0, size[d]) in any dimension.

    Parameters
    ----------
    image : sitk.Image
        Reference volume (same geometry as the image used for tracing).
    potential_branches : list
        Initialization steps from seqseg.modules.initialization; each dict must
        contain 'old point' and 'point' (3D physical coordinates).
    case : str, optional
        Case name for error messages.
    image_path : str, optional
        Path to the image file for error messages.

    Raises
    ------
    ValueError
        If any seed is out of bounds or not 3D (for a 3D image).
    """
    if not potential_branches:
        return

    dim = image.GetDimension()
    size = image.GetSize()
    failures = []

    for bi, step in enumerate(potential_branches):
        for role, key in (("old point", "old point"), ("initial point", "point")):
            pt = step[key]
            arr = np.asarray(pt, dtype=float).ravel()
            if arr.size != dim:
                failures.append(
                    (bi, role, tuple(arr.tolist()),
                     f"expected {dim} coordinates, got length {arr.size}")
                )
                continue
            phys = tuple(arr.tolist())
            cix = image.TransformPhysicalPointToContinuousIndex(phys)
            if any(cix[d] < 0 or cix[d] >= size[d] for d in range(dim)):
                failures.append(
                    (bi, role, phys,
                     tuple(round(float(cix[i]), 4) for i in range(dim)))
                )

    if not failures:
        return

    lines = []
    for bi, role, phys, detail in failures:
        lines.append(f"  branch {bi} {role}: physical={phys!r} ({detail})")

    prefix_parts = []
    if case is not None:
        prefix_parts.append(f"case {case!r}")
    if image_path is not None:
        prefix_parts.append(f"image {image_path!r}")
    prefix = (" (" + ", ".join(prefix_parts) + ")") if prefix_parts else ""

    raise ValueError(
        f"Seed point(s) lie outside the image volume bounds{prefix}. "
        f"Image size={size!r}, spacing={image.GetSpacing()!r}, "
        f"origin={image.GetOrigin()!r}.\n" + "\n".join(lines)
    )


def keep_component_seeds(image, seeds):
    """
    Keep only the component containing the seed point
    Args:
        SITK image, seed point pointing to point in vessel of interest
        seed(s): location of seeds; list of np arrays (one or multiple points)
    """
    # create list of index values for each seed
    index_seeds = []
    for i in range(len(seeds)):
        index_seeds.append(image
                           .TransformPhysicalPointToIndex(seeds[i].tolist()))
    print(f"Index of seeds: {index_seeds}")

    # create a new image with only the labels of interest
    labelImage = remove_other_vessels(image, index_seeds)

    return labelImage


def remove_other_vessels(image, seed):
    """
    Remove all labelled vessels except the one of interest
    Args:
        SITK image, seed point pointing to point in vessel of interest
        seed(s): location of seed; either list (single point)
        or list of lists (multiple points)
    Returns:
        binary image file (either 0 or 1)
    """

    ccimage = sitk.ConnectedComponent(image)
    size = ccimage.GetSize()

    # Normalize input to a list of 3D tuple indices.
    if isinstance(seed, (tuple, list, np.ndarray)) and len(seed) == 3 and \
       all(isinstance(v, (int, np.integer)) for v in seed):
        seed_list = [tuple(int(v) for v in seed)]
    elif isinstance(seed, (tuple, list, np.ndarray)):
        seed_list = [tuple(int(v) for v in s) for s in seed]
    else:
        raise TypeError("seed must be a 3D index or a list of 3D indices")

    labels_seeds = []
    for s in seed_list:
        # Skip out-of-bounds seeds instead of crashing or mis-indexing.
        if any(s[d] < 0 or s[d] >= size[d] for d in range(3)):
            continue
        label = int(ccimage[s])
        # Ignore background labels for multi-seed union logic.
        if label > 0:
            labels_seeds.append(label)

    labels_seeds = sorted(set(labels_seeds))
    print(f"Labels at seed indices: {labels_seeds}")

    if not labels_seeds:
        # Fallback: keep largest connected component when no seed lands
        # in vessel foreground.
        relabeled = sitk.RelabelComponent(ccimage, sortByObjectSize=True)
        labelImage = sitk.BinaryThreshold(relabeled,
                                          lowerThreshold=1,
                                          upperThreshold=1)
    else:
        # Create a new image with union of all seed-connected labels.
        labelImage = sitk.BinaryThreshold(ccimage,
                                          lowerThreshold=labels_seeds[0],
                                          upperThreshold=labels_seeds[0])
        for label in labels_seeds[1:]:
            labelImage = labelImage + sitk.BinaryThreshold(
                ccimage,
                lowerThreshold=label,
                upperThreshold=label
            )

    return labelImage


def connected_comp_info(original_seg, print_condition=False):
    """
    Print info on the component being kept
    """
    removed_seg = sitk.ConnectedComponent(original_seg)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.Execute(removed_seg, original_seg)
    means = []
    sizes = []

    for label in stats.GetLabels():
        if print_condition:
            print("Label: {0} -> Mean: {1} Size: {2}"
                  .format(label, stats.GetMean(label),
                          stats.GetPhysicalSize(label)))
        means.append(stats.GetMean(label))
        sizes.append(stats.GetPhysicalSize(label))
    return stats.GetLabels(), means, sizes


def extract_volume(reader_im, index_extract, size_extract):
    """
    Extract a subvolume from a full image.

    Parameters
    ----------
    reader_im : sitk.ImageFileReader or sitk.Image
        Full volume on disk (reader) or in memory (image).
    index_extract, size_extract : sequence of int
        Lower corner and size in voxels (SimpleITK index order).
    """
    if isinstance(reader_im, sitk.Image):
        roi = sitk.RegionOfInterestImageFilter()
        roi.SetIndex([int(x) for x in index_extract])
        roi.SetSize([int(x) for x in size_extract])
        return roi.Execute(reader_im)

    reader_im.SetExtractIndex(index_extract)
    reader_im.SetExtractSize(size_extract)
    new_img = reader_im.Execute()

    return new_img


def map_to_image(point, radius, size_volume, origin_im, spacing_im,
                 size_im, min_resolution_any_dim=5, direction_im=None):
    """
    Function to map a point and radius to volume metrics
    Also checks if sub-volume is within global

    Note: only corrects if less than (1/3) of the subvolume
    is outside the global volume. This can be changed
    -> less means it will fail sooner when approaching the
    global volume border

    args:
        point: point of volume center (physical coordinates)
        radius: radius at that point
        size_volume: multiple of radius equal the intended
            volume size
        origin_im: image origin
        spacing_im: image spacing
        size_im: image size in voxels (per dimension)
        min_resolution_any_dim: minimum voxel count required along any
            dimension; the radius is grown by 5% per iteration until met
        direction_im: image direction cosines as either a 3x3 array or a
            flat 9-element sequence (SimpleITK ``GetDirection()`` order).
            If ``None``, an identity direction is assumed (legacy
            behavior). Passing the actual direction is required for
            images whose direction matrix is not diagonal (e.g. oblique
            acquisitions); otherwise the extraction index will be wrong.
    return:
        size_extract: number of voxels to extract in each dim
        index_extract: index for sitk volume extraction
        border: boolean, if subvolume is on global border
    """
    point = np.asarray(point, dtype=float).ravel()
    origin_im = np.asarray(origin_im, dtype=float).ravel()
    spacing_im = np.asarray(spacing_im, dtype=float).ravel()
    dim = origin_im.size

    # Build the inverse of the physical-to-index transform. SimpleITK
    # convention: physical = origin + D @ diag(spacing) @ index, so
    # index = diag(1/spacing) @ D^{-1} @ (physical - origin). For an
    # orthonormal direction D^{-1} == D.T, but use the true inverse so
    # we remain robust to mild numerical noise.
    if direction_im is None:
        d_inv = np.eye(dim)
    else:
        d_mat = np.asarray(direction_im, dtype=float).reshape(dim, dim)
        d_inv = np.linalg.inv(d_mat)

    min_res = 0
    border = False

    while min_res < min_resolution_any_dim:
        # Voxel extent of the requested physical box, independent of the
        # image direction (extraction is always axis-aligned in index
        # space).
        size_extract = np.ceil(size_volume*radius/spacing_im).astype(int)
        # Continuous index of the box center via the proper inverse
        # direction transform, then step back the same physical
        # half-extent the original code used. For an identity direction
        # this collapses to round((point - origin - (size_volume/2)*r) /
        # spacing), i.e. it reproduces the legacy result exactly.
        center_cidx = (d_inv @ (point - origin_im)) / spacing_im
        half_extent_idx = (size_volume / 2.0) * radius / spacing_im
        index_extract = np.rint(center_cidx - half_extent_idx).astype(int)
        radius *= 1.05
        min_res = size_extract.min()
        print(f"Subvolume resolution: {size_extract}, Radius: {radius:.3f}")

    end_bounds = index_extract+size_extract

    for i, ind in enumerate(np.logical_and(end_bounds > size_im,
                                           (end_bounds - size_im)
                                           < 1/3 * size_extract)):
        if ind:
            print('\nsub-volume outside global volume, correcting\n')
            size_extract[i] = size_im[i] - index_extract[i]
            border = True

    for i, ind in enumerate(np.logical_and(index_extract < np.zeros(3),
                                           (np.zeros(3) - index_extract)
                                           < 1/3 * size_extract)):
        if ind:
            print('\nsub-volume outside global volume, correcting\n')
            index_extract[i] = 0
            border = True

    return size_extract.tolist(), index_extract.tolist(), border


def import_image(image_dir_or_volume):
    """
    Load image metadata and a handle suitable for :func:`extract_volume`.

    Parameters
    ----------
    image_dir_or_volume : str or sitk.Image
        Path to a volume on disk, or an in-memory ``sitk.Image`` (same geometry
        used for cropping and assembly).
    """
    if isinstance(image_dir_or_volume, sitk.Image):
        reader_im = image_dir_or_volume
        origin_im = np.array(list(reader_im.GetOrigin()))
        size_im = np.array(list(reader_im.GetSize()))
        spacing_im = np.array(list(reader_im.GetSpacing()))
        return reader_im, origin_im, size_im, spacing_im

    reader_im = read_image(image_dir_or_volume)
    origin_im = np.array(list(reader_im.GetOrigin()))
    size_im = np.array(list(reader_im.GetSize()))
    spacing_im = np.array(list(reader_im.GetSpacing()))

    return reader_im, origin_im, size_im, spacing_im


def sitk_to_numpy(Image):
    """
    Function to convert sitk image to numpy array
    args:
        Image: sitk image
    return:
        np_array: numpy array
    """
    np_array = sitk.GetArrayFromImage(Image)

    return np_array


def numpy_to_sitk(numpy, file_reader=None):
    """
    Function to convert numpy array to sitk image
    args:
        numpy: numpy array
        file_reader: sitk image reader
    return:
        Image: sitk image
    """

    Image = sitk.GetImageFromArray(numpy)

    if file_reader:

        Image.SetSpacing(file_reader.GetSpacing())
        Image.SetOrigin(file_reader.GetOrigin())
        Image.SetDirection(file_reader.GetDirection())

    return Image


def distance_map_from_seg(sitk_img):
    """
    Function to calculate distance map
    Inside is negative, outside is positive

    args:
        sitk_img: sitk image
    return:
        distance_map: sitk image
    """
    distance_map = sitk.SignedMaurerDistanceMap(sitk_img,
                                                squaredDistance=False,
                                                useImageSpacing=True)
    return distance_map


def check_seg_border(size_extract, index_extract,
                     predicted_vessel, size_im):
    """
    This function takes in the size and index of the extracted volume,
    its segmentation and the size of the global image

    This function is only called if the subvolume is on the global border
    This function returns true if there is at least one vessel voxel on the
    border of the global volume

    This function:

    1. Finds the face(s) of the subvolume that are on the border of
       the global volume
    2. Extracts the part of the subvolume segmentation that corresponds
       to those faces
    3. Checks if there is a vessel voxel in that part of the segmentation
    4. Returns true if there is a vessel voxel on the border
       of the global volume

    args:
        size_extract: list of ints, size of the extracted volume
        index_extract: list of ints, index of the lower corner for extraction
        predicted_vessel: sitk volume, segmentation of the extracted volume
        size_im: list of ints, size of the global volume
    return:
        border: boolean, if there exists a vessel voxel on the border
                of the global volume
    """

    # Find the faces of the subvolume that are on the border
    # of the global volume
    faces = []
    for i in range(3):
        if index_extract[i] == 0:
            faces.append(i)
        if index_extract[i] + size_extract[i] == size_im[i]:
            faces.append(i+3)

    # Extract the part of the subvolume segmentation that corresponds
    # to those faces
    seg_np = sitk_to_numpy(predicted_vessel).transpose(2, 1, 0)
    border = False
    for face in faces:
        if face == 0:
            border = np.any(seg_np[0, :, :])
        elif face == 1:
            border = np.any(seg_np[:, 0, :])
        elif face == 2:
            border = np.any(seg_np[:, :, 0])
        elif face == 3:
            border = np.any(seg_np[-1, :, :])
        elif face == 4:
            border = np.any(seg_np[:, -1, :])
        elif face == 5:
            border = np.any(seg_np[:, :, -1])

    return border


def is_point_in_image(assembly_image, location):

    index = assembly_image.TransformPhysicalPointToIndex(location.tolist())

    try:
        vessel_value = assembly_image[index]
        # print("Vessel value is: " + str(vessel_value))
        is_inside = vessel_value > 0.5
    except Exception as e:
        print(e)
        is_inside = False

    return is_inside
