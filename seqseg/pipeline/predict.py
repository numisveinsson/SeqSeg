"""Whole-volume nnU-Net prediction for one image or a folder of images."""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import SimpleITK as sitk

from seqseg.modules.nnunet import initialize_predictor
from seqseg.modules.sweep import predict_volume

# ``.nii.gz`` is first so a gzipped NIfTI name is not treated as ``.nii``.
_VOLUME_EXTS = (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd")


def _normalize_img_ext(ext: str) -> str:
    ext = str(ext).strip().lower()
    if not ext.startswith("."):
        ext = "." + ext
    return ext


def _volume_ext_of(name: str, extra: Optional[str] = None) -> Optional[str]:
    lower = name.lower()
    candidates: Sequence[str] = (extra,) + _VOLUME_EXTS if extra else _VOLUME_EXTS
    for ext in candidates:
        if ext and lower.endswith(ext):
            return ext
    return None


def volume_stem(path: str, img_ext: Optional[str] = None) -> str:
    """Filename without a known volume suffix (``.nii.gz`` included)."""
    name = os.path.basename(path)
    extra = _normalize_img_ext(img_ext) if img_ext else None
    ext = _volume_ext_of(name, extra)
    if ext:
        return name[: -len(ext)]
    return os.path.splitext(name)[0]


def list_input_volumes(
    image: Optional[str] = None,
    image_dir: Optional[str] = None,
    img_ext: Optional[str] = None,
) -> List[str]:
    """One image path, or sorted volume files in a directory.

    ``img_ext`` limits a folder to that suffix. When it is omitted, files
    ending in ``.nii.gz``, ``.nii``, ``.mha``, ``.mhd``, or ``.nrrd`` are kept.
    """
    if bool(image) == bool(image_dir):
        raise ValueError("Provide exactly one of image or image_dir")
    if image:
        return [os.path.abspath(os.path.expanduser(image))]

    root = os.path.abspath(os.path.expanduser(image_dir))
    wanted = _normalize_img_ext(img_ext) if img_ext else None
    paths: List[str] = []
    for name in sorted(os.listdir(root)):
        if name.startswith("."):
            continue
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            continue
        if wanted:
            if name.lower().endswith(wanted):
                paths.append(path)
        elif _volume_ext_of(name):
            paths.append(path)
    if not paths:
        detail = f" with extension {wanted}" if wanted else ""
        raise FileNotFoundError(f"No volumes found in {root}{detail}")
    return paths


def run_global_prediction(
    *,
    model_folder: str,
    outdir: str,
    image: Optional[str] = None,
    image_dir: Optional[str] = None,
    img_ext: Optional[str] = None,
    fold: str = "all",
    scale: float = 1.0,
    force_cpu: bool = False,
) -> List[str]:
    """Predict each volume once and write binary and probability images.

    The nnU-Net predictor is loaded a single time. Each case writes
    ``{stem}_segmentation.mha`` and ``{stem}_probability.mha`` under ``outdir``.

    Returns the segmentation paths in input order.
    """
    paths = list_input_volumes(image=image, image_dir=image_dir, img_ext=img_ext)
    os.makedirs(outdir, exist_ok=True)
    predictor = initialize_predictor(model_folder, fold, force_cpu=force_cpu)
    written: List[str] = []
    for path in paths:
        pred, prob = predict_volume(predictor, path, scale=scale)
        stem = volume_stem(path, img_ext)
        seg_path = os.path.join(outdir, f"{stem}_segmentation.mha")
        prob_path = os.path.join(outdir, f"{stem}_probability.mha")
        sitk.WriteImage(pred, seg_path)
        sitk.WriteImage(prob, prob_path)
        print(seg_path)
        print(prob_path)
        written.append(seg_path)
    return written
