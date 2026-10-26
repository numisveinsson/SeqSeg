"""Training-data preparation and nnU-Net training wrappers."""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Sequence, Union

import numpy as np

from seqseg.modules.nnunet import select_training_device
from seqseg.user_paths import apply_nnunet_env, ensure_nnunet_dirs, resolve_path

_VOLUME_EXTS = (".nii.gz", ".nrrd", ".nii", ".mha", ".mhd")
_STAGING_DIR_NAME = "_seqseg_extracted"


class TrainDependencyError(RuntimeError):
    """Raised when training dependencies are missing."""


def _require_sampler():
    try:
        from vascular_segment_sampler.nnunet import write_nnunet_dataset
        from vascular_segment_sampler.sampling import extract_patches
    except ImportError as e:
        raise TrainDependencyError(
            "vascular-segment-sampler is required for training-data preparation.\n"
            "It is installed with SeqSeg. Reinstall with:\n"
            "  pip install -U seqseg\n"
            "or:\n"
            "  pip install vascular-segment-sampler"
        ) from e
    _harden_sampler_vtk()
    return extract_patches, write_nnunet_dataset


def _require_gather_global_volumes():
    try:
        from vascular_segment_sampler.sampling import gather_global_volumes
    except ImportError as e:
        raise TrainDependencyError(
            "Whole-volume dataset construction requires gather_global_volumes "
            "from a recent vascular-segment-sampler.\n"
            "Upgrade with:\n"
            "  pip install -U seqseg\n"
            "or:\n"
            "  pip install -U vascular-segment-sampler"
        ) from e
    return gather_global_volumes


def _reduction_or_nan(arr: Any, fn: Callable[[Any], Any]) -> float:
    """``np.amin``/``mean`` on an empty crop raise; treat that as missing."""
    arr = np.asarray(arr)
    if arr.size == 0:
        return float("nan")
    return float(fn(arr))


def _point_list(location: Any) -> list:
    if hasattr(location, "tolist"):
        return location.tolist()
    return list(location)


def _safe_add_image_stats(stats: dict, im_np: Any) -> dict:
    stats.update(
        {
            "IM_MEAN": _reduction_or_nan(im_np, np.mean),
            "IM_MIN": _reduction_or_nan(im_np, np.amin),
            "IM_STD": _reduction_or_nan(im_np, np.std),
            "IM_MAX": _reduction_or_nan(im_np, np.amax),
        }
    )
    return stats


def _safe_add_local_stats(
    stats: dict,
    location: Any,
    diff_cent: Any,
    blood_np: Any,
    ground_truth: Any,
    means: Sequence,
    removed_seg: Any,
    im_np: Any,
    O: int,
) -> tuple[dict, int]:
    """Same stats as the sampler, but empty labeled crops become NaN not errors."""
    stats.update(
        {
            "DIFF_CENT": diff_cent,
            "POINT_CENT": _point_list(location),
            "BLOOD_MEAN": _reduction_or_nan(blood_np, np.mean),
            "BLOOD_MIN": _reduction_or_nan(blood_np, np.amin),
            "BLOOD_STD": _reduction_or_nan(blood_np, np.std),
            "BLOOD_MAX": _reduction_or_nan(blood_np, np.amax),
            "GT_MEAN": _reduction_or_nan(ground_truth, np.mean),
            "GT_STD": _reduction_or_nan(ground_truth, np.std),
            "GT_MAX": _reduction_or_nan(ground_truth, np.amax),
            "GT_MIN": _reduction_or_nan(ground_truth, np.amin),
        }
    )
    if len(means) != 1:
        import SimpleITK as sitk

        larg_np = sitk.GetArrayFromImage(removed_seg)
        rem_np = np.asarray(im_np)[np.asarray(larg_np) > 0.1]
        stats.update(
            {
                "LARGEST_MEAN": _reduction_or_nan(rem_np, np.mean),
                "LARGEST_STD": _reduction_or_nan(rem_np, np.std),
                "LARGEST_MAX": _reduction_or_nan(rem_np, np.amax),
                "LARGEST_MIN": _reduction_or_nan(rem_np, np.amin),
            }
        )
        O += 1
    return stats, O


def _harden_sampler_vtk() -> None:
    """Patch sampler helpers that break on common SeqSeg datasets.

    Centerline .vtp files from SimVascular/Slicer often include string/abstract
    arrays. ``GetArray(i)`` is None for those, which became
    ``'NoneType' object has no attribute 'GetDataType'``.

    Crops with no labeled voxels used to raise
    ``zero-size array to reduction operation minimum`` in ``add_local_stats``.
    """
    try:
        import vascular_segment_sampler.sampling.extract as extract_mod
        import vascular_segment_sampler.sampling.functions as samp_fn
        import vascular_segment_sampler.vtk_functions as vf
        from seqseg.modules.vtk_functions import (
            collect_arrays,
            convertPolyDataToImageData,
        )
    except ImportError:
        return

    vf.collect_arrays = collect_arrays
    samp_fn.collect_arrays = collect_arrays
    vf.convertPolyDataToImageData = convertPolyDataToImageData

    if not getattr(samp_fn.add_local_stats, "_seqseg_hardened", False):
        _safe_add_local_stats._seqseg_hardened = True  # type: ignore[attr-defined]
        samp_fn.add_local_stats = _safe_add_local_stats
    if not getattr(samp_fn.add_image_stats, "_seqseg_hardened", False):
        _safe_add_image_stats._seqseg_hardened = True  # type: ignore[attr-defined]
        samp_fn.add_image_stats = _safe_add_image_stats

    orig_sort = getattr(samp_fn, "sort_centerline", None)
    if orig_sort is not None and not getattr(orig_sort, "_seqseg_hardened", False):

        def _sort_centerline(centerline, *args, **kwargs):
            if centerline is None or centerline.GetNumberOfPoints() == 0:
                raise RuntimeError(
                    "Centerline has no points. Each case needs a non-empty .vtp "
                    "in centerlines/ (VMTK/SimVascular)."
                )
            pts = centerline.GetPoints()
            if pts is None or pts.GetData() is None:
                raise RuntimeError(
                    "Centerline vtkPoints are empty. Check centerlines/*.vtp."
                )
            try:
                return orig_sort(centerline, *args, **kwargs)
            except KeyError as e:
                key = str(e).strip("'\"")
                if key in ("f", "MaximumInscribedSphereRadius", "Radius"):
                    raise RuntimeError(
                        "Centerline has no MaximumInscribedSphereRadius (or "
                        "Radius) point-data array. Export VMTK/SimVascular "
                        "centerlines."
                    ) from e
                raise

        _sort_centerline._seqseg_hardened = True
        samp_fn.sort_centerline = _sort_centerline

    extract_mod.sort_centerline = samp_fn.sort_centerline


def _parse_modalities(modality: Optional[str]) -> List[str]:
    if not modality:
        return ["CT"]
    return [m.strip().upper() for m in str(modality).split(",") if m.strip()]


def _dir_for_sampler(path: str) -> str:
    """Return an absolute directory path with a trailing separator.

    ``vascular-segment-sampler`` concatenates ``outdir`` with names such as
    ``ct_train_Sample_stats.csv`` instead of using ``os.path.join``. A
    resolved path like ``/scratch/.../seqseg_train`` otherwise becomes
    ``.../seqseg_trainct_train_Sample_stats.csv``.
    """
    return os.path.join(os.path.abspath(os.path.expanduser(path)), "")


def _is_missing_stats_csv(exc: BaseException) -> bool:
    return isinstance(exc, FileNotFoundError) and "Sample_stats.csv" in str(exc)


def _normalize_img_ext(ext: str) -> str:
    ext = str(ext).strip()
    if not ext.startswith("."):
        ext = "." + ext
    return ext


def _volume_ext_of(name: str) -> Optional[str]:
    lower = name.lower()
    for ext in _VOLUME_EXTS:
        if lower.endswith(ext):
            return ext
    return None


def _detect_img_ext(images_dir: Path) -> Optional[str]:
    """Pick the most common volume suffix under ``images/`` (``.nii.gz`` first)."""
    if not images_dir.is_dir():
        return None
    counts: dict[str, int] = {}
    for name in os.listdir(images_dir):
        if name.startswith("."):
            continue
        ext = _volume_ext_of(name)
        if ext:
            counts[ext] = counts.get(ext, 0) + 1
    if not counts:
        return None
    best = max(counts.values())
    for ext in _VOLUME_EXTS:
        if counts.get(ext) == best:
            return ext
    return None


_SAMPLER_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "sampler"
_SAMPLER_DEFAULT_CONFIG = _SAMPLER_CONFIG_DIR / "default_mm.yaml"
_SAMPLER_EMPTY_LIST_KEYS = ("TEST_CASES", "BAD_CASES")


def _sampler_config_dirs() -> List[str]:
    """Directories searched for sampler YAML overlays.

    ``./config`` in the working directory, then SeqSeg's bundled
    ``seqseg/config/sampler``. The configs shipped inside
    ``vascular-segment-sampler`` are never used.
    """
    dirs: List[str] = []
    cwd_cfg = os.path.join(os.getcwd(), "config")
    if os.path.isdir(cwd_cfg):
        dirs.append(os.path.abspath(cwd_cfg))
    bundled = str(_SAMPLER_CONFIG_DIR)
    if os.path.isdir(bundled) and bundled not in dirs:
        dirs.append(bundled)
    return dirs


def _yaml_defines(path: str, key: str) -> bool:
    try:
        import yaml
    except ImportError:
        return False
    try:
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception:  # noqa: BLE001 — unreadable overlay is not a match
        return False
    return isinstance(data, dict) and data.get(key) is not None


def _pick_config_file(directory: str, name: str) -> Optional[str]:
    """Choose ``name.yml`` / ``name.yaml`` inside one directory.

    A bare name prefers the file that sets ``MAX_SAMPLES``. ``config/test.yaml``
    and ``config/test.yml`` both exist; only the ``.yml`` smoke overlay sets
    the cap, and ``--config-name test`` has to select that file.
    """
    yml = os.path.join(directory, name + ".yml")
    yaml_path = os.path.join(directory, name + ".yaml")
    yml_ok = os.path.isfile(yml)
    yaml_ok = os.path.isfile(yaml_path)
    if yml_ok and yaml_ok:
        yml_cap = _yaml_defines(yml, "MAX_SAMPLES")
        yaml_cap = _yaml_defines(yaml_path, "MAX_SAMPLES")
        if yml_cap and not yaml_cap:
            return os.path.abspath(yml)
        if yaml_cap and not yml_cap:
            return os.path.abspath(yaml_path)
        return os.path.abspath(yaml_path)
    if yml_ok:
        return os.path.abspath(yml)
    if yaml_ok:
        return os.path.abspath(yaml_path)
    return None


def _resolve_sampler_config_file(config: str) -> Optional[str]:
    """Return an existing sampler YAML path, or None."""
    raw = str(config)
    if os.path.isfile(raw):
        return os.path.abspath(raw)
    base = os.path.basename(raw)
    if base.endswith((".yaml", ".yml")):
        names = [base]
        bare = None
    else:
        names = []
        bare = base
    for directory in _sampler_config_dirs():
        if bare is not None:
            picked = _pick_config_file(directory, bare)
            if picked:
                return picked
        for name in names:
            path = os.path.join(directory, name)
            if os.path.isfile(path):
                return os.path.abspath(path)
    return None


def _load_sampler_config(config: Union[str, dict]) -> dict:
    if isinstance(config, dict):
        return dict(config)
    path = _resolve_sampler_config_file(str(config))
    if path is None:
        looked = ", ".join(_sampler_config_dirs()) or "(none)"
        raise FileNotFoundError(
            f"Sampler config {config!r} was not found.\n"
            f"  looked in: {looked}\n"
            "Pass a YAML path, or a name such as default_mm, default_cm, or test."
        )
    defaults = _read_sampler_yaml(str(_SAMPLER_DEFAULT_CONFIG))
    if os.path.abspath(path) == str(_SAMPLER_DEFAULT_CONFIG):
        merged = defaults
    else:
        merged = {**defaults, **_read_sampler_yaml(path)}
    for key in _SAMPLER_EMPTY_LIST_KEYS:
        if merged.get(key) is None:
            merged[key] = []
    return merged


def _read_sampler_yaml(path: str) -> dict:
    import yaml

    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Sampler config {path!r} must parse to a mapping")
    return dict(data)


def _cap_from_sampler_config(
    sampler_config: dict,
    max_samples: Optional[float],
) -> Optional[float]:
    """CLI ``--max-samples`` wins; otherwise use YAML ``MAX_SAMPLES``."""
    if max_samples is not None:
        return max_samples
    configured = sampler_config.get("MAX_SAMPLES")
    if configured is None:
        return None
    return configured


def _resolve_img_ext(
    data_dir: str,
    config: dict,
    img_ext: Optional[str] = None,
) -> str:
    if img_ext:
        return _normalize_img_ext(img_ext)
    detected = _detect_img_ext(Path(data_dir) / "images")
    if detected:
        return detected
    return _normalize_img_ext(str(config.get("IMG_EXT") or ".nrrd"))


def _stems_with_suffix(directory: Path, suffix: str) -> List[str]:
    if not directory.is_dir():
        return []
    stems: List[str] = []
    for name in os.listdir(directory):
        if name.startswith("."):
            continue
        if name.endswith(suffix):
            stems.append(name[: -len(suffix)])
    return sorted(stems)


def _count_volume_files(directory: str) -> int:
    if not os.path.isdir(directory):
        return 0
    n = 0
    for name in os.listdir(directory):
        if name.startswith("."):
            continue
        if name.endswith(".nii.gz") or name.endswith(_VOLUME_EXTS[1:]):
            n += 1
    return n


def _preflight_training_data(
    data_dir: str,
    *,
    img_ext: str,
    truth_from_surface: bool,
) -> None:
    """Fail fast when the sampler would iterate zero usable cases."""
    root = Path(data_dir)
    images = root / "images"
    centerlines = root / "centerlines"
    missing = [
        str(p)
        for p, label in ((images, "images"), (centerlines, "centerlines"))
        if not p.is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            "Training data_dir must contain images/ and centerlines/.\n"
            f"  data_dir: {root}\n"
            "  missing: " + ", ".join(missing)
        )

    img_stems = _stems_with_suffix(images, img_ext)
    if not img_stems:
        present = sorted(
            {
                name
                for name in os.listdir(images)
                if not name.startswith(".")
            }
        )
        preview = ", ".join(present[:8]) if present else "(empty)"
        raise FileNotFoundError(
            f"No images with IMG_EXT={img_ext!r} in {images}.\n"
            f"  files found: {preview}\n"
            "Pass --img-ext (e.g. .nii.gz or .mha) or convert images."
        )

    cent_stems = _stems_with_suffix(centerlines, ".vtp")
    if not cent_stems:
        raise FileNotFoundError(
            f"No .vtp centerlines in {centerlines}."
        )

    overlap = sorted(set(img_stems) & set(cent_stems))
    if not overlap:
        raise FileNotFoundError(
            "No case has both an image and a matching centerline stem.\n"
            f"  image stems ({img_ext}): {img_stems[:8]}\n"
            f"  centerline stems (.vtp): {cent_stems[:8]}"
        )

    if truth_from_surface:
        surfaces = root / "surfaces"
        if not surfaces.is_dir():
            raise FileNotFoundError(
                "--truth-from-surface requires a surfaces/ folder under "
                f"{root}."
            )
        surf_stems = _stems_with_suffix(surfaces, ".vtp") + _stems_with_suffix(
            surfaces, ".stl"
        )
        if not surf_stems:
            raise FileNotFoundError(
                f"No .vtp/.stl surfaces in {surfaces}."
            )
        surf_overlap = sorted(set(overlap) & set(surf_stems))
        if not surf_overlap:
            raise FileNotFoundError(
                "No case has an image, centerline, and surface with the same stem.\n"
                f"  usable image+centerline stems: {overlap[:8]}\n"
                f"  surface stems: {sorted(set(surf_stems))[:8]}"
            )


def _missing_volumes_error(
    outdir: str,
    modalities: Sequence[str],
    *,
    global_volumes: bool = False,
) -> RuntimeError:
    if global_volumes:
        lines = [
            "Whole-volume gathering finished without writing any cases.",
            f"  outdir: {os.path.abspath(outdir)}",
        ]
    else:
        lines = [
            "Patch extraction finished without writing any samples.",
            f"  outdir: {os.path.abspath(outdir)}",
        ]
    for mod in modalities:
        img_dir = os.path.join(outdir, f"{mod.lower()}_train")
        mask_dir = os.path.join(outdir, f"{mod.lower()}_train_masks")
        lines.append(
            f"  {mod}: {_count_volume_files(img_dir)} images in {img_dir}, "
            f"{_count_volume_files(mask_dir)} masks in {mask_dir}"
        )
    if global_volumes:
        lines.extend(
            [
                "",
                "Typical causes:",
                "  - Image extension mismatch (pass --img-ext, e.g. .nii.gz).",
                "  - Missing truths/ (or surfaces/ when using --truth-from-surface).",
                "  - All cases listed in outdir/done.txt, so they were skipped.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "vascular-segment-sampler always opens "
                "{modality}_train_Sample_stats.csv at the end, even when no case "
                "completed (so that FileNotFoundError is a symptom, not the cause).",
                "Typical causes:",
                "  - Worker crashes with --num-cores > 1 (exceptions are not re-raised).",
                "    Re-run with --num-cores 1 to see the real error.",
                "  - Image extension mismatch (pass --img-ext, e.g. .nii.gz).",
                "  - Missing truths/ (or surfaces/ when using --truth-from-surface).",
                "  - All cases listed in outdir/done.txt, so they were skipped.",
            ]
        )
    return RuntimeError("\n".join(lines))


def _assert_volumes_written(
    outdir: str,
    modalities: Sequence[str],
    *,
    global_volumes: bool = False,
) -> None:
    for mod in modalities:
        img_dir = os.path.join(outdir, f"{mod.lower()}_train")
        mask_dir = os.path.join(outdir, f"{mod.lower()}_train_masks")
        if _count_volume_files(img_dir) == 0 or _count_volume_files(mask_dir) == 0:
            raise _missing_volumes_error(
                outdir, modalities, global_volumes=global_volumes
            )


@contextmanager
def _tolerate_missing_sampler_stats() -> Iterator[None]:
    """Skip the sampler's end-of-run CSV summary when no case wrote it."""
    try:
        import vascular_segment_sampler.sampling.extract as extract_mod
    except ImportError:
        yield
        return

    orig = getattr(extract_mod, "print_csv_stats", None)
    if orig is None:
        yield
        return

    def _safe(out_dir, global_config, modality):
        suffix = global_config.get("OUTPUT_SUFFIX", "") or ""
        if global_config.get("TESTING"):
            csv_file = f"{modality}_test{suffix}_Sample_stats.csv"
        else:
            csv_file = f"{modality}_train{suffix}_Sample_stats.csv"
        path = out_dir + csv_file
        if not os.path.isfile(path):
            print(
                f"Warning: no sampler stats file at {path}; continuing."
            )
            return
        orig(out_dir, global_config, modality)

    extract_mod.print_csv_stats = _safe
    try:
        yield
    finally:
        extract_mod.print_csv_stats = orig


def _run_extract_patches(extract_patches, **kwargs) -> None:
    try:
        with _tolerate_missing_sampler_stats():
            extract_patches(**kwargs)
    except FileNotFoundError as e:
        if not _is_missing_stats_csv(e):
            raise
        print(f"Warning: {e}")


def _link_or_copy(src: str, dst: str, copy_fn: Callable, *args, **kwargs) -> None:
    """Hardlink ``src`` to ``dst``; fall back to ``copy_fn`` across filesystems."""
    try:
        os.link(src, dst)
    except OSError:
        copy_fn(src, dst, *args, **kwargs)


@contextmanager
def _prefer_hardlinks_for_nnunet_copy() -> Iterator[None]:
    """Make sampler dataset conversion link files instead of copying them."""
    try:
        import vascular_segment_sampler.nnunet.convert as convert_mod
    except ImportError:
        yield
        return

    orig = convert_mod.shutil.copy

    def _copy(src, dst, *args, **kwargs):
        _link_or_copy(src, dst, orig, *args, **kwargs)

    convert_mod.shutil.copy = _copy
    try:
        yield
    finally:
        convert_mod.shutil.copy = orig


def _resolve_training_dirs(
    outdir: Optional[str],
    nnunet_raw: Optional[str],
    *,
    name: str,
    dataset_number: int,
    modality: str,
) -> tuple[str, str, bool]:
    """Return ``(staging_dir, nnunet_raw, managed_staging)``.

    Staging is ``nnUNet_raw/DatasetXXX_*/_seqseg_extracted`` unless ``outdir``
    was passed. A passed ``outdir`` is never treated as disposable.
    ``modality`` is the first modality; a multi-modality prepare shares that
    folder.
    """
    raw = nnunet_raw if nnunet_raw else resolve_path("nnunet_raw")
    if not raw:
        raise ValueError(
            "nnUNet_raw is required. Pass --nnunet-raw or set it once with:\n"
            "  seqseg paths init --nnunet-root ~/nnunet_data"
        )
    raw_abs = os.path.abspath(os.path.expanduser(raw))
    explicit = outdir is not None and str(outdir) != ""
    if explicit:
        staging = str(outdir)
        managed = False
    else:
        dataset_name = expected_nnunet_dataset_name(name, dataset_number, modality)
        staging = os.path.join(raw_abs, dataset_name, _STAGING_DIR_NAME)
        managed = True
    return _dir_for_sampler(staging), raw_abs, managed


def _staging_is_safe_to_remove(staging: str, raw: str) -> bool:
    """True when deleting ``staging`` cannot remove ``raw``."""
    staging_norm = os.path.normpath(staging)
    raw_norm = os.path.normpath(raw)
    if staging_norm == os.path.abspath(os.sep) or staging_norm == raw_norm:
        return False
    try:
        common = os.path.commonpath([staging_norm, raw_norm])
    except ValueError:
        return True
    return common != staging_norm


def _read_done_cases(outdir: str) -> set[str]:
    """Case names the sampler has already finished in this staging folder."""
    path = os.path.join(os.path.normpath(outdir), "done.txt")
    if not os.path.isfile(path):
        return set()
    with open(path, encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def _local_training_cases(
    data_dir: str,
    img_ext: str,
    *,
    testing: bool,
    test_cases: Sequence[str],
) -> List[str]:
    """Image/centerline stems, matching the sampler's ``other`` dataset list."""
    images = Path(data_dir) / "images"
    if testing:
        img_stems = set(_stems_with_suffix(images, img_ext))
        return sorted(case for case in test_cases if case in img_stems)
    cent_stems = set(_stems_with_suffix(Path(data_dir) / "centerlines", ".vtp"))
    held_out = set(test_cases)
    return [
        case
        for case in _stems_with_suffix(images, img_ext)
        if case in cent_stems and case not in held_out
    ]


def _select_cases(
    cases: Sequence[str],
    *,
    perc_dataset: float,
    start_from: int,
    end_at: int,
) -> List[str]:
    """Apply the sampler's seeded shuffle, then perc and start/end slices."""
    chosen = sorted(cases)
    random.Random(42).shuffle(chosen)
    chosen = chosen[: int(perc_dataset * len(chosen))]
    if end_at != -1:
        return chosen[start_from:end_at]
    return chosen[start_from:]


def _cases_this_run(
    data_dir: str,
    sampler_config: dict,
    modality: str,
    *,
    img_ext: str,
    perc_dataset: float,
    start_from: int,
    end_at: int,
    testing: bool,
) -> List[str]:
    dataset_name = str(sampler_config.get("DATASET_NAME") or "other")
    test_cases = list(sampler_config.get("TEST_CASES") or [])
    if dataset_name == "vmr":
        from vascular_segment_sampler.datasets import create_dataset

        cfg = dict(sampler_config)
        cfg["DATA_DIR"] = data_dir
        cfg["IMG_EXT"] = img_ext
        cfg.setdefault("TESTING", testing)
        cfg.setdefault("TEST_CASES", [])
        cfg.setdefault("BAD_CASES", [])
        cases = list(create_dataset(cfg, modality))
    else:
        cases = _local_training_cases(
            data_dir,
            img_ext,
            testing=testing,
            test_cases=test_cases,
        )
    return _select_cases(
        cases,
        perc_dataset=perc_dataset,
        start_from=start_from,
        end_at=end_at,
    )


def _extraction_already_complete(
    outdir: str,
    data_dir: str,
    sampler_config: dict,
    modalities: Sequence[str],
    *,
    img_ext: str,
    perc_dataset: float,
    start_from: int,
    end_at: int,
    testing: bool,
) -> bool:
    """True when ``done.txt`` already lists every case this run would sample."""
    done = _read_done_cases(outdir)
    if not done:
        return False
    for mod in modalities:
        selected = _cases_this_run(
            data_dir,
            sampler_config,
            mod,
            img_ext=img_ext,
            perc_dataset=perc_dataset,
            start_from=start_from,
            end_at=end_at,
            testing=testing,
        )
        if not selected or any(case not in done for case in selected):
            return False
    return True


def _staging_has_volumes(outdir: str, modalities: Sequence[str]) -> bool:
    root = os.path.normpath(outdir)
    for mod in modalities:
        kind = mod.lower()
        for split in ("train", "val", "test"):
            if _count_volume_files(os.path.join(root, f"{kind}_{split}")):
                return True
            if _count_volume_files(os.path.join(root, f"{kind}_{split}_masks")):
                return True
    return False


def _raw_dataset_has_volumes(
    nnunet_raw: str,
    name: str,
    dataset_number: int,
    modalities: Sequence[str],
) -> bool:
    for i, mod in enumerate(modalities):
        dataset = os.path.join(
            nnunet_raw,
            expected_nnunet_dataset_name(name, dataset_number + i, mod),
        )
        if _count_volume_files(os.path.join(dataset, "imagesTr")) == 0:
            return False
        if _count_volume_files(os.path.join(dataset, "labelsTr")) == 0:
            return False
    return True


def _existing_dataset_records(
    nnunet_raw: str,
    name: str,
    dataset_number: int,
    modalities: Sequence[str],
) -> tuple[List[str], List[str]]:
    dirs: List[str] = []
    names: List[str] = []
    for i, mod in enumerate(modalities):
        ds_name = expected_nnunet_dataset_name(name, dataset_number + i, mod)
        dirs.append(os.path.abspath(os.path.join(nnunet_raw, ds_name)))
        names.append(ds_name)
    return dirs, names


def _prune_extracted_volumes(staging: str) -> None:
    """Delete subvolume directories and leave ``.txt`` / ``.csv`` metadata."""
    root = os.path.normpath(staging)
    if not os.path.isdir(root):
        return
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif _volume_ext_of(name):
            os.remove(path)
    print(
        f"Removed extracted subvolumes at {root}; "
        "kept .txt and .csv metadata."
    )


def expected_nnunet_dataset_name(name: str, dataset_number: int, modality: str) -> str:
    """nnU-Net folder name: ``Dataset`` + id (at least 3 digits) + name + modality.

    nnU-Net finds a dataset by the prefix ``Dataset{id:03d}``. The sampler
    writes ``Dataset0`` plus the id, which matches for ids 1–99 and not for
    100 and above (``999`` must be ``Dataset999``, not ``Dataset0999``).
    """
    return f"Dataset{int(dataset_number):03d}_{name}{modality.upper()}"


def _align_dataset_dir_with_nnunet(
    written: str, expected_name: str, nnunet_raw: str
) -> str:
    """Rename a sampler folder so nnU-Net's id lookup finds it."""
    written_abs = os.path.abspath(written)
    expected_abs = os.path.join(os.path.abspath(nnunet_raw), expected_name)
    if os.path.normpath(written_abs) == os.path.normpath(expected_abs):
        return written_abs
    if not os.path.isdir(written_abs):
        return written_abs
    if os.path.exists(expected_abs):
        raise FileExistsError(
            f"Cannot rename {written_abs} to {expected_abs}; destination exists."
        )
    os.rename(written_abs, expected_abs)
    print(
        f"  Renamed {os.path.basename(written_abs)} to {expected_name} "
        "so nnU-Net can find this dataset id."
    )
    return expected_abs


@dataclass
class PrepareResult:
    """Outputs from ``prepare_training_dataset``."""

    extracted_dir: str
    dataset_dirs: List[str]
    dataset_names: List[str]
    modalities: List[str]
    removed_staging: bool = False
    preprocessed: bool = False


def prepare_training_dataset(
    data_dir: Optional[str] = None,
    outdir: Optional[str] = None,
    *,
    name: Optional[str] = None,
    dataset_number: int,
    modality: str = "CT",
    config: Union[str, dict] = "default_mm",
    nnunet_raw: Optional[str] = None,
    perc_dataset: float = 1.0,
    num_cores: int = 1,
    start_from: int = 0,
    end_at: int = -1,
    testing: bool = False,
    validation_prop: Optional[float] = None,
    max_samples: Optional[float] = None,
    truth_from_surface: bool = False,
    truth_target_spacing: Optional[Sequence[float]] = None,
    truth_regenerate: bool = False,
    skip_sample: bool = False,
    skip_convert: bool = False,
    also_test: bool = False,
    verbose: bool = False,
    img_ext: Optional[str] = None,
    global_volumes: bool = False,
    keep_extracted: bool = False,
    configuration: str = "3d_fullres",
    plans: str = "nnUNetPlans",
    np: Optional[int] = None,
    preprocess_only: bool = False,
) -> PrepareResult:
    """
    Prepare training volumes, convert them to nnU-Net raw datasets, and preprocess.

    By default extracts SeqSeg-style patches. With ``global_volumes=True``,
    copies each case's full image and label instead (``gather_global_volumes``).

    Extracted files are written under
    ``nnUNet_raw/DatasetXXX_*/_seqseg_extracted`` unless ``outdir`` is set,
    then hardlinked into ``nnUNet_raw``. After a successful conversion the
    subvolume directories are deleted and ``done.txt`` plus ``.txt`` / ``.csv``
    metadata stay, unless ``keep_extracted`` is set or ``outdir`` was passed
    explicitly. A later run skips extraction when ``done.txt`` already lists
    every case, and skips conversion when those subvolumes are already gone
    and ``imagesTr`` / ``labelsTr`` exist. ``nnUNetv2_plan_and_preprocess``
    then runs for ``configuration`` (default ``3d_fullres``).

    ``preprocess_only=True`` uses the dataset already in ``nnUNet_raw``. It
    does not sample or convert. When the plans file is missing it runs
    ``nnUNetv2_plan_and_preprocess``. When that file exists it resamples
    ``configuration`` with ``nnUNetv2_preprocess``.

    Uses ``vascular-segment-sampler``, which is installed with SeqSeg.
    Sampler prompts are accepted automatically.
    """
    if preprocess_only:
        return _preprocess_existing_dataset(
            dataset_number,
            configuration=configuration,
            plans=plans,
            np=np,
            num_cores=num_cores,
            nnunet_raw=nnunet_raw,
        )
    if not name:
        raise ValueError(
            "--name is required unless --preprocess-only is set."
        )
    extract_patches, write_nnunet_dataset = _require_sampler()

    resolved_data = resolve_path("data_dir", data_dir)
    if not resolved_data:
        raise ValueError(
            "data_dir is required. Pass --data-dir or set it with:\n"
            "  seqseg paths set --nnunet-root ~/nnunet_data --data-dir /path/to/cases"
        )
    data_dir = resolved_data
    config_path = (
        None if isinstance(config, dict) else _resolve_sampler_config_file(str(config))
    )
    sampler_config = _load_sampler_config(config)
    resolved_img_ext = _resolve_img_ext(data_dir, sampler_config, img_ext)
    sampler_config["IMG_EXT"] = resolved_img_ext
    max_samples = _cap_from_sampler_config(sampler_config, max_samples)

    modalities = _parse_modalities(modality)
    modality_arg = ",".join(modalities)
    outdir, nnunet_raw, managed_staging = _resolve_training_dirs(
        outdir,
        nnunet_raw,
        name=name,
        dataset_number=dataset_number,
        modality=modalities[0],
    )
    os.makedirs(outdir, exist_ok=True)
    ensure_nnunet_dirs({"nnunet_raw": nnunet_raw})

    extraction_complete = False
    if not skip_sample:
        extraction_complete = _extraction_already_complete(
            outdir,
            data_dir,
            sampler_config,
            modalities,
            img_ext=resolved_img_ext,
            perc_dataset=perc_dataset,
            start_from=start_from,
            end_at=end_at,
            testing=testing,
        )
    volumes_present = _staging_has_volumes(outdir, modalities)
    reuse_converted = extraction_complete and not volumes_present

    if skip_sample:
        print(
            "Skipping volume gathering (--skip-sample)"
            if global_volumes
            else "Skipping patch extraction (--skip-sample)"
        )
    elif reuse_converted:
        if not _raw_dataset_has_volumes(
            nnunet_raw, name, dataset_number, modalities
        ):
            raise RuntimeError(
                "done.txt lists every case for this run, and the extracted "
                "subvolumes are already gone. nnUNet_raw is missing "
                "imagesTr/labelsTr volumes.\n"
                f"  staging: {os.path.normpath(outdir)}\n"
                "Delete done.txt in that folder and run prepare again to extract."
            )
        print(
            "All cases are listed in done.txt and nnUNet_raw already has "
            "the volumes. Skipping extraction and conversion."
        )
    elif extraction_complete:
        print(
            "All cases are listed in done.txt. Skipping extraction; "
            "converting volumes still in the staging folder."
        )
    else:
        _preflight_training_data(
            data_dir,
            img_ext=resolved_img_ext,
            truth_from_surface=truth_from_surface,
        )
        print("=" * 72)
        if global_volumes:
            print("Gathering whole-case volumes (no patch sampling)")
        else:
            print("Extracting vascular segment patches")
        print(f"  data_dir: {data_dir}")
        print(f"  outdir:   {outdir}")
        if config_path:
            print(f"  config:   {config_path}")
        if max_samples is not None and not global_volumes:
            print(f"  max_samples: {max_samples}")
        if managed_staging and not keep_extracted:
            print(
                "  (subvolumes removed after conversion; "
                ".txt and .csv metadata kept)"
            )
        print(f"  IMG_EXT:  {resolved_img_ext}")
        print("=" * 72)
        if global_volumes:
            gather_global_volumes = _require_gather_global_volumes()
            gather_global_volumes(
                data_dir=data_dir,
                outdir=outdir,
                config=sampler_config,
                perc_dataset=perc_dataset,
                num_cores=num_cores,
                start_from=start_from,
                end_at=end_at,
                testing=testing,
                validation_prop=validation_prop,
                modality=modality_arg,
                truth_from_surface=truth_from_surface,
                truth_target_spacing=(
                    list(truth_target_spacing)
                    if truth_target_spacing is not None
                    else None
                ),
                truth_regenerate=truth_regenerate,
                yes=True,
                verbose=verbose,
            )
        else:
            _run_extract_patches(
                extract_patches,
                data_dir=data_dir,
                outdir=outdir,
                config=sampler_config,
                perc_dataset=perc_dataset,
                num_cores=num_cores,
                start_from=start_from,
                end_at=end_at,
                testing=testing,
                validation_prop=validation_prop,
                max_samples=max_samples,
                modality=modality_arg,
                truth_from_surface=truth_from_surface,
                truth_target_spacing=(
                    list(truth_target_spacing)
                    if truth_target_spacing is not None
                    else None
                ),
                truth_regenerate=truth_regenerate,
                yes=True,
                verbose=verbose,
            )
        _assert_volumes_written(
            outdir, modalities, global_volumes=global_volumes
        )

    dataset_dirs: List[str] = []
    dataset_names: List[str] = []
    removed_staging = False

    if skip_convert:
        print("Skipping nnU-Net conversion (--skip-convert)")
    elif reuse_converted:
        dataset_dirs, dataset_names = _existing_dataset_records(
            nnunet_raw, name, dataset_number, modalities
        )
    else:
        os.makedirs(nnunet_raw, exist_ok=True)

        print("=" * 72)
        print(
            f"Converting {'volumes' if global_volumes else 'patches'} "
            f"to nnU-Net format under {nnunet_raw}"
        )
        print("=" * 72)

        # Dataset numbers must be unique per modality when converting several.
        with _prefer_hardlinks_for_nnunet_copy():
            for i, mod in enumerate(modalities):
                ds_num = dataset_number + i
                ds_path = write_nnunet_dataset(
                    indir=outdir,
                    name=name,
                    dataset_number=ds_num,
                    modality=mod.lower(),
                    outdir=nnunet_raw,
                    also_test=also_test,
                )
                ds_name = expected_nnunet_dataset_name(name, ds_num, mod)
                ds_path = _align_dataset_dir_with_nnunet(ds_path, ds_name, nnunet_raw)
                dataset_dirs.append(os.path.abspath(ds_path))
                dataset_names.append(ds_name)
                print(f"  {mod}: {ds_path}")

        if managed_staging and not keep_extracted:
            staging_path = os.path.normpath(outdir)
            if _staging_is_safe_to_remove(staging_path, nnunet_raw):
                _prune_extracted_volumes(staging_path)
                removed_staging = True
            else:
                print(
                    f"Keeping extracts at {staging_path}; "
                    "removing them would also remove nnUNet_raw."
                )

    preprocessed = False
    if not skip_convert:
        workers = num_cores if np is None else np
        dataset_ids = [dataset_id_from_name(ds_name) for ds_name in dataset_names]
        _run_nnunet_plan_and_preprocess(
            nnunet_raw,
            dataset_ids,
            configuration=configuration,
            np=workers,
            plans=plans,
        )
        preprocessed = True

    return PrepareResult(
        extracted_dir=outdir,
        dataset_dirs=dataset_dirs,
        dataset_names=dataset_names,
        modalities=modalities,
        removed_staging=removed_staging,
        preprocessed=preprocessed,
    )


def _nnunet_env_or_raise() -> dict:
    """Return env with nnU-Net paths from env vars and/or ``seqseg paths``."""
    env = os.environ.copy()
    apply_nnunet_env(env, create_dirs=True)
    missing = [
        key
        for key in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")
        if not env.get(key)
    ]
    if missing:
        raise RuntimeError(
            "nnU-Net paths are not set: "
            + ", ".join(missing)
            + "\nSet them once with:\n"
            "  seqseg paths init\n"
            "  # or: seqseg paths set --nnunet-root ~/nnunet_data\n"
            "Or export:\n"
            "  export nnUNet_raw=/path/to/nnUNet_raw\n"
            "  export nnUNet_preprocessed=/path/to/nnUNet_preprocessed\n"
            "  export nnUNet_results=/path/to/nnUNet_results"
        )
    return env


def _resolve_nnunet_cli(exe: str) -> str:
    """Prefer PATH, then the directory that contains the running Python.

    A venv installs ``nnUNetv2_train`` next to ``python`` even
    when that bin directory is not on ``PATH``.
    """
    found = shutil.which(exe)
    if found:
        return found
    sibling = os.path.join(os.path.dirname(sys.executable), exe)
    for candidate in (sibling, sibling + ".exe"):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return exe


def _remove_nnunet_dataset_copies(env: dict, dataset_id: int) -> None:
    """Delete raw and preprocessed folders for ``dataset_id``. Leave results."""
    removed = False
    for key in ("nnUNet_raw", "nnUNet_preprocessed"):
        root = env.get(key)
        if not root or not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if not os.path.isdir(path):
                continue
            try:
                found_id = dataset_id_from_name(name)
            except ValueError:
                continue
            if found_id != dataset_id:
                continue
            print(f"Removing {path}")
            shutil.rmtree(path)
            removed = True
    if not removed:
        print(
            "No nnUNet_raw or nnUNet_preprocessed folder matched "
            f"dataset id {dataset_id}."
        )


def _nnunet_env_for_raw(nnunet_raw: str) -> dict:
    """Env for nnU-Net commands, with ``nnUNet_raw`` set to ``nnunet_raw``."""
    env = _nnunet_env_or_raise()
    env["nnUNet_raw"] = os.path.abspath(nnunet_raw)
    return env


def _dataset_dirs_matching_id(root: str, dataset_id: int) -> List[str]:
    """Dataset folders under ``root`` whose name encodes ``dataset_id``."""
    if not root or not os.path.isdir(root):
        return []
    found: List[str] = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        try:
            found_id = dataset_id_from_name(name)
        except ValueError:
            continue
        if found_id == dataset_id:
            found.append(path)
    return found


def _one_dataset_dir(root: str, dataset_id: int, label: str) -> str:
    matches = _dataset_dirs_matching_id(root, dataset_id)
    if not matches:
        raise FileNotFoundError(
            f"No {label} folder for dataset id {dataset_id} under {root}."
        )
    if len(matches) > 1:
        names = ", ".join(os.path.basename(path) for path in matches)
        raise RuntimeError(
            f"Multiple {label} folders match dataset id {dataset_id}: {names}"
        )
    return matches[0]


_NNUNET_CLI = (sys.executable, "-m", "seqseg.pipeline.nnunet_cli")


def _plan_and_preprocess_command(
    dataset_id: int,
    configuration: str,
    np: int,
    plans: str,
) -> List[str]:
    """``nnUNetv2_plan_and_preprocess`` argv, through ``seqseg.pipeline.nnunet_cli``.

    ``plans`` is the plans file name (training ``-p``), not the planner class.
    The default name ``nnUNetPlans`` is nnU-Net's own default, so it is omitted.
    """
    cmd = [
        *_NNUNET_CLI,
        "plan_and_preprocess",
        "-d",
        str(dataset_id),
        "-c",
        configuration,
        "-np",
        str(np),
    ]
    if plans != "nnUNetPlans":
        cmd.extend(["-overwrite_plans_name", plans])
    return cmd


def _preprocess_only_command(
    dataset_id: int,
    configuration: str,
    np: int,
    plans: str,
) -> List[str]:
    """``nnUNetv2_preprocess`` argv for one configuration of existing plans."""
    return [
        *_NNUNET_CLI,
        "preprocess",
        "-d",
        str(dataset_id),
        "-plans_name",
        plans,
        "-c",
        configuration,
        "-np",
        str(np),
    ]


def _run_nnunet_cli(cmd: List[str], env: dict, title: str) -> None:
    print("=" * 72)
    print(title)
    print(" ", " ".join(cmd))
    print("=" * 72)
    subprocess.run(cmd, check=True, env=env)


def _run_nnunet_plan_and_preprocess(
    nnunet_raw: str,
    dataset_ids: Sequence[int],
    *,
    configuration: str,
    np: int,
    plans: str,
) -> None:
    """Plan and preprocess each dataset id that prepare just wrote."""
    if not dataset_ids:
        return
    env = _nnunet_env_for_raw(nnunet_raw)
    for dataset_id in dataset_ids:
        _run_nnunet_cli(
            _plan_and_preprocess_command(dataset_id, configuration, np, plans),
            env,
            "Running nnU-Net plan and preprocess",
        )


def _run_nnunet_preprocess_only(
    nnunet_raw: str,
    dataset_id: int,
    *,
    configuration: str,
    np: int,
    plans: str,
) -> None:
    """Resample one configuration from plans that already exist."""
    env = _nnunet_env_for_raw(nnunet_raw)
    _run_nnunet_cli(
        _preprocess_only_command(dataset_id, configuration, np, plans),
        env,
        f"Running nnU-Net preprocess for {configuration}",
    )


def _resolve_nnunet_raw(nnunet_raw: Optional[str]) -> str:
    raw = nnunet_raw if nnunet_raw else resolve_path("nnunet_raw")
    if not raw:
        raise ValueError(
            "nnUNet_raw is required. Pass --nnunet-raw or set it once with:\n"
            "  seqseg paths init --nnunet-root ~/nnunet_data"
        )
    return os.path.abspath(os.path.expanduser(raw))


def _preprocess_existing_dataset(
    dataset_number: int,
    *,
    configuration: str,
    plans: str,
    np: Optional[int],
    num_cores: int,
    nnunet_raw: Optional[str],
) -> PrepareResult:
    """Plan or preprocess ``configuration`` from an existing raw dataset.

    Returns before the sampler is imported. Patch extraction and conversion
    do not run, and ``_seqseg_extracted`` is not required. A missing plans
    file runs ``nnUNetv2_plan_and_preprocess``. An existing plans file
    resamples ``configuration`` with ``nnUNetv2_preprocess``.
    """
    raw_root = _resolve_nnunet_raw(nnunet_raw)
    prepare_hint = (
        "Run a full prepare before --preprocess-only:\n"
        f"  seqseg train prepare --name NAME --dataset-id {dataset_number}"
    )
    try:
        raw_dir = _one_dataset_dir(raw_root, dataset_number, "nnUNet_raw")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{exc}\n{prepare_hint}") from exc
    env = _nnunet_env_for_raw(raw_root)
    pre_root = env.get("nnUNet_preprocessed") or ""
    pre_matches = _dataset_dirs_matching_id(pre_root, dataset_number)
    if len(pre_matches) > 1:
        names = ", ".join(os.path.basename(path) for path in pre_matches)
        raise RuntimeError(
            f"Multiple nnUNet_preprocessed folders match dataset id "
            f"{dataset_number}: {names}"
        )
    workers = num_cores if np is None else np
    plans_file = (
        os.path.join(pre_matches[0], f"{plans}.json") if pre_matches else ""
    )
    if plans_file and os.path.isfile(plans_file):
        _run_nnunet_preprocess_only(
            raw_root,
            dataset_number,
            configuration=configuration,
            np=workers,
            plans=plans,
        )
    else:
        print(
            "No plans file for this dataset. "
            "Planning and preprocessing the existing raw dataset."
        )
        _run_nnunet_plan_and_preprocess(
            raw_root,
            [dataset_number],
            configuration=configuration,
            np=workers,
            plans=plans,
        )
    dataset_name = os.path.basename(raw_dir)
    return PrepareResult(
        extracted_dir=raw_dir,
        dataset_dirs=[raw_dir],
        dataset_names=[dataset_name],
        modalities=[],
        removed_staging=False,
        preprocessed=True,
    )


def _require_preprocessed_dataset(env: dict, dataset_id: int) -> str:
    root = env.get("nnUNet_preprocessed") or ""
    matches = _dataset_dirs_matching_id(root, dataset_id)
    if not matches:
        raise FileNotFoundError(
            f"No preprocessed data for dataset id {dataset_id}"
            + (f" under {root}." if root else ".")
            + "\nseqseg train nnunet only trains. Preprocessing runs in prepare:\n"
            f"  seqseg train prepare --name NAME --dataset-id {dataset_id}"
        )
    if len(matches) > 1:
        names = ", ".join(os.path.basename(path) for path in matches)
        raise RuntimeError(
            f"Multiple nnUNet_preprocessed folders match dataset id {dataset_id}: "
            f"{names}"
        )
    return matches[0]


def _require_continue_checkpoint(
    env: dict,
    dataset_id: int,
    *,
    trainer: str,
    plans: str,
    configuration: str,
    fold: str,
) -> None:
    root = env.get("nnUNet_results") or ""
    try:
        dataset_dir = _one_dataset_dir(root, dataset_id, "nnUNet_results")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No training output for dataset id {dataset_id} to continue.\n"
            "Train this fold once without --c."
        ) from exc
    checkpoint = os.path.join(
        dataset_dir,
        f"{trainer}__{plans}__{configuration}",
        f"fold_{fold}",
        "checkpoint_latest.pth",
    )
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            f"No checkpoint to continue at {checkpoint}.\n"
            "Train this fold once without --c, or pass the fold that has "
            "checkpoint_latest.pth."
        )


def _configuration_n_stages(plans_file: str, configuration: str) -> Optional[int]:
    """Number of encoder stages nnU-Net planned for ``configuration``."""
    if not os.path.isfile(plans_file):
        return None
    with open(plans_file, "r", encoding="utf-8") as f:
        configurations = json.load(f).get("configurations", {})
    merged: dict = {}
    name: Optional[str] = configuration
    while name is not None and name in configurations:
        merged = {**configurations[name], **merged}
        name = configurations[name].get("inherits_from")
    n_stages = merged.get("architecture", {}).get("arch_kwargs", {}).get("n_stages")
    if n_stages is None and "pool_op_kernel_sizes" in merged:
        n_stages = len(merged["pool_op_kernel_sizes"])
    return n_stages


def default_training_trainer(
    preprocessed_dir: str, plans: str, configuration: str
) -> str:
    """``nnUNetTrainer``, or ``nnUNetTrainerNoDeepSupervision`` for shallow plans.

    With two stages or fewer, nnU-Net's deep supervision zeroes the weight
    of its only output and then normalizes by that zero, so every loss is NaN.
    """
    plans_file = os.path.join(preprocessed_dir, f"{plans}.json")
    n_stages = _configuration_n_stages(plans_file, configuration)
    if n_stages is not None and n_stages <= 2:
        return "nnUNetTrainerNoDeepSupervision"
    return "nnUNetTrainer"


def run_nnunet_training(
    dataset_id: int,
    *,
    configuration: str = "3d_fullres",
    fold: str = "0",
    trainer: Optional[str] = None,
    plans: str = "nnUNetPlans",
    cleanup: bool = False,
    continue_training: bool = False,
) -> None:
    """
    Run ``nnUNetv2_train``.

    Planning and preprocessing happen in ``prepare_training_dataset``.
    Training is started with ``-device`` set from what is available:
    CUDA, then MPS, then CPU.

    ``continue_training=True`` passes nnU-Net's ``--c`` and, for a single
    fold, requires ``checkpoint_latest.pth``. ``--fold all`` leaves that
    check to nnU-Net.

    Requires ``nnUNet_raw``, ``nnUNet_preprocessed``, and ``nnUNet_results``.
    Preprocessed data for ``dataset_id`` must already exist.

    With ``cleanup=True``, after training succeeds, delete that dataset's
    folders under ``nnUNet_raw`` and ``nnUNet_preprocessed``. ``nnUNet_results``
    is kept. Cleanup is skipped when training fails. Cleanup removes the
    preprocessed data a later ``--c`` run would need.

    Without ``trainer``, ``default_training_trainer`` picks it from the plans.
    """
    env = _nnunet_env_or_raise()
    preprocessed_dir = _require_preprocessed_dataset(env, dataset_id)
    if trainer is None:
        trainer = default_training_trainer(preprocessed_dir, plans, configuration)
        if trainer != "nnUNetTrainer":
            print(
                f"The {configuration} plan downsamples at most once, so deep "
                f"supervision would make the loss NaN. Training with {trainer}."
            )
    if continue_training and str(fold) != "all":
        _require_continue_checkpoint(
            env,
            dataset_id,
            trainer=trainer,
            plans=plans,
            configuration=configuration,
            fold=str(fold),
        )

    device = select_training_device()
    train_cmd = [
        _resolve_nnunet_cli("nnUNetv2_train"),
        str(dataset_id),
        configuration,
        str(fold),
        "-tr",
        trainer,
        "-p",
        plans,
        "-device",
        device,
    ]
    if continue_training:
        train_cmd.append("--c")
    print("=" * 72)
    print("Running nnU-Net training")
    print(f"  device: {device}")
    print(" ", " ".join(train_cmd))
    print("=" * 72)
    subprocess.run(train_cmd, check=True, env=env)

    if cleanup:
        _remove_nnunet_dataset_copies(env, dataset_id)

    results = env["nnUNet_results"]
    # Folder layout used by SeqSeg NnUNetModelSpec.model_folder()
    print(
        "\nTraining finished (or launched). Point SeqSeg at weights with:\n"
        f"  -nnunet_results_path {results} \\\n"
        f"  -train_dataset DatasetXXX_YOURNAME \\\n"
        f"  -nnunet_type {configuration} \\\n"
        f"  -fold {fold}\n"
    )


def dataset_id_from_name(dataset_name: str) -> int:
    """Parse ``Dataset010_FOO`` / ``Dataset999_BAR`` → integer id."""
    stem = Path(dataset_name).name
    if not stem.startswith("Dataset"):
        raise ValueError(f"Not an nnU-Net dataset name: {dataset_name!r}")
    rest = stem[len("Dataset") :]
    digits = []
    for ch in rest:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    if not digits:
        raise ValueError(f"Could not parse dataset id from {dataset_name!r}")
    return int("".join(digits))
