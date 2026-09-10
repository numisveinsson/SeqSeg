"""Training-data preparation and nnU-Net training wrappers."""

from __future__ import annotations

import os
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Union

from seqseg.user_paths import apply_nnunet_env, ensure_nnunet_dirs, resolve_path

_VOLUME_EXTS = (".nii.gz", ".nrrd", ".nii", ".mha", ".mhd")


class TrainDependencyError(RuntimeError):
    """Raised when optional training dependencies are missing."""


def _require_sampler():
    try:
        from vascular_segment_sampler.nnunet import write_nnunet_dataset
        from vascular_segment_sampler.sampling import extract_patches
    except ImportError as e:
        raise TrainDependencyError(
            "vascular-segment-sampler is required for training-data preparation.\n"
            "Install with:\n"
            '  pip install "seqseg[train]"\n'
            "or:\n"
            "  pip install vascular-segment-sampler"
        ) from e
    _harden_sampler_vtk()
    return extract_patches, write_nnunet_dataset


def _harden_sampler_vtk() -> None:
    """Skip non-numeric VTK arrays the sampler used to pass to vtk_to_numpy.

    Centerline .vtp files from SimVascular/Slicer often include string/abstract
    arrays. ``GetArray(i)`` is None for those, which became
    ``'NoneType' object has no attribute 'GetDataType'``.
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


def _load_sampler_config(config: Union[str, dict]) -> dict:
    if isinstance(config, dict):
        return dict(config)
    try:
        from vascular_segment_sampler.sampling.extract import _load_config

        return dict(_load_config(config))
    except Exception:  # noqa: BLE001 — tests and missing sampler YAML
        return {"IMG_EXT": ".nrrd"}


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


def _missing_patches_error(outdir: str, modalities: Sequence[str]) -> RuntimeError:
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


def _assert_patches_extracted(outdir: str, modalities: Sequence[str]) -> None:
    for mod in modalities:
        img_dir = os.path.join(outdir, f"{mod.lower()}_train")
        mask_dir = os.path.join(outdir, f"{mod.lower()}_train_masks")
        if _count_volume_files(img_dir) == 0 or _count_volume_files(mask_dir) == 0:
            raise _missing_patches_error(outdir, modalities)


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


def expected_nnunet_dataset_name(name: str, dataset_number: int, modality: str) -> str:
    """Match vascular-segment-sampler naming for DatasetXXX_* folders."""
    if dataset_number < 10:
        dataset_number_str = "0" + str(dataset_number)
    else:
        dataset_number_str = str(dataset_number)
    return f"Dataset0{dataset_number_str}_{name}{modality.upper()}"


@dataclass
class PrepareResult:
    """Outputs from ``prepare_training_dataset``."""

    extracted_dir: str
    dataset_dirs: List[str]
    dataset_names: List[str]
    modalities: List[str]


def prepare_training_dataset(
    data_dir: Optional[str] = None,
    outdir: Optional[str] = None,
    *,
    name: str,
    dataset_number: int,
    modality: str = "CT",
    config: Union[str, dict] = "global",
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
    yes: bool = False,
    verbose: bool = False,
    img_ext: Optional[str] = None,
) -> PrepareResult:
    """
    Extract SeqSeg-style patches and convert them to nnU-Net raw datasets.

    Uses ``vascular-segment-sampler`` (``pip install seqseg[train]``).
    """
    extract_patches, write_nnunet_dataset = _require_sampler()

    resolved_data = resolve_path("data_dir", data_dir)
    resolved_out = resolve_path("outdir", outdir) or "./extracted_data/"
    if not resolved_data:
        raise ValueError(
            "data_dir is required. Pass --data-dir or set it with:\n"
            "  seqseg paths set --data-dir /path/to/cases"
        )
    data_dir = resolved_data
    outdir = _dir_for_sampler(resolved_out)
    os.makedirs(outdir, exist_ok=True)

    if nnunet_raw is None:
        nnunet_raw = resolve_path("nnunet_raw")
    if nnunet_raw:
        ensure_nnunet_dirs({"nnunet_raw": nnunet_raw})

    modalities = _parse_modalities(modality)
    modality_arg = ",".join(modalities)

    sampler_config = _load_sampler_config(config)
    resolved_img_ext = _resolve_img_ext(data_dir, sampler_config, img_ext)
    sampler_config["IMG_EXT"] = resolved_img_ext

    if not skip_sample:
        _preflight_training_data(
            data_dir,
            img_ext=resolved_img_ext,
            truth_from_surface=truth_from_surface,
        )
        print("=" * 72)
        print("Extracting vascular segment patches")
        print(f"  data_dir: {data_dir}")
        print(f"  outdir:   {outdir}")
        print(f"  IMG_EXT:  {resolved_img_ext}")
        print("=" * 72)
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
                list(truth_target_spacing) if truth_target_spacing is not None else None
            ),
            truth_regenerate=truth_regenerate,
            yes=yes,
            verbose=verbose,
        )
        _assert_patches_extracted(outdir, modalities)
    else:
        print("Skipping patch extraction (--skip-sample)")

    dataset_dirs: List[str] = []
    dataset_names: List[str] = []

    if not skip_convert:
        convert_outdir = (
            os.path.abspath(os.path.expanduser(nnunet_raw))
            if nnunet_raw
            else outdir
        )
        os.makedirs(convert_outdir, exist_ok=True)

        print("=" * 72)
        print(f"Converting patches to nnU-Net format under {convert_outdir}")
        print("=" * 72)

        # Dataset numbers must be unique per modality when converting several.
        for i, mod in enumerate(modalities):
            ds_num = dataset_number + i
            ds_path = write_nnunet_dataset(
                indir=outdir,
                name=name,
                dataset_number=ds_num,
                modality=mod.lower(),
                outdir=convert_outdir,
                also_test=also_test,
            )
            ds_name = expected_nnunet_dataset_name(name, ds_num, mod)
            dataset_dirs.append(os.path.abspath(ds_path))
            dataset_names.append(ds_name)
            print(f"  {mod}: {ds_path}")
    else:
        print("Skipping nnU-Net conversion (--skip-convert)")

    return PrepareResult(
        extracted_dir=outdir,
        dataset_dirs=dataset_dirs,
        dataset_names=dataset_names,
        modalities=modalities,
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
    """Prefer PATH; fall back to python -m for common nnU-Net entry points."""
    found = shutil.which(exe)
    if found:
        return found
    # nnUNetv2 installs console scripts; if missing, try module form where available.
    return exe


def run_nnunet_training(
    dataset_id: int,
    *,
    configuration: str = "3d_fullres",
    fold: str = "0",
    skip_plan: bool = False,
    plan_only: bool = False,
    np: Optional[int] = None,
    trainer: str = "nnUNetTrainer",
    plans: str = "nnUNetPlans",
) -> None:
    """
    Run ``nnUNetv2_plan_and_preprocess`` and/or ``nnUNetv2_train``.

    Requires ``nnUNet_raw``, ``nnUNet_preprocessed``, and ``nnUNet_results``.
    """
    env = _nnunet_env_or_raise()

    if not skip_plan:
        plan_cmd = [
            _resolve_nnunet_cli("nnUNetv2_plan_and_preprocess"),
            "-d",
            str(dataset_id),
            "-c",
            configuration,
        ]
        if np is not None:
            plan_cmd.extend(["-np", str(np)])
        print("=" * 72)
        print("Running nnU-Net plan and preprocess")
        print(" ", " ".join(plan_cmd))
        print("=" * 72)
        subprocess.run(plan_cmd, check=True, env=env)

    if plan_only:
        print("Plan/preprocess only; skipping training (--plan-only).")
        return

    train_cmd = [
        _resolve_nnunet_cli("nnUNetv2_train"),
        str(dataset_id),
        configuration,
        str(fold),
        "-tr",
        trainer,
        "-p",
        plans,
    ]
    print("=" * 72)
    print("Running nnU-Net training")
    print(" ", " ".join(train_cmd))
    print("=" * 72)
    subprocess.run(train_cmd, check=True, env=env)

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
    """Parse ``Dataset010_FOO`` / ``Dataset0999_BAR`` → integer id."""
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
