"""Training-data preparation and nnU-Net training wrappers."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

from seqseg.user_paths import apply_nnunet_env, ensure_nnunet_dirs, resolve_path


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
    return extract_patches, write_nnunet_dataset


def _parse_modalities(modality: Optional[str]) -> List[str]:
    if not modality:
        return ["CT"]
    return [m.strip().upper() for m in str(modality).split(",") if m.strip()]


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
    outdir = resolved_out
    os.makedirs(outdir, exist_ok=True)

    if nnunet_raw is None:
        nnunet_raw = resolve_path("nnunet_raw")
    if nnunet_raw:
        ensure_nnunet_dirs({"nnunet_raw": nnunet_raw})

    modalities = _parse_modalities(modality)
    modality_arg = ",".join(modalities)

    if not skip_sample:
        print("=" * 72)
        print("Extracting vascular segment patches")
        print("=" * 72)
        extract_patches(
            data_dir=data_dir,
            outdir=outdir,
            config=config,
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
