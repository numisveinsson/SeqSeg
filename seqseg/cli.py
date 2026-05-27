"""SeqSeg CLI: subcommands with legacy ``seqseg <classic-args>`` compatibility."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from typing import List, Optional, Sequence

import faulthandler
import numpy as np

from seqseg.config_models import AlgorithmConfig, NnUNetModelSpec, load_yaml_config
from seqseg.modules import sitk_functions as sf
from seqseg.modules.datasets import get_testing_samples
from seqseg.pipeline.classic import run_classic_batch
from seqseg.pipeline.single_trace import prepare_single_image_dataset
from seqseg.pipeline.post import (
    bootstrap_simvascular_project,
    bootstrap_simvascular_project_batch,
    run_global_centerline_batch,
    run_global_centerline_single,
)

_TOP_LEVEL_COMMANDS = frozenset(
    {
        "run",
        "post",
        "simvascular",
        "config",
        "doctor",
        "init",
        "-h",
        "--help",
        "--version",
    }
)


def _inject_legacy_run_batch(argv: List[str]) -> List[str]:
    if not argv:
        return argv
    if argv[0] not in _TOP_LEVEL_COMMANDS:
        return ["run", "batch", *argv]
    return argv


def _add_classic_trace_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "-data_dir",
        "--data_directory",
        type=str,
        help="Folder containing the testing data",
    )
    p.add_argument(
        "-nnunet_results_path",
        "--nnunet_results_path",
        type=str,
        help="Path to nnUNet results folder",
    )
    p.add_argument(
        "-nnunet_type",
        "--nnunet_type",
        choices=["3d_fullres", "2d"],
        default="3d_fullres",
        type=str,
        help="nnUNet model type",
    )
    p.add_argument(
        "-train_dataset",
        "--train_dataset",
        type=str,
        default="Dataset010_SEQCOROASOCACT",
        help="nnU-Net training dataset name",
    )
    p.add_argument(
        "-fold",
        "--fold",
        default="all",
        type=str,
        help="nnU-Net fold",
    )
    p.add_argument("-img_ext", "--img_ext", type=str, help="Image extension, eg .nii.gz")
    p.add_argument("-outdir", "--outdir", type=str, help="Output directory")
    p.add_argument(
        "-scale",
        "--scale",
        default=1,
        type=float,
        help="Scale image intensities for nnU-Net vs data units",
    )
    p.add_argument("-start", "--start", default=0, type=int, help="Batch start index")
    p.add_argument("-stop", "--stop", default=-1, type=int, help="Batch stop index (-1=all)")
    p.add_argument("-max_n_steps", "--max_n_steps", default=1000, type=int)
    p.add_argument("-max_n_steps_per_branch", "--max_n_steps_per_branch", default=100, type=int)
    p.add_argument("-max_n_branches", "--max_n_branches", default=100, type=int)
    p.add_argument("-unit", "--unit", default="cm", type=str)
    p.add_argument("-config_name", "--config_name", default="global", type=str)
    p.add_argument("-pt_centerline", "--pt_centerline", default=50, type=int)
    p.add_argument("-num_seeds_centerline", "--num_seeds_centerline", default=1, type=int)
    p.add_argument("-write_steps", "--write_steps", default=0, type=int)
    p.add_argument("-extract_global_centerline", "--extract_global_centerline", default=0, type=int)
    p.add_argument("-cap_surface_cent", "--cap_surface_cent", default=0, type=int)
    p.add_argument("-assembly_threshold", "--assembly_threshold", default=0.5, type=float)
    p.add_argument(
        "-resample_spacing",
        "--resample_spacing",
        nargs="+",
        type=float,
        default=None,
        metavar="S",
        help="Isotropic spacing or sx sy sz before SeqSeg",
    )


def _add_plus_trace_args(p: argparse.ArgumentParser) -> None:
    _add_classic_trace_args(p)
    p.set_defaults(max_n_steps_per_branch=10000)
    p.add_argument(
        "-seqseg_test_name",
        "--seqseg_test_name",
        default="3d_fullres",
        type=str,
        help="nnUNet test name for SeqSeg model",
    )
    p.add_argument(
        "-seqseg_train_dataset",
        "--seqseg_train_dataset",
        type=str,
        default="Dataset010_SEQCOROASOCACT",
    )
    p.add_argument("-seqseg_fold", "--seqseg_fold", default="all", type=str)
    p.add_argument("-seqseg_scale", "--seqseg_scale", default=1, type=float)
    p.add_argument("-global_test_name", "--global_test_name", default="3d_fullres", type=str)
    p.add_argument("-global_fold", "--global_fold", default="all", type=str)
    p.add_argument("-global_scale", "--global_scale", default=1, type=float)
    p.add_argument(
        "-global_train_dataset",
        "--global_train_dataset",
        type=str,
        default="Dataset012_COROASOCACT",
    )


def _seqseg_version() -> str:
    try:
        return version("seqseg")
    except PackageNotFoundError:
        return "0.0.0"


def _validate_trace_batch(ns: argparse.Namespace) -> None:
    missing = []
    if not ns.data_directory:
        missing.append("-data_dir / --data_directory")
    if not ns.outdir:
        missing.append("-outdir / --outdir")
    if not ns.img_ext:
        missing.append("-img_ext / --img_ext")
    if not ns.nnunet_results_path:
        missing.append("-nnunet_results_path / --nnunet_results_path")
    if missing:
        print(
            "seqseg run batch: missing required arguments:\n  "
            + "\n  ".join(missing)
            + "\n\nMinimal example:\n"
            "  seqseg run batch \\\n"
            "    -data_dir /path/to/dataset/ \\\n"
            "    -outdir /path/to/out/ \\\n"
            "    -img_ext .nii.gz \\\n"
            "    -nnunet_results_path /path/to/nnUNet_results \\\n"
            "    -train_dataset Dataset010_SEQCOROASOCACT\n",
            file=sys.stderr,
        )
        sys.exit(2)


def _cmd_trace_batch(ns: argparse.Namespace) -> None:
    _validate_trace_batch(ns)
    faulthandler.enable()
    t0 = time.time()
    global_config = AlgorithmConfig.from_name(ns.config_name)
    print(f"\nUsing config file: {ns.config_name}")

    dir_output0 = ns.outdir
    data_dir = ns.data_directory
    try:
        os.mkdir(dir_output0)
    except Exception as e:  # noqa: BLE001
        print(e)

    spec = NnUNetModelSpec(
        train_dataset=ns.train_dataset,
        nnunet_type=ns.nnunet_type,
        results_path=ns.nnunet_results_path,
        fold=ns.fold,
        scale=ns.scale,
    )
    dir_model_weights = spec.model_folder()

    testing_samples, directory_data = get_testing_samples(ns.train_dataset, data_dir)
    print("Testing samples about to run:")
    for sample in testing_samples:
        print(sample)

    stop = ns.stop
    if stop == -1:
        stop = len(testing_samples)
    print(f"Running from {ns.start} to {stop} of {len(testing_samples)} samples")

    resample_spacing = sf.parse_resample_spacing_arg(ns.resample_spacing)

    run_classic_batch(
        dir_output0=dir_output0,
        data_dir=data_dir,
        dataset=ns.train_dataset,
        dir_model_weights=dir_model_weights,
        fold=ns.fold,
        img_format=ns.img_ext,
        scale=ns.scale,
        test_name=ns.nnunet_type,
        pt_centerline=ns.pt_centerline,
        num_seeds=ns.num_seeds_centerline,
        testing_samples=testing_samples,
        directory_data=directory_data,
        start=ns.start,
        stop=stop,
        global_config=global_config,
        unit=ns.unit,
        max_step_size=ns.max_n_steps,
        max_n_branches=ns.max_n_branches,
        max_n_steps_per_branch=ns.max_n_steps_per_branch,
        write_samples=bool(ns.write_steps),
        take_time=global_config["TIME_ANALYSIS"],
        calc_global_centerline=bool(ns.extract_global_centerline),
        cap_surface_cent=bool(ns.cap_surface_cent),
        assembly_threshold=ns.assembly_threshold,
        resample_spacing=resample_spacing,
        start_time_global=t0,
    )
    print("\nTotal calculation time for all cases is: ")
    print(f"{((time.time() - t0) / 60):.2f} min\n")


def _cmd_trace_plus_batch(ns: argparse.Namespace) -> None:
    _validate_trace_batch(ns)
    faulthandler.enable()
    t0 = time.time()
    global_config = AlgorithmConfig.from_name(ns.config_name)
    print(f"Using config file: {ns.config_name}")

    dir_output0 = ns.outdir
    data_dir = ns.data_directory
    try:
        os.mkdir(dir_output0)
    except Exception as e:  # noqa: BLE001
        print(e)

    dir_model_weights_seqseg = NnUNetModelSpec(
        train_dataset=ns.seqseg_train_dataset,
        nnunet_type=ns.seqseg_test_name,
        results_path=ns.nnunet_results_path,
        fold=ns.seqseg_fold,
        scale=ns.seqseg_scale,
    ).model_folder()
    dir_model_weights_global = NnUNetModelSpec(
        train_dataset=ns.global_train_dataset,
        nnunet_type=ns.global_test_name,
        results_path=ns.nnunet_results_path,
        fold=ns.global_fold,
        scale=ns.global_scale,
    ).model_folder()
    print("Using nnUNet model weights from:")
    print(dir_model_weights_seqseg)
    print("Using nnUNet model weights for global segmentation from:")
    print(dir_model_weights_global)

    testing_samples, directory_data = get_testing_samples(ns.seqseg_train_dataset, data_dir)
    print("Testing samples about to run:")
    for sample in testing_samples:
        print(sample)

    stop = ns.stop
    if stop == -1:
        stop = len(testing_samples)

    resample_spacing = sf.parse_resample_spacing_arg(ns.resample_spacing)

    run_plus_batch(
        dir_output0=dir_output0,
        data_dir=data_dir,
        seqseg_dataset=ns.seqseg_train_dataset,
        dir_model_weights_seqseg=dir_model_weights_seqseg,
        seqseg_fold=ns.seqseg_fold,
        seqseg_test_name=ns.seqseg_test_name,
        seqseg_scale=ns.seqseg_scale,
        dir_model_weights_global=dir_model_weights_global,
        global_fold=ns.global_fold,
        global_scale=ns.global_scale,
        img_format=ns.img_ext,
        unit=ns.unit,
        max_step_size=ns.max_n_steps,
        max_n_branches=ns.max_n_branches,
        max_n_steps_per_branch=ns.max_n_steps_per_branch,
        write_samples=bool(ns.write_steps),
        take_time=global_config.get("TIME_ANALYSIS", False),
        calc_global_centerline=bool(ns.extract_global_centerline),
        cap_surface_cent=bool(ns.cap_surface_cent),
        assembly_threshold=ns.assembly_threshold,
        num_seeds=ns.num_seeds_centerline,
        resample_spacing=resample_spacing,
        global_config=global_config,
        start=ns.start,
        stop=stop,
        start_time_global=t0,
    )
    print("\nTotal calculation time is: ")
    print(f"{((time.time() - t0) / 60):.2f} min\n")


def _cmd_config_dump(ns: argparse.Namespace) -> None:
    data = load_yaml_config(ns.name)
    print(json.dumps(data, indent=2, sort_keys=True))


def _cmd_config_fingerprint(ns: argparse.Namespace) -> None:
    """Print keys whose values differ from a baseline packaged YAML."""
    try:
        cur = load_yaml_config(ns.name)
        base = load_yaml_config(ns.baseline)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)
    keys = sorted(set(cur) | set(base))
    diff = 0
    for k in keys:
        if cur.get(k) != base.get(k):
            diff += 1
            print(f"{k}:\n  {ns.name}: {cur.get(k)!r}\n  {ns.baseline}: {base.get(k)!r}\n")
    if diff == 0:
        print(f"No differences between {ns.name!r} and {ns.baseline!r}.")
    else:
        print(f"Total differing keys: {diff}")


def _cmd_init_dataset(ns: argparse.Namespace) -> None:
    root = os.path.abspath(ns.path)
    os.makedirs(os.path.join(root, "images"), exist_ok=True)
    os.makedirs(os.path.join(root, "centerlines"), exist_ok=True)
    os.makedirs(os.path.join(root, "truths"), exist_ok=True)
    seeds_path = os.path.join(root, "seeds.json")
    if not os.path.isfile(seeds_path) or ns.force:
        template = [{"name": "case001", "seeds": [], "cardiac_mesh": False}]
        with open(seeds_path, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=2)
        print(f"Wrote template {seeds_path}")
    else:
        print(f"Skipped existing {seeds_path} (use --force to overwrite)")
    print(
        "Next: add volumes under images/, optional centerlines/ and truths/, "
        "edit seeds.json, then run:\n"
        f"  seqseg run batch -data_dir {root}/ -outdir ... -img_ext .nii.gz "
        "-nnunet_results_path ...\n"
    )


def _cmd_trace_single(ns: argparse.Namespace) -> None:
    faulthandler.enable()
    t0 = time.time()
    if not ns.outdir or not ns.model_folder:
        print(
            "seqseg run single: --outdir and --model-folder are required.",
            file=sys.stderr,
        )
        sys.exit(2)
    if not ns.image:
        print("seqseg run single: --image is required.", file=sys.stderr)
        sys.exit(2)
    if ns.seeds_json:
        if not os.path.isfile(ns.seeds_json):
            print(f"--seeds-json not found: {ns.seeds_json}", file=sys.stderr)
            sys.exit(2)
    elif not ns.seed:
        print(
            "seqseg run single: provide --seeds-json or at least one "
            "--seed X Y Z R (repeat --seed for multiple).",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        os.makedirs(ns.outdir, exist_ok=True)
    except Exception as e:  # noqa: BLE001
        print(e)

    seeds_xyzr: Optional[List[List[float]]] = None
    if ns.seed:
        seeds_xyzr = [list(map(float, s)) for s in ns.seed]

    data_dir, img_ext, _case = prepare_single_image_dataset(
        ns.image,
        ns.outdir,
        seeds_xyzr=seeds_xyzr,
        tangent=tuple(ns.tangent),
        seed_step=ns.seed_step,
        seeds_json_path=ns.seeds_json,
    )

    global_config = AlgorithmConfig.from_name(ns.config_name)
    print(f"Using config file: {ns.config_name}")
    print(f"Staged single-case data_dir: {data_dir}")

    testing_samples, directory_data = get_testing_samples(ns.train_dataset, data_dir)
    print("Testing samples about to run:")
    for sample in testing_samples:
        print(sample)

    run_classic_batch(
        dir_output0=ns.outdir,
        data_dir=data_dir,
        dataset=ns.train_dataset,
        dir_model_weights=os.path.expanduser(ns.model_folder),
        fold=ns.fold,
        img_format=img_ext,
        scale=ns.scale,
        test_name=ns.nnunet_type,
        pt_centerline=ns.pt_centerline,
        num_seeds=ns.num_seeds_centerline,
        testing_samples=testing_samples,
        directory_data=directory_data,
        start=0,
        stop=len(testing_samples),
        global_config=global_config,
        unit=ns.unit,
        max_step_size=ns.max_n_steps,
        max_n_branches=ns.max_n_branches,
        max_n_steps_per_branch=ns.max_n_steps_per_branch,
        write_samples=bool(ns.write_steps),
        take_time=global_config["TIME_ANALYSIS"],
        calc_global_centerline=bool(ns.extract_global_centerline),
        cap_surface_cent=bool(ns.cap_surface_cent),
        assembly_threshold=ns.assembly_threshold,
        resample_spacing=None,
        start_time_global=t0,
    )
    print("\nTotal calculation time for run single: ")
    print(f"{((time.time() - t0) / 60):.2f} min\n")


def _cmd_doctor(ns: argparse.Namespace) -> None:
    import importlib

    print(f"seqseg version {_seqseg_version()}")
    print("SeqSeg environment check")
    for label, modname in (
        ("SimpleITK", "SimpleITK"),
        ("vtk", "vtk"),
        ("nnunetv2", "nnunetv2"),
        ("scipy", "scipy"),
    ):
        try:
            importlib.import_module(modname)
        except ImportError as e:
            print(f"  {label}: MISSING ({e})")
        else:
            print(f"  {label}: OK")

    for env_key in (
        "nnUNet_results",
        "nnUNet_raw",
        "nnUNet_preprocessed",
    ):
        val = os.environ.get(env_key)
        print(f"  {env_key}: {val if val else '(unset)'}")

    if getattr(ns, "model_folder", None):
        mf = os.path.expanduser(ns.model_folder)
        if os.path.isdir(mf):
            print(f"  --model-folder: OK ({mf})")
        else:
            print(f"  --model-folder: NOT FOUND ({mf})")


def _cmd_simvascular_init(ns: argparse.Namespace) -> None:
    bootstrap_simvascular_project(
        ns.case_dir,
        source_image=ns.source_image,
        case_id=ns.case_id,
    )


def _cmd_simvascular_init_batch(ns: argparse.Namespace) -> None:
    bootstrap_simvascular_project_batch(ns.parent_dir, case_glob=ns.case_glob)


def _cmd_post_gcl_single(ns: argparse.Namespace) -> None:
    gc: Optional[AlgorithmConfig] = None
    if ns.config_name:
        gc = AlgorithmConfig.from_name(ns.config_name)
    run_global_centerline_single(
        ns.seg,
        ns.out,
        seeds_json=ns.seeds_json,
        case_name=ns.case_name,
        merge_method=ns.merge_method,
        directory_data=ns.directory_data,
        unit=ns.unit,
        global_config=gc,
    )


def _cmd_post_gcl_batch(ns: argparse.Namespace) -> None:
    gc: Optional[AlgorithmConfig] = None
    if ns.config_name:
        gc = AlgorithmConfig.from_name(ns.config_name)
    run_global_centerline_batch(
        ns.seg_dir,
        seg_glob=ns.seg_glob,
        seeds_json=ns.seeds_json,
        out_dir=ns.out_dir,
        merge_method=ns.merge_method,
        directory_data=ns.directory_data,
        unit=ns.unit,
        global_config=gc,
    )


def _build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="SeqSeg: sequential vessel segmentation and centerline tracing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    root.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_seqseg_version()}",
    )
    sub = root.add_subparsers(dest="command", required=False)

    run_grp = sub.add_parser(
        "run",
        help="Run segmentation and vessel tracing (batch, plus, or single volume)",
    )
    run_sub = run_grp.add_subparsers(dest="run_cmd", required=True)

    p_batch = run_sub.add_parser(
        "batch",
        help="Classic SeqSeg (dataset / nnU-Net batch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_classic_trace_args(p_batch)
    p_batch.set_defaults(_handler=_cmd_trace_batch)

    p_plus = run_sub.add_parser(
        "plus",
        help="Global sweep then SeqSeg",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    plus_sub = p_plus.add_subparsers(dest="plus_cmd", required=True)
    p_plus_batch = plus_sub.add_parser(
        "batch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_plus_trace_args(p_plus_batch)
    p_plus_batch.set_defaults(_handler=_cmd_trace_plus_batch)

    p_single = run_sub.add_parser(
        "single",
        help="One image + seeds (stages data under <outdir>/_seqseg_single_staging/)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_single.add_argument("--image", required=True, type=str, help="Input volume path")
    p_single.add_argument("--outdir", required=True, type=str)
    p_single.add_argument(
        "--model-folder",
        required=True,
        type=str,
        help="nnU-Net trainer folder (…/nnUNetTrainer__nnUNetPlans__3d_fullres)",
    )
    p_single.add_argument("--fold", default="all", type=str)
    p_single.add_argument(
        "--nnunet-type",
        default="3d_fullres",
        choices=["3d_fullres", "2d"],
    )
    p_single.add_argument(
        "--train-dataset",
        default="Dataset010_SEQCOROASOCACT",
        type=str,
        help="Used for output naming (same as batch -train_dataset)",
    )
    p_single.add_argument("--config-name", default="global", type=str)
    p_single.add_argument("--scale", default=1.0, type=float)
    p_single.add_argument("--unit", default="cm", type=str)
    p_single.add_argument("--max-n-steps", default=1000, type=int)
    p_single.add_argument("--max-n-branches", default=100, type=int)
    p_single.add_argument("--max-n-steps-per-branch", default=100, type=int)
    p_single.add_argument("--pt-centerline", default=50, type=int)
    p_single.add_argument("--num-seeds-centerline", default=1, type=int)
    p_single.add_argument("--write-steps", default=0, type=int)
    p_single.add_argument("--extract-global-centerline", default=0, type=int)
    p_single.add_argument("--cap-surface-cent", default=0, type=int)
    p_single.add_argument("--assembly-threshold", default=0.5, type=float)
    p_single.add_argument(
        "--seed",
        action="append",
        nargs=4,
        type=float,
        metavar=("X", "Y", "Z", "R"),
        help="Repeat for multiple seeds (world coordinates + radius)",
    )
    p_single.add_argument(
        "--tangent",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 1.0],
        metavar=("TX", "TY", "TZ"),
        help="Used with --seed to place the paired old point",
    )
    p_single.add_argument(
        "--seed-step",
        default=1.0,
        type=float,
        help="Physical step along --tangent for old-point when using --seed",
    )
    p_single.add_argument(
        "--seeds-json",
        default=None,
        type=str,
        help="If set, copy this seeds.json; first case name must match image stem "
        "unless JSON is edited after staging",
    )
    p_single.set_defaults(_handler=_cmd_trace_single)

    post = sub.add_parser("post", help="Post-process existing outputs")
    post_sub = post.add_subparsers(dest="post_cmd", required=True)
    gcl = post_sub.add_parser("global-centerline")
    gcl_sub = gcl.add_subparsers(dest="gcl_cmd", required=True)

    g1 = gcl_sub.add_parser("single", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g1.add_argument("--seg", required=True, type=str)
    g1.add_argument("--out", required=True, type=str)
    g1.add_argument("--seeds-json", required=True, type=str)
    g1.add_argument("--case-name", required=True, type=str)
    g1.add_argument("--config-name", default=None, type=str)
    g1.add_argument("--merge-method", default="clean", type=str)
    g1.add_argument("--directory-data", default=None, type=str)
    g1.add_argument("--unit", default="cm", type=str)
    g1.set_defaults(_handler=_cmd_post_gcl_single)

    g2 = gcl_sub.add_parser("batch", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g2.add_argument("--seg-dir", required=True, type=str)
    g2.add_argument("--seg-glob", required=True, type=str)
    g2.add_argument("--seeds-json", required=True, type=str)
    g2.add_argument("--out-dir", default=None, type=str)
    g2.add_argument("--config-name", default=None, type=str)
    g2.add_argument("--merge-method", default="clean", type=str)
    g2.add_argument("--directory-data", default=None, type=str)
    g2.add_argument("--unit", default="cm", type=str)
    g2.set_defaults(_handler=_cmd_post_gcl_batch)

    sv = sub.add_parser("simvascular", help="SimVascular project layout")
    sv_sub = sv.add_subparsers(dest="sv_cmd", required=True)
    s1 = sv_sub.add_parser("init", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    s1.add_argument("--case-dir", required=True, type=str)
    s1.add_argument("--source-image", default=None, type=str)
    s1.add_argument("--case-id", default=None, type=str)
    s1.set_defaults(_handler=_cmd_simvascular_init)

    s2 = sv_sub.add_parser("init-batch", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    s2.add_argument("--parent-dir", required=True, type=str)
    s2.add_argument("--case-glob", default="*", type=str)
    s2.set_defaults(_handler=_cmd_simvascular_init_batch)

    cfg = sub.add_parser("config", help="Configuration utilities")
    cfg_sub = cfg.add_subparsers(dest="cfg_cmd", required=True)
    c_dump = cfg_sub.add_parser("dump")
    c_dump.add_argument("--name", default="global", type=str)
    c_dump.set_defaults(_handler=_cmd_config_dump)

    c_fp = cfg_sub.add_parser(
        "fingerprint",
        help="List YAML keys whose values differ from a baseline packaged config",
    )
    c_fp.add_argument("--name", default="global", type=str)
    c_fp.add_argument("--baseline", default="global_default", type=str)
    c_fp.set_defaults(_handler=_cmd_config_fingerprint)

    init_p = sub.add_parser("init", help="Scaffold on-disk dataset layout")
    init_sub = init_p.add_subparsers(dest="init_cmd", required=True)
    id_ds = init_sub.add_parser(
        "dataset",
        help="Create images/, centerlines/, truths/, and seeds.json template",
    )
    id_ds.add_argument("--path", required=True, type=str)
    id_ds.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing seeds.json",
    )
    id_ds.set_defaults(_handler=_cmd_init_dataset)

    doc = sub.add_parser("doctor", help="Verify dependencies and optional nnU-Net paths")
    doc.add_argument(
        "--model-folder",
        default=None,
        type=str,
        help="If set, verify this nnU-Net trainer directory exists",
    )
    doc.set_defaults(_handler=_cmd_doctor)

    return root


def dispatch(argv: Optional[Sequence[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        argv = ["--help"]
    argv = _inject_legacy_run_batch(argv)
    parser = _build_parser()
    ns = parser.parse_args(argv)
    if ns.command is None:
        parser.print_help()
        return
    handler = getattr(ns, "_handler", None)
    if handler is None:
        parser.print_help()
        return
    handler(ns)


def main() -> None:
    dispatch()
