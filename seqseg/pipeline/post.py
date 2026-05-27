"""Post-processing helpers: SimVascular project layout and global centerline."""

from __future__ import annotations

import fnmatch
import json
import os
from pathlib import Path
from typing import Any, List, Mapping, Optional

import numpy as np
import SimpleITK as sitk

from seqseg.modules import initialization as init
from seqseg.modules import vtk_functions as vf
from seqseg.modules.assembly import calc_centerline_global
from seqseg.modules.datasets import normalize_dataset_root
from seqseg.modules.simvascular import write_simvascular_proj


def bootstrap_simvascular_project(
    case_dir: str,
    *,
    source_image: Optional[str] = None,
    case_id: Optional[str] = None,
) -> None:
    """
    Ensure ``case_dir/simvascular`` exists with ``simvascular.proj``.
    If ``source_image`` is set, writes ``Images/{case_id}.vti`` with SimVascular sidecars.
    """
    case_dir = os.path.abspath(case_dir)
    sv = os.path.join(case_dir, "simvascular")
    for sub in (
        sv,
        os.path.join(sv, "Images"),
        os.path.join(sv, "Paths"),
        os.path.join(sv, "Segmentations"),
        os.path.join(sv, "Models"),
    ):
        os.makedirs(sub, exist_ok=True)
    write_simvascular_proj(sv)

    if source_image:
        if not case_id:
            case_id = Path(source_image).stem.split(".")[0]
        out_vti = os.path.join(sv, "Images", f"{case_id}.vti")
        vf.write_image_as_vti(source_image, out_vti)


def bootstrap_simvascular_project_batch(
    parent_dir: str,
    *,
    case_glob: str = "*",
) -> None:
    """Run :func:`bootstrap_simvascular_project` on each matching child directory."""
    parent_dir = os.path.abspath(parent_dir)
    for name in sorted(os.listdir(parent_dir)):
        if not fnmatch.fnmatch(name, case_glob):
            continue
        path = os.path.join(parent_dir, name)
        if os.path.isdir(path):
            bootstrap_simvascular_project(path)


def _load_seeds_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _resolve_test_case(seeds_data: Any, case_name: str) -> dict:
    if isinstance(seeds_data, dict):
        if seeds_data.get("name") == case_name:
            return seeds_data
        raise ValueError(f"No entry with name={case_name!r} in JSON object")
    if isinstance(seeds_data, list):
        for entry in seeds_data:
            if isinstance(entry, dict) and entry.get("name") == case_name:
                return entry
        raise ValueError(f"No entry with name={case_name!r} in JSON list")
    raise ValueError("seeds JSON must be a list or dict")


def _initial_seeds_from_test_case(
    test_case: dict,
    *,
    dir_output: str,
    dir_cent: str,
    dir_data: str,
    unit: str,
) -> List[np.ndarray]:
    """Use initialization.initialize_json to obtain ``initial_seeds``."""
    potential, initial = init.initialize_json(
        test_case,
        dir_output,
        dir_cent,
        dir_data,
        unit,
        write_samples=False,
    )
    del potential
    return initial


def run_global_centerline_single(
    seg_path: str,
    out_vtp: str,
    *,
    seeds_json: str,
    case_name: str,
    merge_method: str = "clean",
    directory_data: Optional[str] = None,
    unit: str = "cm",
    global_config: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Compute global centerline from an existing binary/probability segmentation.

    ``seeds_json`` must be the dataset ``seeds.json`` (or compatible) and ``case_name``
    selects the entry. If seeds are empty in JSON, ``directory_data`` must point to the
    dataset root (with ``centerlines/``) as in a full SeqSeg dataset layout.
    """
    seg = sitk.ReadImage(seg_path)
    seeds_data = _load_seeds_json(seeds_json)
    test_case = _resolve_test_case(seeds_data, case_name)
    if directory_data is None:
        directory_data = normalize_dataset_root(
            str(Path(seeds_json).resolve().parent)
        )
    else:
        directory_data = normalize_dataset_root(directory_data)
    dir_cent = os.path.join(directory_data, "centerlines", case_name + ".vtp")
    tmp_out = os.path.join(os.path.dirname(out_vtp), ".seqseg_post_tmp")
    os.makedirs(tmp_out, exist_ok=True)
    initial_seeds = _initial_seeds_from_test_case(
        test_case,
        dir_output=tmp_out + os.sep,
        dir_cent=dir_cent,
        dir_data=directory_data,
        unit=unit,
    )
    mm = merge_method
    if global_config is not None:
        mm = global_config.get("CENTERLINE_MERGE_METHOD", merge_method)
    nr = len(initial_seeds)
    global_centerline, targets, success = calc_centerline_global(
        seg,
        initial_seeds,
        nr_seeds=nr,
        merge_method=mm,
    )
    if not success:
        raise RuntimeError("Global centerline extraction failed (success=False).")
    vf.write_vtk_polydata(global_centerline, out_vtp)
    if targets:
        targets_pd = vf.points2polydata([t.tolist() for t in targets])
        base, _ = os.path.splitext(out_vtp)
        vf.write_vtk_polydata(targets_pd, base + "_targets.vtp")


def _case_name_from_seg_filename(path: str) -> str:
    name = Path(path).name
    lower_name = name.lower()
    for suf in (".nii.gz", ".nii", ".mha", ".mhd"):
        if lower_name.endswith(suf):
            stem = name[: -len(suf)]
            break
    else:
        stem = Path(path).stem
    marker = "_segmentation_"
    if marker in stem:
        return stem.split(marker)[0]
    return stem


def run_global_centerline_batch(
    seg_dir: str,
    *,
    seg_glob: str,
    seeds_json: str,
    out_dir: Optional[str] = None,
    merge_method: str = "clean",
    directory_data: Optional[str] = None,
    unit: str = "cm",
    global_config: Optional[Mapping[str, Any]] = None,
) -> None:
    """Batch variant of :func:`run_global_centerline_single` using filename-derived case ids."""
    seg_dir = os.path.abspath(seg_dir)
    out_dir = out_dir or seg_dir
    os.makedirs(out_dir, exist_ok=True)
    seeds_data = _load_seeds_json(seeds_json)
    if directory_data is None:
        directory_data = normalize_dataset_root(
            str(Path(seeds_json).resolve().parent)
        )
    else:
        directory_data = normalize_dataset_root(directory_data)

    for name in sorted(os.listdir(seg_dir)):
        if not fnmatch.fnmatch(name, seg_glob):
            continue
        seg_path = os.path.join(seg_dir, name)
        if not os.path.isfile(seg_path):
            continue
        case_name = _case_name_from_seg_filename(seg_path)
        _resolve_test_case(seeds_data, case_name)
        out_vtp = os.path.join(out_dir, f"{case_name}_global_centerline.vtp")
        run_global_centerline_single(
            seg_path,
            out_vtp,
            seeds_json=seeds_json,
            case_name=case_name,
            merge_method=merge_method,
            directory_data=directory_data,
            unit=unit,
            global_config=global_config,
        )
