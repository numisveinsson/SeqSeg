"""Save and load tracing state so a run can be continued for more steps.

A checkpoint lives in ``{case_dir}/checkpoint/``:

- ``manifest.json``: run inputs, limits, loop counters, stop reason
- ``config.json``: algorithm config used by the run
- ``tree.json``: vessel tree with numeric step dicts
- ``assembly.mha``, ``number_updates.mha`` (and ``n_updates.mha`` if weighted)
- ``start_seg.mha`` when the run started from a segmentation
- ``completed_*.vtp``: geometry of completed branches
- ``open_branch/*.vtp``: per-step geometry still held by unfinished branches
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np
import SimpleITK as sitk

from .assembly import VesselTree
from .vtk_functions import appendPolyData, read_geo, write_vtk_polydata

CHECKPOINT_DIRNAME = "checkpoint"
CHECKPOINT_VERSION = 1

# Step keys holding VTK polydata; written as .vtp files next to tree.json.
_POLYDATA_KEYS = ("surface", "centerline", "point_pd")
_COMPLETED_LISTS = ("centerlines", "surfaces", "points", "inside_pts")


@dataclass
class ResumeState:
    """Everything :func:`trace_centerline` needs to re-enter its main loop."""

    vessel_tree: VesselTree
    assembly: sitk.Image
    number_updates: np.ndarray
    n_updates: Optional[np.ndarray]
    i: int
    branch: int
    num_steps_direction: int
    inside_branch: int
    simvascular_path_counter: int
    resume_mode: str
    steps_taken: Optional[int] = None
    branches_started: Optional[int] = None
    centerlines: list = field(default_factory=list)
    surfaces: list = field(default_factory=list)
    points: list = field(default_factory=list)
    inside_pts: list = field(default_factory=list)

    def apply_to(self, segmentation) -> None:
        """Overwrite a freshly built :class:`Segmentation` with saved volumes."""
        expected = tuple(segmentation.number_updates.shape)
        if tuple(self.number_updates.shape) != expected:
            raise ValueError(
                "Checkpoint assembly shape "
                f"{tuple(self.number_updates.shape)} does not match image {expected}"
            )
        segmentation.assembly = self.assembly
        segmentation.number_updates = self.number_updates.astype(
            segmentation.number_updates.dtype, copy=False
        )
        if segmentation.weighted:
            if self.n_updates is not None:
                segmentation.n_updates = self.n_updates.astype(np.int32, copy=False)
            else:
                segmentation.n_updates = np.zeros(expected, dtype=np.int32)


@dataclass
class Checkpoint:
    manifest: dict
    config: dict
    state: ResumeState
    start_seg: Optional[sitk.Image] = None


def checkpoint_dir(case_dir: str) -> str:
    return os.path.join(case_dir, CHECKPOINT_DIRNAME)


def has_checkpoint(case_dir: str) -> bool:
    return os.path.isfile(os.path.join(checkpoint_dir(case_dir), "manifest.json"))


def _encode(value: Any) -> Any:
    """JSON-safe copy of a step value; images and VTK objects become ``None``."""
    if isinstance(value, np.ndarray):
        return {"__ndarray__": value.tolist(), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _encode(v) for k, v in value.items()}
    return None


def _decode(value: Any) -> Any:
    if isinstance(value, dict):
        if "__ndarray__" in value:
            return np.asarray(value["__ndarray__"], dtype=value.get("dtype", None))
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v) for v in value]
    return value


def _image_from_array(arr: np.ndarray, reference: sitk.Image) -> sitk.Image:
    img = sitk.GetImageFromArray(arr)
    img.CopyInformation(reference)
    return img


def save_checkpoint(
    case_dir: str,
    result: Any,
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    start_seg: Optional[sitk.Image] = None,
) -> str:
    """Write a checkpoint for ``result`` (a :class:`TracingResult`).

    The previous checkpoint is replaced only after the new one is fully written.
    """
    final_dir = checkpoint_dir(case_dir)
    tmp_dir = final_dir + ".tmp"
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(os.path.join(tmp_dir, "open_branch"))

    tree = result.vessel_tree
    objects: list = []
    index_of: dict = {}

    def obj_index(step: dict) -> int:
        key = id(step)
        if key not in index_of:
            index_of[key] = len(objects)
            objects.append(step)
        return index_of[key]

    step_ids = [obj_index(s) for s in tree.steps]
    pot_ids = [obj_index(p) for p in tree.potential_branches]

    loop_state = dict(result.loop_state)
    pre_flush = loop_state.pop("pre_flush", None)
    pending = {}
    if pre_flush is not None:
        pending = {id(step): prob for step, prob in pre_flush["pending"]}

    encoded_objects = []
    for k, step in enumerate(objects):
        enc = {key: _encode(val) for key, val in step.items()
               if key not in _POLYDATA_KEYS}
        vtp = {}
        for key in _POLYDATA_KEYS:
            poly = step.get(key)
            if poly is not None:
                fname = f"step{k}_{key}.vtp"
                write_vtk_polydata(poly, os.path.join(tmp_dir, "open_branch", fname))
                vtp[key] = fname
        if vtp:
            enc["__vtp__"] = vtp
        prob = pending.get(id(step), step.get("prob_predicted_vessel"))
        if isinstance(prob, sitk.Image):
            fname = f"step{k}_prob.mha"
            sitk.WriteImage(prob, os.path.join(tmp_dir, "open_branch", fname), True)
            enc["__prob__"] = fname
        encoded_objects.append(enc)

    tree_json = {
        "objects": encoded_objects,
        "steps": step_ids,
        "potential_branches": pot_ids,
        "branches": [list(map(int, b)) for b in tree.branches],
        "bifurcations": [int(b) for b in tree.bifurcations],
        "caps": _encode(list(tree.caps)),
    }
    with open(os.path.join(tmp_dir, "tree.json"), "w", encoding="utf-8") as f:
        json.dump(tree_json, f)

    seg = result.assembly
    weighted = bool(getattr(seg, "weighted", False))
    if pre_flush is not None:
        assembly = pre_flush["assembly"]
        number_updates = pre_flush["number_updates"]
        n_updates = pre_flush["n_updates"]
    else:
        assembly = seg.assembly
        number_updates = seg.number_updates
        n_updates = seg.n_updates if weighted else None
    sitk.WriteImage(assembly, os.path.join(tmp_dir, "assembly.mha"), True)
    sitk.WriteImage(
        _image_from_array(number_updates, assembly),
        os.path.join(tmp_dir, "number_updates.mha"),
        True,
    )
    if weighted and n_updates is not None:
        sitk.WriteImage(
            _image_from_array(n_updates, assembly),
            os.path.join(tmp_dir, "n_updates.mha"),
            True,
        )
    if start_seg is not None:
        sitk.WriteImage(start_seg, os.path.join(tmp_dir, "start_seg.mha"), True)

    completed = {}
    for name in _COMPLETED_LISTS:
        polys = [p for p in getattr(result, name) if p is not None]
        if polys:
            fname = f"completed_{name}.vtp"
            write_vtk_polydata(appendPolyData(polys), os.path.join(tmp_dir, fname))
            completed[name] = fname

    out_manifest = dict(manifest)
    out_manifest.update(loop_state)
    out_manifest["version"] = CHECKPOINT_VERSION
    out_manifest["stop_reason"] = result.stop_reason
    out_manifest["n_steps_taken"] = int(result.n_steps_taken)
    out_manifest["weighted"] = weighted
    out_manifest["completed"] = completed
    with open(os.path.join(tmp_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(_encode(out_manifest), f, indent=2)
    with open(os.path.join(tmp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(_encode(dict(config)), f, indent=2)

    if os.path.isdir(final_dir):
        shutil.rmtree(final_dir)
    os.replace(tmp_dir, final_dir)
    return final_dir


def try_save_checkpoint(case_dir, result, manifest, config, start_seg=None):
    """Like :func:`save_checkpoint`, but report failures instead of raising."""
    try:
        path = save_checkpoint(case_dir, result, manifest, config, start_seg)
        print(f"Wrote checkpoint ({result.stop_reason}, step {result.n_steps_taken}): "
              f"{path}")
        return path
    except Exception as e:  # noqa: BLE001
        print(f"Warning: could not write checkpoint in {case_dir}: {e}")
        return None


def seeds_to_lists(seeds) -> list:
    return [np.asarray(s, dtype=np.float64).ravel().tolist() for s in seeds]


def load_checkpoint(case_dir: str) -> Checkpoint:
    """Read a checkpoint written by :func:`save_checkpoint`."""
    ckpt = checkpoint_dir(case_dir)
    manifest_path = os.path.join(ckpt, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"No SeqSeg checkpoint found in {ckpt}")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {manifest.get('version')!r} "
            f"(expected {CHECKPOINT_VERSION})"
        )
    with open(os.path.join(ckpt, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    with open(os.path.join(ckpt, "tree.json"), encoding="utf-8") as f:
        tree_json = json.load(f)

    objects = []
    for enc in tree_json["objects"]:
        vtp = enc.pop("__vtp__", {})
        prob_file = enc.pop("__prob__", None)
        step = _decode(enc)
        for key in _POLYDATA_KEYS:
            step[key] = None
        for key, fname in vtp.items():
            step[key] = read_geo(os.path.join(ckpt, "open_branch", fname)).GetOutput()
        if prob_file:
            step["prob_predicted_vessel"] = sitk.ReadImage(
                os.path.join(ckpt, "open_branch", prob_file))
        objects.append(step)

    tree = VesselTree.from_state(
        case=manifest.get("case"),
        image=manifest.get("image_file"),
        steps=[objects[k] for k in tree_json["steps"]],
        branches=tree_json["branches"],
        bifurcations=tree_json["bifurcations"],
        potential_branches=[objects[k] for k in tree_json["potential_branches"]],
        caps=_decode(tree_json["caps"]),
    )

    assembly = sitk.ReadImage(os.path.join(ckpt, "assembly.mha"))
    number_updates = sitk.GetArrayFromImage(
        sitk.ReadImage(os.path.join(ckpt, "number_updates.mha"))
    )
    n_updates = None
    n_path = os.path.join(ckpt, "n_updates.mha")
    if os.path.isfile(n_path):
        n_updates = sitk.GetArrayFromImage(sitk.ReadImage(n_path))

    start_seg = None
    ss_path = os.path.join(ckpt, "start_seg.mha")
    if os.path.isfile(ss_path):
        start_seg = sitk.ReadImage(ss_path)

    completed = {}
    for name in _COMPLETED_LISTS:
        fname = manifest.get("completed", {}).get(name)
        completed[name] = (
            [read_geo(os.path.join(ckpt, fname)).GetOutput()] if fname else []
        )

    state = ResumeState(
        vessel_tree=tree,
        assembly=assembly,
        number_updates=number_updates,
        n_updates=n_updates,
        i=int(manifest["i"]),
        branch=int(manifest["branch"]),
        num_steps_direction=int(manifest["num_steps_direction"]),
        inside_branch=int(manifest["inside_branch"]),
        simvascular_path_counter=int(manifest["simvascular_path_counter"]),
        resume_mode=manifest["resume_mode"],
        steps_taken=(None if manifest.get("steps_taken") is None
                     else int(manifest["steps_taken"])),
        branches_started=(None if manifest.get("branches_started") is None
                          else int(manifest["branches_started"])),
        **completed,
    )
    return Checkpoint(manifest=manifest, config=config, state=state,
                      start_seg=start_seg)
