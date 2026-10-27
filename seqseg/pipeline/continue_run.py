"""Continue a traced case from its checkpoint for more steps, optionally with new seeds."""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, List, Optional, Sequence

import numpy as np
import SimpleITK as sitk

from seqseg.config_models import AlgorithmConfig
from seqseg.modules import sitk_functions as sf
from seqseg.modules.assembly import create_step_dict
from seqseg.modules.checkpoint import (
    load_checkpoint,
    seeds_to_lists,
    try_save_checkpoint,
)
from seqseg.modules.tracing import (
    TracingContext,
    TracingResult,
    trace_centerline_from_context,
)
from seqseg.pipeline.classic import finish_classic_case
from seqseg.pipeline.plus import finish_plus_case


def seeds_from_xyzr(
    seeds_xyzr: Sequence[Sequence[float]],
    tangent: Sequence[float] = (0.0, 0.0, 1.0),
    seed_step: float = 1.0,
) -> List[list]:
    """``[x, y, z, r]`` rows to ``[old_point, new_point, radius]`` triples."""
    t = np.asarray(tangent, dtype=np.float64).ravel()
    n = float(np.linalg.norm(t))
    if n < 1e-9:
        raise ValueError("tangent norm is too small")
    t = t / n
    out = []
    for row in seeds_xyzr:
        if len(row) != 4:
            raise ValueError("each seed must be x y z radius")
        new_pt = np.asarray(row[:3], dtype=np.float64)
        old_pt = new_pt - float(seed_step) * t
        out.append([old_pt.tolist(), new_pt.tolist(), float(row[3])])
    return out


def seeds_from_json(path: str, case: Optional[str] = None) -> List[list]:
    """Read ``[old_point, new_point, radius]`` triples from a JSON file.

    Accepts a ``seeds.json`` list of cases (the entry named ``case`` is used,
    or the only entry), a single case object with ``"seeds"``, or a bare list
    of triples.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return list(data.get("seeds", []))
    if isinstance(data, list) and data and isinstance(data[0], dict):
        matches = [d for d in data if d.get("name") == case]
        if not matches and len(data) == 1:
            matches = data
        if not matches:
            raise ValueError(f"No entry named {case!r} in {path}")
        return list(matches[0].get("seeds", []))
    return list(data)


def seeds_to_steps(seeds: Sequence[Sequence[Any]]) -> list:
    """``[old_point, new_point, radius]`` triples to init-seed step dicts."""
    steps = []
    for seed in seeds:
        if len(seed) != 3:
            raise ValueError("each seed must be [old_point, new_point, radius]")
        old = np.asarray(seed[0], dtype=np.float64).ravel()
        new = np.asarray(seed[1], dtype=np.float64).ravel()
        r = float(seed[2])
        step = create_step_dict(old, r, new, r, 0)
        step["connection"] = [0, 0]
        step["init_seed"] = True
        steps.append(step)
    return steps


def run_continue(
    case_dir: str,
    extra_steps: int,
    *,
    seeds: Optional[Sequence[Sequence[Any]]] = None,
    max_n_branches: Optional[int] = None,
    max_n_steps_per_branch: Optional[int] = None,
    force_cpu: Optional[bool] = None,
) -> TracingResult:
    """Continue the run saved in ``{case_dir}/checkpoint`` for ``extra_steps`` steps.

    ``seeds`` are ``[old_point, new_point, radius]`` triples traced as new
    initialization seeds once the open branch ends. Outputs are written as in
    the original run, named with the new total step count, and the checkpoint
    is replaced so the run can be continued again.
    """
    t0 = time.time()
    if extra_steps < 0:
        raise ValueError("extra_steps must be >= 0")
    case_dir = os.path.abspath(os.path.expanduser(case_dir)) + os.sep
    ckpt = load_checkpoint(case_dir)
    m = dict(ckpt.manifest)
    global_config = AlgorithmConfig(ckpt.config)
    case = m["case"]

    new_steps = seeds_to_steps(seeds or [])
    if new_steps:
        ref_image = sitk.ReadImage(m["image_file"])
        sf.validate_potential_branches_image_bounds(
            ref_image, new_steps, case=case, image_path=m["image_file"]
        )

    initial_seeds = [np.asarray(p, dtype=np.float64) for p in m["initial_seeds"]]
    initial_seeds += [s["point"].copy() for s in new_steps]

    start_step = int(m["i"])
    max_step_size = start_step + int(extra_steps)
    if max_n_branches is None:
        max_n_branches = int(m["max_n_branches"])
    if max_n_steps_per_branch is None:
        max_n_steps_per_branch = int(m["max_n_steps_per_branch"])
    if force_cpu is None:
        force_cpu = bool(m.get("force_cpu", False))

    debug = global_config.get("DEBUG", False)
    if not debug:
        sys.stdout = open(case_dir + "out.txt", "a")
    try:
        print(f"\nContinuing {case} from step {start_step} "
              f"({m['stop_reason']}) for {extra_steps} more steps, "
              f"{len(new_steps)} new seeds")

        ctx = TracingContext(
            output_folder=case_dir,
            image_file=m["image_file"],
            case=case,
            model_folder=m["model_folder"],
            fold=m["fold"],
            potential_branches=ckpt.state.vessel_tree.potential_branches,
            max_step_size=max_step_size,
            max_n_branches=max_n_branches,
            max_n_steps_per_branch=max_n_steps_per_branch,
            global_config=global_config,
            unit=m["unit"],
            scale=m["scale"],
            seg_file=m.get("seg_file"),
            write_samples=bool(m["write_samples"]),
            simvascular=bool(m["simvascular"]),
            force_cpu=force_cpu,
            resume_state=ckpt.state,
            new_seeds=new_steps,
            save_resume_state=True,
        )
        tr = trace_centerline_from_context(ctx)

        m.update(
            max_n_steps=max_step_size,
            max_n_branches=max_n_branches,
            max_n_steps_per_branch=max_n_steps_per_branch,
            force_cpu=force_cpu,
            initial_seeds=seeds_to_lists(initial_seeds),
        )
        try_save_checkpoint(case_dir, tr, m, global_config, ckpt.start_seg)

        dir_output0 = os.path.dirname(case_dir.rstrip(os.sep)) + os.sep
        common = dict(
            case=case,
            idx=m["idx"],
            dir_output=case_dir,
            dir_output0=dir_output0,
            initial_seeds=initial_seeds,
            global_config=global_config,
            unit=m["unit"],
            take_time=global_config.get("TIME_ANALYSIS", False),
            calc_global_centerline=bool(m["calc_global_centerline"]),
            cap_surface_cent=bool(m["cap_surface_cent"]),
            assembly_threshold=float(m["assembly_threshold"]),
            simvascular=bool(m["simvascular"]),
            start_time_global=t0,
        )
        if m["pipeline"] == "plus":
            finish_plus_case(
                tr,
                seqseg_test_name=m["test_name"],
                user_start_seg=ckpt.start_seg,
                num_seeds=int(m.get("num_seeds", len(initial_seeds))),
                **common,
            )
        else:
            finish_classic_case(
                tr,
                test_name=m["test_name"],
                start_seg=ckpt.start_seg,
                write_samples=bool(m["write_samples"]),
                **common,
            )
    finally:
        if not debug:
            sys.stdout.close()
            sys.stdout = sys.__stdout__

    print(f"Continued {case}: now at step {tr.n_steps_taken} ({tr.stop_reason})")
    return tr
