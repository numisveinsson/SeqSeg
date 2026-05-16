"""
High-level SeqSeg tracing API (simple seeds, bundled options, one-call trace).

nnU-Net weights are still loaded from ``model_folder`` on disk via the existing
predictor initialization inside :func:`trace_centerline`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union

import numpy as np
import SimpleITK as sitk

from seqseg.config_models import AlgorithmConfig
from seqseg.modules.assembly import create_step_dict
from seqseg.modules.tracing import (
    TracingContext,
    TracingResult,
    trace_centerline_from_context,
)


@dataclass(frozen=True)
class BranchSeed:
    """One vessel seed as an old point, new point, and lumen radius (same units as ``unit``)."""

    old_point: np.ndarray
    new_point: np.ndarray
    radius: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "old_point", np.asarray(self.old_point, dtype=np.float64).ravel())
        object.__setattr__(self, "new_point", np.asarray(self.new_point, dtype=np.float64).ravel())
        if self.old_point.size != 3 or self.new_point.size != 3:
            raise ValueError("old_point and new_point must have length 3")


def branch_seed_at_point(
    point: Sequence[float],
    radius: float,
    tangent: Optional[Sequence[float]] = None,
    step: float = 1.0,
) -> BranchSeed:
    """
    Build a :class:`BranchSeed` from a single world-space point and radius.

    The old point is ``point - step * tangent`` (default tangent ``(0,0,1)``),
    so :func:`create_step_dict` receives a valid direction.
    """
    pt = np.asarray(point, dtype=np.float64).ravel()
    if pt.size != 3:
        raise ValueError("point must have length 3")
    if tangent is None:
        t = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        t = np.asarray(tangent, dtype=np.float64).ravel()
    if t.size != 3:
        raise ValueError("tangent must have length 3")
    n = float(np.linalg.norm(t))
    if n < 1e-9:
        raise ValueError("tangent norm is too small")
    t = t / n
    old_pt = pt - float(step) * t
    return BranchSeed(old_point=old_pt, new_point=pt.copy(), radius=float(radius))


def seeds_to_potential_branches(
    seeds: Sequence[Union[BranchSeed, Mapping[str, Any], Sequence[Any]]],
) -> list:
    """
    Convert high-level seeds into ``potential_branches`` step dicts for tracing.

    Accepted elements:

    - :class:`BranchSeed`
    - ``dict`` / mapping with keys ``old_point``, ``new_point``, ``radius``
    - length-3 sequence ``(old_point, new_point, radius)``
    - length-2 sequence ``(point, radius)`` — uses :func:`branch_seed_at_point`
    """
    out: list = []
    for raw in seeds:
        if isinstance(raw, BranchSeed):
            old, new, r = raw.old_point, raw.new_point, raw.radius
        elif isinstance(raw, Mapping):
            old = np.asarray(raw["old_point"], dtype=np.float64).ravel()
            new = np.asarray(raw["new_point"], dtype=np.float64).ravel()
            r = float(raw["radius"])
        elif isinstance(raw, (list, tuple)):
            if len(raw) == 3:
                old = np.asarray(raw[0], dtype=np.float64).ravel()
                new = np.asarray(raw[1], dtype=np.float64).ravel()
                r = float(raw[2])
            elif len(raw) == 2:
                bs = branch_seed_at_point(raw[0], float(raw[1]))
                old, new, r = bs.old_point, bs.new_point, bs.radius
            else:
                raise ValueError(
                    f"sequence seed must have length 2 or 3, got {len(raw)!r}"
                )
        else:
            raise TypeError(f"unsupported seed type: {type(raw)!r}")

        if old.size != 3 or new.size != 3:
            raise ValueError("old_point and new_point must have length 3")
        step = create_step_dict(old, r, new, r, None)
        step["connection"] = [0, 0]
        out.append(step)
    return out


@dataclass
class TracingOptions:
    """Common tracing limits and I/O flags (optional wrapper for :class:`TracingContext`)."""

    max_n_steps: int = 1000
    max_n_branches: int = 100
    max_n_steps_per_branch: int = 100
    write_samples: bool = False
    disk_io: bool = True
    unit: str = "cm"
    scale: float = 1.0
    fold: str = "all"
    seg_file: Optional[Union[str, sitk.Image]] = None
    start_seg: Optional[sitk.Image] = None


def run_tracing(
    image: sitk.Image,
    seeds: Sequence[Union[BranchSeed, Mapping[str, Any], Sequence[Any]]],
    model_folder: str,
    *,
    case: str = "seqseg_case",
    config: Union[str, AlgorithmConfig, MutableMapping[str, Any]] = "global",
    options: Optional[TracingOptions] = None,
    output_folder: str = "",
) -> TracingResult:
    """
    Run vessel tracing on an in-memory image with simple seed definitions.

    Parameters
    ----------
    image
        Full reference volume (same role as the file passed as ``image_file``).
    seeds
        See :func:`seeds_to_potential_branches`.
    model_folder
        nnU-Net ``nnUNetTrainer__nnUNetPlans__*`` folder on disk (weights read from there).
    case
        Case label for logging and any outputs when ``disk_io`` is True.
    config
        Packaged config name (e.g. ``\"global\"``), an :class:`~seqseg.config_models.AlgorithmConfig`,
        or a plain mapping (copied into :class:`~seqseg.config_models.AlgorithmConfig`).
    options
        :class:`TracingOptions`; defaults are sensible for a library call.
    output_folder
        Output root when ``disk_io`` is True; may be ``\"\"`` when ``disk_io`` is False.

    Returns
    -------
    TracingResult
        Use ``result.assembly.assembly`` for the accumulated probability ``sitk.Image``.
    """
    opts = options or TracingOptions()
    if isinstance(config, AlgorithmConfig):
        gc: MutableMapping[str, Any] = config
    elif isinstance(config, str):
        gc = AlgorithmConfig.from_name(config)
    else:
        gc = AlgorithmConfig(dict(config))

    potential = seeds_to_potential_branches(seeds)
    ctx = TracingContext(
        output_folder=output_folder,
        image_file=image,
        case=case,
        model_folder=model_folder,
        fold=opts.fold,
        potential_branches=potential,
        max_step_size=opts.max_n_steps,
        max_n_branches=opts.max_n_branches,
        max_n_steps_per_branch=opts.max_n_steps_per_branch,
        global_config=gc,
        unit=opts.unit,
        scale=opts.scale,
        seg_file=opts.seg_file,
        start_seg=opts.start_seg,
        write_samples=opts.write_samples,
        disk_io=opts.disk_io,
    )
    return trace_centerline_from_context(ctx)
