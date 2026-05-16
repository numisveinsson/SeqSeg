"""Lightweight tests for :mod:`seqseg.api` seed helpers (no nnU-Net)."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import SimpleITK as sitk

from seqseg.api import (
    BranchSeed,
    TracingOptions,
    branch_seed_at_point,
    run_tracing,
    seeds_to_potential_branches,
)


def test_branch_seed_at_point_default_tangent():
    bs = branch_seed_at_point([0, 0, 5], 1.5, step=2.0)
    np.testing.assert_allclose(bs.old_point, [0, 0, 3])
    np.testing.assert_allclose(bs.new_point, [0, 0, 5])
    assert bs.radius == 1.5


def test_branch_seed_at_point_custom_tangent():
    t = [1.0, 0.0, 0.0]
    bs = branch_seed_at_point([10, 0, 0], 2.0, tangent=t, step=1.0)
    np.testing.assert_allclose(bs.old_point, [9, 0, 0])
    np.testing.assert_allclose(bs.new_point, [10, 0, 0])


def test_seeds_branch_seed_roundtrip():
    bs = BranchSeed(
        old_point=[0, 0, 0],
        new_point=[0, 0, 1],
        radius=0.5,
    )
    steps = seeds_to_potential_branches([bs])
    assert len(steps) == 1
    assert steps[0]["connection"] == [0, 0]


def test_seeds_mapping_and_tuple_short_form():
    steps = seeds_to_potential_branches(
        [
            {"old_point": [0, 0, 0], "new_point": [1, 0, 0], "radius": 1.0},
            ([0, 0, 0], 0.5),
        ]
    )
    assert len(steps) == 2


def test_seeds_invalid_length_raises():
    with pytest.raises(ValueError, match="length 2 or 3"):
        seeds_to_potential_branches([(1, 2, 3, 4)])


def test_seeds_unsupported_type_raises():
    with pytest.raises(TypeError, match="unsupported seed type"):
        seeds_to_potential_branches([42])


def test_tracing_options_defaults():
    opts = TracingOptions()
    assert opts.disk_io is True
    assert opts.max_n_steps == 1000
    assert opts.fold == "all"


def test_run_tracing_builds_context_and_calls_tracing(monkeypatch):
    fake_result = MagicMock(name="TracingResult")
    captured = {}

    def fake_trace(ctx):
        captured["ctx"] = ctx
        return fake_result

    monkeypatch.setattr("seqseg.api.trace_centerline_from_context", fake_trace)
    img = sitk.Image(8, 8, 8, sitk.sitkUInt8)
    res = run_tracing(
        img,
        [([0.0, 0.0, 0.0], 1.0)],
        "/fake/nnUNetTrainer",
        case="c1",
        config="global",
        options=TracingOptions(disk_io=False, max_n_steps=50),
        output_folder="",
    )
    assert res is fake_result
    ctx = captured["ctx"]
    assert ctx.case == "c1"
    assert ctx.disk_io is False
    assert ctx.max_step_size == 50
    assert ctx.model_folder == "/fake/nnUNetTrainer"
    assert len(ctx.potential_branches) == 1


def test_lazy_seqseg_exports_run_tracing():
    import seqseg

    fn = seqseg.run_tracing
    assert callable(fn)
    assert fn.__module__ == "seqseg.api"
