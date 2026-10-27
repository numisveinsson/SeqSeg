"""Checkpoint save/load and resume helpers (no nnU-Net)."""

import json
from unittest.mock import MagicMock

import numpy as np
import pytest
import SimpleITK as sitk

from seqseg.api import continue_tracing
from seqseg.modules.assembly import Segmentation, VesselTree, create_step_dict
from seqseg.modules.checkpoint import (
    CHECKPOINT_VERSION,
    ResumeState,
    has_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from seqseg.modules.tracing import (
    TracingResult,
    _pop_unused_init_seed,
    _queue_resume_seeds,
    trace_centerline,
)
from seqseg.pipeline.continue_run import (
    seeds_from_json,
    seeds_from_xyzr,
    seeds_to_steps,
)
from seqseg.modules.vtk_functions import points2polydata


def _step(old, new, r=1.0, init=False):
    s = create_step_dict(np.array(old, dtype=float), r,
                         np.array(new, dtype=float), r, 0)
    s["connection"] = [0, 0]
    if init:
        s["init_seed"] = True
    return s


def _tiny_image(value=0.0, size=(4, 4, 4)):
    arr = np.full(size[::-1], value, dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    return img


def _tiny_tree():
    s0 = _step([0, 0, 0], [0, 0, 1], init=True)
    s1 = _step([0, 0, 1], [0, 0, 2])
    pot = _step([0, 0, 2], [1, 0, 2])
    tree = VesselTree("c", "<sitk.Image>", s0, [s0, pot])
    tree.add_step(1, s1, 0)
    return tree, s0, s1, pot


def _seg_from_image(img, weighted=False, fill=0.25):
    start = None
    if fill:
        start = _tiny_image(fill, size=img.GetSize())
    seg = Segmentation(
        case="c",
        image=img,
        weighted=weighted,
        weight_type="radius" if weighted else None,
        start_seg=start,
    )
    if fill:
        seg.number_updates[:] = 2
        if weighted:
            seg.n_updates[:] = 3
    return seg


def _result(tree, seg, i=1, stop="max_steps", resume_mode="pending_step"):
    return TracingResult(
        centerlines=[],
        surfaces=[],
        points=[],
        inside_pts=[],
        assembly=seg,
        vessel_tree=tree,
        n_steps_taken=i,
        stop_reason=stop,
        loop_state={
            "i": i,
            "branch": 0,
            "num_steps_direction": 1,
            "inside_branch": 0,
            "simvascular_path_counter": 0,
            "resume_mode": resume_mode,
        },
    )


def test_checkpoint_roundtrip(tmp_path):
    tree, s0, s1, pot = _tiny_tree()
    s1["point_pd"] = points2polydata([s1["point"].tolist()])
    img = _tiny_image(0.4)
    seg = _seg_from_image(img, weighted=True, fill=0.4)
    start = _tiny_image(1.0)
    result = _result(tree, seg, i=1)
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    manifest = {
        "pipeline": "classic",
        "case": "c",
        "idx": 0,
        "test_name": "3d_fullres",
        "image_file": "/img.mha",
        "model_folder": "/model",
        "fold": "all",
        "unit": "cm",
        "scale": 1.0,
        "write_samples": False,
        "simvascular": False,
        "calc_global_centerline": False,
        "cap_surface_cent": False,
        "assembly_threshold": 0.5,
        "max_n_steps": 200,
        "max_n_branches": 100,
        "max_n_steps_per_branch": 100,
        "initial_seeds": [[0.0, 0.0, 1.0]],
    }
    save_checkpoint(str(case_dir), result, manifest, {"USE_BUFFER": True}, start)
    assert has_checkpoint(str(case_dir))

    loaded = load_checkpoint(str(case_dir))
    assert loaded.manifest["version"] == CHECKPOINT_VERSION
    assert loaded.manifest["stop_reason"] == "max_steps"
    assert loaded.manifest["i"] == 1
    assert loaded.manifest["resume_mode"] == "pending_step"
    assert loaded.state.i == 1
    assert loaded.state.branch == 0
    np.testing.assert_allclose(loaded.state.vessel_tree.steps[1]["point"], [0, 0, 2])
    assert loaded.state.vessel_tree.steps[1]["point_pd"] is not None
    assert len(loaded.state.vessel_tree.potential_branches) == 2
    np.testing.assert_allclose(
        sitk.GetArrayFromImage(loaded.state.assembly),
        sitk.GetArrayFromImage(seg.assembly),
    )
    np.testing.assert_array_equal(loaded.state.number_updates, seg.number_updates)
    np.testing.assert_array_equal(loaded.state.n_updates, seg.n_updates)
    assert loaded.start_seg is not None


def test_apply_to_restores_assembly():
    img = _tiny_image(0.0)
    saved = _seg_from_image(img, weighted=True, fill=0.7)
    fresh = Segmentation(
        case="c", image=img, weighted=True, weight_type="radius"
    )
    state = ResumeState(
        vessel_tree=_tiny_tree()[0],
        assembly=saved.assembly,
        number_updates=saved.number_updates,
        n_updates=saved.n_updates,
        i=3,
        branch=0,
        num_steps_direction=1,
        inside_branch=0,
        simvascular_path_counter=0,
        resume_mode="pending_step",
    )
    state.apply_to(fresh)
    np.testing.assert_allclose(
        sitk.GetArrayFromImage(fresh.assembly),
        sitk.GetArrayFromImage(saved.assembly),
    )
    np.testing.assert_array_equal(fresh.number_updates, saved.number_updates)
    np.testing.assert_array_equal(fresh.n_updates, saved.n_updates)


def test_queue_pending_step_appends_seeds():
    tree, *_ = _tiny_tree()
    extra = _step([5, 0, 0], [6, 0, 0], init=True)
    n_before = len(tree.potential_branches)
    branch, started, run = _queue_resume_seeds(
        tree, "pending_step", i=1, branch=0, new_seeds=[extra]
    )
    assert branch == 0
    assert started is False
    assert run is True
    assert any(p is extra for p in tree.potential_branches)
    assert extra.get("init_seed")
    assert len(tree.potential_branches) == n_before + 1
    assert extra is _pop_unused_init_seed(tree)


def test_queue_needs_new_branch_without_seeds_stops():
    tree, *_ = _tiny_tree()
    old = tree.steps[1]
    branch, started, run = _queue_resume_seeds(
        tree, "needs_new_branch", i=1, branch=0, new_seeds=[]
    )
    assert run is False
    assert started is False
    assert tree.steps[1] is old


def test_queue_needs_new_branch_installs_first_seed():
    tree, *_ = _tiny_tree()
    old = tree.steps[1]
    first = _step([9, 0, 0], [10, 0, 0], init=True)
    second = _step([11, 0, 0], [12, 0, 0], init=True)
    branch, started, run = _queue_resume_seeds(
        tree, "needs_new_branch", i=1, branch=0, new_seeds=[first, second]
    )
    assert run is True
    assert started is True
    assert branch == 1
    assert tree.steps[1] is first
    assert tree.steps[1] is not old
    assert any(p is second for p in tree.potential_branches)
    assert all(p is not first for p in tree.potential_branches)


def test_pop_unused_init_seed_before_bifurcation():
    tree, s0, *_ = _tiny_tree()
    bif = _step([0, 1, 0], [0, 2, 0])
    bif["init_seed"] = False
    unused = _step([3, 0, 0], [4, 0, 0], init=True)
    tree.potential_branches = [s0, bif, unused]
    popped = _pop_unused_init_seed(tree)
    assert popped is unused
    assert all(p is not unused for p in tree.potential_branches)
    assert any(p is bif for p in tree.potential_branches)


def _resume_config():
    return {
        "SEGMENTATION": True,
        "DEBUG": False,
        "DEBUG_STEP": 0,
        "RETRACE": False,
        "TIME_ANALYSIS": False,
        "ANIMATION": False,
        "ANIMATION_STEPS": 10,
        "NR_ALLOW_RETRACE_STEPS": 2,
        "PREVENT_RETRACE": False,
        "SORT_NEXT": True,
        "MERGE_NEXT": False,
        "VOLUME_SIZE_RATIO": 4,
        "MAX_PERC_ENLARGE": 0.33,
        "MAGN_RADIUS": 1.0,
        "NR_CHANCES": 1,
        "MIN_RADIUS": 0.1,
        "ADD_RADIUS": 0.0,
        "FORCEFUL_SIDEBRANCH": False,
        "FORCEFUL_SIDEBRANCH_MAGN_RADIUS": 1.0,
        "STOP_PRE": False,
        "STOP_RADIUS": 0.1,
        "USE_BUFFER": False,
        "ASSEMBLY_EVERY_N": 5,
        "BUFFER_N": 2,
        "WEIGHTED_ASSEMBLY": False,
        "WEIGHT_TYPE": "radius",
        "MEGA_SUBVOLUME": False,
        "NR_MEGA_SUB": 0,
        "MIN_RES": 16,
        "CENTERLINE_EXTRACTION_VMTK": False,
        "TREE_ANALYSIS": False,
    }


def test_resume_keeps_step_index(tmp_path):
    img = _tiny_image()
    tree, *_ = _tiny_tree()
    seg = _seg_from_image(img, fill=0.2)
    state = ResumeState(
        vessel_tree=tree,
        assembly=seg.assembly,
        number_updates=seg.number_updates,
        n_updates=None,
        i=7,
        branch=0,
        num_steps_direction=3,
        inside_branch=0,
        simvascular_path_counter=2,
        resume_mode="needs_new_branch",
    )
    tup = trace_centerline(
        output_folder=str(tmp_path) + "/",
        image_file=img,
        case="c",
        model_folder="/unused",
        fold="all",
        potential_branches=tree.potential_branches,
        max_step_size=20,
        max_n_branches=100,
        max_n_steps_per_branch=100,
        global_config=_resume_config(),
        unit="cm",
        seg_file=img,
        write_samples=False,
        disk_io=False,
        resume_state=state,
        new_seeds=[],
    )
    assert tup[6] == 7
    assert tup[7] == "exhausted"
    assert tup[8]["i"] == 7
    assert tup[8]["simvascular_path_counter"] == 2


def test_resume_pending_does_not_reset_i(tmp_path):
    img = _tiny_image()
    tree, *_ = _tiny_tree()
    while len(tree.steps) <= 5:
        k = len(tree.steps)
        tree.add_step(k, _step([0, 0, k], [0, 0, k + 1]), 0)
    extra = _step([8, 0, 0], [9, 0, 0], init=True)
    seg = _seg_from_image(img, fill=0.2)
    state = ResumeState(
        vessel_tree=tree,
        assembly=seg.assembly,
        number_updates=seg.number_updates,
        n_updates=None,
        i=5,
        branch=0,
        num_steps_direction=2,
        inside_branch=1,
        simvascular_path_counter=0,
        resume_mode="pending_step",
    )
    tup = trace_centerline(
        output_folder=str(tmp_path) + "/",
        image_file=img,
        case="c",
        model_folder="/unused",
        fold="all",
        potential_branches=tree.potential_branches,
        max_step_size=5,
        max_n_branches=100,
        max_n_steps_per_branch=100,
        global_config=_resume_config(),
        unit="cm",
        seg_file=img,
        write_samples=False,
        disk_io=False,
        resume_state=state,
        new_seeds=[extra],
    )
    assert tup[6] == 5
    assert tup[7] == "max_steps"
    assert any(p is extra for p in tup[5].potential_branches)
    assert extra.get("init_seed")


def test_seeds_from_xyzr_and_json(tmp_path):
    triples = seeds_from_xyzr([[1, 2, 3, 0.5]], tangent=(0, 0, 1), seed_step=1.0)
    np.testing.assert_allclose(triples[0][0], [1, 2, 2])
    np.testing.assert_allclose(triples[0][1], [1, 2, 3])
    assert triples[0][2] == 0.5

    path = tmp_path / "seeds.json"
    path.write_text(json.dumps([
        {"name": "c1", "seeds": [[[0, 0, 0], [0, 0, 1], 1.0]]},
    ]))
    assert seeds_from_json(str(path), "c1") == [[[0, 0, 0], [0, 0, 1], 1.0]]

    steps = seeds_to_steps(triples)
    assert steps[0]["init_seed"] is True
    assert steps[0]["connection"] == [0, 0]


def test_continue_tracing_delegates(monkeypatch, tmp_path):
    captured = {}
    fake = MagicMock(name="TracingResult")

    def fake_run(case_dir, extra_steps, seeds=None, **kwargs):
        captured["case_dir"] = case_dir
        captured["extra_steps"] = extra_steps
        captured["seeds"] = seeds
        captured["kwargs"] = kwargs
        return fake

    monkeypatch.setattr("seqseg.pipeline.continue_run.run_continue", fake_run)
    res = continue_tracing(
        str(tmp_path),
        50,
        seeds=[([0.0, 0.0, 0.0], 1.0)],
        max_n_branches=200,
    )
    assert res is fake
    assert captured["extra_steps"] == 50
    assert len(captured["seeds"]) == 1
    assert captured["kwargs"]["max_n_branches"] == 200
