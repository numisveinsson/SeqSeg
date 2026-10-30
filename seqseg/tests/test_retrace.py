"""Centerline retrace checks. No image and no network."""

import numpy as np

from seqseg.modules.assembly import VesselTree, create_step_dict
from seqseg.modules.tracing import (
    discard_empty_retrace_branch,
    drop_failed_branch_step,
)
from seqseg.modules.tracing_functions import (
    centerline_retrace_kind,
    is_centerline_retrace,
    retrace_stop_action,
)


def _step(point, tangent, radius=1.0):
    point = np.asarray(point, dtype=float)
    tangent = np.asarray(tangent, dtype=float)
    tangent = tangent / np.linalg.norm(tangent)
    return create_step_dict(point - tangent, radius, point, radius, None)


def _tree(steps, branches):
    tree = VesselTree("case", "image", steps[0], [steps[0]])
    tree.steps = list(steps)
    tree.branches = [list(ids) for ids in branches]
    tree.potential_branches = []
    return tree


def _parent(n=9, radius=1.0, spacing=1.0):
    steps = [
        _step([spacing * i, 0.0, 0.0], [1.0, 0.0, 0.0], radius)
        for i in range(n)
    ]
    return _tree(steps, [list(range(n))])


def test_next_step_along_the_branch_is_not_a_retrace():
    tree = _parent()
    last = tree.steps[-1]["point"]
    ahead = last + np.array([0.4, 0.0, 0.0])
    tangent = np.array([1.0, 0.0, 0.0])

    assert not is_centerline_retrace(
        tree, ahead, tangent, branch=0)
    assert is_centerline_retrace(
        tree, ahead, tangent, branch=0, tip_ignore_steps=0)


def test_reversal_onto_the_parent_is_a_retrace():
    tree = _parent(n=9, radius=2.0)
    child = _step([8.0, 0.0, 0.0], [-1.0, 0.0, 0.0], radius=2.0)
    tree.steps.append(child)
    tree.branches.append([4, len(tree.steps) - 1])

    assert is_centerline_retrace(
        tree, child["point"], child["tangent"], branch=1)


def test_child_leaving_inside_the_parent_radius_is_not_a_retrace():
    tree = _parent(n=9, radius=2.0)
    child = _step([4.2, 0.3, 0.0], [0.0, 1.0, 0.0], radius=0.4)
    tree.steps.append(child)
    tree.branches.append([4, len(tree.steps) - 1])

    assert np.linalg.norm(child["point"] - tree.steps[4]["point"]) < 2.0
    assert not is_centerline_retrace(
        tree, child["point"], child["tangent"], branch=1)


def test_shallow_departure_at_the_ostium_uses_the_birth_zone():
    tree = _parent(n=9, radius=2.0)
    angle = np.deg2rad(15.0)
    child = _step(
        [4.2, 0.1, 0.0],
        [np.cos(angle), np.sin(angle), 0.0],
        radius=0.4,
    )
    tree.steps.append(child)
    tree.branches.append([4, len(tree.steps) - 1])

    assert not is_centerline_retrace(
        tree, child["point"], child["tangent"], branch=1)
    assert is_centerline_retrace(
        tree, child["point"], child["tangent"], branch=1,
        birth_ignore_factor=0.0)


def test_parallel_centerline_two_radii_away_is_not_a_retrace():
    radius = 1.0
    tree = _parent(n=9, radius=radius)
    point = np.array([4.0, 2.0 * radius, 0.0])
    tangent = np.array([1.0, 0.0, 0.0])

    assert not is_centerline_retrace(tree, point, tangent, branch=None)


def test_loop_that_leaves_and_returns_is_kept():
    tree = _parent(n=9, radius=1.0)
    away = [
        _step([0.0, 3.0, 0.0], [0.0, 1.0, 0.0]),
        _step([0.0, 6.0, 0.0], [0.0, 1.0, 0.0]),
        _step([4.0, 6.0, 0.0], [1.0, 0.0, 0.0]),
        _step([8.0, 6.0, 0.0], [1.0, 0.0, 0.0]),
        _step([8.0, 4.0, 0.0], [0.0, -1.0, 0.0]),
        _step([8.0, 2.5, 0.0], [0.0, -1.0, 0.0]),
        _step([8.0, 1.5, 0.0], [0.0, -1.0, 0.0]),
        _step([8.0, 1.0, 0.0], [0.0, -1.0, 0.0]),
    ]
    ids = [0] + list(range(len(tree.steps), len(tree.steps) + len(away)))
    tree.steps.extend(away)
    tree.branches.append(ids)
    closing = np.array([8.0, 0.2, 0.0])
    tangent = np.array([1.0, 0.0, 0.0])

    assert centerline_retrace_kind(
        tree, closing, tangent, branch=1) == "loop"
    assert not is_centerline_retrace(tree, closing, tangent, branch=1)
    assert is_centerline_retrace(
        tree, closing, tangent, branch=1, allow_loops=False)
    assert retrace_stop_action(tree.branches[1], kind="loop") == "keep_prefix"
    assert retrace_stop_action([0, 1], kind="loop") == "keep_prefix"
    assert ids[1] in tree.branches[1]


def test_return_to_an_earlier_part_of_the_same_branch_is_a_retrace():
    tree = _parent(n=9, radius=1.0)
    early = tree.steps[1]["point"]

    assert is_centerline_retrace(
        tree, early, np.array([1.0, 0.0, 0.0]), branch=0)
    assert is_centerline_retrace(
        tree, early, np.array([-1.0, 0.0, 0.0]), branch=0)


def test_retrace_after_real_steps_keeps_prefix_and_children():
    steps = [_step([float(i), 0.0, 0.0], [1.0, 0.0, 0.0]) for i in range(6)]
    child_real = _step([4.0, 1.0, 0.0], [0.0, 1.0, 0.0])
    child_failed = _step([4.0, 2.0, 0.0], [0.0, 1.0, 0.0])
    steps.extend([child_real, child_failed])
    tree = _tree(steps, [list(range(6)), [4, 6, 7]])
    grandchild = _step([4.0, 1.5, 0.5], [0.0, 0.0, 1.0])
    grandchild["connection"] = [1, 6]
    tree.potential_branches.append(grandchild)

    assert retrace_stop_action(tree.branches[1]) == "keep_prefix"
    drop_failed_branch_step(tree, 1)

    assert tree.branches[1] == [4, 6]
    assert tree.potential_branches == [grandchild]
    assert grandchild["connection"] == [1, 6]


def test_first_step_retrace_does_not_count_as_a_started_branch():
    steps = [_step([float(i), 0.0, 0.0], [1.0, 0.0, 0.0]) for i in range(5)]
    rejected = _step([2.0, 0.2, 0.0], [0.0, 1.0, 0.0])
    steps.append(rejected)
    tree = _tree(steps, [list(range(5)), [2, 5]])
    from_rejected = _step([2.2, 0.4, 0.0], [0.0, 1.0, 0.0])
    from_rejected["connection"] = [1, 5]
    unrelated = _step([1.0, 1.0, 0.0], [0.0, 1.0, 0.0])
    unrelated["connection"] = [0, 1]
    tree.potential_branches.extend([from_rejected, unrelated])

    assert retrace_stop_action(tree.branches[1]) == "discard_empty"
    branch, i, started = discard_empty_retrace_branch(
        tree, 1, branches_started=2)

    assert started == 1
    assert branch == 0
    assert i == 5
    assert len(tree.branches) == 1
    assert any(pot is unrelated for pot in tree.potential_branches)
    assert unrelated["connection"] == [0, 1]
    assert any(pot is from_rejected for pot in tree.potential_branches)
    assert from_rejected["connection"][0] != 1
