"""Regression tests for vectorized ``get_end_points`` (centerline)."""

import numpy as np
import pytest
import SimpleITK as sitk

from seqseg.modules.centerline import get_end_points


def _legacy_get_end_points(cluster_map, distance_map, end_clusters):
    end_points = []
    for end_cluster in end_clusters:
        indices = np.argwhere(cluster_map == end_cluster)
        max_distance = 0
        end_point = None
        for index in indices:
            distance = distance_map[tuple(index)]
            if distance > max_distance:
                max_distance = distance
                end_point = index
        end_points.append(end_point)
    return end_points


def _volume_to_sitk(arr_np):
    """Invert ``GetArrayFromImage(...).transpose(2, 1, 0)`` for test images."""
    sitk_layout = np.transpose(arr_np, (2, 1, 0))
    return sitk.GetImageFromArray(sitk_layout)


def _assert_roundtrip(arr_np):
    img = _volume_to_sitk(arr_np)
    back = sitk.GetArrayFromImage(img).transpose(2, 1, 0)
    np.testing.assert_array_equal(back, arr_np)


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_get_end_points_matches_legacy_random(seed):
    rng = np.random.default_rng(seed)
    shape = (5, 6, 7)
    cluster_map = rng.integers(0, 5, size=shape, dtype=np.int32)
    distance_map = rng.random(shape).astype(np.float64)
    end_clusters = [1, 2, 3, 4]

    _assert_roundtrip(cluster_map)
    _assert_roundtrip(distance_map.astype(np.float32))

    img_c = _volume_to_sitk(cluster_map)
    img_d = _volume_to_sitk(distance_map.astype(np.float32))

    legacy = _legacy_get_end_points(cluster_map, distance_map, end_clusters)
    fast = get_end_points(
        img_c,
        end_clusters,
        img_d,
        cluster_map_np=cluster_map,
        distance_map_np=distance_map,
    )
    assert len(legacy) == len(fast)
    for a, b in zip(legacy, fast):
        if a is None:
            assert b is None
        else:
            np.testing.assert_array_equal(a, b)


def test_get_end_points_tie_breaker_first_argwhere_order():
    """Equal max distance: strict ``>`` keeps first argwhere row."""
    cluster_map = np.zeros((2, 2, 3), dtype=np.int32)
    cluster_map[:, :, :] = 7
    distance_map = np.zeros_like(cluster_map, dtype=np.float64)
    distance_map[0, 0, 0] = 5.0
    distance_map[1, 1, 2] = 5.0

    img_c = _volume_to_sitk(cluster_map)
    img_d = _volume_to_sitk(distance_map.astype(np.float32))

    legacy = _legacy_get_end_points(cluster_map, distance_map, [7])
    fast = get_end_points(
        img_c,
        [7],
        img_d,
        cluster_map_np=cluster_map,
        distance_map_np=distance_map,
    )
    np.testing.assert_array_equal(legacy[0], fast[0])
    np.testing.assert_array_equal(fast[0], np.array([0, 0, 0]))


def test_get_end_points_empty_end_clusters():
    cluster_map = np.ones((2, 2, 2), dtype=np.int32)
    img_c = _volume_to_sitk(cluster_map)
    img_d = _volume_to_sitk(np.ones_like(cluster_map, dtype=np.float32))
    assert get_end_points(img_c, [], img_d) == []


def test_get_end_points_missing_label_returns_none():
    cluster_map = np.zeros((3, 3, 3), dtype=np.int32)
    cluster_map[0, 0, 0] = 1
    distance_map = np.ones_like(cluster_map, dtype=np.float64)
    img_c = _volume_to_sitk(cluster_map)
    img_d = _volume_to_sitk(distance_map.astype(np.float32))
    out = get_end_points(
        img_c,
        [2],
        img_d,
        cluster_map_np=cluster_map,
        distance_map_np=distance_map,
    )
    assert out == [None]
