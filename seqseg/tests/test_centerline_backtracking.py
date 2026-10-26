"""Automatic targets stay in the lumen, and backtracking does too."""

import numpy as np
import SimpleITK as sitk

from seqseg.modules.centerline import (
    backtracking_gradient,
    calc_centerline_fmm,
    cluster_map,
    masked_world_frame_gradient,
    snap_end_points_to_lumen,
    world_frame_gradient,
)

_IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def _image_from_xyz(seg_xyz, spacing=(1.0, 1.0, 1.0)):
    image = sitk.GetImageFromArray(np.transpose(seg_xyz, (2, 1, 0)))
    image.SetSpacing(tuple(spacing))
    return image


def _inscribed_distance(image):
    """Inscribed distance as calc_centerline_fmm builds it: + inside, - outside."""
    signed = sitk.SignedMaurerDistanceMap(
        image, squaredDistance=False, useImageSpacing=True)
    return -sitk.GetArrayFromImage(signed).transpose(2, 1, 0).astype(np.float64)


def _tube(length=40, radius=1.0, start=4, size=12):
    seg = np.zeros((length, size, size), dtype=np.uint8)
    yy, zz = np.ogrid[:size, :size]
    center = size // 2
    cross = np.hypot(yy - center, zz - center) <= radius
    for x in range(start, length - start):
        seg[x][cross] = 1
    return seg


def test_snap_end_point_stays_on_the_local_cross_section():
    radius = np.zeros((50, 20, 20), dtype=np.float64)
    for x in range(5, 25):
        yy, zz = np.ogrid[:20, :20]
        dist = np.hypot(yy - 10, zz - 10)
        tube = dist <= 3
        radius[x][tube] = 3.0 - dist[tube]
    radius[45, 10, 10] = 8.0

    snapped = snap_end_points_to_lumen(
        [np.array([5, 10, 13])], radius, (1.0, 1.0, 1.0))

    assert len(snapped) == 1
    point = snapped[0]
    assert point[0] == 5
    assert point[1] == 10
    assert point[2] == 10
    assert radius[tuple(point)] == 3.0

    speck = np.zeros((5, 5, 5), dtype=np.float64)
    speck[2, 2, 2] = 0.2
    assert snap_end_points_to_lumen(
        [np.array([2, 2, 2])], speck, (1.0, 1.0, 1.0)) == []


def test_masked_gradient_follows_a_thin_tube_at_the_wall():
    seg = _tube(radius=1.0)
    image = _image_from_xyz(seg)
    inside = seg > 0
    inward = _inscribed_distance(image)
    xs = np.arange(seg.shape[0], dtype=np.float64)[:, None, None]
    arrival = np.where(inside, np.broadcast_to(xs, seg.shape), 0.0)

    wall = (20, 6, 7)
    assert inside[wall] and not inside[20, 6, 8]

    # Differencing across the wall: the step at the wall points sideways
    # and is far larger than the step along the vessel.
    plain = world_frame_gradient(arrival, (1.0, 1.0, 1.0), _IDENTITY)
    assert abs(plain[2][wall]) > 5.0 * abs(plain[0][wall])

    # Masked: the wall voxel follows the tube, and an exterior voxel steps
    # back into the lumen with an along-vessel sized nudge.
    masked = masked_world_frame_gradient(
        arrival, inside, inward, (1.0, 1.0, 1.0), _IDENTITY)
    np.testing.assert_allclose(masked[:, 20, 6, 7], (1.0, 0.0, 0.0))
    outside = (20, 6, 8)
    descent = -masked[:, outside[0], outside[1], outside[2]]
    assert descent[2] < 0.0
    assert np.linalg.norm(descent) <= 1.0 + 1e-9


def test_backtracking_in_a_thin_tube_stays_inside_the_mask():
    seg = _tube(radius=1.0)
    image = _image_from_xyz(seg)
    inside = seg > 0
    inward = _inscribed_distance(image)
    xs = np.arange(seg.shape[0], dtype=np.float64)[:, None, None]
    arrival = np.where(inside, np.broadcast_to(xs, seg.shape), 0.0)
    gradient = masked_world_frame_gradient(
        arrival, inside, inward, (1.0, 1.0, 1.0), _IDENTITY)

    seed = np.array([6.0, 6.0, 6.0])
    for target in (np.array([32.0, 6.0, 6.0]), np.array([32.0, 6.0, 7.0])):
        points, success = backtracking_gradient(
            gradient, inward, image, seed, target, verbose=False)
        assert success
        assert len(points) < 5000
        for point in points:
            index = image.TransformPhysicalPointToIndex(point.tolist())
            assert inside[index[0], index[1], index[2]]

    # A sample one voxel outside the mask is pulled in, not rejected.
    pulled, pulled_ok = backtracking_gradient(
        gradient, inward, image, seed, np.array([32.0, 6.0, 8.0]))
    assert pulled_ok
    for point in pulled[1:]:
        index = image.TransformPhysicalPointToIndex(point.tolist())
        assert inside[index[0], index[1], index[2]]


def test_cluster_map_starts_the_wave_at_the_given_seed():
    seg = _tube(radius=1.5)
    yy, zz = np.ogrid[:12, :12]
    seg[30:35][:, (yy - 6) ** 2 + (zz - 6) ** 2 <= 9] = 1
    image = _image_from_xyz(seg)
    seed = np.array([8.0, 6.0, 6.0])

    wave_seed, _targets, wave = cluster_map(
        image, return_wave_distance_map=True, seed=seed)

    np.testing.assert_allclose(wave_seed, seed)
    wave_np = sitk.GetArrayFromImage(wave).transpose(2, 1, 0)
    assert wave_np[8, 6, 6] == 0.0


def test_cluster_map_snaps_the_distal_tip_to_the_lumen():
    seg = _tube(length=40, radius=2.0, size=16)
    yy, zz = np.ogrid[:16, :16]
    fat = (yy - 8) ** 2 + (zz - 8) ** 2 <= 16
    for x in range(32, 37):
        seg[x][fat] = 1
    image = _image_from_xyz(seg)

    _seed, targets, _wave = cluster_map(
        image, return_wave_distance_map=True, verbose=False)

    distal = min(targets, key=lambda point: point[0])
    index = image.TransformPhysicalPointToIndex(distal.tolist())
    assert index[0] < 12
    assert abs(index[1] - 8) <= 1
    assert abs(index[2] - 8) <= 1


def test_automatic_targets_reach_a_seed_away_from_the_widest_voxel():
    seg = _tube(radius=1.5)
    yy, zz = np.ogrid[:12, :12]
    seg[30:35][:, (yy - 6) ** 2 + (zz - 6) ** 2 <= 9] = 1
    image = _image_from_xyz(seg)
    seed = np.array([10.0, 6.0, 6.0])

    _centerline, success, successes = calc_centerline_fmm(
        image, seed=seed, targets=None, min_res=10,
        return_success_list=True)

    assert success
    assert all(successes)
