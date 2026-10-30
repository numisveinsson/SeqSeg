"""Seed placement keeps the largest bodies and drops speckle."""

import numpy as np
import SimpleITK as sitk

from seqseg.modules.centerline import create_seeds_from_disconnected_bodies


def _segmentation_with_blobs(sizes):
    """Binary image of separated boxes. ``sizes`` are (nx, ny, nz)."""
    gap = 2
    width = sum(nx + gap for nx, _, _ in sizes)
    height = max(ny for _, ny, _ in sizes)
    depth = max(nz for _, _, nz in sizes)
    arr = np.zeros((depth, height, width), dtype=np.uint8)
    x = 0
    for nx, ny, nz in sizes:
        arr[:nz, :ny, x:x + nx] = 1
        x += nx + gap
    image = sitk.GetImageFromArray(arr)
    image.SetSpacing((1.0, 1.0, 1.0))
    return image


def _selected_sizes(image, nr_seeds, min_component_voxels):
    _, info = create_seeds_from_disconnected_bodies(
        image,
        nr_seeds=nr_seeds,
        min_component_voxels=min_component_voxels,
    )
    return [comp['size'] for comp in info['selected_components']]


def test_top_five_drops_bodies_under_the_voxel_cutoff():
    # 800, 400, 200, 150, 120 stay; 40 and 8 are speckle.
    image = _segmentation_with_blobs([
        (20, 8, 5),
        (10, 8, 5),
        (8, 5, 5),
        (6, 5, 5),
        (6, 5, 4),
        (4, 5, 2),
        (2, 2, 2),
    ])
    assert _selected_sizes(image, nr_seeds=5, min_component_voxels=100) == [
        800, 400, 200, 150, 120,
    ]


def test_fewer_than_five_large_bodies_uses_only_those():
    image = _segmentation_with_blobs([
        (20, 8, 5),
        (10, 8, 5),
        (4, 5, 2),
    ])
    assert _selected_sizes(image, nr_seeds=5, min_component_voxels=100) == [
        800, 400,
    ]


def test_no_cutoff_still_takes_the_requested_largest_bodies():
    image = _segmentation_with_blobs([
        (20, 8, 5),
        (10, 8, 5),
        (4, 5, 2),
    ])
    assert _selected_sizes(image, nr_seeds=2, min_component_voxels=None) == [
        800, 400,
    ]


def test_all_bodies_below_cutoff_keeps_the_largest():
    image = _segmentation_with_blobs([
        (4, 5, 2),
        (2, 2, 2),
    ])
    assert _selected_sizes(image, nr_seeds=5, min_component_voxels=100) == [40]
