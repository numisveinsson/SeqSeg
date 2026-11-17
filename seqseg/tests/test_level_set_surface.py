"""Logit level set used to contour the final surface."""

import numpy as np
import SimpleITK as sitk

from seqseg.modules import vtk_functions as vf
from seqseg.modules.sitk_functions import (
    keep_level_set_components,
    probability_to_level_set,
)

SPACING = 0.5


def _sigmoid_ball(center, radius, size=48, sharpness=10.0):
    """Probability image: a sigmoid of signed distance to a sphere."""
    idx = np.indices((size,) * 3).astype(np.float64) * SPACING
    dist = np.sqrt(sum((idx[k] - center[2 - k]) ** 2 for k in range(3)))
    prob = 1.0 / (1.0 + np.exp(-sharpness * (radius - dist)))
    img = sitk.GetImageFromArray(prob.astype(np.float32))
    img.SetSpacing((SPACING,) * 3)
    return img


def _radii(surface, center):
    n = surface.GetNumberOfPoints()
    pts = np.array([surface.GetPoint(i) for i in range(n)])
    return np.linalg.norm(pts - np.asarray(center), axis=1)


def test_level_set_is_finite_and_zero_at_threshold():
    img = sitk.Image(2, 2, 2, sitk.sitkFloat32)
    img[0, 0, 0] = 1.0
    img[1, 0, 0] = 0.3
    level_set = sitk.GetArrayFromImage(
        probability_to_level_set(img, threshold=0.3)
    )
    assert np.isfinite(level_set).all()
    assert level_set[0, 0, 0] > 0 > level_set[1, 1, 1]
    assert abs(level_set[0, 0, 1]) < 1e-6


def test_level_set_contour_recovers_sphere_radius():
    center, radius = (11.9, 12.2, 12.4), 6.3
    prob = _sigmoid_ball(center, radius)
    binary = sitk.BinaryThreshold(prob, lowerThreshold=0.5, upperThreshold=1)

    on_mask = _radii(vf.evaluate_surface(binary, 1), center)
    on_logit = _radii(
        vf.evaluate_surface(probability_to_level_set(prob), 0), center
    )

    assert abs(on_logit.mean() - radius) < 0.02 * SPACING
    assert on_logit.std() < 0.05 * SPACING
    # The mask contour sits on voxel centers, inside the true surface.
    assert radius - on_mask.mean() > 0.25 * SPACING


def test_keep_level_set_components_drops_unseeded_piece():
    kept_center, stray_center = (7.0, 12.0, 12.0), (18.0, 12.0, 12.0)
    prob = sitk.Maximum(
        _sigmoid_ball(kept_center, 3.0), _sigmoid_ball(stray_center, 2.0)
    )
    level_set = probability_to_level_set(prob)
    binary = sitk.BinaryThreshold(
        _sigmoid_ball(kept_center, 3.0), lowerThreshold=0.5, upperThreshold=1
    )

    out = keep_level_set_components(level_set, binary, [np.array(kept_center)])

    before = sitk.GetArrayFromImage(level_set)
    after = sitk.GetArrayFromImage(out)
    stray = sitk.GetArrayFromImage(_sigmoid_ball(stray_center, 2.0)) >= 0.5
    assert (after[stray] < 0).all()
    assert np.array_equal(after[~stray], before[~stray])


def test_empty_probability_gives_no_surface():
    prob = sitk.Image(8, 8, 8, sitk.sitkFloat32)
    surface = vf.evaluate_surface(probability_to_level_set(prob), 0)
    assert surface.GetNumberOfPoints() == 0
