"""Tests for ImageCAS / CAS-X direction canonicalization."""

import numpy as np
import SimpleITK as sitk

from seqseg.scripts.prepare_casx_dataset import (
    IDENTITY_DIRECTION,
    canonicalize_identity_direction,
)


def test_canonicalize_flips_negative_y_and_keeps_physical_point():
    arr = np.arange(2 * 4 * 3, dtype=np.int16).reshape(2, 4, 3)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.5, 0.4, 1.0))
    img.SetOrigin((1.0, 10.0, 3.0))
    img.SetDirection((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))

    ijk = (1, 3, 0)
    phys = np.array(img.TransformIndexToPhysicalPoint(ijk))

    out = canonicalize_identity_direction(img)
    assert tuple(round(v, 9) for v in out.GetDirection()) == IDENTITY_DIRECTION

    ny = img.GetSize()[1]
    ijk_flipped = (ijk[0], ny - 1 - ijk[1], ijk[2])
    phys_out = np.array(out.TransformIndexToPhysicalPoint(ijk_flipped))
    np.testing.assert_allclose(phys, phys_out)

    # Sampler-style mapping (ignores direction) must land in-bounds.
    origin = np.array(out.GetOrigin())
    spacing = np.array(out.GetSpacing())
    size = np.array(out.GetSize())
    idx = (phys_out - origin) / spacing
    assert np.all(idx >= 0) and np.all(idx <= size - 1)


def test_canonicalize_noop_when_already_identity():
    img = sitk.GetImageFromArray(np.ones((2, 2, 2), dtype=np.uint8))
    img.SetDirection(IDENTITY_DIRECTION)
    origin = img.GetOrigin()
    out = canonicalize_identity_direction(img)
    assert out.GetDirection() == IDENTITY_DIRECTION
    assert out.GetOrigin() == origin
    np.testing.assert_array_equal(sitk.GetArrayFromImage(img), sitk.GetArrayFromImage(out))
