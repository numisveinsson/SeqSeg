"""Assembly spacing cap and resample size guards (CI OOM prevention)."""

import numpy as np
import pytest
import SimpleITK as sitk

from seqseg.config_models import AlgorithmConfig
from seqseg.modules.sitk_functions import (
    capped_target_spacing,
    resample_to_spacing,
    resampled_grid_size,
)


def test_capped_target_spacing_mm_caps_coarse_dims():
    target = capped_target_spacing((0.65, 0.65, 1.0), 0.3, "mm")
    assert target == pytest.approx([0.3, 0.3, 0.3])


def test_capped_target_spacing_already_fine_returns_none():
    assert capped_target_spacing((0.2, 0.2, 0.2), 0.3, "mm") is None


def test_capped_target_spacing_cm_is_tenfold_finer():
    # Image spacing stored in mm, but -unit cm treats 0.3 mm as 0.03.
    target = capped_target_spacing((0.65, 0.65, 1.0), 0.3, "cm")
    assert target == pytest.approx([0.03, 0.03, 0.03])
    n_vox = int(np.prod(resampled_grid_size(
        (512, 512, 258), (0.65, 0.65, 1.0), target
    )))
    # ~1e9 voxels — this is what SIGKILL'd the macos CI job.
    assert n_vox > 200_000_000


def test_resampled_grid_size_identity():
    size = resampled_grid_size((10, 20, 30), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    np.testing.assert_array_equal(size, [10, 20, 30])


def test_resample_to_spacing_refuses_huge_grid():
    img = sitk.Image(8, 8, 8, sitk.sitkUInt8)
    img.SetSpacing((1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="max_voxels"):
        resample_to_spacing(img, (0.01, 0.01, 0.01), max_voxels=1_000)


def test_global_test_config_skips_assembly_upsample():
    cfg = AlgorithmConfig.from_name("global_test")
    assert cfg.get("CENT_MAX_SPACING") is None


def test_resample_to_spacing_small_grid_runs():
    img = sitk.Image(4, 4, 4, sitk.sitkFloat32)
    img.SetSpacing((2.0, 2.0, 2.0))
    out = resample_to_spacing(img, (1.0, 1.0, 1.0))
    assert tuple(out.GetSize()) == (8, 8, 8)
    assert out.GetSpacing() == pytest.approx((1.0, 1.0, 1.0))
