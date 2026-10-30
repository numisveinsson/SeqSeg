"""Assembly spacing cap and resample size guards (CI OOM prevention)."""

import numpy as np
import pytest
import SimpleITK as sitk

from seqseg.config_models import AlgorithmConfig
from seqseg.modules.sitk_functions import (
    DEFAULT_MAX_RESAMPLE_VOXELS,
    capped_target_spacing,
    fit_spacing_to_max_voxels,
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
    # ~1e12 voxels — this is what SIGKILL'd the macos CI job.
    assert n_vox > DEFAULT_MAX_RESAMPLE_VOXELS


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


def test_fit_spacing_keeps_target_when_under_limit():
    orig_size = (64, 256, 256)
    orig_spacing = (0.15, 0.117188, 0.117188)
    target = [0.03, 0.03, 0.03]
    n_vox = int(np.prod(resampled_grid_size(orig_size, orig_spacing, target)))
    assert n_vox == 320_000_000
    fitted = fit_spacing_to_max_voxels(orig_size, orig_spacing, target)
    assert fitted == pytest.approx(target)


def test_fit_spacing_coarsens_to_voxel_budget():
    orig_size = (512, 512, 258)
    orig_spacing = (0.65, 0.65, 1.0)
    target = [0.03, 0.03, 0.03]
    fitted = fit_spacing_to_max_voxels(orig_size, orig_spacing, target)
    assert fitted is not None
    n_vox = int(np.prod(resampled_grid_size(orig_size, orig_spacing, fitted)))
    assert n_vox <= DEFAULT_MAX_RESAMPLE_VOXELS
    assert all(f <= s for f, s in zip(fitted, orig_spacing))
    assert any(f < s for f, s in zip(fitted, orig_spacing))


def test_fit_spacing_returns_none_if_original_already_over_budget():
    fitted = fit_spacing_to_max_voxels(
        (100, 100, 100),
        (1.0, 1.0, 1.0),
        (0.01, 0.01, 0.01),
        max_voxels=1_000,
    )
    assert fitted is None


def _plans(spacing, transpose=None, configuration="3d_fullres"):
    plans = {
        "configurations": {
            configuration: {"spacing": list(spacing)},
        }
    }
    if transpose is not None:
        plans["transpose_forward"] = list(transpose)
    return plans


def test_plan_spacing_identity_transpose_is_reversed_sitk_order():
    from seqseg.modules.sitk_functions import plan_spacing_to_image_units

    # plans.json stores (z, y, x). SimpleITK order is (x, y, z).
    got = plan_spacing_to_image_units(
        [1.5, 0.7, 0.5], [0, 1, 2], (1.0, 1.0, 2.0), scale=1.0
    )
    assert got == pytest.approx([0.5, 0.7, 1.5])


def test_plan_spacing_divides_by_scale_into_image_units():
    from seqseg.modules.sitk_functions import plan_spacing_to_image_units

    # 0.05 cm model spacing, image header in mm, -scale 0.1.
    got = plan_spacing_to_image_units(
        [0.05, 0.05, 0.05], [0, 1, 2], (1.0, 1.0, 1.0), scale=0.1
    )
    assert got == pytest.approx([0.5, 0.5, 0.5])


def test_plan_spacing_follows_transpose_forward():
    from seqseg.modules.sitk_functions import plan_spacing_to_image_units

    got = plan_spacing_to_image_units(
        [0.4, 1.5, 0.8], [2, 0, 1], (1.0, 2.0, 3.0), scale=1.0
    )
    assert got == pytest.approx([0.4, 0.8, 1.5])


def test_plan_spacing_2d_keeps_through_plane_axis():
    from seqseg.modules.sitk_functions import plan_spacing_to_image_units

    got = plan_spacing_to_image_units(
        [0.5, 0.5], [0, 1, 2], (1.0, 1.0, 2.0), scale=1.0
    )
    assert got == pytest.approx([0.5, 0.5, 2.0])


def test_finer_axis_spacing_keeps_native_where_it_is_finer():
    from seqseg.modules.sitk_functions import (
        automatic_resample_spacing,
        finer_axis_spacing,
    )

    assert finer_axis_spacing((0.3, 0.8, 1.0), (0.5, 0.5, 0.5)) == pytest.approx(
        [0.3, 0.5, 0.5]
    )
    assert finer_axis_spacing((0.2, 0.2, 0.2), (0.5, 0.5, 0.5)) is None
    target, ideal = automatic_resample_spacing(
        (0.3, 0.8, 1.0), (8, 8, 8), [(0.5, 0.5, 0.5)]
    )
    assert target == pytest.approx([0.3, 0.5, 0.5])
    assert ideal == pytest.approx([0.3, 0.5, 0.5])


def test_automatic_resample_spacing_respects_voxel_budget():
    from seqseg.modules.sitk_functions import automatic_resample_spacing

    target, ideal = automatic_resample_spacing(
        (0.65, 0.65, 1.0),
        (512, 512, 258),
        [(0.03, 0.03, 0.03)],
    )
    assert ideal == pytest.approx([0.03, 0.03, 0.03])
    assert target is not None
    n_vox = int(np.prod(resampled_grid_size((512, 512, 258), (0.65, 0.65, 1.0), target)))
    assert n_vox <= DEFAULT_MAX_RESAMPLE_VOXELS
    assert any(t < s for t, s in zip(target, (0.65, 0.65, 1.0)))


def test_model_spacing_reads_plans_json(tmp_path):
    import json

    from seqseg.modules.sitk_functions import model_spacing_in_image_units

    folder = tmp_path / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    folder.mkdir()
    (folder / "plans.json").write_text(
        json.dumps(_plans([0.05, 0.05, 0.08], transpose=[0, 1, 2]))
    )
    got = model_spacing_in_image_units(folder, (1.0, 1.0, 2.0), scale=0.1)
    assert got == pytest.approx([0.8, 0.5, 0.5])


def test_resample_inputs_uses_finer_model_spacing(tmp_path):
    import json

    from seqseg.modules.sitk_functions import resample_inputs_for_case

    img_path = tmp_path / "case.mha"
    img = sitk.Image(4, 4, 4, sitk.sitkFloat32)
    img.SetSpacing((1.0, 1.0, 2.0))
    sitk.WriteImage(img, str(img_path))
    folder = tmp_path / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    folder.mkdir()
    (folder / "plans.json").write_text(
        json.dumps(_plans([1.0, 0.5, 0.5], transpose=[0, 1, 2]))
    )
    out = tmp_path / "out"
    image_out, seg_out = resample_inputs_for_case(
        str(img_path),
        None,
        str(out),
        "case",
        ".mha",
        [(str(folder), 1.0)],
    )
    assert seg_out is None
    assert image_out != str(img_path)
    resampled = sitk.ReadImage(image_out)
    assert resampled.GetSpacing() == pytest.approx((0.5, 0.5, 1.0))
    assert tuple(resampled.GetSize()) == (8, 8, 8)


def test_resample_inputs_keeps_native_when_image_is_finer(tmp_path):
    import json

    from seqseg.modules.sitk_functions import resample_inputs_for_case

    img_path = tmp_path / "case.mha"
    img = sitk.Image(4, 4, 4, sitk.sitkFloat32)
    img.SetSpacing((0.2, 0.2, 0.2))
    sitk.WriteImage(img, str(img_path))
    folder = tmp_path / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    folder.mkdir()
    (folder / "plans.json").write_text(
        json.dumps(_plans([0.5, 0.5, 0.5]))
    )
    image_out, _seg = resample_inputs_for_case(
        str(img_path),
        None,
        str(tmp_path / "out"),
        "case",
        ".mha",
        [(str(folder), 1.0)],
    )
    assert image_out == str(img_path)


def test_resample_inputs_keeps_native_without_model_specs(tmp_path):
    from seqseg.modules.sitk_functions import resample_inputs_for_case

    img_path = tmp_path / "case.mha"
    img = sitk.Image(4, 4, 4, sitk.sitkFloat32)
    img.SetSpacing((0.65, 0.65, 1.0))
    sitk.WriteImage(img, str(img_path))
    image_out, seg_out = resample_inputs_for_case(
        str(img_path),
        None,
        str(tmp_path / "out"),
        "case",
        ".mha",
        [],
    )
    assert image_out == str(img_path)
    assert seg_out is None
    assert not (tmp_path / "out").exists()


def test_resample_inputs_explicit_spacing_overrides_model(tmp_path):
    from seqseg.modules.sitk_functions import resample_inputs_for_case

    img_path = tmp_path / "case.mha"
    img = sitk.Image(4, 4, 4, sitk.sitkFloat32)
    img.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(img, str(img_path))
    image_out, _seg = resample_inputs_for_case(
        str(img_path),
        None,
        str(tmp_path / "out"),
        "case",
        ".mha",
        [],
        explicit_spacing=(0.5, 0.5, 0.5),
    )
    resampled = sitk.ReadImage(image_out)
    assert resampled.GetSpacing() == pytest.approx((0.5, 0.5, 0.5))
