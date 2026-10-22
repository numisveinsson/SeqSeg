"""Unit tests for seqseg.pipeline.train helpers (no GPU / sampler required)."""

from unittest.mock import MagicMock, patch

import pytest

import os
import shutil
import subprocess

from pathlib import Path

import numpy as np

from seqseg.pipeline.train import (
    TrainDependencyError,
    _STAGING_DIR_NAME,
    _cap_from_sampler_config,
    _detect_img_ext,
    _dir_for_sampler,
    _harden_sampler_vtk,
    _link_or_copy,
    _load_sampler_config,
    _normalize_img_ext,
    _pick_config_file,
    _reduction_or_nan,
    _resolve_img_ext,
    _resolve_sampler_config_file,
    _safe_add_image_stats,
    _safe_add_local_stats,
    _staging_is_safe_to_remove,
    dataset_id_from_name,
    expected_nnunet_dataset_name,
    prepare_training_dataset,
    run_nnunet_training,
)


@pytest.fixture(autouse=True)
def _stub_nnunet_preprocess_runners(monkeypatch):
    """Sampling tests should not launch nnU-Net. Override to assert the call."""
    plan = MagicMock()
    only = MagicMock()
    monkeypatch.setattr(
        "seqseg.pipeline.train._run_nnunet_plan_and_preprocess", plan
    )
    monkeypatch.setattr(
        "seqseg.pipeline.train._run_nnunet_preprocess_only", only
    )
    return plan, only


@pytest.fixture(autouse=True)
def _stub_missing_sampler_config(monkeypatch):
    """Prepare tests do not ship a sampler YAML. A missing name is an error."""

    def _stub(config):
        if isinstance(config, dict):
            return dict(config)
        return {"IMG_EXT": ".nrrd"}

    monkeypatch.setattr("seqseg.pipeline.train._load_sampler_config", _stub)


def _make_training_data(tmp_path, *, surfaces=False, img_ext=".nrrd"):
    data = tmp_path / "data"
    (data / "images").mkdir(parents=True)
    (data / "centerlines").mkdir()
    (data / "images" / f"case1{img_ext}").write_bytes(b"x")
    (data / "centerlines" / "case1.vtp").write_bytes(b"x")
    if surfaces:
        (data / "surfaces").mkdir()
        (data / "surfaces" / "case1.vtp").write_bytes(b"x")
    return str(data)


def _make_patch_dirs(extracted, modality="ct"):
    img_dir = extracted / f"{modality}_train"
    mask_dir = extracted / f"{modality}_train_masks"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir()
    (img_dir / "a.nrrd").write_bytes(b"x")
    (mask_dir / "a.nrrd").write_bytes(b"x")
    return img_dir, mask_dir


def test_dir_for_sampler_adds_trailing_sep(tmp_path):
    raw = str(tmp_path / "seqseg_train")
    out = _dir_for_sampler(raw)
    assert out.endswith(os.sep)
    assert out.rstrip("/\\") == os.path.abspath(raw)
    # Sampler concatenates outdir + "ct_train_Sample_stats.csv"
    stats = out + "ct_train_Sample_stats.csv"
    assert os.path.dirname(stats) == os.path.abspath(raw)


def test_align_dataset_dir_renames_four_digit_sampler_name(tmp_path):
    from seqseg.pipeline.train import _align_dataset_dir_with_nnunet

    raw = tmp_path / "nnUNet_raw"
    written = raw / "Dataset0999_MYDATACT"
    (written / "imagesTr").mkdir(parents=True)
    (written / "imagesTr" / "a.nii.gz").write_bytes(b"x")
    aligned = _align_dataset_dir_with_nnunet(
        str(written), "Dataset999_MYDATACT", str(raw)
    )
    assert os.path.basename(aligned) == "Dataset999_MYDATACT"
    assert (raw / "Dataset999_MYDATACT" / "imagesTr" / "a.nii.gz").is_file()
    assert not written.exists()


def test_expected_nnunet_dataset_name_padding():
    assert expected_nnunet_dataset_name("AORTAS", 1, "ct") == "Dataset001_AORTASCT"
    assert expected_nnunet_dataset_name("AORTAS", 10, "mr") == "Dataset010_AORTASMR"
    assert expected_nnunet_dataset_name("MYDATA", 999, "CT") == "Dataset999_MYDATACT"


def test_dataset_id_from_name_errors():
    with pytest.raises(ValueError):
        dataset_id_from_name("not_a_dataset")
    with pytest.raises(ValueError):
        dataset_id_from_name("Dataset_FOO")


def test_prepare_requires_sampler():
    with patch(
        "seqseg.pipeline.train._require_sampler",
        side_effect=TrainDependencyError("nope"),
    ):
        with pytest.raises(TrainDependencyError):
            prepare_training_dataset(
                "/tmp/data",
                "/tmp/out",
                name="X",
                dataset_number=1,
            )


def test_prepare_calls_sampler_apis(tmp_path):
    extract = MagicMock()
    write = MagicMock(return_value=str(tmp_path / "Dataset999_MYDATACT"))
    data = _make_training_data(tmp_path)
    extracted = tmp_path / "extracted"
    _make_patch_dirs(extracted)

    with patch(
        "seqseg.pipeline.train._require_sampler",
        return_value=(extract, write),
    ):
        result = prepare_training_dataset(
            data,
            str(extracted),
            name="MYDATA",
            dataset_number=999,
            nnunet_raw=str(tmp_path / "nnUNet_raw"),
            modality="CT",
        )

    extract.assert_called_once()
    write.assert_called_once()
    outdir_arg = extract.call_args.kwargs["outdir"]
    assert outdir_arg.endswith(os.sep)
    assert os.path.basename(os.path.normpath(outdir_arg)) == "extracted"
    assert extract.call_args.kwargs["config"]["IMG_EXT"] == ".nrrd"
    assert extract.call_args.kwargs["yes"] is True
    assert result.dataset_names == ["Dataset999_MYDATACT"]
    assert result.modalities == ["CT"]
    assert result.removed_staging is False
    assert (extracted / "ct_train" / "a.nrrd").is_file()
    assert write.call_args.kwargs["outdir"] == str(tmp_path / "nnUNet_raw")


def test_pick_config_file_prefers_overlay_that_sets_max_samples(tmp_path):
    (tmp_path / "test.yaml").write_text("MOVE_DIST: 10\n")
    (tmp_path / "test.yml").write_text("MAX_SAMPLES: 12\n")
    picked = _pick_config_file(str(tmp_path), "test")
    assert picked.endswith("test.yml")


def test_resolve_sampler_config_file_bare_name(tmp_path, monkeypatch):
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "test.yaml").write_text("MOVE_DIST: 10\n")
    (cfg / "test.yml").write_text("MAX_SAMPLES: 12\nMOVE_DIST: 8\n")
    monkeypatch.setattr(
        "seqseg.pipeline.train._sampler_config_dirs", lambda: [str(cfg)]
    )
    resolved = _resolve_sampler_config_file("test")
    assert resolved.endswith("test.yml")
    explicit = _resolve_sampler_config_file("test.yaml")
    assert explicit.endswith("test.yaml")


def test_prepare_missing_config_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "seqseg.pipeline.train._load_sampler_config", _load_sampler_config
    )
    monkeypatch.setattr(
        "seqseg.pipeline.train._sampler_config_dirs", lambda: [str(tmp_path)]
    )
    with patch(
        "seqseg.pipeline.train._require_sampler",
        return_value=(MagicMock(), MagicMock()),
    ):
        with pytest.raises(FileNotFoundError, match="was not found"):
            prepare_training_dataset(
                str(tmp_path / "data"),
                str(tmp_path / "extracted"),
                name="MYDATA",
                dataset_number=1,
                nnunet_raw=str(tmp_path / "nnUNet_raw"),
                config="does_not_exist",
            )


def test_cap_from_sampler_config_cli_overrides_yaml():
    assert _cap_from_sampler_config({"MAX_SAMPLES": 12}, None) == 12
    assert _cap_from_sampler_config({"MAX_SAMPLES": 12}, 3) == 3
    assert _cap_from_sampler_config({}, None) is None


def test_prepare_passes_yaml_max_samples(tmp_path):
    extract = MagicMock()
    write = MagicMock(return_value=str(tmp_path / "Dataset999_MYDATACT"))
    data = _make_training_data(tmp_path)
    extracted = tmp_path / "extracted"
    _make_patch_dirs(extracted)

    with patch(
        "seqseg.pipeline.train._require_sampler",
        return_value=(extract, write),
    ), patch(
        "seqseg.pipeline.train._load_sampler_config",
        return_value={"IMG_EXT": ".nrrd", "MAX_SAMPLES": 12},
    ), patch(
        "seqseg.pipeline.train._resolve_sampler_config_file",
        return_value=str(tmp_path / "test.yml"),
    ):
        prepare_training_dataset(
            data,
            str(extracted),
            name="MYDATA",
            dataset_number=999,
            nnunet_raw=str(tmp_path / "nnUNet_raw"),
            modality="CT",
            config="test",
        )

    assert extract.call_args.kwargs["max_samples"] == 12
    assert extract.call_args.kwargs["config"]["MAX_SAMPLES"] == 12


def test_prepare_global_volumes_calls_gather_not_extract(tmp_path):
    extract = MagicMock()
    write = MagicMock(return_value=str(tmp_path / "Dataset999_MYDATACT"))
    gather = MagicMock()
    data = _make_training_data(tmp_path)
    extracted = tmp_path / "extracted"
    _make_patch_dirs(extracted)

    with patch(
        "seqseg.pipeline.train._require_sampler",
        return_value=(extract, write),
    ), patch(
        "seqseg.pipeline.train._require_gather_global_volumes",
        return_value=gather,
    ):
        result = prepare_training_dataset(
            data,
            str(extracted),
            name="MYDATA",
            dataset_number=999,
            nnunet_raw=str(tmp_path / "nnUNet_raw"),
            modality="CT",
            global_volumes=True,
        )

    extract.assert_not_called()
    gather.assert_called_once()
    write.assert_called_once()
    kwargs = gather.call_args.kwargs
    assert kwargs["config"]["IMG_EXT"] == ".nrrd"
    assert kwargs["modality"] == "CT"
    assert kwargs["yes"] is True
    assert "max_samples" not in kwargs
    write_kwargs = write.call_args.kwargs
    assert write_kwargs["name"] == "MYDATA"
    assert write_kwargs["dataset_number"] == 999
    assert write_kwargs["modality"] == "ct"
    assert result.dataset_names == ["Dataset999_MYDATACT"]
    assert result.modalities == ["CT"]


def test_prepare_global_volumes_requires_recent_sampler(tmp_path):
    extract = MagicMock()
    write = MagicMock()
    with patch(
        "seqseg.pipeline.train._require_sampler",
        return_value=(extract, write),
    ), patch(
        "seqseg.pipeline.train._require_gather_global_volumes",
        side_effect=TrainDependencyError("upgrade sampler"),
    ):
        with pytest.raises(TrainDependencyError, match="upgrade sampler"):
            prepare_training_dataset(
                _make_training_data(tmp_path),
                str(tmp_path / "extracted"),
                name="MYDATA",
                dataset_number=999,
                nnunet_raw=str(tmp_path / "nnUNet_raw"),
                global_volumes=True,
            )
    extract.assert_not_called()
    write.assert_not_called()


def test_prepare_global_volumes_skip_sample_converts_only(tmp_path):
    extract = MagicMock()
    write = MagicMock(return_value=str(tmp_path / "Dataset999_MYDATACT"))
    gather = MagicMock()
    extracted = tmp_path / "extracted"
    _make_patch_dirs(extracted)

    with patch(
        "seqseg.pipeline.train._require_sampler",
        return_value=(extract, write),
    ), patch(
        "seqseg.pipeline.train._require_gather_global_volumes",
        return_value=gather,
    ):
        prepare_training_dataset(
            str(tmp_path / "data"),
            str(extracted),
            name="MYDATA",
            dataset_number=999,
            nnunet_raw=str(tmp_path / "nnUNet_raw"),
            skip_sample=True,
            global_volumes=True,
        )

    extract.assert_not_called()
    gather.assert_not_called()
    write.assert_called_once()


def test_detect_img_ext_prefers_nii_gz(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "716.nii.gz").write_bytes(b"x")
    (images / "notes.txt").write_text("nope")
    assert _detect_img_ext(images) == ".nii.gz"
    assert _normalize_img_ext("nii.gz") == ".nii.gz"
    assert _resolve_img_ext(str(tmp_path), {"IMG_EXT": ".nrrd"}) == ".nii.gz"


def test_prepare_detects_nii_gz_and_sets_sampler_config(tmp_path):
    extract = MagicMock()
    write = MagicMock(return_value=str(tmp_path / "Dataset999_MYDATACT"))
    data = _make_training_data(tmp_path, img_ext=".nii.gz")
    extracted = tmp_path / "extracted"
    _make_patch_dirs(extracted)

    with patch(
        "seqseg.pipeline.train._require_sampler",
        return_value=(extract, write),
    ):
        prepare_training_dataset(
            data,
            str(extracted),
            name="MYDATA",
            dataset_number=999,
            nnunet_raw=str(tmp_path / "nnUNet_raw"),
        )

    assert extract.call_args.kwargs["config"]["IMG_EXT"] == ".nii.gz"


def test_prepare_explicit_img_ext_overrides_detection(tmp_path):
    extract = MagicMock()
    write = MagicMock(return_value=str(tmp_path / "Dataset999_MYDATACT"))
    data = Path(_make_training_data(tmp_path, img_ext=".nii.gz"))
    (data / "images" / "case1.mha").write_bytes(b"x")
    extracted = tmp_path / "extracted"
    _make_patch_dirs(extracted)

    with patch(
        "seqseg.pipeline.train._require_sampler",
        return_value=(extract, write),
    ):
        prepare_training_dataset(
            str(data),
            str(extracted),
            name="MYDATA",
            dataset_number=999,
            nnunet_raw=str(tmp_path / "nnUNet_raw"),
            img_ext="mha",
        )

    assert extract.call_args.kwargs["config"]["IMG_EXT"] == ".mha"


def test_prepare_multi_modality_increments_ids(tmp_path):
    extract = MagicMock()
    write = MagicMock(
        side_effect=[
            str(tmp_path / "Dataset999_MYDATACT"),
            str(tmp_path / "Dataset1000_MYDATAMR"),
        ]
    )

    with patch(
        "seqseg.pipeline.train._require_sampler",
        return_value=(extract, write),
    ):
        result = prepare_training_dataset(
            str(tmp_path / "data"),
            str(tmp_path / "extracted"),
            name="MYDATA",
            dataset_number=999,
            nnunet_raw=str(tmp_path / "nnUNet_raw"),
            modality="CT,MR",
            skip_sample=True,
        )

    assert write.call_count == 2
    assert result.dataset_names == [
        "Dataset999_MYDATACT",
        "Dataset1000_MYDATAMR",
    ]


def test_prepare_preflight_rejects_missing_images(tmp_path):
    extract = MagicMock()
    write = MagicMock()
    empty = tmp_path / "data"
    empty.mkdir()
    with patch(
        "seqseg.pipeline.train._require_sampler",
        return_value=(extract, write),
    ):
        with pytest.raises(FileNotFoundError, match="images/ and centerlines"):
            prepare_training_dataset(
                str(empty),
                str(tmp_path / "extracted"),
                name="MYDATA",
                dataset_number=999,
                nnunet_raw=str(tmp_path / "nnUNet_raw"),
            )
    extract.assert_not_called()


def test_prepare_continues_if_stats_csv_missing_but_patches_exist(tmp_path):
    extracted = tmp_path / "extracted"
    _make_patch_dirs(extracted)
    extract = MagicMock(
        side_effect=FileNotFoundError(
            f"[Errno 2] No such file or directory: "
            f"'{extracted}/ct_train_Sample_stats.csv'"
        )
    )
    write = MagicMock(return_value=str(tmp_path / "Dataset999_MYDATACT"))

    with patch(
        "seqseg.pipeline.train._require_sampler",
        return_value=(extract, write),
    ):
        result = prepare_training_dataset(
            _make_training_data(tmp_path),
            str(extracted),
            name="MYDATA",
            dataset_number=999,
            nnunet_raw=str(tmp_path / "nnUNet_raw"),
        )

    write.assert_called_once()
    assert result.dataset_names == ["Dataset999_MYDATACT"]


def test_prepare_missing_stats_csv_without_patches_is_actionable(tmp_path):
    extract = MagicMock(
        side_effect=FileNotFoundError(
            "[Errno 2] No such file or directory: "
            "'/scratch/11178/numi/seqseg_train/ct_train_Sample_stats.csv'"
        )
    )
    write = MagicMock()

    with patch(
        "seqseg.pipeline.train._require_sampler",
        return_value=(extract, write),
    ):
        with pytest.raises(RuntimeError, match="--num-cores 1"):
            prepare_training_dataset(
                _make_training_data(tmp_path),
                str(tmp_path / "extracted"),
                name="MYDATA",
                dataset_number=999,
                nnunet_raw=str(tmp_path / "nnUNet_raw"),
            )
    write.assert_not_called()


def test_collect_arrays_skips_string_arrays():
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    from seqseg.modules.vtk_functions import collect_arrays

    pts = vtk.vtkPoints()
    pts.InsertNextPoint(0.0, 0.0, 0.0)
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)

    rad = numpy_to_vtk(np.array([1.5], dtype=np.float64))
    rad.SetName("MaximumInscribedSphereRadius")
    poly.GetPointData().AddArray(rad)

    names = vtk.vtkStringArray()
    names.SetName("ModelName")
    names.InsertNextValue("aorta")
    poly.GetPointData().AddArray(names)

    data = collect_arrays(poly.GetPointData())
    assert "MaximumInscribedSphereRadius" in data
    assert "ModelName" not in data
    assert float(data["MaximumInscribedSphereRadius"][0]) == 1.5


def test_convert_poly_to_image_without_scalars():
    import vtk

    from seqseg.modules.vtk_functions import convertPolyDataToImageData

    sphere = vtk.vtkSphereSource()
    sphere.SetCenter(4, 4, 4)
    sphere.SetRadius(2.0)
    sphere.Update()

    ref = vtk.vtkImageData()
    ref.SetDimensions(8, 8, 8)
    ref.SetSpacing(1.0, 1.0, 1.0)
    ref.SetOrigin(0, 0, 0)

    out = convertPolyDataToImageData(sphere.GetOutput(), ref)
    assert out.GetPointData().GetScalars() is not None
    assert out.GetDimensions() == (8, 8, 8)


def test_harden_sampler_vtk_patches_collect_arrays():
    pytest.importorskip("vascular_segment_sampler")
    import vascular_segment_sampler.sampling.functions as samp_fn

    from seqseg.modules.vtk_functions import collect_arrays

    _harden_sampler_vtk()
    assert samp_fn.collect_arrays is collect_arrays
    assert samp_fn.add_local_stats is _safe_add_local_stats
    assert samp_fn.add_image_stats is _safe_add_image_stats


def test_safe_add_local_stats_empty_blood_does_not_raise():
    stats, extra = _safe_add_local_stats(
        {},
        np.array([1.0, 2.0, 3.0]),
        0.0,
        np.array([]),
        np.zeros((2, 2, 2)),
        [1],
        None,
        np.ones((2, 2, 2)),
        0,
    )
    assert extra == 0
    assert np.isnan(stats["BLOOD_MIN"])
    assert np.isnan(stats["BLOOD_MEAN"])
    assert stats["GT_MIN"] == 0.0
    assert stats["POINT_CENT"] == [1.0, 2.0, 3.0]


def test_safe_add_local_stats_empty_largest_component():
    sitk = pytest.importorskip("SimpleITK")
    empty = sitk.Image(2, 2, 2, sitk.sitkUInt8)
    im = np.arange(8, dtype=float).reshape(2, 2, 2)
    stats, extra = _safe_add_local_stats(
        {},
        [0, 0, 0],
        0.0,
        im.ravel(),
        np.ones((2, 2, 2)),
        [1, 2],
        empty,
        im,
        0,
    )
    assert extra == 1
    assert np.isnan(stats["LARGEST_MIN"])
    assert np.isnan(stats["LARGEST_MEAN"])


def test_safe_add_image_stats_empty_volume():
    stats = _safe_add_image_stats({}, np.array([]))
    assert np.isnan(stats["IM_MIN"])
    assert np.isnan(stats["IM_MAX"])


def test_reduction_or_nan_nonempty():
    assert _reduction_or_nan(np.array([3.0, 1.0, 2.0]), np.amin) == 1.0
    assert np.isnan(_reduction_or_nan(np.array([]), np.amin))


def test_run_nnunet_training_requires_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SEQSEG_PATHS_FILE", str(tmp_path / "no_paths.yaml"))
    for key in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="nnU-Net paths"):
        run_nnunet_training(999)


def test_run_nnunet_training_passes_detected_device(tmp_path, monkeypatch):
    raw = tmp_path / "nnUNet_raw"
    pre = tmp_path / "nnUNet_preprocessed" / "Dataset001_X"
    res = tmp_path / "nnUNet_results"
    pre.mkdir(parents=True)
    raw.mkdir()
    res.mkdir()
    _isolate_nnunet_env(monkeypatch, raw, pre.parent, res)
    monkeypatch.setattr(
        "seqseg.pipeline.train.select_training_device", lambda: "mps"
    )

    with patch("seqseg.pipeline.train.subprocess.run") as run:
        run_nnunet_training(1, configuration="3d_fullres", fold="0")

    assert run.call_count == 1
    train_cmd = run.call_args[0][0]
    assert "nnUNetv2_train" in train_cmd[0]
    assert "nnUNetv2_plan_and_preprocess" not in train_cmd[0]
    assert "--c" not in train_cmd
    assert train_cmd[train_cmd.index("-device") + 1] == "mps"


def test_run_nnunet_training_requires_preprocessed_data(tmp_path, monkeypatch):
    raw = tmp_path / "nnUNet_raw"
    pre = tmp_path / "nnUNet_preprocessed"
    res = tmp_path / "nnUNet_results"
    for path in (raw, pre, res):
        path.mkdir()
    _isolate_nnunet_env(monkeypatch, raw, pre, res)
    with pytest.raises(FileNotFoundError, match="seqseg train prepare"):
        run_nnunet_training(999)


def test_run_nnunet_training_continue_appends_c(tmp_path, monkeypatch):
    raw = tmp_path / "nnUNet_raw"
    pre = tmp_path / "nnUNet_preprocessed" / "Dataset999_MYDATACT"
    checkpoint = (
        tmp_path
        / "nnUNet_results"
        / "Dataset999_MYDATACT"
        / "nnUNetTrainer__nnUNetPlans__3d_fullres"
        / "fold_0"
        / "checkpoint_latest.pth"
    )
    pre.mkdir(parents=True)
    raw.mkdir()
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"x")
    _isolate_nnunet_env(monkeypatch, raw, pre.parent, checkpoint.parents[3])
    monkeypatch.setattr(
        "seqseg.pipeline.train.select_training_device", lambda: "cpu"
    )

    with patch("seqseg.pipeline.train.subprocess.run") as run:
        run_nnunet_training(999, fold="0", continue_training=True)

    train_cmd = run.call_args[0][0]
    assert train_cmd[-1] == "--c"
    assert "nnUNetv2_plan_and_preprocess" not in train_cmd[0]


def test_run_nnunet_training_continue_requires_checkpoint(tmp_path, monkeypatch):
    raw = tmp_path / "nnUNet_raw"
    pre = tmp_path / "nnUNet_preprocessed" / "Dataset999_MYDATACT"
    res = tmp_path / "nnUNet_results" / "Dataset999_MYDATACT"
    pre.mkdir(parents=True)
    raw.mkdir()
    res.mkdir(parents=True)
    _isolate_nnunet_env(monkeypatch, raw, pre.parent, res.parent)
    with pytest.raises(FileNotFoundError, match="checkpoint_latest.pth"):
        run_nnunet_training(999, fold="0", continue_training=True)


def test_run_nnunet_training_continue_fold_all_skips_checkpoint_check(
    tmp_path, monkeypatch
):
    raw = tmp_path / "nnUNet_raw"
    pre = tmp_path / "nnUNet_preprocessed" / "Dataset999_MYDATACT"
    res = tmp_path / "nnUNet_results"
    pre.mkdir(parents=True)
    raw.mkdir()
    res.mkdir()
    _isolate_nnunet_env(monkeypatch, raw, pre.parent, res)
    monkeypatch.setattr(
        "seqseg.pipeline.train.select_training_device", lambda: "cpu"
    )
    with patch("seqseg.pipeline.train.subprocess.run") as run:
        run_nnunet_training(999, fold="all", continue_training=True)
    assert run.call_args[0][0][-1] == "--c"


def _isolate_nnunet_env(monkeypatch, raw, pre, res):
    monkeypatch.setenv("SEQSEG_PATHS_FILE", str(raw.parent / "no_paths.yaml"))
    monkeypatch.setenv("nnUNet_raw", str(raw))
    monkeypatch.setenv("nnUNet_preprocessed", str(pre))
    monkeypatch.setenv("nnUNet_results", str(res))


def test_prefer_hardlinks_patches_sampler_copy(tmp_path):
    pytest.importorskip("vascular_segment_sampler.nnunet.convert")
    import vascular_segment_sampler.nnunet.convert as convert_mod

    from seqseg.pipeline.train import _prefer_hardlinks_for_nnunet_copy

    src = tmp_path / "a.nrrd"
    src.write_bytes(b"volume")
    dst = tmp_path / "b.nrrd"
    orig = convert_mod.shutil.copy
    with _prefer_hardlinks_for_nnunet_copy():
        convert_mod.shutil.copy(str(src), str(dst))
    assert dst.stat().st_ino == src.stat().st_ino
    assert convert_mod.shutil.copy is orig


def test_link_or_copy_hardlinks_same_filesystem(tmp_path):
    src = tmp_path / "a.nrrd"
    src.write_bytes(b"volume")
    dst = tmp_path / "b.nrrd"
    _link_or_copy(str(src), str(dst), shutil.copy)
    assert src.stat().st_ino == dst.stat().st_ino
    src.unlink()
    assert dst.read_bytes() == b"volume"


def test_link_or_copy_falls_back_when_link_fails(tmp_path, monkeypatch):
    src = tmp_path / "a.nrrd"
    src.write_bytes(b"volume")
    dst = tmp_path / "b.nrrd"
    copied = {}

    def boom(*_args, **_kwargs):
        raise OSError("cross device")

    def fake_copy(src_path, dst_path, *_args, **_kwargs):
        copied["src"] = src_path
        Path(dst_path).write_bytes(Path(src_path).read_bytes())

    monkeypatch.setattr("seqseg.pipeline.train.os.link", boom)
    _link_or_copy(str(src), str(dst), fake_copy)
    assert copied["src"] == str(src)
    assert dst.read_bytes() == b"volume"


def test_staging_is_safe_to_remove(tmp_path):
    raw = tmp_path / "nn" / "nnUNet_raw"
    staging = tmp_path / "nn" / _STAGING_DIR_NAME
    assert _staging_is_safe_to_remove(str(staging), str(raw))
    assert not _staging_is_safe_to_remove(str(tmp_path / "nn"), str(raw))
    assert not _staging_is_safe_to_remove(str(raw), str(raw))
    assert not _staging_is_safe_to_remove(os.path.abspath(os.sep), str(raw))


def test_prepare_requires_nnunet_raw(tmp_path, monkeypatch):
    monkeypatch.setenv("SEQSEG_PATHS_FILE", str(tmp_path / "no_paths.yaml"))
    monkeypatch.delenv("nnUNet_raw", raising=False)
    with patch(
        "seqseg.pipeline.train._require_sampler",
        return_value=(MagicMock(), MagicMock()),
    ):
        with pytest.raises(ValueError, match="nnUNet_raw"):
            prepare_training_dataset(
                _make_training_data(tmp_path),
                name="MYDATA",
                dataset_number=999,
            )


def test_prepare_defaults_staging_and_removes_it(tmp_path, monkeypatch):
    monkeypatch.setenv("SEQSEG_PATHS_FILE", str(tmp_path / "no_paths.yaml"))
    monkeypatch.delenv("nnUNet_raw", raising=False)
    raw = tmp_path / "nn" / "nnUNet_raw"
    staging = tmp_path / "nn" / _STAGING_DIR_NAME
    _make_patch_dirs(staging)
    extract = MagicMock()
    write = MagicMock(return_value=str(raw / "Dataset999_MYDATACT"))

    with patch(
        "seqseg.pipeline.train._require_sampler",
        return_value=(extract, write),
    ):
        result = prepare_training_dataset(
            _make_training_data(tmp_path),
            name="MYDATA",
            dataset_number=999,
            nnunet_raw=str(raw),
        )

    outdir_arg = extract.call_args.kwargs["outdir"]
    assert os.path.basename(os.path.normpath(outdir_arg)) == _STAGING_DIR_NAME
    assert write.call_args.kwargs["outdir"] == str(raw)
    assert result.removed_staging is True
    assert not staging.exists()


def test_prepare_keep_extracted_leaves_managed_staging(tmp_path):
    raw = tmp_path / "nn" / "nnUNet_raw"
    staging = tmp_path / "nn" / _STAGING_DIR_NAME
    _make_patch_dirs(staging)
    extract = MagicMock()
    write = MagicMock(return_value=str(raw / "Dataset999_MYDATACT"))

    with patch(
        "seqseg.pipeline.train._require_sampler",
        return_value=(extract, write),
    ):
        result = prepare_training_dataset(
            _make_training_data(tmp_path),
            name="MYDATA",
            dataset_number=999,
            nnunet_raw=str(raw),
            keep_extracted=True,
        )

    assert result.removed_staging is False
    assert (staging / "ct_train" / "a.nrrd").is_file()


def test_prepare_failed_convert_keeps_staging(tmp_path):
    raw = tmp_path / "nn" / "nnUNet_raw"
    staging = tmp_path / "nn" / _STAGING_DIR_NAME
    _make_patch_dirs(staging)
    extract = MagicMock()
    write = MagicMock(side_effect=RuntimeError("convert failed"))

    with patch(
        "seqseg.pipeline.train._require_sampler",
        return_value=(extract, write),
    ):
        with pytest.raises(RuntimeError, match="convert failed"):
            prepare_training_dataset(
                _make_training_data(tmp_path),
                name="MYDATA",
                dataset_number=999,
                nnunet_raw=str(raw),
            )

    assert (staging / "ct_train" / "a.nrrd").is_file()


def test_prepare_skip_convert_keeps_staging(tmp_path):
    raw = tmp_path / "nn" / "nnUNet_raw"
    staging = tmp_path / "nn" / _STAGING_DIR_NAME
    write = MagicMock()

    with patch(
        "seqseg.pipeline.train._require_sampler",
        return_value=(MagicMock(), write),
    ):
        result = prepare_training_dataset(
            str(tmp_path / "data"),
            name="MYDATA",
            dataset_number=999,
            nnunet_raw=str(raw),
            skip_sample=True,
            skip_convert=True,
        )

    write.assert_not_called()
    assert result.removed_staging is False
    assert result.preprocessed is False
    assert staging.is_dir()


def test_prepare_skip_convert_does_not_preprocess(
    tmp_path, _stub_nnunet_preprocess_runners
):
    plan, only = _stub_nnunet_preprocess_runners
    raw = tmp_path / "nn" / "nnUNet_raw"
    with patch(
        "seqseg.pipeline.train._require_sampler",
        return_value=(MagicMock(), MagicMock()),
    ):
        prepare_training_dataset(
            str(tmp_path / "data"),
            name="MYDATA",
            dataset_number=999,
            nnunet_raw=str(raw),
            skip_sample=True,
            skip_convert=True,
        )
    plan.assert_not_called()
    only.assert_not_called()


def test_prepare_preprocesses_written_dataset_ids(
    tmp_path, _stub_nnunet_preprocess_runners
):
    plan, only = _stub_nnunet_preprocess_runners
    extract = MagicMock()
    write = MagicMock(return_value=str(tmp_path / "nnUNet_raw" / "Dataset999_MYDATACT"))
    data = _make_training_data(tmp_path)
    extracted = tmp_path / "extracted"
    _make_patch_dirs(extracted)

    with patch(
        "seqseg.pipeline.train._require_sampler",
        return_value=(extract, write),
    ):
        result = prepare_training_dataset(
            data,
            str(extracted),
            name="MYDATA",
            dataset_number=999,
            nnunet_raw=str(tmp_path / "nnUNet_raw"),
            modality="CT",
            num_cores=4,
        )

    only.assert_not_called()
    plan.assert_called_once_with(
        str(tmp_path / "nnUNet_raw"),
        [999],
        configuration="3d_fullres",
        np=4,
        plans="nnUNetPlans",
    )
    assert result.preprocessed is True


def test_resolve_nnunet_cli_uses_python_directory(tmp_path, monkeypatch):
    from seqseg.pipeline.train import _resolve_nnunet_cli

    bindir = tmp_path / "bin"
    bindir.mkdir()
    script = bindir / "nnUNetv2_plan_and_preprocess"
    script.write_text("#!/bin/sh\n")
    script.chmod(0o755)
    monkeypatch.setattr(
        "seqseg.pipeline.train.sys.executable", str(bindir / "python")
    )
    monkeypatch.setattr("seqseg.pipeline.train.shutil.which", lambda _exe: None)
    assert _resolve_nnunet_cli("nnUNetv2_plan_and_preprocess") == str(script)


def test_plan_and_preprocess_command_defaults():
    from seqseg.pipeline.train import _plan_and_preprocess_command

    cmd = _plan_and_preprocess_command(999, "3d_fullres", 4, "nnUNetPlans")
    assert "nnUNetv2_plan_and_preprocess" in cmd[0]
    assert cmd[cmd.index("-d") + 1] == "999"
    assert cmd[cmd.index("-c") + 1] == "3d_fullres"
    assert cmd[cmd.index("-np") + 1] == "4"
    assert "-pl" not in cmd
    assert "-overwrite_plans_name" not in cmd


def test_plan_and_preprocess_command_custom_plans():
    from seqseg.pipeline.train import _plan_and_preprocess_command

    cmd = _plan_and_preprocess_command(999, "2d", 1, "MyPlans")
    assert cmd[cmd.index("-overwrite_plans_name") + 1] == "MyPlans"
    assert "-pl" not in cmd


def test_preprocess_only_command():
    from seqseg.pipeline.train import _preprocess_only_command

    cmd = _preprocess_only_command(999, "2d", 8, "nnUNetPlans")
    assert "nnUNetv2_preprocess" in cmd[0]
    assert cmd[cmd.index("-d") + 1] == "999"
    assert cmd[cmd.index("-plans_name") + 1] == "nnUNetPlans"
    assert cmd[cmd.index("-c") + 1] == "2d"
    assert cmd[cmd.index("-np") + 1] == "8"


def test_preprocess_only_does_not_sample(tmp_path, monkeypatch):
    raw = tmp_path / "nnUNet_raw" / "Dataset999_MYDATACT"
    pre = tmp_path / "nnUNet_preprocessed" / "Dataset999_MYDATACT"
    res = tmp_path / "nnUNet_results"
    raw.mkdir(parents=True)
    pre.mkdir(parents=True)
    (pre / "nnUNetPlans.json").write_text("{}")
    res.mkdir()
    _isolate_nnunet_env(monkeypatch, raw.parent, pre.parent, res)
    called = {}

    def boom(*_args, **_kwargs):
        raise AssertionError("sampler should not run")

    def capture(nnunet_raw, dataset_id, *, configuration, np, plans):
        called["raw"] = nnunet_raw
        called["id"] = dataset_id
        called["configuration"] = configuration
        called["np"] = np
        called["plans"] = plans

    monkeypatch.setattr("seqseg.pipeline.train._require_sampler", boom)
    monkeypatch.setattr(
        "seqseg.pipeline.train._run_nnunet_preprocess_only", capture
    )
    result = prepare_training_dataset(
        dataset_number=999,
        preprocess_only=True,
        configuration="2d",
        np=8,
        nnunet_raw=str(raw.parent),
    )
    assert called["id"] == 999
    assert called["configuration"] == "2d"
    assert called["np"] == 8
    assert called["plans"] == "nnUNetPlans"
    assert called["raw"] == str(raw.parent)
    assert result.preprocessed is True
    assert result.dataset_names == ["Dataset999_MYDATACT"]


def test_preprocess_only_missing_plans_tells_user_to_prepare(tmp_path, monkeypatch):
    raw = tmp_path / "nnUNet_raw" / "Dataset999_MYDATACT"
    pre = tmp_path / "nnUNet_preprocessed" / "Dataset999_MYDATACT"
    res = tmp_path / "nnUNet_results"
    raw.mkdir(parents=True)
    pre.mkdir(parents=True)
    res.mkdir()
    _isolate_nnunet_env(monkeypatch, raw.parent, pre.parent, res)
    with pytest.raises(FileNotFoundError, match="seqseg train prepare"):
        prepare_training_dataset(
            dataset_number=999,
            preprocess_only=True,
            nnunet_raw=str(raw.parent),
        )


def test_cleanup_removes_raw_and_preprocessed_keeps_results(tmp_path, monkeypatch):
    raw = tmp_path / "nnUNet_raw"
    pre = tmp_path / "nnUNet_preprocessed"
    res = tmp_path / "nnUNet_results"
    kept_raw = raw / "Dataset010_OTHER"
    kept_pre = pre / "Dataset010_OTHER"
    drop_raw = raw / "Dataset999_MYDATACT"
    drop_pre = pre / "Dataset999_MYDATACT"
    kept_res = res / "Dataset999_MYDATACT"
    for path in (kept_raw, kept_pre, drop_raw, drop_pre, kept_res):
        path.mkdir(parents=True)
        (path / "file.txt").write_text("x")
    _isolate_nnunet_env(monkeypatch, raw, pre, res)

    with patch("seqseg.pipeline.train.subprocess.run"):
        run_nnunet_training(999, cleanup=True)

    assert not drop_raw.exists()
    assert not drop_pre.exists()
    assert (kept_raw / "file.txt").is_file()
    assert (kept_pre / "file.txt").is_file()
    assert (kept_res / "file.txt").is_file()


def test_cleanup_skipped_when_training_fails(tmp_path, monkeypatch):
    raw = tmp_path / "nnUNet_raw"
    pre = tmp_path / "nnUNet_preprocessed"
    res = tmp_path / "nnUNet_results"
    dataset = pre / "Dataset999_MYDATACT"
    dataset.mkdir(parents=True)
    (dataset / "file.txt").write_text("x")
    _isolate_nnunet_env(monkeypatch, raw, pre, res)

    with patch(
        "seqseg.pipeline.train.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "nnUNetv2_train"),
    ):
        with pytest.raises(subprocess.CalledProcessError):
            run_nnunet_training(999, cleanup=True)

    assert (dataset / "file.txt").is_file()
