"""Unit tests for seqseg.pipeline.train helpers (no GPU / sampler required)."""

from unittest.mock import MagicMock, patch

import pytest

import os

from pathlib import Path

from seqseg.pipeline.train import (
    TrainDependencyError,
    _detect_img_ext,
    _dir_for_sampler,
    _normalize_img_ext,
    _resolve_img_ext,
    dataset_id_from_name,
    expected_nnunet_dataset_name,
    prepare_training_dataset,
    run_nnunet_training,
)


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


def test_expected_nnunet_dataset_name_padding():
    assert expected_nnunet_dataset_name("AORTAS", 1, "ct") == "Dataset001_AORTASCT"
    assert expected_nnunet_dataset_name("AORTAS", 10, "mr") == "Dataset010_AORTASMR"
    assert expected_nnunet_dataset_name("MYDATA", 999, "CT") == "Dataset0999_MYDATACT"


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
    write = MagicMock(return_value=str(tmp_path / "Dataset0999_MYDATACT"))
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
            modality="CT",
            yes=True,
        )

    extract.assert_called_once()
    write.assert_called_once()
    outdir_arg = extract.call_args.kwargs["outdir"]
    assert outdir_arg.endswith(os.sep)
    assert os.path.basename(os.path.normpath(outdir_arg)) == "extracted"
    assert extract.call_args.kwargs["config"]["IMG_EXT"] == ".nrrd"
    assert result.dataset_names == ["Dataset0999_MYDATACT"]
    assert result.modalities == ["CT"]


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
    write = MagicMock(return_value=str(tmp_path / "Dataset0999_MYDATACT"))
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
            yes=True,
        )

    assert extract.call_args.kwargs["config"]["IMG_EXT"] == ".nii.gz"


def test_prepare_explicit_img_ext_overrides_detection(tmp_path):
    extract = MagicMock()
    write = MagicMock(return_value=str(tmp_path / "Dataset0999_MYDATACT"))
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
            img_ext="mha",
            yes=True,
        )

    assert extract.call_args.kwargs["config"]["IMG_EXT"] == ".mha"


def test_prepare_multi_modality_increments_ids(tmp_path):
    extract = MagicMock()
    write = MagicMock(
        side_effect=[
            str(tmp_path / "Dataset0999_MYDATACT"),
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
            modality="CT,MR",
            skip_sample=True,
        )

    assert write.call_count == 2
    assert result.dataset_names == [
        "Dataset0999_MYDATACT",
        "Dataset01000_MYDATAMR",
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
                yes=True,
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
    write = MagicMock(return_value=str(tmp_path / "Dataset0999_MYDATACT"))

    with patch(
        "seqseg.pipeline.train._require_sampler",
        return_value=(extract, write),
    ):
        result = prepare_training_dataset(
            _make_training_data(tmp_path),
            str(extracted),
            name="MYDATA",
            dataset_number=999,
            yes=True,
        )

    write.assert_called_once()
    assert result.dataset_names == ["Dataset0999_MYDATACT"]


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
                yes=True,
            )
    write.assert_not_called()


def test_run_nnunet_training_requires_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SEQSEG_PATHS_FILE", str(tmp_path / "no_paths.yaml"))
    for key in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="nnU-Net paths"):
        run_nnunet_training(999, plan_only=True)


def test_run_nnunet_training_plan_only(monkeypatch):
    monkeypatch.setenv("nnUNet_raw", "/raw")
    monkeypatch.setenv("nnUNet_preprocessed", "/pre")
    monkeypatch.setenv("nnUNet_results", "/res")

    with patch("seqseg.pipeline.train.subprocess.run") as run:
        run_nnunet_training(999, plan_only=True, configuration="3d_fullres")
    assert run.call_count == 1
    cmd = run.call_args[0][0]
    assert "nnUNetv2_plan_and_preprocess" in cmd[0]
    assert cmd[cmd.index("-d") + 1] == "999"
    assert cmd[cmd.index("-c") + 1] == "3d_fullres"
