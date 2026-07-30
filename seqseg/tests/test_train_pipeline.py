"""Unit tests for seqseg.pipeline.train helpers (no GPU / sampler required)."""

from unittest.mock import MagicMock, patch

import pytest

from seqseg.pipeline.train import (
    TrainDependencyError,
    dataset_id_from_name,
    expected_nnunet_dataset_name,
    prepare_training_dataset,
    run_nnunet_training,
)


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

    with patch(
        "seqseg.pipeline.train._require_sampler",
        return_value=(extract, write),
    ):
        result = prepare_training_dataset(
            str(tmp_path / "data"),
            str(tmp_path / "extracted"),
            name="MYDATA",
            dataset_number=999,
            modality="CT",
            yes=True,
        )

    extract.assert_called_once()
    write.assert_called_once()
    assert result.dataset_names == ["Dataset0999_MYDATACT"]
    assert result.modalities == ["CT"]


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


def test_run_nnunet_training_requires_env(monkeypatch):
    for key in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="nnUNet_raw"):
        run_nnunet_training(999, plan_only=True)


def test_run_nnunet_training_plan_only(monkeypatch):
    monkeypatch.setenv("nnUNet_raw", "/raw")
    monkeypatch.setenv("nnUNet_preprocessed", "/pre")
    monkeypatch.setenv("nnUNet_results", "/res")

    with patch("seqseg.pipeline.train.subprocess.run") as run:
        run_nnunet_training(999, plan_only=True)
    assert run.call_count == 1
    cmd = run.call_args[0][0]
    assert "nnUNetv2_plan_and_preprocess" in cmd[0]
    assert cmd[cmd.index("-d") + 1] == "999"
