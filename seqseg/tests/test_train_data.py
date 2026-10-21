"""Train-prepare smoke test on the bundled case in ``test_data_train``.

The case is one CT volume (``195``) with a truth, centerline, and surface.
Patch extraction is capped with ``--max-samples`` so nnU-Net plans and
preprocesses those subvolumes instead of the whole scan.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from seqseg.cli import dispatch

DATA_DIR = Path(__file__).resolve().parent / "test_data_train"
DATASET_ID = "901"
DATASET_NAME = f"Dataset{DATASET_ID}_TESTDATACT"


def _isolate_nnunet(monkeypatch, root: Path) -> tuple[Path, Path, Path]:
    raw = root / "nnUNet_raw"
    pre = root / "nnUNet_preprocessed"
    res = root / "nnUNet_results"
    monkeypatch.setenv("SEQSEG_PATHS_FILE", str(root / "no_paths.yaml"))
    monkeypatch.setenv("nnUNet_raw", str(raw))
    monkeypatch.setenv("nnUNet_preprocessed", str(pre))
    monkeypatch.setenv("nnUNet_results", str(res))
    return raw, pre, res


def _prepare(data_dir: Path, *, max_samples: str = "2") -> None:
    dispatch(
        [
            "train",
            "prepare",
            "--data-dir",
            str(data_dir),
            "--name",
            "TESTDATA",
            "--dataset-id",
            DATASET_ID,
            "--modality",
            "CT",
            "--img-ext",
            ".nii.gz",
            "--max-samples",
            max_samples,
            "--num-cores",
            "1",
            "--yes",
        ]
    )


@pytest.mark.skipif(not (DATA_DIR / "images").is_dir(), reason="test_data_train is missing")
def test_prepare_sample_case_preprocesses_extracted_patches(tmp_path, monkeypatch):
    image = DATA_DIR / "images" / "195.nii.gz"
    before = image.stat().st_mtime_ns
    raw, pre, _res = _isolate_nnunet(monkeypatch, tmp_path)

    _prepare(DATA_DIR)

    dataset = raw / DATASET_NAME
    assert (dataset / "dataset.json").is_file()
    images_tr = dataset / "imagesTr"
    labels_tr = dataset / "labelsTr"
    assert images_tr.is_dir() and labels_tr.is_dir()
    assert any(images_tr.iterdir())
    assert any(labels_tr.iterdir())

    pre_dir = pre / DATASET_NAME
    assert (pre_dir / "nnUNetPlans.json").is_file()
    assert (pre_dir / "nnUNetPlans_3d_fullres").is_dir()
    assert image.stat().st_mtime_ns == before
    staging = tmp_path / "_seqseg_extracted"
    assert not staging.exists()

    dispatch(
        [
            "train",
            "prepare",
            "--dataset-id",
            DATASET_ID,
            "--configuration",
            "2d",
            "--preprocess-only",
            "--np",
            "1",
        ]
    )
    assert (pre_dir / "nnUNetPlans_2d").is_dir()
    assert (pre_dir / "nnUNetPlans_3d_fullres").is_dir()
    assert image.stat().st_mtime_ns == before
    assert dataset.is_dir()
