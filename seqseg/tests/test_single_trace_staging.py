"""Tests for :func:`seqseg.pipeline.single_trace.prepare_single_image_dataset`."""

import json
from pathlib import Path

import pytest

from seqseg.pipeline.single_trace import prepare_single_image_dataset, _stem_and_suffix


def test_stem_and_suffix_nii_gz():
    stem, ext = _stem_and_suffix("/data/subject_01.nii.gz")
    assert stem == "subject_01"
    assert ext == ".nii.gz"


def test_stem_and_suffix_mha():
    stem, ext = _stem_and_suffix("C:/vol.mha")
    assert stem == "vol"
    assert ext == ".mha"


def test_prepare_single_with_xyzr_seeds(tmp_path):
    src = tmp_path / "my_case.nii.gz"
    src.write_bytes(b"")
    out = tmp_path / "run"
    data_dir, ext, case_name = prepare_single_image_dataset(
        str(src),
        str(out),
        seeds_xyzr=[[1.0, 2.0, 3.0, 0.5]],
        tangent=(0.0, 0.0, 1.0),
        seed_step=2.0,
    )
    assert data_dir.endswith("/") or data_dir.endswith("\\")
    assert ext == ".nii.gz"
    assert case_name == "my_case"

    staging = Path(data_dir.rstrip("/\\"))
    assert (staging / "images" / "my_case.nii.gz").is_file()
    seeds_path = staging / "seeds.json"
    payload = json.loads(seeds_path.read_text(encoding="utf-8"))
    assert len(payload) == 1
    assert payload[0]["name"] == "my_case"
    assert len(payload[0]["seeds"]) == 1
    old, new, r = payload[0]["seeds"][0]
    assert new == [1.0, 2.0, 3.0]
    assert r == 0.5
    assert old[0] == 1.0 and old[1] == 2.0 and old[2] == 1.0


def test_prepare_single_with_seeds_json_case_rename(tmp_path):
    src = tmp_path / "raw_name.nii.gz"
    src.write_bytes(b"")
    seeds_in = tmp_path / "seeds.json"
    seeds_in.write_text(
        json.dumps(
            [{"name": "renamed_case", "seeds": [], "cardiac_mesh": False}],
        ),
        encoding="utf-8",
    )
    data_dir, ext, case_name = prepare_single_image_dataset(
        str(src),
        str(tmp_path / "out"),
        seeds_json_path=str(seeds_in),
    )
    assert case_name == "renamed_case"
    assert ext == ".nii.gz"
    staging = Path(data_dir.rstrip("/\\"))
    assert (staging / "images" / "renamed_case.nii.gz").is_file()


def test_prepare_single_requires_seeds_without_json(tmp_path):
    src = tmp_path / "x.mha"
    src.write_bytes(b"")
    with pytest.raises(ValueError, match="seeds_xyzr or seeds_json_path"):
        prepare_single_image_dataset(str(src), str(tmp_path / "o"))


def test_prepare_single_missing_image(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare_single_image_dataset(
            str(tmp_path / "nope.nii.gz"),
            str(tmp_path / "o"),
            seeds_xyzr=[[0, 0, 0, 1.0]],
        )


def test_prepare_single_unknown_extension(tmp_path):
    src = tmp_path / "bad"
    src.write_bytes(b"")
    with pytest.raises(ValueError, match="Could not infer file extension"):
        prepare_single_image_dataset(
            str(src),
            str(tmp_path / "o"),
            seeds_xyzr=[[0, 0, 0, 1.0]],
        )
