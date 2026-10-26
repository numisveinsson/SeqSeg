"""Whole-volume prediction listing and writing, with the nnU-Net model mocked."""

import os

import pytest
import SimpleITK as sitk

from seqseg.pipeline.predict import (
    list_input_volumes,
    run_global_prediction,
    volume_stem,
)


def test_volume_stem_keeps_nii_gz_as_one_suffix():
    assert volume_stem("/data/case.nii.gz") == "case"
    assert volume_stem("/data/case.nii") == "case"
    assert volume_stem("/data/scan.mha") == "scan"


def test_list_input_volumes_filters_extension(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.nii.gz").write_bytes(b"x")
    (images / "b.mha").write_bytes(b"x")
    (images / "notes.txt").write_bytes(b"x")
    (images / ".hidden.nii.gz").write_bytes(b"x")

    found = list_input_volumes(image_dir=str(images))
    assert [os.path.basename(p) for p in found] == ["a.nii.gz", "b.mha"]

    only_nii = list_input_volumes(image_dir=str(images), img_ext="nii.gz")
    assert [os.path.basename(p) for p in only_nii] == ["a.nii.gz"]

    with pytest.raises(FileNotFoundError, match="No volumes"):
        list_input_volumes(image_dir=str(images), img_ext=".nrrd")


def test_run_global_prediction_loads_model_once(tmp_path, monkeypatch):
    calls = {"n": 0, "images": []}

    def fake_init(model_folder, fold, force_cpu=False):
        calls["n"] += 1
        calls["model"] = model_folder
        calls["fold"] = fold
        calls["cpu"] = force_cpu
        return object()

    def fake_predict(predictor, dir_image, scale=1):
        calls["images"].append(os.path.basename(dir_image))
        calls["scale"] = scale
        label = sitk.Image(2, 2, 2, sitk.sitkUInt8)
        prob = sitk.Image(2, 2, 2, sitk.sitkFloat32)
        return label, prob

    monkeypatch.setattr("seqseg.pipeline.predict.initialize_predictor", fake_init)
    monkeypatch.setattr("seqseg.pipeline.predict.predict_volume", fake_predict)

    model = tmp_path / "trainer"
    model.mkdir()
    image = tmp_path / "case.nii.gz"
    image.write_bytes(b"x")
    out = tmp_path / "out"

    written = run_global_prediction(
        model_folder=str(model),
        outdir=str(out),
        image=str(image),
        fold="0",
        scale=0.1,
        force_cpu=True,
    )
    assert calls["n"] == 1
    assert calls["model"] == str(model)
    assert calls["fold"] == "0"
    assert calls["cpu"] is True
    assert calls["scale"] == 0.1
    assert written == [str(out / "case_segmentation.mha")]
    assert (out / "case_probability.mha").is_file()
    assert sitk.ReadImage(written[0]).GetSize() == (2, 2, 2)

    folder = tmp_path / "images"
    folder.mkdir()
    (folder / "a.nii.gz").write_bytes(b"x")
    (folder / "b.mha").write_bytes(b"x")
    (folder / "skip.txt").write_bytes(b"x")
    out2 = tmp_path / "out2"
    calls["n"] = 0
    calls["images"] = []
    written = run_global_prediction(
        model_folder=str(model),
        outdir=str(out2),
        image_dir=str(folder),
    )
    assert calls["n"] == 1
    assert calls["images"] == ["a.nii.gz", "b.mha"]
    assert (out2 / "a_segmentation.mha").is_file()
    assert (out2 / "a_probability.mha").is_file()
    assert (out2 / "b_segmentation.mha").is_file()
    assert (out2 / "b_probability.mha").is_file()
    assert written == [
        str(out2 / "a_segmentation.mha"),
        str(out2 / "b_segmentation.mha"),
    ]
