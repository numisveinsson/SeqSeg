"""Starting-segmentation load / merge helpers."""

import numpy as np
import pytest
import SimpleITK as sitk

from seqseg.modules.sitk_functions import (
    as_probability_image,
    geometry_matches,
    load_global_segmentation,
    load_start_segmentation,
    merge_binary_with_start,
    merge_probability_with_start,
    resample_to_reference,
    resolve_global_seg_files,
)


def _box(size=(8, 8, 8), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), value=0):
    img = sitk.Image(*size, sitk.sitkUInt8)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    if value:
        arr = sitk.GetArrayFromImage(img)
        arr[:] = value
        out = sitk.GetImageFromArray(arr)
        out.CopyInformation(img)
        return out
    return img


def test_as_probability_image_scales_255():
    img = _box(value=255)
    prob = as_probability_image(img)
    arr = sitk.GetArrayFromImage(prob)
    assert arr.dtype == np.float32
    assert float(arr.max()) == pytest.approx(1.0)


def test_resample_to_reference_when_grids_differ():
    src = _box(size=(4, 4, 4), spacing=(2.0, 2.0, 2.0), value=1)
    ref = _box(size=(8, 8, 8), spacing=(1.0, 1.0, 1.0))
    aligned = resample_to_reference(src, ref, is_label=True)
    assert geometry_matches(aligned, ref)
    assert tuple(aligned.GetSize()) == (8, 8, 8)


def test_load_start_segmentation_aligns_and_normalizes(tmp_path):
    ref = _box()
    arr = np.zeros((8, 8, 8), dtype=np.uint8)
    arr[1:4, 1:4, 1:4] = 1
    start = sitk.GetImageFromArray(arr)
    start.CopyInformation(ref)
    path = str(tmp_path / "start.mha")
    sitk.WriteImage(start, path)

    loaded = load_start_segmentation(path, ref)
    assert geometry_matches(loaded, ref)
    loaded_arr = sitk.GetArrayFromImage(loaded)
    assert float(loaded_arr.max()) == pytest.approx(1.0)
    assert float(loaded_arr[2, 2, 2]) == pytest.approx(1.0)
    assert float(loaded_arr[0, 0, 0]) == pytest.approx(0.0)


def test_load_start_segmentation_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_start_segmentation(str(tmp_path / "nope.mha"), _box())


def test_merge_binary_with_start_unions_masks():
    seqseg = _box()
    a = sitk.GetArrayFromImage(seqseg)
    a[0:3, 0:3, 0:3] = 1
    seqseg = sitk.GetImageFromArray(a)
    seqseg.CopyInformation(_box())

    start = _box()
    b = sitk.GetArrayFromImage(start)
    b[5:8, 5:8, 5:8] = 1
    start = sitk.GetImageFromArray(b)
    start.CopyInformation(_box())

    merged = merge_binary_with_start(seqseg, as_probability_image(start))
    m = sitk.GetArrayFromImage(merged)
    assert int(m[1, 1, 1]) == 1
    assert int(m[6, 6, 6]) == 1
    assert int(m[4, 4, 4]) == 0


def test_merge_probability_with_start_takes_maximum():
    seqseg = sitk.Image(4, 4, 4, sitk.sitkFloat32)
    start = sitk.Image(4, 4, 4, sitk.sitkFloat32)
    sa = sitk.GetArrayFromImage(seqseg)
    st = sitk.GetArrayFromImage(start)
    sa[0, 0, 0] = 0.2
    st[0, 0, 0] = 0.9
    seqseg = sitk.GetImageFromArray(sa)
    start = sitk.GetImageFromArray(st)
    seqseg.SetSpacing((1.0, 1.0, 1.0))
    start.CopyInformation(seqseg)
    merged = merge_probability_with_start(seqseg, start)
    assert float(sitk.GetArrayFromImage(merged)[0, 0, 0]) == pytest.approx(0.9)


def _write(img, path):
    sitk.WriteImage(img, str(path))


def test_resolve_global_seg_files_prefers_segmentation_and_probability(tmp_path):
    seg = tmp_path / "case_segmentation.mha"
    prob = tmp_path / "case_probability.mha"
    named = tmp_path / "case.mha"
    _write(_box(value=1), seg)
    _write(_box(value=1), prob)
    _write(_box(value=1), named)

    binary_path, prob_path = resolve_global_seg_files(tmp_path, "case", ".mha")
    assert binary_path == str(seg)
    assert prob_path == str(prob)


def test_resolve_global_seg_files_image_name(tmp_path):
    named = tmp_path / "case.nii.gz"
    _write(_box(value=1), named)
    binary_path, prob_path = resolve_global_seg_files(tmp_path, "case", ".nii.gz")
    assert binary_path == str(named)
    assert prob_path is None


def test_resolve_global_seg_files_probability_only(tmp_path):
    prob = tmp_path / "case_probability.mha"
    _write(_box(value=1), prob)
    binary_path, prob_path = resolve_global_seg_files(tmp_path, "case", ".mha")
    assert binary_path is None
    assert prob_path == str(prob)


def test_resolve_global_seg_files_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="case_segmentation.mha") as exc:
        resolve_global_seg_files(tmp_path, "case", ".nii.gz")
    message = str(exc.value)
    assert "case.nii.gz" in message
    assert "case_probability.mha" in message


def test_load_global_segmentation_label_and_probability_stay_separate(tmp_path):
    ref = _box()
    label = _box()
    arr = sitk.GetArrayFromImage(label)
    arr[1, 1, 1] = 2
    label = sitk.GetImageFromArray(arr)
    label.CopyInformation(ref)
    prob = sitk.Image(8, 8, 8, sitk.sitkFloat32)
    prob.CopyInformation(ref)
    pa = sitk.GetArrayFromImage(prob)
    pa[1, 1, 1] = 0.25
    pa[2, 2, 2] = 0.8
    prob = sitk.GetImageFromArray(pa)
    prob.CopyInformation(ref)
    seg_path = tmp_path / "seg.mha"
    prob_path = tmp_path / "prob.mha"
    _write(label, seg_path)
    _write(prob, prob_path)

    binary, loaded_prob = load_global_segmentation(str(seg_path), str(prob_path), ref)
    assert geometry_matches(binary, ref)
    assert geometry_matches(loaded_prob, ref)
    b = sitk.GetArrayFromImage(binary)
    p = sitk.GetArrayFromImage(loaded_prob)
    assert int(b[1, 1, 1]) == 1
    assert int(b[2, 2, 2]) == 0
    assert float(p[1, 1, 1]) == pytest.approx(0.25)
    assert float(p[2, 2, 2]) == pytest.approx(0.8)


def test_load_global_segmentation_image_name_is_binary_prior(tmp_path):
    ref = _box()
    label = _box(value=255)
    path = tmp_path / "case.mha"
    _write(label, path)
    binary, prob = load_global_segmentation(str(path), None, ref)
    assert int(sitk.GetArrayFromImage(binary).max()) == 1
    assert float(sitk.GetArrayFromImage(prob).max()) == pytest.approx(1.0)
    assert sitk.GetArrayFromImage(prob).dtype == np.float32


def test_load_global_segmentation_probability_only_thresholds(tmp_path):
    ref = _box(size=(4, 4, 4))
    prob = sitk.Image(4, 4, 4, sitk.sitkFloat32)
    prob.CopyInformation(ref)
    pa = sitk.GetArrayFromImage(prob)
    pa[0, 0, 0] = 0.4
    pa[1, 1, 1] = 0.5
    pa[2, 2, 2] = 1.4
    prob = sitk.GetImageFromArray(pa)
    prob.CopyInformation(ref)
    path = tmp_path / "case_probability.mha"
    _write(prob, path)

    binary, loaded = load_global_segmentation(None, str(path), ref)
    b = sitk.GetArrayFromImage(binary)
    p = sitk.GetArrayFromImage(loaded)
    assert int(b[0, 0, 0]) == 0
    assert int(b[1, 1, 1]) == 1
    assert float(p[0, 0, 0]) == pytest.approx(0.4)
    assert float(p[1, 1, 1]) == pytest.approx(0.5)
    assert float(p[2, 2, 2]) == pytest.approx(1.0)


def test_load_global_segmentation_resamples_onto_image(tmp_path):
    ref = _box(size=(8, 8, 8), spacing=(1.0, 1.0, 1.0))
    coarse = _box(size=(4, 4, 4), spacing=(2.0, 2.0, 2.0))
    arr = sitk.GetArrayFromImage(coarse)
    arr[0, 0, 0] = 3
    coarse = sitk.GetImageFromArray(arr)
    coarse.CopyInformation(_box(size=(4, 4, 4), spacing=(2.0, 2.0, 2.0)))
    path = tmp_path / "case_segmentation.mha"
    _write(coarse, path)

    binary, prob = load_global_segmentation(str(path), None, ref)
    assert geometry_matches(binary, ref)
    assert geometry_matches(prob, ref)
    b = sitk.GetArrayFromImage(binary)
    p = sitk.GetArrayFromImage(prob)
    assert set(np.unique(b)).issubset({0, 1})
    assert int(b.sum()) > 0
    assert float(p[b > 0].min()) == pytest.approx(1.0)
    assert float(p[b == 0].max()) == pytest.approx(0.0)
