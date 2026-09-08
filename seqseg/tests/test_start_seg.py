"""Starting-segmentation load / merge helpers."""

import numpy as np
import pytest
import SimpleITK as sitk

from seqseg.modules.sitk_functions import (
    as_probability_image,
    geometry_matches,
    load_start_segmentation,
    merge_binary_with_start,
    merge_probability_with_start,
    resample_to_reference,
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
