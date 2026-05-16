"""Dataset path normalization (trailing slash invariant)."""

import json
import os

import pytest

from seqseg.modules.datasets import get_directories, get_testing_samples, normalize_dataset_root


@pytest.mark.parametrize("suffix", ["", os.sep])
def test_get_directories_ignores_trailing_sep(tmp_path, suffix):
    root = str(tmp_path) + suffix
    os.makedirs(tmp_path / "images")
    a = get_directories(root, "case1", ".mha")
    b = get_directories(str(tmp_path), "case1", ".mha")
    assert a == b
    assert os.path.basename(a[0]) == "case1.mha"


def test_normalize_dataset_root_no_double_separators_in_join(tmp_path):
    sub = tmp_path / "data"
    sub.mkdir()
    n = normalize_dataset_root(str(sub) + os.sep)
    joined = os.path.join(n, "images", "x.mha")
    assert ".." not in joined
    assert os.path.join(str(sub), "images", "x.mha") == joined


def test_get_testing_samples_with_or_without_trailing_slash(tmp_path):
    d = tmp_path / "ds"
    (d / "images").mkdir(parents=True)
    (d / "centerlines").mkdir(parents=True)
    (d / "images" / "a.mha").write_bytes(b"")
    (d / "centerlines" / "a.vtp").write_bytes(b"")

    samples1, dir1 = get_testing_samples("Dataset999_FAKE", str(d))
    samples2, dir2 = get_testing_samples("Dataset999_FAKE", str(d) + os.sep)
    assert dir1 == dir2
    assert len(samples1) == len(samples2)


def test_get_testing_samples_json_path(tmp_path):
    d = tmp_path / "ds2"
    d.mkdir()
    seeds = [{"name": "c1", "seeds": [[0, 0, 0]], "cardiac_mesh": False}]
    with open(d / "seeds.json", "w", encoding="utf-8") as f:
        json.dump(seeds, f)

    _, dir_a = get_testing_samples("X", str(d))
    _, dir_b = get_testing_samples("X", str(d) + os.sep)
    assert dir_a == dir_b
