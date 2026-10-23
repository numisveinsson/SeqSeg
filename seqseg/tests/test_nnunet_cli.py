"""Tests for seqseg.pipeline.nnunet_cli (single-pass nnU-Net file lookup)."""

import json
import sys

import pytest

pytest.importorskip("nnunetv2")

import nnunetv2.experiment_planning.plan_and_preprocess_entrypoints  # noqa: E402,F401
import nnunetv2.preprocessing.preprocessors.default_preprocessor  # noqa: E402,F401
from nnunetv2.utilities.utils import (  # noqa: E402
    get_filenames_of_train_images_and_targets as nnunet_folder_scan,
)

from seqseg.pipeline import nnunet_cli
from seqseg.pipeline.nnunet_cli import install_fast_lookup, training_files_from_folder


def _raw_dataset(root, cases, *, channels=1, ending=".nii.gz"):
    dataset = root / "Dataset999_MYDATACT"
    (dataset / "imagesTr").mkdir(parents=True)
    (dataset / "labelsTr").mkdir()
    for case in cases:
        for ch in range(channels):
            (dataset / "imagesTr" / f"{case}_{ch:04d}{ending}").write_text("x")
        (dataset / "labelsTr" / f"{case}{ending}").write_text("x")
    (dataset / "imagesTr" / ".DS_Store").write_text("x")
    (dataset / "imagesTr" / "notes.txt").write_text("x")
    meta = {
        "channel_names": {str(ch): "CT" for ch in range(channels)},
        "labels": {"background": 0, "vessel": 1},
        "numTraining": len(cases),
        "file_ending": ending,
    }
    (dataset / "dataset.json").write_text(json.dumps(meta))
    return dataset, meta


@pytest.mark.parametrize(
    "cases, channels, ending",
    [
        (["mydatact_002", "mydatact_001", "mydatact_010"], 1, ".nii.gz"),
        (["mydatact_001", "mydatact_1000"], 2, ".nrrd"),
        (["a_001", "a_001_x"], 1, ".mha"),
    ],
)
def test_lookup_matches_nnunet_folder_scan(tmp_path, cases, channels, ending):
    dataset, meta = _raw_dataset(tmp_path, cases, channels=channels, ending=ending)
    expected = nnunet_folder_scan(str(dataset), dict(meta))

    got = training_files_from_folder(str(dataset), meta)

    assert list(got) == [str(k) for k in expected]
    assert got == {str(k): v for k, v in expected.items()}


def test_install_patches_every_nnunet_caller(tmp_path, monkeypatch):
    import nnunetv2.utilities.utils as utils

    for name, module in list(sys.modules.items()):
        if name.startswith("nnunetv2") and hasattr(
            module, "get_filenames_of_train_images_and_targets"
        ):
            monkeypatch.setattr(
                module,
                "get_filenames_of_train_images_and_targets",
                module.get_filenames_of_train_images_and_targets,
            )
    original = nnunet_folder_scan

    install_fast_lookup()

    from nnunetv2.experiment_planning import plan_and_preprocess_api
    from nnunetv2.experiment_planning.dataset_fingerprint import fingerprint_extractor
    from nnunetv2.experiment_planning.experiment_planners import (
        default_experiment_planner,
    )
    from nnunetv2.preprocessing.preprocessors import default_preprocessor

    for module in (
        utils,
        plan_and_preprocess_api,
        fingerprint_extractor,
        default_experiment_planner,
        default_preprocessor,
    ):
        assert module.get_filenames_of_train_images_and_targets is not original

    dataset, meta = _raw_dataset(tmp_path, ["mydatact_001"])
    assert set(utils.get_filenames_of_train_images_and_targets(str(dataset))) == {
        "mydatact_001"
    }


def test_explicit_dataset_list_uses_nnunet_lookup(tmp_path):
    dataset, meta = _raw_dataset(tmp_path, ["mydatact_001"])
    meta["dataset"] = {
        "listed": {
            "images": ["imagesTr/mydatact_001_0000.nii.gz"],
            "label": "labelsTr/mydatact_001.nii.gz",
        }
    }
    calls = []

    def original(folder, dataset_json):
        calls.append(folder)
        return {"listed": dataset_json["dataset"]["listed"]}

    fast = nnunet_cli._fast_lookup(original)
    assert list(fast(str(dataset), meta)) == ["listed"]
    assert calls == [str(dataset)]


def test_main_rejects_unknown_command():
    with pytest.raises(SystemExit, match="usage"):
        nnunet_cli.main(["train"])


def test_main_runs_entry_with_nnunet_argv(monkeypatch):
    from nnunetv2.experiment_planning import plan_and_preprocess_entrypoints

    seen = {}
    monkeypatch.setattr(
        nnunet_cli, "install_fast_lookup", lambda: seen.setdefault("installed", True)
    )
    monkeypatch.setattr(
        plan_and_preprocess_entrypoints,
        "preprocess_entry",
        lambda: seen.setdefault("argv", list(sys.argv)),
    )
    monkeypatch.setattr(sys, "argv", ["x"])

    nnunet_cli.main(["preprocess", "-d", "999", "-c", "2d"])

    assert seen == {
        "installed": True,
        "argv": ["nnUNetv2_preprocess", "-d", "999", "-c", "2d"],
    }
