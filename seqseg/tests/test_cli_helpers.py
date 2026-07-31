"""CLI parser wiring and small handlers (no full tracing runs)."""

import argparse
import io
import sys
from unittest.mock import patch

import pytest

from seqseg.cli import (
    _build_parser,
    _cmd_config_fingerprint,
    _cmd_init_dataset,
    _cmd_train_nnunet,
    _cmd_train_prepare,
    _inject_legacy_run_batch,
    dispatch,
)
from seqseg.pipeline.train import (
    TrainDependencyError,
    dataset_id_from_name,
    expected_nnunet_dataset_name,
)


def test_inject_legacy_run_batch():
    assert _inject_legacy_run_batch(["run", "batch", "-data_dir", "x"]) == [
        "run",
        "batch",
        "-data_dir",
        "x",
    ]
    assert _inject_legacy_run_batch(["-data_dir", "x"]) == [
        "run",
        "batch",
        "-data_dir",
        "x",
    ]


def test_parser_version_exits_zero(capsys):
    parser = _build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "seqseg" in out.lower() or len(out) > 0


def test_parser_run_single_required_flags():
    parser = _build_parser()
    ns = parser.parse_args(
        [
            "run",
            "single",
            "--image",
            "/tmp/img.nii.gz",
            "--outdir",
            "/tmp/out",
            "--model-folder",
            "/tmp/model",
            "--seed",
            "0",
            "0",
            "0",
            "1",
        ]
    )
    assert ns.command == "run"
    assert ns.run_cmd == "single"
    assert ns.image == "/tmp/img.nii.gz"
    assert ns.model_folder == "/tmp/model"


def test_parser_init_dataset():
    parser = _build_parser()
    ns = parser.parse_args(["init", "dataset", "--path", "/tmp/ds"])
    assert ns.command == "init"
    assert ns.init_cmd == "dataset"
    assert ns.path == "/tmp/ds"


def test_parser_config_fingerprint_defaults():
    parser = _build_parser()
    ns = parser.parse_args(["config", "fingerprint"])
    assert ns.command == "config"
    assert ns.cfg_cmd == "fingerprint"
    assert ns.name == "global"
    assert ns.baseline == "global_default"


def test_cmd_init_dataset_writes_template(tmp_path):
    root = str(tmp_path / "dataset")
    ns = argparse.Namespace(path=root, force=False)
    buf = io.StringIO()
    with patch.object(sys, "stdout", buf):
        _cmd_init_dataset(ns)
    assert (tmp_path / "dataset" / "images").is_dir()
    assert (tmp_path / "dataset" / "seeds.json").is_file()
    out = buf.getvalue()
    assert "seeds.json" in out


def test_cmd_init_dataset_skips_without_force(tmp_path):
    root = tmp_path / "ds2"
    root.mkdir()
    seeds = root / "seeds.json"
    seeds.write_text('{"existing": true}', encoding="utf-8")
    ns = argparse.Namespace(path=str(root), force=False)
    buf = io.StringIO()
    with patch.object(sys, "stdout", buf):
        _cmd_init_dataset(ns)
    assert seeds.read_text() == '{"existing": true}'
    assert "Skipped" in buf.getvalue()


def test_cmd_config_fingerprint_same_config(capsys):
    ns = argparse.Namespace(name="global", baseline="global")
    _cmd_config_fingerprint(ns)
    out = capsys.readouterr().out
    assert "No differences" in out


def test_dispatch_prints_citation_banner(tmp_path):
    root = str(tmp_path / "dataset")
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    with patch.object(sys, "stdout", buf_out), patch.object(sys, "stderr", buf_err):
        dispatch(["init", "dataset", "--path", root])
    err = buf_err.getvalue()
    assert "Please cite the following paper when using SeqSeg" in err
    assert "Please cite the following paper when using nnU-Net" in err
    assert "Sveinsson Cepero" in err
    assert "Isensee" in err


def test_dispatch_no_citation_on_help(capsys):
    with pytest.raises(SystemExit) as exc:
        dispatch(["--help"])
    assert exc.value.code == 0
    err = capsys.readouterr().err
    assert "Please cite" not in err


def test_parser_train_prepare():
    parser = _build_parser()
    ns = parser.parse_args(
        [
            "train",
            "prepare",
            "--data-dir",
            "/tmp/data",
            "--name",
            "MYDATA",
            "--dataset-number",
            "999",
            "--modality",
            "CT",
        ]
    )
    assert ns.command == "train"
    assert ns.train_cmd == "prepare"
    assert ns.data_dir == "/tmp/data"
    assert ns.name == "MYDATA"
    assert ns.dataset_number == 999


def test_parser_paths_init():
    parser = _build_parser()
    ns = parser.parse_args(
        [
            "paths",
            "init",
            "--data-dir",
            "/tmp/data",
            "--outdir",
            "/tmp/out",
        ]
    )
    assert ns.command == "paths"
    assert ns.paths_cmd == "init"
    assert ns.data_dir == "/tmp/data"
    assert ns.outdir == "/tmp/out"


def test_parser_train_nnunet():
    parser = _build_parser()
    ns = parser.parse_args(
        ["train", "nnunet", "--dataset-id", "999", "--fold", "all", "--plan-only"]
    )
    assert ns.command == "train"
    assert ns.train_cmd == "nnunet"
    assert ns.dataset_id == 999
    assert ns.plan_only is True


def test_dataset_id_from_name():
    assert dataset_id_from_name("Dataset010_SEQCOROASOCACT") == 10
    assert dataset_id_from_name("Dataset0999_MYDATACT") == 999
    assert expected_nnunet_dataset_name("MYDATA", 999, "ct") == "Dataset0999_MYDATACT"


def test_cmd_train_prepare_missing_dep(tmp_path, capsys):
    ns = argparse.Namespace(
        data_dir=str(tmp_path),
        outdir=str(tmp_path / "out"),
        name="MYDATA",
        dataset_number=999,
        modality="CT",
        config_name="global",
        nnunet_raw=None,
        perc_dataset=1.0,
        num_cores=1,
        start_from=0,
        end_at=-1,
        testing=False,
        validation_prop=None,
        max_samples=None,
        truth_from_surface=False,
        truth_target_spacing=None,
        truth_regenerate=False,
        skip_sample=False,
        skip_convert=False,
        also_test=False,
        yes=False,
        verbose=False,
    )
    with patch(
        "seqseg.cli.prepare_training_dataset",
        side_effect=TrainDependencyError("missing"),
    ):
        with pytest.raises(SystemExit) as exc:
            _cmd_train_prepare(ns)
    assert exc.value.code == 1
    assert "missing" in capsys.readouterr().err


def test_cmd_train_nnunet_resolves_name(monkeypatch):
    called = {}

    def fake_run(dataset_id, **kwargs):
        called["dataset_id"] = dataset_id
        called["kwargs"] = kwargs

    monkeypatch.setattr("seqseg.cli.run_nnunet_training", fake_run)
    ns = argparse.Namespace(
        dataset_id=None,
        dataset_name="Dataset0999_MYDATACT",
        configuration="3d_fullres",
        fold="0",
        skip_plan=False,
        plan_only=True,
        np=None,
        trainer="nnUNetTrainer",
        plans="nnUNetPlans",
    )
    _cmd_train_nnunet(ns)
    assert called["dataset_id"] == 999
    assert called["kwargs"]["plan_only"] is True
