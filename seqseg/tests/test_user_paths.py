"""Tests for persisted SeqSeg user paths."""

import os
from pathlib import Path

from seqseg.user_paths import (
    apply_nnunet_env,
    clear_paths,
    default_nnunet_layout,
    format_paths_report,
    load_paths,
    merge_path_updates,
    resolve_path,
    save_paths,
    shell_exports,
)


def test_save_load_roundtrip(tmp_path, monkeypatch):
    pf = tmp_path / "paths.yaml"
    monkeypatch.setenv("SEQSEG_PATHS_FILE", str(pf))
    save_paths(
        {
            "nnunet_raw": str(tmp_path / "raw"),
            "data_dir": str(tmp_path / "data"),
            "outdir": str(tmp_path / "out"),
        }
    )
    loaded = load_paths()
    assert loaded["nnunet_raw"].endswith("raw")
    assert loaded["data_dir"].endswith("data")
    assert resolve_path("outdir") == str((tmp_path / "out").resolve())


def test_env_wins_over_file(tmp_path, monkeypatch):
    pf = tmp_path / "paths.yaml"
    monkeypatch.setenv("SEQSEG_PATHS_FILE", str(pf))
    save_paths({"nnunet_raw": str(tmp_path / "from_file")})
    monkeypatch.setenv("nnUNet_raw", str(tmp_path / "from_env"))
    assert resolve_path("nnunet_raw").endswith("from_env")


def test_apply_nnunet_env_fills_missing(tmp_path, monkeypatch):
    pf = tmp_path / "paths.yaml"
    monkeypatch.setenv("SEQSEG_PATHS_FILE", str(pf))
    for key in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        monkeypatch.delenv(key, raising=False)
    layout = default_nnunet_layout(str(tmp_path / "nnunet"))
    save_paths(layout)
    env: dict = {}
    apply_nnunet_env(env, create_dirs=True)
    assert Path(env["nnUNet_raw"]).is_dir()
    assert "nnUNet_preprocessed" in env
    assert "nnUNet_results" in env


def test_merge_nnunet_root():
    out = merge_path_updates({}, nnunet_root="~/nnunet_data")
    assert out["nnunet_raw"].endswith("nnUNet_raw")
    assert out["nnunet_preprocessed"].endswith("nnUNet_preprocessed")
    assert out["nnunet_results"].endswith("nnUNet_results")


def test_clear_and_export(tmp_path, monkeypatch):
    pf = tmp_path / "paths.yaml"
    monkeypatch.setenv("SEQSEG_PATHS_FILE", str(pf))
    for key in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        monkeypatch.delenv(key, raising=False)
    save_paths(default_nnunet_layout(str(tmp_path / "nn")))
    text = shell_exports()
    assert "export nnUNet_raw=" in text
    assert clear_paths() == pf.resolve()
    assert load_paths() == {}


def test_format_report_mentions_file(tmp_path, monkeypatch):
    pf = tmp_path / "paths.yaml"
    monkeypatch.setenv("SEQSEG_PATHS_FILE", str(pf))
    report = format_paths_report()
    assert str(pf.resolve()) in report
    assert "missing" in report
