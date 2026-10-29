"""done.txt skipping for classic and plus trace batches."""

from unittest.mock import MagicMock

import pytest

from seqseg.pipeline.classic import run_classic_batch
from seqseg.pipeline.done import (
    append_done_case,
    case_name_from_sample,
    read_done_cases,
)
from seqseg.pipeline.plus import run_plus_batch


def test_case_name_from_seeds_or_centerline_sample():
    assert case_name_from_sample({"name": "3", "seeds": []}) == "3"
    assert case_name_from_sample({"name": "S-008"}) == "S-008"
    assert case_name_from_sample(["S-008", 0, -50, -60]) == "S-008"


def test_read_ignores_blanks_duplicates_and_missing_file(tmp_path):
    assert read_done_cases(str(tmp_path)) == set()
    (tmp_path / "done.txt").write_text("3\n\n3\n  S-008  \n", encoding="utf-8")
    assert read_done_cases(str(tmp_path)) == {"3", "S-008"}


def test_append_does_not_rewrite_existing_lines(tmp_path):
    path = tmp_path / "done.txt"
    path.write_bytes(b"3")
    append_done_case(str(tmp_path), "S-008")
    assert path.read_bytes() == b"3\nS-008\n"
    append_done_case(str(tmp_path), "9")
    assert path.read_text(encoding="utf-8") == "3\nS-008\n9\n"


def _stub_trace(monkeypatch, module, traced, finish_name, fail_case=None):
    def process_init(test_case, directory_data, dir_output0, img_format, test):
        name = case_name_from_sample(test_case)
        traced.append(name)
        return (
            f"{dir_output0}{test}_{name}/",
            "img.mha",
            None,
            "cent.vtp",
            name,
            0,
            isinstance(test_case, dict),
        )

    def finish(*args, **kwargs):
        if fail_case is not None and kwargs["case"] == fail_case:
            raise RuntimeError("trace failed")

    monkeypatch.setattr(f"{module}.init.process_init", process_init)
    monkeypatch.setattr(
        f"{module}.sf.resample_inputs_for_case",
        lambda *args, **kwargs: ("img.mha", None),
    )
    monkeypatch.setattr(f"{module}.sitk.ReadImage", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(
        f"{module}.sf.validate_potential_branches_image_bounds",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        f"{module}.trace_centerline_from_context",
        lambda *args, **kwargs: MagicMock(),
    )
    monkeypatch.setattr(f"{module}.try_save_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(f"{module}.create_case_directories", lambda *args, **kwargs: None)
    monkeypatch.setattr(f"{module}.{finish_name}", finish)


def _stub_plus(monkeypatch, samples, data_dir, traced, fail_case=None):
    _stub_trace(
        monkeypatch,
        "seqseg.pipeline.plus",
        traced,
        "finish_plus_case",
        fail_case=fail_case,
    )
    monkeypatch.setattr(
        "seqseg.pipeline.plus.get_testing_samples",
        lambda *args, **kwargs: (samples, str(data_dir)),
    )
    monkeypatch.setattr(
        "seqseg.pipeline.plus.sf.resolve_global_seg_files",
        lambda *args, **kwargs: ("binary.mha", "prob.mha"),
    )
    monkeypatch.setattr(
        "seqseg.pipeline.plus.sf.load_global_segmentation",
        lambda *args, **kwargs: (MagicMock(), MagicMock()),
    )
    monkeypatch.setattr(
        "seqseg.pipeline.plus.sitk.WriteImage",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "seqseg.pipeline.plus.vf.evaluate_surface",
        lambda *args, **kwargs: MagicMock(),
    )
    monkeypatch.setattr(
        "seqseg.pipeline.plus.vf.write_vtk_polydata",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "seqseg.pipeline.plus.vf.taubin_smooth_polydata",
        lambda *args, **kwargs: MagicMock(),
    )
    monkeypatch.setattr(
        "seqseg.pipeline.plus.vf.compute_polydata_normals",
        lambda *args, **kwargs: MagicMock(),
    )
    monkeypatch.setattr(
        "seqseg.pipeline.plus.init.initialize_from_seg",
        lambda *args, **kwargs: ([], [], None),
    )


def _stub_classic(monkeypatch, traced, fail_case=None):
    _stub_trace(
        monkeypatch,
        "seqseg.pipeline.classic",
        traced,
        "finish_classic_case",
        fail_case=fail_case,
    )
    monkeypatch.setattr(
        "seqseg.pipeline.classic.init.initialization",
        lambda *args, **kwargs: ([], []),
    )


def _run_plus(outdir, data_dir, samples, start, stop):
    run_plus_batch(
        dir_output0=str(outdir),
        data_dir=str(data_dir),
        seqseg_dataset="Dataset010_SEQCOROASOCACT",
        dir_model_weights_seqseg="/unused/seqseg",
        seqseg_fold="0",
        seqseg_test_name="3d_fullres",
        seqseg_scale=1.0,
        dir_model_weights_global=None,
        global_fold="0",
        global_scale=1.0,
        img_format=".mha",
        unit="mm",
        max_step_size=1,
        max_n_branches=1,
        max_n_steps_per_branch=1,
        write_samples=False,
        take_time=False,
        calc_global_centerline=False,
        cap_surface_cent=False,
        assembly_threshold=0.5,
        num_seeds=1,
        resample_spacing=None,
        global_config={"DEBUG": True},
        start=start,
        stop=stop,
        global_seg_dir=str(outdir / "global_segs"),
        start_time_global=0.0,
    )


def _run_classic(outdir, data_dir, samples, start, stop):
    run_classic_batch(
        dir_output0=str(outdir),
        data_dir=str(data_dir),
        dataset="Dataset010_SEQCOROASOCACT",
        dir_model_weights="/unused/seqseg",
        fold="0",
        img_format=".mha",
        scale=1.0,
        test_name="3d_fullres",
        pt_centerline=0,
        num_seeds=1,
        testing_samples=samples,
        directory_data=str(data_dir),
        start=start,
        stop=stop,
        global_config={"DEBUG": True},
        unit="mm",
        max_step_size=1,
        max_n_branches=1,
        max_n_steps_per_branch=1,
        write_samples=False,
        take_time=False,
        calc_global_centerline=False,
        cap_surface_cent=False,
        assembly_threshold=0.5,
        resample_spacing=None,
        start_time_global=0.0,
    )


def _prepare_dirs(tmp_path, kind, done_text):
    outdir = tmp_path / kind
    data_dir = tmp_path / f"{kind}_data"
    outdir.mkdir()
    data_dir.mkdir()
    # A segmentation file and a dataset-folder list must not cause a skip.
    (outdir / "S-008_segmentation_1_steps.mha").write_bytes(b"")
    (data_dir / "done.txt").write_text("S-008\n", encoding="utf-8")
    if done_text is not None:
        (outdir / "done.txt").write_text(done_text, encoding="utf-8")
    return outdir, data_dir


@pytest.mark.parametrize("kind", ["classic", "plus"])
def test_listed_case_skipped_and_success_appends(tmp_path, monkeypatch, capsys, kind):
    samples = [{"name": "3"}, {"name": "S-008"}, {"name": "9"}]
    original = "3\n\n3\n"
    outdir, data_dir = _prepare_dirs(tmp_path, kind, original)
    traced = []
    if kind == "plus":
        _stub_plus(monkeypatch, samples, data_dir, traced)
        _run_plus(outdir, data_dir, samples, start=0, stop=2)
    else:
        _stub_classic(monkeypatch, traced)
        _run_classic(outdir, data_dir, samples, start=0, stop=2)

    assert traced == ["S-008"]
    assert (outdir / "done.txt").read_text(encoding="utf-8") == original + "S-008\n"
    output = capsys.readouterr().out
    assert "Skipping 3: listed in done.txt" in output
    assert "Skipping S-008" not in output
    assert "Skipping 9" not in output


@pytest.mark.parametrize("kind", ["classic", "plus"])
def test_failed_case_is_not_marked_done(tmp_path, monkeypatch, kind):
    samples = [{"name": "3"}, {"name": "S-008"}]
    outdir, data_dir = _prepare_dirs(tmp_path, kind, None)
    traced = []
    if kind == "plus":
        _stub_plus(monkeypatch, samples, data_dir, traced, fail_case="S-008")
        with pytest.raises(RuntimeError, match="trace failed"):
            _run_plus(outdir, data_dir, samples, start=0, stop=2)
    else:
        _stub_classic(monkeypatch, traced, fail_case="S-008")
        with pytest.raises(RuntimeError, match="trace failed"):
            _run_classic(outdir, data_dir, samples, start=0, stop=2)

    assert traced == ["3", "S-008"]
    assert (outdir / "done.txt").read_text(encoding="utf-8") == "3\n"


def test_classic_list_sample_appends_image_stem(tmp_path, monkeypatch):
    outdir = tmp_path / "classic_list"
    data_dir = tmp_path / "classic_list_data"
    outdir.mkdir()
    data_dir.mkdir()
    traced = []
    _stub_classic(monkeypatch, traced)
    _run_classic(
        outdir,
        data_dir,
        [["S-008", 0, -50, -60]],
        start=0,
        stop=1,
    )
    assert traced == ["S-008"]
    assert (outdir / "done.txt").read_text(encoding="utf-8") == "S-008\n"
