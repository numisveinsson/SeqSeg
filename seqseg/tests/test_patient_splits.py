"""Patient-level nnU-Net folds built from name_mapping.txt."""

import json
from pathlib import Path

import pytest

from seqseg.pipeline.train import (
    _install_patient_level_splits,
    _kfold_id_splits,
    _patient_id_from_extracted_name,
    _splits_match,
)


def _write_mapping(folder: Path, pairs, modality="ct"):
    """pairs: (nnU-Net case id, extracted stem)."""
    folder.mkdir(parents=True, exist_ok=True)
    lines = ["# nnunet_name -> original_path\n"]
    for case_id, source in pairs:
        lines.append(
            f"imagesTr/{case_id}_0000.nii.gz -> {modality}_train/{source}.nii.gz\n"
        )
        lines.append(
            f"labelsTr/{case_id}.nii.gz -> {modality}_train_masks/{source}.nii.gz\n"
        )
    (folder / "name_mapping.txt").write_text("".join(lines))


def _pairs_for(patients, prefix):
    pairs = []
    n = 1
    for patient in patients:
        for index, sub in ((0, 0), (0, 1), (1, 0)):
            pairs.append((f"{prefix}_{n:03d}", f"{patient}_{index}_{sub}"))
            n += 1
    return pairs


def _load(folder: Path):
    return json.loads((folder / "splits_final.json").read_text())


def _patients_of(case_ids, case_to_patient):
    return {case_to_patient[case] for case in case_ids}


def _case_to_patient(pairs):
    return {case_id: _patient_id_from_extracted_name(source) for case_id, source in pairs}


def test_patient_id_strips_patch_indices_only():
    assert _patient_id_from_extracted_name("foo_bar_3_1.nii.gz") == "foo_bar"
    assert _patient_id_from_extracted_name("patient_01_3_2.nii.gz") == "patient_01"
    assert _patient_id_from_extracted_name("foo_bar.nii.gz") == "foo_bar"


def test_two_patients_stay_on_one_side_of_each_fold(tmp_path):
    patients = ["alpha", "beta"]
    pairs = _pairs_for(patients, "ct")
    folder = tmp_path / "Dataset999_MYDATACT"
    _write_mapping(folder, pairs)
    _install_patient_level_splits([str(folder)])

    case_to_patient = _case_to_patient(pairs)
    splits = _load(folder)
    assert len(splits) == 2
    seen_val = []
    for fold in splits:
        train_p = _patients_of(fold["train"], case_to_patient)
        val_p = _patients_of(fold["val"], case_to_patient)
        assert train_p.isdisjoint(val_p)
        assert train_p | val_p == set(patients)
        assert len(fold["train"]) == 3
        assert len(fold["val"]) == 3
        seen_val.append(val_p)
    assert seen_val[0] != seen_val[1]


def test_same_patients_hold_out_the_same_people_in_fold_0(tmp_path):
    patients = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    ct_pairs = _pairs_for(patients, "ct")
    mr_pairs = _pairs_for(patients, "mr")
    ct = tmp_path / "Dataset999_MYDATACT"
    mr = tmp_path / "Dataset1000_MYDATAMR"
    _write_mapping(ct, ct_pairs, modality="ct")
    _write_mapping(mr, mr_pairs, modality="mr")
    _install_patient_level_splits([str(ct), str(mr)])

    ct_splits = _load(ct)
    mr_splits = _load(mr)
    assert len(ct_splits) == 5
    assert len(mr_splits) == 5
    ct_map = _case_to_patient(ct_pairs)
    mr_map = _case_to_patient(mr_pairs)
    assert _patients_of(ct_splits[0]["val"], ct_map) == _patients_of(
        mr_splits[0]["val"], mr_map
    )
    for fold in ct_splits:
        assert _patients_of(fold["train"], ct_map).isdisjoint(
            _patients_of(fold["val"], ct_map)
        )


def test_one_patient_does_not_write_a_split(tmp_path, capsys):
    folder = tmp_path / "Dataset999_MYDATACT"
    _write_mapping(folder, _pairs_for(["only"], "ct"))
    _install_patient_level_splits([str(folder)])
    assert not (folder / "splits_final.json").exists()
    assert "held-out fold is impossible" in capsys.readouterr().out


def test_patch_split_is_replaced_and_custom_split_is_kept(tmp_path, capsys):
    patients = ["a", "b", "c", "d", "e", "f"]
    pairs = _pairs_for(patients, "ct")
    raw = tmp_path / "raw" / "Dataset999_MYDATACT"
    pre = tmp_path / "pre" / "Dataset999_MYDATACT"
    _write_mapping(raw, pairs)
    pre.mkdir(parents=True)
    case_ids = sorted(case for case, _source in pairs)
    patch = _kfold_id_splits(case_ids, 5)
    (pre / "splits_final.json").write_text(json.dumps(patch))

    _install_patient_level_splits([str(raw)], str(tmp_path / "pre"))

    patient = _load(raw)
    assert _splits_match(patient, _load(pre))
    assert not _splits_match(patient, patch)
    assert "Replacing patch-level split" in capsys.readouterr().out

    custom = [{"train": ["kept_train"], "val": ["kept_val"]}]
    (raw / "splits_final.json").write_text(json.dumps(custom))
    with pytest.raises(RuntimeError, match="custom split"):
        _install_patient_level_splits([str(raw)], str(tmp_path / "pre"))
    assert json.loads((raw / "splits_final.json").read_text()) == custom
    assert _splits_match(_load(pre), patient)


def test_kfold_matches_nnunet_default():
    from nnunetv2.utilities.crossval_split import generate_crossval_split

    ids = [f"case_{i:03d}" for i in range(8)]
    theirs = generate_crossval_split(list(ids), seed=12345, n_splits=5)
    ours = _kfold_id_splits(ids, 5)
    normalized = [
        {
            "train": [str(case) for case in fold["train"]],
            "val": [str(case) for case in fold["val"]],
        }
        for fold in theirs
    ]
    assert _splits_match(normalized, ours)
