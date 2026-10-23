"""
Typed configuration helpers for SeqSeg runs.

Algorithm parameters that still live in YAML are exposed as AlgorithmConfig
(subclass of UserDict) so existing code can use config["KEY"] unchanged.
"""

from __future__ import annotations

import os
from collections import UserDict
from dataclasses import dataclass
from importlib import resources
from typing import Any, Mapping, Optional

import yaml


def load_yaml_config(config_name: str) -> dict[str, Any]:
    """
    Load a YAML file from seqseg.config package data, with filesystem fallback.
    """
    try:
        with resources.files("seqseg.config").joinpath(
            f"{config_name}.yaml"
        ).open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except (ImportError, FileNotFoundError, ModuleNotFoundError, OSError):
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        config_dir = os.path.join(pkg_dir, "config")
        config_path = os.path.join(config_dir, f"{config_name}.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        raise FileNotFoundError(
            f"Configuration file '{config_name}.yaml' not found in {config_dir}"
        ) from None


class AlgorithmConfig(UserDict[str, Any]):
    """YAML-backed algorithm settings; supports dict-style access."""

    @classmethod
    def from_name(cls, config_name: str) -> AlgorithmConfig:
        data = load_yaml_config(config_name)
        if not isinstance(data, dict):
            raise ValueError(f"Config {config_name!r} must parse to a mapping")
        return cls(data)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.data)


@dataclass(frozen=True)
class NnUNetModelSpec:
    """Resolved nnU-Net trainer folder (under nnUNet_results or relative)."""

    train_dataset: str
    nnunet_type: str = "3d_fullres"
    results_path: Optional[str] = None
    fold: str = "all"
    scale: float = 1.0

    def model_folder(self) -> str:
        rel = (
            f"{self.train_dataset}/nnUNetTrainer__nnUNetPlans__{self.nnunet_type}"
        )
        if self.results_path is not None:
            return os.path.join(self.results_path, rel)
        return rel


def trainer_folders_for_configuration(
    dataset_dir: str, configuration: str
) -> list[str]:
    """Trainer folders ``<Trainer>__<Plans>__<configuration>`` under a dataset."""
    if not os.path.isdir(dataset_dir):
        return []
    found = []
    for name in sorted(os.listdir(dataset_dir)):
        parts = name.split("__")
        if len(parts) != 3 or parts[2] != configuration:
            continue
        path = os.path.join(dataset_dir, name)
        if os.path.isdir(path):
            found.append(path)
    return found


def resolve_nnunet_model_folder(spec: NnUNetModelSpec) -> str:
    """Return the trainer folder ``seqseg run`` should load.

    The dataset under ``results_path`` is searched for the trainer folder of
    ``nnunet_type``, whatever trainer and plans wrote it. When none is found
    the default ``nnUNetTrainer`` path is returned. More than one raises
    ``ValueError``.
    """
    if spec.results_path is None:
        return spec.model_folder()
    dataset_dir = os.path.join(spec.results_path, spec.train_dataset)
    found = trainer_folders_for_configuration(dataset_dir, spec.nnunet_type)
    if not found:
        return spec.model_folder()
    if len(found) > 1:
        names = "\n  ".join(os.path.basename(path) for path in found)
        raise ValueError(
            f"Several {spec.nnunet_type} trainer folders under {dataset_dir}:\n"
            f"  {names}\n"
            "Keep one trainer folder per configuration; move the others out "
            "of this dataset folder."
        )
    return found[0]


@dataclass(frozen=True)
class TracingLimits:
    max_n_steps: int = 1000
    max_n_branches: int = 100
    max_n_steps_per_branch: int = 100
    write_samples: bool = False
    unit: str = "cm"
    scale: float = 1.0


@dataclass(frozen=True)
class CaseIO:
    """Per-case filesystem layout for a SeqSeg run."""

    dir_output: str
    dir_image: str
    dir_seg: Optional[str]
    dir_cent: str
    case: str
    sample_index: int
    json_file_present: bool
    dir_output_root: str
    test_name: str

    @property
    def simvascular_dir(self) -> str:
        return os.path.join(self.dir_output, "simvascular")
