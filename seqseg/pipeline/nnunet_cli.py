"""Run an nnU-Net planning or preprocessing entry point with a faster file lookup.

``python -m seqseg.pipeline.nnunet_cli plan_and_preprocess -d 999 ...`` takes the
same arguments as ``nnUNetv2_plan_and_preprocess``.

nnU-Net finds a dataset's training files by matching every case id against
every name in ``imagesTr``. That is quadratic in the number of cases, and it
runs again in fingerprinting, planning, preprocessing, and the label copy:
about an hour each for several hundred thousand patches. Listing the files
under ``dataset`` in dataset.json avoids it, but nnU-Net sends dataset.json to
every preprocessing job, so a long list slows preprocessing far more.

This module swaps in a single-pass lookup before nnU-Net starts. It returns
the same mapping as nnU-Net's folder scan. nnU-Net's files are not modified.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Callable, Dict, List, Optional

_ENTRY_POINTS = {
    "plan_and_preprocess": "plan_and_preprocess_entry",
    "preprocess": "preprocess_entry",
}

_CHANNEL_SUFFIX = re.compile(r"_\d{4}$")


def training_files_from_folder(raw_dataset_folder: str, dataset_json: dict) -> Dict[str, dict]:
    """Same result as nnU-Net's ``imagesTr`` scan, in one directory listing."""
    ending = dataset_json["file_ending"]
    crop = len(ending) + 5
    images_dir = os.path.join(raw_dataset_folder, "imagesTr")
    labels_dir = os.path.join(raw_dataset_folder, "labelsTr")

    images: Dict[str, List[str]] = {}
    with os.scandir(images_dir) as entries:
        for entry in entries:
            name = entry.name
            if not name.endswith(ending) or not entry.is_file():
                continue
            files = images.setdefault(name[:-crop], [])
            if _CHANNEL_SUFFIX.search(name[: -len(ending)]):
                files.append(name)

    return {
        case: {
            "images": [os.path.join(images_dir, name) for name in sorted(images[case])],
            "label": os.path.join(labels_dir, case + ending),
        }
        for case in sorted(images)
    }


def _fast_lookup(original: Callable) -> Callable:
    def get_filenames_of_train_images_and_targets(
        raw_dataset_folder: str, dataset_json: Optional[dict] = None
    ):
        if dataset_json is None:
            from batchgenerators.utilities.file_and_folder_operations import load_json

            dataset_json = load_json(os.path.join(raw_dataset_folder, "dataset.json"))
        if "dataset" in dataset_json:
            return original(raw_dataset_folder, dataset_json)
        return training_files_from_folder(raw_dataset_folder, dataset_json)

    return get_filenames_of_train_images_and_targets


def install_fast_lookup() -> None:
    """Point every loaded nnU-Net module at the single-pass lookup.

    Modules bind the function with ``from ... import``, so each one is patched.
    Only the parent process looks up files; spawned workers do not need it.
    """
    import nnunetv2.experiment_planning.plan_and_preprocess_entrypoints  # noqa: F401
    import nnunetv2.preprocessing.preprocessors.default_preprocessor  # noqa: F401
    import nnunetv2.utilities.utils as nnunet_utils

    original = nnunet_utils.get_filenames_of_train_images_and_targets
    fast = _fast_lookup(original)
    for name, module in list(sys.modules.items()):
        if not name.startswith("nnunetv2") or module is None:
            continue
        if getattr(module, "get_filenames_of_train_images_and_targets", None) is original:
            module.get_filenames_of_train_images_and_targets = fast


def main(argv: Optional[List[str]] = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in _ENTRY_POINTS:
        raise SystemExit(
            "usage: python -m seqseg.pipeline.nnunet_cli "
            f"{{{','.join(_ENTRY_POINTS)}}} [nnU-Net arguments]"
        )
    install_fast_lookup()
    from nnunetv2.experiment_planning import plan_and_preprocess_entrypoints

    command = args.pop(0)
    sys.argv = [f"nnUNetv2_{command}", *args]
    getattr(plan_and_preprocess_entrypoints, _ENTRY_POINTS[command])()


if __name__ == "__main__":
    main()
