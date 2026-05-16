"""Output directory layout for SeqSeg case folders."""

import os


def create_case_directories(output_folder: str, write_samples: bool) -> None:
    """Create per-case folder tree (classic and plus compatible)."""
    directories = [
        output_folder,
        output_folder + "errors",
        output_folder + "assembly",
        output_folder + "simvascular",
        output_folder + "simvascular/Images",
        output_folder + "simvascular/Paths",
        output_folder + "simvascular/Segmentations",
        output_folder + "simvascular/Models",
    ]
    if write_samples:
        directories.extend(
            [
                output_folder + "volumes",
                output_folder + "predictions",
                output_folder + "centerlines",
                output_folder + "surfaces",
                output_folder + "points",
                output_folder + "animation",
            ]
        )
    for directory in directories:
        try:
            os.mkdir(directory)
        except Exception as e:  # noqa: BLE001 — match legacy behavior
            print(e)
