"""Output directory layout for SeqSeg case folders."""

import os


def create_case_directories(
    output_folder: str,
    write_samples: bool,
    simvascular: bool = False,
) -> None:
    """Create per-case folder tree (classic and plus compatible).

    ``output_folder`` may or may not end with a separator. Subfolders are
    joined, so a missing slash does not glue the name onto the parent.
    """
    root = os.fspath(output_folder)
    directories = [
        root,
        os.path.join(root, "errors"),
        os.path.join(root, "assembly"),
    ]
    if simvascular:
        directories.extend(
            [
                os.path.join(root, "simvascular"),
                os.path.join(root, "simvascular", "Images"),
                os.path.join(root, "simvascular", "Paths"),
                os.path.join(root, "simvascular", "Segmentations"),
                os.path.join(root, "simvascular", "Models"),
            ]
        )
    if write_samples:
        directories.extend(
            [
                os.path.join(root, "volumes"),
                os.path.join(root, "predictions"),
                os.path.join(root, "centerlines"),
                os.path.join(root, "surfaces"),
                os.path.join(root, "points"),
                os.path.join(root, "animation"),
            ]
        )
    for directory in directories:
        try:
            os.mkdir(directory)
        except Exception as e:  # noqa: BLE001 — match legacy behavior
            print(e)
