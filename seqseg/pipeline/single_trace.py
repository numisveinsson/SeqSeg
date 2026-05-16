"""Prepare a minimal dataset tree for ``seqseg run single``."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


def _stem_and_suffix(image_path: str) -> Tuple[str, str]:
    """Return (stem, extension_with_dot) for common medical filenames."""
    name = Path(image_path).name
    for suf in (".nii.gz", ".NII.GZ"):
        if name.lower().endswith(suf.lower()):
            return name[: -len(suf)], suf.lower()
    p = Path(image_path)
    return p.stem, p.suffix.lower() if p.suffix else ""


def prepare_single_image_dataset(
    image_path: str,
    output_root: str,
    *,
    seeds_xyzr: Optional[Sequence[Sequence[float]]] = None,
    tangent: Sequence[float] = (0.0, 0.0, 1.0),
    seed_step: float = 1.0,
    seeds_json_path: Optional[str] = None,
) -> Tuple[str, str, str]:
    """
    Build ``images/``, ``centerlines/``, ``truths/``, ``seeds.json`` under a staging folder.

    Returns
    -------
    data_dir, img_ext, case_name
        ``data_dir`` ends with ``os.sep`` (expected by initialization helpers).
    """
    image_path = os.path.abspath(image_path)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"image not found: {image_path}")

    staging = os.path.join(os.path.abspath(output_root), "_seqseg_single_staging")
    images_d = os.path.join(staging, "images")
    os.makedirs(images_d, exist_ok=True)

    stem, ext = _stem_and_suffix(image_path)
    if not ext:
        raise ValueError(
            f"Could not infer file extension from {image_path!r}; use .nii.gz, .mha, etc."
        )

    if seeds_json_path:
        shutil.copy2(seeds_json_path, os.path.join(staging, "seeds.json"))
        with open(os.path.join(staging, "seeds.json"), encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            case_name = str(data[0].get("name", stem))
        elif isinstance(data, dict):
            case_name = str(data.get("name", stem))
        else:
            case_name = stem
        dest_img = os.path.join(images_d, case_name + ext)
        shutil.copy2(image_path, dest_img)
    else:
        case_name = stem
        if not seeds_xyzr:
            raise ValueError("Provide seeds_xyzr or seeds_json_path")
        dest_img = os.path.join(images_d, case_name + ext)
        shutil.copy2(image_path, dest_img)

        t = np.asarray(tangent, dtype=np.float64).ravel()
        n = float(np.linalg.norm(t))
        if n < 1e-9:
            raise ValueError("tangent norm is too small")
        t = t / n
        seed_rows: List = []
        for row in seeds_xyzr:
            if len(row) != 4:
                raise ValueError("each --seed must be x y z radius")
            nx, ny, nz, r = map(float, row)
            new_pt = [nx, ny, nz]
            old_pt = (np.array(new_pt, dtype=np.float64) - float(seed_step) * t).tolist()
            seed_rows.append([old_pt, new_pt, r])
        payload = [
            {"name": case_name, "seeds": seed_rows, "cardiac_mesh": False},
        ]
        with open(os.path.join(staging, "seeds.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    data_dir = staging + os.sep
    return data_dir, ext, case_name
