"""Skip finished trace-batch cases via ``done.txt`` in the output directory.

One case name per line, the same format as training's ``done.txt``. For a
trace batch that name is the ``name`` field from ``seeds.json`` (for example
``3`` or ``S-008``). When the dataset has no seeds file it is the image stem.
A missing file means nothing is done. Blank lines and duplicate names are
ignored. The segmentation ``.mha`` is not a skip list.
"""

from __future__ import annotations

import os
from typing import Mapping, Set


def done_txt_path(outdir: str) -> str:
    """Path of the skip list for a batch output directory (``-outdir``)."""
    return os.path.join(os.path.normpath(outdir), "done.txt")


def read_done_cases(outdir: str) -> Set[str]:
    """Case names already listed in ``outdir/done.txt``."""
    path = done_txt_path(outdir)
    if not os.path.isfile(path):
        return set()
    with open(path, encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def append_done_case(outdir: str, name: str) -> None:
    """Append ``name`` without rewriting or deleting existing lines."""
    text = str(name).strip()
    if not text:
        return
    path = done_txt_path(outdir)
    separator = ""
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) not in b"\r\n":
                separator = "\n"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"{separator}{text}\n")


def case_name_from_sample(test_case) -> str:
    """Case name stored in ``done.txt`` for one testing sample."""
    if isinstance(test_case, Mapping):
        return str(test_case["name"]).strip()
    return str(test_case[0]).strip()


def skip_if_done(done: Set[str], name: str) -> bool:
    """Print a one-line skip and return True when ``name`` is already listed."""
    if name not in done:
        return False
    print(f"Skipping {name}: listed in done.txt")
    return True


def mark_case_done(outdir: str, done: Set[str], name: str) -> None:
    """Record a case that finished after its final segmentation was written."""
    append_done_case(outdir, name)
    done.add(name)
