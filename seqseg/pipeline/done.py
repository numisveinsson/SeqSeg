"""Skip finished trace-batch cases via ``done.txt`` in the output directory.

One case per line. For a trace batch the name is the ``name`` field from
``seeds.json`` (for example ``3`` or ``S-008``). When the dataset has no
seeds file it is the image stem. A finished case also records how long that
case took, in seconds, after a tab::

    S-008<TAB>842.105 s

Lines written before timing was recorded are only the case name. A missing
file means nothing is done. Blank lines and duplicate names are ignored.
The segmentation ``.mha`` is not a skip list.
"""

from __future__ import annotations

import os
from typing import Mapping, Set


def done_txt_path(outdir: str) -> str:
    """Path of the skip list for a batch output directory (``-outdir``)."""
    return os.path.join(os.path.normpath(outdir), "done.txt")


def case_name_from_done_line(line: str) -> str:
    """Case name from one ``done.txt`` line.

    The name is the text before the first tab. Older lines are only the name.
    """
    return line.strip().split("\t", 1)[0].strip()


def read_done_cases(outdir: str) -> Set[str]:
    """Case names already listed in ``outdir/done.txt``."""
    path = done_txt_path(outdir)
    if not os.path.isfile(path):
        return set()
    names: Set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            name = case_name_from_done_line(line)
            if name:
                names.add(name)
    return names


def append_done_case(
    outdir: str,
    name: str,
    seconds: float | None = None,
) -> None:
    """Append ``name`` without rewriting or deleting existing lines.

    ``seconds`` is how long that case took. It is written after a tab so a
    later run still reads the case name.
    """
    text = str(name).strip()
    if not text:
        return
    line = text if seconds is None else f"{text}\t{float(seconds):.3f} s"
    path = done_txt_path(outdir)
    separator = ""
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) not in b"\r\n":
                separator = "\n"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"{separator}{line}\n")


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


def mark_case_done(
    outdir: str,
    done: Set[str],
    name: str,
    seconds: float | None = None,
) -> None:
    """Record a case that finished after its final segmentation was written."""
    append_done_case(outdir, name, seconds=seconds)
    done.add(name)
