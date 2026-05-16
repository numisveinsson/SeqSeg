"""
Sweep + SeqSeg pipeline (global nnU-Net sweep, then sequential tracing).

Prefer: ``seqseg run plus batch <args>``. This module remains for
``python -m seqseg.seqseg_plus`` and existing job scripts.
"""

import sys

from seqseg.cli import dispatch


def main() -> None:
    argv = sys.argv[1:]
    sys.argv = [sys.argv[0], "run", "plus", "batch", *argv]
    dispatch()


if __name__ == "__main__":
    main()
