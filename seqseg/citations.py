"""Citation banners printed when running SeqSeg CLI commands."""

from __future__ import annotations

import sys

_CITATION_BANNER = """\
#######################################################################
Please cite the following paper when using SeqSeg:
Sveinsson Cepero, N., & Shadden, S. C. (2024). SeqSeg: Learning Local Segments for Automatic Vascular Model Construction. Annals of Biomedical Engineering.
#######################################################################
Please cite the following paper when using nnU-Net:
Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.
#######################################################################"""


def print_citation_banner() -> None:
    print(_CITATION_BANNER, file=sys.stderr)
