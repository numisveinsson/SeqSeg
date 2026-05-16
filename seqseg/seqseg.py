"""
SeqSeg: Sequential Vessel Segmentation and Centerline Tracing

Console script entry: ``seqseg`` → :func:`main`.
"""

from seqseg.config_models import load_yaml_config
from seqseg.cli import main, dispatch

__all__ = ("main", "dispatch", "load_yaml_config")


if __name__ == "__main__":
    main()
