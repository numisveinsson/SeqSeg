"""Device selection for nnU-Net inference (no model load)."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

from seqseg.modules.nnunet import select_inference_device


def _fake_torch(*, cuda=False, mps=False):
    device = MagicMock(side_effect=lambda kind, index=None: (kind, index))
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        device=device,
    )


def test_select_inference_device_force_cpu_skips_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=True, mps=True))
    assert select_inference_device(".../3d_fullres", force_cpu=True) == ("cpu", 0)


def test_select_inference_device_uses_cuda_when_available(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=True, mps=False))
    assert select_inference_device(".../3d_fullres", force_cpu=False) == ("cuda", 0)


def test_select_inference_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False, mps=False))
    assert select_inference_device(".../3d_fullres", force_cpu=False) == ("cpu", 0)


def test_select_inference_device_mps_only_for_2d(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False, mps=True))
    assert select_inference_device(".../2d", force_cpu=False) == ("mps", None)
    assert select_inference_device(".../3d_fullres", force_cpu=False) == ("cpu", 0)
