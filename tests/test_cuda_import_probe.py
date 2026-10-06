"""import_cupy must raise ImportError (the exception every caller catches to
fall back to CPU) when CuPy imports fine but no CUDA device is reachable."""
from __future__ import annotations

import sys
import types

import pytest

from calliope.core import _cuda_import


def _fake_cupy(get_count):
    cp = types.ModuleType("cupy")
    cuda = types.SimpleNamespace(runtime=types.SimpleNamespace(
        getDeviceCount=get_count))
    cp.cuda = cuda
    return cp


def _raise():
    raise RuntimeError("cudaErrorNoDevice")


@pytest.mark.parametrize("get_count", [lambda: 0, _raise])
def test_no_device_raises_importerror(monkeypatch, get_count):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(get_count))
    with pytest.raises(ImportError):
        _cuda_import.import_cupy()


def test_device_present_returns_module(monkeypatch):
    fake = _fake_cupy(lambda: 1)
    monkeypatch.setitem(sys.modules, "cupy", fake)
    assert _cuda_import.import_cupy() is fake
