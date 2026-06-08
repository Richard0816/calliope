"""Centralised CuPy import (optional GPU dependency).

CuPy and ``cupyx.scipy.signal`` are optional: they accelerate the dF/F
and cross-correlation paths on machines with a CUDA GPU, but a CPU-only
machine simply doesn't have them installed and every GPU code path falls
back to CPU. All GPU code imports CuPy through these helpers so the
optional dependency is handled in exactly one place.

History: this module used to assemble the module name at runtime via
``bytes.fromhex(...)`` to hide it from the Nuitka freezer's static import
scanner (which would otherwise bundle the large CUDA stack). Nuitka was
dropped 2026-06-08, so these are now plain imports. If a freezer is
reintroduced, this single chokepoint is where to exclude/hide CuPy again.
"""

from __future__ import annotations

from types import ModuleType


def import_cupy() -> ModuleType:
    """Import and return the ``cupy`` module.

    Raises ``ImportError`` when CuPy isn't installed / no CUDA device is
    reachable; callers catch that and fall back to the CPU path.
    """
    import cupy
    return cupy


def import_cupyx_scipy_signal() -> ModuleType:
    """Import and return the ``cupyx.scipy.signal`` module.

    Raises ``ImportError`` when unavailable (same handling as
    :func:`import_cupy`).
    """
    import cupyx.scipy.signal as cupyx_signal
    return cupyx_signal
