"""Centralised CuPy import (optional GPU dependency).

CuPy is optional: it accelerates the dF/F and cross-correlation paths on
machines with a CUDA GPU, but a CPU-only machine simply doesn't have it
installed and every GPU code path falls back to CPU. All GPU code imports CuPy through these helpers so the
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

    Raises ``ImportError`` when CuPy isn't installed **or** no CUDA device
    is reachable; callers catch that and fall back to the CPU path.

    ``import cupy`` on its own succeeds on a machine that has CuPy installed
    but no usable GPU/driver -- the failure would only surface at the first
    device operation, past every caller's ``except ImportError``. So after
    importing we ask the CUDA runtime for the device count (cheap; it does
    not create a device context) and turn "no device" into ImportError.
    This is a best-effort availability probe: it cannot guarantee that a
    later allocation or kernel launch succeeds.
    """
    import cupy
    try:
        n_devices = int(cupy.cuda.runtime.getDeviceCount())
    except Exception as exc:  # CUDARuntimeError, missing driver, ...
        raise ImportError(
            f"CuPy is installed but no CUDA device is reachable ({exc})"
        ) from exc
    if n_devices < 1:
        raise ImportError("CuPy is installed but no CUDA device was found")
    return cupy

