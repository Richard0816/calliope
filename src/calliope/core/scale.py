"""Pixel <-> micrometre conversion for the imaging FOV.

Why this module exists
----------------------
Several tabs and post-processing helpers need to render ROI / spatial
plots on physical (µm) axes instead of raw pixels. This module is the
single place the conversion maths lives, and it accepts either the
scope's zoom factor or a *direct* µm-per-pixel calibration straight
from the microscope.

Where the calibration is stored
-------------------------------
Tab 3 stamps the resolved value into ``<plane0>/calliope_calibration.npy``
(``utils.save_pix_to_um``). Readers normally pass
``utils.load_plane_view(plane0)`` as ``ops``: that view layers the
calibration file in as ``view['pix_to_um']`` (legacy plane folders
without the file fall back to an ``ops.npy['pix_to_um']`` key).

Conversion formula
------------------
The lab's reference FOV is ~3080.9 µm wide at 1x zoom. At zoom Z the
FOV shrinks proportionally: ``fov_um(Z) = FOV_UM_REFERENCE / Z``.
Suite2p reports the registered movie shape as ``Lx`` x ``Ly``
pixels, so::

    pix_to_um = (FOV_UM_REFERENCE / zoom) / Lx

For non-square pixels you'd need a separate y-conversion; the lab's
two-photon setup is square-pixel so a scalar suffices.

Public API
----------
``pix_to_um_from_zoom(zoom, Lx, fov_um=FOV_UM_REFERENCE)``
    Compute µm-per-pixel from the scope's zoom factor + image width.
``pix_to_um_direct(um_per_pixel)``
    Validation passthrough for an explicit calibration.
``resolve_pix_to_um(ops, *, zoom=None, um_per_pixel=None, fov_um=FOV_UM_REFERENCE)``
    Single resolver used by callers. Priority order:
    explicit ``um_per_pixel`` -> explicit ``zoom`` -> ``ops['pix_to_um']``
    (the calibration layered in by ``load_plane_view``) -> ``None``.
    ``zoom`` requires ``ops['Lx']`` to be available.
"""

from __future__ import annotations

from typing import Optional


# Reference FOV width at 1x scope zoom for the lab's two-photon rig
# (manufacturer calibration). Override at the call site via ``fov_um``
# if a recording was made on a different rig.
FOV_UM_REFERENCE = 3080.90169


def pix_to_um_from_zoom(zoom: float, Lx: int,
                        fov_um: float = FOV_UM_REFERENCE) -> float:
    """Return µm per pixel from scope zoom and image width.

    Raises ``ValueError`` if ``zoom`` or ``Lx`` is non-positive.
    """
    z = float(zoom)
    if z <= 0:
        raise ValueError(f"zoom must be > 0, got {zoom!r}")
    L = int(Lx)
    if L <= 0:
        raise ValueError(f"Lx must be > 0, got {Lx!r}")
    return (float(fov_um) / z) / L


def pix_to_um_direct(um_per_pixel: float) -> float:
    """Validate and return an explicit µm-per-pixel calibration.

    Raises ``ValueError`` if ``um_per_pixel`` is non-positive.
    """
    v = float(um_per_pixel)
    if v <= 0:
        raise ValueError(f"um_per_pixel must be > 0, got {um_per_pixel!r}")
    return v


def resolve_pix_to_um(ops: Optional[dict] = None, *,
                      zoom: Optional[float] = None,
                      um_per_pixel: Optional[float] = None,
                      fov_um: float = FOV_UM_REFERENCE) -> Optional[float]:
    """Pick a µm-per-pixel value from the available sources.

    Priority order (first match wins):
      1. ``um_per_pixel`` explicit kwarg (direct calibration).
      2. ``zoom`` explicit kwarg + ``ops['Lx']``.
      3. ``ops['pix_to_um']`` -- pass ``utils.load_plane_view(plane0)``
         as ``ops`` so this is the value from
         ``calliope_calibration.npy`` (or a legacy ``ops.npy`` key).
      4. ``None`` (caller should treat axes as raw pixels).

    Returns ``None`` if no source resolved to a usable positive value.
    """
    if um_per_pixel is not None:
        try:
            return pix_to_um_direct(um_per_pixel)
        except (TypeError, ValueError):
            pass
    if zoom is not None and ops is not None and ops.get("Lx"):
        try:
            return pix_to_um_from_zoom(zoom, ops["Lx"], fov_um=fov_um)
        except (TypeError, ValueError):
            pass
    if ops is not None:
        v = ops.get("pix_to_um")
        if v is not None:
            try:
                return pix_to_um_direct(v)
            except (TypeError, ValueError):
                return None
    return None
