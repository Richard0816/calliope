"""Pixel <-> micrometre conversion for the imaging FOV.

Why this module exists
----------------------
Several tabs and post-processing helpers need to render ROI / spatial
plots on physical (µm) axes instead of raw pixels. The conversion
factor used to be inlined in two places:

* ``clustering_cmap._resolve_pix_to_um`` -- read ``ops['pix_to_um']``,
  fall back to a notes-CSV zoom lookup that was actually broken
  (called ``utils.get_zoom_from_notes`` which doesn't exist; the
  fallback was silently dead).
* ``spatial_propagation/tab.py`` -- read ``ops['pix_to_um']`` directly.

This module centralises the maths so callers only have one entry
point and adds a *direct* µm-per-pixel override (in addition to the
zoom-based formula) so users with a calibration straight from their
microscope can plug it in without computing zoom.

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
``resolve_pix_to_um(ops, *, zoom=None, um_per_pixel=None)``
    Single resolver used by callers. Priority order:
    explicit ``um_per_pixel`` -> explicit ``zoom`` -> ``ops['pix_to_um']``
    -> ``None``. ``zoom`` requires ``ops['Lx']`` to be available.
"""

from __future__ import annotations

from typing import Optional


# Reference FOV width at 1x scope zoom for the lab's two-photon rig.
# Sourced from the manufacturer calibration that was hard-coded in
# the legacy ``clustering_cmap`` resolver. Override at the call site
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
      3. ``ops['pix_to_um']`` previously stored on the ops dict.
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
