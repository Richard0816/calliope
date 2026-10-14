"""Frame rate and pixel size read from the recording's own TIFF metadata.

Detection used to fall back silently to a rig-specific 15.08 Hz when the
user left Tab 3's FPS override at 0, so a 30 Hz recording ran with every
time and duration doubled (UX audit finding 3). The acquisition software
almost always records the real values, so read them from the raw TIFFs:

* ImageJ: ``finterval`` (s/frame) or ``fps``; pixel size from the
  X resolution tag when ``unit`` is microns.
* ScanImage: ``SI.hRoiManager.scanVolumeRate`` (falls back to
  ``scanFrameRate``), divided by ``SI.hScan2D.logAverageFactor``.
* OME-TIFF: ``Pixels@TimeIncrement`` / ``Pixels@PhysicalSizeX``.

:func:`fill_from_metadata` is the one entry point the detection paths
call; a value the user typed (non-zero) always wins.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

_MICRONS = {"micron", "microns", "um", "µm", "\\u00b5m", "μm"}


def _imagej(tf) -> dict:
    md = tf.imagej_metadata or {}
    out = {}
    fi = float(md.get("finterval", 0) or 0)
    fps = float(md.get("fps", 0) or 0)
    if fi > 0:
        out["fps"] = 1.0 / fi
    elif fps > 0:
        out["fps"] = fps
    if str(md.get("unit", "")).strip().lower() in _MICRONS:
        xres = tf.pages[0].tags.get("XResolution")
        if xres is not None:
            num, den = xres.value
            if num > 0:
                out["um_per_px"] = den / num
    return out


def _scanimage(tf) -> dict:
    fd = (tf.scanimage_metadata or {}).get("FrameData", {}) or {}
    rate = float(fd.get("SI.hRoiManager.scanVolumeRate", 0) or 0) or \
        float(fd.get("SI.hRoiManager.scanFrameRate", 0) or 0)
    avg = float(fd.get("SI.hScan2D.logAverageFactor", 1) or 1)
    return {"fps": rate / max(avg, 1.0)} if rate > 0 else {}


def _ome(tf) -> dict:
    xml = tf.ome_metadata or ""
    m = re.search(r"<Pixels\b[^>]*>", xml)
    if not m:
        return {}
    attrs = dict(re.findall(r'(\w+)="([^"]*)"', m.group(0)))
    out = {}
    dt = float(attrs.get("TimeIncrement", 0) or 0)
    unit = attrs.get("TimeIncrementUnit", "s")
    dt *= {"s": 1.0, "ms": 1e-3, "us": 1e-6, "µs": 1e-6}.get(unit, 0.0)
    if dt > 0:
        out["fps"] = 1.0 / dt
    px = float(attrs.get("PhysicalSizeX", 0) or 0)
    if px > 0 and attrs.get("PhysicalSizeXUnit", "µm") in _MICRONS:
        out["um_per_px"] = px
    return out


def read_tiff_metadata(path) -> dict:
    """``{"fps", "um_per_px", "source"}`` from one TIFF; keys absent when
    not recorded. Never raises."""
    try:
        import tifffile
        with tifffile.TiffFile(str(path)) as tf:
            for name, fn in (("ImageJ", _imagej), ("ScanImage", _scanimage),
                             ("OME", _ome)):
                try:
                    found = fn(tf)
                except Exception:
                    found = {}
                if found:
                    return {**found, "source": f"{name} metadata of "
                                               f"{Path(path).name}"}
    except Exception:
        pass
    return {}


def recording_metadata(tiff_folder) -> dict:
    """Metadata for the recording preprocessed into ``tiff_folder``.

    Prefers the original raw TIFF (listed in the raw-paths sidecar, or its
    archived copy in the folder) because the shifted TIFF does not carry
    the acquisition metadata; falls back to any TIFF in the folder.
    """
    from .preprocessing import read_raw_paths_sidecar
    folder = Path(tiff_folder)
    candidates = []
    for raw in read_raw_paths_sidecar(folder):
        candidates += [raw, folder / raw.name]
    candidates += sorted(folder.glob("*.tif*"))
    for c in candidates:
        if Path(c).is_file():
            md = read_tiff_metadata(c)
            if md:
                return md
    return {}


def fill_from_metadata(params: dict, tiff_folder, log=print) -> dict:
    """Return a copy of ``params`` with ``fps_override`` / ``um_per_pixel``
    filled from TIFF metadata where the user left them at 0 (and no scope
    zoom is set for the pixel size). Logs what it used."""
    out = dict(params)
    md = recording_metadata(tiff_folder)
    if not float(out.get("fps_override", 0) or 0):
        if md.get("fps"):
            out["fps_override"] = round(float(md["fps"]), 4)
            log(f"[fps] {out['fps_override']:g} Hz from {md['source']}")
        else:
            from .calliope_settings import DEFAULT_FPS
            log(f"[fps] WARNING: no frame rate in the TIFF metadata and none "
                f"set; using the {DEFAULT_FPS:g} Hz fallback. Times and "
                f"durations are wrong if your acquisition rate differs.")
    if (not float(out.get("um_per_pixel", 0) or 0)
            and not float(out.get("scope_zoom", 0) or 0)
            and md.get("um_per_px")):
        out["um_per_pixel"] = float(md["um_per_px"])
        log(f"[scale] {out['um_per_pixel']:g} µm/px from {md['source']}")
    return out
