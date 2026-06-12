"""NWB (Neurodata Without Borders) export.

Writes a single ``.nwb`` file from a finished CalLIOPE ``suite2p/plane0``
folder so recordings can be shared via DANDI / opened in neurosift, pynwb,
MATLAB, etc. The exporter is intentionally read-only with respect to the
pipeline outputs: it consumes ``F.npy`` / ``Fneu.npy`` / ``spks.npy`` /
``stat.npy`` / ``iscell.npy``, the ``r0p7_dff.memmap.float32`` dF/F memmap,
and the :mod:`calliope.core.recording_meta` ``meta.json`` sidecar as its
single source of truth for imaging metadata (fps, tau, neuropil coef,
GCaMP variant). Event times from ``calliope_summary.xlsx`` are added when
present.

``pynwb`` is an *optional* dependency (``pip install calliope[nwb]``). All
imports are deferred and guarded by :func:`nwb_available` so importing this
module never fails on a base install.

Layout of the produced file::

    /general/devices/Microscope
    /general/optophysiology/ImagingPlane
    /processing/ophys/ImageSegmentation/PlaneSegmentation   (per-ROI ImageMask + iscell prob)
    /processing/ophys/Fluorescence/{Fluorescence,Neuropil,DfOverF}  RoiResponseSeries
    /processing/ophys/Deconvolved/Deconvolved               RoiResponseSeries (OASIS)
    /intervals/events                                       (optional) population event windows
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union
from uuid import uuid4

import numpy as np

from . import recording_meta, utils

PathLike = Union[str, Path]


def nwb_available() -> bool:
    """Return True iff ``pynwb`` is importable (the optional ``[nwb]`` extra)."""
    try:
        import pynwb  # noqa: F401

        return True
    except Exception:
        return False


def _session_start_time(plane0: Path) -> datetime:
    """Best-effort acquisition timestamp.

    NWB requires a timezone-aware ``session_start_time``. We don't store the
    original acquisition time, so we use the ``stat.npy`` mtime (close to
    when detection produced the outputs) and fall back to "now" in UTC.
    """
    try:
        stat = plane0 / "stat.npy"
        ts = stat.stat().st_mtime if stat.exists() else None
        if ts:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
    except OSError:
        pass
    return datetime.now(timezone.utc)


def _build_image_mask(s: dict, Ly: int, Lx: int) -> np.ndarray:
    """Build a single ROI's (Ly, Lx) float32 weighted mask from a stat dict."""
    mask = np.zeros((Ly, Lx), dtype=np.float32)
    ypix = np.asarray(s.get("ypix", []), dtype=np.int64)
    xpix = np.asarray(s.get("xpix", []), dtype=np.int64)
    lam = np.asarray(s.get("lam", []), dtype=np.float32)
    if ypix.size == 0:
        return mask
    if lam.size != ypix.size:  # uniform weight fallback
        lam = np.ones(ypix.size, dtype=np.float32)
    ok = (ypix >= 0) & (ypix < Ly) & (xpix >= 0) & (xpix < Lx)
    mask[ypix[ok], xpix[ok]] = lam[ok]
    return mask


def _read_event_windows(plane0: Path) -> Optional[np.ndarray]:
    """Return an (n_events, 2) array of [start_s, end_s] from the summary
    workbook, or None if no event sheet is present/readable."""
    xlsx = plane0 / "calliope_summary.xlsx"
    if not xlsx.exists():
        return None
    try:
        import pandas as pd

        df = pd.read_excel(xlsx, sheet_name="EventWindows")
    except Exception:
        return None
    if "start_s" not in df.columns or "end_s" not in df.columns:
        return None
    starts = df["start_s"].to_numpy(dtype=np.float64)
    ends = df["end_s"].to_numpy(dtype=np.float64)
    if starts.size == 0:
        return None
    return np.column_stack([starts, ends])


def export_recording_to_nwb(
    plane0: PathLike,
    out_path: PathLike,
    *,
    recording_id: Optional[str] = None,
    session_description: str = "CalLIOPE two-photon calcium imaging recording",
    include_events: bool = True,
    overwrite: bool = True,
    subject_metadata: Optional[dict] = None,
) -> Path:
    """Export a finished ``plane0`` folder to an ``.nwb`` file.

    Parameters
    ----------
    plane0 : path
        A populated ``suite2p/plane0`` folder (must contain at least
        ``stat.npy``, ``F.npy`` and ``ops.npy``).
    out_path : path
        Destination ``.nwb`` file.
    recording_id : str, optional
        Human-readable id; inferred from the path when omitted.
    include_events : bool
        When True, add population event windows from
        ``calliope_summary.xlsx`` (if present) as an NWB TimeIntervals
        table named ``events``.
    overwrite : bool
        Overwrite an existing ``out_path``.

    Returns
    -------
    Path
        The written ``.nwb`` path.

    Raises
    ------
    ImportError
        If ``pynwb`` is not installed (install the ``[nwb]`` extra).
    FileNotFoundError
        If required inputs are missing from ``plane0``.
    """
    if not nwb_available():
        raise ImportError(
            "NWB export requires pynwb. Install the optional extra: "
            "pip install 'calliope[nwb]'  (or pip install pynwb)."
        )

    from pynwb import NWBHDF5IO, NWBFile
    from pynwb.ophys import (
        Fluorescence,
        ImageSegmentation,
        OpticalChannel,
    )

    plane0 = Path(plane0)
    out_path = Path(out_path)
    if not (plane0 / "stat.npy").exists() or not (plane0 / "F.npy").exists():
        raise FileNotFoundError(
            f"{plane0} is not a finished plane0 folder (need stat.npy + F.npy)"
        )
    if out_path.exists() and not overwrite:
        raise FileExistsError(out_path)

    rec_id = recording_id or utils.infer_recording_id(plane0)
    meta = recording_meta.read_meta(plane0)

    # ---- load arrays -----------------------------------------------------
    F = np.load(plane0 / "F.npy", allow_pickle=False)          # (N, T)
    Fneu = np.load(plane0 / "Fneu.npy", allow_pickle=False)
    N, T = F.shape
    spks = None
    if (plane0 / "spks.npy").exists():
        spks = np.load(plane0 / "spks.npy", allow_pickle=False)
    iscell = None
    if (plane0 / "iscell.npy").exists():
        iscell = np.load(plane0 / "iscell.npy", allow_pickle=False)
    stat = list(np.load(plane0 / "stat.npy", allow_pickle=True))

    view = utils.load_plane_view(plane0)
    Ly = int(view.get("Ly"))
    Lx = int(view.get("Lx"))

    # frame rate: meta.json first, then notes lookup default
    fps = meta.get("dff", {}).get("fps")
    if not fps or float(fps) <= 0:
        fps = float(utils.get_fps_from_notes(str(plane0)))
    fps = float(fps)

    pix_to_um = utils.load_pix_to_um(plane0)  # microns per pixel or None

    # dF/F memmap (T, N) -- optional
    dff = None
    dff_path = plane0 / "r0p7_dff.memmap.float32"
    if dff_path.exists():
        try:
            dff = np.memmap(
                str(dff_path), dtype="float32", mode="r", shape=(T, N)
            )
        except (OSError, ValueError) as exc:
            print(f"[nwb] could not open dF/F memmap ({exc}); skipping DfOverF")
            dff = None

    gcamp = meta.get("extraction", {}).get("gcamp_variant_label")
    tau = meta.get("extraction", {}).get("tau_seconds")

    # ---- build the NWBFile ----------------------------------------------
    nwbfile = NWBFile(
        session_description=session_description,
        identifier=str(uuid4()),
        session_start_time=_session_start_time(plane0),
        session_id=rec_id,
        institution="",
        source_script="calliope.core.nwb_export",
        source_script_file_name="nwb_export.py",
    )

    # DANDI requires a Subject. The pipeline does not capture animal
    # metadata, so we stamp a minimal placeholder keyed off the recording
    # id; callers heading for DANDI upload should pass ``subject_metadata``
    # (subject_id / species / sex / age / date_of_birth) with real values.
    # We never fabricate age/date_of_birth -- without ``subject_metadata``
    # nwbinspector will (correctly) flag missing demographics.
    from pynwb.file import Subject

    subj_kwargs = {
        "subject_id": rec_id,
        "description": "subject metadata not captured by the CalLIOPE pipeline",
        "species": "unknown",
        "sex": "U",
    }
    if subject_metadata:
        subj_kwargs.update(
            {k: v for k, v in subject_metadata.items() if v is not None}
        )
    nwbfile.subject = Subject(**subj_kwargs)

    device = nwbfile.create_device(
        name="Microscope",
        description="Two-photon microscope (CalLIOPE pipeline)",
    )
    optical_channel = OpticalChannel(
        name="OpticalChannel",
        description="green calcium-indicator channel",
        emission_lambda=513.0,
    )
    imaging_plane = nwbfile.create_imaging_plane(
        name="ImagingPlane",
        optical_channel=optical_channel,
        imaging_rate=fps,
        description=(
            f"GCaMP variant: {gcamp or 'unknown'}; "
            f"OASIS tau={tau} s; neuropil_coef="
            f"{meta.get('extraction', {}).get('neuropil_coef')}"
        ),
        device=device,
        excitation_lambda=920.0,
        indicator=str(gcamp or "GCaMP"),
        location="unknown",
        grid_spacing=(
            [float(pix_to_um), float(pix_to_um)] if pix_to_um else None
        ),
        grid_spacing_unit="micrometers" if pix_to_um else "n/a",
    )

    ophys = nwbfile.create_processing_module(
        name="ophys", description="CalLIOPE optical physiology outputs"
    )

    # ---- plane segmentation (ROI masks + iscell) ------------------------
    img_seg = ImageSegmentation()
    ophys.add(img_seg)
    plane_seg = img_seg.create_plane_segmentation(
        name="PlaneSegmentation",
        description="Suite2p Sparsery + Cellpose ROIs curated by cellfilter",
        imaging_plane=imaging_plane,
    )
    for s in stat:
        plane_seg.add_roi(image_mask=_build_image_mask(s, Ly, Lx))

    if iscell is not None:
        cell_prob = (
            iscell[:, 0] if iscell.ndim == 2 else iscell.astype(np.float32)
        )
        plane_seg.add_column(
            name="iscell_prob",
            description="cellfilter / classifier P(cell)",
            data=np.asarray(cell_prob, dtype=np.float32).tolist(),
        )
    # per-ROI source tag, if present
    sources = [str(s.get("_source", "")) for s in stat]
    if any(sources):
        plane_seg.add_column(
            name="roi_source",
            description="detector that produced the ROI",
            data=sources,
        )

    rt_region = plane_seg.create_roi_table_region(
        region=list(range(len(stat))), description="all ROIs"
    )

    # ---- fluorescence traces (RoiResponseSeries are (time, roi)) --------
    fluorescence = Fluorescence()
    ophys.add(fluorescence)
    fluorescence.create_roi_response_series(
        name="Fluorescence",
        data=np.ascontiguousarray(F.T),  # (T, N)
        rois=rt_region,
        unit="lumens",
        rate=fps,
    )
    fluorescence.create_roi_response_series(
        name="Neuropil",
        data=np.ascontiguousarray(Fneu.T),
        rois=rt_region,
        unit="lumens",
        rate=fps,
    )
    if dff is not None:
        fluorescence.create_roi_response_series(
            name="DfOverF",
            data=np.asarray(dff),  # already (T, N)
            rois=rt_region,
            unit="ratio",
            rate=fps,
        )

    # ---- deconvolved / inferred spike rate (OASIS) ----------------------
    if spks is not None:
        deconv = Fluorescence(name="Deconvolved")
        ophys.add(deconv)
        deconv.create_roi_response_series(
            name="Deconvolved",
            description="OASIS non-negative deconvolved spike estimate",
            data=np.ascontiguousarray(spks.T),  # (T, N)
            rois=rt_region,
            unit="a.u.",
            rate=fps,
        )

    # ---- optional population event windows ------------------------------
    if include_events:
        windows = _read_event_windows(plane0)
        if windows is not None:
            from pynwb.epoch import TimeIntervals

            events = TimeIntervals(
                name="events",
                description="CalLIOPE population event windows",
            )
            for start_s, end_s in windows:
                events.add_interval(
                    start_time=float(start_s), stop_time=float(end_s)
                )
            nwbfile.add_time_intervals(events)

    # ---- write ----------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and overwrite:
        os.unlink(out_path)
    with NWBHDF5IO(str(out_path), mode="w") as io:
        io.write(nwbfile)
    return out_path
