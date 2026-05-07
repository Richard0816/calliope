"""summary_writer.py - cross-recording XLSX summary file.

What this file does, in one paragraph
-------------------------------------
As the user works through the pipeline, each tab incrementally adds
or refreshes a sheet in a single Excel workbook,
``<plane0>/calliope_summary.xlsx``. Tab 3 fills in the ROI table, Tab
4 stamps the lowpass parameters, Tab 5 writes the event windows and
per-ROI onset times, Tab 6 writes cluster assignments. Sheets written
by other tabs are preserved on rewrite, so the workbook is the single
source of truth for "what did we do to this recording".

Why a workbook and not just NumPy files?
- Lab members often want to reopen the data in Excel / GraphPad to
  sanity-check it.
- The sheet layout is also the cross-recording comparison format: a
  separate analysis script can glob every recording's
  ``calliope_summary.xlsx`` and stack the EventOnsets sheets into a
  single tidy table.

Sheet conventions
-----------------
- ``Recording``: single key/value table (recording_id, plane0, prefix,
  fps, T, N, generated_at). Updated by every tab.
- ``ROIs``: one row per ROI from the Suite2p detection tab.
- ``FilterParams``: filter knobs from the Low-pass tab (key/value).
- ``EventWindows``: one row per detected population event window
  (event_id, start_s, end_s, ...).
- ``EventOnsets``: long format, one row per (event_id, roi, onset_s).
  This is the sheet most downstream analyses consume.
- ``RoiEventTimes``: wide format, one column per ROI with every
  detected onset time in seconds (NaN-padded). Includes onsets that
  fall outside any population event window.
- ``Clusters``: one row per ROI with cluster_id, cluster_color,
  threshold.

Public surface
--------------
``summary_path(plane0)`` -> resolve the workbook path.
``update_recording_meta(plane0, fps, T, N)`` -> stamp the
``Recording`` sheet.
``write_rois_sheet(plane0, ...)`` -> Tab 3 calls this on completion.
``write_filter_params_sheet(plane0, ...)`` -> Tab 4.
``write_events_sheets(plane0, event_windows, onsets_by_roi, fps)``
    -> Tab 5; produces the EventWindows + EventOnsets + RoiEventTimes
    triple in one call.
``write_clusters_sheet(plane0, ...)`` -> Tab 6.
"""

from __future__ import annotations

import datetime
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

DEFAULT_FILENAME = "calliope_summary.xlsx"


# ---------------------------------------------------------------------------
# Path + low-level sheet I/O
# ---------------------------------------------------------------------------


def summary_path(plane0, filename: str = DEFAULT_FILENAME) -> Path:
    """Resolve the workbook path for a given plane0 directory."""
    return Path(plane0) / filename


def _to_dataframe(data) -> pd.DataFrame:
    """Coerce ``data`` to a DataFrame without copying when possible."""
    """Coerce ``data`` to a pandas DataFrame.

    Handles the three idioms callers use:
        * already a DataFrame -> return as-is.
        * dict ``{column_name: sequence}`` -> wide DataFrame.
        * list of dicts (one record per dict) -> long DataFrame.
        * anything else -> defer to pandas' constructor.

    Encapsulating the conversion lets every writer below stay
    agnostic about how the caller chose to assemble its rows.
    """
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, dict):
        # dict of column_name -> sequence
        return pd.DataFrame(data)
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return pd.DataFrame(data)
    return pd.DataFrame(data)


def write_sheet(
    plane0,
    sheet_name: str,
    data,
    filename: str = DEFAULT_FILENAME,
) -> Path:
    """Replace (or create) one sheet of the workbook in place.

    The interesting bit
    -------------------
    pandas' ``ExcelWriter`` with ``mode="a"`` + ``if_sheet_exists=
    "replace"`` lets us write **one** sheet without trampling other
    sheets that another tab wrote earlier. So Tab 4 stamping the
    FilterParams sheet doesn't erase the ROIs sheet that Tab 3
    wrote.

    Caller is responsible for retrying on ``PermissionError`` if
    Excel has the file open. We surface the OS error rather than
    swallowing it -- silent failure here would corrupt the
    accumulated state of the workbook.
    """
    path = summary_path(plane0, filename)
    df = _to_dataframe(data)

    if path.exists():
        # Append-mode writer keeps the existing workbook contents
        # and only touches the named sheet.
        with pd.ExcelWriter(
            path, engine="openpyxl", mode="a", if_sheet_exists="replace"
        ) as w:
            df.to_excel(w, sheet_name=sheet_name, index=False)
    else:
        # Fresh workbook -- create parents and write in mode="w".
        path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(path, engine="openpyxl", mode="w") as w:
            df.to_excel(w, sheet_name=sheet_name, index=False)
    return path


def write_sheets(plane0, sheets: dict, filename: str = DEFAULT_FILENAME) -> Path:
    """Write multiple sheets in a single workbook open. ``sheets`` maps
    sheet_name -> DataFrame/dict/records."""
    path = summary_path(plane0, filename)
    if path.exists():
        with pd.ExcelWriter(
            path, engine="openpyxl", mode="a", if_sheet_exists="replace"
        ) as w:
            for name, data in sheets.items():
                _to_dataframe(data).to_excel(w, sheet_name=name, index=False)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(path, engine="openpyxl", mode="w") as w:
            for name, data in sheets.items():
                _to_dataframe(data).to_excel(w, sheet_name=name, index=False)
    return path


# ---------------------------------------------------------------------------
# Recording metadata sheet
# ---------------------------------------------------------------------------


def update_recording_meta(
    plane0,
    *,
    recording_id: Optional[str] = None,
    prefix: Optional[str] = None,
    fps: Optional[float] = None,
    T: Optional[int] = None,
    N: Optional[int] = None,
    extra: Optional[dict] = None,
    filename: str = DEFAULT_FILENAME,
) -> Path:
    """Write/update the "Recording" sheet (key/value pairs).

    Every tab calls this on completion -- the sheet acts as a single-
    row description of the recording and a "last touched" timestamp.
    Subsequent calls overwrite the sheet rather than append, so the
    Recording sheet always reflects the most recent state.

    The ``*`` in the parameter list separates positional from
    keyword-only arguments: callers must specify ``fps=...``,
    ``T=...`` etc. by name. Pythonic way to enforce explicit naming
    on a long call site.
    """
    plane0 = Path(plane0)
    # If the caller didn't pass a recording id, infer it from the
    # path (``F:/data/Cx/2024-07-01_00018/suite2p/plane0`` ->
    # ``2024-07-01_00018``).
    rid = recording_id or _infer_recording_id(plane0)
    # List of (key, value) tuples becomes the two-column "Recording"
    # sheet. Empty strings (rather than NaN) when a value wasn't
    # supplied so the cells render cleanly in Excel.
    rows = [
        ("recording_id", rid),
        ("plane0", str(plane0)),
        ("prefix", prefix if prefix is not None else ""),
        ("fps", float(fps) if fps is not None else ""),
        ("T", int(T) if T is not None else ""),
        ("N", int(N) if N is not None else ""),
        ("generated_at",
         datetime.datetime.now().isoformat(timespec="seconds")),
    ]
    if extra:
        for k, v in extra.items():
            rows.append((str(k), v))
    df = pd.DataFrame(rows, columns=["key", "value"])
    return write_sheet(plane0, "Recording", df, filename=filename)


def _infer_recording_id(plane0: Path) -> str:
    """Recover the human-readable recording id from a plane0 path.

    The pipeline used to assume ``<recording_id>/suite2p/plane0/...``
    so two ``.parent`` calls reached the recording root. The newer
    ``sparse_plus_cellpose`` layout writes plane0 deeper:
    ``<recording_id>/detection/final/suite2p/plane0/...`` -- in that
    layout, the old "two parents up" walk lands on ``final`` instead
    of the recording id.

    Defer to ``utils.infer_recording_id`` (in ``core/utils.py``) for
    the robust resolution: it tries the canonical date-pattern regex
    first, falls back to skipping known intermediate folder names
    (``suite2p`` / ``plane0`` / ``detection`` / ``final`` /
    ``_shared_reg`` / etc.), and finally to the legacy two-parents-up
    behaviour.
    """
    from .utils import infer_recording_id
    return infer_recording_id(plane0)


# ---------------------------------------------------------------------------
# Per-tab convenience writers
# ---------------------------------------------------------------------------


def write_rois_sheet(
    plane0,
    stat_list,
    *,
    p_cell: Optional[Sequence[float]] = None,
    predicted_mask: Optional[Sequence[bool]] = None,
    iscell: Optional[Sequence[bool]] = None,
    filename: str = DEFAULT_FILENAME,
) -> Path:
    """Write a "ROIs" sheet with one row per Suite2p ROI.

    ``stat_list`` is the suite2p stat.npy structure (list of dicts with
    ``xpix``/``ypix``/``med`` keys).
    """
    rows = []
    for i, s in enumerate(stat_list):
        med = s.get("med", None) if isinstance(s, dict) else None
        if med is None:
            xpix = np.asarray(s["xpix"])
            ypix = np.asarray(s["ypix"])
            x_med = float(np.median(xpix)) if xpix.size else float("nan")
            y_med = float(np.median(ypix)) if ypix.size else float("nan")
        else:
            y_med, x_med = float(med[0]), float(med[1])
        area = int(np.asarray(s["xpix"]).size) if "xpix" in s else 0
        rows.append({
            "roi": i,
            "x_med": x_med,
            "y_med": y_med,
            "area_px": area,
            "p_cell": (float(p_cell[i]) if p_cell is not None
                       and i < len(p_cell) else ""),
            "predicted_cell": (bool(predicted_mask[i])
                               if predicted_mask is not None
                               and i < len(predicted_mask) else ""),
            "iscell": (bool(iscell[i]) if iscell is not None
                       and i < len(iscell) else ""),
        })
    return write_sheet(plane0, "ROIs", rows, filename=filename)


def write_filter_params_sheet(
    plane0,
    params: dict,
    filename: str = DEFAULT_FILENAME,
) -> Path:
    """Write the Low-pass tab parameters as a key/value sheet."""
    rows = [{"key": str(k), "value": v} for k, v in params.items()]
    return write_sheet(plane0, "FilterParams", rows, filename=filename)


def write_events_sheets(
    plane0,
    event_windows,  # iterable of (start_s, end_s) or (start, end) frames
    onsets_by_roi,  # list[np.ndarray] of onset times in seconds, len = N
    *,
    fps: Optional[float] = None,
    in_seconds: bool = True,
    filename: str = DEFAULT_FILENAME,
) -> Path:
    """Write the "EventWindows", "EventOnsets" and "RoiEventTimes" sheets.

    EventWindows:  event_id, start_s, end_s, duration_s, n_active_rois
    EventOnsets:   event_id, roi, onset_s, t_relative_s  (only onsets that
                   fall inside an event window).
    RoiEventTimes: wide table, one column ``roi_<i>`` per ROI listing every
                   detected onset time in seconds (NaN-padded so all
                   columns share the same length). Includes onsets that
                   fall outside any event window.
    """
    if event_windows is None:
        event_windows = []
    if not in_seconds:
        if fps is None:
            raise ValueError(
                "fps required when event_windows are in frame indices.")
        event_windows = [
            (float(s) / fps, float(e) / fps) for s, e in event_windows
        ]
    else:
        event_windows = [(float(s), float(e)) for s, e in event_windows]

    onsets_by_roi = [np.asarray(x, dtype=float) for x in onsets_by_roi]

    win_rows = []
    onset_rows = []
    for ev_id, (s, e) in enumerate(event_windows):
        active_count = 0
        for roi, onsets in enumerate(onsets_by_roi):
            if onsets.size == 0:
                continue
            mask = (onsets >= s) & (onsets <= e)
            if not mask.any():
                continue
            active_count += 1
            for t in onsets[mask]:
                onset_rows.append({
                    "event_id": ev_id,
                    "roi": roi,
                    "onset_s": float(t),
                    "t_relative_s": float(t) - s,
                })
        win_rows.append({
            "event_id": ev_id,
            "start_s": s,
            "end_s": e,
            "duration_s": e - s,
            "n_active_rois": active_count,
        })

    # Wide per-ROI sheet: every onset for every ROI, NaN-padded to a
    # common length so the columns line up. Useful for downstream scripts
    # that want all events per ROI without filtering by population window.
    max_len = max((arr.size for arr in onsets_by_roi), default=0)
    roi_event_times: dict[str, np.ndarray] = {}
    for roi, arr in enumerate(onsets_by_roi):
        padded = np.full(max_len, np.nan, dtype=float)
        if arr.size:
            padded[:arr.size] = arr
        roi_event_times[f"roi_{roi}"] = padded
    roi_event_times_df = pd.DataFrame(roi_event_times)

    return write_sheets(
        plane0,
        {
            "EventWindows": win_rows,
            "EventOnsets": onset_rows,
            "RoiEventTimes": roi_event_times_df,
        },
        filename=filename,
    )


def write_clusters_sheet(
    plane0,
    cluster_labels,  # length N, fcluster() ints
    *,
    cluster_colors: Optional[dict] = None,  # cluster_id -> hex
    threshold: Optional[float] = None,
    method: str = "average",
    metric: str = "correlation",
    roi_indices: Optional[Sequence[int]] = None,
    filename: str = DEFAULT_FILENAME,
) -> Path:
    """Write a "Clusters" sheet.

    Schema: roi, cluster_id, cluster_color, threshold_used, method, metric

    ``roi_indices`` (optional, len N) is written into the ``roi`` column.
    Pass the Suite2p ROI indices when the cluster_labels were computed on
    a filter-restricted view of dF/F so the column lines up with the
    ``roi`` column in the ROIs sheet. If omitted, the column falls back
    to ``0..N-1``.
    """
    cluster_labels = np.asarray(cluster_labels)
    if roi_indices is None:
        roi_indices = range(len(cluster_labels))
    elif len(roi_indices) != len(cluster_labels):
        raise ValueError(
            f"roi_indices length {len(roi_indices)} does not match "
            f"cluster_labels length {len(cluster_labels)}.")
    rows = []
    for roi, lbl in zip(roi_indices, cluster_labels):
        rows.append({
            "roi": int(roi),
            "cluster_id": int(lbl),
            "cluster_color": (cluster_colors.get(int(lbl), "")
                              if cluster_colors else ""),
            "threshold_used": (float(threshold) if threshold is not None
                               else ""),
            "method": method,
            "metric": metric,
        })
    return write_sheet(plane0, "Clusters", rows, filename=filename)
