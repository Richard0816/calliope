"""Trimmed copy of ``utils.py`` for the calliope GUI.

This file is the workhorse of the analysis pipeline. Almost every
quantitative step the GUI exposes lives here, plus a handful of small
file/IO helpers. The functions are deliberately pure NumPy / SciPy
with no GUI dependencies, so the same code can be re-used from
notebooks or batch scripts.

Reader's map (function -> what it does in plain English)
--------------------------------------------------------
Per-trace signal processing
    lowpass_causal_1d
        Smooth a 1-D trace with a Butterworth low-pass that runs
        forward only (causal -- no peeking at the future). Used by
        Tab 4 to clean up shot noise before event detection.
    sg_first_derivative_1d
        Estimate a 1-D trace's first time-derivative with a
        Savitzky-Golay polynomial fit. Lets us threshold on
        "rate of brightening" rather than absolute brightness.
    mad_z
        Robust z-score using median + MAD (instead of mean + sd).
        Less sensitive to outliers than a regular z-score.
    hysteresis_onsets
        Two-threshold (Schmitt-trigger style) onset detector. A cell
        "starts firing" when its z crosses ``z_hi`` and only stops
        once it falls below ``z_lo``. Returns the list of onset frame
        indices.

Painting per-ROI values onto the FOV
    paint_spatial
        Take an array of "one number per ROI" and a list of Suite2p
        ``stat`` entries (which carry each ROI's pixel mask) and draw
        the values onto an Ly x Lx image. Reused by every spatial
        figure in the GUI.

Population events (Tab 5 + Tab 8)
    EventDetectionParams
        Dataclass bundling every knob the density-based detector
        accepts. Fields are documented inline below.
    detect_event_windows
        The main entry point: given per-ROI onset times, produce
        event windows ``(t0, t1)`` plus an active-mask matrix
        ``A`` and a per-ROI first-onset matrix ``first_time``.
    plot_event_detection / shade_event_windows
        Diagnostic-plot helpers used by Tab 5.

Suite2p / cellfilter helpers
    s2p_infer_orientation, s2p_load_raw, s2p_open_memmaps
        Read Suite2p outputs (``F.npy`` / ``Fneu.npy`` / dF/F memmaps)
        with shape inference so the rest of the code never has to
        worry about ``(T, N)`` vs ``(N, T)`` orientation.
    robust_df_over_f_1d, first_n_min_df_over_f_1d
        Two dF/F baselines: a rolling 10th-percentile window (for
        long recordings with slow drift) or the mean of the first
        N minutes (for short, stable recordings).
    change_batch_according_to_free_ram /
    change_nbinned_according_to_free_ram
        Auto-tune Suite2p batch sizes based on free RAM and the
        movie's per-frame footprint, so we don't OOM on big slices.

Recording metadata
    get_fps_from_notes
        Locate the lab's ``notes.csv`` and read the frame rate for a
        given recording. Falls back to a default if missing.
    aav_cleanup_and_dictionary_lookup, get_row_number_csv_module,
    file_name_to_aav_to_dictionary_lookup
        Cross-reference a recording filename against a CSV of AAV
        metadata.
"""

# Type-hint forward-reference shim. See pipeline_gui.py for full
# explanation.
from __future__ import annotations

# --- standard library ---
import os                  # OS utilities: path joining, listing, env vars
import re                  # Regex utilities for parsing filenames

# --- third-party scientific stack ---
# NumPy is the array library that underpins almost everything below.
# Roughly the equivalent of R's base + matrix arithmetic, but a
# separate package rather than built-in. ``np`` is the conventional
# alias.
import numpy as np
# pandas: DataFrame / Series, used here for reading the notes.csv
# metadata file. The "df.iloc[i]" / "df.loc[name]" idiom is
# pandas-specific row indexing.
import pandas as pd
# psutil exposes system info (free RAM, CPU count). We only use it
# for the auto-batch-size logic at the bottom of the file.
import psutil
# SciPy is broken into many sub-modules; we import only what we need.
from scipy.ndimage import gaussian_filter1d, percentile_filter
from scipy.signal import butter, find_peaks, savgol_filter, sosfilt
from scipy.special import erfinv  # inverse error function for Gaussian quantile
# pathlib provides ``Path`` -- a path object. Use ``Path("a") / "b"``
# instead of ``os.path.join``.
from pathlib import Path
# Type-hint utilities. ``Optional[X]`` == ``X | None``;
# ``Sequence[X]`` is "any list-like of Xs"; ``Union[A, B]`` == ``A | B``.
from typing import Optional, Sequence, Union
# Dataclasses give us cheap "record" types -- a class whose
# ``__init__`` is auto-generated from the field annotations. Roughly
# the same idea as R's ``setClass`` for S4 records, but lighter.
from dataclasses import dataclass

# matplotlib is imported here only because two helper functions below
# (plot_event_detection, shade_event_windows) take an existing Axes
# and draw onto it. We don't open any windows from this file.
import matplotlib.pyplot as plt


def _load_keep_mask(root: Path, n_total: int) -> np.ndarray:
    """Load a boolean ROI keep-mask of length ``n_total`` from a Suite2p
    plane0 folder. Source preference, in order:

    1. ``predicted_cell_mask.npy`` -- canonical output from the cell-filter
       prediction step (run_cellfilter / cellfilter.predict_recording).
    2. ``r0p7_cell_mask_bool.npy`` -- legacy duplicate written by the
       run_full_pipeline.make_filtered_memmaps step, kept for backward
       compatibility with older recordings that don't have (1).
    3. ``iscell.npy`` column 0 -- suite2p's built-in classifier; fallback
       when no cell-filter prediction is present.

    Raises FileNotFoundError when none of the above exist."""
    pred_path = root / "predicted_cell_mask.npy"
    if pred_path.exists():
        return np.load(pred_path, allow_pickle=False).astype(bool).ravel()
    legacy_path = root / "r0p7_cell_mask_bool.npy"
    if legacy_path.exists():
        return np.load(legacy_path, allow_pickle=False).astype(bool).ravel()
    iscell_path = root / "iscell.npy"
    if iscell_path.exists():
        ic = np.load(iscell_path, allow_pickle=False)
        return (ic[:, 0] > 0 if ic.ndim == 2 else ic > 0).astype(bool).ravel()
    raise FileNotFoundError(
        f"No filter mask found in {root}. Expected one of: "
        f"predicted_cell_mask.npy, r0p7_cell_mask_bool.npy, iscell.npy."
    )


def lowpass_causal_1d(x, fps, cutoff_hz=5.0, order=2, zi=None, sos=None):
    """Causal SOS low-pass for a 1-D trace.

    "Causal" means the filter only uses past samples to compute each
    output -- no peeking forward. That matters for event detection
    because we don't want a filter to introduce artificial pre-event
    activity.

    "SOS" = second-order sections, the numerically stable
    representation of an IIR filter. SciPy's ``sosfilt`` runs the
    filter in one forward pass.

    Parameters
    ----------
    x : 1-D array
        Trace to filter (e.g. one cell's dF/F).
    fps : float
        Frame rate in Hz; used to convert ``cutoff_hz`` to a
        normalised frequency.
    cutoff_hz : float
        -3 dB cutoff in Hz. Clamped to ``[1e-4, 0.95 * Nyquist]`` so
        the design step never sees impossible numbers.
    order : int
        Butterworth order; default 2 = gentle roll-off.
    zi : array, optional
        Filter state from a previous call. Pass it back in to
        continue filtering a chunked stream.
    sos : array, optional
        Pre-built filter coefficients; if None, designed on the fly.

    Returns
    -------
    y : ndarray (float32)
        Filtered trace, same shape as ``x``.
    zf : ndarray
        Final filter state -- pass back as ``zi`` next call.
    sos : ndarray
        Filter coefficients -- pass back as ``sos`` next call to
        skip redesign.
    """
    # ``np.asarray`` either returns the input unchanged (if it's
    # already an ndarray of the right dtype) or copies it to one --
    # cheap idempotent way to coerce inputs.
    x = np.asarray(x, dtype=np.float32)
    n = x.size
    if n < 3:
        # Filter design needs at least a few samples; bail safely.
        return x.copy(), zi, sos

    # Nyquist = max representable frequency given the sample rate.
    nyq = fps / 2.0
    # Clamp so we never design a filter that's outside (0, nyq).
    cutoff = min(max(1e-4, cutoff_hz), 0.95 * nyq)
    if sos is None:
        # ``butter(..., output='sos')`` returns the filter as a stack
        # of biquad sections.
        sos = butter(order, cutoff / nyq, btype='low', output='sos')

    if zi is None:
        # Initialise the filter state with the first sample so the
        # filter doesn't "start from zero" and produce a giant
        # transient.
        zi = np.zeros((sos.shape[0], 2), dtype=np.float32)
        zi[:, 0] = x[0]
        zi[:, 1] = x[0]

    y, zf = sosfilt(sos, x, zi=zi)
    # ``y.astype(np.float32)`` is essentially a copy with type
    # conversion -- needed because sosfilt returns float64.
    return y.astype(np.float32), zf.astype(np.float32), sos


def sg_first_derivative_1d(x, fps, win_ms=333, poly=3):
    """Savitzky-Golay smoothed first derivative on a 1-D trace.

    Idea: fit a degree-``poly`` polynomial to a sliding window
    centred on each sample, then read off the analytic derivative of
    the fit at the centre point. Less noisy than naively diffing
    consecutive samples.

    Used by Tab 4 to produce the per-ROI ``_dt`` memmap that Tab 5
    thresholds for onset detection.
    """
    x = np.asarray(x, dtype=np.float32)
    n = x.size
    # Convert window length from milliseconds to samples; SG requires
    # an odd-length window. ``| 1`` sets the lowest bit, which forces
    # the integer to odd (e.g. 6 -> 7, 7 -> 7).
    win = max(3, int((win_ms / 1000.0) * fps) | 1)
    if win >= n:
        # Trace is shorter than the requested window -- shrink to the
        # largest odd window that fits.
        win = max(3, (n - (1 - n % 2)))
    if win < 3 or n < 3:
        # Too short for SG; fall back to a one-sample finite-
        # difference (multiplied by fps to convert to per-second).
        g = np.empty_like(x)
        g[0] = 0.0
        g[1:] = (x[1:] - x[:-1]) * fps
        return g
    # ``deriv=1`` => first derivative; ``delta=1/fps`` => sample
    # spacing in seconds, so the result is in units of (signal/sec).
    return savgol_filter(x, window_length=win, polyorder=poly, deriv=1, delta=1.0 / fps).astype(np.float32)


def mad_z(x):
    """Robust z-score using median + MAD instead of mean + sd.

    For Gaussian-distributed data, ``MAD * 1.4826`` is an unbiased
    estimate of the standard deviation. Because both median and MAD
    are robust to outliers, occasional huge transients in the data
    don't blow up the z-score scale -- which would otherwise cause
    real events to fall *below* a fixed threshold.

    Returns
    -------
    z : ndarray
        ``(x - median) / (1.4826 * mad)``.
    med : float
        The median of ``x`` (handy for inverse transforms).
    mad : float
        The MAD of ``x``.
    """
    med = np.median(x)
    # ``+ 1e-12`` is a tiny epsilon to avoid division by zero on
    # perfectly flat traces.
    mad = np.median(np.abs(x - med)) + 1e-12
    return (x - med) / (1.4826 * mad), med, mad


def hysteresis_onsets(z, z_hi, z_lo, fps, min_sep_s=0.0):
    """Two-threshold (Schmitt-trigger) onset detector.

    A sample marks an onset when ``z`` crosses *up* through ``z_hi``;
    we then stay "active" until ``z`` falls below ``z_lo``. Using two
    thresholds (with ``z_hi > z_lo``) prevents flickering at the
    boundary that a single-threshold detector would produce.

    Parameters
    ----------
    z : 1-D array
        The robust-z trace from ``mad_z(...)[0]`` of the SG derivative.
    z_hi, z_lo : float
        Enter / exit thresholds in robust-z units.
    fps : float
        Frame rate; only used for the ``min_sep_s`` merge step.
    min_sep_s : float
        If > 0, drop onsets that fall within this many seconds of
        the previous accepted onset.

    Returns
    -------
    onsets : ndarray of int
        Frame indices where each event began.
    """
    above_hi = z >= z_hi
    onsets = []
    active = False
    # Single-pass linear scan. ``range(z.size)`` is Python's lazy
    # integer-iterator equivalent of R's ``seq_along(z)``.
    for i in range(z.size):
        if not active and above_hi[i]:
            active = True
            onsets.append(i)
        elif active and z[i] <= z_lo:
            active = False
    if not onsets:
        # Important to return an empty *array*, not an empty list,
        # because callers slice / index it as numeric.
        return np.array([], dtype=int)
    onsets = np.array(onsets, dtype=int)
    if min_sep_s > 0:
        # Merge: keep only onsets that are at least ``min_sep`` frames
        # past the previous kept onset.
        min_sep = int(min_sep_s * fps)
        merged = [onsets[0]]
        for k in onsets[1:]:
            if k - merged[-1] >= min_sep:
                merged.append(k)
        onsets = np.asarray(merged, dtype=int)
    return onsets


def paint_spatial(values_per_roi, stat_list, Ly, Lx):
    """Paint per-ROI scalar values onto the imaging plane.

    Each Suite2p ROI carries a list of pixel positions (``ypix`` /
    ``xpix``) and a per-pixel weight ``lam`` -- the soft cell mask
    saying "this pixel is 80% cell, this one is 30% cell". To turn a
    "one number per ROI" array into a ``(Ly, Lx)`` image we paint
    every ROI's value onto its pixels weighted by ``lam``, then
    normalise pixel-wise by the accumulated weight so overlapping
    ROIs blend cleanly.

    Parameters
    ----------
    values_per_roi : array of length n_rois
        One scalar per ROI (event rate, mean dF/F, rank, etc.). NaN
        entries are still assigned but propagate through the
        weighted sum.
    stat_list : list of dict
        Suite2p ``stat`` entries; each must have ``ypix``, ``xpix``,
        ``lam`` arrays.
    Ly, Lx : int
        Imaging plane height and width in pixels.

    Returns
    -------
    img : (Ly, Lx) float32 ndarray
        Painted image. Pixels touched by no ROI stay at 0.
    """
    # Two parallel image-shaped accumulators: one for the weighted
    # value sum, one for the weight sum so we can normalise.
    img = np.zeros((Ly, Lx), dtype=np.float32)
    w = np.zeros((Ly, Lx), dtype=np.float32)

    for j, s in enumerate(stat_list):
        v = values_per_roi[j]
        ypix = s['ypix']
        xpix = s['xpix']
        lam = s['lam'].astype(np.float32)

        # Fancy indexing: ``img[ypix, xpix]`` selects the pixels that
        # belong to this ROI; ``+=`` adds the weighted value into
        # those pixels in-place. NumPy automatically broadcasts ``v``
        # across the ROI's pixel mask.
        img[ypix, xpix] += v * lam
        w[ypix, xpix] += lam

    # Boolean mask of "pixels that received any weight". Avoids a
    # divide-by-zero on the background.
    m = w > 0
    img[m] /= w[m]

    return img


# Pre-compile a regex matching folder names like ``2024-07-01_00018``
# **and** ``+``-joined groups like ``2024-07-01_00018+2024-07-01_00019``
# (preprocess_tiff_group merges multiple TIFFs into one recording named
# by joining the source stems with ``+``). The ``(?:...)`` wraps a
# non-capturing optional repetition: any number of ``+YYYY-MM-DD_NNN``
# pieces tacked onto the end.
RECORDING_DIR_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}_\d+(?:\+\d{4}-\d{2}-\d{2}_\d+)*"
)


def _find_recording_root(path: Path) -> Path | None:
    """Walk upward from ``path`` until we hit a folder named
    ``YYYY-MM-DD_#####`` (or a ``+``-joined group of those).

    Returns ``None`` if no ancestor matches.

    The recording is identified by date + serial number; this is the
    folder that holds raw data, suite2p output, etc., so most of the
    pipeline keys things off this name.
    """
    # ``[path] + list(path.parents)`` builds a list starting with the
    # path itself and walking up: ``foo/bar/baz`` -> [baz, bar, foo].
    for p in [path] + list(path.parents):
        # ``fullmatch`` requires the entire string to match the regex.
        if RECORDING_DIR_RE.fullmatch(p.name):
            return p
    return None


def find_recording_root(path) -> Path:
    """Public entry point: find the recording root for any path
    inside that recording.

    The pipeline used to assume a fixed layout
    ``<recording_id>/suite2p/plane0/...``, so callers reached the
    recording id with ``plane0.parent.parent.name``. The newer
    ``sparse_plus_cellpose`` layout writes deeper:
    ``<recording_id>/detection/final/suite2p/plane0/...``. Reaching
    "two parents up" now lands on ``final``, which is wrong.

    This helper unifies both:

    1. Walk every ancestor (including ``path`` itself) looking for a
       folder whose name matches the canonical recording-id regex
       (``YYYY-MM-DD_#####`` or ``+``-joined groups). First match
       wins -- works for both old and new layouts.
    2. **Fallback for arbitrary user-named recordings**: if step 1
       fails, look for the deepest ancestor whose name is *not*
       part of the standard suite2p chain (``suite2p``, ``plane0``,
       ``detection``, ``final``, ``shared_reg``, ``sparsery_pass``,
       ``cellpose_pass``). Returns that as the best guess.
    3. **Last resort**: legacy two-parents-up if ``path`` ends in
       ``suite2p/plane0``.
    4. **Otherwise**: return ``path`` unchanged so callers always get
       a Path back (the ``recording id`` they end up displaying may
       be wrong, but at least nothing crashes).
    """
    p = Path(path).resolve()

    # Strategy 1: canonical recording-id regex.
    hit = _find_recording_root(p)
    if hit is not None:
        return hit

    # Strategy 2: skip past well-known intermediate folder names and
    # take the next one up. Ordered from "deepest" to "shallowest"
    # so we land on the user-named recording folder.
    INTERMEDIATE_NAMES = {
        "plane0", "suite2p",                # Suite2p output structure
        "final", "detection",               # sparse_plus_cellpose layout
        "_shared_reg", "sparsery_pass",
        "cellpose_pass", "merged",          # legacy multi-pass layout
    }
    for ancestor in p.parents:
        if ancestor.name not in INTERMEDIATE_NAMES:
            return ancestor
        # Keep walking; e.g. plane0 -> suite2p -> final -> detection -> rec.

    # Strategy 3: legacy "<rec>/suite2p/plane0" assumption.
    if p.name == "plane0" and p.parent.name == "suite2p":
        return p.parent.parent

    # Last resort.
    return p


def infer_recording_id(path) -> str:
    """Return the human-readable recording id (folder name) inferred
    from any path inside the recording.

    Convenience wrapper around ``find_recording_root``: callers that
    only want the id (e.g. for filling the ``Recording`` sheet in
    ``calliope_summary.xlsx``) can do
    ``utils.infer_recording_id(plane0)`` instead of walking parents
    by hand.
    """
    return find_recording_root(path).name


def get_fps_from_notes(
        path: str,
        notes_root: str = r"F:\notes_recordings",
        default_fps: float = 15.07,
) -> float:
    """Resolve FPS for a recording, given ANY path inside that recording.

    The lab keeps notes XLSX files keyed by date in ``F:\\notes_recordings``;
    each contains a "2P settings" sheet whose rows describe one
    recording per row, with columns including ``filename`` and
    ``rate (hz)``. We search that sheet for the row matching the
    recording folder we're inside and read the frame rate.

    Safe fallback: any failure (missing notes, missing column, weird
    value) returns ``default_fps`` instead of raising.
    """
    # The whole function is wrapped in a try/except because we'd
    # rather hand the caller a sane default than crash the GUI on a
    # missing notes file.
    try:
        path = Path(path)

        rec_root = _find_recording_root(path)
        if rec_root is None:
            return default_fps

        # Folder name "2024-07-01_00018" -> date_str="2024-07-01",
        # rec_str="00018". ``split("_", 1)`` splits at most once,
        # leaving any trailing underscores in the second piece.
        date_str, rec_str = rec_root.name.split("_", 1)
        target = f"{date_str}-{rec_str}"

        notes_root = Path(notes_root)
        # Try ``YYYY-MM-DD.xlsx`` first; if not present, glob for any
        # xlsx with the date in its name (some files have suffixes).
        notes_path = notes_root / f"{date_str}.xlsx"

        if not notes_path.exists():
            candidates = sorted(notes_root.glob(f"*{date_str}*.xlsx"))
            if not candidates:
                return default_fps
            notes_path = candidates[0]

        df = pd.read_excel(notes_path, sheet_name="2P settings")
        # Strip whitespace from the column names so a stray space in
        # the spreadsheet doesn't break the lookup.
        df.columns = [str(c).strip() for c in df.columns]
        cols = {c.lower(): c for c in df.columns}

        if "filename" not in cols or "rate (hz)" not in cols:
            return default_fps

        fn_col = cols["filename"]
        rate_col = cols["rate (hz)"]

        fn = df[fn_col].astype(str).str.strip()

        hits = df.loc[fn == target]

        if hits.empty:
            hits = df.loc[
                fn.str.endswith(f"-{rec_str}", na=False) |
                (fn == rec_str)
                ]

        if hits.empty:
            return default_fps

        rate_val = hits.iloc[0][rate_col]
        if pd.isna(rate_val):
            return default_fps

        return float(rate_val)

    except Exception:
        return default_fps


# -------- EVENT DETECTION  --------
# ---------- config ----------

@dataclass
class EventDetectionParams:
    """All parameters for density-based event detection and boundary
    refinement.

    The ``@dataclass`` decorator above the class header tells Python:
    "look at the type-annotated fields below and auto-generate an
    ``__init__`` for me". So instantiating
    ``EventDetectionParams(bin_sec=0.05)`` works without us writing a
    constructor ourselves -- every field becomes a keyword argument
    with the listed default. R's nearest equivalent is
    ``setRefClass(..., fields=list(...))``.

    NOTE (short-event rework, 2026-05-01): defaults were retuned to match
    event_detection_short.py. Epileptiform events should be < ~0.5 s long;
    wide windows from the OLD merge logic are now broken up via a watershed
    split + hard duration clamp. The OLD defaults are kept commented out
    next to each field for easy revert.
    """
    # Density construction
    bin_sec: float = 0.025                 # OLD: 0.05
    smooth_sigma_bins: float = 1.5         # OLD: 2.0
    normalize_by_num_rois: bool = True

    # Peak detection on smoothed density (scipy.signal.find_peaks)
    min_prominence: float = 0.002          # OLD: 0.007
    min_width_bins: float = 1.0            # OLD: 2.0  (looser; fine bins => narrow real peaks)
    min_distance_bins: float = 4.0         # OLD: 3.0  (~100 ms at bin_sec=0.025)
    prominence_wlen_s: float = 1.0         # NEW: local-window for find_peaks prominence

    # Baseline / noise (for boundary walking)
    baseline_mode: str = "rolling"  # "rolling" or "global"
    baseline_percentile: float = 5.0
    baseline_window_s: float = 5.0         # OLD: 30.0  (short window tracks within-burst variation)
    noise_quiet_percentile: float = 40.0
    noise_mad_factor: float = 1.4826

    # Boundary walking
    end_threshold_k: float = 1.5           # OLD: 2.0
    max_walk_duration_s: float = 2.0       # NEW: cap on one-sided walk distance from peak (s).
    max_event_duration_s: float = 0.5      # OLD: 10.0  (now: hard physiological cap on FINAL event duration)

    # Overlap merging  (DISABLED in favour of watershed split below)
    merge_gap_s: float = 0.0  # OLD path: merge events closer than this

    # Drop events whose population is below this many active ROIs.
    # Default 3 = excludes 2-ROI "events" which are almost always noise
    # on epileptiform recordings where real events recruit dozens to
    # hundreds of cells. Raise to 4-5 to also exclude 3-4-ROI events.
    # Applied AFTER boundary walking + watershed split, so the
    # diagnostic arrays still describe the unfiltered detection (the
    # returned event_windows / A / first_time are sliced by the
    # population mask but ``diagnostics["prominence"]`` etc. are not).
    min_active_rois: int = 3

    # NEW: watershed split + hard duration cap  ----------------------------
    enable_watershed_split: bool = True
        # If consecutive baseline-walk windows overlap, split them at the
        # local minimum of the smoothed density between the two peaks.
    enforce_symmetric_clamp: bool = False
        # If True, force every window to exactly peak +/- max_event_duration_s/2.
        # If False, only clamp when the walked/split window exceeds the cap.

    # Gaussian-fit refinement (DISABLED for short-event mode -- the hard
    # clamp + watershed take its place. Kept here for parameter compatibility
    # with older callers; ignored when use_gaussian_boundary=False.)
    use_gaussian_boundary: bool = False    # OLD: True
    gaussian_quantile: float = 0.99
    gaussian_fit_pad_s: float = 0.5
    gaussian_min_sigma_s: float = 0.05


# ---------- top-level entry point ----------

def detect_event_windows(
        onsets_by_roi: Sequence[np.ndarray],
        T: int,
        fps: float,
        params: Optional[EventDetectionParams] = None,
        return_diagnostics: bool = False,
):
    """
    Detect population events from per-ROI onset times and return event windows
    alongside per-event activation matrices ready to substitute for the old
    (A, first_time, keep_bins) triplet.

    Parameters
    ----------
    onsets_by_roi : list of 1D arrays
        Per-ROI onset times in SECONDS (already in recording time, not frames).
        Equivalent to the output of _event_onsets_by_roi / extract_onsets_by_roi.
        Length = number of ROIs (post cell-mask filtering, if any).
    T : int
        Number of frames in the recording.
    fps : float
        Sampling rate in Hz.
    params : EventDetectionParams, optional
        Detection / boundary parameters. Uses defaults if None.
    return_diagnostics : bool
        If True, also returns a dict with baseline, noise, peak heights,
        gaussian fits, etc. -- useful for plotting and debugging.

    Returns
    -------
    event_windows : (n_events, 2) float array
        Per-event [start_s, end_s] in seconds.
    A : (N_rois, n_events) bool array
        A[i, e] = True iff ROI i had >= 1 onset inside event e's window.
    first_time : (N_rois, n_events) float array
        first_time[i, e] = time (s) of ROI i's earliest onset inside event e,
        NaN if inactive.
    diagnostics : dict (only if return_diagnostics=True)
        Contains: time_centers_s, binned_density, smoothed_density,
        baseline_trace, end_threshold_trace, baseline_noise,
        peak_s, peak_height, mu_s, sigma_s,
        boundary_source_left, boundary_source_right, prominence, duration_s.
    """
    # ``params is None`` -> use the dataclass defaults. We avoid using
    # ``params or EventDetectionParams()`` because Python treats some
    # legitimate dataclass instances as "falsy" if all fields are
    # zero, which would silently replace user input.
    if params is None:
        params = EventDetectionParams()

    # 1) Flat onset times across all ROIs -> density
    # ``duration_s`` is the total recording length. ``T`` is in
    # frames; dividing by ``fps`` converts to seconds.
    duration_s = float(T) / float(fps)
    # ``_build_density`` flattens every ROI's onset list into one
    # vector, histograms it into ``bin_sec``-wide bins, and Gaussian-
    # smooths the histogram. We get back:
    #   centers : bin centre time (s) for each histogram bin
    #   counts  : raw integer counts in each bin
    #   smooth  : Gaussian-smoothed density (per-ROI rate if normalize)
    centers, counts, smooth = _build_density(
        onsets_by_roi=onsets_by_roi,
        duration_s=duration_s,
        bin_sec=params.bin_sec,
        smooth_sigma_bins=params.smooth_sigma_bins,
        n_rois=len(onsets_by_roi),
        normalize_by_num_rois=params.normalize_by_num_rois,
    )

    # 2) Peak detection
    # NEW: pass prominence_wlen_s through so prominence is measured
    # locally (matches event_detection_short.py behaviour). We need
    # this in *bin* units, so we convert seconds -> bins via
    # ``prominence_wlen_s / bin_sec``. ``| 1`` forces an odd number
    # because scipy.signal.find_peaks's ``wlen`` parameter wants odd.
    wlen_bins_val = max(3, int(round(params.prominence_wlen_s / max(params.bin_sec, 1e-9))) | 1)
    # ``_detect_density_peaks`` returns the indices of bins where the
    # smoothed density has a local peak satisfying our prominence /
    # width / distance thresholds.
    peak_indices = _detect_density_peaks(
        smooth=smooth,
        min_prominence=params.min_prominence,
        min_width_bins=params.min_width_bins,
        min_distance_bins=params.min_distance_bins,
        wlen_bins=wlen_bins_val,
    )

    # 3) Per-event boundaries
    # For each peak, walk outward until the smoothed density falls
    # back to ``baseline + k*noise`` -- that's the event window.
    # Also handles the watershed-split, hard duration cap, and
    # optional Gaussian-fit refinement.
    boundaries = _boundaries_from_peaks(
        time_s=centers,
        smooth=smooth,
        peak_indices=peak_indices,
        params=params,
    )

    # ``np.column_stack`` glues two 1-D arrays into a 2-D array
    # column-wise: shape becomes (n_events, 2) with [start, end].
    event_windows = np.column_stack([boundaries["start_s"], boundaries["end_s"]])

    # 4) Activation matrix + first_time per event
    # For every (ROI, event) pair this returns whether the ROI fired
    # inside the event window and the time of its first onset there.
    # Used downstream by Tab 5 (heatmap sort), Tab 7 (per-event
    # cross-correlation cropping) and Tab 8 (activation-order maps).
    A, first_time = _activation_matrix_from_windows(onsets_by_roi, event_windows)

    # 4b) Drop events whose population is below ``min_active_rois``.
    # Computed from the activation matrix (sum across ROIs per event
    # gives the number of cells that fired inside that window). Both
    # the published event_windows / A / first_time AND the per-peak
    # diagnostic arrays (peak_s / peak_height / prominence / mu_s /
    # sigma_s / boundary_source_* / duration_s) are sliced so the
    # diagnostic plot's red C3 peak markers only mark surviving
    # events. Continuous-time diagnostic arrays (smoothed_density,
    # baseline_trace, etc.) stay full-length so the density plot
    # keeps its context.
    if event_windows.shape[0] > 0 and params.min_active_rois > 0:
        n_active_per_event = A.sum(axis=0).astype(int)
        keep_mask = n_active_per_event >= int(params.min_active_rois)
    else:
        keep_mask = np.ones(event_windows.shape[0], dtype=bool)
    if not keep_mask.all():
        event_windows = event_windows[keep_mask]
        A = A[:, keep_mask]
        first_time = first_time[:, keep_mask]

    if not return_diagnostics:
        return event_windows, A, first_time

    # Per-peak fields slice with keep_mask; continuous-time fields stay
    # at full resolution so the smoothed-density curve still shows
    # context around the dropped events.
    def _slice_per_peak(arr):
        if arr is None:
            return arr
        a = np.asarray(arr)
        if a.size != keep_mask.size:
            return a
        return a[keep_mask]

    diagnostics = {
        "time_centers_s": centers,
        "binned_density": counts,
        "smoothed_density": smooth,
        "baseline_trace": boundaries["baseline_trace"],
        "end_threshold_trace": boundaries["end_threshold_trace"],
        "baseline_noise": boundaries["baseline_noise"],
        "peak_s": _slice_per_peak(boundaries["peak_s"]),
        "peak_height": _slice_per_peak(boundaries["peak_height"]),
        "mu_s": _slice_per_peak(boundaries["mu_s"]),
        "sigma_s": _slice_per_peak(boundaries["sigma_s"]),
        "boundary_source_left": _slice_per_peak(
            boundaries["boundary_source_left"]),
        "boundary_source_right": _slice_per_peak(
            boundaries["boundary_source_right"]),
        "prominence": _slice_per_peak(boundaries["prominence"]),
        "duration_s": _slice_per_peak(boundaries["duration_s"]),
    }
    return event_windows, A, first_time, diagnostics


# ---------- plotting helper ----------

def plot_event_detection(
        diagnostics: dict,
        ylabel: str = "Onset density (per bin per ROI)",
        title: str = "Detected events on onset density",
        ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Diagnostic plot for ``detect_event_windows``.

    Renders five layers stacked on the same axis:
        - blue bars   : raw histogram of per-ROI onsets per bin
        - blue line   : Gaussian-smoothed density (the curve we
                         actually peak-pick on)
        - dotted grey : baseline trace (rolling 5th-percentile of
                         the smoothed density)
        - dashed black: end-threshold trace (baseline + k*noise) --
                         where the boundary walk stops
        - red markers : detected peaks
    Tab 5 calls this on the bottom panel after every Render.

    Parameters
    ----------
    diagnostics : dict
        Output of ``detect_event_windows(..., return_diagnostics=True)``.
    ylabel, title : str
        Cosmetic labels.
    ax : matplotlib Axes, optional
        Draw onto this axis. If None, builds a fresh
        ``(14, 4.5)`` figure.

    Returns
    -------
    fig : matplotlib Figure
        Either the passed-in axis's figure or the freshly created
        one. Caller is responsible for ``plt.show()`` / saving.
    """
    if ax is None:
        # ``plt.subplots`` with no rows/cols creates one Figure +
        # one Axes and returns ``(fig, ax)``.
        fig, ax = plt.subplots(figsize=(14, 4.5))
    else:
        # Caller supplied an axis (Tab 5 does this so the diagnostic
        # lands inside its existing figure). Pull its Figure handle.
        fig = ax.figure

    centers = diagnostics["time_centers_s"]
    smooth = diagnostics["smoothed_density"]
    counts = diagnostics["binned_density"]

    bin_width = float(np.median(np.diff(centers))) if centers.size > 1 else 1.0

    ax.bar(centers, counts, width=bin_width, alpha=0.35, align="center",
           color="C0", edgecolor="none", label="Binned counts")
    ax.plot(centers, smooth, linewidth=1.5, color="C0", label="Smoothed density")
    ax.plot(centers, diagnostics["baseline_trace"], color="gray", linestyle=":",
            linewidth=1.0, label="Baseline")
    ax.plot(centers, diagnostics["end_threshold_trace"], color="black",
            linestyle="--", linewidth=1.0, label="End threshold")

    peak_s = diagnostics["peak_s"]
    peak_h = diagnostics["peak_height"]
    ax.plot(peak_s, peak_h, "v", color="C3", markersize=6,
            label=f"Peaks (n={peak_s.size})")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


def shade_event_windows(ax: plt.Axes, event_windows: np.ndarray, color="C1", alpha=0.20):
    """Overlay shaded event windows onto an existing axis.

    Tab 5 calls this after ``plot_event_detection`` to highlight
    where each detected event lives. ``ax.axvspan(s, e, ...)`` draws
    a translucent vertical strip from x=s to x=e spanning the full
    y-range. Looping is the simplest way to draw N strips since
    matplotlib doesn't accept a vector of (start, end) pairs.
    """
    # ``for s, e in event_windows:`` destructures each row of the
    # (n_events, 2) array into two scalars at once -- equivalent to
    # ``for ev in event_windows: s, e = ev[0], ev[1]``.
    for s, e in event_windows:
        ax.axvspan(s, e, color=color, alpha=alpha)


# =====================================================================
# Internals
# =====================================================================

def _build_density(
        onsets_by_roi: Sequence[np.ndarray],
        duration_s: float,
        bin_sec: float,
        smooth_sigma_bins: float,
        n_rois: int,
        normalize_by_num_rois: bool,
):
    """Flatten every ROI's onset list, histogram into time bins, and
    Gaussian-smooth the histogram.

    Conceptually: each ROI casts one "vote" at each of its onset
    times; the histogram is "how many cells fired in each ``bin_sec``
    window". Smoothing turns the spiky histogram into a continuous
    density we can peak-pick on.

    Returns (centers, counts, smooth) where:
        - centers is the bin centre time (seconds) for each bin
        - counts is the raw integer count (or per-ROI rate if
          ``normalize_by_num_rois``)
        - smooth is the Gaussian-smoothed version
    """
    # ``[... for x in onsets_by_roi if len(x) > 0]`` is a list
    # comprehension that drops empty ROI onset arrays before we
    # concat them. ``np.asarray(x, dtype=np.float64)`` is a no-op if
    # the input is already a float64 array.
    nonempty = [np.asarray(x, dtype=np.float64) for x in onsets_by_roi if len(x) > 0]
    # ``np.concatenate`` on an empty list raises; guard against the
    # all-quiet recording case with an explicit empty array.
    flat = np.concatenate(nonempty) if nonempty else np.array([], dtype=np.float64)

    # Build histogram bin edges starting at 0 with ``bin_sec`` width
    # up to (and just past) the recording duration. ``np.arange``
    # excludes the upper limit so we add ``+ bin_sec`` in the call.
    edges = np.arange(0.0, duration_s + bin_sec, bin_sec, dtype=np.float64)
    if edges[-1] < duration_s:
        # Pad to make sure the last bin covers the actual recording
        # end. ``np.append`` returns a new array; the trailing bin
        # may end up shorter than ``bin_sec`` but that's fine.
        edges = np.append(edges, duration_s)

    # ``np.histogram`` returns (counts, edges); we already have edges
    # but it returns them anyway -- harmless reassignment.
    counts, edges = np.histogram(flat, bins=edges)
    # Bin centres = mean of consecutive edges. Vectorised slice
    # arithmetic, much faster than a Python loop.
    centers = 0.5 * (edges[:-1] + edges[1:])

    counts = counts.astype(np.float64)
    if normalize_by_num_rois and n_rois > 0:
        # Convert count-per-bin to "fraction of cells that fired in
        # this bin" so recordings with very different N stay
        # comparable.
        counts = counts / float(n_rois)

    # Gaussian smoothing: replaces each bin with a weighted average
    # of itself and its neighbours, weights = Gaussian with sigma in
    # *bin* units. ``mode="nearest"`` extends edge values rather than
    # zero-padding (avoids artificial dips at recording start/end).
    smooth = gaussian_filter1d(counts, sigma=smooth_sigma_bins, mode="nearest")
    return centers, counts, smooth


def _detect_density_peaks(
        smooth: np.ndarray,
        min_prominence: float,
        min_width_bins: float,
        min_distance_bins: float,
        wlen_bins: int = 0,
) -> np.ndarray:
    """Find significant peaks in the smoothed density.

    Wraps ``scipy.signal.find_peaks`` with three gates:
        - ``prominence``: minimum vertical distance from each peak to
          the lowest contour that still contains it. Filters out
          tiny ripples.
        - ``width``: minimum width at half-prominence in bins.
          Filters out single-bin spikes.
        - ``distance``: minimum bin separation between adjacent
          accepted peaks. Prevents counting the same event twice.

    ``wlen_bins`` (optional) makes the prominence calculation
    *local*: instead of "the deepest valley anywhere on the trace",
    use the deepest valley within ``wlen_bins`` of the peak. Matches
    the behaviour of ``event_detection_short.py``.

    Returns
    -------
    peaks : ndarray of int
        Bin indices of accepted peaks.
    """
    # Build the keyword-argument dict incrementally so we only set
    # ``wlen`` when it's actually requested. Python's ``**kw``
    # unpacking lets us pass it through to find_peaks as if we'd
    # written each kwarg by hand.
    kw = dict(
        prominence=min_prominence,
        width=min_width_bins,
        distance=max(1, int(round(min_distance_bins))),
    )
    if wlen_bins and wlen_bins >= 3:
        kw["wlen"] = int(wlen_bins)
    # ``find_peaks`` returns (indices, properties); we throw the
    # properties dict away with the underscore convention.
    peaks, _props = find_peaks(smooth, **kw)
    return peaks


# ---- baseline / noise ----

def _estimate_global_baseline(density: np.ndarray, baseline_percentile: float) -> np.ndarray:
    """Single-scalar baseline: the Pth percentile of the smoothed
    density, broadcast to a same-shape array.

    Used when ``params.baseline_mode == "global"``. Cheap; appropriate
    for short recordings with stable baselines.
    """
    if density.size == 0:
        return np.zeros_like(density)
    b = float(np.percentile(density, baseline_percentile))
    # ``np.full_like`` makes a new array shaped like ``density`` but
    # filled with the constant ``b``.
    return np.full_like(density, b, dtype=np.float64)


def _estimate_rolling_baseline(
        density: np.ndarray,
        fps_density: float,
        baseline_window_s: float,
        baseline_percentile: float,
) -> np.ndarray:
    """Time-varying baseline: rolling Pth percentile in a window of
    ``baseline_window_s`` seconds.

    Used when ``params.baseline_mode == "rolling"`` (the default).
    Lets the baseline track slow within-burst drift so events that
    sit on top of an elevated baseline still get accepted.

    ``fps_density`` is the *bin* sampling rate (1 / bin_sec), not the
    movie frame rate. So a 5 s window at bin_sec=0.025 is
    ``5 * 40 = 200`` bins.
    """
    if density.size == 0:
        return np.zeros_like(density)
    # Convert window length from seconds to bins; ``| 1`` forces an
    # odd number (percentile_filter wants an odd ``size``).
    win = max(3, int(round(baseline_window_s * fps_density)) | 1)
    # Don't exceed the array length, and stay odd.
    win = min(win, density.size if density.size % 2 == 1 else density.size - 1)
    if win < 3:
        # Fall back to global baseline if the window doesn't fit.
        return _estimate_global_baseline(density, baseline_percentile)
    # ``percentile_filter`` slides a window across the array and
    # replaces each centre value with the Pth percentile of the
    # window. ``mode="reflect"`` mirrors edges to avoid bias at the
    # start/end.
    return percentile_filter(
        density.astype(np.float64),
        size=win,
        percentile=baseline_percentile,
        mode="reflect",
    )


def _estimate_noise_from_quiet(
        density: np.ndarray,
        baseline_trace: np.ndarray,
        quiet_percentile: float = 40.0,
        noise_mad_factor: float = 1.4826,
) -> float:
    """Estimate the density's "noise level" using only its quiet
    bins.

    Reasoning: the bins where stuff actually happens are loud and
    should not contribute to the noise estimate. We define "quiet"
    as the bottom ``quiet_percentile`` of ``density - baseline``
    (default 40%) -- a lenient cutoff that still excludes the
    obvious peaks. Noise = MAD of the quiet residuals scaled by
    1.4826 (so it's roughly comparable to a Gaussian sigma).

    Returns a scalar -- the same noise level is used at every bin
    when computing the boundary-walk threshold.
    """
    if density.size == 0:
        return 0.0
    resid = density - baseline_trace
    cutoff = float(np.percentile(resid, quiet_percentile))
    quiet_mask = resid <= cutoff
    # Safety: if very few bins are below the cutoff (e.g. a tiny
    # recording), use the whole trace.
    if quiet_mask.sum() < 32:
        quiet_mask = np.ones_like(resid, dtype=bool)
    sample = resid[quiet_mask]
    med = np.median(sample)
    mad = np.median(np.abs(sample - med)) + 1e-12
    return float(noise_mad_factor * mad)


# ---- boundary walk ----

def _walk_boundary(
        density: np.ndarray,
        peak_idx: int,
        end_threshold_trace: np.ndarray,
        direction: int,
        max_steps: int,
) -> int:
    """Walk outward from ``peak_idx`` in ``density`` until the value
    falls below ``end_threshold_trace[i]`` or we hit ``max_steps``.

    Used to find one event-window boundary per peak. ``direction``
    is +1 (right -> end) or -1 (left -> start).

    Returns the bin index where the walk stopped (inclusive).
    """
    n = density.size
    i = peak_idx
    steps = 0
    # ``0 <= i + direction < n`` keeps us inside the array; the
    # walk also stops once the density falls under the threshold.
    while 0 <= i + direction < n and steps < max_steps:
        nxt = i + direction
        if density[nxt] <= end_threshold_trace[nxt]:
            return i
        i = nxt
        steps += 1
    return i


# ---- Gaussian fit ----

def _gaussian_z(q: float) -> float:
    """Inverse-CDF of the standard normal at quantile ``q``.

    Used by the optional Gaussian-fit boundary refinement: if you
    want to define an event window as "everywhere within the q-
    quantile of a Gaussian fit to the peak", you need the matching
    z value (e.g. q=0.99 -> z≈2.33). We compute it via the inverse
    error function rather than depending on scipy.stats.
    """
    if not (0.5 < q < 1.0):
        raise ValueError("gaussian_quantile must be in (0.5, 1.0)")
    return float(np.sqrt(2.0) * erfinv(2.0 * q - 1.0))


def _fit_gaussian_to_peak(
        time_s: np.ndarray,
        density: np.ndarray,
        baseline_trace: np.ndarray,
        peak_idx: int,
        left_idx: int,
        right_idx: int,
        pad_samples: int,
        min_sigma_s: float,
) -> tuple[float, float]:
    """Compute the weighted mean (mu) and standard deviation (sigma)
    of a peak in time, treating ``density - baseline`` (clipped to
    >= 0) as the weights.

    This is a *moments-based* Gaussian "fit" -- not a least-squares
    fit. We just compute the first two moments of the residual.
    Cheaper, robust, and good enough for the boundary-quantile
    refinement.

    Returns (mu_s, sigma_s) in seconds. ``min_sigma_s`` is a floor
    so a too-narrow peak doesn't collapse to sigma=0.
    """
    n = density.size
    # Pad the fit window outward from the boundary-walk indices so
    # the moments capture the tails of the peak.
    L = max(0, left_idx - pad_samples)
    R = min(n - 1, right_idx + pad_samples)
    if R <= L:
        # Degenerate fit window; just return the peak time + the
        # minimum sigma.
        return float(time_s[peak_idx]), float(min_sigma_s)

    # ``time_s[L:R+1]`` is array slicing -- the +1 is because Python
    # slice ends are exclusive but we want R inclusive.
    t_win = time_s[L:R + 1].astype(np.float64)
    d_win = density[L:R + 1].astype(np.float64)
    b_win = baseline_trace[L:R + 1].astype(np.float64)
    # Use density-above-baseline as a non-negative weight; clamp
    # below at 0 so noise dips don't produce negative weights.
    w = np.clip(d_win - b_win, 0.0, None)

    total = w.sum()
    if total <= 0:
        return float(time_s[peak_idx]), float(min_sigma_s)

    # Weighted mean and variance, the textbook formulas.
    mu = float((t_win * w).sum() / total)
    var = float(((t_win - mu) ** 2 * w).sum() / total)
    sigma = max(float(np.sqrt(max(var, 0.0))), float(min_sigma_s))
    return mu, sigma


# ---- peaks -> boundaries ----

def _boundaries_from_peaks(
        time_s: np.ndarray,
        smooth: np.ndarray,
        peak_indices: np.ndarray,
        params: EventDetectionParams,
) -> dict:
    """Turn the peak-bin indices into per-event window
    ``(start_s, end_s)`` pairs.

    For each peak this routine performs four passes:

    1. **Baseline walk** -- starting from the peak, walk left and
       right until the smoothed density drops below the
       ``baseline + k*noise`` threshold or until ``max_walk_duration_s``
       is exceeded.
    2. **Watershed split** (when ``params.enable_watershed_split``) --
       if two walked windows overlap, find the deepest valley
       between them and split the windows there. Stops two close
       events from getting fused.
    3. **Hard duration clamp** -- enforce
       ``max_event_duration_s``, optionally as a symmetric clamp
       around the peak.
    4. **Optional Gaussian-fit refinement** -- swap the walked
       boundary for the q-quantile of a moments-based Gaussian fit
       to the peak shape (only if ``use_gaussian_boundary``).

    Returns a dict packed with arrays for each event plus the
    full-trace baseline / noise for diagnostics.
    """
    time_s = np.asarray(time_s, dtype=np.float64)
    smooth = np.asarray(smooth, dtype=np.float64)
    peaks = np.asarray(peak_indices, dtype=np.int64)

    dt = float(np.median(np.diff(time_s))) if time_s.size > 1 else 1.0
    fps_density = 1.0 / max(dt, 1e-12)

    # baseline + noise
    if params.baseline_mode == "global":
        baseline_trace = _estimate_global_baseline(smooth, params.baseline_percentile)
    else:
        baseline_trace = _estimate_rolling_baseline(
            smooth, fps_density, params.baseline_window_s, params.baseline_percentile,
        )
    noise = _estimate_noise_from_quiet(
        smooth, baseline_trace,
        quiet_percentile=params.noise_quiet_percentile,
        noise_mad_factor=params.noise_mad_factor,
    )
    end_threshold_trace = baseline_trace + params.end_threshold_k * noise

    empty_f = np.array([], dtype=np.float64)
    empty_o = np.array([], dtype=object)
    if peaks.size == 0:
        return dict(
            start_s=empty_f, peak_s=empty_f, end_s=empty_f,
            peak_height=empty_f, prominence=empty_f, duration_s=empty_f,
            mu_s=empty_f, sigma_s=empty_f,
            boundary_source_left=empty_o, boundary_source_right=empty_o,
            baseline_trace=baseline_trace, end_threshold_trace=end_threshold_trace,
            baseline_noise=noise,
        )

    peaks = np.sort(peaks)

    # baseline walk
    # NEW: walk distance is now controlled by max_walk_duration_s (the
    # "search radius" for boundaries). The final event duration is then
    # tightened by the watershed split + hard cap below.
    walk_steps = max(1, int(round(params.max_walk_duration_s / dt)))
    # OLD: max_steps = max(1, int(round(params.max_event_duration_s / dt)))
    start_idx = np.empty(peaks.size, dtype=np.int64)
    end_idx = np.empty(peaks.size, dtype=np.int64)
    for i, p in enumerate(peaks):
        start_idx[i] = _walk_boundary(smooth, int(p), end_threshold_trace, -1, walk_steps)
        end_idx[i] = _walk_boundary(smooth, int(p), end_threshold_trace, +1, walk_steps)

    # ------------------------------------------------------------------
    # OLD: merge overlapping / touching events (DISABLED 2026-05-01).
    # Replaced by watershed split below so closely-spaced epileptiform
    # events stay separate instead of being fused into one wide window.
    # ------------------------------------------------------------------
    # merge_gap_samples = max(0, int(round(params.merge_gap_s / dt)))
    # keep = np.ones(peaks.size, dtype=bool)
    # for i in range(1, peaks.size):
    #     j = i - 1
    #     while j >= 0 and not keep[j]:
    #         j -= 1
    #     if j < 0:
    #         continue
    #     if start_idx[i] - end_idx[j] <= merge_gap_samples:
    #         if smooth[peaks[i]] > smooth[peaks[j]]:
    #             peaks[j] = peaks[i]
    #         end_idx[j] = max(end_idx[j], end_idx[i])
    #         start_idx[j] = min(start_idx[j], start_idx[i])
    #         keep[i] = False
    # peaks = peaks[keep]
    # start_idx = start_idx[keep]
    # end_idx = end_idx[keep]

    # NEW: watershed split -- where consecutive walked windows overlap,
    # cut at the local minimum (valley) of `smooth` between the two peaks.
    if params.enable_watershed_split and peaks.size > 1:
        for i in range(1, peaks.size):
            j = i - 1
            if start_idx[i] <= end_idx[j]:
                seg_lo = int(peaks[j])
                seg_hi = int(peaks[i])
                if seg_hi - seg_lo >= 2:
                    valley = seg_lo + int(np.argmin(smooth[seg_lo:seg_hi + 1]))
                else:
                    valley = (seg_lo + seg_hi) // 2
                end_idx[j] = valley
                start_idx[i] = valley + 1 if valley + 1 <= seg_hi else valley
                start_idx[i] = min(start_idx[i], int(peaks[i]))

    n_events = peaks.size

    # Gaussian-fit refinement (DISABLED by default in short-event mode --
    # use_gaussian_boundary now defaults to False. Block kept for backward
    # compatibility when callers explicitly enable it.)
    mu_s = np.full(n_events, np.nan, dtype=np.float64)
    sigma_s = np.full(n_events, np.nan, dtype=np.float64)
    src_left = np.empty(n_events, dtype=object)
    src_right = np.empty(n_events, dtype=object)

    base_start_s = time_s[start_idx].astype(np.float64)
    base_end_s = time_s[end_idx].astype(np.float64)

    if params.use_gaussian_boundary and n_events > 0:
        z = _gaussian_z(params.gaussian_quantile)
        pad_samples = max(0, int(round(params.gaussian_fit_pad_s / dt)))
        start_s_out = np.empty(n_events, dtype=np.float64)
        end_s_out = np.empty(n_events, dtype=np.float64)
        for i in range(n_events):
            mu, sig = _fit_gaussian_to_peak(
                time_s=time_s, density=smooth, baseline_trace=baseline_trace,
                peak_idx=int(peaks[i]), left_idx=int(start_idx[i]), right_idx=int(end_idx[i]),
                pad_samples=pad_samples, min_sigma_s=params.gaussian_min_sigma_s,
            )
            mu_s[i] = mu
            sigma_s[i] = sig
            g_start = mu - z * sig
            g_end = mu + z * sig
            # Whichever comes first
            chosen_start = max(g_start, base_start_s[i])
            chosen_end = min(g_end, base_end_s[i])
            # Never cross the peak
            peak_t = float(time_s[peaks[i]])
            chosen_start = min(chosen_start, peak_t)
            chosen_end = max(chosen_end, peak_t)
            start_s_out[i] = chosen_start
            end_s_out[i] = chosen_end
            src_left[i] = "gaussian" if g_start >= base_start_s[i] else "baseline"
            src_right[i] = "gaussian" if g_end <= base_end_s[i] else "baseline"
    else:
        start_s_out = base_start_s.copy()
        end_s_out = base_end_s.copy()
        src_left[:] = "baseline"
        src_right[:] = "baseline"

    # NEW: HARD DURATION CAP -- physiology says a single epileptiform event
    # cannot be more than ~max_event_duration_s. Clamp every window to
    # peak +/- max/2 (only tighten -- never widen).
    if n_events > 0:
        cap = float(params.max_event_duration_s)
        half_cap = cap / 2.0
        for i in range(n_events):
            peak_t = float(time_s[peaks[i]])
            cur_dur = end_s_out[i] - start_s_out[i]
            if params.enforce_symmetric_clamp or cur_dur > cap:
                new_left = peak_t - half_cap
                new_right = peak_t + half_cap
                if not params.enforce_symmetric_clamp:
                    # Only tighten the window, never widen it
                    new_left = max(new_left, start_s_out[i])
                    new_right = min(new_right, end_s_out[i])
                # Defensive: never cross the peak
                start_s_out[i] = min(new_left, peak_t)
                end_s_out[i] = max(new_right, peak_t)

    peak_height = smooth[peaks]
    prominences_final = peak_height - end_threshold_trace[peaks]

    return dict(
        start_s=start_s_out,
        peak_s=time_s[peaks].astype(np.float64),
        end_s=end_s_out,
        peak_height=peak_height,
        prominence=prominences_final,
        duration_s=end_s_out - start_s_out,
        mu_s=mu_s, sigma_s=sigma_s,
        boundary_source_left=src_left,
        boundary_source_right=src_right,
        baseline_trace=baseline_trace,
        end_threshold_trace=end_threshold_trace,
        baseline_noise=noise,
    )


# ---- activation matrix within event windows ----

def _activation_matrix_from_windows(
        onsets_by_roi: Sequence[np.ndarray],
        event_windows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """For every (ROI, event) pair, did the ROI have any onset
    inside that event's time window? And if so, when was its
    earliest one?

    This routine is the bridge between the per-ROI onset detector
    and downstream per-event analyses (Tabs 7/8 both consume the
    matrices it produces).

    Returns
    -------
    A : (N, E) bool
        ``A[i, e]`` is True iff ROI i had at least one onset inside
        event e's ``[start_s, end_s]`` window.
    first_time : (N, E) float
        ``first_time[i, e]`` is the earliest onset time (s) of ROI
        i inside event e, or NaN if the ROI didn't fire there.
    """
    N = len(onsets_by_roi)
    E = event_windows.shape[0]
    # ``np.zeros((N, E), dtype=bool)`` allocates an N x E False-
    # filled boolean matrix -- one of the cheapest array allocations
    # NumPy can do.
    A = np.zeros((N, E), dtype=bool)
    # ``np.full(shape, value)`` is the constant-fill version of
    # ``zeros`` / ``ones``.
    first_time = np.full((N, E), np.nan, dtype=np.float64)

    if E == 0:
        return A, first_time

    # Slice the event windows array into start / end vectors.
    starts = event_windows[:, 0].astype(np.float64)
    ends = event_windows[:, 1].astype(np.float64)

    for i, ts in enumerate(onsets_by_roi):
        ts = np.asarray(ts, dtype=np.float64)
        if ts.size == 0:
            continue
        # Broadcasting trick: ``ts[None, :]`` adds a new leading
        # axis, turning the (n_onsets,) vector into shape
        # (1, n_onsets). ``starts[:, None]`` does the opposite,
        # turning (E,) into (E, 1). NumPy then broadcasts both to
        # (E, n_onsets) so the comparison checks every (event,
        # onset) pair simultaneously. Equivalent to a nested for-
        # loop but a hundred times faster.
        inside = (ts[None, :] >= starts[:, None]) & (ts[None, :] <= ends[:, None])  # (E, n_onsets)
        # ``any(axis=1)`` collapses across onsets -> per-event bool.
        any_inside = inside.any(axis=1)
        A[i, any_inside] = True
        # For each event the ROI participated in, record the first
        # onset time. ``np.where(...)`` returns the indices of True
        # values; we iterate just those.
        for e in np.where(any_inside)[0]:
            first_time[i, e] = float(ts[inside[e]].min())

    return A, first_time


# ---------------------------------------------------------------------------
# Suite2p detection / dF/F helpers (used by the Suite2p tab and the
# suite2p_pipeline / cellfilter modules under calliope.core).
# ---------------------------------------------------------------------------


def change_batch_according_to_free_ram() -> int:
    """Pick a Suite2p ``batch_size`` based on currently-available RAM.

    Suite2p's per-batch memory footprint is roughly linear in
    ``batch_size``, so we want to scale the parameter with whatever
    RAM we've got. The linear model below was tuned empirically:

        16 GiB available  -> batch_size 150
        200 GiB available -> batch_size 4000
        less than 13.5 GiB-> floor at 100 (safe baseline)

    Tab 3 calls this when seeding the Suite2p ops dict so a powerful
    workstation gets a faster run without manual tuning.
    """
    # ``psutil.virtual_memory().available`` is bytes; convert to GiB.
    available_mem = round(psutil.virtual_memory().available / (1024 ** 3), 1)
    if available_mem <= 13.5:
        return 100
    # ``int(...)`` truncates toward zero; the linear function is
    # ``20 * GiB - 170``.
    return int(20 * available_mem - 170)


def _peek_tiff_frame_shape(tiff_folder: str):
    """Return ``(Ly, Lx)`` from the first TIFF in the folder, or None.

    "Peek" because we don't read pixel data -- we only open the file
    and look at the page header to find the height and width. Used
    by ``change_nbinned_according_to_free_ram`` to size things
    appropriately for the recording's FOV without spending the time
    or RAM to actually load a frame.
    """
    try:
        import tifffile
    except ImportError:
        return None
    try:
        for name in sorted(os.listdir(tiff_folder)):
            if name.lower().endswith(('.tif', '.tiff')):
                with tifffile.TiffFile(os.path.join(tiff_folder, name)) as tf:
                    page = tf.pages[0]
                    shape = page.shape
                    if len(shape) >= 2:
                        return int(shape[-2]), int(shape[-1])
                break
    except Exception:
        return None
    return None


def change_nbinned_according_to_free_ram(tiff_folder: str,
                                         ram_fraction: float = 0.35,
                                         default: int = 1500,
                                         floor: int = 500,
                                         ceiling: int = 5000,
                                         peak_multiplier: float = 3.0) -> int:
    """Cap Suite2p's ``nbinned`` so sparsery's peak intermediate fits in RAM.

    Models the peak as ``peak_multiplier * nbinned * Ly * Lx * 4`` bytes and
    budgets ``ram_fraction`` of currently-available RAM for it. Falls back
    to ``default`` if the TIFF frame shape can't be read.
    """
    available_bytes = psutil.virtual_memory().available
    budget = available_bytes * ram_fraction

    shape = _peek_tiff_frame_shape(tiff_folder)
    if shape is None:
        print(f"[nbinned] could not peek TIFF shape in {tiff_folder}; "
              f"using default={default}")
        return default

    Ly, Lx = shape
    bytes_per_nbinned = Ly * Lx * 4 * peak_multiplier
    if bytes_per_nbinned <= 0:
        return default

    nbinned = int(budget // bytes_per_nbinned)
    capped = max(floor, min(ceiling, nbinned))
    avail_gb = available_bytes / (1024 ** 3)
    print(f"[nbinned] avail={avail_gb:.1f}GB frame={Ly}x{Lx} "
          f"raw_nbinned={nbinned} -> {capped} "
          f"(peak~{capped * Ly * Lx * 4 * peak_multiplier / 1024 ** 3:.2f}GB)")
    return capped


def s2p_infer_orientation(F: np.ndarray) -> tuple[int, int, bool]:
    """Look at the shape of an ``F.npy`` array and pull out
    ``(num_frames, num_rois, time_major)``.

    Suite2p's ``F.npy`` is canonically saved as ``(N_ROIs, T)`` --
    so a row per cell. We trust that layout; the third return value
    is a vestigial flag from older versions of this code that
    auto-detected the orientation.

    Raises ``ValueError`` if the array isn't 2D (defensive against
    accidentally loading the wrong file).
    """
    if F.ndim != 2:
        raise ValueError(f"Expected 2D array, got {F.shape}")
    n_rois, n_frames = F.shape
    return n_frames, n_rois, False


def s2p_load_raw(root: Union[str, Path]) -> tuple[np.ndarray, np.ndarray, int, int, bool]:
    """Load ``F.npy`` + ``Fneu.npy`` from a Suite2p plane0 folder.

    Returns
    -------
    F : (N_ROIs, T) ndarray
        Raw cell fluorescence trace per ROI.
    Fneu : (N_ROIs, T) ndarray
        Surrounding-neuropil trace per ROI.
    num_frames, num_rois : int
    time_major : bool
        Always False for canonical Suite2p output (rows = ROIs).

    Raises ``ValueError`` if F and Fneu have mismatched shapes
    (which would mean the folder is corrupted).
    """
    root = Path(root)
    # ``allow_pickle=False`` is the safe default -- prevents
    # arbitrary code execution if someone hands us a malicious .npy.
    F = np.load(root / "F.npy", allow_pickle=False)
    Fneu = np.load(root / "Fneu.npy", allow_pickle=False)
    if F.shape != Fneu.shape:
        raise ValueError(f"F and Fneu shapes differ: {F.shape} vs {Fneu.shape}")
    num_frames, num_rois, time_major = s2p_infer_orientation(F)
    return F, Fneu, num_frames, num_rois, time_major


def s2p_load_spks(plane0: Union[str, Path],
                  kept_idx: np.ndarray) -> np.ndarray:
    """Read Suite2p's ``spks.npy`` and subset it to a chosen set of
    ROIs.

    What ``spks.npy`` is
    --------------------
    Suite2p's deconvolution step (OASIS) produces a per-ROI sparse
    estimate of when each cell fired. Mostly zero with isolated
    non-zero peaks marking the deconvolved spikes. Tabs 5 and 8 both
    expose toggles to use this signal instead of the hysteresis-onset
    detector.

    Layout / orientation
    --------------------
    Modern Suite2p saves ``(N_total, T)``; some older variants save
    the transpose ``(T, N_total)``. We auto-detect by which axis can
    accommodate ``kept_idx.max()``.

    Parameters
    ----------
    plane0 : path-like
        The Suite2p plane0 folder.
    kept_idx : 1-D int array
        Suite2p ROI ids to keep -- typically the cell-filter pass
        intersected with any manual subset Tab 5 applied. ``spks_full``
        is sliced along the ROI axis by this array.

    Returns
    -------
    spks : (T, N_kept) float32 ndarray
        Always ``(T, N_kept)``-shaped regardless of the on-disk
        orientation. ``np.ascontiguousarray`` so subsequent vectorised
        slicing isn't slowed down by stride pessimism.

    Raises
    ------
    FileNotFoundError if ``spks.npy`` is missing.
    ValueError if its shape doesn't have exactly two axes.
    """
    plane0 = Path(plane0)
    spks_path = plane0 / "spks.npy"
    if not spks_path.exists():
        raise FileNotFoundError(spks_path)
    kept_idx = np.asarray(kept_idx, dtype=int)
    spks_full = np.load(spks_path)
    if spks_full.ndim != 2:
        raise ValueError(
            f"Unexpected spks.npy shape {spks_full.shape}")
    # Pick whichever axis matches the kept-ROI range.
    n_rois_axis = (0 if spks_full.shape[0] >= int(kept_idx.max()) + 1
                   else 1)
    if n_rois_axis == 0:
        spks = spks_full[kept_idx, :].T            # (T, N_kept)
    else:
        spks = spks_full[:, kept_idx]              # (T, N_kept)
    return np.ascontiguousarray(spks, dtype=np.float32)


def s2p_open_memmaps(root: Union[str, Path], prefix: str = "r0p7_") -> tuple[np.memmap, np.memmap, np.memmap, int, int]:
    """Open the three dF/F memmaps written by Tab 3 / Tab 4.

    On disk we keep three parallel ``(T, N)`` float32 memmaps:
        ``<prefix>dff.memmap.float32``
            Neuropil-corrected dF/F.
        ``<prefix>dff_lowpass.memmap.float32``
            ``dff`` after the user-chosen Butterworth low-pass.
        ``<prefix>dff_dt.memmap.float32``
            Savitzky-Golay first derivative of the low-passed
            trace.

    We use ``np.memmap`` rather than ``np.load`` because each file
    can be hundreds of MB; mmap'ing them lets us slice columns /
    frames without ever loading the whole array into RAM.

    For filtered prefixes (e.g. ``r0p7_filtered_``), the column count
    is derived from a keep-mask resolved via ``_load_keep_mask`` --
    which prefers ``predicted_cell_mask.npy`` and falls back to
    legacy / suite2p-classifier sources if it's missing.

    Returns ``(dff, low, dt, T, N_kept)``.
    """
    root = Path(root)
    F, _, num_frames, num_rois, _ = s2p_load_raw(root)

    if prefix.split("_")[-2] == "filtered":
        mask = _load_keep_mask(root, num_rois)
        if mask.size != num_rois:
            raise ValueError(
                f"keep-mask length {mask.size} != num_rois {num_rois} "
                f"in {root}"
            )
        F = F[mask, :]
        num_rois = int(mask.sum())

    dff = np.memmap(root / f"{prefix}dff.memmap.float32", dtype="float32", mode="r", shape=(num_frames, num_rois))
    low = np.memmap(root / f"{prefix}dff_lowpass.memmap.float32", dtype="float32", mode="r",
                    shape=(num_frames, num_rois))
    dt = np.memmap(root / f"{prefix}dff_dt.memmap.float32", dtype="float32", mode="r", shape=(num_frames, num_rois))
    return dff, low, dt, num_frames, num_rois


# ============================================================================
# Suite2p 1.0 plane-folder view
# ============================================================================
# Suite2p 1.0 split the legacy monolithic ``ops.npy`` into four files:
#
#     db.npy             -- per-plane database (Ly, Lx, nframes, nchannels,
#                           data_path, save_path[0], reg_file, ...)
#     settings.npy       -- pipeline settings (tau, fs, all detection /
#                           registration / extraction knobs, nested by
#                           subsystem: settings['detection'][...], etc.)
#     reg_outputs.npy    -- registration artefacts (meanImg, meanImgE,
#                           refImg, yoff, xoff, corrXY, badframes,
#                           yrange, xrange, bidiphase, ...)
#     detect_outputs.npy -- detection artefacts (max_proj, Vcorr,
#                           meanImg_crop, diameter, Vmax, Vmap, ...)
#
# Suite2p still writes a merged ``ops.npy`` when ``settings.io.save_ops_orig
# = True`` (the default), so ``ops.npy`` reads keep working on fresh outputs
# but are not the canonical 1.0 location.
#
# ``load_plane_view`` returns a single flat dict synthesised from the four
# canonical files (with an ``ops.npy`` fallback for legacy plane folders),
# so call-sites that historically did ``np.load("ops.npy").item()`` can
# switch to ``load_plane_view(plane0)`` without changing key names.

# Fields stored under nested settings paths in suite2p 1.0. Map flat key ->
# (settings, sub-keys...). Used to flatten settings.npy into the view dict.
_FLAT_FROM_SETTINGS: dict[str, tuple[str, ...]] = {
    # top-level
    "tau":                       ("tau",),
    "fs":                        ("fs",),
    "diameter":                  ("diameter",),
    "torch_device":              ("torch_device",),
    # run flags
    "do_registration":           ("run", "do_registration"),
    "do_detection":              ("run", "do_detection"),
    "do_deconvolution":          ("run", "do_deconvolution"),
    # io
    "save_mat":                  ("io", "save_mat"),
    "save_NWB":                  ("io", "save_NWB"),
    "save_ops_orig":             ("io", "save_ops_orig"),
    "delete_bin":                ("io", "delete_bin"),
    "move_bin":                  ("io", "move_bin"),
    "combined":                  ("io", "combined"),
    # registration
    "nimg_init":                 ("registration", "nimg_init"),
    "maxregshift":               ("registration", "maxregshift"),
    "nonrigid":                  ("registration", "nonrigid"),
    "smooth_sigma":              ("registration", "smooth_sigma"),
    "smooth_sigma_time":         ("registration", "smooth_sigma_time"),
    "two_step_registration":     ("registration", "two_step_registration"),
    # detection
    "denoise":                   ("detection", "denoise"),
    "threshold_scaling":         ("detection", "threshold_scaling"),
    "max_overlap":               ("detection", "max_overlap"),
    "spatial_scale":             ("detection", "sparsery_settings", "spatial_scale"),
    "max_iterations":            ("detection", "sourcery_settings", "max_iterations"),
    # classification
    "preclassify":               ("classification", "preclassify"),
    "classifier_path":           ("classification", "classifier_path"),
    # extraction
    "neuropil_extract":          ("extraction", "neuropil_extract"),
    "inner_neuropil_radius":     ("extraction", "inner_neuropil_radius"),
    "min_neuropil_pixels":       ("extraction", "min_neuropil_pixels"),
    "lam_percentile":            ("extraction", "lam_percentile"),
    "allow_overlap":             ("extraction", "allow_overlap"),
    # extraction (renamed)
    "neuropil_coefficient":      ("extraction", "neuropil_coefficient"),
    # deconvolution preprocess
    "baseline":                  ("dcnv_preprocess", "baseline"),
    "win_baseline":              ("dcnv_preprocess", "win_baseline"),
    "sig_baseline":              ("dcnv_preprocess", "sig_baseline"),
    "prctile_baseline":          ("dcnv_preprocess", "prctile_baseline"),
}


def _flatten_settings(settings: dict) -> dict:
    """Walk the nested ``settings`` dict and emit the flat keys readers
    historically used. Keys whose nested path is missing are skipped
    silently (older settings.npy files won't have every subsystem).
    """
    out: dict = {}
    for flat_key, path in _FLAT_FROM_SETTINGS.items():
        node = settings
        ok = True
        for p in path:
            if not isinstance(node, dict) or p not in node:
                ok = False
                break
            node = node[p]
        if ok:
            out[flat_key] = node
    return out


def load_plane_view(plane0: Union[str, Path]) -> dict:
    """Load a flat ops-like view of a suite2p 1.0 plane folder.

    Reads ``db.npy``, ``settings.npy``, ``reg_outputs.npy``, and
    ``detect_outputs.npy`` if present and merges their keys into one
    dict. Settings are flattened (e.g. ``settings['detection']
    ['threshold_scaling']`` becomes ``view['threshold_scaling']``).
    Falls back to a legacy ``ops.npy`` if the canonical files are
    missing, so old plane folders keep working.

    Calliope-side calibration (``pix_to_um``) is layered on top via
    :func:`load_pix_to_um` so callers see one merged dict.

    Returns an empty dict if nothing readable is found.
    """
    plane0 = Path(plane0)
    view: dict = {}

    # Load the four canonical files, ignoring missing ones.
    db_path = plane0 / "db.npy"
    settings_path = plane0 / "settings.npy"
    reg_path = plane0 / "reg_outputs.npy"
    det_path = plane0 / "detect_outputs.npy"

    have_canonical = False
    if db_path.exists():
        try:
            db = np.load(db_path, allow_pickle=True).item()
            if isinstance(db, dict):
                view.update(db)
                have_canonical = True
        except Exception:
            pass
    if settings_path.exists():
        try:
            settings = np.load(settings_path, allow_pickle=True).item()
            if isinstance(settings, dict):
                view.update(_flatten_settings(settings))
                have_canonical = True
        except Exception:
            pass
    if reg_path.exists():
        try:
            reg = np.load(reg_path, allow_pickle=True).item()
            if isinstance(reg, dict):
                view.update(reg)
                have_canonical = True
        except Exception:
            pass
    if det_path.exists():
        try:
            det = np.load(det_path, allow_pickle=True).item()
            if isinstance(det, dict):
                view.update(det)
                have_canonical = True
        except Exception:
            pass

    # Fallback: legacy ops.npy. Suite2p still writes this by default
    # (save_ops_orig=True), so fresh outputs have it too -- but the
    # canonical-file reads above take precedence on a per-key basis.
    if not have_canonical:
        ops_path = plane0 / "ops.npy"
        if ops_path.exists():
            try:
                ops = np.load(ops_path, allow_pickle=True).item()
                if isinstance(ops, dict):
                    view.update(ops)
            except Exception:
                pass

    # Layer calliope-side calibration on top.
    pix = load_pix_to_um(plane0)
    if pix is not None:
        view["pix_to_um"] = pix

    return view


# ============================================================================
# Calliope-side per-plane calibration
# ============================================================================
# The micrometre-per-pixel calibration is a calliope-only field (suite2p
# does not produce or read it). It used to be stamped onto ``ops.npy``,
# which couples calliope state to suite2p's output schema. The calibration
# now lives in its own ``calliope_calibration.npy`` next to the plane so
# we can stop touching ops.npy and so suite2p re-runs do not blow it away.

_CALLIOPE_CAL_NAME = "calliope_calibration.npy"


def _calibration_path(plane0: Union[str, Path]) -> Path:
    return Path(plane0) / _CALLIOPE_CAL_NAME


def load_pix_to_um(plane0: Union[str, Path]) -> Optional[float]:
    """Return the calliope-stamped µm-per-pixel calibration, or None.

    Looks first at ``<plane0>/calliope_calibration.npy``. Falls back to
    a legacy ``ops['pix_to_um']`` read if the new file is missing, so
    older plane folders keep working until they're re-stamped.
    """
    plane0 = Path(plane0)
    cal_path = _calibration_path(plane0)
    if cal_path.exists():
        try:
            cal = np.load(cal_path, allow_pickle=True).item()
            if isinstance(cal, dict) and "pix_to_um" in cal:
                v = cal["pix_to_um"]
                return float(v) if v is not None else None
        except Exception:
            pass
    # Legacy fallback.
    ops_path = plane0 / "ops.npy"
    if ops_path.exists():
        try:
            ops = np.load(ops_path, allow_pickle=True).item()
            if isinstance(ops, dict) and "pix_to_um" in ops:
                v = ops["pix_to_um"]
                return float(v) if v is not None else None
        except Exception:
            pass
    return None


def save_pix_to_um(plane0: Union[str, Path], value: float) -> None:
    """Stamp the calibration into ``<plane0>/calliope_calibration.npy``."""
    plane0 = Path(plane0)
    plane0.mkdir(parents=True, exist_ok=True)
    cal_path = _calibration_path(plane0)
    payload: dict = {}
    if cal_path.exists():
        try:
            existing = np.load(cal_path, allow_pickle=True).item()
            if isinstance(existing, dict):
                payload = existing
        except Exception:
            payload = {}
    payload["pix_to_um"] = float(value)
    np.save(cal_path, payload, allow_pickle=True)


def robust_df_over_f_1d(F, win_sec=45, perc=10, fps=30.0):
    """Rolling-percentile baseline dF/F for one trace.

    Computes ``dF/F = (F - F0) / eps`` where ``F0`` is a rolling
    Pth-percentile (default 10th) over a window of ``win_sec``
    seconds. Why a rolling baseline:
        - Long recordings drift slowly (photobleaching, focus
          shifts). A constant baseline either misses real activity
          early on or mis-flags later activity as no-event.
        - The 10th-percentile is robust: a window full of activity
          still has at least 10% "quiet" samples that we can lock
          onto.

    Tab 3 calls this for every ROI when the user picks "rolling 45 s
    percentile" as the baseline mode.

    NaN handling: any non-finite samples are linearly interpolated
    from neighbours before filtering.
    """
    F = np.asarray(F, dtype=np.float32)
    n = F.size

    # ``np.isfinite`` is True for normal numbers, False for inf/NaN.
    # If any are non-finite, replace them with linear interpolation
    # using ``np.interp(query, known_x, known_y)`` -- so the rolling
    # filter doesn't propagate NaNs.
    finite = np.isfinite(F)
    if not finite.all():
        F = np.interp(np.arange(n), np.flatnonzero(finite), F[finite]).astype(np.float32)

    # Window length in samples, forced odd. ``percentile_filter``
    # wants an odd-length kernel so the centre sample is well-defined.
    win = max(3, int(win_sec * fps) | 1)
    # Don't exceed the trace length.
    win = min(win, n if n % 2 == 1 else n - 1)
    if win < 3:
        # Trace is too short; fall back to a single global Pth-
        # percentile baseline.
        F0 = np.full_like(F, np.nanpercentile(F, perc))
    else:
        # The actual rolling Pth-percentile.
        F0 = percentile_filter(F, size=win, percentile=perc, mode='nearest').astype(np.float32)

    # ``eps`` (epsilon) is what we divide by. We use the 1st
    # percentile of the baseline -- not the baseline itself -- so a
    # bright ROI ends up normalised to a *cell-relative* scale, not
    # a per-frame moving target. ``eps = max(eps, 1e-9)`` guards
    # against literal zeros.
    eps = np.nanpercentile(F0, 1) if np.isfinite(F0).any() else 1.0
    eps = max(eps, 1e-9)
    dff = (F - F0) / eps
    return dff


def first_n_min_df_over_f_1d(F, baseline_min=2.0, perc=10, fps=30.0):
    """Constant baseline dF/F using only the first ``baseline_min``
    minutes of the trace.

    Used for short, stable recordings where the baseline doesn't
    drift (e.g. a 5-minute slice with constant aCSF). Cheaper than
    the rolling version and easier to interpret -- F0 is just one
    number.

    Tab 3 picks between this and ``robust_df_over_f_1d`` based on
    the user's "rolling vs. first-N-min" radio button.
    """
    F = np.asarray(F, dtype=np.float32)
    n = F.size

    # Same NaN-interpolation step as robust_df_over_f_1d.
    finite = np.isfinite(F)
    if not finite.all():
        F = np.interp(np.arange(n), np.flatnonzero(finite),
                      F[finite]).astype(np.float32)

    # Number of frames covering the requested baseline window.
    # ``round(baseline_min * 60 * fps)`` gives frames; ``int(...)``
    # truncates; ``max(1, ...)`` keeps it at least one frame.
    n_baseline = max(1, int(round(float(baseline_min) * 60.0 * float(fps))))
    n_baseline = min(n_baseline, n)
    # ``F[:n_baseline]`` is the slice "first n_baseline samples".
    F0_scalar = float(np.nanpercentile(F[:n_baseline], perc))
    eps = max(F0_scalar, 1e-9)
    return ((F - F0_scalar) / eps).astype(np.float32)


# ---------------------------------------------------------------------------
# AAV / notes-CSV lookups (used by suite2p_pipeline.load_base_settings)
# ---------------------------------------------------------------------------
#
# Why these helpers exist:
#   - The lab's AAV metadata sits in a CSV with one row per recording.
#   - Each row's "AAV" column carries a string like "GCaMP6f-rg",
#     "8m+CC", or "6s_GR" describing the GECI variant injected.
#   - Suite2p's deconvolution needs a numeric ``tau`` constant for the
#     calcium transient decay; that constant differs between 6f, 6m,
#     6s, 8m... so we look it up from a small in-code dictionary.
# These three functions wire the lookup together.


def aav_cleanup_and_dictionary_lookup(aav: str, dic: dict) -> float:
    """Find which dictionary key appears in the AAV string and return
    the matching value.

    Example
    -------
        dic = {"6f": 0.7, "6m": 1.0, "6s": 1.3, "8m": 0.137}
        aav_cleanup_and_dictionary_lookup("GCaMP6f-rg", dic)  # -> 0.7
    """
    # The "rg" suffix is a lab convention indicating "rabies-G"
    # delivery; strip it so it doesn't trip up the lookup.
    aav = aav.replace("rg", "")
    # Split on any of ``-``, ``_``, ``+``.
    components = re.split(r"[-_+]", aav)
    # Lower-case both sides so "6F" and "6f" match.
    dict_lower = {k.lower(): v for k, v in dic.items()}
    list_lower = {item.lower() for item in components}
    # ``set & set`` intersection. ``next(iter(...))`` grabs an
    # arbitrary element from the intersection -- we assume there's
    # at most one match.
    common = dict_lower.keys() & list_lower
    return dict_lower[next(iter(common))]


def get_row_number_csv_module(csv_filename: str, header_name: str, target_element: str) -> int:
    """Find the row number in ``csv_filename`` whose ``header_name``
    column matches ``target_element`` after splitting on ``-``/``_``
    and casting to ints.

    Why the int-list comparison: filenames in the lab look like
    ``2024-07-01_00018``; the metadata CSV's ``video`` column may
    have the same recording typed as ``20240701-00018`` or
    ``2024-07-01-00018``. By extracting the integer fragments we
    match across all those punctuation variants.

    Returns 1-based row number (so callers can use it as
    ``df.iloc[row_num - 1]``), or None if no row matches.
    """
    try:
        col = pd.read_csv(csv_filename, usecols=[header_name])
    except ValueError:
        print(f"Error: Header '{header_name}' not found in the CSV file.")
        return None

    # Local helper closure: turn a string into a list of ints
    # extracted from its hyphen / underscore-separated fragments.
    def to_int_list(s: str):
        return [int(x) for x in re.split(r"[-_]", str(s)) if x.isdigit()]

    target_list = to_int_list(target_element)
    # ``col[header_name]`` is a pandas Series; ``.items()`` iterates
    # (index, value) pairs.
    for idx, val in col[header_name].items():
        if to_int_list(val) == target_list:
            return idx + 1
    return None


def file_name_to_aav_to_dictionary_lookup(file_name, aav_info_csv, dic):
    """Compose the two helpers above: take a recording filename, find
    its row in the AAV CSV, read the ``AAV`` column, and resolve the
    matching dictionary value (e.g. tau).

    Used by ``suite2p_pipeline.load_base_settings`` to set the
    recording-specific tau in the suite2p settings dict before
    detection.
    """
    row_num = get_row_number_csv_module(aav_info_csv, 'video', file_name)
    col = pd.read_csv(aav_info_csv, usecols=["AAV"])
    # ``iloc[row_num - 1]`` is positional indexing: 1-based -> 0-based.
    element = col["AAV"].iloc[row_num - 1]
    element = str(element)
    return aav_cleanup_and_dictionary_lookup(element, dic)
