"""Trimmed copy of ``utils.py`` for the calliope GUI.

Public symbols used by tabs and supporting modules (cellfilter,
adaptive_detection, sparse_plus_cellpose):

    Tab-driven:
        paint_spatial, get_fps_from_notes, mad_z, hysteresis_onsets,
        detect_event_windows, plot_event_detection, shade_event_windows,
        EventDetectionParams, _load_keep_mask, lowpass_causal_1d,
        sg_first_derivative_1d
    Suite2p detection / cellfilter:
        change_batch_according_to_free_ram, change_nbinned_according_to_free_ram,
        s2p_infer_orientation, s2p_load_raw, s2p_open_memmaps,
        robust_df_over_f_1d, first_n_min_df_over_f_1d,
        aav_cleanup_and_dictionary_lookup, get_row_number_csv_module,
        file_name_to_aav_to_dictionary_lookup
"""

from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd
import psutil
from scipy.ndimage import gaussian_filter1d, percentile_filter
from scipy.signal import butter, find_peaks, savgol_filter, sosfilt
from scipy.special import erfinv
from pathlib import Path
from typing import Optional, Sequence, Union
from dataclasses import dataclass

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
    """Causal SOS low-pass (no filtfilt copies). Returns y, zf, sos."""
    x = np.asarray(x, dtype=np.float32)
    n = x.size
    if n < 3:
        return x.copy(), zi, sos

    nyq = fps / 2.0
    cutoff = min(max(1e-4, cutoff_hz), 0.95 * nyq)
    if sos is None:
        sos = butter(order, cutoff / nyq, btype='low', output='sos')

    if zi is None:
        # zi shape: (sections, 2). Init with first sample to avoid transient.
        zi = np.zeros((sos.shape[0], 2), dtype=np.float32)
        zi[:, 0] = x[0]
        zi[:, 1] = x[0]

    y, zf = sosfilt(sos, x, zi=zi)
    return y.astype(np.float32), zf.astype(np.float32), sos


def sg_first_derivative_1d(x, fps, win_ms=333, poly=3):
    """Savitzky-Golay smoothed first derivative on 1D."""
    x = np.asarray(x, dtype=np.float32)
    n = x.size
    win = max(3, int((win_ms / 1000.0) * fps) | 1)  # odd
    if win >= n:
        win = max(3, (n - (1 - n % 2)))  # largest valid odd <= n
    if win < 3 or n < 3:
        # fallback simple gradient
        g = np.empty_like(x)
        g[0] = 0.0
        g[1:] = (x[1:] - x[:-1]) * fps
        return g
    return savgol_filter(x, window_length=win, polyorder=poly, deriv=1, delta=1.0 / fps).astype(np.float32)


def mad_z(x):
    """
    Robust z-score using MAD (per ROI), with 1.4826 factor to approximate σ for normal data. (stolen from Stern et al. 2024)
    Returns z, and the median/MAD for optional inverse transforms.
    """
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return (x - med) / (1.4826 * mad), med, mad


def hysteresis_onsets(z, z_hi, z_lo, fps, min_sep_s=0.0):
    """
    Hysteresis onset detection on robust z:
      - enter when z >= z_hi; exit when z <= z_lo
      - merge onsets closer than min_sep_s (sec)
    Returns onset frame indices (np.int32).
    """
    above_hi = z >= z_hi
    onsets = []
    active = False
    for i in range(z.size):
        if not active and above_hi[i]:
            active = True
            onsets.append(i)
        elif active and z[i] <= z_lo:
            active = False
    if not onsets:
        return np.array([], dtype=int)
    onsets = np.array(onsets, dtype=int)
    if min_sep_s > 0:
        min_sep = int(min_sep_s * fps)
        merged = [onsets[0]]
        for k in onsets[1:]:
            if k - merged[-1] >= min_sep:
                merged.append(k)
        onsets = np.asarray(merged, dtype=int)
    return onsets


def paint_spatial(values_per_roi, stat_list, Ly, Lx):
    """
    Paint per-ROI scalar values onto the imaging plane using ROI masks.
    Uses 'lam' weights for soft assignment; normalizes by accumulated weight.
    Returns (Ly, Lx) float32 image.
    """
    # Initialize the output image array with zeros, shape matches imaging plane dimensions
    img = np.zeros((Ly, Lx), dtype=np.float32)

    # Initialize weight accumulator array to track total lambda weights per pixel
    w = np.zeros((Ly, Lx), dtype=np.float32)

    # Iterate through each ROI and its corresponding statistics dictionary
    for j, s in enumerate(stat_list):
        # Get the scalar value (metric) for the current ROI
        v = values_per_roi[j]

        # Extract y and x-coordinates of all pixels belonging to this ROI
        ypix = s['ypix']
        xpix = s['xpix']

        # Extract lambda weights (pixel-wise contribution strengths) and convert to float32
        lam = s['lam'].astype(np.float32)

        # Add weighted ROI value to image: each pixel gets v * its lambda weight
        img[ypix, xpix] += v * lam

        # Accumulate the lambda weights at each pixel (for normalization)
        w[ypix, xpix] += lam

    # Create boolean mask identifying pixels with non-zero accumulated weights
    m = w > 0

    # Normalize weighted values by dividing by accumulated weights (only where w > 0)
    img[m] /= w[m]

    return img


RECORDING_DIR_RE = re.compile(r"\d{4}-\d{2}-\d{2}_\d+")


def _find_recording_root(path: Path) -> Path | None:
    """
    Walk upward until we find a folder named YYYY-MM-DD_#####.
    """
    for p in [path] + list(path.parents):
        if RECORDING_DIR_RE.fullmatch(p.name):
            return p
    return None


def get_fps_from_notes(
        path: str,
        notes_root: str = r"F:\notes_recordings",
        default_fps: float = 15.07,
) -> float:
    """
    Resolve FPS for a recording, given ANY path inside that recording.
    Safe fallback to default_fps if metadata is missing.
    """
    try:
        path = Path(path)

        rec_root = _find_recording_root(path)
        if rec_root is None:
            return default_fps

        date_str, rec_str = rec_root.name.split("_", 1)
        target = f"{date_str}-{rec_str}"

        notes_root = Path(notes_root)
        notes_path = notes_root / f"{date_str}.xlsx"

        if not notes_path.exists():
            candidates = sorted(notes_root.glob(f"*{date_str}*.xlsx"))
            if not candidates:
                return default_fps
            notes_path = candidates[0]

        df = pd.read_excel(notes_path, sheet_name="2P settings")
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
    """
    All parameters for density-based event detection and boundary refinement.

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
    if params is None:
        params = EventDetectionParams()

    # 1) Flat onset times across all ROIs -> density
    duration_s = float(T) / float(fps)
    centers, counts, smooth = _build_density(
        onsets_by_roi=onsets_by_roi,
        duration_s=duration_s,
        bin_sec=params.bin_sec,
        smooth_sigma_bins=params.smooth_sigma_bins,
        n_rois=len(onsets_by_roi),
        normalize_by_num_rois=params.normalize_by_num_rois,
    )

    # 2) Peak detection
    # NEW: pass prominence_wlen_s through so prominence is measured locally
    # (matches event_detection_short.py behaviour).
    wlen_bins_val = max(3, int(round(params.prominence_wlen_s / max(params.bin_sec, 1e-9))) | 1)
    peak_indices = _detect_density_peaks(
        smooth=smooth,
        min_prominence=params.min_prominence,
        min_width_bins=params.min_width_bins,
        min_distance_bins=params.min_distance_bins,
        wlen_bins=wlen_bins_val,
    )

    # 3) Per-event boundaries
    boundaries = _boundaries_from_peaks(
        time_s=centers,
        smooth=smooth,
        peak_indices=peak_indices,
        params=params,
    )

    event_windows = np.column_stack([boundaries["start_s"], boundaries["end_s"]])

    # 4) Activation matrix + first_time per event (drop-in for _activation_matrix)
    A, first_time = _activation_matrix_from_windows(onsets_by_roi, event_windows)

    if not return_diagnostics:
        return event_windows, A, first_time

    diagnostics = {
        "time_centers_s": centers,
        "binned_density": counts,
        "smoothed_density": smooth,
        "baseline_trace": boundaries["baseline_trace"],
        "end_threshold_trace": boundaries["end_threshold_trace"],
        "baseline_noise": boundaries["baseline_noise"],
        "peak_s": boundaries["peak_s"],
        "peak_height": boundaries["peak_height"],
        "mu_s": boundaries["mu_s"],
        "sigma_s": boundaries["sigma_s"],
        "boundary_source_left": boundaries["boundary_source_left"],
        "boundary_source_right": boundaries["boundary_source_right"],
        "prominence": boundaries["prominence"],
        "duration_s": boundaries["duration_s"],
    }
    return event_windows, A, first_time, diagnostics


# ---------- plotting helper ----------

def plot_event_detection(
        diagnostics: dict,
        ylabel: str = "Onset density (per bin per ROI)",
        title: str = "Detected events on onset density",
        ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Figure: binned counts (bars) + smoothed density (line) + baseline +
    end threshold + shaded event windows + peak markers.
    Takes the `diagnostics` dict returned by detect_event_windows(..., return_diagnostics=True).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 4.5))
    else:
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
    """Overlay shaded event windows onto an existing axis."""
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
    """Flatten onsets, histogram them, and Gaussian-smooth."""
    nonempty = [np.asarray(x, dtype=np.float64) for x in onsets_by_roi if len(x) > 0]
    flat = np.concatenate(nonempty) if nonempty else np.array([], dtype=np.float64)

    edges = np.arange(0.0, duration_s + bin_sec, bin_sec, dtype=np.float64)
    if edges[-1] < duration_s:
        edges = np.append(edges, duration_s)

    counts, edges = np.histogram(flat, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    counts = counts.astype(np.float64)
    if normalize_by_num_rois and n_rois > 0:
        counts = counts / float(n_rois)

    smooth = gaussian_filter1d(counts, sigma=smooth_sigma_bins, mode="nearest")
    return centers, counts, smooth


def _detect_density_peaks(
        smooth: np.ndarray,
        min_prominence: float,
        min_width_bins: float,
        min_distance_bins: float,
        wlen_bins: int = 0,
) -> np.ndarray:
    """Run scipy find_peaks with the same gates as event_detection.detect_density_peaks."""
    kw = dict(
        prominence=min_prominence,
        width=min_width_bins,
        distance=max(1, int(round(min_distance_bins))),
    )
    if wlen_bins and wlen_bins >= 3:
        kw["wlen"] = int(wlen_bins)
    peaks, _props = find_peaks(smooth, **kw)
    return peaks


# ---- baseline / noise ----

def _estimate_global_baseline(density: np.ndarray, baseline_percentile: float) -> np.ndarray:
    if density.size == 0:
        return np.zeros_like(density)
    b = float(np.percentile(density, baseline_percentile))
    return np.full_like(density, b, dtype=np.float64)


def _estimate_rolling_baseline(
        density: np.ndarray,
        fps_density: float,
        baseline_window_s: float,
        baseline_percentile: float,
) -> np.ndarray:
    if density.size == 0:
        return np.zeros_like(density)
    win = max(3, int(round(baseline_window_s * fps_density)) | 1)
    win = min(win, density.size if density.size % 2 == 1 else density.size - 1)
    if win < 3:
        return _estimate_global_baseline(density, baseline_percentile)
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
    if density.size == 0:
        return 0.0
    resid = density - baseline_trace
    cutoff = float(np.percentile(resid, quiet_percentile))
    quiet_mask = resid <= cutoff
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
    n = density.size
    i = peak_idx
    steps = 0
    while 0 <= i + direction < n and steps < max_steps:
        nxt = i + direction
        if density[nxt] <= end_threshold_trace[nxt]:
            return i
        i = nxt
        steps += 1
    return i


# ---- Gaussian fit ----

def _gaussian_z(q: float) -> float:
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
    n = density.size
    L = max(0, left_idx - pad_samples)
    R = min(n - 1, right_idx + pad_samples)
    if R <= L:
        return float(time_s[peak_idx]), float(min_sigma_s)

    t_win = time_s[L:R + 1].astype(np.float64)
    d_win = density[L:R + 1].astype(np.float64)
    b_win = baseline_trace[L:R + 1].astype(np.float64)
    w = np.clip(d_win - b_win, 0.0, None)

    total = w.sum()
    if total <= 0:
        return float(time_s[peak_idx]), float(min_sigma_s)

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
    """Baseline walk + Gaussian fit + merge. Returns a dict of per-event arrays."""
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
    """
    Drop-in for the old _activation_matrix, but using variable-width event
    windows instead of fixed bins.

    Returns
    -------
    A : (N, E) bool     -- any onset in [start_s, end_s] ?
    first_time : (N, E) float -- earliest onset time (s) in window, NaN if inactive
    """
    N = len(onsets_by_roi)
    E = event_windows.shape[0]
    A = np.zeros((N, E), dtype=bool)
    first_time = np.full((N, E), np.nan, dtype=np.float64)

    if E == 0:
        return A, first_time

    starts = event_windows[:, 0].astype(np.float64)
    ends = event_windows[:, 1].astype(np.float64)

    for i, ts in enumerate(onsets_by_roi):
        ts = np.asarray(ts, dtype=np.float64)
        if ts.size == 0:
            continue
        # For each event window, find the earliest onset inside [start, end]
        # Vectorised: broadcast onsets against window boundaries (E can be large,
        # but typically 20-200 events so (E x ts.size) is fine).
        # If N*E*ts is ever too large we can switch to searchsorted per ROI.
        inside = (ts[None, :] >= starts[:, None]) & (ts[None, :] <= ends[:, None])  # (E, n_onsets)
        any_inside = inside.any(axis=1)
        A[i, any_inside] = True
        for e in np.where(any_inside)[0]:
            first_time[i, e] = float(ts[inside[e]].min())

    return A, first_time


# ---------------------------------------------------------------------------
# Suite2p detection / dF/F helpers (used by the Suite2p tab and the
# adaptive_detection / cellfilter modules under calliope.core).
# ---------------------------------------------------------------------------


def change_batch_according_to_free_ram() -> int:
    """Pick a Suite2p batch_size by interpolating between (16 GiB -> 150)
    and (200 GiB -> 4000) of free system RAM. Floors at 100."""
    available_mem = round(psutil.virtual_memory().available / (1024 ** 3), 1)
    if available_mem <= 13.5:
        return 100
    return int(20 * available_mem - 170)


def _peek_tiff_frame_shape(tiff_folder: str):
    """Return (Ly, Lx) from the first TIFF in the folder, or None if unavailable."""
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
    """Suite2p's F.npy / Fneu.npy are canonically (N_ROIs, T). Trust that
    layout. Returns (num_frames, num_rois, time_major) where time_major=False.
    """
    if F.ndim != 2:
        raise ValueError(f"Expected 2D array, got {F.shape}")
    n_rois, n_frames = F.shape
    return n_frames, n_rois, False


def s2p_load_raw(root: Union[str, Path]) -> tuple[np.ndarray, np.ndarray, int, int, bool]:
    """Load F.npy + Fneu.npy. Returns (F, Fneu, num_frames, num_rois, time_major)."""
    root = Path(root)
    F = np.load(root / "F.npy", allow_pickle=False)
    Fneu = np.load(root / "Fneu.npy", allow_pickle=False)
    if F.shape != Fneu.shape:
        raise ValueError(f"F and Fneu shapes differ: {F.shape} vs {Fneu.shape}")
    num_frames, num_rois, time_major = s2p_infer_orientation(F)
    return F, Fneu, num_frames, num_rois, time_major


def s2p_open_memmaps(root: Union[str, Path], prefix: str = "r0p7_") -> tuple[np.memmap, np.memmap, np.memmap, int, int]:
    """Open dF/F, low-pass dF/F, and derivative Suite2p memmaps with a given
    prefix. Returns (dff, low, dt, T, N) with shape (T, N).

    For filtered prefixes (e.g. ``r0p7_filtered_``), the column count is
    derived from a keep-mask resolved via ``_load_keep_mask`` -- which
    prefers ``predicted_cell_mask.npy`` and falls back to legacy /
    suite2p-classifier sources if it's missing.
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


def robust_df_over_f_1d(F, win_sec=45, perc=10, fps=30.0):
    """Rolling percentile baseline on a 1D array, low-RAM."""
    F = np.asarray(F, dtype=np.float32)
    n = F.size

    finite = np.isfinite(F)
    if not finite.all():
        F = np.interp(np.arange(n), np.flatnonzero(finite), F[finite]).astype(np.float32)

    win = max(3, int(win_sec * fps) | 1)  # odd
    win = min(win, n if n % 2 == 1 else n - 1)
    if win < 3:
        F0 = np.full_like(F, np.nanpercentile(F, perc))
    else:
        F0 = percentile_filter(F, size=win, percentile=perc, mode='nearest').astype(np.float32)

    eps = np.nanpercentile(F0, 1) if np.isfinite(F0).any() else 1.0
    eps = max(eps, 1e-9)
    dff = (F - F0) / eps
    return dff


def first_n_min_df_over_f_1d(F, baseline_min=2.0, perc=10, fps=30.0):
    """Constant baseline taken from the first ``baseline_min`` minutes."""
    F = np.asarray(F, dtype=np.float32)
    n = F.size

    finite = np.isfinite(F)
    if not finite.all():
        F = np.interp(np.arange(n), np.flatnonzero(finite),
                      F[finite]).astype(np.float32)

    n_baseline = max(1, int(round(float(baseline_min) * 60.0 * float(fps))))
    n_baseline = min(n_baseline, n)
    F0_scalar = float(np.nanpercentile(F[:n_baseline], perc))
    eps = max(F0_scalar, 1e-9)
    return ((F - F0_scalar) / eps).astype(np.float32)


# ---------------------------------------------------------------------------
# AAV / notes-CSV lookups (used by adaptive_detection)
# ---------------------------------------------------------------------------


def aav_cleanup_and_dictionary_lookup(aav: str, dic: dict) -> float:
    """Return the dict value matching an AAV string after stripping 'rg' and
    splitting on -, _, +."""
    aav = aav.replace("rg", "")
    components = re.split(r"[-_+]", aav)
    dict_lower = {k.lower(): v for k, v in dic.items()}
    list_lower = {item.lower() for item in components}
    common = dict_lower.keys() & list_lower
    return dict_lower[next(iter(common))]


def get_row_number_csv_module(csv_filename: str, header_name: str, target_element: str) -> int:
    """Return the 1-based row number under ``header_name`` whose value
    splits (on -/_) into the same int list as ``target_element``."""
    try:
        col = pd.read_csv(csv_filename, usecols=[header_name])
    except ValueError:
        print(f"Error: Header '{header_name}' not found in the CSV file.")
        return None

    def to_int_list(s: str):
        return [int(x) for x in re.split(r"[-_]", str(s)) if x.isdigit()]

    target_list = to_int_list(target_element)
    for idx, val in col[header_name].items():
        if to_int_list(val) == target_list:
            return idx + 1
    return None


def file_name_to_aav_to_dictionary_lookup(file_name, aav_info_csv, dic):
    """Look up ``file_name`` under the ``video`` column of ``aav_info_csv``,
    pull the matching ``AAV`` string, and resolve a dict value via
    ``aav_cleanup_and_dictionary_lookup``."""
    row_num = get_row_number_csv_module(aav_info_csv, 'video', file_name)
    col = pd.read_csv(aav_info_csv, usecols=["AAV"])
    element = col["AAV"].iloc[row_num - 1]
    element = str(element)
    return aav_cleanup_and_dictionary_lookup(element, dic)
