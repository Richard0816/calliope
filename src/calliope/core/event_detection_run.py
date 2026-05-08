"""Headless event detection for batch execution and Tab 5.

Why this module exists
----------------------
Tab 5's full compute body (~150 lines: per-ROI MAD-z + hysteresis,
event-window detection, summary writing) used to live inside
``EventDetectionTab._compute_render_data`` and ``_on_render_done``.
Tab 0's batch runner needs to invoke the same logic without a Tk
window, so we lift it here. The tab still owns the matplotlib
canvases and slider interactions, but the actual computation goes
through this module.

Public surface
--------------
``run_event_detection(plane0, params, *, figures_dir=None,
                      write_summary=True, progress_cb=None)``
    The headless entry point. Loads memmaps, runs MAD-z +
    hysteresis_onsets per ROI, runs ``detect_event_windows`` on the
    pooled onsets, optionally renders heatmap / raster / detection
    figures + writes the events sheets to ``calliope_summary.xlsx``.
    Returns the same data dict the tab consumes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import utils as core_utils
from ..gui_common import format_roi_indices, parse_manual_roi_spec


# Knob names that map 1:1 to ``EventDetectionParams`` fields. Kept in
# sync with Tab 5's PARAM_SPEC -- if you add a knob there, add it
# here too.
_EVENT_DETECTION_FIELDS = (
    "bin_sec", "smooth_sigma_bins", "normalize_by_num_rois",
    "min_prominence", "min_width_bins", "min_distance_bins",
    "prominence_wlen_s",
    "baseline_mode", "baseline_percentile",
    "baseline_window_s", "noise_quiet_percentile",
    "noise_mad_factor", "end_threshold_k",
    "max_walk_duration_s", "max_event_duration_s",
    "enable_watershed_split", "enforce_symmetric_clamp",
    "merge_gap_s",
    "use_gaussian_boundary", "gaussian_quantile",
    "gaussian_fit_pad_s", "gaussian_min_sigma_s",
)


def _resolve_keep_mask(plane0: Path, N_total: int) -> np.ndarray:
    """Pick the same keep mask Tab 5 uses (cellfilter -> iscell ->
    all)."""
    mask_path = plane0 / "predicted_cell_mask.npy"
    if mask_path.exists():
        return np.load(mask_path).astype(bool)
    iscell_path = plane0 / "iscell.npy"
    if iscell_path.exists():
        ic = np.load(iscell_path)
        return ((ic[:, 0] > 0) if ic.ndim == 2 else (ic > 0)).astype(bool)
    return np.ones(N_total, dtype=bool)


def _build_event_params(params: dict):
    """Translate the flat ``params`` dict into an
    :class:`EventDetectionParams` instance, using class defaults for
    any missing field.
    """
    from ..tabs.event_detection.logic import EventDetectionParams
    obj = EventDetectionParams()
    for fname in _EVENT_DETECTION_FIELDS:
        if fname in params:
            try:
                setattr(obj, fname, params[fname])
            except Exception:
                pass
    return obj


def run_event_detection(
    plane0: Path,
    params: dict,
    *,
    figures_dir: Optional[Path] = None,
    write_summary: bool = True,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """Run Tab 5's full event-detection pipeline on ``plane0``.

    Parameters
    ----------
    plane0 : Path
        Suite2p plane folder (from Tab 3) containing F.npy + the
        ``r0p7_filtered_dff_lowpass.memmap.float32`` /
        ``r0p7_filtered_dff_dt.memmap.float32`` memmaps that Tab 4
        wrote.
    params : dict
        Same shape as ``EventDetectionTab._params``: per-ROI
        hysteresis (``z_enter``, ``z_exit``, ``min_sep_s``), display
        target (``time_cols_target``), the
        :class:`EventDetectionParams` fields, plus optional
        ``onset_source`` (``"derivative"`` (default) or ``"spks"``),
        ``manual_subset_enabled``, ``manual_roi_spec``.
    figures_dir : Path or None
        If set, save heatmap / raster / event-detection PNGs.
    write_summary : bool
        If True (default), append events sheets to
        ``<plane0>/calliope_summary.xlsx`` via
        :func:`summary_writer.write_events_sheets`.
    progress_cb : callable
        Optional ``callback(message)`` for status strings.

    Returns
    -------
    dict
        Same shape as Tab 5's render payload (heatmap / raster /
        kept_idx / event_windows / A / first_time / fps / T /
        N_kept / onsets_by_roi / diagnostics / downsample).
    """
    from ..tabs.event_detection import logic as ed_utils

    plane0 = Path(plane0)
    F = np.load(plane0 / "F.npy", mmap_mode="r")
    N_total, T = F.shape

    mask = _resolve_keep_mask(plane0, N_total)
    N_kept_full = int(mask.sum())
    if N_kept_full == 0:
        raise RuntimeError("No ROIs survive the cell-filter mask.")

    lp_path = plane0 / "r0p7_filtered_dff_lowpass.memmap.float32"
    dt_path = plane0 / "r0p7_filtered_dff_dt.memmap.float32"
    if not lp_path.exists() or not dt_path.exists():
        raise FileNotFoundError(
            f"Missing low-pass memmaps in {plane0}. "
            "Run Tab 4 (or core.lowpass_run.run_lowpass) first.")
    lowpass = np.memmap(str(lp_path), dtype="float32", mode="r",
                        shape=(T, N_kept_full))
    derivative = np.memmap(str(dt_path), dtype="float32", mode="r",
                           shape=(T, N_kept_full))

    kept_full_idx = np.flatnonzero(mask)
    pos_in_memmap = np.arange(N_kept_full)
    kept_idx = kept_full_idx.copy()

    if params.get("manual_subset_enabled"):
        spec = params.get("manual_roi_spec", "")
        try:
            manual_ids = parse_manual_roi_spec(spec, N_total)
        except ValueError as e:
            raise RuntimeError(
                f"Manual ROI spec invalid: {e!s} (input {spec!r})")
        sid_to_pos = {int(sid): pos
                      for pos, sid in enumerate(kept_full_idx)}
        valid: list[tuple[int, int]] = []
        skipped: list[int] = []
        for sid in manual_ids:
            if sid in sid_to_pos:
                valid.append((sid, sid_to_pos[sid]))
            else:
                skipped.append(sid)
        if not valid:
            raise RuntimeError(
                f"None of the manual ROI ids survive the keep mask "
                f"(skipped: {skipped[:8]}"
                + (' ...' if len(skipped) > 8 else '') + ').')
        if skipped and progress_cb is not None:
            progress_cb(
                f"manual ROI ids skipped (not in keep mask): "
                f"{format_roi_indices(skipped)}")
        kept_idx = np.array([sid for sid, _ in valid], dtype=int)
        pos_in_memmap = np.array([pos for _, pos in valid], dtype=int)

    N_kept = int(pos_in_memmap.size)

    fps = float(ed_utils.get_fps_from_notes(str(plane0)))

    z_enter = float(params.get("z_enter", 3.5))
    z_exit = float(params.get("z_exit", 1.5))
    min_sep_s = float(params.get("min_sep_s", 0.1))
    time_cols_target = int(params.get("time_cols_target", 1200))

    downsample = max(1, T // time_cols_target)
    num_cols = T // downsample
    heatmap = np.zeros((N_kept, num_cols), dtype=np.uint8)
    raster = np.zeros((N_kept, num_cols), dtype=np.uint8)
    event_counts = np.zeros(N_kept, dtype=np.int32)
    onsets_by_roi: list = []

    onset_source = params.get("onset_source", "derivative")
    spks_arr = None
    if onset_source == "spks":
        spks_arr = core_utils.s2p_load_spks(plane0, kept_full_idx)

    report_every = max(1, N_kept // 20)
    for i in range(N_kept):
        mem_col = int(pos_in_memmap[i])
        lp_i = np.asarray(lowpass[:, mem_col], dtype=np.float32)
        if onset_source == "spks":
            src_i = np.asarray(spks_arr[:, mem_col], dtype=np.float32)
        else:
            src_i = np.asarray(derivative[:, mem_col], dtype=np.float32)
        z, _, _ = ed_utils.mad_z(src_i)
        onsets = ed_utils.hysteresis_onsets(
            z, z_enter, z_exit, fps, min_sep_s=min_sep_s)
        event_counts[i] = onsets.size
        onsets_by_roi.append(np.asarray(onsets, dtype=np.float64) / fps)

        if downsample > 1:
            trimmed = lp_i[:num_cols * downsample].reshape(
                num_cols, downsample)
            lp_ds = trimmed.mean(axis=1)
            if onsets.size:
                bins = (onsets // downsample).clip(0, num_cols - 1)
                raster[i, np.unique(bins)] = 1
        else:
            lp_ds = lp_i
            if onsets.size:
                raster[i, onsets.clip(0, num_cols - 1)] = 1

        lo, hi = np.percentile(lp_ds, [1, 99])
        if hi <= lo:
            heatmap[i, :] = 0
        else:
            norm = np.clip((lp_ds - lo) / (hi - lo), 0, 1)
            heatmap[i, :] = (norm * 255.0 + 0.5).astype(np.uint8)

        if progress_cb is not None and (i + 1) % report_every == 0:
            progress_cb(f"event detection: {i + 1}/{N_kept} ROIs")

    order = np.argsort(-event_counts)
    heatmap = heatmap[order]
    raster = raster[order]

    ed_kwargs: dict = {"params": _build_event_params(params)}
    event_windows, A, first_time, diagnostics = \
        ed_utils.detect_event_windows(
            onsets_by_roi, T=T, fps=fps,
            return_diagnostics=True, **ed_kwargs,
        )

    data = {
        "heatmap": heatmap,
        "raster": raster,
        "kept_idx": kept_idx,
        "pos_in_memmap": pos_in_memmap,
        "N_kept_full": N_kept_full,
        "event_counts": event_counts,
        "diagnostics": diagnostics,
        "event_windows": event_windows,
        "A": A,
        "first_time": first_time,
        "onsets_by_roi": onsets_by_roi,
        "fps": fps,
        "T": T,
        "N_kept": N_kept,
        "downsample": downsample,
    }

    if figures_dir is not None:
        figs = _render_event_figures(data, figures_dir=Path(figures_dir))
        data["figures"] = figs

    if write_summary:
        from ..tabs.event_detection import summary_writer
        summary_writer.update_recording_meta(
            plane0, fps=fps, T=T, N=N_kept)
        summary_writer.write_events_sheets(
            plane0,
            event_windows=event_windows,
            onsets_by_roi=onsets_by_roi,
            fps=fps,
            in_seconds=True,
        )

    return data


def _render_event_figures(data: dict, *, figures_dir: Path) -> list[str]:
    """Save the heatmap, event raster, and event-detection diagnostic
    plots that Tab 5 normally shows live, as PNGs under
    ``figures_dir``.
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from ..tabs.event_detection import logic as ed_utils

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    written = []

    hm = data["heatmap"]
    rs = data["raster"]
    fps = data["fps"]; T = data["T"]; ds = data["downsample"]
    bin_ms = ds * 1000.0 / fps if fps > 0 else 0.0

    fig = plt.Figure(figsize=(9, 4), tight_layout=True)
    ax = fig.add_subplot(111)
    ax.imshow(hm, aspect="auto", interpolation="nearest")
    ax.set_xlabel(f"Time (downsampled bins ~{bin_ms:.0f} ms)")
    ax.set_ylabel("ROIs (most active at top)")
    ax.set_title("Low-pass dF/F heatmap")
    p = figures_dir / "heatmap.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(str(p))

    fig = plt.Figure(figsize=(9, 4), tight_layout=True)
    ax = fig.add_subplot(111)
    ax.imshow(rs, aspect="auto", interpolation="nearest", cmap="Greys")
    ax.set_xlabel(f"Time (downsampled bins ~{bin_ms:.0f} ms)")
    ax.set_ylabel("ROIs (most active at top)")
    ax.set_title("Onset raster")
    p = figures_dir / "raster.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(str(p))

    fig = plt.Figure(figsize=(9, 4), tight_layout=True)
    ax = fig.add_subplot(111)
    duration_s = float(T) / float(fps) if fps > 0 else 1.0
    try:
        ed_utils.plot_event_detection(data["diagnostics"], ax=ax)
        ev = data["event_windows"]
        if ev is not None and len(ev) > 0:
            ed_utils.shade_event_windows(ax, ev, color="C1", alpha=0.20)
        ax.set_xlim(0.0, duration_s)
        ax.relim(); ax.autoscale_view(scalex=False, scaley=True)
    except Exception as e:
        ax.text(0.5, 0.5, f"plot_event_detection error:\n{e}",
                ha="center", va="center", transform=ax.transAxes)
    p = figures_dir / "event_detection.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(str(p))

    return written
