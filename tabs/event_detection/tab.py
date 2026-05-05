"""calliope.tabs.event_detection.tab - Tab 5: Event detection.

Three panels driven by ``utils.detect_event_windows``:
    1. Sorted lowpass dF/F heatmap.
    2. Sorted event raster (per-ROI hysteresis onsets).
    3. plot_event_detection diagnostics with detected event windows.

Reads plane0/r0p7_filtered_dff_lowpass.memmap.float32 +
r0p7_filtered_dff_dt.memmap.float32 produced by tab 4.

"Save Data..." on panels 1 & 2 exports a per-frame CSV (rows = frames,
columns = ROIs) using the full-resolution lowpass / onset arrays - not
the downsampled display matrices.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .logic import summary_writer

from ...gui_common import AppState, attach_fig_toolbar, open_advanced, spec_defaults


class EventDetectionTab(ttk.Frame):
    """Three panels:
        1. Sorted lowpass dF/F heatmap (mirrors image_all overview_heatmap).
        2. Sorted event raster (per-ROI hysteresis onsets, 1 px per onset).
        3. plot_event_detection from utils - population event-window
           detection on the per-ROI onset stream.
    """

    POLL_MS = 80
    Z_ENTER = 3.5
    Z_EXIT = 1.5
    MIN_SEP_S = 0.1
    TIME_COLS_TARGET = 1200

    PARAM_SPEC: list = [
        # Per-ROI onset hysteresis (mad_z + hysteresis_onsets)
        {"name": "z_enter", "label": "Hysteresis enter (z)",
         "type": "float", "default": 3.5, "group": "Per-ROI hysteresis",
         "help": "robust z threshold for entering an event"},
        {"name": "z_exit", "label": "Hysteresis exit (z)",
         "type": "float", "default": 1.5, "group": "Per-ROI hysteresis",
         "help": "lower threshold to exit (hysteresis)"},
        {"name": "min_sep_s", "label": "Min separation (s)",
         "type": "float", "default": 0.1, "group": "Per-ROI hysteresis",
         "help": "merge onsets closer than this into one event"},
        # Display
        {"name": "time_cols_target", "label": "Heatmap time columns",
         "type": "int", "default": 1200, "group": "Display"},
        # Population event detection (utils.EventDetectionParams).
        # NOTE (short-event rework, 2026-05-01): defaults retuned to match
        # the new short-duration detector. OLD values shown in `help` text
        # so they're easy to revert from the Advanced... dialog.
        {"name": "bin_sec", "label": "bin_sec",
         "type": "float", "default": 0.025,
         "group": "Population events - density",
         "help": "histogram bin width for the onset density (s). OLD=0.05"},
        {"name": "smooth_sigma_bins", "label": "smooth_sigma_bins",
         "type": "float", "default": 1.5,
         "group": "Population events - density",
         "help": "Gaussian smoothing sigma in bins. OLD=2.0"},
        {"name": "normalize_by_num_rois",
         "label": "normalize_by_num_rois",
         "type": "bool", "default": True,
         "group": "Population events - density"},
        {"name": "min_prominence", "label": "min_prominence",
         "type": "float", "default": 0.002,
         "group": "Population events - peaks",
         "help": "OLD=0.007"},
        {"name": "min_width_bins", "label": "min_width_bins",
         "type": "float", "default": 1.0,
         "group": "Population events - peaks",
         "help": "OLD=2.0"},
        {"name": "min_distance_bins", "label": "min_distance_bins",
         "type": "float", "default": 4.0,
         "group": "Population events - peaks",
         "help": "OLD=3.0; ~100 ms at bin_sec=0.025"},
        {"name": "prominence_wlen_s", "label": "prominence_wlen_s",
         "type": "float", "default": 1.0,
         "group": "Population events - peaks",
         "help": "NEW: local window (s) for find_peaks prominence"},
        {"name": "baseline_mode", "label": "baseline_mode",
         "type": "choice", "choices": ["rolling", "global"],
         "default": "rolling", "group": "Population events - baseline"},
        {"name": "baseline_percentile", "label": "baseline_percentile",
         "type": "float", "default": 5.0,
         "group": "Population events - baseline"},
        {"name": "baseline_window_s", "label": "baseline_window_s",
         "type": "float", "default": 5.0,
         "group": "Population events - baseline",
         "help": "rolling-mode window length. OLD=30.0"},
        {"name": "noise_quiet_percentile",
         "label": "noise_quiet_percentile",
         "type": "float", "default": 40.0,
         "group": "Population events - baseline"},
        {"name": "noise_mad_factor", "label": "noise_mad_factor",
         "type": "float", "default": 1.4826,
         "group": "Population events - baseline"},
        {"name": "end_threshold_k", "label": "end_threshold_k",
         "type": "float", "default": 1.5,
         "group": "Population events - boundaries",
         "help": "end = baseline + k * noise. OLD=2.0"},
        {"name": "max_walk_duration_s",
         "label": "max_walk_duration_s",
         "type": "float", "default": 2.0,
         "group": "Population events - boundaries",
         "help": "NEW: cap on one-sided walk distance from peak (s)"},
        {"name": "max_event_duration_s",
         "label": "max_event_duration_s",
         "type": "float", "default": 0.5,
         "group": "Population events - boundaries",
         "help": "Hard physiological cap on FINAL event duration (s). "
                 "OLD=10.0 (was walk distance, now final cap)"},
        {"name": "enable_watershed_split",
         "label": "enable_watershed_split",
         "type": "bool", "default": True,
         "group": "Population events - boundaries",
         "help": "NEW: split overlapping windows at the local minimum "
                 "between consecutive peaks"},
        {"name": "enforce_symmetric_clamp",
         "label": "enforce_symmetric_clamp",
         "type": "bool", "default": False,
         "group": "Population events - boundaries",
         "help": "NEW: force every window to peak +/- max/2"},
        {"name": "merge_gap_s", "label": "merge_gap_s",
         "type": "float", "default": 0.0,
         "group": "Population events - boundaries",
         "help": "(DISABLED in code by enable_watershed_split=True)"},
        {"name": "use_gaussian_boundary",
         "label": "use_gaussian_boundary",
         "type": "bool", "default": False,
         "group": "Population events - gaussian fit",
         "help": "OLD=True; replaced by hard duration cap"},
        {"name": "gaussian_quantile", "label": "gaussian_quantile",
         "type": "float", "default": 0.99,
         "group": "Population events - gaussian fit"},
        {"name": "gaussian_fit_pad_s", "label": "gaussian_fit_pad_s",
         "type": "float", "default": 0.5,
         "group": "Population events - gaussian fit"},
        {"name": "gaussian_min_sigma_s",
         "label": "gaussian_min_sigma_s",
         "type": "float", "default": 0.05,
         "group": "Population events - gaussian fit"},
    ]

    def __init__(self, master, state: AppState) -> None:
        super().__init__(master, padding=10)
        self.state = state
        self._plane0: Optional[Path] = None
        self._render_queue: queue.Queue = queue.Queue()
        self._render_worker: Optional[threading.Thread] = None
        self._params: dict = spec_defaults(self.PARAM_SPEC)
        self._last_data: Optional[dict] = None

        self._build_ui()
        self.after(self.POLL_MS, self._drain_render_queue)
        state.subscribe_plane0(self._on_plane0)
        state.subscribe_lowpass_ready(self._on_plane0)
        if state.lowpass_plane0 is not None:
            self._on_plane0(state.lowpass_plane0)
        elif state.plane0 is not None:
            self._on_plane0(state.plane0)

    # -- UI -----------------------------------------------------------------

    def _build_ui(self) -> None:
        header = ttk.LabelFrame(
            self, text="Event detection (uses filtered lowpass + derivative)",
            padding=8)
        header.pack(fill="x", pady=(0, 6))

        row = ttk.Frame(header); row.pack(fill="x", pady=2)
        self.render_btn = ttk.Button(
            row, text="Render", command=self._on_render,
            state="disabled")
        self.render_btn.pack(side="left")
        self.render_progress = ttk.Progressbar(
            row, mode="indeterminate", length=160)
        self.render_progress.pack(side="left", padx=12)
        self.status_var = tk.StringVar(
            value="Compute lowpass + derivative on tab 4 first.")
        ttk.Label(row, textvariable=self.status_var,
                  font=("", 9, "italic")).pack(side="left")
        ttk.Button(row, text="Advanced...",
                   command=self._on_advanced).pack(side="right",
                                                   padx=(0, 6))
        self.summary_btn = ttk.Button(
            row, text="Save summary",
            command=self._on_save_summary, state="disabled")
        self.summary_btn.pack(side="right", padx=(0, 6))
        ttk.Button(row, text="Reload from folder...",
                   command=self._reload_from_folder).pack(side="right")

        body = ttk.Frame(self); body.pack(fill="both", expand=True)
        # Panel 3 is the interactive event-detection trace and gets the
        # bulk of the height + a navigation toolbar (mirrors plt.show()).
        body.rowconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)
        body.rowconfigure(2, weight=4)
        body.columnconfigure(0, weight=1)

        f1 = ttk.LabelFrame(
            body, text="1. Filtered lowpass dF/F heatmap "
                       "(sorted by event count)", padding=4)
        f1.grid(row=0, column=0, sticky="nsew", pady=(0, 4))
        self.hm_fig = plt.Figure(figsize=(8, 1.6), tight_layout=True)
        self.hm_ax = self.hm_fig.add_subplot(111)
        self.hm_ax.set_axis_off()
        self.hm_ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=self.hm_ax.transAxes)
        self.hm_canvas = FigureCanvasTkAgg(self.hm_fig, master=f1)
        attach_fig_toolbar(
            self.hm_canvas, f1, data_basename="event_heatmap",
            data_provider=lambda p: self._save_panel_csv(p, "heatmap"))
        self.hm_canvas.get_tk_widget().pack(fill="both", expand=True)

        f2 = ttk.LabelFrame(
            body, text="2. Filtered event raster (sorted by event count)",
            padding=4)
        f2.grid(row=1, column=0, sticky="nsew", pady=(0, 4))
        self.er_fig = plt.Figure(figsize=(8, 1.6), tight_layout=True)
        self.er_ax = self.er_fig.add_subplot(111)
        self.er_ax.set_axis_off()
        self.er_ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=self.er_ax.transAxes)
        self.er_canvas = FigureCanvasTkAgg(self.er_fig, master=f2)
        attach_fig_toolbar(
            self.er_canvas, f2, data_basename="event_raster",
            data_provider=lambda p: self._save_panel_csv(p, "raster"))
        self.er_canvas.get_tk_widget().pack(fill="both", expand=True)

        f3 = ttk.LabelFrame(
            body, text="3. Population event detection "
                       "(utils.plot_event_detection)", padding=4)
        f3.grid(row=2, column=0, sticky="nsew")
        self.ed_fig = plt.Figure(figsize=(8, 4.8), tight_layout=True)
        self.ed_ax = self.ed_fig.add_subplot(111)
        self.ed_ax.set_axis_off()
        self.ed_ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=self.ed_ax.transAxes)
        self.ed_canvas = FigureCanvasTkAgg(self.ed_fig, master=f3)
        self.ed_toolbar = attach_fig_toolbar(self.ed_canvas, f3)
        self.ed_canvas.get_tk_widget().pack(side="top", fill="both",
                                            expand=True)

    # -- Plane0 / readiness handling ---------------------------------------

    def _reload_from_folder(self) -> None:
        path = filedialog.askdirectory(
            title="Select a suite2p plane0 folder containing the "
                  "r0p7_filtered_dff_lowpass / _dt memmaps")
        if not path:
            return
        self._on_plane0(Path(path))

    def _on_plane0(self, plane0: Path) -> None:
        self._plane0 = Path(plane0)
        if self._inputs_ready(self._plane0):
            self.render_btn.config(state="normal")
            self.status_var.set(
                f"Ready. Click Render to compute and plot.  "
                f"({self._plane0})")
        else:
            self.render_btn.config(state="disabled")
            self.status_var.set(
                "Filtered lowpass / derivative memmaps not present yet. "
                "Compute them on tab 4 first.")

    def _inputs_ready(self, plane0: Path) -> bool:
        for name in ("F.npy", "r0p7_filtered_dff_lowpass.memmap.float32",
                     "r0p7_filtered_dff_dt.memmap.float32"):
            if not (plane0 / name).exists():
                return False
        return True

    # -- Advanced parameters ------------------------------------------------

    def _on_advanced(self) -> None:
        if open_advanced(
                self, "Event detection - Advanced parameters",
                self.PARAM_SPEC, self._params):
            self.status_var.set(
                "Advanced parameters updated. Click Render to recompute "
                "with the new settings.")

    # -- Render worker ------------------------------------------------------

    def _on_render(self) -> None:
        if (self._render_worker is not None
                and self._render_worker.is_alive()):
            messagebox.showinfo("Busy", "Render is already running.")
            return
        if self._plane0 is None or not self._inputs_ready(self._plane0):
            messagebox.showerror(
                "Not ready",
                "Compute lowpass + derivative on tab 4 first.")
            return

        plane0 = self._plane0
        self.render_btn.config(state="disabled")
        self.render_progress.start(12)
        self.status_var.set("Rendering ...")

        def worker():
            try:
                payload = self._compute_render_data(plane0)
                self._render_queue.put(("done", payload))
            except Exception as e:
                self._render_queue.put(
                    ("error", f"{e}\n{traceback.format_exc()}"))

        self._render_worker = threading.Thread(target=worker, daemon=True)
        self._render_worker.start()

    def _compute_render_data(self, plane0: Path) -> dict:
        from . import logic as utils
        F = np.load(plane0 / "F.npy", mmap_mode="r")
        N_total, T = F.shape

        mask_path = plane0 / "predicted_cell_mask.npy"
        if mask_path.exists():
            mask = np.load(mask_path).astype(bool)
        else:
            iscell_path = plane0 / "iscell.npy"
            if iscell_path.exists():
                ic = np.load(iscell_path)
                mask = ((ic[:, 0] > 0) if ic.ndim == 2
                        else (ic > 0)).astype(bool)
            else:
                mask = np.ones(N_total, dtype=bool)
        N_kept = int(mask.sum())
        if N_kept == 0:
            raise RuntimeError("No ROIs survive the cell-filter mask.")

        lp_path = plane0 / "r0p7_filtered_dff_lowpass.memmap.float32"
        dt_path = plane0 / "r0p7_filtered_dff_dt.memmap.float32"
        lowpass = np.memmap(str(lp_path), dtype="float32", mode="r",
                            shape=(T, N_kept))
        derivative = np.memmap(str(dt_path), dtype="float32", mode="r",
                               shape=(T, N_kept))

        fps = float(utils.get_fps_from_notes(str(plane0)))

        z_enter = float(self._params.get("z_enter", 3.5))
        z_exit = float(self._params.get("z_exit", 1.5))
        min_sep_s = float(self._params.get("min_sep_s", 0.1))
        time_cols_target = int(
            self._params.get("time_cols_target", 1200))

        downsample = max(1, T // time_cols_target)
        num_cols = T // downsample
        heatmap = np.zeros((N_kept, num_cols), dtype=np.uint8)
        raster = np.zeros((N_kept, num_cols), dtype=np.uint8)
        event_counts = np.zeros(N_kept, dtype=np.int32)
        onsets_by_roi: list = []

        for i in range(N_kept):
            lp_i = np.asarray(lowpass[:, i], dtype=np.float32)
            dt_i = np.asarray(derivative[:, i], dtype=np.float32)
            z, _, _ = utils.mad_z(dt_i)
            onsets = utils.hysteresis_onsets(
                z, z_enter, z_exit, fps, min_sep_s=min_sep_s)
            event_counts[i] = onsets.size
            onsets_by_roi.append(
                np.asarray(onsets, dtype=np.float64) / fps)

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

        order = np.argsort(-event_counts)
        heatmap = heatmap[order]
        raster = raster[order]

        # All 18 PARAM_SPEC knobs map 1:1 to EventDetectionParams fields.
        ed_kwargs: dict = {}
        try:
            from .logic import EventDetectionParams
            params_obj = EventDetectionParams()
            for field_name in (
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
            ):
                if field_name in self._params:
                    setattr(params_obj, field_name,
                            self._params[field_name])
            ed_kwargs["params"] = params_obj
        except Exception as e:
            print(f"[GUI] EventDetectionParams build failed: {e}")

        event_windows, A, first_time, diagnostics = \
            utils.detect_event_windows(
                onsets_by_roi, T=T, fps=fps,
                return_diagnostics=True, **ed_kwargs,
            )

        return {
            "heatmap": heatmap,
            "raster": raster,
            "kept_idx": np.flatnonzero(mask),
            "event_counts": event_counts,
            "diagnostics": diagnostics,
            "event_windows": event_windows,
            "onsets_by_roi": onsets_by_roi,
            "fps": fps,
            "T": T,
            "N_kept": N_kept,
            "downsample": downsample,
        }

    def _drain_render_queue(self) -> None:
        try:
            while True:
                kind, payload = self._render_queue.get_nowait()
                if kind == "done":
                    self._on_render_done(payload)
                elif kind == "error":
                    self.render_progress.stop()
                    self.render_btn.config(state="normal")
                    self.status_var.set("Render error.")
                    messagebox.showerror(
                        "Render failed", payload.split("\n", 1)[0])
        except queue.Empty:
            pass
        self.after(self.POLL_MS, self._drain_render_queue)

    def _on_render_done(self, data: dict) -> None:
        from . import logic as utils
        self.render_progress.stop()
        self.render_btn.config(state="normal")

        hm = data["heatmap"]
        rs = data["raster"]
        N = data["N_kept"]; T = data["T"]
        ds = data["downsample"]; fps = data["fps"]

        ax = self.hm_ax
        ax.clear(); ax.set_axis_on()
        ax.imshow(hm, aspect="auto", interpolation="nearest")
        ax.set_xlabel(f"Time (downsampled bins ~{ds * 1000.0 / fps:.0f} ms)",
                      fontsize=8)
        ax.set_ylabel("ROIs (most active at top)", fontsize=8)
        ax.tick_params(labelsize=7)
        self.hm_canvas.draw_idle()

        ax = self.er_ax
        ax.clear(); ax.set_axis_on()
        ax.imshow(rs, aspect="auto", interpolation="nearest", cmap="Greys")
        ax.set_xlabel(f"Time (downsampled bins ~{ds * 1000.0 / fps:.0f} ms)",
                      fontsize=8)
        ax.set_ylabel("ROIs (most active at top)", fontsize=8)
        ax.tick_params(labelsize=7)
        self.er_canvas.draw_idle()

        ax = self.ed_ax
        ax.clear(); ax.set_axis_on()
        duration_s = float(T) / float(fps) if fps > 0 else 1.0
        try:
            utils.plot_event_detection(data["diagnostics"], ax=ax)
            ev = data["event_windows"]
            if ev is not None and len(ev) > 0:
                utils.shade_event_windows(ax, ev, color="C1", alpha=0.20)
            # Force the time axis to span the full recording so the trace
            # doesn't look cut off when the auto-fit picks up only the
            # smoothed-density extent.
            ax.set_xlim(0.0, duration_s)
            ax.relim(); ax.autoscale_view(scalex=False, scaley=True)
        except Exception as e:
            ax.text(0.5, 0.5, f"plot_event_detection error:\n{e}",
                    ha="center", va="center", transform=ax.transAxes)
        self.ed_canvas.draw_idle()
        # Reset the toolbar's home/back/forward stack so 'home' returns
        # to this fresh view (not the empty placeholder it captured at
        # widget construction time).
        try:
            self.ed_toolbar.update()
        except Exception:
            pass

        n_events = (len(data["event_windows"])
                    if data["event_windows"] is not None else 0)
        self.status_var.set(
            f"Rendered  N_kept={N}  T={T}  fps={fps:.2f}  "
            f"duration={duration_s:.1f}s  events={n_events}")

        # Drop the display-only arrays from the cached payload now that
        # they live on the matplotlib figures. ``heatmap`` and ``raster``
        # are read once here and never again -- the heatmap CSV export
        # reloads the lowpass memmap fresh, and the raster CSV export
        # rebuilds 0/1 markers from ``onsets_by_roi``. Keeping them was
        # holding tens of MB per recording for no functional gain.
        data.pop("heatmap", None)
        data.pop("raster", None)
        self._last_data = data
        self.summary_btn.config(state="normal")
        if self._plane0 is not None:
            try:
                self._write_summary(self._plane0, data)
            except Exception as e:
                print(f"[GUI] event summary write failed: {e}")

    # -- Summary export -----------------------------------------------------

    def _write_summary(self, plane0: Path, data: dict) -> Path:
        summary_writer.update_recording_meta(
            plane0, fps=data.get("fps"),
            T=data.get("T"), N=data.get("N_kept"))
        return summary_writer.write_events_sheets(
            plane0,
            event_windows=data.get("event_windows"),
            onsets_by_roi=data.get("onsets_by_roi") or [],
            fps=data.get("fps"),
            in_seconds=True,
        )

    def _on_save_summary(self) -> None:
        if self._plane0 is None or self._last_data is None:
            messagebox.showinfo(
                "No data", "Click 'Render' first to compute event windows.")
            return
        try:
            path = self._write_summary(self._plane0, self._last_data)
            self.status_var.set(f"Summary -> {path}")
        except Exception as e:
            messagebox.showerror("Summary failed", str(e))

    # -- Plot data export ---------------------------------------------------

    def _save_panel_csv(self, parent, which: str) -> None:
        """Write a full-resolution per-frame CSV for the heatmap or raster:
        rows = video frames (no downsampling), columns = individual ROIs
        in original Suite2p index order, plus a leading ``time_s`` column
        equal to ``frame_index / fps``. ``which`` is "heatmap" or "raster".

        Heatmap values are the actual lowpass dF/F floats from
        ``r0p7_filtered_dff_lowpass.memmap.float32`` (not the 0-255
        normalised display version). Raster values are 0/1 onset markers
        rebuilt at frame resolution from ``onsets_by_roi``.
        """
        import pandas as pd
        if self._last_data is None or self._plane0 is None:
            messagebox.showinfo(
                "No data", "Click 'Render' first to compute panel data.",
                parent=parent)
            return
        data = self._last_data
        kept_idx = data.get("kept_idx")
        fps = float(data.get("fps") or 0.0)
        T = int(data.get("T") or 0)
        N_kept = int(data.get("N_kept") or 0)
        if kept_idx is None or fps <= 0 or T <= 0 or N_kept <= 0:
            messagebox.showerror(
                "Export failed",
                "Missing kept_idx / fps / T / N_kept in last_data; "
                "rerun Render.", parent=parent)
            return

        if which == "heatmap":
            lp_path = (self._plane0
                       / "r0p7_filtered_dff_lowpass.memmap.float32")
            if not lp_path.exists():
                messagebox.showerror(
                    "Export failed",
                    f"Missing {lp_path.name}; recompute on Tab 4.",
                    parent=parent)
                return
            lowpass = np.memmap(str(lp_path), dtype="float32", mode="r",
                                shape=(T, N_kept))
            wide = np.asarray(lowpass)
            value_label = "lp_dff"
            default_basename = "event_heatmap"
        elif which == "raster":
            onsets_by_roi = data.get("onsets_by_roi") or []
            wide = np.zeros((T, N_kept), dtype=np.uint8)
            for i, onsets_s in enumerate(onsets_by_roi[:N_kept]):
                if onsets_s is None:
                    continue
                arr = np.asarray(onsets_s, dtype=np.float64)
                if arr.size == 0:
                    continue
                frames = np.clip(np.round(arr * fps).astype(np.int64),
                                 0, T - 1)
                wide[frames, i] = 1
            value_label = "onset_binary"
            default_basename = "event_raster"
        else:
            raise ValueError(f"unknown panel {which!r}")

        roi_columns = [f"roi_{int(i)}"
                       for i in np.asarray(kept_idx)[:N_kept]]
        df = pd.DataFrame(wide, columns=roi_columns)
        df.insert(0, "time_s",
                  np.arange(T, dtype=np.float64) / fps)

        target = filedialog.asksaveasfilename(
            parent=parent, defaultextension=".csv",
            initialfile=f"{default_basename}.csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
            title=f"Save {which} as CSV "
                  f"(rows = frames, columns = ROIs)")
        if not target:
            return
        # Header row stamps the frame->time conversion so downstream
        # scripts don't have to parse the time_s column.
        with open(target, "w", encoding="utf-8", newline="") as fh:
            fh.write(
                f"# {which} ({value_label}); "
                f"fps={fps:.6f}, frame_seconds={1.0 / fps:.6f}, "
                f"n_frames={T}, n_rois_kept={N_kept}\n")
            df.to_csv(fh, index=False)
        messagebox.showinfo(
            "Saved", f"Wrote {Path(target).name}", parent=parent)
