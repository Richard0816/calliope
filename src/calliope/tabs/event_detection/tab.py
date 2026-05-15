"""calliope.tabs.event_detection.tab - Tab 5: Event detection.

What this tab does
------------------
This is the analytical heart of the pipeline -- the bridge from
"continuous calcium traces" to "discrete events that a
neurophysiologist would talk about". For each kept ROI it detects:

* **Per-ROI onsets** -- moments when that one cell starts firing,
  using MAD-z thresholding + hysteresis on the smoothed first
  derivative.
* **Population events** -- moments when many cells fire together,
  detected as peaks in the histogram of all per-ROI onsets after
  Gaussian smoothing. Each peak gets walked outward to a baseline-
  noise cutoff; overlapping windows are watershed-split; everything
  is hard-clamped to a maximum duration (epileptiform events are
  short).

What the user sees
------------------
Three stacked panels, all updating after each Render:
    1. dF/F heatmap, ROIs sorted by event count (most active on top).
    2. Event raster, same sort order, one dot per onset.
    3. The smoothed onset-density curve with detected event windows
       shaded -- this is the ``plot_event_detection`` diagnostic
       view.

Header controls let you toggle a manual ROI subset (Suite2p ids
intersected with the cell-filter keep mask) and pop the Advanced
dialog for every per-ROI hysteresis + density-detector knob.

What gets published
-------------------
On every successful render the tab:

* Saves the EventWindows / EventOnsets / RoiEventTimes sheets to
  ``calliope_summary.xlsx`` via ``summary_writer.write_events_sheets``.
* Publishes the in-memory results
  (event_windows, A, first_time, kept_idx, fps, T, plane0,
  onsets_by_roi) onto ``AppState.event_results`` so Tabs 7 and 8
  can refresh automatically without re-detecting.

Reads
-----
``plane0/r0p7_filtered_dff_lowpass.memmap.float32``
``plane0/r0p7_filtered_dff_dt.memmap.float32``
both produced by Tab 4.

"Save Data..." on panels 1 & 2 exports a per-frame CSV (rows = frames,
columns = ROIs) using the full-resolution lowpass / onset arrays - not
the downsampled display matrices.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
import traceback

import customtkinter as ctk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .logic import summary_writer

from ...gui_common import (
    AppState, attach_fig_toolbar, attach_resize_handle, drain_queue,
    format_roi_indices, open_advanced, parse_manual_roi_spec,
    spec_defaults,
)


class EventDetectionTab(ctk.CTkFrame):
    """Tab 5: per-ROI onsets + population events.

    Three panels:
        1. Sorted lowpass dF/F heatmap (mirrors image_all overview_heatmap).
        2. Sorted event raster (per-ROI hysteresis onsets, 1 px per onset).
        3. plot_event_detection from utils - population event-window
           detection on the per-ROI onset stream.

    Class attributes
    ----------------
    ``POLL_MS``
        Render-queue polling interval (main thread).
    ``Z_ENTER`` / ``Z_EXIT`` / ``MIN_SEP_S``
        Default per-ROI hysteresis-onset thresholds (also exposed in
        PARAM_SPEC).
    ``TIME_COLS_TARGET``
        Target column count for the down-sampled heatmap so the
        figure stays interactive for long recordings.
    ``PARAM_SPEC``
        The 26-entry knob list driving the Advanced... dialog --
        every per-ROI threshold + every ``EventDetectionParams``
        field. See ``calliope.core.utils.EventDetectionParams`` for
        what each one does mathematically.

    Lifecycle
    ---------
    ``__init__``
        Build widgets, prime ``_params`` from PARAM_SPEC defaults,
        subscribe to ``state.subscribe_plane0`` and
        ``state.subscribe_lowpass_ready`` so the tab unlocks once
        the upstream memmaps are written.
    ``_on_render``
        Snapshot the manual-subset Tk vars on the main thread (Tk
        vars are not safe to read from a worker), then launch a
        worker thread.
    ``_compute_render_data`` (worker thread)
        Load the lowpass + dt memmaps, run ``mad_z`` +
        ``hysteresis_onsets`` per ROI, run
        ``utils.detect_event_windows`` over all onsets, return a
        dict of arrays.
    ``_on_render_done`` (main thread)
        Update the three figures, write summary sheets, publish
        ``state.set_event_results(...)``.
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
         "type": "int", "default": 1200, "group": "Display",
         "help": "approximate target column count for the downsampled "
                 "heatmap and raster"},
        # Population event detection (utils.EventDetectionParams).
        {"name": "bin_sec", "label": "Density bin (s)",
         "type": "float", "default": 0.05,
         "group": "Population events - density",
         "help": "histogram bin width for the onset density"},
        {"name": "smooth_sigma_bins", "label": "Smoothing sigma (bins)",
         "type": "float", "default": 1.5,
         "group": "Population events - density",
         "help": "Gaussian smoothing sigma applied to the binned density"},
        {"name": "normalize_by_num_rois",
         "label": "Normalize by ROI count",
         "type": "bool", "default": True,
         "group": "Population events - density",
         "help": "scale density to per-ROI rate so recordings with "
                 "different ROI counts are comparable"},
        {"name": "min_prominence", "label": "Min peak prominence",
         "type": "float", "default": 0.002,
         "group": "Population events - peaks",
         "help": "find_peaks prominence floor on the smoothed density"},
        {"name": "min_width_bins", "label": "Min peak width (bins)",
         "type": "float", "default": 1.0,
         "group": "Population events - peaks",
         "help": "find_peaks minimum width in density bins"},
        {"name": "min_distance_bins", "label": "Min peak separation (bins)",
         "type": "float", "default": 4.0,
         "group": "Population events - peaks",
         "help": "minimum bin distance between adjacent peaks "
                 "(~200 ms at default bin_sec=0.05)"},
        {"name": "prominence_wlen_s", "label": "Prominence window (s)",
         "type": "float", "default": 1.0,
         "group": "Population events - peaks",
         "help": "local window for find_peaks prominence"},
        {"name": "min_active_rois",
         "label": "Min active ROIs per event",
         "type": "int", "default": 3,
         "group": "Population events - peaks",
         "help": "drop events with fewer than this many participating "
                 "ROIs; default 3 excludes 2-ROI noise events. Raise "
                 "to 4-5 to also exclude 3-4-ROI events."},
        {"name": "baseline_mode", "label": "Baseline mode",
         "type": "choice", "choices": ["rolling", "global"],
         "default": "rolling", "group": "Population events - baseline",
         "help": "rolling = per-bin percentile over baseline_window_s; "
                 "global = scalar percentile of the whole density"},
        {"name": "baseline_percentile", "label": "Baseline percentile",
         "type": "float", "default": 5.0,
         "group": "Population events - baseline",
         "help": "percentile used to estimate the density baseline"},
        {"name": "baseline_window_s", "label": "Rolling window (s)",
         "type": "float", "default": 5.0,
         "group": "Population events - baseline",
         "help": "rolling-mode window length"},
        {"name": "noise_quiet_percentile",
         "label": "Quiet percentile (noise)",
         "type": "float", "default": 40.0,
         "group": "Population events - baseline",
         "help": "values below this percentile feed the MAD noise estimate"},
        {"name": "noise_mad_factor", "label": "MAD -> sigma factor",
         "type": "float", "default": 1.4826,
         "group": "Population events - baseline",
         "help": "MAD scaling so noise estimates approach a Gaussian sigma"},
        {"name": "end_threshold_k", "label": "End threshold k",
         "type": "float", "default": 1.5,
         "group": "Population events - boundaries",
         "help": "boundary = baseline + k * noise"},
        {"name": "max_walk_duration_s",
         "label": "Max walk duration (s)",
         "type": "float", "default": 2.0,
         "group": "Population events - boundaries",
         "help": "cap on one-sided walk distance from each peak"},
        {"name": "max_event_duration_s",
         "label": "Max event duration (s)",
         "type": "float", "default": 1.3,
         "group": "Population events - boundaries",
         "help": "hard physiological cap on the final event duration"},
        {"name": "enable_watershed_split",
         "label": "Watershed-split overlaps",
         "type": "bool", "default": True,
         "group": "Population events - boundaries",
         "help": "split overlapping windows at the local minimum between "
                 "consecutive peaks"},
        {"name": "enforce_symmetric_clamp",
         "label": "Force symmetric clamp",
         "type": "bool", "default": False,
         "group": "Population events - boundaries",
         "help": "force every window to peak +/- max_event_duration_s/2"},
        {"name": "merge_gap_s", "label": "Merge gap (s)",
         "type": "float", "default": 0.0,
         "group": "Population events - boundaries",
         "help": "legacy overlap-merge gap; ignored when watershed-split "
                 "is enabled"},
        {"name": "use_gaussian_boundary",
         "label": "Use Gaussian-fit boundary",
         "type": "bool", "default": False,
         "group": "Population events - gaussian fit",
         "help": "fit a Gaussian to each peak and use its quantile cut "
                 "instead of the walked boundary"},
        {"name": "gaussian_quantile", "label": "Gaussian quantile",
         "type": "float", "default": 0.99,
         "group": "Population events - gaussian fit",
         "help": "quantile of the Gaussian fit used to set the boundary"},
        {"name": "gaussian_fit_pad_s", "label": "Gaussian fit pad (s)",
         "type": "float", "default": 0.5,
         "group": "Population events - gaussian fit",
         "help": "padding around each peak included in the Gaussian fit"},
        {"name": "gaussian_min_sigma_s",
         "label": "Gaussian min sigma (s)",
         "type": "float", "default": 0.05,
         "group": "Population events - gaussian fit",
         "help": "lower bound on the Gaussian fit's sigma"},
    ]

    def __init__(self, master, state: AppState) -> None:
        super().__init__(master, fg_color="transparent")
        self.state = state
        self._plane0: Optional[Path] = None
        self._render_queue: queue.Queue = queue.Queue()
        self._render_worker: Optional[threading.Thread] = None
        self._params: dict = spec_defaults(self.PARAM_SPEC)
        self._last_data: Optional[dict] = None
        # Single-instance reference to the prominence-distribution
        # popout (Tab 5 only allows one at a time per tab instance).
        self._prominence_popout = None
        # Onset source: 'derivative' (default, MAD-z + hysteresis on
        # the SG derivative of the lowpass dF/F) or 'spks' (MAD-z +
        # hysteresis on Suite2p's deconvolved spike trace from
        # ``spks.npy``). The radio buttons in the header drive this.
        self.onset_source_var = tk.StringVar(value="derivative")

        self._build_ui()
        drain_queue(self, self._render_queue,
                    {"done": self._on_render_done,
                     "error": self._on_render_error},
                    poll_ms=self.POLL_MS)
        state.subscribe_plane0(self._on_plane0)
        state.subscribe_lowpass_ready(self._on_plane0)
        if state.lowpass_plane0 is not None:
            self._on_plane0(state.lowpass_plane0)
        elif state.plane0 is not None:
            self._on_plane0(state.plane0)

    # -- UI -----------------------------------------------------------------

    def _build_ui(self) -> None:
        header = ctk.CTkFrame(self)
        header.pack(fill="x", padx=10, pady=(10, 6))
        ctk.CTkLabel(
            header,
            text="Event detection (uses filtered lowpass + derivative)",
            font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(6, 0))

        # Persistent recording header. ``status_var`` below carries
        # transient state messages ("Loading...", "Ready", etc.);
        # this label keeps the recording id + plane0 visible at all
        # times so the user can tell which dataset the panels show.
        rec_row = ctk.CTkFrame(header, fg_color="transparent")
        rec_row.pack(fill="x", padx=8, pady=(0, 4))
        self.recording_var = tk.StringVar(value="No recording loaded.")
        ctk.CTkLabel(rec_row, textvariable=self.recording_var,
                     font=ctk.CTkFont(size=12, slant="italic")).pack(
            side="left")

        row = ctk.CTkFrame(header, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=2)
        self.render_btn = ctk.CTkButton(
            row, text="Render", width=100,
            command=self._on_render, state="disabled")
        self.render_btn.pack(side="left")
        self.render_progress = ctk.CTkProgressBar(
            row, mode="indeterminate", width=160)
        self.render_progress.pack(side="left", padx=12)
        self.status_var = tk.StringVar(
            value="Compute lowpass + derivative on tab 4 first.")
        ctk.CTkLabel(row, textvariable=self.status_var,
                     font=ctk.CTkFont(size=11, slant="italic")).pack(
            side="left")
        ctk.CTkButton(row, text="Advanced...", width=100,
                      command=self._on_advanced).pack(side="right",
                                                      padx=(0, 6))
        self.prominence_btn = ctk.CTkButton(
            row, text="Prominence distribution...", width=200,
            command=self._on_open_prominence_popout, state="disabled")
        self.prominence_btn.pack(side="right", padx=(0, 6))
        self.summary_btn = ctk.CTkButton(
            row, text="Save summary", width=110,
            command=self._on_save_summary, state="disabled")
        self.summary_btn.pack(side="right", padx=(0, 6))
        ctk.CTkButton(row, text="Reload from folder...", width=160,
                      command=self._reload_from_folder).pack(side="right")

        # ---- Onset source row ----
        # Default uses MAD-z + hysteresis on the Savitzky-Golay
        # derivative of the lowpass dF/F (everything Tab 4 produced).
        # The "spks" alternative loads Suite2p's deconvolved
        # ``spks.npy`` and runs the same MAD-z + hysteresis on it.
        src_row = ctk.CTkFrame(header, fg_color="transparent")
        src_row.pack(fill="x", padx=8, pady=(4, 0))
        ctk.CTkLabel(src_row, text="Onset source:").pack(side="left")
        ctk.CTkRadioButton(
            src_row, text="Derivative (default; MAD-z hysteresis on dt)",
            value="derivative", variable=self.onset_source_var,
        ).pack(side="left", padx=(8, 12))
        ctk.CTkRadioButton(
            src_row, text="Suite2p spks (deconvolved)",
            value="spks", variable=self.onset_source_var,
        ).pack(side="left")
        ctk.CTkLabel(
            src_row,
            text="  -- spks mode reads <plane0>/spks.npy (one onset "
                 "per MAD-z hysteresis crossing on the spike trace).",
            text_color="gray", font=ctk.CTkFont(size=11, slant="italic"),
        ).pack(side="left", padx=(8, 0))

        # Manual ROI subset row: when enabled, only those Suite2p ROI ids
        # (intersected with the cell-filter keep mask) feed the heatmap,
        # raster, and population event detection.
        sub_row = ctk.CTkFrame(header, fg_color="transparent")
        sub_row.pack(fill="x", padx=8, pady=(4, 6))
        self.manual_subset_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            sub_row, text="Restrict to manual ROI subset (Suite2p ids):",
            variable=self.manual_subset_var,
        ).pack(side="left")
        self.manual_roi_var = tk.StringVar(value="")
        ctk.CTkEntry(sub_row, textvariable=self.manual_roi_var, width=240).pack(
            side="left", padx=(4, 4), fill="x", expand=True)
        ctk.CTkLabel(
            sub_row,
            text="(e.g. 0,3,5-9; ids outside the keep mask are skipped)",
            text_color="gray",
        ).pack(side="left")

        # Three stacked matplotlib panels, each in its own CTkFrame
        # wrap so the height is controllable. A draggable grip below
        # each panel lets the user grow that panel independently --
        # without squeezing its neighbours -- and the scrollable wrapper
        # extends the tab body to absorb the new height.
        f1_wrap = ctk.CTkFrame(self, fg_color="transparent")
        f1_wrap.pack(fill="x", padx=14, pady=(0, 0))
        f1 = ctk.CTkFrame(f1_wrap)
        f1.pack(fill="both", expand=True)
        ctk.CTkLabel(f1,
                     text="1. Filtered lowpass dF/F heatmap "
                          "(sorted by event count)",
                     font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(6, 0))
        self.hm_fig = plt.Figure(figsize=(8, 1.6), tight_layout=True)
        self.hm_ax = self.hm_fig.add_subplot(111)
        self.hm_ax.set_axis_off()
        self.hm_ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=self.hm_ax.transAxes)
        self.hm_canvas = FigureCanvasTkAgg(self.hm_fig, master=f1)
        attach_fig_toolbar(
            self.hm_canvas, f1, data_basename="event_heatmap",
            data_provider=lambda p: self._save_panel_csv(p, "heatmap"))
        self.hm_canvas.get_tk_widget().pack(fill="both", expand=True,
                                            padx=4, pady=(0, 4))
        attach_resize_handle(self, f1_wrap, min_height=140,
                             initial_height=180)

        f2_wrap = ctk.CTkFrame(self, fg_color="transparent")
        f2_wrap.pack(fill="x", padx=14, pady=(0, 0))
        f2 = ctk.CTkFrame(f2_wrap)
        f2.pack(fill="both", expand=True)
        ctk.CTkLabel(
            f2, text="2. Filtered event raster (sorted by event count)",
            font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(6, 0))
        self.er_fig = plt.Figure(figsize=(8, 1.6), tight_layout=True)
        self.er_ax = self.er_fig.add_subplot(111)
        self.er_ax.set_axis_off()
        self.er_ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=self.er_ax.transAxes)
        self.er_canvas = FigureCanvasTkAgg(self.er_fig, master=f2)
        attach_fig_toolbar(
            self.er_canvas, f2, data_basename="event_raster",
            data_provider=lambda p: self._save_panel_csv(p, "raster"))
        self.er_canvas.get_tk_widget().pack(fill="both", expand=True,
                                            padx=4, pady=(0, 4))
        attach_resize_handle(self, f2_wrap, min_height=140,
                             initial_height=180)

        f3_wrap = ctk.CTkFrame(self, fg_color="transparent")
        f3_wrap.pack(fill="x", padx=14, pady=(0, 10))
        f3 = ctk.CTkFrame(f3_wrap)
        f3.pack(fill="both", expand=True)
        ctk.CTkLabel(
            f3,
            text="3. Population event detection "
                 "(utils.plot_event_detection)",
            font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(6, 0))
        self.ed_fig = plt.Figure(figsize=(8, 4.8), tight_layout=True)
        self.ed_ax = self.ed_fig.add_subplot(111)
        self.ed_ax.set_axis_off()
        self.ed_ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=self.ed_ax.transAxes)
        self.ed_canvas = FigureCanvasTkAgg(self.ed_fig, master=f3)
        self.ed_toolbar = attach_fig_toolbar(self.ed_canvas, f3)
        self.ed_canvas.get_tk_widget().pack(side="top", fill="both",
                                            expand=True,
                                            padx=4, pady=(0, 4))
        attach_resize_handle(self, f3_wrap, min_height=240,
                             initial_height=380)

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
        from ...core.utils import infer_recording_id
        try:
            rec_id = infer_recording_id(self._plane0)
        except Exception:
            rec_id = self._plane0.parent.name
        self.recording_var.set(
            f"Recording: {rec_id}    plane0: {self._plane0}")
        if self._inputs_ready(self._plane0):
            self.render_btn.configure(state="normal")
            self.status_var.set(
                "Ready. Click Render to compute and plot.")
        else:
            self.render_btn.configure(state="disabled")
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

    # -- Prominence-distribution popout -------------------------------------

    def _on_open_prominence_popout(self) -> None:
        """Open (or focus) the prominence-distribution popout.

        Pulls the cached smoothed onset density from the most recent
        render's diagnostics, computes every candidate peak's
        prominence (find_peaks with the same width/distance/wlen as
        detection but prominence floor lifted to ~0), and shows the
        full distribution with a draggable threshold line.
        """
        existing = getattr(self, "_prominence_popout", None)
        if existing is not None and existing.winfo_exists():
            existing.lift()
            existing.focus_set()
            return
        if self._last_data is None:
            messagebox.showinfo(
                "No data",
                "Click Render at least once before opening the "
                "prominence-distribution popout.")
            return
        diagnostics = self._last_data.get("diagnostics") or {}
        smoothed = diagnostics.get("smoothed_density")
        if smoothed is None or len(smoothed) == 0:
            messagebox.showinfo(
                "No density",
                "The last render did not produce a smoothed density "
                "trace. Re-render Tab 5 and try again.")
            return

        from ...core.utils import (
            EventDetectionParams, compute_candidate_prominences,
        )
        # Mirror the field set in EventDetectionParams; ignore any
        # PARAM_SPEC keys that aren't part of the dataclass so the
        # constructor doesn't raise on UI-only knobs (manual_subset_*,
        # onset_source, ...).
        edp_fields = {f for f in EventDetectionParams.__dataclass_fields__}
        edp_kwargs = {k: v for k, v in self._params.items()
                      if k in edp_fields}
        try:
            params = EventDetectionParams(**edp_kwargs)
        except TypeError:
            params = EventDetectionParams()
        proms = compute_candidate_prominences(np.asarray(smoothed),
                                              params=params)

        from .prominence_popout import ProminencePopout
        self._prominence_popout = ProminencePopout(
            self,
            prominences=proms,
            current_value=float(self._params.get("min_prominence",
                                                 params.min_prominence)),
            on_apply=self._on_prominence_apply,
        )

    def _on_prominence_apply(self, new_value: float) -> None:
        """Callback from the popout's Apply button: update
        ``min_prominence`` and kick off a re-render."""
        self._params["min_prominence"] = float(new_value)
        self.status_var.set(
            f"min_prominence set to {new_value:.4f}. Re-rendering...")
        # Trigger the normal render path so all downstream artefacts
        # (heatmap, raster, diagnostic panel, AppState.event_results,
        # summary workbook) refresh against the new threshold.
        self._on_render()

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
        # Snapshot Tk-var-backed knobs on the main thread; Tcl
        # interpreter calls (BooleanVar.get / StringVar.get) from a
        # background thread can raise "main thread is not in main loop"
        # under threaded Tcl. Park the snapshot in self._params so the
        # worker reads from a plain dict like every other knob does.
        self._params["manual_subset_enabled"] = bool(
            self.manual_subset_var.get())
        self._params["manual_roi_spec"] = self.manual_roi_var.get()
        # Onset source: 'derivative' or 'spks'. Spks mode reads
        # ``<plane0>/spks.npy`` instead of the ``_dt`` memmap.
        self._params["onset_source"] = self.onset_source_var.get()
        if (self._params["onset_source"] == "spks"
                and not (plane0 / "spks.npy").exists()):
            messagebox.showerror(
                "Spks not available",
                f"{plane0}/spks.npy does not exist. Run Suite2p "
                f"detection on Tab 3 first, or switch back to the "
                f"Derivative onset source.")
            return

        self.render_btn.configure(state="disabled")
        self.render_progress.start()
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
        """Run the full per-ROI hysteresis + population-event detection.

        Delegates the actual computation to
        :func:`calliope.core.event_detection_run.run_event_detection`
        so Tab 0's batch runner uses the same code path. The summary
        write is suppressed here because the existing tab flow calls
        ``_write_summary`` separately on the rendered payload.
        """
        from ...core.event_detection_run import run_event_detection
        return run_event_detection(
            plane0, dict(self._params),
            figures_dir=None,
            write_summary=False,
        )


    def _on_render_error(self, payload: str) -> None:
        self.render_progress.stop()
        self.render_btn.configure(state="normal")
        self.status_var.set("Render error.")
        messagebox.showerror(
            "Render failed", payload.split("\n", 1)[0])

    def _on_render_done(self, data: dict) -> None:
        from . import logic as utils
        self.render_progress.stop()
        self.render_btn.configure(state="normal")

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
        self.summary_btn.configure(state="normal")
        self.prominence_btn.configure(state="normal")
        if self._plane0 is not None:
            try:
                self._write_summary(self._plane0, data)
            except Exception as e:
                print(f"[GUI] event summary write failed: {e}")

        # Publish to AppState so downstream tabs (e.g. Tab 8 spatial
        # propagation) can render the same events without re-detecting.
        try:
            self.state.set_event_results({
                "plane0": self._plane0,
                "event_windows": data.get("event_windows"),
                "A": data.get("A"),
                "first_time": data.get("first_time"),
                "kept_idx": data.get("kept_idx"),
                "fps": data.get("fps"),
                "T": data.get("T"),
                "onsets_by_roi": data.get("onsets_by_roi"),
                # Forward the per-event monotonicity scalars Tab 8's
                # bottom panel uses so it can render from the cached
                # result instead of re-running the 10k-shuffle sweep
                # on every click.
                "event_monotonicity": data.get("event_monotonicity"),
            })
        except Exception as e:
            print(f"[GUI] publish event_results failed: {e}")

    # -- Summary export -----------------------------------------------------

    def _write_summary(self, plane0: Path, data: dict) -> Path:
        summary_writer.update_recording_meta(
            plane0, fps=data.get("fps"),
            T=data.get("T"), N=data.get("N_kept"))
        path = summary_writer.write_events_sheets(
            plane0,
            event_windows=data.get("event_windows"),
            onsets_by_roi=data.get("onsets_by_roi") or [],
            fps=data.get("fps"),
            in_seconds=True,
        )
        # Stamp the EventMonotonicity sheet from the per-event
        # Spearman scalars ``run_event_detection`` already computed
        # and stashed on the data dict. Without this call the GUI
        # path silently skips the sheet -- the headless / batch
        # path in ``run_event_detection.run_event_detection`` writes
        # it via the same module but only when ``write_summary=True``
        # (the tab passes ``write_summary=False`` and runs its own
        # writer here instead).
        em = data.get("event_monotonicity")
        if em:
            try:
                summary_writer.write_event_monotonicity_sheet(plane0, em)
            except Exception as e:
                print(f"[GUI] EventMonotonicity sheet write failed: {e}")
        return path

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
            N_kept_full = int(data.get("N_kept_full") or N_kept)
            lowpass = np.memmap(str(lp_path), dtype="float32", mode="r",
                                shape=(T, N_kept_full))
            pos = data.get("pos_in_memmap")
            if pos is None or len(pos) == N_kept_full:
                wide = np.asarray(lowpass)
            else:
                wide = np.asarray(lowpass[:, np.asarray(pos, dtype=int)])
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
