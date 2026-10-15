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
    AppState, attach_fig_toolbar, attach_resize_handle, drain_queue, format_roi_indices,
    report_stage_error, open_advanced, short_roi_label, spec_defaults,
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
    ``PARAM_SPEC``
        The 32-entry knob list driving the Advanced... dialog -- the
        per-ROI hysteresis / onset-refinement knobs (``z_enter``,
        ``z_exit``, ``min_sep_s``, ``min_rise_dff``,
        ``onset_backtrack_s``), the heatmap ``time_cols_target``, and
        every ``EventDetectionParams`` field. See
        ``calliope.core.utils.EventDetectionParams`` for what each one
        does mathematically.

    Lifecycle
    ---------
    ``__init__``
        Build widgets, prime ``_params`` from PARAM_SPEC defaults,
        subscribe to ``state.subscribe_plane0`` and
        ``state.subscribe_lowpass_ready`` so the tab unlocks once
        the upstream memmaps are written.
    ``_on_render``
        Snapshot the manual-subset Tk vars on the main thread (Tk
        vars are not safe to read from a worker), then offload
        ``core.event_detection_run.run_event_detection_offload`` to a
        child process via ``core.offload.run_offloaded`` (a thread
        would let the pure-Python per-ROI loop hold the GIL and freeze
        the GUI). Load the lowpass + dt memmaps, run ``mad_z`` +
        ``hysteresis_onsets`` per ROI, run
        ``utils.detect_event_windows`` over all onsets, return a
        dict of arrays.
    ``_on_render_done`` (main thread)
        Update the three figures and publish
        ``state.set_event_results(...)``, then defer the summary-sheet
        + manifest writes to a follow-up ``after`` tick so the disk /
        openpyxl work doesn't block the event loop right after a run.
    """

    POLL_MS = 80

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
        {"name": "min_rise_dff", "label": "Min onset rise (dF/F)",
         "type": "float", "default": 0.0, "group": "Per-ROI hysteresis",
         "help": "drop onsets whose foot->peak rise on the low-pass dF/F is "
                 "below this many ABSOLUTE dF/F units; removes noise-amplitude "
                 "excursions on low-SNR ROIs (0 = off; ~0.2 is a typical "
                 "starting point, but the right value is recording-dependent)"},
        {"name": "onset_backtrack_s", "label": "Onset backtrack (s)",
         "type": "float", "default": 0.0, "group": "Per-ROI hysteresis",
         "help": "max window to walk an onset back to the foot of its rise. "
                 "0 = auto: max(indicator tau, 0.8/lowpass-cutoff) per "
                 "recording, using the cutoff Tab 4 last computed at "
                 "(~0.8s for fast GCaMP at 1Hz); set a positive value to "
                 "override"},
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
        {"name": "auto_min_prominence",
         "label": "Auto min prominence (null estimate)",
         "type": "bool", "default": True,
         "group": "Population events - peaks",
         "help": "set the prominence floor automatically from the "
                 "circular-shift null per recording; uncheck to use the "
                 "fixed value below"},
        {"name": "auto_min_prominence_percentile",
         "label": "Auto prominence percentile",
         "type": "float", "default": 99.0,
         "group": "Population events - peaks",
         "help": "null percentile used when Auto min prominence is on "
                 "(fitted to ~99 against hand-tuned recordings)"},
        {"name": "auto_min_prominence_n_shuffles",
         "label": "Auto prominence shuffles",
         "type": "int", "default": 200,
         "group": "Population events - peaks",
         "help": "circular-shift count for the null estimate "
                 "(more = steadier, slower)"},
        {"name": "auto_min_prominence_seed",
         "label": "Auto prominence seed",
         "type": "int", "default": 0,
         "group": "Population events - peaks",
         "help": "RNG seed for the null estimate (reproducibility)"},
        {"name": "min_prominence", "label": "Min peak prominence",
         "type": "float", "default": 0.002,
         "group": "Population events - peaks",
         "help": "find_peaks prominence floor on the smoothed density; "
                 "used only when Auto min prominence is off"},
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
         "type": "float", "default": 1.0,
         "group": "Population events - boundaries",
         "help": "join back-to-back windows whose gap is <= this into one "
                 "event (a seizure becomes one long event); 0 = off"},
        {"name": "timing_method", "label": "Within-event timing",
         "type": "choice", "choices": ["half-max", "first onset"],
         "default": "half-max", "group": "Population events - timing",
         "help": "per-ROI activation time inside each event (Tabs 7/8, "
                 "monotonicity): half-max crossing on lightly filtered "
                 "dF/F, or the detection onset (more jitter)"},
        {"name": "timing_cutoff_hz", "label": "Timing filter cutoff (Hz)",
         "type": "float", "default": 5.0,
         "group": "Population events - timing",
         "help": "zero-phase low-pass for the half-max timing; skipped "
                 "when at or above ~Nyquist"},
        {"name": "seizure_min_duration_s",
         "label": "Seizure-like if >= (s)",
         "type": "float", "default": 5.0,
         "group": "Population events - classes",
         "help": "events at least this long are classed 'seizure-like'"},
        {"name": "population_fraction",
         "label": "Population burst if ROI fraction >=",
         "type": "float", "default": 0.5,
         "group": "Population events - classes",
         "help": "shorter events where at least this fraction of ROIs "
                 "fire are 'population burst', else 'local burst'"},
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
        self._summary_job = None
        self._after_summary_write = None
        # Heavy detection runs in a child process (see core.offload) so
        # the pure-Python per-ROI hysteresis loop can't hold the GIL and
        # freeze the GUI. ``None`` until the first render.
        self._render_job = None
        # Interactive "Export figures" also runs in a child process
        # (re-detection + figure render); ``None`` until the first export.
        self._export_job = None
        self._params: dict = spec_defaults(self.PARAM_SPEC)
        self._last_data: Optional[dict] = None
        # When True, the next render skips persisting params to the run
        # manifest. Set by the "Open run folder…" reload so its own
        # recompute can't overwrite a good manifest with the params it
        # just loaded (or with defaults, when none were found).
        self._suppress_manifest_write = False
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
                     "error": self._on_render_error,
                     "status": self._on_render_status,
                     "export_done": self._on_export_done,
                     "summary_done": self._on_summary_written,
                     "summary_error": self._on_summary_failed,
                     "export_error": self._on_export_error},
                    poll_ms=self.POLL_MS)
        state.subscribe_plane0(self._on_plane0)
        state.subscribe_lowpass_ready(self._on_plane0)
        if state.lowpass_plane0 is not None:
            self._on_plane0(state.lowpass_plane0)
        elif state.plane0 is not None:
            self._on_plane0(state.plane0)

    # -- UI -----------------------------------------------------------------

    def apply_batch_row(self, params: dict, log=None) -> None:
        """Apply a batch row's PARAM_SPEC fields + the manual-subset /
        onset-source Tk vars that ``_on_render`` snapshots into ``_params``.

        Public seam the Tab 0 batch runner calls instead of poking this
        tab's internals.
        """
        p = params or {}
        for entry in self.PARAM_SPEC:
            name = entry["name"]
            if name in p:
                self._params[name] = p[name]
        if "manual_subset_enabled" in p:
            self.manual_subset_var.set(bool(p["manual_subset_enabled"]))
        if "manual_roi_spec" in p:
            self.manual_roi_var.set(str(p["manual_roi_spec"]))
        if "onset_source" in p:
            self.onset_source_var.set(str(p["onset_source"]))

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
                     font=ctk.CTkFont(size=11, slant="italic"),
                     wraplength=380, justify="left").pack(
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
        self.export_btn = ctk.CTkButton(
            row, text="Export figures", width=130,
            command=self._on_export_figures, state="disabled")
        self.export_btn.pack(side="right", padx=(0, 6))
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
                             initial_height=300)

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
                             initial_height=300)

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

        # Circular-shift null floor: computed off-thread (a few seconds at
        # 200 shuffles) so the popout opens instantly with the observed
        # histogram and the null p95/p99 reference lines fill in when ready.
        # By default (auto_min_prominence) the null p99 drives the applied
        # per-recording floor; in this popout the null lines are a reference
        # for picking a manual value, and applying one here turns auto off.
        onsets_by_roi = self._last_data.get("onsets_by_roi")
        T = self._last_data.get("T")
        fps = self._last_data.get("fps")
        null_ready = bool(onsets_by_roi) and bool(T) and bool(fps and fps > 0)

        from .prominence_popout import ProminencePopout
        self._prominence_popout = ProminencePopout(
            self,
            prominences=proms,
            current_value=float(self._params.get("min_prominence",
                                                 params.min_prominence)),
            on_apply=self._on_prominence_apply,
            null_pending=null_ready,
            auto_on=bool(self._params.get("auto_min_prominence",
                                          params.auto_min_prominence)),
            auto_percentile=float(params.auto_min_prominence_percentile),
        )

        if null_ready:
            self._spawn_null_floor_worker(
                self._prominence_popout, list(onsets_by_roi),
                int(T), float(fps), params)

    def _spawn_null_floor_worker(self, popout, onsets_by_roi, T, fps,
                                 params) -> None:
        """Compute the circular-shift null prominence floor in a daemon
        thread and push p95/p99 back into ``popout`` on the Tk main thread.

        Errors are swallowed to a "null unavailable" state -- the floor is
        an optional reference, never a blocker on tuning.
        """
        from ...core.utils import null_prominence_percentiles

        def worker():
            try:
                # Honour the user-editable Advanced knobs rather than
                # hardcoding. Always include 95/99 so the popout's fixed
                # p95/p99 reference styling keeps working, plus the user's
                # configured percentile (which drives the actual AUTO floor
                # in detect_event_windows).
                pct = float(params.auto_min_prominence_percentile)
                wanted = sorted({95.0, 99.0, pct})
                pcts = null_prominence_percentiles(
                    onsets_by_roi, T, fps, params,
                    percentiles=tuple(wanted),
                    n_shuffles=int(params.auto_min_prominence_n_shuffles),
                    seed=int(params.auto_min_prominence_seed))
            except Exception as e:
                print(f"[GUI] null-floor computation failed: {e}")
                pcts = None
            # Marshal back to the Tk thread; the popout may have been
            # closed in the meantime (set_null_percentiles guards that).
            try:
                self.after(0, lambda: popout.set_null_percentiles(pcts))
            except Exception:
                pass

        threading.Thread(target=worker, daemon=True).start()

    def _on_prominence_apply(self, new_value: float) -> None:
        """Callback from the popout's Apply button: update
        ``min_prominence`` and kick off a re-render."""
        self._params["min_prominence"] = float(new_value)
        # Applying a value by hand is an explicit override: switch off the
        # auto (null-estimate) floor so this value actually takes effect.
        self._params["auto_min_prominence"] = False
        self.status_var.set(
            f"min_prominence set to {new_value:.4f} (auto off). "
            "Re-rendering...")
        # Trigger the normal render path so all downstream artefacts
        # (heatmap, raster, diagnostic panel, AppState.event_results,
        # summary workbook) refresh against the new threshold.
        self._on_render()

    # -- Render worker ------------------------------------------------------

    def _on_render(self) -> None:
        if self._render_job is not None and self._render_job.running():
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

        # Run detection in a child process (not a thread): the per-ROI
        # hysteresis loop is pure-Python and would hold the GIL, freezing
        # the GUI. ``run_offloaded`` re-posts ("done"|"error"|"status",
        # payload) onto ``_render_queue`` from the main thread, so the
        # existing drain_queue handlers (and their AppState/batch
        # ``set_*`` calls) keep running on the main thread untouched.
        from ...core.offload import run_offloaded
        from ...core import event_detection_run as edr
        self._render_job = run_offloaded(
            self, self._render_queue,
            edr.run_event_detection_offload,
            args=(str(plane0), dict(self._params)),
            progress_kind="status",
            poll_ms=self.POLL_MS,
        )

    def _on_render_status(self, payload) -> None:
        """Live progress string from the detection child process.

        ``payload`` is the tuple of positional args the child's
        ``progress_cb(message)`` was called with; Tab 5 only sends a
        single status string.
        """
        try:
            msg = payload[0] if isinstance(payload, tuple) and payload \
                else str(payload)
        except Exception:
            return
        self.status_var.set(msg)


    def _on_render_error(self, payload: str) -> None:
        self.render_progress.stop()
        self.render_btn.configure(state="normal")
        self.status_var.set("Render error.")
        if report_stage_error(self.state, payload):
            return
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

        # x in seconds, not downsampled bins whose width changes per
        # recording (UX audit papercut).
        def _extent(m):
            return (0.0, m.shape[1] * ds / fps if fps > 0 else m.shape[1],
                    m.shape[0], 0)

        ax = self.hm_ax
        ax.clear(); ax.set_axis_on()
        ax.imshow(hm, aspect="auto", interpolation="nearest",
                  extent=_extent(hm))
        ax.set_xlabel(f"Time (s)  [{ds * 1000.0 / fps:.0f} ms bins]",
                      fontsize=8)
        ax.set_ylabel("ROIs (most active at top)", fontsize=8)
        ax.tick_params(labelsize=7)
        self.hm_canvas.draw_idle()

        ax = self.er_ax
        ax.clear(); ax.set_axis_on()
        ax.imshow(rs, aspect="auto", interpolation="nearest", cmap="Greys",
                  extent=_extent(rs))
        ax.set_xlabel(f"Time (s)  [{ds * 1000.0 / fps:.0f} ms bins]",
                      fontsize=8)
        ax.set_ylabel("ROIs (most active at top)", fontsize=8)
        ax.tick_params(labelsize=7)
        self.er_canvas.draw_idle()

        ax = self.ed_ax
        ax.clear(); ax.set_axis_on()
        duration_s = float(T) / float(fps) if fps > 0 else 1.0
        try:
            diagnostics = data["diagnostics"]
            utils.plot_event_detection(diagnostics, ax=ax)
            ev = data["event_windows"]
            if ev is not None and len(ev) > 0:
                utils.shade_event_windows(ax, ev, color="C1", alpha=0.20,
                                          classes=data.get("event_class"))
            # Force the time axis to span the full recording so the trace
            # doesn't look cut off when the auto-fit picks up only the
            # smoothed-density extent.
            ax.set_xlim(0.0, duration_s)
            # Explicit y-axis fit: anchor the bottom at 0 (densities are
            # non-negative) and put the top at the tallest detected
            # peak + 10% headroom. ``peak_height`` is the smoothed-
            # density value at each picked peak, so it's by construction
            # the highest point the user cares about. Fall back to 1.0
            # when no events were detected.
            peak_h = np.asarray(
                diagnostics.get("peak_height", []), dtype=float)
            finite = peak_h[np.isfinite(peak_h)] if peak_h.size else peak_h
            ymax = float(finite.max()) if finite.size else 1.0
            if ymax <= 0:
                ymax = 1.0
            ax.set_ylim(0.0, ymax * 1.10)
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
        subset = (self._params.get("manual_subset_enabled")
                  and data.get("kept_idx") is not None)
        self.status_var.set(
            f"Rendered  N_kept={N}  T={T}  fps={fps:.2f}  "
            f"duration={duration_s:.1f}s  events={n_events}"
            + (f"\nSubset ROIs:{format_roi_indices(sorted(int(i) for i in data['kept_idx']))}"
               if subset else ""))
        if subset:
            for a, c in ((self.hm_ax, self.hm_canvas),
                         (self.er_ax, self.er_canvas)):
                a.set_title(f"Subset {short_roi_label(data['kept_idx'])}",
                            fontsize=8)
                c.draw_idle()

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
        # An interactive export still running keeps its button disabled;
        # _on_export_done / _on_export_error re-enable it.
        job = getattr(self, "_export_job", None)
        if job is None or not job.running():
            self.export_btn.configure(state="normal")

        # The summary workbook write runs in a child process: openpyxl
        # took ~23 s on a 679-ROI recording and froze the GUI when it ran
        # here. In a batch, publish only once it has finished, so the next
        # stage (and the finalize move) never races the write; otherwise
        # publish now so Tab 8 can render straight away.
        publish = lambda d=data: self._publish_event_results(d)
        in_batch = bool(getattr(self.state, "batch_active", False))
        if not in_batch:
            publish()
        if self._plane0 is None:
            if in_batch:
                publish()
            return
        self._after_summary_write = publish if in_batch else None
        keys = ("fps", "T", "N_kept", "event_windows", "onsets_by_roi",
                "event_monotonicity", "event_class")
        from ...core.offload import run_offloaded
        from ...core import event_detection_run as edr
        self._summary_job = run_offloaded(
            self, self._render_queue, edr.write_event_summary_offload,
            args=(str(self._plane0), {k: data.get(k) for k in keys}),
            done_kind="summary_done", error_kind="summary_error",
            poll_ms=self.POLL_MS)

    def _publish_event_results(self, data: dict) -> None:
        """Publish to AppState so downstream tabs (e.g. Tab 8 spatial
        propagation) render the same events, and the batch runner advances."""
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
                # Per-event monotonicity scalars (theta*, rho, p, u_x,
                # u_y) for any consumer that only needs the scalars.
                # Tab 8 does not read them: its plots need the full rho
                # curve + null distribution, so it re-runs the sweep.
                "event_monotonicity": data.get("event_monotonicity"),
            })
        except Exception as e:
            print(f"[GUI] publish event_results failed: {e}")

    def _on_summary_failed(self, payload) -> None:
        print(f"[GUI] event summary write failed:\n{payload}")
        self._on_summary_written(None)

    def _on_summary_written(self, payload) -> None:
        # Persist the params used so a future "Open run folder…" reload
        # can restore them instead of recomputing with GUI defaults.
        # Suppressed during a reload's own recompute (writing back the
        # just-loaded params is pointless, and writing defaults when none
        # were found would silently record them as the run's settings).
        if not self._suppress_manifest_write and self._plane0 is not None:
            try:
                self._write_event_manifest(self._plane0)
            except Exception as e:
                print(f"[GUI] event manifest update failed: {e}")
        self._suppress_manifest_write = False
        then, self._after_summary_write = self._after_summary_write, None
        if then is not None:
            then()

    def _write_event_manifest(self, plane0: Path) -> None:
        """Merge the current event-detection params into the run manifest.

        ``write_export_manifest`` merges rather than overwrites, so this
        contributes the Tab 5 params to the shared manifest without
        clobbering the other stages' entries -- which is what lets a
        reload restore events without falling back to defaults.
        """
        from ...core.export_manifest import write_export_manifest
        from ...core.utils import safe_recording_id, save_folder_for_plane0
        save_folder = save_folder_for_plane0(plane0)
        figures_root = save_folder / "calliope_figures"
        rec_id = safe_recording_id(plane0)
        write_export_manifest(
            figures_root, rec_id=rec_id, params=dict(self._params),
            plane0=plane0, ckpt_path=None)

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
            event_classes=data.get("event_class"),
        )
        # Stamp the EventMonotonicity sheet from the per-event
        # Spearman scalars ``run_event_detection`` already computed
        # and stashed on the data dict. Without this call the GUI
        # path silently skips the sheet -- the headless / batch
        # path in ``core.event_detection_run.run_event_detection``
        # writes it via the same module but only when
        # ``write_summary=True`` (the tab passes ``write_summary=False``
        # and runs its own writer here instead).
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

    def _on_export_figures(self, quiet: bool = False) -> None:
        """Re-render Tab 5's heatmap / raster / event-detection figures
        for the currently-loaded recording as PNG + SVG under
        ``<save_folder>/calliope_figures/event_detection/``, then update
        the manifest. Re-runs detection on the existing lowpass memmap
        (minus the per-event monotonicity sweep, which the figures don't
        use) so the in-memory state can stay memory-light.

        An interactive click runs that work in a child process via
        ``core.offload.run_offloaded`` (like Render), so the GUI stays
        responsive; ``_on_export_done`` / ``_on_export_error`` report the
        result. ``quiet=True`` (Tab 0 batch toggle) runs it synchronously
        -- the batch advances to the next stage as soon as this returns --
        and suppresses messageboxes.
        """
        plane0 = self._plane0
        if plane0 is None:
            if not quiet:
                messagebox.showinfo("No data", "Load a recording first.")
            return
        if not quiet and any(
                job is not None and job.running()
                for job in (self._render_job, self._export_job)):
            messagebox.showinfo(
                "Busy", "Render / export is already running.")
            return
        from ...core.utils import save_folder_for_plane0
        from ...core import event_detection_run as edr
        save_folder = save_folder_for_plane0(plane0)
        figures_root = save_folder / "calliope_figures"
        params = dict(self._params)
        self.status_var.set("Exporting event figures ...")

        if not quiet:
            from ...core.offload import run_offloaded
            self.export_btn.configure(state="disabled")
            try:
                self._export_job = run_offloaded(
                    self, self._render_queue,
                    edr.export_event_figures_offload,
                    args=(str(plane0), params, str(figures_root)),
                    done_kind="export_done", error_kind="export_error",
                    progress_kind="status",
                    poll_ms=self.POLL_MS,
                )
            except Exception as e:
                # Dispatch itself failed (spawn / pickling): no job will
                # ever post export_error, so recover the button here.
                self._export_job = None
                self._on_export_error(f"could not start export: {e}")
            return

        self.update_idletasks()
        try:
            result = edr.export_event_figures_offload(
                str(plane0), params, str(figures_root))
        except Exception as e:
            self.status_var.set(f"Export failed: {e}")
            return
        self.status_var.set(
            f"Exported event figures -> {result['events_dir']}")

    def _on_export_done(self, result: dict) -> None:
        """Main-thread completion handler for an interactive export."""
        self.export_btn.configure(state="normal")
        events_dir = result.get("events_dir", "")
        self.status_var.set(f"Exported event figures -> {events_dir}")
        messagebox.showinfo(
            "Export complete",
            f"Wrote heatmap / raster / event-detection figures "
            f"(PNG + SVG) to:\n  {events_dir}\n\n"
            f"Manifest:\n  {result.get('manifest', '')}")

    def _on_export_error(self, payload: str) -> None:
        self.export_btn.configure(state="normal")
        first = payload.split("\n", 1)[0]
        self.status_var.set(f"Export failed: {first}")
        messagebox.showerror("Export failed", first)

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
