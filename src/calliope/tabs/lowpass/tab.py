"""calliope.tabs.lowpass.tab - Tab 4: Low-pass filter.

What this tab is for
--------------------
Calcium imaging traces are noisy at the per-frame level (shot noise
from photon counting), but real GCaMP transients live at frequencies
below ~5 Hz. So we low-pass filter to suppress the noise and keep
the signal. The right cutoff depends on the recording's frame rate,
the GCaMP variant, and how brief the events of interest are -- so
this tab makes that choice interactive.

What the user sees
------------------
Three panels updating live as you move a cutoff slider:
    1. FFT power spectrum of the chosen trace, with the cutoff
       marked as a vertical line. Used to pick the cutoff at the
       elbow between signal and noise.
    2. The raw dF/F trace.
    3. The same trace passed through a causal Butterworth low-pass
       at the slider cutoff.

The "trace source" radio buttons let the user pick which trace to
inspect: population mean, median, the most-active ROI, or a manual
list of Suite2p ROI ids (mean / median aggregation).

When the user is happy, "Compute" runs the same filter + Savitzky-
Golay first derivative on **every** kept ROI and writes two
``(T, N_kept)`` memmaps:
    r0p7_filtered_dff_lowpass.memmap.float32
    r0p7_filtered_dff_dt.memmap.float32
Then it publishes ``state.set_lowpass_ready(plane0)`` so Tab 5 picks
up automatically.

Class layout
------------
``LowpassTab(ttk.Frame)``
    ``CUTOFF_MIN`` / ``CUTOFF_MAX`` / ``CUTOFF_DEFAULT``
        Slider bounds and starting value (also exposed in PARAM_SPEC
        so power users can override).
    ``PARAM_SPEC``
        Butterworth order, SG window length / polynomial, slider
        bounds.
    ``__init__``
        Subscribes to ``state.subscribe_plane0`` so it knows when a
        Suite2p detection has finished and the dF/F memmap is on
        disk.
    ``_on_slider_drag``
        Slider callback: redraws panel 3 + panel 1 cutoff line in
        real time. Debounced via ``_slider_job`` so dragging doesn't
        recompute on every pixel of motion.
    ``_on_compute``
        Worker-thread launcher that filters every ROI and writes the
        memmaps. Same threading pattern as Tabs 1 / 3.

Default 0.01 - 7 Hz cutoff range; the slider bounds themselves are
user-tunable from the Advanced dialog (``cutoff_min``/``cutoff_max``
in PARAM_SPEC).

Trace source can be the population mean, median, the highest-scoring
ROI, or a user-supplied list / range of original Suite2p ROI ids
(aggregated by mean/median).

The Compute button writes r0p7_filtered_dff_lowpass.memmap.float32 +
r0p7_filtered_dff_dt.memmap.float32 at the user's chosen cutoff and
notifies the event-detection tab via ``state.set_lowpass_ready``.
"""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional

import customtkinter as ctk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ...gui_common import (
    AppState, attach_fig_toolbar, drain_queue, format_roi_indices, report_stage_error,
    open_advanced, parse_manual_roi_spec, spec_defaults,
)


class LowpassTab(ctk.CTkFrame):
    """Three panels:
        1. FFT power spectrum of the population-mean filtered dF/F.
        2. Population-mean dF/F (raw).
        3. Population-mean dF/F passed through a causal low-pass at
           the slider's cutoff (default 0.01 - 7 Hz, default 1 Hz; the
           bounds are governed by the ``cutoff_min`` / ``cutoff_max``
           PARAM_SPEC entries).
    Panel 3 (and the cutoff line on panel 1) update live as the slider
    moves; panels 1 and 2 are computed once when the data loads.
    """

    CUTOFF_MIN = 0.01
    CUTOFF_MAX = 7.0
    CUTOFF_DEFAULT = 1.0

    POLL_MS = 80

    PARAM_SPEC: list = [
        {"name": "filter_order", "label": "Butterworth order",
         "type": "int", "default": 2, "group": "Low-pass filter"},
        {"name": "sg_win_ms", "label": "SG derivative window (ms)",
         "type": "int", "default": 333, "group": "Derivative"},
        {"name": "sg_poly", "label": "SG polynomial order",
         "type": "int", "default": 2, "group": "Derivative"},
        {"name": "cutoff_min", "label": "Slider min (Hz)",
         "type": "float", "default": 0.01, "group": "Slider bounds"},
        {"name": "cutoff_max", "label": "Slider max (Hz)",
         "type": "float", "default": 7.0, "group": "Slider bounds"},
    ]

    def __init__(self, master, state: AppState) -> None:
        super().__init__(master, fg_color="transparent")
        self.state = state
        self._plane0: Optional[Path] = None
        self._fps: float = 15.0
        self._mean_dff: Optional[np.ndarray] = None
        self._trace_label: str = "mean"
        self._fft_xf: Optional[np.ndarray] = None
        self._fft_power: Optional[np.ndarray] = None
        self._cutoff_line = None
        self._slider_job: Optional[str] = None
        self._params: dict = spec_defaults(self.PARAM_SPEC)

        self._compute_queue: queue.Queue = queue.Queue()
        # Heavy filter write runs in a child process (see core.offload)
        # so the per-ROI scipy loop can't hold the GIL and freeze the
        # GUI. ``None`` until the first compute.
        self._compute_job = None
        self._n_kept: int = 0
        self._T: int = 0

        self._build_ui()
        drain_queue(self, self._compute_queue,
                    {"status": self.compute_status_var.set,
                     "done": self._on_compute_done,
                     "error": self._on_compute_error},
                    poll_ms=self.POLL_MS)
        state.subscribe_plane0(self._on_plane0)
        if state.plane0 is not None:
            self._on_plane0(state.plane0)

    # -- UI -----------------------------------------------------------------

    def apply_batch_row(self, params: dict, log=None) -> None:
        """Apply a batch row's PARAM_SPEC fields + the live cutoff slider.

        Public seam the Tab 0 batch runner calls instead of poking this
        tab's internals.
        """
        p = params or {}
        for entry in self.PARAM_SPEC:
            name = entry["name"]
            if name in p:
                self._params[name] = p[name]
        if "cutoff_hz" in p:
            try:
                self.cutoff_var.set(float(p["cutoff_hz"]))
            except (TypeError, ValueError):
                pass

    def _build_ui(self) -> None:
        header = ctk.CTkFrame(self)
        header.pack(fill="x", padx=10, pady=(10, 6))
        ctk.CTkLabel(
            header,
            text="Low-pass filter (causal Butterworth, order 2)",
            font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(6, 0))

        row = ctk.CTkFrame(header, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=2)
        ctk.CTkLabel(row, text="Cutoff (Hz):", width=90,
                     anchor="w").pack(side="left")
        self.cutoff_var = tk.DoubleVar(value=self.CUTOFF_DEFAULT)
        self.slider = ctk.CTkSlider(
            row, from_=self.CUTOFF_MIN, to=self.CUTOFF_MAX,
            orientation="horizontal", variable=self.cutoff_var,
            command=self._on_slider,
        )
        self.slider.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.cutoff_str = tk.StringVar(value=f"{self.CUTOFF_DEFAULT:.2f}")
        entry = ctk.CTkEntry(row, textvariable=self.cutoff_str, width=80)
        entry.pack(side="left")
        entry.bind("<Return>", self._on_entry)
        ctk.CTkLabel(row, text="Hz").pack(side="left", padx=(2, 8))
        ctk.CTkButton(row, text="Advanced...", width=90,
                      command=self._on_advanced).pack(side="right",
                                                      padx=(0, 6))
        ctk.CTkButton(row, text="Reload from folder...", width=150,
                      command=self._reload_from_folder).pack(side="right")

        row = ctk.CTkFrame(header, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=(4, 0))
        ctk.CTkLabel(row, text="", width=90).pack(side="left")
        self.compute_btn = ctk.CTkButton(
            row, text="Compute low-pass + derivative (write memmaps)",
            command=self._on_compute, state="disabled", width=320)
        self.compute_btn.pack(side="left")
        self.export_btn = ctk.CTkButton(
            row, text="Export figures", width=130,
            command=self._on_export_figures, state="disabled")
        self.export_btn.pack(side="left", padx=(6, 0))
        self.compute_progress = ctk.CTkProgressBar(
            row, mode="indeterminate", width=160)
        self.compute_progress.pack(side="left", padx=12)
        self.compute_status_var = tk.StringVar(value="")
        ctk.CTkLabel(row, textvariable=self.compute_status_var,
                     font=ctk.CTkFont(size=11, slant="italic")).pack(
            side="left")

        row = ctk.CTkFrame(header, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=(4, 6))
        ctk.CTkLabel(row, text="Trace source:", width=90,
                     anchor="w").pack(side="left")
        # Default to median: an arithmetic mean across ROIs is dominated by a
        # few near-zero-baseline ROIs whose dF/F runs to ~1e9, which blows up
        # the live preview (and matches the headless figure default).
        self.source_var = tk.StringVar(value="median")
        ctk.CTkRadioButton(
            row, text="Mean across kept ROIs",
            value="mean", variable=self.source_var,
            command=self._on_source_change,
        ).pack(side="left", padx=(0, 12))
        ctk.CTkRadioButton(
            row, text="Median across kept ROIs",
            value="median", variable=self.source_var,
            command=self._on_source_change,
        ).pack(side="left", padx=(0, 12))
        ctk.CTkRadioButton(
            row, text="Best-scoring ROI (max predicted_cell_prob)",
            value="best", variable=self.source_var,
            command=self._on_source_change,
        ).pack(side="left", padx=(0, 12))
        ctk.CTkRadioButton(
            row, text="Manual ROI(s):",
            value="manual", variable=self.source_var,
            command=self._on_source_change,
        ).pack(side="left")
        self.manual_roi_var = tk.StringVar(value="0")
        manual_entry = ctk.CTkEntry(row, textvariable=self.manual_roi_var,
                                    width=180)
        manual_entry.pack(side="left", padx=(2, 0))
        manual_entry.bind("<Return>", self._on_manual_entry)
        manual_entry.bind("<FocusOut>", self._on_manual_entry)
        # Aggregation for multi-ROI manual input (single-ROI inputs ignore it).
        self.manual_agg_var = tk.StringVar(value="mean")
        manual_agg = ctk.CTkComboBox(
            row, variable=self.manual_agg_var,
            values=["mean", "median"], state="readonly", width=90,
            command=lambda _v: self._on_manual_entry())
        manual_agg.pack(side="left", padx=(4, 0))

        # View-mode toggle: dF/F vs robust z-score. dF/F is the canonical
        # display the on-disk memmaps carry; robust z is a *view-only*
        # transform that re-expresses each trace as ``(x - median(x)) /
        # (1.4826 * MAD(x))``. Useful because dF/F with a rolling baseline
        # can create "positive artifacts following negative deviations"
        # for inhibited cells (Vanwalleghem & Constantin, Frontiers Neural
        # Circuits 2021, "The Curse of Negativity") and miscalibrates
        # cross-recording magnitudes; robust z corrects both at the cost
        # of losing the absolute fluorescence interpretation. No memmap is
        # written -- the toggle just rebuilds the figure.
        row2 = ctk.CTkFrame(header, fg_color="transparent")
        row2.pack(fill="x", padx=8, pady=(0, 6))
        ctk.CTkLabel(row2, text="View:", width=90,
                     anchor="w").pack(side="left")
        self.view_mode_var = tk.StringVar(value="dff")
        ctk.CTkRadioButton(
            row2, text="dF/F", value="dff",
            variable=self.view_mode_var,
            command=self._on_view_mode_change,
        ).pack(side="left", padx=(0, 12))
        ctk.CTkRadioButton(
            row2, text="Robust z (median ± 1.4826·MAD)",
            value="robust_z", variable=self.view_mode_var,
            command=self._on_view_mode_change,
        ).pack(side="left")

        self.status_var = tk.StringVar(
            value="Run detection first (or reload from a finished folder).")
        ctk.CTkLabel(self, textvariable=self.status_var,
                     font=ctk.CTkFont(size=11, slant="italic")).pack(
            anchor="w", padx=10, pady=(0, 6))

        # Persistent recording header so the user can always tell
        # which plane0 the FFT / raw / low-pass panels are showing.
        # Updated by ``_on_plane0`` once data has loaded.
        self.recording_var = tk.StringVar(value="No recording loaded.")
        ctk.CTkLabel(self, textvariable=self.recording_var,
                     font=ctk.CTkFont(size=12, slant="italic")).pack(
            anchor="w", padx=10, pady=(0, 6))

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        body.rowconfigure(0, weight=1, uniform="rows")
        body.rowconfigure(1, weight=1, uniform="rows")
        body.rowconfigure(2, weight=1, uniform="rows")
        body.columnconfigure(0, weight=1)

        f1 = ctk.CTkFrame(body)
        f1.grid(row=0, column=0, sticky="nsew", pady=(0, 4))
        ctk.CTkLabel(f1, text="1. FFT power spectrum",
                     font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(6, 0))
        # Y-axis scale toggle. Log is the default (better for visualising
        # power-law roll-off at higher frequencies); linear is useful for
        # sanity-checking a single dominant peak at e.g. line frequency.
        fft_top = ctk.CTkFrame(f1, fg_color="transparent")
        fft_top.pack(fill="x", padx=4)
        ctk.CTkLabel(fft_top, text="y-axis:").pack(side="left", padx=(0, 4))
        self.fft_yscale_var = tk.StringVar(value="log")
        ctk.CTkRadioButton(
            fft_top, text="log", value="log",
            variable=self.fft_yscale_var,
            command=self._draw_fft).pack(side="left")
        ctk.CTkRadioButton(
            fft_top, text="linear", value="linear",
            variable=self.fft_yscale_var,
            command=self._draw_fft).pack(side="left", padx=(4, 0))
        self.fft_fig = plt.Figure(figsize=(8, 2.4), tight_layout=True)
        self.fft_ax = self.fft_fig.add_subplot(111)
        self.fft_ax.set_axis_off()
        self.fft_ax.text(0.5, 0.5, "No data", ha="center", va="center",
                         transform=self.fft_ax.transAxes)
        self.fft_canvas = FigureCanvasTkAgg(self.fft_fig, master=f1)
        attach_fig_toolbar(self.fft_canvas, f1)
        self.fft_canvas.get_tk_widget().pack(fill="both", expand=True,
                                             padx=4, pady=(0, 4))

        f2 = ctk.CTkFrame(body)
        f2.grid(row=1, column=0, sticky="nsew", pady=(0, 4))
        ctk.CTkLabel(f2, text="2. Raw dF/F",
                     font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(6, 0))
        self.raw_fig = plt.Figure(figsize=(8, 2.4), tight_layout=True)
        self.raw_ax = self.raw_fig.add_subplot(111)
        self.raw_ax.set_axis_off()
        self.raw_ax.text(0.5, 0.5, "No data", ha="center", va="center",
                         transform=self.raw_ax.transAxes)
        self.raw_canvas = FigureCanvasTkAgg(self.raw_fig, master=f2)
        attach_fig_toolbar(self.raw_canvas, f2)
        self.raw_canvas.get_tk_widget().pack(fill="both", expand=True,
                                             padx=4, pady=(0, 4))

        f3 = ctk.CTkFrame(body)
        f3.grid(row=2, column=0, sticky="nsew")
        ctk.CTkLabel(f3, text="3. Low-pass dF/F",
                     font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(6, 0))
        self.lp_fig = plt.Figure(figsize=(8, 2.4), tight_layout=True)
        self.lp_ax = self.lp_fig.add_subplot(111)
        self.lp_ax.set_axis_off()
        self.lp_ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=self.lp_ax.transAxes)
        self.lp_canvas = FigureCanvasTkAgg(self.lp_fig, master=f3)
        attach_fig_toolbar(self.lp_canvas, f3)
        self.lp_canvas.get_tk_widget().pack(fill="both", expand=True,
                                            padx=4, pady=(0, 4))

    # -- Plane0 handling ----------------------------------------------------

    def _reload_from_folder(self) -> None:
        path = filedialog.askdirectory(
            title="Select a suite2p plane0 folder containing "
                  "r0p7_filtered_dff.memmap.float32")
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
        self.status_var.set(f"Loading dF/F from {self._plane0} ...")
        self.update_idletasks()
        try:
            self._load_data(self._plane0)
        except Exception as e:
            self.status_var.set(f"Load error: {e}")
            self.compute_btn.configure(state="disabled")
            return
        self._draw_fft()
        self._draw_raw()
        self._draw_lowpass()
        self.status_var.set(
            f"Loaded T={self._mean_dff.size}  N_kept={self._n_kept}  "
            f"fps={self._fps:.2f}  cutoff={self.cutoff_var.get():.3f} Hz")
        self.compute_btn.configure(state="normal")
        self.export_btn.configure(state="normal")

    def _load_data(self, plane0: Path) -> None:
        from . import logic as utils
        F = np.load(plane0 / "F.npy", mmap_mode="r")
        N_total, T = F.shape
        self._T = int(T)

        prob_path = plane0 / "predicted_cell_prob.npy"
        filtered_path = plane0 / "r0p7_filtered_dff.memmap.float32"

        if filtered_path.exists():
            # Resolve the keep mask against the filtered memmap's on-disk
            # size, NOT the live predicted_cell_mask/iscell. Curation or a
            # "Promote to filter mask" after the memmap was written drifts
            # those files, so mask.sum() no longer matches the file's column
            # count and the np.memmap request below fails on Windows with
            # [WinError 8] Not enough memory resources. See
            # utils.resolve_filtered_mask.
            mask, N_kept = utils.resolve_filtered_mask(
                plane0, N_total, memmap_path=filtered_path, T=T)
            self._n_kept = N_kept
            if N_kept == 0:
                raise RuntimeError("No ROIs survive the cell-filter mask.")
            dff = np.memmap(str(filtered_path), dtype="float32", mode="r",
                            shape=(T, N_kept))
            kept_idx = np.flatnonzero(mask)
        else:
            # No filtered memmap to size against: fall back to the full
            # (T, N_total) memmap sliced by the live mask.
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
            self._n_kept = N_kept
            if N_kept == 0:
                raise RuntimeError("No ROIs survive the cell-filter mask.")
            dff_path = plane0 / "r0p7_dff.memmap.float32"
            if not dff_path.exists():
                raise FileNotFoundError(
                    f"Neither {filtered_path.name} nor {dff_path.name} "
                    f"exists in {plane0}.")
            full = np.memmap(str(dff_path), dtype="float32", mode="r",
                             shape=(T, N_total))
            dff = np.asarray(full[:, mask])
            kept_idx = np.flatnonzero(mask)

        source = self.source_var.get() if hasattr(self, "source_var") \
            else "mean"
        if source == "best":
            if not prob_path.exists():
                print("[GUI] predicted_cell_prob.npy missing; "
                      "falling back to population mean trace.")
                source = "mean"

        if source == "manual":
            # Manual ROI(s) are *original* suite2p ROI indices (0..N_total-1),
            # regardless of whether they survived the cell-filter mask.
            spec = self.manual_roi_var.get()
            try:
                roi_indices = parse_manual_roi_spec(spec, N_total)
            except ValueError as e:
                print(f"[GUI] Manual ROI(s): {e!s} (input {spec!r}); "
                      f"falling back to mean.")
                source = "mean"
            else:
                dff_full_path = plane0 / "r0p7_dff.memmap.float32"
                if not dff_full_path.exists():
                    print(f"[GUI] {dff_full_path.name} missing; cannot "
                          f"extract manual ROI(s); falling back to mean.")
                    source = "mean"
                else:
                    full = np.memmap(str(dff_full_path),
                                     dtype="float32", mode="r",
                                     shape=(T, N_total))
                    if len(roi_indices) == 1:
                        roi_idx = roi_indices[0]
                        trace = np.asarray(full[:, roi_idx],
                                           dtype=np.float32)
                        self._mean_dff = trace
                        excluded = "" if mask[roi_idx] \
                            else " [excluded by mask]"
                        self._trace_label = f"ROI {roi_idx}{excluded}"
                    else:
                        cols = np.asarray(full[:, roi_indices],
                                          dtype=np.float32)
                        agg = self.manual_agg_var.get()
                        reducer = (np.median if agg == "median"
                                   else np.mean)
                        self._mean_dff = reducer(
                            cols, axis=1).astype(np.float32)
                        n_excl = int(sum(1 for i in roi_indices
                                         if not mask[i]))
                        excl_str = (f" [{n_excl} excluded]"
                                    if n_excl else "")
                        self._trace_label = (
                            f"{agg} of ROIs "
                            f"{format_roi_indices(roi_indices)} "
                            f"(n={len(roi_indices)}){excl_str}")

        if source == "best":
            probs_full = np.load(prob_path).astype(np.float32)
            probs_kept = probs_full[mask] if probs_full.shape[0] == N_total \
                else probs_full
            best_in_kept = int(np.argmax(probs_kept))
            best_orig = int(kept_idx[best_in_kept])
            best_score = float(probs_kept[best_in_kept])
            trace = np.asarray(dff[:, best_in_kept], dtype=np.float32)
            self._mean_dff = trace
            self._trace_label = (f"ROI {best_orig} (score {best_score:.3f})")
        elif source in ("mean", "median"):
            agg = np.zeros(T, dtype=np.float32)
            from ...core import utils as _u
            chunk = max(1, min(T, _u.DFF_TIME_CHUNK))
            reducer = (np.median if source == "median" else np.mean)
            for t0 in range(0, T, chunk):
                t1 = min(T, t0 + chunk)
                agg[t0:t1] = reducer(np.asarray(dff[t0:t1, :]), axis=1)
            self._mean_dff = agg
            self._trace_label = f"{source} across {N_kept} kept ROIs"

        self._fps = float(utils.get_fps_from_notes(str(plane0)))

        N = self._mean_dff.size
        yf = np.fft.rfft(self._mean_dff)
        self._fft_xf = np.fft.rfftfreq(N, 1.0 / self._fps)
        self._fft_power = np.abs(yf) ** 2

    # -- Slider / entry binding --------------------------------------------

    def _on_slider(self, _value: str) -> None:
        if self._slider_job is not None:
            self.after_cancel(self._slider_job)
        self._slider_job = self.after(80, self._apply_cutoff)

    def _on_entry(self, _event=None) -> None:
        try:
            val = float(self.cutoff_str.get())
        except ValueError:
            self.status_var.set("Cutoff must be a number.")
            return
        val = max(self.CUTOFF_MIN, min(self.CUTOFF_MAX, val))
        self.cutoff_var.set(val)
        self._apply_cutoff()

    def _apply_cutoff(self) -> None:
        cutoff = float(self.cutoff_var.get())
        cutoff = max(self.CUTOFF_MIN, min(self.CUTOFF_MAX, cutoff))
        self.cutoff_str.set(f"{cutoff:.3f}")
        if self._mean_dff is None:
            return
        self._update_cutoff_marker(cutoff)
        self._draw_lowpass()

    # -- Plotting -----------------------------------------------------------

    def _draw_fft(self) -> None:
        if self._fft_xf is None or self._fft_power is None:
            return
        ax = self.fft_ax
        ax.clear(); ax.set_axis_on()
        xf = self._fft_xf[1:]; power = self._fft_power[1:]
        # Y-axis scale comes from the radio button next to the plot.
        # Log clips the floor so log10 doesn't underflow; linear shows
        # raw values so dominant peaks (e.g. line frequency) are obvious.
        yscale = self.fft_yscale_var.get()
        if yscale == "linear":
            ax.plot(xf, power, lw=0.8, color="black")
            ax.set_yscale("linear")
            ax.set_ylabel("Power", fontsize=8)
        else:
            ax.semilogy(xf, np.maximum(power, 1e-12), lw=0.8, color="black")
            ax.set_ylabel("Power (log)", fontsize=8)
        ax.set_xlim(0, min(self._fps / 2.0, 15.0))
        ax.set_xlabel("Frequency (Hz)", fontsize=8)
        ax.set_title(f"FFT - {self._trace_label}", fontsize=9)
        ax.tick_params(labelsize=7)
        self._cutoff_line = ax.axvline(
            self.cutoff_var.get(), color="tab:red", lw=1.2,
            linestyle="--", label=f"cutoff {self.cutoff_var.get():.2f} Hz")
        ax.legend(loc="upper right", fontsize=7)
        self.fft_canvas.draw_idle()

    def _update_cutoff_marker(self, cutoff: float) -> None:
        if self._cutoff_line is None:
            return
        self._cutoff_line.set_xdata([cutoff, cutoff])
        self._cutoff_line.set_label(f"cutoff {cutoff:.2f} Hz")
        leg = self.fft_ax.get_legend()
        if leg is not None:
            leg.get_texts()[0].set_text(f"cutoff {cutoff:.2f} Hz")
        self.fft_canvas.draw_idle()

    def _apply_view_mode(self, y: np.ndarray) -> tuple[np.ndarray, str]:
        """Return (transformed trace, y-axis label suffix) for the
        current view-mode toggle. ``dff`` mode passes through; ``robust_z``
        mode normalises by the trace's median + 1.4826·MAD via
        ``core.utils.mad_z``. Computed lazily on the trace being plotted
        -- no memmap is touched.
        """
        try:
            mode = str(self.view_mode_var.get())
        except Exception:
            mode = "dff"
        if mode != "robust_z":
            return y, "dF/F"
        # Robust z: (x - median) / (1.4826 * MAD). ``mad_z`` returns
        # (z, median, mad); we only need the first element.
        from ...core import utils as core_utils
        z, _med, _mad = core_utils.mad_z(np.asarray(y, dtype=np.float32))
        return z.astype(np.float32, copy=False), "Robust z (median ± 1.4826·MAD)"

    def _on_view_mode_change(self) -> None:
        # Re-render both raw + low-pass panels (FFT is amplitude-scale-
        # agnostic so it doesn't need a redraw).
        self._draw_raw()
        self._draw_lowpass()

    def _draw_raw(self) -> None:
        if self._mean_dff is None:
            return
        y, ylabel = self._apply_view_mode(self._mean_dff)
        t = np.arange(y.size, dtype=np.float32) / self._fps
        ax = self.raw_ax
        ax.clear(); ax.set_axis_on()
        ax.plot(t, y, lw=0.6, color="black")
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(f"Raw {ylabel.split(' ')[0]} - {self._trace_label}",
                     fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, t[-1] if t.size else 1.0)
        self.raw_canvas.draw_idle()

    def _draw_lowpass(self) -> None:
        if self._mean_dff is None:
            return
        from . import logic as utils
        cutoff = float(self.cutoff_var.get())
        order = int(self._params.get("filter_order", 2))
        # Zero-phase here so the previewed trace lines up in time with
        # the raw trace shown above; the on-disk lowpass memmap that
        # feeds event detection still uses the causal filter.
        lp, _ = utils.lowpass_zero_phase_1d(
            self._mean_dff, fps=self._fps, cutoff_hz=cutoff,
            order=order, sos=None)
        # Apply the view-mode transform to the filtered trace so the
        # robust-z toggle is consistent across both panels. The robust-z
        # statistics (median, MAD) are recomputed on the filtered trace
        # rather than reused from the raw -- this matches how a
        # downstream consumer would interpret the figure.
        y, ylabel = self._apply_view_mode(lp)
        t = np.arange(y.size, dtype=np.float32) / self._fps
        ax = self.lp_ax
        ax.clear(); ax.set_axis_on()
        ax.plot(t, y, lw=0.7, color="tab:blue")
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel(f"{ylabel} low-pass @ {cutoff:.2f} Hz", fontsize=8)
        ax.set_title(f"Low-pass - {self._trace_label}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, t[-1] if t.size else 1.0)
        self.lp_canvas.draw_idle()

    def _on_source_change(self) -> None:
        if self._plane0 is None:
            return
        try:
            self._load_data(self._plane0)
        except Exception as e:
            self.status_var.set(f"Source switch error: {e}")
            return
        self._draw_fft(); self._draw_raw(); self._draw_lowpass()
        self.status_var.set(
            f"Source: {self._trace_label}  "
            f"T={self._mean_dff.size}  fps={self._fps:.2f}  "
            f"cutoff={self.cutoff_var.get():.3f} Hz")

    def _on_manual_entry(self, _event=None) -> None:
        # Snap source to manual and reload. Also fired on FocusOut so clicking
        # elsewhere applies the value without forcing the user to hit Enter.
        if self._plane0 is None:
            return
        self.source_var.set("manual")
        self._on_source_change()

    # -- Advanced parameters ------------------------------------------------

    def _on_advanced(self) -> None:
        if not open_advanced(
                self, "Low-pass filter - Advanced parameters",
                self.PARAM_SPEC, self._params):
            return
        new_min = float(self._params.get("cutoff_min", self.CUTOFF_MIN))
        new_max = float(self._params.get("cutoff_max", self.CUTOFF_MAX))
        if new_min < new_max:
            self.CUTOFF_MIN = new_min
            self.CUTOFF_MAX = new_max
            self.slider.configure(from_=new_min, to=new_max)
            cur = float(self.cutoff_var.get())
            self.cutoff_var.set(max(new_min, min(new_max, cur)))
            self._apply_cutoff()
        self.status_var.set(
            "Advanced parameters updated. The next 'Compute' click will "
            "use the new filter / derivative settings.")

    # -- Compute (writes the lowpass + derivative memmaps) ------------------

    def _on_compute(self) -> None:
        if self._compute_job is not None and self._compute_job.running():
            messagebox.showinfo(
                "Busy", "Low-pass / derivative compute already running.")
            return
        if self._plane0 is None or self._n_kept == 0 or self._T == 0:
            messagebox.showerror(
                "Not ready",
                "Load a plane0 with r0p7_filtered_dff.memmap.float32 first.")
            return
        cutoff = float(self.cutoff_var.get())
        cutoff = max(self.CUTOFF_MIN, min(self.CUTOFF_MAX, cutoff))

        plane0 = self._plane0
        T = self._T
        N = self._n_kept
        fps = self._fps

        self.compute_btn.configure(state="disabled")
        self.compute_progress.start()
        self.compute_status_var.set(
            f"Writing lowpass + derivative @ {cutoff:.3f} Hz ...")

        # Offload the filter write to a child process (not a thread):
        # the per-ROI scipy loop would otherwise hold the GIL and freeze
        # the GUI. ``T``/``N`` are re-derived inside the helper from the
        # source memmap, so only the path string + scalars cross the
        # process boundary. The done payload is the cutoff, matching
        # ``_on_compute_done``.
        from ...core.offload import run_offloaded
        from ...core import lowpass_run
        self._compute_job = run_offloaded(
            self, self._compute_queue,
            lowpass_run.compute_lowpass_offload,
            args=(str(plane0),),
            kwargs=dict(
                fps=fps,
                cutoff_hz=cutoff,
                filter_order=int(self._params.get("filter_order", 2)),
                sg_win_ms=int(self._params.get("sg_win_ms", 333)),
                sg_poly=int(self._params.get("sg_poly", 2)),
            ),
            progress_kind="status",
            poll_ms=self.POLL_MS,
        )

    def _on_compute_done(self, cutoff_hz: float) -> None:
        self.compute_progress.stop()
        self.compute_btn.configure(state="normal")
        self.compute_status_var.set(
            f"Wrote r0p7_filtered_dff_lowpass.memmap.float32 + "
            f"r0p7_filtered_dff_dt.memmap.float32 "
            f"@ {cutoff_hz:.3f} Hz")
        if self._plane0 is not None:
            try:
                self.state.set_lowpass_ready(self._plane0)
            except Exception as e:
                print(f"lowpass_ready publish error: {e}")

    def _on_compute_error(self, payload: str) -> None:
        self.compute_progress.stop()
        self.compute_btn.configure(state="normal")
        self.compute_status_var.set("Error.")
        if report_stage_error(self.state, payload):
            return
        messagebox.showerror(
            "Compute failed", payload.split("\n", 1)[0])

    def _on_export_figures(self, quiet: bool = False) -> None:
        """Re-render the Tab 4 lowpass figures (FFT, raw dF/F, lowpass
        dF/F) for the currently-loaded recording. Writes PNG + SVG into
        ``<save_folder>/calliope_figures/lowpass/`` and refreshes the
        recording's ``manifest.json``.

        ``quiet=True`` (called by the Tab 0 batch toggle) suppresses
        messageboxes; failures fall back to the status bar.
        """
        plane0 = self._plane0
        if plane0 is None:
            if not quiet:
                messagebox.showinfo("No data", "Load a recording first.")
            return
        from ...core.utils import save_folder_for_plane0
        save_folder = save_folder_for_plane0(plane0)
        figures_root = save_folder / "calliope_figures"
        lowpass_dir = figures_root / "lowpass"
        try:
            from ...core.lowpass_run import render_lowpass_figures
            from ...core.export_manifest import write_export_manifest
            from ...core.utils import safe_recording_id
            rec_id = safe_recording_id(plane0)
            cutoff_hz = float(self.cutoff_var.get())
            written = render_lowpass_figures(
                plane0, fps=self._fps, cutoff_hz=cutoff_hz,
                figures_dir=lowpass_dir,
                filter_order=int(self._params.get("filter_order", 2)),
            )
            manifest_path = write_export_manifest(
                figures_root,
                rec_id=rec_id, params=dict(self._params),
                plane0=plane0, ckpt_path=None,
            )
            n_figs = len(written) * 2  # PNG + SVG per panel
            self.status_var.set(
                f"Exported {n_figs} lowpass files -> {lowpass_dir}")
            if not quiet:
                messagebox.showinfo(
                    "Export complete",
                    f"Wrote {n_figs} lowpass figure files (PNG + SVG) "
                    f"to:\n  {lowpass_dir}\n\n"
                    f"Manifest:\n  {manifest_path}")
        except Exception as e:
            if not quiet:
                messagebox.showerror("Export failed", str(e))
            else:
                self.status_var.set(f"Export failed: {e}")
