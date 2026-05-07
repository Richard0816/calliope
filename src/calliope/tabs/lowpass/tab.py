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
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ...gui_common import (
    AppState, attach_fig_toolbar, format_roi_indices, open_advanced,
    parse_manual_roi_spec, spec_defaults,
)


class LowpassTab(ttk.Frame):
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
        super().__init__(master, padding=10)
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
        self._compute_worker: Optional[threading.Thread] = None
        self._n_kept: int = 0
        self._T: int = 0

        self._build_ui()
        self.after(self.POLL_MS, self._drain_compute_queue)
        state.subscribe_plane0(self._on_plane0)
        if state.plane0 is not None:
            self._on_plane0(state.plane0)

    # -- UI -----------------------------------------------------------------

    def _build_ui(self) -> None:
        header = ttk.LabelFrame(
            self, text="Low-pass filter (causal Butterworth, order 2)",
            padding=8)
        header.pack(fill="x", pady=(0, 6))

        row = ttk.Frame(header); row.pack(fill="x", pady=2)
        ttk.Label(row, text="Cutoff (Hz):", width=12).pack(side="left")
        self.cutoff_var = tk.DoubleVar(value=self.CUTOFF_DEFAULT)
        self.slider = ttk.Scale(
            row, from_=self.CUTOFF_MIN, to=self.CUTOFF_MAX,
            orient="horizontal", variable=self.cutoff_var,
            command=self._on_slider,
        )
        self.slider.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.cutoff_str = tk.StringVar(value=f"{self.CUTOFF_DEFAULT:.2f}")
        entry = ttk.Entry(row, textvariable=self.cutoff_str, width=8)
        entry.pack(side="left")
        entry.bind("<Return>", self._on_entry)
        ttk.Label(row, text="Hz").pack(side="left", padx=(2, 8))
        ttk.Button(row, text="Advanced...",
                   command=self._on_advanced).pack(side="right",
                                                   padx=(0, 6))
        ttk.Button(row, text="Reload from folder...",
                   command=self._reload_from_folder).pack(side="right")

        row = ttk.Frame(header); row.pack(fill="x", pady=(4, 0))
        ttk.Label(row, text="", width=12).pack(side="left")
        self.compute_btn = ttk.Button(
            row, text="Compute low-pass + derivative (write memmaps)",
            command=self._on_compute, state="disabled")
        self.compute_btn.pack(side="left")
        self.compute_progress = ttk.Progressbar(
            row, mode="indeterminate", length=160)
        self.compute_progress.pack(side="left", padx=12)
        self.compute_status_var = tk.StringVar(value="")
        ttk.Label(row, textvariable=self.compute_status_var,
                  font=("", 9, "italic")).pack(side="left")

        row = ttk.Frame(header); row.pack(fill="x", pady=(4, 0))
        ttk.Label(row, text="Trace source:", width=12).pack(side="left")
        self.source_var = tk.StringVar(value="mean")
        ttk.Radiobutton(
            row, text="Mean across kept ROIs",
            value="mean", variable=self.source_var,
            command=self._on_source_change,
        ).pack(side="left", padx=(0, 12))
        ttk.Radiobutton(
            row, text="Median across kept ROIs",
            value="median", variable=self.source_var,
            command=self._on_source_change,
        ).pack(side="left", padx=(0, 12))
        ttk.Radiobutton(
            row, text="Best-scoring ROI (max predicted_cell_prob)",
            value="best", variable=self.source_var,
            command=self._on_source_change,
        ).pack(side="left", padx=(0, 12))
        ttk.Radiobutton(
            row, text="Manual ROI(s):",
            value="manual", variable=self.source_var,
            command=self._on_source_change,
        ).pack(side="left")
        self.manual_roi_var = tk.StringVar(value="0")
        manual_entry = ttk.Entry(row, textvariable=self.manual_roi_var,
                                 width=18)
        manual_entry.pack(side="left", padx=(2, 0))
        manual_entry.bind("<Return>", self._on_manual_entry)
        manual_entry.bind("<FocusOut>", self._on_manual_entry)
        # Aggregation for multi-ROI manual input (single-ROI inputs ignore it).
        self.manual_agg_var = tk.StringVar(value="mean")
        manual_agg = ttk.Combobox(
            row, textvariable=self.manual_agg_var,
            values=["mean", "median"], state="readonly", width=7)
        manual_agg.pack(side="left", padx=(4, 0))
        manual_agg.bind("<<ComboboxSelected>>", self._on_manual_entry)

        self.status_var = tk.StringVar(
            value="Run detection first (or reload from a finished folder).")
        ttk.Label(self, textvariable=self.status_var,
                  font=("", 9, "italic")).pack(anchor="w", pady=(0, 6))

        body = ttk.Frame(self); body.pack(fill="both", expand=True)
        body.rowconfigure(0, weight=1, uniform="rows")
        body.rowconfigure(1, weight=1, uniform="rows")
        body.rowconfigure(2, weight=1, uniform="rows")
        body.columnconfigure(0, weight=1)

        f1 = ttk.LabelFrame(body, text="1. FFT power spectrum", padding=4)
        f1.grid(row=0, column=0, sticky="nsew", pady=(0, 4))
        self.fft_fig = plt.Figure(figsize=(8, 2.4), tight_layout=True)
        self.fft_ax = self.fft_fig.add_subplot(111)
        self.fft_ax.set_axis_off()
        self.fft_ax.text(0.5, 0.5, "No data", ha="center", va="center",
                         transform=self.fft_ax.transAxes)
        self.fft_canvas = FigureCanvasTkAgg(self.fft_fig, master=f1)
        attach_fig_toolbar(self.fft_canvas, f1)
        self.fft_canvas.get_tk_widget().pack(fill="both", expand=True)

        f2 = ttk.LabelFrame(body, text="2. Raw dF/F", padding=4)
        f2.grid(row=1, column=0, sticky="nsew", pady=(0, 4))
        self.raw_fig = plt.Figure(figsize=(8, 2.4), tight_layout=True)
        self.raw_ax = self.raw_fig.add_subplot(111)
        self.raw_ax.set_axis_off()
        self.raw_ax.text(0.5, 0.5, "No data", ha="center", va="center",
                         transform=self.raw_ax.transAxes)
        self.raw_canvas = FigureCanvasTkAgg(self.raw_fig, master=f2)
        attach_fig_toolbar(self.raw_canvas, f2)
        self.raw_canvas.get_tk_widget().pack(fill="both", expand=True)

        f3 = ttk.LabelFrame(body, text="3. Low-pass dF/F", padding=4)
        f3.grid(row=2, column=0, sticky="nsew")
        self.lp_fig = plt.Figure(figsize=(8, 2.4), tight_layout=True)
        self.lp_ax = self.lp_fig.add_subplot(111)
        self.lp_ax.set_axis_off()
        self.lp_ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=self.lp_ax.transAxes)
        self.lp_canvas = FigureCanvasTkAgg(self.lp_fig, master=f3)
        attach_fig_toolbar(self.lp_canvas, f3)
        self.lp_canvas.get_tk_widget().pack(fill="both", expand=True)

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
        self.status_var.set(f"Loading dF/F from {self._plane0} ...")
        self.update_idletasks()
        try:
            self._load_data(self._plane0)
        except Exception as e:
            self.status_var.set(f"Load error: {e}")
            self.compute_btn.config(state="disabled")
            return
        self._draw_fft()
        self._draw_raw()
        self._draw_lowpass()
        self.status_var.set(
            f"Loaded T={self._mean_dff.size}  N_kept={self._n_kept}  "
            f"fps={self._fps:.2f}  cutoff={self.cutoff_var.get():.3f} Hz")
        self.compute_btn.config(state="normal")

    def _load_data(self, plane0: Path) -> None:
        from . import logic as utils
        F = np.load(plane0 / "F.npy", mmap_mode="r")
        N_total, T = F.shape
        self._T = int(T)

        mask_path = plane0 / "predicted_cell_mask.npy"
        prob_path = plane0 / "predicted_cell_prob.npy"
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

        filtered_path = plane0 / "r0p7_filtered_dff.memmap.float32"
        if filtered_path.exists():
            dff = np.memmap(str(filtered_path), dtype="float32", mode="r",
                            shape=(T, N_kept))
            kept_idx = np.flatnonzero(mask)
        else:
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
            chunk = max(1, min(T, 8192))
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
        ax.semilogy(xf, np.maximum(power, 1e-12), lw=0.8, color="black")
        ax.set_xlim(0, min(self._fps / 2.0, 15.0))
        ax.set_xlabel("Frequency (Hz)", fontsize=8)
        ax.set_ylabel("Power (log)", fontsize=8)
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

    def _draw_raw(self) -> None:
        if self._mean_dff is None:
            return
        t = np.arange(self._mean_dff.size, dtype=np.float32) / self._fps
        ax = self.raw_ax
        ax.clear(); ax.set_axis_on()
        ax.plot(t, self._mean_dff, lw=0.6, color="black")
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("dF/F", fontsize=8)
        ax.set_title(f"Raw dF/F - {self._trace_label}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, t[-1] if t.size else 1.0)
        self.raw_canvas.draw_idle()

    def _draw_lowpass(self) -> None:
        if self._mean_dff is None:
            return
        from . import logic as utils
        cutoff = float(self.cutoff_var.get())
        order = int(self._params.get("filter_order", 2))
        lp, _, _ = utils.lowpass_causal_1d(
            self._mean_dff, fps=self._fps, cutoff_hz=cutoff,
            order=order, zi=None, sos=None)
        t = np.arange(lp.size, dtype=np.float32) / self._fps
        ax = self.lp_ax
        ax.clear(); ax.set_axis_on()
        ax.plot(t, lp, lw=0.7, color="tab:blue")
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel(f"dF/F low-pass @ {cutoff:.2f} Hz", fontsize=8)
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
        if self._compute_worker is not None and self._compute_worker.is_alive():
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

        self.compute_btn.config(state="disabled")
        self.compute_progress.start(12)
        self.compute_status_var.set(
            f"Writing lowpass + derivative @ {cutoff:.3f} Hz ...")

        def worker():
            try:
                self._run_compute(plane0, T, N, fps, cutoff)
                self._compute_queue.put(("done", cutoff))
            except Exception as e:
                self._compute_queue.put(
                    ("error", f"{e}\n{traceback.format_exc()}"))

        self._compute_worker = threading.Thread(target=worker, daemon=True)
        self._compute_worker.start()

    def _run_compute(self, plane0: Path, T: int, N: int,
                     fps: float, cutoff: float) -> None:
        """Causal lowpass + SG first derivative, per ROI, written into
        r0p7_filtered_dff_lowpass.memmap.float32 and
        r0p7_filtered_dff_dt.memmap.float32 (both shape (T, N) float32).
        """
        from . import logic as utils
        src_path = plane0 / "r0p7_filtered_dff.memmap.float32"
        if not src_path.exists():
            raise FileNotFoundError(
                f"Missing {src_path.name}; run tab 3 first.")
        src = np.memmap(str(src_path), dtype="float32", mode="r",
                        shape=(T, N))

        lp_path = plane0 / "r0p7_filtered_dff_lowpass.memmap.float32"
        dt_path = plane0 / "r0p7_filtered_dff_dt.memmap.float32"
        lp_mm = np.memmap(str(lp_path), dtype="float32", mode="w+",
                          shape=(T, N))
        dt_mm = np.memmap(str(dt_path), dtype="float32", mode="w+",
                          shape=(T, N))

        sg_win_ms = int(self._params.get("sg_win_ms", 333))
        sg_poly = int(self._params.get("sg_poly", 2))
        order = int(self._params.get("filter_order", 2))
        sos = None
        report_every = max(1, N // 20)
        t0 = time.time()
        for i in range(N):
            trace = np.asarray(src[:, i], dtype=np.float32)
            lp, _, sos = utils.lowpass_causal_1d(
                trace, fps=fps, cutoff_hz=cutoff, order=order,
                zi=None, sos=sos)
            dt = utils.sg_first_derivative_1d(
                lp, fps=fps, win_ms=sg_win_ms, poly=sg_poly)
            lp_mm[:, i] = lp
            dt_mm[:, i] = dt
            if (i + 1) % report_every == 0:
                self._compute_queue.put(
                    ("status",
                     f"{i + 1}/{N} ROIs ({time.time() - t0:.1f}s)"))

        lp_mm.flush()
        dt_mm.flush()
        del lp_mm, dt_mm, src

    def _drain_compute_queue(self) -> None:
        try:
            while True:
                kind, payload = self._compute_queue.get_nowait()
                if kind == "status":
                    self.compute_status_var.set(payload)
                elif kind == "done":
                    self.compute_progress.stop()
                    self.compute_btn.config(state="normal")
                    self.compute_status_var.set(
                        f"Wrote r0p7_filtered_dff_lowpass.memmap.float32 + "
                        f"r0p7_filtered_dff_dt.memmap.float32 "
                        f"@ {payload:.3f} Hz")
                    if self._plane0 is not None:
                        try:
                            self.state.set_lowpass_ready(self._plane0)
                        except Exception as e:
                            print(f"lowpass_ready publish error: {e}")
                elif kind == "error":
                    self.compute_progress.stop()
                    self.compute_btn.config(state="normal")
                    self.compute_status_var.set("Error.")
                    messagebox.showerror(
                        "Compute failed", payload.split("\n", 1)[0])
        except queue.Empty:
            pass
        self.after(self.POLL_MS, self._drain_compute_queue)
