"""calliope.tabs.suite2p.tab - Tab 3: Suite2p detection.

What this tab does, end to end
------------------------------
The biggest workhorse tab in the GUI. Drives the entire detection +
trace-extraction pipeline:

    shifted TIFF (from Tab 1)
        |
        v
    Sparsery + Cellpose ROI detection
        (calliope.core.sparse_plus_cellpose.run)
        |
        v
    Suite2p extract -> F.npy / Fneu.npy / spks.npy / iscell.npy / stat.npy
        |
        v
    dF/F computation with neuropil correction
        (writes r0p7_dff*.memmap.float32)
        |
        v
    Cell-filter CNN prediction
        (calliope.core.cellfilter.predict.predict_recording)
        writes predicted_cell_prob.npy + predicted_cell_mask.npy
        |
        v
    Filtered dF/F memmap (r0p7_filtered_dff*.memmap.float32) for the
        ROIs that passed the keep mask.

When the worker finishes, the tab publishes the populated plane0
folder onto ``AppState.plane0`` so Tabs 4-8 can pick up the work
without re-running anything.

What the user sees
------------------
- An ops profile dropdown + paths to the suite2p ops .npy file, the
  cell-filter checkpoint, and the AAV metadata CSV.
- An Advanced... dialog with every Sparsery, Cellpose, dF/F and
  classifier knob.
- A "Run detection" button that kicks off the worker thread.
- Three panels:
    * top: live console (stdout/stderr captured via ``QueueWriter``)
    * bottom-left:  raw detected ROIs over the chosen background image
    * bottom-right: ROIs that *passed* the cell-filter mask
- A "Reload from folder..." button to load a previously-detected
  recording without re-running anything.

Threading
---------
Detection takes minutes. The button handler ``_on_run`` snapshots
form values, swaps stdout/stderr to a ``QueueWriter`` so library
prints land in the GUI log, and launches a worker thread. The main
thread polls the queue every POLL_MS and updates the log + figures.

Writes r0p7_dff and r0p7_filtered_dff memmaps under <out_dir>/detection/
final/suite2p/plane0/.
"""

from __future__ import annotations

import contextlib
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

from .logic import PreprocessResult, summary_writer

from ...gui_common import (
    AppState, QueueWriter, attach_fig_toolbar, open_advanced, spec_defaults,
)


# Bundled resource files: suite2p ops .npy + AAV metadata .csv ship inside
# calliope/data/. The detection worker also lazy-imports the heavyweight
# detection modules from calliope.core (sparse_plus_cellpose, utils,
# cellfilter) — see the worker bodies below.
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class Suite2pTab(ttk.Frame):
    """Runs sparse_plus_cellpose -> dF/F -> cell filter on the preprocessed
    shifted TIFF and renders three panels: live console, raw detected ROIs,
    and ROIs surviving the cell-filter prediction mask.

    Class-attribute layout
    ----------------------
    ``DEFAULT_OPS_PATH`` / ``DEFAULT_CKPT_PATH`` / ``DEFAULT_AAV_CSV``
        File paths the form is pre-populated with on startup. Users
        can override via the Browse... buttons. The ops file ships
        bundled in ``calliope/data``; the checkpoint and AAV CSV
        live on the lab F: drive.

    ``KNOWN_BG_IMAGES``
        Background-image options shown in the dropdown above the
        spatial map (mean, max projection, correlation map, etc.).
        Each option is a (suite2p ops key, display label) pair.

    ``PARAM_SPEC``
        The full list of Sparsery / Cellpose / dF/F / classifier knobs
        the Advanced... dialog will expose.

    ``POLL_MS``
        Main-thread polling interval for draining the worker's log
        queue.
    """

    POLL_MS = 80

    DEFAULT_OPS_PATH = _DATA_DIR / "updated_settings.npy"
    DEFAULT_CKPT_PATH = Path(r"F:\cellfilter_checkpoints\best.pt")
    DEFAULT_AAV_CSV = _DATA_DIR / "human_SLE_2p_meta.csv"

    # ---- GCaMP variant -> tau lookup table ----
    # The Suite2p deconvolution step needs a tau (calcium decay
    # constant in seconds) that matches the GECI variant in the
    # recording. Picking it wrong throws off the per-cell spike
    # estimate. The user can either pick a variant explicitly
    # (skipping the AAV CSV lookup) or stay on "Auto" to let the
    # legacy lookup run.
    GCAMP_OPTIONS: list[tuple[str, Optional[float]]] = [
        ("Auto (from AAV CSV)", None),     # default
        ("GCaMP6f  (tau = 0.7 s)", 0.7),
        ("GCaMP6m  (tau = 1.0 s)", 1.0),
        ("GCaMP6s  (tau = 1.3 s)", 1.3),
        ("GCaMP8m  (tau = 0.137 s)", 0.137),
    ]

    KNOWN_BG_IMAGES: list[tuple[str, str]] = [
        ("meanImg",       "Mean"),
        ("meanImgE",      "Mean (enhanced)"),
        ("max_proj",      "Max projection"),
        ("Vcorr",         "Correlation"),
        ("refImg",        "Reg. ref"),
        ("meanImg_chan2", "Mean ch2"),
    ]
    DEFAULT_BG_KEY = "meanImgE"

    PARAM_SPEC: list = [
        # Sparsery (suite2p detection)
        {"name": "threshold_scaling", "label": "threshold_scaling",
         "type": "float", "default": 0.85, "group": "Sparsery"},
        {"name": "high_pass", "label": "high_pass (frames)",
         "type": "int", "default": 100, "group": "Sparsery"},
        {"name": "smooth_sigma", "label": "smooth_sigma",
         "type": "float", "default": 1.0, "group": "Sparsery"},
        {"name": "max_iterations", "label": "max_iterations",
         "type": "int", "default": 1500, "group": "Sparsery"},
        {"name": "spatial_scale", "label": "spatial_scale (0=auto)",
         "type": "int", "default": 0, "group": "Sparsery"},
        {"name": "preclassify", "label": "preclassify",
         "type": "float", "default": 0.0, "group": "Sparsery"},
        {"name": "hard_cap", "label": "ROI hard cap",
         "type": "int", "default": 60000, "group": "Sparsery",
         "help": "abort sparsery if it exceeds this many ROIs"},
        # Cellpose
        {"name": "cellpose_model_type", "label": "model_type",
         "type": "choice", "choices": ["cyto", "cyto2", "nuclei"],
         "default": "cyto2", "group": "Cellpose"},
        {"name": "cellpose_diameter", "label": "diameter (0=auto)",
         "type": "int", "default": 0, "group": "Cellpose"},
        {"name": "cellpose_flow_threshold", "label": "flow_threshold",
         "type": "float", "default": 0.8, "group": "Cellpose"},
        {"name": "cellpose_cellprob_threshold", "label": "cellprob_threshold",
         "type": "float", "default": -1.0, "group": "Cellpose"},
        # Merge
        {"name": "max_overlap", "label": "Cellpose merge max overlap",
         "type": "float", "default": 0.3, "group": "Merge",
         "help": "drop cellpose ROIs > this fraction covered by sparsery"},
        # dF/F
        {"name": "fps_override", "label": "FPS override (0=auto)",
         "type": "float", "default": 0.0, "group": "dF/F",
         "help": "non-zero overrides utils.get_fps_from_notes "
                 "(fallback default is 15.07)"},
        {"name": "neuropil_coef", "label": "Neuropil coefficient (r)",
         "type": "float", "default": 0.7, "group": "dF/F"},
        {"name": "perc", "label": "Baseline percentile",
         "type": "int", "default": 10, "group": "dF/F"},
        {"name": "win_sec", "label": "Rolling window (s)",
         "type": "float", "default": 45.0, "group": "dF/F",
         "help": "only used in 'rolling' baseline mode"},
        # Default lowpass written here (Tab 4 overwrites filtered_*)
        {"name": "default_lowpass_hz", "label": "Default low-pass (Hz)",
         "type": "float", "default": 1.0, "group": "Default low-pass"},
        {"name": "default_sg_win_ms", "label": "SG derivative window (ms)",
         "type": "int", "default": 333, "group": "Default low-pass"},
        {"name": "default_sg_poly", "label": "SG polynomial order",
         "type": "int", "default": 2, "group": "Default low-pass"},
        # Pixel <-> micrometre calibration (written to ops['pix_to_um']
        # after detection so Tabs 6 & 7 render spatial plots in µm).
        # Direct override wins over zoom; leave both at 0 to skip and
        # render in raw pixels.
        {"name": "scope_zoom", "label": "Scope zoom (0=skip)",
         "type": "float", "default": 0.0, "group": "Pixel scale",
         "help": "compute pix_to_um as (FOV_um / zoom) / Lx using the "
                 "lab reference FOV (3080.9 µm at 1x)"},
        {"name": "um_per_pixel", "label": "µm per pixel direct (0=skip)",
         "type": "float", "default": 0.0, "group": "Pixel scale",
         "help": "explicit calibration; takes priority over scope_zoom"},
        # GPU dF/F via analyze_output_gpu (CuPy)
        {"name": "use_gpu_dff", "label": "Use GPU for dF/F",
         "type": "bool", "default": True, "group": "GPU",
         "help": "falls back to CPU if CuPy missing or baseline mode "
                 "is 'rolling' (GPU only supports first-N-min mean)"},
        {"name": "gpu_roi_chunk", "label": "GPU ROI chunk (0=all)",
         "type": "int", "default": 0, "group": "GPU",
         "help": "split ROIs into chunks if VRAM is tight"},
    ]

    def __init__(self, master, state: AppState) -> None:
        super().__init__(master, padding=10)
        self.state = state
        self._log_queue: queue.Queue = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._final_plane0: Optional[Path] = None
        self._params: dict = spec_defaults(self.PARAM_SPEC)
        self._bg_var = tk.StringVar(value=self.DEFAULT_BG_KEY)
        self._panel_cache: Optional[dict] = None

        self._build_ui()
        self.after(self.POLL_MS, self._drain_log_queue)
        state.subscribe(self._on_preprocess_result)

    # -- UI -----------------------------------------------------------------

    def _build_ui(self) -> None:
        header = ttk.LabelFrame(
            self, text="Suite2p detection (sparsery + cellpose) "
                       "-> dF/F -> cell filter", padding=8)
        header.pack(fill="x", pady=(0, 6))

        row = ttk.Frame(header); row.pack(fill="x", pady=2)
        ttk.Label(row, text="Base ops:", width=12).pack(side="left")
        self.ops_var = tk.StringVar(value=str(self.DEFAULT_OPS_PATH))
        ttk.Entry(row, textvariable=self.ops_var).pack(
            side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(row, text="Browse...", command=self._browse_ops).pack(
            side="left")

        row = ttk.Frame(header); row.pack(fill="x", pady=2)
        ttk.Label(row, text="Cell filter:", width=12).pack(side="left")
        self.ckpt_var = tk.StringVar(value=str(self.DEFAULT_CKPT_PATH))
        ttk.Entry(row, textvariable=self.ckpt_var).pack(
            side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(row, text="Browse...", command=self._browse_ckpt).pack(
            side="left")

        row = ttk.Frame(header); row.pack(fill="x", pady=2)
        ttk.Label(row, text="dF/F baseline:", width=12).pack(side="left")
        self.baseline_var = tk.StringVar(value="rolling")
        ttk.Radiobutton(
            row, text="Rolling (45 s window, 10th pct)",
            value="rolling", variable=self.baseline_var,
        ).pack(side="left", padx=(0, 12))
        ttk.Radiobutton(
            row, text="First N minutes:",
            value="first_n", variable=self.baseline_var,
        ).pack(side="left")
        self.baseline_min_var = tk.StringVar(value="2")
        ttk.Entry(row, textvariable=self.baseline_min_var,
                  width=6).pack(side="left", padx=(4, 2))
        ttk.Label(row, text="min").pack(side="left")

        # ---- GCaMP variant picker (drives Suite2p's tau) ----
        # Default keeps the legacy AAV-CSV-driven lookup for users
        # who haven't migrated. Picking a specific variant short-
        # circuits the lookup and writes its tau directly into the
        # Suite2p ops dict. See GCAMP_OPTIONS class attribute above.
        row = ttk.Frame(header); row.pack(fill="x", pady=2)
        ttk.Label(row, text="GCaMP variant:", width=14).pack(side="left")
        self.gcamp_var = tk.StringVar(value=self.GCAMP_OPTIONS[0][0])
        gcamp_combo = ttk.Combobox(
            row, textvariable=self.gcamp_var, state="readonly",
            values=[label for label, _tau in self.GCAMP_OPTIONS],
            width=28,
        )
        gcamp_combo.pack(side="left", padx=(0, 8))
        ttk.Label(
            row,
            text=("Pick the GECI you injected; \"Auto\" reads the AAV "
                  "metadata CSV by recording filename."),
            foreground="gray", font=("", 8, "italic"),
        ).pack(side="left")

        row = ttk.Frame(header); row.pack(fill="x", pady=2)
        self.csv_dff_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            row,
            text="Also write filtered dF/F as CSV  "
                 "(column headers = Suite2p ROI ids)",
            variable=self.csv_dff_var,
        ).pack(side="left")

        row = ttk.Frame(header); row.pack(fill="x", pady=(4, 0))
        self.run_btn = ttk.Button(
            row, text="Run detection + cell filter",
            command=self._on_run, state="disabled")
        self.run_btn.pack(side="left")
        self.load_btn = ttk.Button(
            row, text="Load existing panels", command=self._on_load_existing,
            state="disabled")
        self.load_btn.pack(side="left", padx=(6, 0))
        self.summary_btn = ttk.Button(
            row, text="Save summary",
            command=self._on_save_summary, state="disabled")
        self.summary_btn.pack(side="left", padx=(6, 0))
        ttk.Button(row, text="Advanced...",
                   command=self._on_advanced).pack(side="left", padx=(6, 0))
        self.progress = ttk.Progressbar(row, mode="indeterminate", length=200)
        self.progress.pack(side="left", padx=12)
        self.status_var = tk.StringVar(value="Run preprocessing first.")
        ttk.Label(row, textvariable=self.status_var).pack(side="left")

        body = ttk.Frame(self); body.pack(fill="both", expand=True)
        body.rowconfigure(0, weight=2)
        body.rowconfigure(1, weight=0)
        body.rowconfigure(2, weight=3)
        body.columnconfigure(0, weight=1, uniform="cols")
        body.columnconfigure(1, weight=1, uniform="cols")

        log_frame = ttk.LabelFrame(body, text="1. Suite2p console", padding=4)
        log_frame.grid(row=0, column=0, columnspan=2, sticky="nsew",
                       pady=(0, 6))
        self.log = tk.Text(log_frame, height=10, wrap="word",
                           state="disabled", font=("Consolas", 9))
        self.log.pack(fill="both", expand=True, side="left")
        sb = ttk.Scrollbar(log_frame, orient="vertical",
                           command=self.log.yview)
        sb.pack(fill="y", side="right")
        self.log.config(yscrollcommand=sb.set)

        bg_row = ttk.LabelFrame(
            body, text="Background image (panels 2 & 3)", padding=4)
        bg_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        self._bg_radio_holder = ttk.Frame(bg_row)
        self._bg_radio_holder.pack(side="left", fill="x", expand=True)
        ttk.Label(
            self._bg_radio_holder,
            text="Run detection (or load existing) to choose a background.",
        ).pack(side="left")

        det_frame = ttk.LabelFrame(
            body, text="2. Detected ROIs (raw suite2p output)", padding=6)
        det_frame.grid(row=2, column=0, sticky="nsew", padx=(0, 3))
        self.det_fig = plt.Figure(figsize=(5, 5), tight_layout=True)
        self.det_ax = self.det_fig.add_subplot(111)
        self.det_ax.set_axis_off()
        self.det_ax.text(0.5, 0.5, "No detection yet", ha="center",
                         va="center", transform=self.det_ax.transAxes)
        self.det_canvas = FigureCanvasTkAgg(self.det_fig, master=det_frame)
        attach_fig_toolbar(self.det_canvas, det_frame)
        self.det_canvas.get_tk_widget().pack(fill="both", expand=True)

        fil_frame = ttk.LabelFrame(
            body, text="3. After cell-filter prediction mask", padding=6)
        fil_frame.grid(row=2, column=1, sticky="nsew", padx=(3, 0))
        self.fil_fig = plt.Figure(figsize=(5, 5), tight_layout=True)
        self.fil_ax = self.fil_fig.add_subplot(111)
        self.fil_ax.set_axis_off()
        self.fil_ax.text(0.5, 0.5, "No filter applied yet", ha="center",
                         va="center", transform=self.fil_ax.transAxes)
        self.fil_canvas = FigureCanvasTkAgg(self.fil_fig, master=fil_frame)
        attach_fig_toolbar(self.fil_canvas, fil_frame)
        self.fil_canvas.get_tk_widget().pack(fill="both", expand=True)

    # -- Handlers -----------------------------------------------------------

    def _browse_ops(self) -> None:
        path = filedialog.askopenfilename(
            title="Select base Suite2p ops file",
            filetypes=[("NumPy", "*.npy"), ("All files", "*.*")])
        if path:
            self.ops_var.set(path)

    def _browse_ckpt(self) -> None:
        path = filedialog.askopenfilename(
            title="Select cell-filter checkpoint",
            filetypes=[("PyTorch", "*.pt"), ("All files", "*.*")])
        if path:
            self.ckpt_var.set(path)

    def _on_preprocess_result(self, result: PreprocessResult) -> None:
        self.run_btn.config(state="normal")
        existing = self._candidate_plane0(result)
        if existing is not None and self._plane0_has_outputs(existing):
            self.load_btn.config(state="normal")
            self.status_var.set(
                f"Ready: {result.shifted_tiff.name}  "
                f"(existing detection at {existing} - 'Load existing' "
                f"will render without re-running)")
        else:
            self.load_btn.config(state="disabled")
            self.status_var.set(
                f"Ready: {result.shifted_tiff.name}  "
                f"({result.shape_yx[0]}x{result.shape_yx[1]})")

    @staticmethod
    def _candidate_plane0(result: PreprocessResult) -> Path:
        return result.out_dir / "detection" / "final" / "suite2p" / "plane0"

    @staticmethod
    def _plane0_has_outputs(plane0: Path) -> bool:
        return (plane0 / "stat.npy").exists() and \
               (plane0 / "ops.npy").exists()

    def _on_load_existing(self) -> None:
        result = self.state.result
        if result is None:
            messagebox.showerror(
                "Missing input",
                "Run preprocessing first so I know which folder to load.")
            return
        plane0 = self._candidate_plane0(result)
        if not self._plane0_has_outputs(plane0):
            messagebox.showerror(
                "No data",
                f"Expected stat.npy / ops.npy at:\n{plane0}\n"
                f"Run detection first.")
            return
        self._append_log(
            f"--- Loading existing panels from {plane0} ---")
        self.status_var.set("Loading existing panels...")
        self._final_plane0 = plane0
        try:
            self._draw_panels(plane0)
            self.status_var.set(f"Loaded -> {plane0}")
        except Exception as e:
            self.status_var.set(f"Render failed: {e}")
            self._append_log(f"[GUI] render error: {e}\n"
                             f"{traceback.format_exc()}")
            return
        try:
            self.state.set_plane0(plane0)
        except Exception as e:
            self._append_log(f"[GUI] plane0 publish error: {e}")

    def _on_advanced(self) -> None:
        if open_advanced(self, "Suite2p Detection - Advanced parameters",
                         self.PARAM_SPEC, self._params):
            self._append_log(
                "[GUI] Advanced parameters updated. Re-run detection "
                "to apply.")

    def _on_run(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            messagebox.showinfo("Busy", "Detection is already running.")
            return
        result = self.state.result
        if result is None:
            messagebox.showerror("Missing input", "Run preprocessing first.")
            return

        ops_path = self.ops_var.get().strip()
        if not ops_path or not Path(ops_path).exists():
            messagebox.showerror(
                "Missing ops", f"Base ops file not found:\n{ops_path}")
            return

        ckpt_path = self.ckpt_var.get().strip()
        ckpt_present = bool(ckpt_path) and Path(ckpt_path).exists()
        if not ckpt_present:
            self._append_log(
                f"[GUI] No cell-filter checkpoint at {ckpt_path}; "
                "the third panel will fall back to suite2p's iscell.")

        baseline_mode = self.baseline_var.get()
        baseline_min = 2.0
        if baseline_mode == "first_n":
            try:
                baseline_min = float(self.baseline_min_var.get().strip())
                if baseline_min <= 0:
                    raise ValueError("must be > 0")
            except ValueError as e:
                messagebox.showerror(
                    "Invalid baseline",
                    f"First-N-minute baseline must be a positive number "
                    f"(got {self.baseline_min_var.get()!r}): {e}")
                return

        # Resolve the GCaMP picker into a tau scalar (or ``None`` to
        # let the AAV-CSV lookup run as before). Picker rows are
        # ``(label, tau)`` pairs; we look up the user-chosen label.
        tau_override: Optional[float] = None
        gcamp_label = self.gcamp_var.get()
        for label, tau in self.GCAMP_OPTIONS:
            if label == gcamp_label:
                tau_override = tau
                break
        if tau_override is not None:
            self._append_log(
                f"[GUI] GCaMP variant: {gcamp_label} -> tau={tau_override}")

        tiff_folder = result.shifted_tiff.parent
        save_folder = result.out_dir / "detection"
        rec_id = result.out_dir.name

        self.run_btn.config(state="disabled")
        self.progress.start(12)
        self.status_var.set("Running...")
        self._append_log(
            f"--- Starting detection on {tiff_folder.name} "
            f"(save -> {save_folder}) ---")
        self._final_plane0 = None

        def worker():
            try:
                self._run_pipeline(
                    tiff_folder=tiff_folder,
                    save_folder=save_folder,
                    ops_path=ops_path,
                    ckpt_path=(ckpt_path if ckpt_present else None),
                    rec_id=rec_id,
                    baseline_mode=baseline_mode,
                    baseline_min=baseline_min,
                    tau_override=tau_override,
                )
                self._log_queue.put(("done", None))
            except Exception as e:
                self._log_queue.put(
                    ("error", f"{e}\n{traceback.format_exc()}"))

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    # -- Worker -------------------------------------------------------------

    def _run_pipeline(
        self,
        tiff_folder: Path,
        save_folder: Path,
        ops_path: str,
        ckpt_path: Optional[str],
        rec_id: str,
        baseline_mode: str,
        baseline_min: float,
        tau_override: Optional[float] = None,
    ) -> None:
        writer = QueueWriter(self._log_queue)
        params = dict(self._params)
        sparsery_ops = {
            "high_pass":         params.get("high_pass", 100),
            "preclassify":       params.get("preclassify", 0.0),
            "smooth_sigma":      params.get("smooth_sigma", 1.0),
            "sparse_mode":       True,
            "spatial_scale":     params.get("spatial_scale", 0),
            "threshold_scaling": params.get("threshold_scaling", 0.85),
            "max_iterations":    params.get("max_iterations", 1500),
        }
        cellpose_cfg = {
            "cellpose_model_type":
                params.get("cellpose_model_type", "cyto2"),
            "cellpose_diameter":
                params.get("cellpose_diameter", 0),
            "cellpose_flow_threshold":
                params.get("cellpose_flow_threshold", 0.8),
            "cellpose_cellprob_threshold":
                params.get("cellpose_cellprob_threshold", -1.0),
            "cellpose_channel_input": "meanImg",
        }
        hard_cap = int(params.get("hard_cap", 60000))
        max_overlap = float(params.get("max_overlap", 0.3))

        with contextlib.redirect_stdout(writer), \
                contextlib.redirect_stderr(writer):
            print("[GUI] running sparse_plus_cellpose...")
            from ...core import sparse_plus_cellpose as spc
            aav_csv = (str(self.DEFAULT_AAV_CSV)
                       if self.DEFAULT_AAV_CSV.exists() else None)
            final_plane0 = spc.run(
                tiff_folder=str(tiff_folder),
                save_folder=str(save_folder),
                path_to_ops=ops_path,
                sparsery_ops=sparsery_ops,
                cellpose_cfg=cellpose_cfg,
                hard_cap=hard_cap,
                max_overlap=max_overlap,
                aav_info_csv=aav_csv,
                tau_vals=spc.DEFAULT_TAU_VALS,
                tau_override=tau_override,
                verbose=True,
            )
            print(f"[GUI] suite2p plane0 -> {final_plane0}")
            self._log_queue.put(("plane0", final_plane0))

            self._stamp_pix_to_um(final_plane0, params)

            self._run_dff(final_plane0, baseline_mode, baseline_min)

            if ckpt_path:
                print("[GUI] running cell-filter prediction...")
                self._run_cellfilter(final_plane0, ckpt_path, rec_id)
                print("[GUI] predicted_cell_mask.npy written")

            self._run_filtered_dff(final_plane0)

    def _stamp_pix_to_um(self, plane0: Path, params: dict) -> None:
        """Write the resolved µm-per-pixel onto ``plane0/ops.npy``.

        Tabs 6 (clustering colormap) and 7 (spatial propagation) read
        ``ops['pix_to_um']`` to render spatial axes in µm. We resolve
        from either the user-supplied direct calibration or the scope
        zoom (combined with the recording's ``Lx``) and stamp the
        result so downstream tabs are oblivious to the source.

        Skipped silently if the user left both inputs at 0.
        """
        from ...core.scale import resolve_pix_to_um
        from ...core import utils
        zoom = params.get("scope_zoom", 0.0) or None
        um_px = params.get("um_per_pixel", 0.0) or None
        if zoom is None and um_px is None:
            return
        view = utils.load_plane_view(plane0)
        if not view:
            print(f"[GUI] pix_to_um: no plane outputs at {plane0}; "
                  "skipping calibration stamp")
            return
        try:
            resolved = resolve_pix_to_um(
                view, zoom=zoom, um_per_pixel=um_px,
            )
        except (TypeError, ValueError) as e:
            print(f"[GUI] pix_to_um: invalid calibration ({e}); skipping")
            return
        if resolved is None:
            return
        utils.save_pix_to_um(plane0, float(resolved))
        source = "direct µm/px" if um_px is not None else f"zoom {zoom}"
        print(f"[GUI] pix_to_um = {resolved:.4f} µm/px (source: {source})")

    def _run_dff(
        self, plane0: Path, baseline_mode: str, baseline_min: float,
    ) -> None:
        """Inline dF/F + low-pass + derivative computation. Writes:
            plane0/r0p7_dff.memmap.float32          (T, N)
            plane0/r0p7_dff_lowpass.memmap.float32  (T, N)  @ default 1 Hz
            plane0/r0p7_dff_dt.memmap.float32       (T, N)
        Tries the GPU path first when ``use_gpu_dff`` is enabled and the
        baseline mode is 'first_n'; otherwise falls back to per-cell CPU.
        """
        from ...core import utils
        F = np.load(plane0 / "F.npy", allow_pickle=False)
        Fneu = np.load(plane0 / "Fneu.npy", allow_pickle=False)
        if F.shape != Fneu.shape or F.ndim != 2:
            raise ValueError(
                f"unexpected F/Fneu shapes: {F.shape}, {Fneu.shape}")
        F = np.asarray(F, dtype=np.float32, order="C")
        Fneu = np.asarray(Fneu, dtype=np.float32, order="C")
        N, T = F.shape

        fps_override = float(self._params.get("fps_override", 0.0))
        if fps_override > 0:
            fps = fps_override
            print(f"[GUI] FPS override active: {fps:.4f} Hz")
        else:
            fps = float(utils.get_fps_from_notes(str(plane0)))
        if baseline_mode == "first_n":
            print(f"[GUI] dF/F: first-{baseline_min:g}-min baseline "
                  f"(fps={fps}, n_baseline_frames~"
                  f"{int(baseline_min * 60 * fps)})")
        else:
            print(f"[GUI] dF/F: rolling baseline (fps={fps})")

        cutoff_hz_default = float(
            self._params.get("default_lowpass_hz", 1.0))
        sg_win_ms_default = int(self._params.get("default_sg_win_ms", 333))
        sg_poly_default = int(self._params.get("default_sg_poly", 2))
        r_default = float(self._params.get("neuropil_coef", 0.7))

        if self._maybe_run_dff_gpu(
                plane0=plane0, baseline_mode=baseline_mode,
                baseline_min=baseline_min, fps=fps,
                cutoff_hz=cutoff_hz_default,
                sg_win_ms=sg_win_ms_default, sg_poly=sg_poly_default,
                r=r_default):
            return
        cutoff_hz = float(self._params.get("default_lowpass_hz", 1.0))
        sg_win_ms = int(self._params.get("default_sg_win_ms", 333))
        sg_poly = int(self._params.get("default_sg_poly", 2))
        r = float(self._params.get("neuropil_coef", 0.7))
        perc = int(self._params.get("perc", 10))
        win_sec = float(self._params.get("win_sec", 45.0))
        print(f"[GUI] low-pass default cutoff {cutoff_hz:.2f} Hz; "
              f"SG derivative win {sg_win_ms} ms / poly {sg_poly}; "
              f"neuropil r={r:.2f}, baseline pct={perc}, win={win_sec:g}s")

        shape = (T, N)
        dff_path = plane0 / "r0p7_dff.memmap.float32"
        lp_path = plane0 / "r0p7_dff_lowpass.memmap.float32"
        dt_path = plane0 / "r0p7_dff_dt.memmap.float32"
        dff_mm = np.memmap(str(dff_path), mode="w+", dtype="float32",
                           shape=shape)
        lp_mm = np.memmap(str(lp_path), mode="w+", dtype="float32",
                          shape=shape)
        dt_mm = np.memmap(str(dt_path), mode="w+", dtype="float32",
                          shape=shape)

        batch = max(8, utils.change_batch_according_to_free_ram() * 20)
        sos = None

        t0 = time.time()
        for i0 in range(0, N, batch):
            i1 = min(N, i0 + batch)
            for i in range(i0, i1):
                trace = (F[i, :] - r * Fneu[i, :]).astype(np.float32)
                if baseline_mode == "first_n":
                    dff = utils.first_n_min_df_over_f_1d(
                        trace, baseline_min=baseline_min,
                        perc=perc, fps=fps)
                else:
                    dff = utils.robust_df_over_f_1d(
                        trace, win_sec=win_sec, perc=perc, fps=fps)
                lp, _, sos = utils.lowpass_causal_1d(
                    dff, fps=fps, cutoff_hz=cutoff_hz,
                    order=2, zi=None, sos=sos)
                dt = utils.sg_first_derivative_1d(
                    lp, fps=fps, win_ms=sg_win_ms, poly=sg_poly)
                dff_mm[:, i] = dff
                lp_mm[:, i] = lp
                dt_mm[:, i] = dt
            dff_mm.flush(); lp_mm.flush(); dt_mm.flush()
            print(f"[GUI] dF/F batch {i0}-{i1 - 1}/{N - 1} "
                  f"({time.time() - t0:.1f}s)")

        del dff_mm, lp_mm, dt_mm
        print(f"[GUI] r0p7_dff / r0p7_dff_lowpass / r0p7_dff_dt "
              f"memmaps written ({time.time() - t0:.1f}s)")

    def _maybe_run_dff_gpu(
        self, plane0: Path, baseline_mode: str, baseline_min: float,
        fps: float, cutoff_hz: float, sg_win_ms: int, sg_poly: int,
        r: float,
    ) -> bool:
        """If the GPU path is enabled and usable, run it and return True.
        The GPU module only supports a 'first-N seconds mean' baseline,
        so we skip it for the rolling mode.
        """
        if not bool(self._params.get("use_gpu_dff", True)):
            print("[GUI] GPU dF/F disabled in Advanced parameters; "
                  "using CPU.")
            return False
        if baseline_mode != "first_n":
            print("[GUI] GPU dF/F only supports first-N-min baseline; "
                  "rolling mode falls back to CPU.")
            return False
        try:
            import analyze_output_gpu as gpu
        except Exception as e:
            print(f"[GUI] analyze_output_gpu not importable ({e}); "
                  "using CPU.")
            return False
        if not getattr(gpu, "_CUPY_OK", False):
            print("[GUI] CuPy not available; using CPU dF/F.")
            return False

        baseline_sec = float(baseline_min) * 60.0
        roi_chunk = int(self._params.get("gpu_roi_chunk", 0))
        if roi_chunk <= 0:
            roi_chunk = None
        F = np.load(plane0 / "F.npy", allow_pickle=False)
        Fneu = np.load(plane0 / "Fneu.npy", allow_pickle=False)
        print(f"[GUI] dF/F (GPU): baseline_sec={baseline_sec:g}  "
              f"cutoff={cutoff_hz:g} Hz  sg=({sg_win_ms} ms, p{sg_poly})")
        try:
            t0 = time.time()
            gpu.process_suite2p_traces_gpu(
                F, Fneu, fps=fps, r=r,
                baseline_sec=baseline_sec,
                cutoff_hz=cutoff_hz,
                sg_win_ms=sg_win_ms, sg_poly=sg_poly,
                out_dir=str(plane0), prefix="r0p7_",
                roi_chunk=roi_chunk,
            )
            print(f"[GUI] GPU dF/F + lowpass + dt complete "
                  f"({time.time() - t0:.1f}s)")
            return True
        except Exception as e:
            print(f"[GUI] GPU dF/F failed ({e}); falling back to CPU.")
            return False

    def _run_filtered_dff(self, plane0: Path) -> None:
        """Slice r0p7_dff.memmap.float32 down to ROIs that pass the
        cell-filter mask (or suite2p iscell as a fallback) and write
        r0p7_filtered_dff.memmap.float32.
        """
        F = np.load(plane0 / "F.npy", mmap_mode="r")
        N_total, T = F.shape

        mask = self._load_keep_mask(plane0, N_total)
        N_kept = int(mask.sum())
        if N_kept == 0:
            print("[GUI] filtered dF/F: 0 ROIs survive the mask; "
                  "skipping filtered memmap")
            return

        dff_path = plane0 / "r0p7_dff.memmap.float32"
        filtered_path = plane0 / "r0p7_filtered_dff.memmap.float32"
        dff = np.memmap(str(dff_path), dtype="float32", mode="r",
                        shape=(T, N_total))
        filt = np.memmap(str(filtered_path), dtype="float32", mode="w+",
                         shape=(T, N_kept))
        chunk = max(1, min(T, 4096))
        for t0 in range(0, T, chunk):
            t1 = min(T, t0 + chunk)
            filt[t0:t1, :] = dff[t0:t1, :][:, mask]
        filt.flush()
        del filt, dff
        # Also persist the keep-mask under the legacy filename downstream tools
        # (clustering_cmap, cluster_mean_dff_gui, Fig1/Fig4, bins_visualization,
        # older crosscorr code paths) still hardcode. Matches
        # run_full_pipeline.make_filtered_memmaps so GUI and CLI runs produce
        # the same on-disk layout.
        np.save(plane0 / "r0p7_cell_mask_bool.npy", mask)
        print(f"[GUI] r0p7_filtered_dff.memmap.float32 written "
              f"(T={T}, N_kept={N_kept}/{N_total})  "
              f"+ r0p7_cell_mask_bool.npy")

        if self.csv_dff_var.get():
            self._write_filtered_dff_csv(plane0, T=T, mask=mask)

    def _write_filtered_dff_csv(self, plane0: Path, *, T: int,
                                mask: np.ndarray) -> None:
        """Mirror r0p7_filtered_dff.memmap.float32 to a CSV with one column
        per kept ROI, headed by its Suite2p ROI index (``roi_<i>``)."""
        import pandas as pd

        filtered_path = plane0 / "r0p7_filtered_dff.memmap.float32"
        csv_path = plane0 / "r0p7_filtered_dff.csv"
        N_kept = int(mask.sum())
        s2p_ids = np.where(mask)[0].astype(int)
        headers = [f"roi_{int(i)}" for i in s2p_ids]

        filt = np.memmap(str(filtered_path), dtype="float32", mode="r",
                         shape=(T, N_kept))
        chunk = max(1, min(T, 4096))
        first = True
        for t0 in range(0, T, chunk):
            t1 = min(T, t0 + chunk)
            chunk_df = pd.DataFrame(
                np.asarray(filt[t0:t1, :]),
                columns=headers,
                index=np.arange(t0, t1),
            )
            chunk_df.index.name = "frame"
            chunk_df.to_csv(
                csv_path,
                mode="w" if first else "a",
                header=first,
                float_format="%.6g",
            )
            first = False
        del filt
        print(f"[GUI] {csv_path.name} written "
              f"({T} frames, {N_kept} ROIs, Suite2p ids in headers)")

    def _run_cellfilter(
        self, plane0: Path, ckpt_path: str, rec_id: str,
    ) -> None:
        import torch
        from ...core.cellfilter.model import CellFilter
        from ...core.cellfilter.predict import predict_recording

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  device: {device}   ckpt: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = CellFilter().to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        predict_recording(rec_id, model, device, plane0=plane0)

    # -- Log queue + completion --------------------------------------------

    def _drain_log_queue(self) -> None:
        try:
            while True:
                kind, payload = self._log_queue.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "plane0":
                    self._final_plane0 = Path(payload)
                elif kind == "done":
                    self._on_done()
                elif kind == "error":
                    self._on_error(payload)
        except queue.Empty:
            pass
        self.after(self.POLL_MS, self._drain_log_queue)

    def _append_log(self, text: str) -> None:
        self.log.config(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def _on_done(self) -> None:
        self.progress.stop()
        self.run_btn.config(state="normal")
        plane0 = self._final_plane0
        if plane0 is None:
            self.status_var.set("Done (no plane0 returned).")
            return
        try:
            self._draw_panels(plane0)
            self.status_var.set(f"Done -> {plane0}")
        except Exception as e:
            self.status_var.set(f"Done, render failed: {e}")
            self._append_log(f"[GUI] render error: {e}\n"
                             f"{traceback.format_exc()}")
        try:
            self.state.set_plane0(plane0)
        except Exception as e:
            self._append_log(f"[GUI] plane0 publish error: {e}")

    def _on_error(self, msg: str) -> None:
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.status_var.set("Error.")
        self._append_log(f"ERROR: {msg}")
        messagebox.showerror("Detection failed", msg.split("\n", 1)[0])

    # -- Panel rendering ----------------------------------------------------

    def _draw_panels(self, plane0: Path) -> None:
        # Load ops only to extract the handful of 2D background images we
        # actually display. Suite2p 1.0's reg_outputs.npy / detect_outputs.npy
        # carry the preview images we want (mean, max_proj, refImg, Vcorr,
        # ...). load_plane_view merges them into a flat dict; pull just the
        # ndarrays in KNOWN_BG_IMAGES (cast to float32 once, since that's
        # the dtype the renderer needs) and let the rest fall out of scope
        # at the end of this function.
        from ...core import utils as core_utils
        view = core_utils.load_plane_view(plane0)
        bg_images: dict[str, np.ndarray] = {}
        for key, _label in self.KNOWN_BG_IMAGES:
            img = view.get(key)
            if isinstance(img, np.ndarray) and img.ndim == 2:
                bg_images[key] = np.ascontiguousarray(img, dtype=np.float32)
        del view  # release every other field

        stat = np.load(plane0 / "stat.npy", allow_pickle=True)
        n_total = len(stat)

        keep = self._load_keep_mask(plane0, n_total)
        prob_path = plane0 / "predicted_cell_prob.npy"
        probs = (np.load(prob_path).astype(np.float32)
                 if prob_path.exists() else None)

        self._panel_cache = dict(
            plane0=plane0, bg_images=bg_images, stat=stat,
            n_total=n_total, keep=keep, probs=probs,
        )

        self._populate_bg_radios(bg_images)
        self._redraw_with_bg()

        self.summary_btn.config(state="normal")
        try:
            self._write_summary(plane0, stat, keep, probs)
        except Exception as e:
            self._append_log(f"[GUI] summary write failed: {e}")

    def _populate_bg_radios(self, bg_images: dict) -> None:
        for w in self._bg_radio_holder.winfo_children():
            w.destroy()

        available = [(key, label) for key, label in self.KNOWN_BG_IMAGES
                     if key in bg_images]

        if not available:
            ttk.Label(self._bg_radio_holder,
                      text="(no 2D images in ops.npy)").pack(side="left")
            return

        keys = {k for k, _ in available}
        if self._bg_var.get() not in keys:
            self._bg_var.set(self.DEFAULT_BG_KEY
                             if self.DEFAULT_BG_KEY in keys
                             else available[0][0])

        for key, label in available:
            ttk.Radiobutton(
                self._bg_radio_holder, text=label, value=key,
                variable=self._bg_var,
                command=self._on_bg_changed,
            ).pack(side="left", padx=(0, 10))

    def _on_bg_changed(self) -> None:
        if self._panel_cache is not None:
            self._redraw_with_bg()

    def _redraw_with_bg(self) -> None:
        c = self._panel_cache
        if c is None:
            return
        bg_images = c["bg_images"]
        stat = c["stat"]
        n_total = c["n_total"]
        keep = c["keep"]
        probs = c["probs"]

        bg_key = self._bg_var.get()
        bg = bg_images.get(bg_key)
        bg_label = dict(self.KNOWN_BG_IMAGES).get(bg_key, bg_key)
        if bg is None:
            bg = bg_images.get("meanImgE", bg_images.get("meanImg"))
            bg_label = "Mean (fallback)"
        Ly, Lx = bg.shape

        vmax = float(np.quantile(bg, 0.995))
        vmin = float(np.quantile(bg, 0.01))

        label_all = self._build_label_image(stat, Ly, Lx)
        self._render_panel(
            self.det_ax, self.det_canvas, bg, label_all, vmin, vmax,
            f"All detected ROIs (n = {n_total})  [bg: {bg_label}]")

        kept_n = int(keep.sum())
        if probs is not None and probs.shape[0] == n_total:
            score_img = self._build_score_image(stat, keep, probs, Ly, Lx)
            title = (f"After filter (n = {kept_n} / {n_total})  "
                     f"[bg: {bg_label}, coloured by predicted_cell_prob]")
            self._render_score_panel(
                self.fil_ax, self.fil_canvas, bg, score_img,
                vmin, vmax, title)
        else:
            kept_stat = [s for i, s in enumerate(stat) if keep[i]]
            label_kept = self._build_label_image(kept_stat, Ly, Lx)
            keep_src = "iscell.npy (suite2p classifier)"
            self._render_panel(
                self.fil_ax, self.fil_canvas, bg, label_kept,
                vmin, vmax,
                f"After filter (n = {kept_n} / {n_total})  "
                f"[bg: {bg_label}, {keep_src}]")

    @staticmethod
    def _build_label_image(stat, Ly: int, Lx: int) -> np.ndarray:
        label = np.zeros((Ly, Lx), dtype=np.int32)
        for i, s in enumerate(stat, start=1):
            yp = np.asarray(s["ypix"]); xp = np.asarray(s["xpix"])
            ok = (yp >= 0) & (yp < Ly) & (xp >= 0) & (xp < Lx)
            label[yp[ok], xp[ok]] = i
        return label

    @staticmethod
    def _build_score_image(stat, keep: np.ndarray, probs: np.ndarray,
                           Ly: int, Lx: int) -> np.ndarray:
        img = np.full((Ly, Lx), np.nan, dtype=np.float32)
        for i, s in enumerate(stat):
            if not keep[i]:
                continue
            yp = np.asarray(s["ypix"]); xp = np.asarray(s["xpix"])
            ok = (yp >= 0) & (yp < Ly) & (xp >= 0) & (xp < Lx)
            img[yp[ok], xp[ok]] = float(probs[i])
        return img

    @staticmethod
    def _load_keep_mask(plane0: Path, n_total: int) -> np.ndarray:
        mask_path = plane0 / "predicted_cell_mask.npy"
        if mask_path.exists():
            return np.load(mask_path).astype(bool)
        iscell_path = plane0 / "iscell.npy"
        if iscell_path.exists():
            ic = np.load(iscell_path)
            return (ic[:, 0] > 0) if ic.ndim == 2 else (ic > 0).astype(bool)
        return np.ones(n_total, dtype=bool)

    # -- Summary export -----------------------------------------------------

    def _write_summary(self, plane0: Path, stat, keep, probs) -> Path:
        prob_path = plane0 / "predicted_cell_prob.npy"
        p_cell = (np.load(prob_path)
                  if prob_path.exists() else None)
        iscell_path = plane0 / "iscell.npy"
        iscell_arr = None
        if iscell_path.exists():
            ic = np.load(iscell_path)
            iscell_arr = (ic[:, 0] > 0) if ic.ndim == 2 else (ic > 0)
        try:
            from ...core import utils
            fps = float(utils.get_fps_from_notes(str(plane0)))
        except Exception:
            fps = None
        summary_writer.update_recording_meta(
            plane0, fps=fps, T=None, N=int(len(stat)))
        return summary_writer.write_rois_sheet(
            plane0, stat, p_cell=p_cell,
            predicted_mask=keep, iscell=iscell_arr)

    def _on_save_summary(self) -> None:
        plane0 = self._final_plane0
        if plane0 is None:
            messagebox.showinfo(
                "No data", "Run detection or 'Load existing panels' first.")
            return
        try:
            stat = np.load(plane0 / "stat.npy", allow_pickle=True)
            n_total = len(stat)
            keep = self._load_keep_mask(plane0, n_total)
            prob_path = plane0 / "predicted_cell_prob.npy"
            probs = (np.load(prob_path).astype(np.float32)
                     if prob_path.exists() else None)
            path = self._write_summary(plane0, stat, keep, probs)
            self.status_var.set(f"Summary -> {path}")
        except Exception as e:
            messagebox.showerror("Summary failed", str(e))

    @staticmethod
    def _render_panel(ax, canvas, mean, label_img, vmin, vmax, title) -> None:
        ax.clear(); ax.set_axis_off()
        ax.imshow(mean, cmap="gray", vmin=vmin, vmax=vmax)
        if label_img.max() > 0:
            overlay = np.ma.masked_where(label_img == 0, label_img)
            ax.imshow(overlay, cmap="nipy_spectral", alpha=0.45,
                      interpolation="nearest")
        ax.set_title(title, fontsize=9)
        canvas.draw_idle()

    @staticmethod
    def _render_score_panel(ax, canvas, mean, score_img, vmin, vmax,
                            title) -> None:
        """Same overlay style as _render_panel but coloured by score across
        [0.5, 1.0] (the cell-filter pass range)."""
        fig = ax.figure
        for old_ax in list(fig.axes):
            if old_ax is not ax:
                fig.delaxes(old_ax)
        ax.clear(); ax.set_axis_off()
        ax.imshow(mean, cmap="gray", vmin=vmin, vmax=vmax)
        overlay = np.ma.masked_invalid(score_img)
        if overlay.count() > 0:
            im = ax.imshow(overlay, cmap="viridis", alpha=0.45,
                           vmin=0.5, vmax=1.0, interpolation="nearest")
            cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            cb.set_label("predicted_cell_prob (>= 0.5 only)",
                         fontsize=8)
            cb.ax.tick_params(labelsize=7)
        ax.set_title(title, fontsize=9)
        canvas.draw_idle()
