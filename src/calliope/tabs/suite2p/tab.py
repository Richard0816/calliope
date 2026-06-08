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
- A path picker for the cell-filter checkpoint (defaults to the
  bundled ``calliope/data/cellfilter_best.pt``) and a GCaMP-variant
  dropdown that resolves to the Suite2p deconvolution tau.
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

import customtkinter as ctk
from tkinter import filedialog, messagebox
from typing import Optional

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .logic import (
    PreprocessResult, build_label_image, build_score_image,
    candidate_plane0, count_leaves, load_keep_mask, plane0_has_outputs,
    render_background_panel, render_panel, render_score_panel, summary_writer,
)
from .curation_popout import CurationPopout

from ...core.scale import FOV_UM_REFERENCE
from ...gui_common import (
    AppState, QueueWriter, attach_fig_toolbar, attach_resize_handle,
    drain_queue, open_advanced, report_stage_error, spec_defaults,
)


# Bundled resource files: the pretrained cell-filter checkpoint ships
# inside calliope/data/. The detection worker also lazy-imports the heavyweight
# detection modules from calliope.core (sparse_plus_cellpose, utils,
# cellfilter) — see the worker bodies below.
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class Suite2pTab(ctk.CTkFrame):
    """Runs sparse_plus_cellpose -> dF/F -> cell filter on the preprocessed
    shifted TIFF and renders three panels: live console, raw detected ROIs,
    and ROIs surviving the cell-filter prediction mask.

    Class-attribute layout
    ----------------------
    ``DEFAULT_OPS_PATH`` / ``DEFAULT_CKPT_PATH``
        File paths the form is pre-populated with on startup. Users
        can override via the Browse... buttons. Both the ops file and
        the cell-filter checkpoint ship bundled in ``calliope/data``.

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

    # Base settings come from calliope.core.calliope_settings (in source);
    # per-session edits go through the "Edit suite2p settings..." popout.
    DEFAULT_CKPT_PATH = _DATA_DIR / "cellfilter_best.pt"

    # ---- GCaMP variant -> tau lookup table ----
    # The Suite2p deconvolution step needs a tau (calcium decay
    # constant in seconds) that matches the GECI variant in the
    # recording. Picking it wrong throws off the per-cell spike
    # estimate, so the user must pick the variant that matches their
    # injection. GCaMP6m is the default since it's the most common
    # variant in the lab's standing recordings.
    #
    # Tau values per published in-vivo measurements / spike-inference
    # optimisations. Source paper per row:
    #   GCaMP6f/m/s : Chen et al., Nature 2013
    #                 https://doi.org/10.1038/nature12354
    #                 (Suite2p-recommended bins used by Pachitariu 2018
    #                  https://doi.org/10.1523/JNEUROSCI.3339-17.2018)
    #   jGCaMP7f   : Dana et al., Nat Methods 2019
    #                 https://doi.org/10.1038/s41592-019-0435-6
    #   jGCaMP8f/m/s base : Zhang et al., Nature 2023
    #                 https://doi.org/10.1038/s41586-023-05828-9
    #   jGCaMP8 in-vivo OASIS tau optimum : Rupprecht et al., bioRxiv 2025
    #                 https://doi.org/10.1101/2025.03.03.641129
    #                 (replaces the older 0.137 s value, which matches
    #                  the cultured-neuron half-decay rather than the
    #                  in-vivo OASIS optimum)
    GCAMP_OPTIONS: list[tuple[str, float]] = [
        ("jGCaMP8m (tau = 0.25 s)", 0.25), # default; Zhang 2023 / Rupprecht 2025
        ("GCaMP6m  (tau = 1.0 s)", 1.0),   # Chen 2013
        ("GCaMP6f  (tau = 0.7 s)", 0.7),   # Chen 2013
        ("GCaMP6s  (tau = 1.5 s)", 1.5),   # Chen 2013 + Pachitariu 2018 slow bin
        ("jGCaMP7f (tau = 0.7 s)", 0.7),   # Dana 2019
        ("jGCaMP8f (tau = 0.15 s)", 0.15), # Zhang 2023 / Rupprecht 2025
        ("jGCaMP8s (tau = 0.5 s)", 0.5),   # Zhang 2023 / Rupprecht 2025 optimum
    ]

    @classmethod
    def _resolve_gcamp_tau(
        cls, label: str
    ) -> tuple[float, str, str]:
        """Resolve a (possibly stale) GCaMP label to a (tau, canonical_label,
        match_kind) tuple.

        Three-stage match:

        1. **exact** -- label string is in ``GCAMP_OPTIONS`` verbatim. Trivial
           case. Returns the matched tau and the original label.
        2. **variant** -- label contains an unambiguous GCaMP variant code
           (``6f`` / ``6m`` / ``6s`` / ``7f`` / ``8f`` / ``8m`` / ``8s``).
           Used when a saved batch CSV / preference carries an OBSOLETE label
           (e.g. the pre-2026-05-26 ``"GCaMP8m  (tau = 0.137 s)"`` with the
           cultured-neuron tau). Returns the current canonical label + tau so
           the user gets the corrected value, the GUI can snap the StringVar
           forward, and the lab can audit which recordings got which tau.
        3. **fallback** -- label parses to nothing recognisable. Returns the
           first dropdown option's tau (the lab's default jGCaMP8m at 0.25 s)
           so the run still proceeds, with the caller responsible for
           surfacing a loud warning.

        Why this isn't a one-line ``dict.get``: the label format has
        changed at least once (the 2026-05-26 tau update went from
        ``"GCaMP8m  (tau = 0.137 s)"`` to ``"jGCaMP8m (tau = 0.25 s)"`` --
        both leading "j" and tau value changed), so an exact-match lookup
        loses information whenever a saved selection round-trips through a
        CSV. Pulling the variant code out is what makes the migration
        idempotent.
        """
        import re
        text = str(label or "")
        # Stage 1: exact match.
        for cand_label, cand_tau in cls.GCAMP_OPTIONS:
            if cand_label == text:
                return float(cand_tau), cand_label, "exact"
        # Stage 2: variant-code match. The variant code is the two
        # characters immediately after "GCaMP" (or "jGCaMP") in canonical
        # labels like "GCaMP8m" / "jGCaMP8m". Anchoring on the GCaMP
        # prefix avoids false positives on stray "8m" / "6f" substrings
        # appearing in unrelated text (e.g. a date "2026-06-08m") and
        # also handles the leading "j" / casing variation across the old
        # and new label schemas.
        m = re.search(r"j?GCaMP\s*([678][fms])(?![A-Za-z0-9])",
                      text, flags=re.IGNORECASE)
        if m:
            code = m.group(1).lower()
            # Map the variant code back to a canonical (label, tau) pair
            # via the dropdown itself. Look at the trailing letter+digit
            # in each canonical label and pick the matching row.
            for cand_label, cand_tau in cls.GCAMP_OPTIONS:
                lm = re.search(r"([678][fms])\b",
                               cand_label, flags=re.IGNORECASE)
                if lm and lm.group(1).lower() == code:
                    return float(cand_tau), cand_label, "variant"
        # Stage 3: fallback.
        fb_label, fb_tau = cls.GCAMP_OPTIONS[0]
        return float(fb_tau), fb_label, "fallback"

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
        # Cellpose-SAM second pass on the correlation image (Vcorr).
        # When enabled, runs Cellpose-SAM (cellpose >= 4.0) on Vcorr
        # after the primary Sparsery + cyto2 cellpose passes. ROIs from
        # this pass are filtered against Sparsery independently and
        # tagged ``_source='cellpose_sam'`` in ``stat.npy``. Adds 1-2x
        # detection time and silently skips if cellpose is 3.x.
        # Reference: Pachitariu et al. Cellpose-SAM bioRxiv 2025,
        # doi:10.1101/2025.04.28.651001.
        {"name": "enable_sam_vcorr_pass",
         "label": "Cellpose-SAM second pass on Vcorr",
         "type": "bool", "default": False, "group": "Cellpose-SAM (Tier 2)",
         "help": "recover silent cells via Cellpose-SAM on the "
                 "correlation image; needs cellpose >= 4.0"},
        {"name": "sam_flow_threshold",
         "label": "SAM flow_threshold", "type": "float", "default": 0.8,
         "group": "Cellpose-SAM (Tier 2)"},
        {"name": "sam_cellprob_threshold",
         "label": "SAM cellprob_threshold", "type": "float", "default": -1.0,
         "group": "Cellpose-SAM (Tier 2)"},
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
                 "reference FOV below"},
        {"name": "fov_um_reference", "label": "Reference FOV µm at 1x",
         "type": "float", "default": FOV_UM_REFERENCE, "group": "Pixel scale",
         "help": "FOV width in µm at 1x zoom for your rig; only used by the "
                 "scope-zoom path. Default is the lab's two-photon rig "
                 f"({FOV_UM_REFERENCE:.1f} µm)"},
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
        super().__init__(master, fg_color="transparent")
        self.state = state
        self._log_queue: queue.Queue = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._final_plane0: Optional[Path] = None
        self._params: dict = spec_defaults(self.PARAM_SPEC)
        # Per-session suite2p settings override produced by the
        # "Edit suite2p settings..." popout. Empty dict means "use the
        # in-source defaults from calliope.core.calliope_settings". Gets
        # passed to spc.run via the settings_override kwarg.
        self._settings_override: dict = {}
        # Post-detection archive (prune + zstd raw compression) runs in a
        # child process (see core.offload) instead of the detection worker
        # thread, so the slow level-19 compression can't freeze the GUI.
        # ``_pending_archive`` carries the paths/params from the worker to
        # the main-thread launcher; ``set_plane0`` (the batch-advance
        # signal) is deferred until the archive finishes, preserving the
        # old ordering where the row completed only after archiving.
        self._archive_job = None
        self._pending_archive: Optional[dict] = None
        self._bg_var = tk.StringVar(value=self.DEFAULT_BG_KEY)
        # Panel 3 ("After filter") ROI-overlay toggle. ON (default) paints
        # the kept-ROI overlay over the background; OFF shows the clean
        # background image only. Controls both the live panel and the
        # exported "kept_rois" figure.
        self._roi_overlay_var = tk.BooleanVar(value=True)
        self._panel_cache: Optional[dict] = None

        self._build_ui()
        drain_queue(self, self._log_queue,
                    {"log": self._append_log,
                     "plane0": self._on_plane0_payload,
                     "done": lambda _p: self._on_done(),
                     "error": self._on_error,
                     "archive_done": self._on_archive_done,
                     "archive_error": self._on_archive_error},
                    poll_ms=self.POLL_MS)
        state.subscribe(self._on_preprocess_result)

    # -- UI -----------------------------------------------------------------

    def _build_ui(self) -> None:
        header = ctk.CTkFrame(self)
        header.pack(fill="x", padx=10, pady=(10, 6))
        ctk.CTkLabel(
            header,
            text="Suite2p detection (sparsery + cellpose) "
                 "-> dF/F -> cell filter",
            font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(6, 0))

        # No "Base ops" file picker here: base settings come from
        # calliope.core.calliope_settings.CALLIOPE_BASE_SETTINGS (in-source)
        # and per-session edits go through the "Edit suite2p settings..."
        # popout next to the Run button.

        row = ctk.CTkFrame(header, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=2)
        ctk.CTkLabel(row, text="Cell filter:", width=90,
                     anchor="w").pack(side="left")
        self.ckpt_var = tk.StringVar(value=str(self.DEFAULT_CKPT_PATH))
        ctk.CTkEntry(row, textvariable=self.ckpt_var).pack(
            side="left", fill="x", expand=True, padx=(0, 4))
        ctk.CTkButton(row, text="Browse...", width=90,
                      command=self._browse_ckpt).pack(side="left")

        row = ctk.CTkFrame(header, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=2)
        ctk.CTkLabel(row, text="dF/F baseline:", width=90,
                     anchor="w").pack(side="left")
        # Default = first-N-minutes mode at 2 minutes. The lab's
        # standard recordings have a clean baseline at the start, and
        # rolling has both higher CPU cost and worse behaviour on long
        # epileptiform sweeps where the rolling window can sit inside
        # an event.
        self.baseline_var = tk.StringVar(value="first_n")
        ctk.CTkRadioButton(
            row, text="First N minutes:",
            value="first_n", variable=self.baseline_var,
        ).pack(side="left", padx=(0, 4))
        self.baseline_min_var = tk.StringVar(value="2")
        ctk.CTkEntry(row, textvariable=self.baseline_min_var,
                     width=60).pack(side="left", padx=(0, 2))
        ctk.CTkLabel(row, text="min").pack(side="left", padx=(0, 12))
        ctk.CTkRadioButton(
            row, text="Rolling (45 s window, 10th pct)",
            value="rolling", variable=self.baseline_var,
        ).pack(side="left")

        # ---- GCaMP variant picker (drives Suite2p's tau) ----
        row = ctk.CTkFrame(header, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=2)
        ctk.CTkLabel(row, text="GCaMP variant:", width=110,
                     anchor="w").pack(side="left")
        self.gcamp_var = tk.StringVar(value=self.GCAMP_OPTIONS[0][0])
        gcamp_combo = ctk.CTkComboBox(
            row, variable=self.gcamp_var, state="readonly",
            values=[label for label, _tau in self.GCAMP_OPTIONS],
            width=240,
        )
        gcamp_combo.pack(side="left", padx=(0, 8))
        # Optional custom tau override: when non-empty, this scalar wins
        # over the dropdown variant (for indicators not in the preset list,
        # or a rig-calibrated tau). Leave blank to use the dropdown.
        ctk.CTkLabel(row, text="or custom τ (s):", anchor="w").pack(
            side="left", padx=(0, 4))
        self.custom_tau_var = tk.StringVar(value="")
        ctk.CTkEntry(
            row, textvariable=self.custom_tau_var, width=70,
            placeholder_text="blank",
        ).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(
            row,
            text=("Pick the GECI you injected; the chosen variant sets "
                  "the suite2p tau. A custom τ overrides the dropdown."),
            text_color="gray", font=ctk.CTkFont(size=11, slant="italic"),
        ).pack(side="left")

        row = ctk.CTkFrame(header, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=2)
        self.csv_dff_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            row,
            text="Also write filtered dF/F as CSV  "
                 "(column headers = Suite2p ROI ids)",
            variable=self.csv_dff_var,
        ).pack(side="left")

        row = ctk.CTkFrame(header, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=(4, 0))
        self.run_btn = ctk.CTkButton(
            row, text="Run detection + cell filter", width=220,
            command=self._on_run, state="disabled")
        self.run_btn.pack(side="left")
        self.load_btn = ctk.CTkButton(
            row, text="Load existing panels", width=170,
            command=self._on_load_existing, state="disabled")
        self.load_btn.pack(side="left", padx=(6, 0))
        self.summary_btn = ctk.CTkButton(
            row, text="Save summary", width=120,
            command=self._on_save_summary, state="disabled")
        self.summary_btn.pack(side="left", padx=(6, 0))
        self.export_btn = ctk.CTkButton(
            row, text="Export figures", width=130,
            command=self._on_export_figures, state="disabled")
        self.export_btn.pack(side="left", padx=(6, 0))
        ctk.CTkButton(row, text="Advanced...", width=110,
                      command=self._on_advanced).pack(side="left",
                                                      padx=(6, 0))
        ctk.CTkButton(row, text="Edit suite2p settings...", width=180,
                      command=self._on_edit_suite2p_settings).pack(
                          side="left", padx=(6, 0))
        self.progress = ctk.CTkProgressBar(row, mode="indeterminate",
                                           width=200)
        self.progress.pack(side="left", padx=12)
        self.status_var = tk.StringVar(value="Run preprocessing first.")
        ctk.CTkLabel(row, textvariable=self.status_var).pack(side="left")

        # Persistent header showing which recording the panels +
        # popout are currently pointing at. Updated by
        # ``_draw_panels`` whenever a plane0 finishes loading.
        rec_row = ctk.CTkFrame(header, fg_color="transparent")
        rec_row.pack(fill="x", padx=8, pady=(4, 6))
        self.recording_var = tk.StringVar(value="No recording loaded.")
        ctk.CTkLabel(rec_row, textvariable=self.recording_var,
                     font=ctk.CTkFont(size=12, slant="italic")).pack(
            side="left")

        # Log on top with a draggable grip below it -- dragging extends
        # the log down without changing the size of the detection
        # panels beneath. The surrounding scrollable tab body absorbs
        # the extra height. CTkFrame wrap so the height drag actually
        # sticks.
        log_wrap = ctk.CTkFrame(self, fg_color="transparent")
        log_wrap.pack(fill="x", padx=14, pady=(0, 0))
        log_frame = ctk.CTkFrame(log_wrap)
        log_frame.pack(fill="both", expand=True)
        ctk.CTkLabel(log_frame, text="1. Suite2p console",
                     font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(6, 0))
        self.log = ctk.CTkTextbox(
            log_frame, height=180, wrap="word", state="disabled",
            font=ctk.CTkFont(family="Consolas", size=11))
        self.log.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        attach_resize_handle(self, log_wrap, min_height=160)

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=14, pady=(0, 10))
        body.rowconfigure(0, weight=0)
        body.rowconfigure(1, weight=1)
        body.columnconfigure(0, weight=1, uniform="cols")
        body.columnconfigure(1, weight=1, uniform="cols")

        bg_row = ctk.CTkFrame(body)
        bg_row.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        ctk.CTkLabel(bg_row, text="Background image (panels 2 & 3)",
                     font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(6, 0))
        self._bg_radio_holder = ctk.CTkFrame(bg_row, fg_color="transparent")
        self._bg_radio_holder.pack(side="left", fill="x", expand=True,
                                   padx=8, pady=(0, 6))
        ctk.CTkLabel(
            self._bg_radio_holder,
            text="Run detection (or load existing) to choose a background.",
        ).pack(side="left")

        det_frame = ctk.CTkFrame(body)
        det_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 3))
        ctk.CTkLabel(det_frame, text="2. Detected ROIs (raw suite2p output)",
                     font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(6, 0))
        self.det_fig = plt.Figure(figsize=(5, 5), tight_layout=True)
        self.det_ax = self.det_fig.add_subplot(111)
        self.det_ax.set_axis_off()
        self.det_ax.text(0.5, 0.5, "No detection yet", ha="center",
                         va="center", transform=self.det_ax.transAxes)
        self.det_canvas = FigureCanvasTkAgg(self.det_fig, master=det_frame)
        attach_fig_toolbar(self.det_canvas, det_frame)
        self.det_canvas.get_tk_widget().pack(fill="both", expand=True,
                                             padx=6, pady=(0, 6))
        # Click on the "All detected ROIs" panel to open the curation
        # popout. We resolve the clicked ROI via the cached label image
        # (built in ``_redraw_with_bg``) so the popout works on whichever
        # background image the user picked.
        self.det_canvas.mpl_connect("button_press_event",
                                    self._on_det_click)
        self._det_label_img: Optional[np.ndarray] = None
        self._curation_popout: Optional[CurationPopout] = None

        fil_frame = ctk.CTkFrame(body)
        fil_frame.grid(row=1, column=1, sticky="nsew", padx=(3, 0))
        fil_hdr = ctk.CTkFrame(fil_frame, fg_color="transparent")
        fil_hdr.pack(fill="x", padx=8, pady=(6, 0))
        ctk.CTkLabel(fil_hdr, text="3. After cell-filter prediction mask",
                     font=ctk.CTkFont(weight="bold")).pack(side="left",
                                                           anchor="w")
        # ON: paint the kept-ROI overlay; OFF: clean background only.
        # Re-renders from the panel cache (no recompute) and is honoured
        # by the figure export.
        ctk.CTkSwitch(fil_hdr, text="ROI overlay",
                      variable=self._roi_overlay_var,
                      command=self._redraw_with_bg).pack(side="right")
        self.fil_fig = plt.Figure(figsize=(5, 5), tight_layout=True)
        self.fil_ax = self.fil_fig.add_subplot(111)
        self.fil_ax.set_axis_off()
        self.fil_ax.text(0.5, 0.5, "No filter applied yet", ha="center",
                         va="center", transform=self.fil_ax.transAxes)
        self.fil_canvas = FigureCanvasTkAgg(self.fil_fig, master=fil_frame)
        attach_fig_toolbar(self.fil_canvas, fil_frame)
        self.fil_canvas.get_tk_widget().pack(fill="both", expand=True,
                                             padx=6, pady=(0, 6))
        # Click on the kept-ROI panel to curate the same way Panel 2 does.
        # The cached label image only contains kept ROIs, so background
        # pixels and rejected ROIs are silent no-ops.
        self.fil_canvas.mpl_connect("button_press_event",
                                    self._on_fil_click)
        self._fil_label_img: Optional[np.ndarray] = None

    # -- Handlers -----------------------------------------------------------

    def _browse_ckpt(self) -> None:
        path = filedialog.askopenfilename(
            title="Select cell-filter checkpoint",
            filetypes=[("PyTorch", "*.pt"), ("All files", "*.*")])
        if path:
            self.ckpt_var.set(path)

    def _on_preprocess_result(self, result: PreprocessResult) -> None:
        self.run_btn.configure(state="normal")
        existing = candidate_plane0(result)
        if existing is not None and plane0_has_outputs(existing):
            self.load_btn.configure(state="normal")
            self.status_var.set(
                f"Ready: {result.shifted_tiff.name}  "
                f"(existing detection at {existing} - 'Load existing' "
                f"will render without re-running)")
        else:
            self.load_btn.configure(state="disabled")
            self.status_var.set(
                f"Ready: {result.shifted_tiff.name}  "
                f"({result.shape_yx[0]}x{result.shape_yx[1]})")

    def _on_load_existing(self) -> None:
        result = self.state.result
        if result is None:
            messagebox.showerror(
                "Missing input",
                "Run preprocessing first so I know which folder to load.")
            return
        plane0 = candidate_plane0(result)
        if not plane0_has_outputs(plane0):
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

    def _on_edit_suite2p_settings(self) -> None:
        """Open the popout that edits the underlying suite2p 1.0 settings.

        Unlike the Advanced... dialog (which exposes the curated
        ``PARAM_SPEC`` subset Calliope cares about), this popout shows
        every leaf in the nested suite2p settings tree. Edits are
        merged on top of :func:`calliope_settings.build_base_settings`
        and persist for the session via ``self._settings_override``.
        """
        from ...core import calliope_settings as _cs
        base = _cs.build_base_settings()
        specs, flat = _cs.flatten_settings_to_specs(
            base, current=self._settings_override or None,
        )
        if not open_advanced(
                self, "Edit suite2p settings (deep merge on top of defaults)",
                specs, flat):
            return
        new_overrides = _cs.nest_flat_values(flat, base)
        if new_overrides == self._settings_override:
            self._append_log(
                "[GUI] suite2p settings: no changes from current overrides.")
            return
        self._settings_override = new_overrides
        if not new_overrides:
            self._append_log(
                "[GUI] suite2p settings reset to in-source defaults.")
        else:
            self._append_log(
                f"[GUI] suite2p settings updated "
                f"({count_leaves(new_overrides)} overrides). "
                "Re-run detection to apply.")

    def _on_run(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            messagebox.showinfo("Busy", "Detection is already running.")
            return
        if self._archive_job is not None and self._archive_job.running():
            messagebox.showinfo(
                "Busy", "Still archiving the previous run's raw data. "
                "Wait for it to finish before starting another detection.")
            return
        result = self.state.result
        if result is None:
            messagebox.showerror("Missing input", "Run preprocessing first.")
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

        # Resolve the GCaMP picker into a tau scalar. Picker rows are
        # ``(label, tau)`` pairs; we look up the user-chosen label.
        gcamp_label = self.gcamp_var.get()
        tau_override, matched_label, match_kind = self._resolve_gcamp_tau(
            gcamp_label)
        # A non-blank custom tau wins over the dropdown variant. Validate
        # it as a positive float; bail loudly on garbage rather than
        # silently falling back to the variant tau.
        custom_tau_raw = self.custom_tau_var.get().strip()
        if custom_tau_raw:
            try:
                custom_tau = float(custom_tau_raw)
                if custom_tau <= 0:
                    raise ValueError("tau must be positive")
            except ValueError as e:
                messagebox.showerror(
                    "Invalid custom τ",
                    f"Custom τ must be a positive number of seconds "
                    f"(got {custom_tau_raw!r}): {e}. Clear the field to "
                    f"use the GCaMP dropdown instead.")
                return
            self._append_log(
                f"[GUI] Custom τ override: {custom_tau} s "
                f"(dropdown variant {gcamp_label!r} ignored)")
            tau_override = custom_tau
        elif match_kind == "exact":
            self._append_log(
                f"[GUI] GCaMP variant: {gcamp_label} -> tau={tau_override}")
        elif match_kind == "variant":
            # The label string is from a previous schema (e.g. an old batch
            # CSV with ``"GCaMP8m  (tau = 0.137 s)"`` from before the
            # 2026-05-26 tau update). Snap to the current canonical label,
            # log loudly so the user knows the saved selection was
            # migrated, and update the StringVar so the GUI reflects the
            # new label after this run.
            self._append_log(
                f"[GUI] GCaMP variant: {gcamp_label!r} not in current "
                f"dropdown; matched variant code -> using "
                f"{matched_label!r} (tau={tau_override}). Update your "
                f"saved batch CSV / preferences to use the new label.")
            try:
                self.gcamp_var.set(matched_label)
            except Exception:
                pass
        else:
            self._append_log(
                f"[GUI] GCaMP variant: {gcamp_label!r} unrecognised; "
                f"falling back to {self.GCAMP_OPTIONS[0][0]!r} "
                f"(tau={tau_override}). Pick a variant from the dropdown.")

        tiff_folder = result.shifted_tiff.parent
        save_folder = result.out_dir / "detection"
        rec_id = result.out_dir.name

        self.run_btn.configure(state="disabled")
        self.progress.start()
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
                    ckpt_path=(ckpt_path if ckpt_present else None),
                    rec_id=rec_id,
                    baseline_mode=baseline_mode,
                    baseline_min=baseline_min,
                    tau_override=tau_override,
                    raw_paths=getattr(result, "raw_paths", None),
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
        ckpt_path: Optional[str],
        rec_id: str,
        baseline_mode: str,
        baseline_min: float,
        tau_override: float = 1.0,
        raw_paths=None,
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

        # Tier 2 #13 Cellpose-SAM second pass on Vcorr. Off by default
        # because (a) it requires cellpose >= 4.0 which not every box
        # has and (b) the SAM checkpoint is ~1 GB. When the user checks
        # the box, build a config that mirrors the primary cellpose
        # settings but with model_type='cpsam' + channel_input='Vcorr'.
        # The runner persists/streams Vcorr next to the shared
        # registration so run_cellpose_pass can pick it up.
        enable_sam = bool(params.get("enable_sam_vcorr_pass", False))
        sam_cfg = None
        if enable_sam:
            sam_cfg = {
                "cellpose_model_type": "cpsam",
                "cellpose_channel_input": "Vcorr",
                "cellpose_diameter": params.get("cellpose_diameter", 0),
                "cellpose_flow_threshold": float(
                    params.get("sam_flow_threshold", 0.8)),
                "cellpose_cellprob_threshold": float(
                    params.get("sam_cellprob_threshold", -1.0)),
            }

        with contextlib.redirect_stdout(writer), \
                contextlib.redirect_stderr(writer):
            print("[GUI] running sparse_plus_cellpose...")
            from ...core import sparse_plus_cellpose as spc
            final_plane0 = spc.run(
                tiff_folder=str(tiff_folder),
                save_folder=str(save_folder),
                # path_to_ops omitted: in-source defaults from
                # calliope.core.calliope_settings are used. The popout's
                # settings_override layers on top.
                sparsery_ops=sparsery_ops,
                cellpose_cfg=cellpose_cfg,
                hard_cap=hard_cap,
                max_overlap=max_overlap,
                tau_override=tau_override,
                # Per-session overrides from the "Edit suite2p settings..."
                # popout. Empty dict means no overrides.
                settings_override=(self._settings_override
                                   or None),
                enable_sam_vcorr_pass=enable_sam,
                sam_vcorr_cfg=sam_cfg,
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

            # Defer prune + raw-TIFF compression to a child process,
            # launched from the main thread in ``_on_done`` (the zstd
            # level-19 archive is slow and would freeze the GUI here).
            # Stash the context for that launcher.
            self._pending_archive = {
                "save_folder": str(save_folder),
                "params": params,
                "raw_paths": raw_paths,
                "plane0": str(final_plane0),
            }

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
        fov_um = params.get("fov_um_reference", FOV_UM_REFERENCE) \
            or FOV_UM_REFERENCE
        if zoom is None and um_px is None:
            return
        view = utils.load_plane_view(plane0)
        if not view:
            print(f"[GUI] pix_to_um: no plane outputs at {plane0}; "
                  "skipping calibration stamp")
            return
        try:
            resolved = resolve_pix_to_um(
                view, zoom=zoom, um_per_pixel=um_px, fov_um=fov_um,
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
            from calliope.core import analyze_output_gpu as gpu
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

        mask = load_keep_mask(plane0, N_total)
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
        from ...core import utils
        chunk = max(1, min(T, utils.DFF_TIME_CHUNK))
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
        from ...core import utils
        chunk = max(1, min(T, utils.DFF_TIME_CHUNK))
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
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model = CellFilter().to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        predict_recording(rec_id, model, device, plane0=plane0)

    # -- Log queue + completion --------------------------------------------

    def _on_plane0_payload(self, payload) -> None:
        # Worker thread ships the final ``plane0`` path through the
        # queue; capture it for ``_on_done`` to consume.
        self._final_plane0 = Path(payload)

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _on_done(self) -> None:
        plane0 = self._final_plane0
        if plane0 is None:
            self.progress.stop()
            self.run_btn.configure(state="normal")
            self.status_var.set("Done (no plane0 returned).")
            return
        # Draw the panels now (fast) so results appear immediately, then
        # hand the slow prune + raw-compression archive to a child
        # process so it can't freeze the GUI. ``set_plane0`` (the batch
        # advance signal) is published only after the archive finishes
        # in ``_on_archive_done`` -- same ordering as when archiving ran
        # inline at the end of the worker.
        try:
            self._draw_panels(plane0)
        except Exception as e:
            self.status_var.set(f"Render failed: {e}")
            self._append_log(f"[GUI] render error: {e}\n"
                             f"{traceback.format_exc()}")

        ctx = self._pending_archive
        if ctx is None:
            # No archive context (shouldn't happen from a detection run);
            # finish immediately so the run never hangs.
            self._finish_detection(plane0)
            return

        self.status_var.set("Detection complete -- archiving raw data "
                            "(compressing) in the background...")
        self._append_log("[GUI] archiving raw data in a background "
                         "process (GUI stays responsive)...")
        from ...core.offload import run_offloaded
        from ...core import detection_run as _dr
        self._archive_job = run_offloaded(
            self, self._log_queue, _dr.archive_offload,
            args=(ctx["save_folder"], ctx["params"]),
            kwargs=dict(raw_paths=ctx["raw_paths"], plane0_str=ctx["plane0"]),
            done_kind="archive_done", error_kind="archive_error",
            progress_kind="log", poll_ms=self.POLL_MS)

    def _finish_detection(self, plane0: Path) -> None:
        """Stop the spinner, re-enable Run, and publish ``plane0`` (the
        batch-advance signal). Called once the archive completes (or is
        skipped / fails) so the row finishes only after archiving."""
        self._pending_archive = None
        self._archive_job = None
        self.progress.stop()
        self.run_btn.configure(state="normal")
        self.status_var.set(f"Done -> {plane0}")
        try:
            self.state.set_plane0(plane0)
        except Exception as e:
            self._append_log(f"[GUI] plane0 publish error: {e}")

    def _on_archive_done(self, _payload) -> None:
        self._append_log("[GUI] archive complete.")
        plane0 = self._final_plane0
        if plane0 is not None:
            self._finish_detection(plane0)

    def _on_archive_error(self, payload) -> None:
        # Archiving is best-effort -- a compression failure must NOT block
        # the pipeline (matches the old inline behaviour, which swallowed
        # archive errors). Log it and still publish plane0 so batch
        # advances and downstream stages run.
        first = str(payload).split("\n", 1)[0]
        self._append_log(f"[archive] step failed (continuing): {first}")
        plane0 = self._final_plane0
        if plane0 is not None:
            self.status_var.set(f"Done (archive failed) -> {plane0}")
            self._pending_archive = None
            self._archive_job = None
            try:
                self.state.set_plane0(plane0)
            except Exception as e:
                self._append_log(f"[GUI] plane0 publish error: {e}")
            self.progress.stop()
            self.run_btn.configure(state="normal")

    def _on_error(self, msg: str) -> None:
        self.progress.stop()
        self.run_btn.configure(state="normal")
        self.status_var.set("Error.")
        self._append_log(f"ERROR: {msg}")
        if report_stage_error(self.state, msg):
            return
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
        # Curation popout reads the full set of background images
        # plus yrange / xrange for the max-projection padder; cache
        # whichever fields the recording happens to carry so the
        # popout can paint one panel per available background and
        # the rest of the view dict can be released.
        _ops_view_keys = (
            tuple(k for k, _ in self.KNOWN_BG_IMAGES)
            + ("maxImg", "yrange", "xrange")
        )
        ops_view = {
            k: view.get(k) for k in _ops_view_keys
            if view.get(k) is not None
        }
        del view  # release every other field

        stat = np.load(plane0 / "stat.npy", allow_pickle=True)
        n_total = len(stat)

        keep = load_keep_mask(plane0, n_total)
        prob_path = plane0 / "predicted_cell_prob.npy"
        probs = (np.load(prob_path).astype(np.float32)
                 if prob_path.exists() else None)

        self._panel_cache = dict(
            plane0=plane0, bg_images=bg_images, stat=stat,
            n_total=n_total, keep=keep, probs=probs, ops_view=ops_view,
        )

        from ...core.utils import infer_recording_id
        try:
            rec_id = infer_recording_id(plane0)
        except Exception:
            rec_id = plane0.parent.name
        self.recording_var.set(f"Recording: {rec_id}    plane0: {plane0}")

        self._populate_bg_radios(bg_images)
        self._redraw_with_bg()

        self.summary_btn.configure(state="normal")
        self.export_btn.configure(state="normal")
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
            ctk.CTkLabel(self._bg_radio_holder,
                         text="(no 2D images in ops.npy)").pack(side="left")
            return

        keys = {k for k, _ in available}
        if self._bg_var.get() not in keys:
            self._bg_var.set(self.DEFAULT_BG_KEY
                             if self.DEFAULT_BG_KEY in keys
                             else available[0][0])

        for key, label in available:
            ctk.CTkRadioButton(
                self._bg_radio_holder, text=label, value=key,
                variable=self._bg_var,
                command=self._on_bg_changed,
            ).pack(side="left", padx=(0, 10))

    def _on_bg_changed(self) -> None:
        if self._panel_cache is not None:
            self._redraw_with_bg()

    def _on_det_click(self, event) -> None:
        """Resolve the clicked ROI on the "All detected ROIs" panel
        and open / refocus the curation popout.

        Skips left-clicks while the matplotlib navigation toolbar is in
        a pan/zoom mode (so dragging a zoom rectangle doesn't open the
        popout) and ignores any click outside ``det_ax``.
        """
        self._handle_panel_click(
            event, self.det_ax, self.det_canvas, self._det_label_img,
            empty_status="No ROI at pixel ({x}, {y}); click on a "
                         "coloured ROI patch.",
        )

    def _on_fil_click(self, event) -> None:
        """Same as ``_on_det_click`` but for Panel 3 (kept ROIs only).

        The cached label image zeroes out rejected ROIs, so clicks on
        non-kept pixels fall through to the background-hint status
        message without opening the popout. The popout itself shows
        any ROI the user wants (kept or not) -- this just makes
        Panel 3 a more convenient entry point when the kept-mask is
        the view the user is already inspecting.
        """
        self._handle_panel_click(
            event, self.fil_ax, self.fil_canvas, self._fil_label_img,
            empty_status="No kept ROI at pixel ({x}, {y}); the "
                         "cell-filter rejected this region or the "
                         "click missed every kept ROI.",
        )

    def _handle_panel_click(self, event, ax, canvas, label_img,
                            *, empty_status: str) -> None:
        if event.inaxes is not ax:
            return
        if event.button != 1:
            return
        toolbar = getattr(canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return
        if label_img is None or self._panel_cache is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        Ly, Lx = label_img.shape
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        if not (0 <= x < Lx and 0 <= y < Ly):
            return
        lbl = int(label_img[y, x])
        if lbl <= 0:
            self.status_var.set(empty_status.format(x=x, y=y))
            return
        roi_idx = lbl - 1
        c = self._panel_cache
        plane0 = c["plane0"]
        stat = c["stat"]
        ops_view = c.get("ops_view", {})
        # Single-instance popout per Tab 3: re-use the existing window
        # if it's still alive and pointing at the same plane0; build a
        # fresh one otherwise.
        popout = self._curation_popout
        if (popout is None or not popout.winfo_exists()
                or getattr(popout, "_plane0", None) != Path(plane0)):
            try:
                popout = CurationPopout(
                    self.winfo_toplevel(), plane0, stat, ops_view,
                    on_iscell_changed=self._on_iscell_flipped,
                )
            except Exception as e:
                messagebox.showerror("Curation popout failed", str(e))
                return
            self._curation_popout = popout
        popout.show_roi(roi_idx)

    def _repoint_after_copy(self, scratch: Path, final: Path) -> None:
        """BatchTab calls this after a row's scratch->HDD copy so we
        rewrite the dict-stored plane0 path inside ``_panel_cache``
        from ``scratch`` to ``final``.

        Why this hook exists separately from BatchTab's attribute-
        walking repoint: ``_panel_cache`` is a dict whose ``"plane0"``
        entry is the Path the curation popout reads ``iscell.npy``
        and ``stat.npy`` from. BatchTab's loop rewrites named tab
        attributes (``_plane0``, ``_final_plane0``, ...) but can't
        reach inside dict values, so a stale scratch path used to
        survive the repoint. After the rmtree improvements made the
        scratch deletion actually succeed (instead of silently
        failing under ``ignore_errors=True``), opening the popout
        post-finalize produced ``iscell=?`` because the cached
        scratch path no longer existed.

        Also rewrites ``_plane0`` on any open ``CurationPopout`` so
        subsequent flips write ``iscell.npy`` at the HDD path. The
        popout's already-loaded ``_iscell`` / ``_probs`` arrays
        stayed in RAM and remain valid.
        """
        def _rewrite(cur):
            if cur is None:
                return None
            try:
                p = Path(cur)
                if p == scratch:
                    return final
                rel = p.relative_to(scratch)
                return final / rel
            except (TypeError, ValueError, OSError):
                return None

        cache = self._panel_cache
        if isinstance(cache, dict):
            new_p = _rewrite(cache.get("plane0"))
            if new_p is not None:
                cache["plane0"] = new_p

        popout = self._curation_popout
        if popout is not None:
            try:
                if popout.winfo_exists():
                    new_p = _rewrite(getattr(popout, "_plane0", None))
                    if new_p is not None:
                        popout._plane0 = new_p
                    # ``popout._dff`` is an ``np.memmap`` opened
                    # against scratch's ``r0p7_dff.memmap.float32``
                    # for the trace strip. The path rewrite above
                    # doesn't release the handle -- drop it so the
                    # bulk rmtree can succeed. ``show_roi`` doesn't
                    # reopen it, so this disables the trace strip on
                    # the open popout; that's the right trade-off
                    # for freeing scratch (user reopens the popout
                    # if they need the trace).
                    dff = getattr(popout, "_dff", None)
                    fname = getattr(dff, "filename", None)
                    if fname is not None:
                        try:
                            Path(fname).relative_to(scratch)
                        except (TypeError, ValueError, OSError):
                            pass
                        else:
                            popout._dff = None
                            popout._T = 0
            except Exception:
                pass

    def _on_iscell_flipped(self, roi_idx: int, new_value: int) -> None:
        """Listener fired by the popout after a successful classification
        flip. Reload the keep mask from the freshly-written iscell.npy
        and repaint both panels so Tab 3 stays in sync with the new
        labels.

        The sentinel ``(-1, -1)`` means "Promote to filter mask" just
        rewrote ``predicted_cell_mask.npy``. In that case also regenerate
        the filtered dF/F memmap so downstream tabs (lowpass ``n_kept``,
        Tab 5/6/7, cross-correlation) reflect the curated cell set --
        see :meth:`_regenerate_after_promote`.
        """
        c = self._panel_cache
        if c is None:
            return
        try:
            c["keep"] = load_keep_mask(c["plane0"], c["n_total"])
        except Exception:
            return
        self._redraw_with_bg()
        if roi_idx == -1 and new_value == -1:
            self._regenerate_after_promote(Path(c["plane0"]))

    def _regenerate_after_promote(self, plane0: Path) -> None:
        """Rewrite the filtered dF/F memmap after a "Promote to filter
        mask" so downstream tabs see the curated cell set.

        Promote only updates ``predicted_cell_mask.npy``. The memmap that
        downstream tabs actually read (``r0p7_filtered_dff.memmap.float32``)
        is column-sliced to the keep mask that was live when it was
        written and is hard-anchored to ``r0p7_cell_mask_bool.npy`` by
        :func:`calliope.core.utils.resolve_filtered_mask` (the WinError 8
        fix). So on a reloaded completed run the promotion is invisible
        until the memmap is rewritten.

        :meth:`_run_filtered_dff` re-slices from the full-ROI source
        ``r0p7_dff.memmap.float32`` using the now-current keep mask, so
        both removing and re-adding cells take effect. Republishing
        ``plane0`` then tells subscribers (lowpass etc.) to reload.

        **Windows file-lock handling.** On a reloaded run, downstream
        tabs hold open ``np.memmap`` handles on the filtered file --
        Tab 7's ``_dff_cache`` and Tab 6's ``self._dff`` (both opened by
        the reload flow's ``_on_reload_inputs`` / ``_on_reload_clusters``).
        On Windows an open mapping locks the file, so the ``mode="w+"``
        truncate inside ``_run_filtered_dff`` fails with
        ``PermissionError``. ``AppState.release_memmaps()`` tells those
        tabs to drop their cached handles first (they reopen lazily / on
        the next run); a ``gc.collect()`` then frees any transient
        mappings before we truncate.

        No-op when the source/filtered memmaps don't exist yet -- there
        is nothing downstream to refresh until a detection run has
        produced them. Note: already-computed downstream *results* (Tab 5
        events, Tab 6/7 clusters, cross-correlation) are NOT recomputed;
        they stay stale until those stages are re-run.
        """
        import gc

        src = plane0 / "r0p7_dff.memmap.float32"
        filt = plane0 / "r0p7_filtered_dff.memmap.float32"
        if not (src.exists() and filt.exists()):
            return
        # Drop downstream tabs' cached handles on the filtered memmap so
        # Windows releases the file lock before we truncate it below.
        try:
            self.state.release_memmaps()
        except Exception:
            pass
        gc.collect()
        try:
            self._run_filtered_dff(plane0)
        except Exception as e:
            messagebox.showwarning(
                "Filtered dF/F not updated",
                f"The promotion was saved, but the filtered dF/F memmap "
                f"could not be regenerated:\n{e}\n\nClose any open "
                f"cross-correlation / cluster popouts and re-run Tab 3 "
                f"'filtered dF/F' to apply the new cell set downstream.")
            return
        # Re-fire the plane0 channel so memmap-reading tabs (lowpass
        # n_kept, etc.) reload from the rewritten memmap. set_plane0
        # re-publishes unconditionally, so the same path still refreshes.
        try:
            self.state.set_plane0(plane0)
        except Exception:
            pass

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

        # Pixel calibration for the µm scale bar (None when the recording
        # has no zoom / µm-per-pixel set -> panels fall back to bare pixels).
        pix_to_um = None
        if self._final_plane0 is not None:
            try:
                from ...core import utils as _utils
                pix_to_um = _utils.load_pix_to_um(self._final_plane0)
            except Exception:
                pix_to_um = None

        from ...core import utils
        vmax = float(np.quantile(bg, utils.DISPLAY_CLIP_HIGH_PCT / 100.0))
        vmin = float(np.quantile(bg, utils.DISPLAY_CLIP_LOW_PCT / 100.0))

        label_all = build_label_image(stat, Ly, Lx)
        # Cache the label image so the click handler can map a
        # (x_data, y_data) cursor position to the Suite2p ROI index
        # that was clicked. The label image stores ``roi_idx + 1`` per
        # pixel, ``0`` for background.
        self._det_label_img = label_all
        # Panel-3 click cache: same label encoding (roi_idx + 1) but with
        # rejected ROIs zeroed out so a click on a non-kept pixel is a
        # silent no-op. Vectorised LUT over the label image -- O(H*W),
        # no per-ROI Python loop.
        lookup = np.concatenate([[False], keep.astype(bool)])
        self._fil_label_img = np.where(
            lookup[label_all], label_all, 0).astype(label_all.dtype)
        render_panel(
            self.det_ax, self.det_canvas, bg, label_all, vmin, vmax,
            f"All detected ROIs (n = {n_total})  [bg: {bg_label}]"
            f"  -- click an ROI to inspect / curate",
            pix_to_um=pix_to_um)

        kept_n = int(keep.sum())
        if not self._roi_overlay_var.get():
            # Overlay off: show the clean background image only.
            self.fil_ax = render_background_panel(
                self.fil_ax, self.fil_canvas, bg, vmin, vmax,
                f"After filter (n = {kept_n} / {n_total})  "
                f"[bg: {bg_label}, no ROI overlay]",
                pix_to_um=pix_to_um)
        elif probs is not None and probs.shape[0] == n_total:
            score_img = build_score_image(stat, keep, probs, Ly, Lx)
            title = (f"After filter (n = {kept_n} / {n_total})  "
                     f"[bg: {bg_label}, coloured by predicted_cell_prob]")
            self.fil_ax = render_score_panel(
                self.fil_ax, self.fil_canvas, bg, score_img,
                vmin, vmax, title, pix_to_um=pix_to_um)
        else:
            kept_stat = [s for i, s in enumerate(stat) if keep[i]]
            label_kept = build_label_image(kept_stat, Ly, Lx)
            keep_src = "iscell.npy (suite2p classifier)"
            render_panel(
                self.fil_ax, self.fil_canvas, bg, label_kept,
                vmin, vmax,
                f"After filter (n = {kept_n} / {n_total})  "
                f"[bg: {bg_label}, {keep_src}]",
                pix_to_um=pix_to_um)

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
            keep = load_keep_mask(plane0, n_total)
            prob_path = plane0 / "predicted_cell_prob.npy"
            probs = (np.load(prob_path).astype(np.float32)
                     if prob_path.exists() else None)
            path = self._write_summary(plane0, stat, keep, probs)
            self.status_var.set(f"Summary -> {path}")
        except Exception as e:
            messagebox.showerror("Summary failed", str(e))

    def _has_pending_curation(self, plane0: Path) -> bool:
        """True when ROI labels have moved but the keep-mask hasn't
        been refreshed -- exporting now would render figures from a
        stale keep-mask.

        Two signals, OR'd:
        1. **In-memory:** an open CurationPopout has flipped ROIs in
           this session that haven't been promoted yet (popout still
           tracks them in ``self._flipped``).
        2. **On-disk:** ``iscell.npy`` mtime > ``predicted_cell_mask.npy``
           mtime. This catches the case where the user closed the popout
           before exporting -- the popout's ``_flipped`` set is gone but
           the iscell flips persisted to disk and the predicted-mask
           never got refreshed.

        Both files always have a fresh mtime ordering after a clean
        ``run_detection`` (predicted_cell_mask is written *after*
        suite2p's classifier writes iscell), so the disk check only
        fires when the user's flips happened later than the last
        promote / detection.
        """
        popout = self._curation_popout
        if (popout is not None and popout.winfo_exists()
                and popout.has_pending_promotion()):
            return True
        iscell = plane0 / "iscell.npy"
        mask = plane0 / "predicted_cell_mask.npy"
        if iscell.is_file() and mask.is_file():
            try:
                return iscell.stat().st_mtime > mask.stat().st_mtime
            except OSError:
                return False
        return False

    def _on_export_figures(self, quiet: bool = False) -> None:
        """Re-render the Tab 3 detection panels as PNG+SVG and drop a
        ``manifest.json`` next to them. Block the export when curation
        is uncommitted (either an open popout has un-promoted flips,
        or ``iscell.npy`` is newer than ``predicted_cell_mask.npy``
        on disk -- catches the case where the user closed the popout
        before clicking Export but the keep-mask never got refreshed).

        For the full per-stage figure set (lowpass / events / clusters
        / xcorr / spatial), run the recording through the batch tab --
        ``core/batch_pipeline.py`` writes figures for every stage as
        part of the normal pipeline.

        When ``quiet=True`` (called by the Tab 0 batch toggle), all
        messageboxes are suppressed and failures fall back to the
        status bar instead.
        """
        plane0 = self._final_plane0
        if plane0 is None:
            if not quiet:
                messagebox.showinfo(
                    "No data",
                    "Run detection or 'Load existing panels' first.")
            return

        if self._has_pending_curation(plane0):
            if not quiet:
                messagebox.showwarning(
                    "Uncommitted curation",
                    "iscell.npy has flips that haven't been promoted to "
                    "predicted_cell_mask.npy.\n\n"
                    "Open the curation popout (click an ROI on Panel 2 "
                    "or Panel 3) and click 'Promote to filter mask' "
                    "before exporting -- otherwise the figures will "
                    "reflect the stale keep-mask, not what you're "
                    "looking at.")
            else:
                self.status_var.set(
                    "Export skipped: uncommitted curation flips.")
            return

        # plane0 = <save_folder>/detection/final/suite2p/plane0; figures
        # land under calliope_figures/<stage>/ to match the batch layout.
        from ...core.utils import save_folder_for_plane0
        save_folder = save_folder_for_plane0(plane0)
        figures_root = save_folder / "calliope_figures"
        detection_dir = figures_root / "detection"
        try:
            from ...core.detection_run import _render_detection_panels
            from ...core.export_manifest import write_export_manifest
            from ...core.utils import safe_recording_id
            rec_id = safe_recording_id(plane0)
            written = _render_detection_panels(
                plane0, detection_dir,
                kept_overlay=bool(self._roi_overlay_var.get()))
            ckpt_path = self.ckpt_var.get().strip() or None
            manifest_path = write_export_manifest(
                figures_root,
                rec_id=rec_id, params=dict(self._params),
                plane0=plane0, ckpt_path=ckpt_path,
            )
            n_figs = len(written) * 2  # PNG + SVG per panel
            self.status_var.set(
                f"Exported {n_figs} files -> {figures_root}")
            if not quiet:
                messagebox.showinfo(
                    "Export complete",
                    f"Wrote {n_figs} figure files (PNG + SVG) to:\n"
                    f"  {detection_dir}\n\n"
                    f"Manifest:\n  {manifest_path}\n\n"
                    f"For the full per-stage figure set (lowpass, "
                    f"events, clustering, xcorr, spatial), drive the "
                    f"recording through Tab 0 (Batch).")
        except Exception as e:
            if not quiet:
                messagebox.showerror("Export failed", str(e))
            else:
                self.status_var.set(f"Export failed: {e}")

