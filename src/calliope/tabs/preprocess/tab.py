"""calliope.tabs.preprocess.tab - Tab 1: Input & Preprocess.

What the user sees
------------------
A simple form with three rows: pick a working directory, pick one or
more TIFFs inside it (multi-selection groups them into a single
recording), and click Run. While the worker thread runs, status
messages stream into the log panel below. When Run finishes, a
``PreprocessResult`` is published on ``AppState`` so Tab 2 can
display the QC GIF and Tab 3 can pick up the shifted TIFF.

Why a worker thread
-------------------
Preprocessing a 10-minute movie can take 30+ seconds. If we ran it
on the main (Tk) thread the GUI would freeze. The pattern used here
shows up in every "long-running" tab in CalLIOPE:

    1. ``_on_run`` (main thread) snapshots the form values into
       plain Python types.
    2. It launches a ``threading.Thread`` whose target is a worker
       function defined inline. That function calls into
       ``preprocessing.preprocess_tiff(...)`` and pushes log lines
       and a final result onto a thread-safe ``queue.Queue``.
    3. ``_drain_log_queue`` (main thread, scheduled with
       ``self.after(...)``) wakes up every POLL_MS, drains the
       queue, and writes lines into the log panel. When it sees the
       "done" sentinel it publishes the result on ``AppState``.

For an R user: same idea as launching a ``future`` in the
``promises`` package and polling it from Shiny's reactive event
loop.

Module layout
-------------
``PreprocessTab`` (subclass of ``ttk.Frame``)
    The actual tab widget. ``ttk.Frame`` is just "a container for
    other widgets" -- subclassing it gives us a Tk widget that *is*
    a frame and adds our own state + methods on top.

``PARAM_SPEC`` (class attribute)
    A list of dicts describing every Advanced... knob. Used to
    auto-build the dialog (see ``gui_common.AdvancedDialog``) and to
    seed ``self._params`` with default values. Editing the list in
    one place adds a knob to the dialog AND wires it through to the
    worker.

Lifecycle
---------
    __init__        -- build widgets, subscribe to AppState, prime defaults.
    _build_ui       -- lay out the Tk widgets.
    _on_browse      -- folder picker -> update the working directory.
    _on_run         -- snapshot params, launch the worker thread.
    _on_advanced    -- pop up the Advanced... dialog.
    _drain_log_queue-- main-thread polling loop draining the log queue.
"""

# Type-hint forward-reference shim (see pipeline_gui.py).
from __future__ import annotations

# Threading + queue: the long-running preprocess step runs on a
# worker thread; we communicate with the GUI thread via a thread-
# safe FIFO queue.
import queue
import threading
import tkinter as tk
from pathlib import Path
# ``filedialog`` for the Browse... folder picker;
# ``messagebox`` for error pop-ups; ``ttk`` for themed widgets.
from tkinter import filedialog, messagebox, ttk
from typing import Optional

# Re-export shim (see logic.py for what this is). We rename it to
# ``preprocessing`` so the call sites read like normal code:
# ``preprocessing.preprocess_tiff(...)``.
from . import logic as preprocessing
from .logic import PreprocessResult

# Shared helpers from the top-level GUI common module: the state bus,
# the advanced-dialog opener, and the "build a defaults dict from
# PARAM_SPEC" helper.
from ...gui_common import AppState, open_advanced, spec_defaults


class PreprocessTab(ttk.Frame):
    """Tab 1: pick TIFFs and run the shift + QC preprocess.

    Inherits from ``ttk.Frame`` so an instance is usable as a Tk
    widget that holds children and can be packed/gridded into the
    parent notebook.
    """

    # ``POLL_MS`` is how often (ms) the main-thread log-drain runs.
    # 80ms ≈ 12 Hz, fast enough to look smooth, slow enough not to
    # eat CPU.
    POLL_MS = 80

    PARAM_SPEC: list = [
        # Blob detector
        {"name": "soma_diameter_px", "label": "Soma diameter (px)",
         "type": "float", "default": 12.0, "group": "Blob detection",
         "help": "expected soma size, used for LoG sigma range"},
        {"name": "scale_tol", "label": "Scale tolerance",
         "type": "float", "default": 0.5, "group": "Blob detection",
         "help": "sigma range = r * (1 ± tol) / sqrt(2)"},
        {"name": "min_contrast", "label": "Min contrast (LoG)",
         "type": "float", "default": 0.10, "group": "Blob detection",
         "help": "blob_log threshold after robust normalisation"},
        {"name": "min_area_px", "label": "Min area (px²)",
         "type": "int", "default": 25, "group": "Blob detection"},
        {"name": "max_area_px", "label": "Max area (px²)",
         "type": "int", "default": 400, "group": "Blob detection"},
        # QC gif
        {"name": "downsample_t", "label": "Time downsample factor",
         "type": "int", "default": 4, "group": "QC gif",
         "help": "keep every Nth frame for the preview"},
        {"name": "max_size_px", "label": "Max gif size (px)",
         "type": "int", "default": 512, "group": "QC gif"},
        {"name": "playback_fps", "label": "Playback FPS",
         "type": "int", "default": 15, "group": "QC gif"},
        {"name": "clip_low", "label": "Clip low (percentile)",
         "type": "float", "default": 1.0, "group": "QC gif"},
        {"name": "clip_high", "label": "Clip high (percentile)",
         "type": "float", "default": 99.5, "group": "QC gif"},
    ]

    def __init__(self, master, state: AppState) -> None:
        """Build the tab.

        ``master`` is the parent Tk widget (the scrollable container
        wrapper from ``pipeline_gui._make_scrollable_tab``).
        ``state`` is the shared ``AppState`` event bus -- we'll
        publish the ``PreprocessResult`` onto it when Run finishes.
        """
        # Initialise the parent ``ttk.Frame`` with 10 px of internal
        # padding. ``super().__init__`` is the standard way to call
        # the parent class's constructor.
        super().__init__(master, padding=10)
        # Stash the state bus so the worker callback can publish onto
        # it later.
        self.state = state
        # The log queue: worker pushes ('log', line) tuples and a
        # final ('done', result) sentinel; main thread's
        # ``_drain_log_queue`` reads them.
        self._log_queue: queue.Queue = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        # ``self._params`` starts as the dict of defaults pulled out
        # of ``PARAM_SPEC``. The Advanced... dialog mutates this dict
        # in place when the user clicks OK.
        self._params: dict = spec_defaults(self.PARAM_SPEC)

        # Lay out the widgets.
        self._build_ui()
        # ``after(ms, callback)`` schedules ``callback`` to run on
        # the Tk main thread ``ms`` milliseconds from now. We use
        # this to start the polling loop that drains the log queue
        # without ever blocking the GUI.
        self.after(self.POLL_MS, self._drain_log_queue)

    # -- UI -----------------------------------------------------------------

    def _build_ui(self) -> None:
        dir_frame = ttk.LabelFrame(self, text="1. Working directory (raw TIFFs)",
                                   padding=8)
        dir_frame.pack(fill="x", pady=(0, 8))

        self.dir_var = tk.StringVar()
        ttk.Entry(dir_frame, textvariable=self.dir_var).grid(
            row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(dir_frame, text="Browse...", command=self._browse_dir).grid(
            row=0, column=1)

        # Subfolder search depth (BFS): 0 = only files in working_dir
        self.depth_var = tk.IntVar(value=0)
        ttk.Label(dir_frame, text="Search depth (subfolders):").grid(
            row=1, column=0, sticky="w", pady=(6, 0))
        depth_spin = ttk.Spinbox(
            dir_frame, from_=0, to=99, width=5,
            textvariable=self.depth_var, command=self._refresh_tiffs,
        )
        depth_spin.grid(row=1, column=1, sticky="w", pady=(6, 0))
        depth_spin.bind("<Return>", lambda _e: self._refresh_tiffs())
        depth_spin.bind("<FocusOut>", lambda _e: self._refresh_tiffs())

        dir_frame.columnconfigure(0, weight=1)

        tiff_frame = ttk.LabelFrame(
            self,
            text="2. TIFF files  (select 1+; multi-selection -> one grouped "
                 "recording)",
            padding=8,
        )
        tiff_frame.pack(fill="x", pady=(0, 8))

        list_holder = ttk.Frame(tiff_frame)
        list_holder.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        list_holder.columnconfigure(0, weight=1)
        self.tiff_listbox = tk.Listbox(
            list_holder, selectmode="extended", height=6,
            exportselection=False,
        )
        self.tiff_listbox.grid(row=0, column=0, sticky="ew")
        list_sb = ttk.Scrollbar(list_holder, orient="vertical",
                                command=self.tiff_listbox.yview)
        list_sb.grid(row=0, column=1, sticky="ns")
        self.tiff_listbox.config(yscrollcommand=list_sb.set)
        self.tiff_listbox.bind(
            "<<ListboxSelect>>",
            lambda _e: self._update_existing_status(),
        )

        ttk.Button(tiff_frame, text="Refresh",
                   command=self._refresh_tiffs).grid(row=0, column=1, sticky="n")

        ttk.Label(
            tiff_frame,
            text="Identifier (optional, used as folder name; "
                 "default = '+'-joined stems for groups):",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 2))
        self.identifier_var = tk.StringVar()
        self.identifier_var.trace_add(
            "write", lambda *_a: self._update_existing_status()
        )
        ttk.Entry(tiff_frame, textvariable=self.identifier_var).grid(
            row=2, column=0, columnspan=2, sticky="ew")

        tiff_frame.columnconfigure(0, weight=1)

        out_frame = ttk.LabelFrame(
            self, text="3. Output root  (creates data/<recording>/ underneath)",
            padding=8)
        out_frame.pack(fill="x", pady=(0, 8))

        self.data_root_var = tk.StringVar()
        ttk.Entry(out_frame, textvariable=self.data_root_var).grid(
            row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(out_frame, text="Browse...", command=self._browse_data_root).grid(
            row=0, column=1)
        out_frame.columnconfigure(0, weight=1)

        action = ttk.Frame(self)
        action.pack(fill="x", pady=(4, 8))
        self.run_btn = ttk.Button(action, text="Run preprocessing",
                                  command=self._on_run)
        self.run_btn.pack(side="left")
        self.rerun_btn = ttk.Button(action, text="Force rerun",
                                    command=self._on_force_rerun,
                                    state="disabled")
        self.rerun_btn.pack(side="left", padx=(6, 0))
        ttk.Button(action, text="Advanced...",
                   command=self._on_advanced).pack(side="left", padx=(6, 0))
        self.progress = ttk.Progressbar(action, mode="indeterminate", length=200)
        self.progress.pack(side="left", padx=12)
        self.status_var = tk.StringVar(value="Idle.")
        ttk.Label(action, textvariable=self.status_var).pack(side="left")

        log_frame = ttk.LabelFrame(self, text="Log", padding=4)
        log_frame.pack(fill="both", expand=True)
        self.log = tk.Text(log_frame, height=12, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True, side="left")
        sb = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        sb.pack(fill="y", side="right")
        self.log.config(yscrollcommand=sb.set)

    # -- Handlers -----------------------------------------------------------

    def _browse_dir(self) -> None:
        path = filedialog.askdirectory(title="Select working directory")
        if not path:
            return
        self.dir_var.set(path)
        self.state.working_dir = Path(path)
        if not self.data_root_var.get():
            default = Path(path).parent / "data"
            self.data_root_var.set(str(default))
        self._refresh_tiffs()

    def _browse_data_root(self) -> None:
        path = filedialog.askdirectory(title="Select output (data) root")
        if path:
            self.data_root_var.set(path)

    def _refresh_tiffs(self) -> None:
        wd = self.dir_var.get().strip()
        self.tiff_listbox.delete(0, "end")
        if not wd:
            return
        try:
            depth = int(self.depth_var.get())
        except (tk.TclError, ValueError):
            depth = 0
        depth = max(0, depth)
        tiffs = preprocessing.list_tiffs(wd, max_depth=depth)
        wd_path = Path(wd)
        names: list[str] = []
        for t in tiffs:
            try:
                rel = t.relative_to(wd_path).as_posix()
            except ValueError:
                rel = t.name
            names.append(rel)
        for n in names:
            self.tiff_listbox.insert("end", n)
        if names:
            self.tiff_listbox.selection_set(0)
        else:
            scope = (f"{wd} (depth={depth})" if depth > 0 else wd)
            self._append_log(f"No .tif/.tiff files in {scope}")
        self._update_existing_status()

    @staticmethod
    def _trailing_index(name: str) -> int:
        # Last run of digits in a filename stem; sentinel for files without
        # any numeric suffix.
        import re
        stem = Path(name).stem
        m = re.search(r"(\d+)(?!.*\d)", stem)
        return int(m.group(1)) if m else -1

    def _selected_tiff_names(self) -> list[str]:
        idxs = self.tiff_listbox.curselection()
        names = [self.tiff_listbox.get(i) for i in idxs]
        names.sort(key=self._trailing_index)
        return names

    def _resolved_identifier(self, names: list[str]) -> str:
        explicit = self.identifier_var.get().strip()
        if explicit:
            return explicit
        return "+".join(Path(n).stem for n in names)

    def _existing_out_dir(self) -> Optional[Path]:
        wd = self.dir_var.get().strip()
        names = self._selected_tiff_names()
        data_root = self.data_root_var.get().strip()
        if not (wd and names and data_root):
            return None
        return Path(data_root) / self._resolved_identifier(names)

    def _update_existing_status(self) -> None:
        out_dir = self._existing_out_dir()
        existing = (preprocessing.load_existing_preprocess(out_dir)
                    if out_dir is not None else None)
        if existing is not None:
            self.rerun_btn.config(state="normal")
            self.status_var.set(
                f"Existing outputs at {out_dir} - Run will load them; "
                f"use Force rerun to redo.")
        else:
            self.rerun_btn.config(state="disabled")
            if out_dir is not None:
                n = len(self._selected_tiff_names())
                suffix = f" ({n} TIFFs as one group)" if n > 1 else ""
                self.status_var.set(
                    f"No existing outputs; Run will preprocess{suffix}.")
            else:
                self.status_var.set("Idle.")

    def _on_run(self) -> None:
        self._start_run(force=False)

    def _on_force_rerun(self) -> None:
        out_dir = self._existing_out_dir()
        if out_dir is not None:
            ok = messagebox.askyesno(
                "Force rerun",
                f"This will overwrite existing outputs in:\n{out_dir}\n\n"
                f"Continue?")
            if not ok:
                return
        self._start_run(force=True)

    def _start_run(self, force: bool) -> None:
        if self._worker is not None and self._worker.is_alive():
            messagebox.showinfo("Busy", "Preprocessing is already running.")
            return

        wd = self.dir_var.get().strip()
        names = self._selected_tiff_names()
        data_root = self.data_root_var.get().strip()

        if not wd or not names:
            messagebox.showerror(
                "Missing input",
                "Select a working directory and at least one TIFF.")
            return
        if not data_root:
            messagebox.showerror("Missing output",
                                 "Set an output root directory.")
            return

        srcs = [Path(wd) / n for n in names]
        identifier = self._resolved_identifier(names)
        self.state.selected_tiff = srcs[0]
        self.state.data_root = Path(data_root)

        if not force:
            out_dir = self._existing_out_dir()
            existing = (preprocessing.load_existing_preprocess(out_dir)
                        if out_dir is not None else None)
            if existing is not None:
                self._append_log(
                    f"--- Loading existing outputs from {out_dir} ---")
                self._on_done(existing)
                return

        self.run_btn.config(state="disabled")
        self.rerun_btn.config(state="disabled")
        self.progress.start(12)
        self.status_var.set("Running...")
        if len(srcs) == 1:
            self._append_log(f"--- Preprocessing {srcs[0].name} ---")
        else:
            self._append_log(
                f"--- Preprocessing group '{identifier}' "
                f"({len(srcs)} TIFFs, in trailing-index order):")
            for s in srcs:
                self._append_log(f"      {s.name}")
            self._append_log("---")

        # Snapshot params on the main thread so the worker sees a stable
        # value even if the user opens Advanced again mid-run.
        params = dict(self._params)
        blob_keys = ("scale_tol", "min_contrast", "min_area_px",
                     "max_area_px")
        qc_keys = ("downsample_t", "max_size_px", "playback_fps",
                   "clip_low", "clip_high")
        blob_params = {k: params[k] for k in blob_keys if k in params}
        qc_params = {k: params[k] for k in qc_keys if k in params}
        soma_diameter_px = float(params.get("soma_diameter_px", 12.0))
        explicit_identifier = self.identifier_var.get().strip() or None

        def worker():
            try:
                if len(srcs) == 1:
                    result = preprocessing.preprocess_tiff(
                        src_tiff=srcs[0],
                        data_root=data_root,
                        recording_name=explicit_identifier,
                        soma_diameter_px=soma_diameter_px,
                        progress_cb=lambda m: self._log_queue.put(("log", m)),
                        blob_params=blob_params,
                        qc_params=qc_params,
                    )
                else:
                    result = preprocessing.preprocess_tiff_group(
                        src_tiffs=srcs,
                        data_root=data_root,
                        recording_name=identifier,
                        soma_diameter_px=soma_diameter_px,
                        progress_cb=lambda m: self._log_queue.put(("log", m)),
                        blob_params=blob_params,
                        qc_params=qc_params,
                    )
                self._log_queue.put(("done", result))
            except Exception as e:
                self._log_queue.put(("error", str(e)))

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    def _on_advanced(self) -> None:
        if open_advanced(self, "Preprocess - Advanced parameters",
                         self.PARAM_SPEC, self._params):
            self._append_log(
                "[GUI] Advanced parameters updated. Re-run preprocessing "
                "to apply.")

    # -- Logging / queue drain ---------------------------------------------

    def _drain_log_queue(self) -> None:
        try:
            while True:
                kind, payload = self._log_queue.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "done":
                    self._on_done(payload)
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

    def _on_done(self, result: PreprocessResult) -> None:
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.rerun_btn.config(state="normal")
        n_frames_str = (f"{result.n_frames} frames"
                        if result.n_frames > 0 else "n_frames=?")
        self.status_var.set(
            f"Done - {result.n_blobs} preview blobs, {n_frames_str}.")
        self._append_log(
            f"Outputs: {result.qc_gif.name}, "
            f"{result.shifted_tiff.name}, mean.npy, blobs.npy")
        self.state.set_result(result)

    def _on_error(self, msg: str) -> None:
        self.progress.stop()
        self.run_btn.config(state="normal")
        self._update_existing_status()
        self.status_var.set("Error.")
        self._append_log(f"ERROR: {msg}")
        messagebox.showerror("Preprocessing failed", msg)
