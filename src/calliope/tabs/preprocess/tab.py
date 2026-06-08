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
    2. It offloads ``preprocessing.preprocess_offload(...)`` to a child
       process via ``core.offload.run_offloaded`` (a thread would let
       the TIFF I/O / mean / QC-GIF work hold the GIL and freeze the
       GUI). Progress strings and the final ``PreprocessResult`` are
       re-posted from the main thread onto a ``queue.Queue``.
    3. ``drain_queue`` (main thread, scheduled with ``self.after(...)``)
       wakes up every POLL_MS, drains the queue, and writes lines into
       the log panel. When it sees the "done" payload it publishes the
       result on ``AppState``.

Module layout
-------------
``PreprocessTab`` (subclass of ``ctk.CTkFrame``)
    The actual tab widget. Subclassing the frame gives us a Tk
    widget that *is* a frame and adds our own state + methods on
    top.

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

# ``from __future__ import annotations`` keeps type hints as strings
# at parse time (see pipeline_gui.py for the full rationale).
from __future__ import annotations

# Threading + queue: the long-running preprocess step runs on a
# worker thread; we communicate with the GUI thread via a thread-
# safe FIFO queue.
import queue
import threading
import tkinter as tk
from pathlib import Path

import customtkinter as ctk
# ``filedialog`` for the Browse... folder picker;
# ``messagebox`` for error pop-ups; ``ttk`` only for the Spinbox
# widget CTk doesn't ship.
from tkinter import filedialog, messagebox, ttk
from typing import Optional

# ``logic`` is the per-tab worker module; aliasing it as
# ``preprocessing`` lets the call sites read like
# ``preprocessing.preprocess_tiff(...)``.
from . import logic as preprocessing
from .logic import PreprocessResult

# Shared helpers from the top-level GUI common module: the state bus,
# the advanced-dialog opener, and the "build a defaults dict from
# PARAM_SPEC" helper.
from ...gui_common import (
    AppState, apply_dark_to_tk_widget, attach_resize_handle,
    drain_queue, open_advanced, report_stage_error, spec_defaults,
)


class PreprocessTab(ctk.CTkFrame):
    """Tab 1: pick TIFFs and run the shift + QC preprocess.

    Subclasses ``ctk.CTkFrame`` so the instance is itself a Tk widget
    that the content host can pack/grid directly.
    """

    # Class attribute (lives on the class, shared by every instance).
    # ``POLL_MS`` is how often (ms) the main-thread log-drain runs --
    # 80 ms is fast enough to look smooth, slow enough not to eat CPU.
    POLL_MS = 80

    PARAM_SPEC: list = [
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
        # ``super().__init__`` runs the CTkFrame constructor; CTk frames
        # don't accept a ``padding=`` kwarg, so the children below use
        # padx/pady for inset spacing.
        super().__init__(master, fg_color="transparent")
        # Stash the state bus so the worker callback can publish onto
        # it later.
        self.state = state
        # The log queue: worker pushes ('log', line) tuples and a
        # final ('done', result) sentinel; main thread's
        # ``_drain_log_queue`` reads them.
        self._log_queue: queue.Queue = queue.Queue()
        # Preprocessing runs in a child process (see core.offload) so the
        # TIFF shift / mean / QC-GIF work can't hold the GIL and freeze
        # the GUI. ``None`` until the first run.
        self._job = None
        # ``self._params`` starts as the dict of defaults pulled out
        # of ``PARAM_SPEC``. The Advanced... dialog mutates this dict
        # in place when the user clicks OK.
        self._params: dict = spec_defaults(self.PARAM_SPEC)

        # Lay out the widgets.
        self._build_ui()
        # ``drain_queue`` polls ``self._log_queue`` from the Tk main
        # thread every ``POLL_MS`` ms and dispatches each
        # ``(kind, payload)`` to the matching handler -- the worker
        # thread can ``print`` (via QueueWriter) without ever blocking
        # the GUI.
        drain_queue(self, self._log_queue,
                    {"log": self._append_log,
                     "done": self._on_done,
                     "error": self._on_error},
                    poll_ms=self.POLL_MS)

    # -- UI -----------------------------------------------------------------

    def apply_batch_row(self, params: dict, log=None) -> None:
        """Apply a batch row's per-row overrides to this tab's state.

        Public seam the Tab 0 batch runner calls before this tab's run
        method fires, instead of poking ``_params`` directly. Consumes
        only the keys this tab owns (the 5 QC-gif knobs) and ignores the
        rest; keys the row didn't override leave tab state untouched.
        """
        p = params or {}
        for k in ("downsample_t", "max_size_px", "playback_fps",
                  "clip_low", "clip_high"):
            if k in p:
                self._params[k] = p[k]

    def _build_ui(self) -> None:
        # Panels stack vertically at their natural heights. Only panels
        # 2 (TIFF files) and 5 (Log) carry a draggable resize grip --
        # dragging extends the panel downward and the surrounding
        # scrollable wrapper lets the tab body grow as long as we want.
        # Panels 1, 3, 4 stay fixed-height; resizing one of the variable
        # panels never shrinks the others.
        dir_frame = ctk.CTkFrame(self)
        dir_frame.pack(fill="x", padx=14, pady=(14, 2))
        ctk.CTkLabel(dir_frame, text="1. Working directory (raw TIFFs)",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, columnspan=2, sticky="w",
            padx=8, pady=(6, 4))

        self.dir_var = tk.StringVar()
        ctk.CTkEntry(dir_frame, textvariable=self.dir_var).grid(
            row=1, column=0, sticky="ew", padx=(8, 6), pady=(0, 4))
        ctk.CTkButton(dir_frame, text="Browse...", width=90,
                      command=self._browse_dir).grid(
            row=1, column=1, padx=(0, 8), pady=(0, 4))

        # Subfolder search depth (BFS): 0 = only files in working_dir
        self.depth_var = tk.IntVar(value=0)
        ctk.CTkLabel(dir_frame, text="Search depth (subfolders):").grid(
            row=2, column=0, sticky="w", padx=(8, 0), pady=(0, 6))
        # ttk.Spinbox: CTk doesn't ship its own spinbox primitive, so
        # we fall back to ttk here and let apply_ttk_dark_theme blend it.
        depth_spin = ttk.Spinbox(
            dir_frame, from_=0, to=99, width=5,
            textvariable=self.depth_var, command=self._refresh_tiffs,
        )
        depth_spin.grid(row=2, column=1, sticky="w", padx=(0, 8),
                        pady=(0, 6))
        depth_spin.bind("<Return>", lambda _e: self._refresh_tiffs())
        depth_spin.bind("<FocusOut>", lambda _e: self._refresh_tiffs())

        dir_frame.columnconfigure(0, weight=1)

        # CTkFrame wrap so configure(height=) is honoured -- the
        # resize handle re-sizes this wrapper without ttk widgets
        # recomputing reqh and undoing the change.
        tiff_wrap = ctk.CTkFrame(self, fg_color="transparent")
        tiff_wrap.pack(fill="x", padx=14, pady=2)
        tiff_frame = ctk.CTkFrame(tiff_wrap)
        tiff_frame.pack(fill="both", expand=True)
        ctk.CTkLabel(
            tiff_frame,
            text="2. TIFF files  (select 1+; multi-selection -> "
                 "one grouped recording)",
            font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, columnspan=2, sticky="w",
            padx=8, pady=(6, 4))

        list_holder = ctk.CTkFrame(tiff_frame, fg_color="transparent")
        list_holder.grid(row=1, column=0, sticky="nsew",
                         padx=(8, 6), pady=(0, 4))
        list_holder.columnconfigure(0, weight=1)
        list_holder.rowconfigure(0, weight=1)
        self.tiff_listbox = tk.Listbox(
            list_holder, selectmode="extended", height=6,
            exportselection=False,
        )
        self.tiff_listbox.grid(row=0, column=0, sticky="nsew")
        list_sb = ctk.CTkScrollbar(list_holder, orientation="vertical",
                                   command=self.tiff_listbox.yview)
        list_sb.grid(row=0, column=1, sticky="ns")
        self.tiff_listbox.config(yscrollcommand=list_sb.set)
        self.tiff_listbox.bind(
            "<<ListboxSelect>>",
            lambda _e: self._update_existing_status(),
        )

        ctk.CTkButton(tiff_frame, text="Refresh", width=80,
                      command=self._refresh_tiffs).grid(
                          row=1, column=1, sticky="n", padx=(0, 8),
                          pady=(0, 4))

        ctk.CTkLabel(
            tiff_frame,
            text="Identifier (optional, used as folder name; "
                 "default = '+'-joined stems for groups):",
            anchor="w",
        ).grid(row=2, column=0, columnspan=2, sticky="w",
               padx=8, pady=(8, 2))
        self.identifier_var = tk.StringVar()
        self.identifier_var.trace_add(
            "write", lambda *_a: self._update_existing_status()
        )
        ctk.CTkEntry(tiff_frame, textvariable=self.identifier_var).grid(
            row=3, column=0, columnspan=2, sticky="ew",
            padx=8, pady=(0, 8))

        tiff_frame.columnconfigure(0, weight=1)
        # Only the list row expands when the panel is dragged taller --
        # the Identifier label + entry (rows 2 and 3) stay anchored at
        # the bottom with default weight=0, so growing the panel always
        # gives more height to the TIFF listbox.
        tiff_frame.rowconfigure(1, weight=1)

        # Draggable grip extends tiff_wrap downward when the listbox
        # needs more rows on screen.
        attach_resize_handle(self, tiff_wrap, min_height=140)

        out_frame = ctk.CTkFrame(self)
        out_frame.pack(fill="x", padx=14, pady=2)
        ctk.CTkLabel(
            out_frame,
            text="3. Output root  (creates data/<recording>/ underneath)",
            font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, columnspan=2, sticky="w",
            padx=8, pady=(6, 4))

        self.data_root_var = tk.StringVar()
        ctk.CTkEntry(out_frame, textvariable=self.data_root_var).grid(
            row=1, column=0, sticky="ew", padx=(8, 6), pady=(0, 8))
        ctk.CTkButton(out_frame, text="Browse...", width=90,
                      command=self._browse_data_root).grid(
            row=1, column=1, padx=(0, 8), pady=(0, 8))
        out_frame.columnconfigure(0, weight=1)

        action = ctk.CTkFrame(self)
        action.pack(fill="x", padx=14, pady=2)
        ctk.CTkLabel(action, text="4. Run",
                     font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(6, 4))
        action_row = ctk.CTkFrame(action, fg_color="transparent")
        action_row.pack(fill="x", padx=8, pady=(0, 8))
        self.run_btn = ctk.CTkButton(action_row, text="Run preprocessing",
                                     command=self._on_run)
        self.run_btn.pack(side="left")
        self.rerun_btn = ctk.CTkButton(action_row, text="Force rerun",
                                       command=self._on_force_rerun,
                                       state="disabled")
        self.rerun_btn.pack(side="left", padx=(6, 0))
        ctk.CTkButton(action_row, text="Advanced...",
                      command=self._on_advanced).pack(side="left", padx=(6, 0))
        self.progress = ctk.CTkProgressBar(
            action_row, mode="indeterminate", width=200)
        self.progress.pack(side="left", padx=12)
        self.status_var = tk.StringVar(value="Idle.")
        ctk.CTkLabel(action_row, textvariable=self.status_var).pack(
            side="left")

        log_wrap = ctk.CTkFrame(self, fg_color="transparent")
        log_wrap.pack(fill="x", padx=14, pady=(2, 14))
        log_frame = ctk.CTkFrame(log_wrap)
        log_frame.pack(fill="both", expand=True)
        ctk.CTkLabel(log_frame, text="5. Log",
                     font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(6, 0))
        self.log = ctk.CTkTextbox(log_frame, height=200, wrap="word",
                                  state="disabled")
        self.log.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        apply_dark_to_tk_widget(self.tiff_listbox)
        # Wheel routing is centralised in the app root's
        # ``install_scroll_router``: when the cursor is over the
        # listbox the walk-up finds the tk.Listbox and lets its
        # class binding scroll; everywhere else falls through to
        # the tab wrapper.
        # Resize grip below the log -- drag down to grow the log; the
        # scrollable tab body absorbs the extra height.
        attach_resize_handle(self, log_wrap, min_height=160, pad=(0, 0))

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
            self.rerun_btn.configure(state="normal")
            self.status_var.set(
                f"Existing outputs at {out_dir} - Run will load them; "
                f"use Force rerun to redo.")
        else:
            self.rerun_btn.configure(state="disabled")
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
        if self._job is not None and self._job.running():
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

        self.run_btn.configure(state="disabled")
        self.rerun_btn.configure(state="disabled")
        self.progress.start()
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
        qc_keys = ("downsample_t", "max_size_px", "playback_fps",
                   "clip_low", "clip_high")
        qc_params = {k: params[k] for k in qc_keys if k in params}
        explicit_identifier = self.identifier_var.get().strip() or None

        # Offload to a child process (not a thread): TIFF I/O + the
        # mean / QC-GIF work would otherwise hold the GIL and freeze the
        # GUI. Only path strings + a flag + the qc_params dict cross the
        # boundary; progress strings come back as ("log", msg) so the
        # existing ``_append_log`` handler is reused. The done payload is
        # the picklable PreprocessResult ``_on_done`` already expects.
        group = len(srcs) > 1
        rec_name = identifier if group else explicit_identifier
        from ...core.offload import run_offloaded
        from ...core import preprocessing as _pp
        self._job = run_offloaded(
            self, self._log_queue,
            _pp.preprocess_offload,
            args=([str(s) for s in srcs], data_root),
            kwargs=dict(recording_name=rec_name, group=group,
                        qc_params=qc_params),
            progress_kind="log",
            poll_ms=self.POLL_MS,
        )

    def _on_advanced(self) -> None:
        if open_advanced(self, "Preprocess - Advanced parameters",
                         self.PARAM_SPEC, self._params):
            self._append_log(
                "[GUI] Advanced parameters updated. Re-run preprocessing "
                "to apply.")

    # -- Logging / queue drain ---------------------------------------------

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _on_done(self, result: PreprocessResult) -> None:
        self.progress.stop()
        self.run_btn.configure(state="normal")
        self.rerun_btn.configure(state="normal")
        n_frames_str = (f"{result.n_frames} frames"
                        if result.n_frames > 0 else "n_frames=?")
        self.status_var.set(f"Done - {n_frames_str}.")
        self._append_log(
            f"Outputs: {result.qc_gif.name}, "
            f"{result.shifted_tiff.name}, mean.npy")
        self.state.set_result(result)

    def _on_error(self, msg: str) -> None:
        self.progress.stop()
        self.run_btn.configure(state="normal")
        self._update_existing_status()
        self.status_var.set("Error.")
        self._append_log(f"ERROR: {msg}")
        if report_stage_error(self.state, msg):
            return
        messagebox.showerror("Preprocessing failed", msg)
