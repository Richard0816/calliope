"""calliope.tabs.batch.tab - Tab 0: Batch runner.

What this tab does
------------------
Drives the full Tabs 3-8 pipeline (detection -> lowpass -> events
-> clustering -> xcorr -> spatial figures) over a list of
recordings. One row per recording: identifier, TIFFs, per-row
"Edit parameters" button. A top-level "Apply to all rows" button
sets defaults across every row.

What the user sees
------------------
- Top: working directory + recursion depth + output folder + Scan;
  "Apply defaults to all rows" + "Run all" / "Abort".
- Middle: scrollable list of recording rows. Each row has an
  identifier entry, a list of TIFFs (with Browse), an "Edit
  parameters..." button, and a per-row status / progress label.
- Bottom: live console (stdout/stderr captured via QueueWriter).

State persistence
-----------------
The list of rows + per-row params is serialised to
``<output>/calliope_batch.json`` on every Run All so closing &
re-opening the GUI restores the queue.

After every Run All, ``<output>/batch_report.csv`` is written with
``recording_id, status, plane0, total_s, <stage>_status,
<stage>_duration_s`` columns for downstream analysis.
"""

from __future__ import annotations

import contextlib
import csv
import json
import queue
import threading
import time
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from ...gui_common import (
    AppState, QueueWriter, open_advanced, spec_defaults,
)


BATCH_JSON_NAME = "calliope_batch.json"
BATCH_REPORT_NAME = "batch_report.csv"
TIFF_GLOBS = ("*.tif", "*.tiff", "*.TIF", "*.TIFF")


def _build_batch_param_spec() -> list:
    """Assemble the union PARAM_SPEC.

    Entries are emitted in pipeline order (preprocess -> detection ->
    low-pass -> events -> clustering -> xcorr -> pipeline-wide) and
    each source group's label is rewritten with a numbered stage
    prefix so the Advanced dialog reads top-to-bottom in operation
    order. Tab 4's "Low-pass" cutoff knob (which Tab 4 binds to a
    slider, not a PARAM_SPEC entry) is inserted alongside the
    rest of the low-pass section.
    """
    # Original-group -> renamed-group. Anything not in the map keeps
    # its native label.
    rename = {
        # 1. Preprocess (Tab 1)
        "Blob detection": "1. Preprocess - Blob detection",
        "QC gif": "1. Preprocess - QC GIF",
        # 2. Detection (Tab 3)
        "Sparsery": "2. Detection - Sparsery",
        "Cellpose": "2. Detection - Cellpose",
        "Merge": "2. Detection - Merge",
        "dF/F": "2. Detection - dF/F",
        "Default low-pass": "2. Detection - Default low-pass",
        "Pixel scale": "2. Detection - Pixel scale",
        "GPU": "2. Detection - GPU",
        # 3. Low-pass + derivative (Tab 4)
        "Low-pass": "3. Low-pass - Cutoff",
        "Low-pass filter": "3. Low-pass - Butterworth",
        "Derivative": "3. Low-pass - SG derivative",
        "Slider bounds": "3. Low-pass - Slider bounds",
        # 4. Event detection (Tab 5)
        "Per-ROI hysteresis": "4. Events - Per-ROI hysteresis",
        "Display": "4. Events - Display",
        "Population events - density":
            "4. Events - Population density",
        "Population events - peaks":
            "4. Events - Population peaks",
        "Population events - baseline":
            "4. Events - Population baseline",
        "Population events - boundaries":
            "4. Events - Population boundaries",
        "Population events - gaussian fit":
            "4. Events - Population Gaussian fit",
        # 5. Clustering / 6. Xcorr / 7. Pipeline-wide added below
    }

    spec: list = []
    seen: set[str] = set()

    def _extend(items):
        for entry in items:
            name = entry.get("name")
            if not name or name in seen:
                continue
            new = dict(entry)
            grp = new.get("group", "Parameters")
            new["group"] = rename.get(grp, grp)
            spec.append(new)
            seen.add(name)

    # 1. Preprocess
    try:
        from ..preprocess.tab import PreprocessTab
        _extend(PreprocessTab.PARAM_SPEC)
    except Exception:
        pass

    # 2. Detection
    try:
        from ..suite2p.tab import Suite2pTab
        _extend(Suite2pTab.PARAM_SPEC)
    except Exception:
        pass

    # 3. Low-pass: cutoff first (binds to the slider value), then the
    # Tab 4 PARAM_SPEC entries for filter / derivative / slider bounds.
    spec.append({
        "name": "cutoff_hz", "label": "Low-pass cutoff (Hz)",
        "type": "float", "default": 1.0,
        "group": "3. Low-pass - Cutoff",
        "help": "cutoff applied to every kept ROI when batch runs Tab 4",
    })
    seen.add("cutoff_hz")
    try:
        from ..lowpass.tab import LowpassTab
        _extend(LowpassTab.PARAM_SPEC)
    except Exception:
        pass

    # 4. Event detection
    try:
        from ..event_detection.tab import EventDetectionTab
        _extend(EventDetectionTab.PARAM_SPEC)
    except Exception:
        pass

    # 5. Clustering (no source PARAM_SPEC)
    spec.extend([
        {"name": "prefix", "label": "dF/F prefix",
         "type": "str", "default": "r0p7_filtered_",
         "group": "5. Clustering",
         "help": "memmap prefix; filtered_ uses the cell-filter mask"},
        {"name": "threshold",
         "label": "Cluster threshold (0=auto)",
         "type": "float", "default": 0.0, "group": "5. Clustering",
         "help": "linkage cut height; 0 -> auto via auto_choose_threshold"},
        {"name": "palette", "label": "Cluster palette",
         "type": "str", "default": "tab10", "group": "5. Clustering"},
    ])

    # 6. Cross-correlation (no source PARAM_SPEC)
    spec.extend([
        {"name": "max_lag_seconds", "label": "xcorr max lag (s)",
         "type": "float", "default": 2.0,
         "group": "6. Cross-correlation"},
        {"name": "zero_lag", "label": "Compute zero-lag correlation",
         "type": "bool", "default": True,
         "group": "6. Cross-correlation"},
        {"name": "use_gpu", "label": "Use GPU for xcorr",
         "type": "bool", "default": True,
         "group": "6. Cross-correlation"},
    ])

    # 7. Pipeline-wide
    spec.extend([
        {"name": "baseline_mode", "label": "dF/F baseline mode",
         "type": "choice", "choices": ["first_n", "rolling"],
         "default": "first_n", "group": "7. Pipeline-wide"},
        {"name": "baseline_min", "label": "Baseline duration (min)",
         "type": "float", "default": 2.0, "group": "7. Pipeline-wide",
         "help": "first-N-min baseline length when baseline_mode=first_n"},
    ])

    return spec


def _pick_tiffs_dialog(parent: tk.Misc, *, title: str,
                       initial_dir: str = "",
                       current_paths: Optional[list[str]] = None,
                       initial_depth: int = 0) -> Optional[list[str]]:
    """Modal Listbox-based TIFF picker (same UX as Tab 1).

    Shows the TIFFs under ``initial_dir`` (recursing to ``initial_depth``
    sub-levels) in a multi-select Listbox. Returns the list of absolute
    paths the user confirmed, or ``None`` if the dialog was cancelled.

    The user can change the search root + depth and hit "Refresh" to
    re-populate the list. Items already in ``current_paths`` are
    pre-selected.
    """
    from ...core import preprocessing

    win = tk.Toplevel(parent)
    win.title(title)
    win.transient(parent.winfo_toplevel())
    win.grab_set()
    win.geometry("700x520")

    body = ttk.Frame(win, padding=8)
    body.pack(fill="both", expand=True)

    row = ttk.Frame(body)
    row.pack(fill="x", pady=(0, 6))
    ttk.Label(row, text="Folder:", width=10).pack(side="left")
    dir_var = tk.StringVar(value=initial_dir or "")
    ttk.Entry(row, textvariable=dir_var).pack(
        side="left", fill="x", expand=True, padx=(0, 4))
    ttk.Button(
        row, text="Browse...", width=10,
        command=lambda: dir_var.set(
            filedialog.askdirectory(title="Pick search folder",
                                    parent=win) or dir_var.get())
    ).pack(side="left")

    row = ttk.Frame(body)
    row.pack(fill="x", pady=(0, 6))
    ttk.Label(row, text="Depth:", width=10).pack(side="left")
    depth_var = tk.IntVar(value=max(0, int(initial_depth)))
    ttk.Spinbox(row, from_=0, to=6, width=4,
                textvariable=depth_var).pack(side="left")
    refresh_btn = ttk.Button(row, text="Refresh", width=10)
    refresh_btn.pack(side="left", padx=(8, 0))
    status_var = tk.StringVar(value="")
    ttk.Label(row, textvariable=status_var,
              font=("", 9, "italic")).pack(side="left", padx=(8, 0))

    list_holder = ttk.Frame(body)
    list_holder.pack(fill="both", expand=True)
    list_holder.columnconfigure(0, weight=1)
    list_holder.rowconfigure(0, weight=1)
    listbox = tk.Listbox(list_holder, selectmode="extended",
                         exportselection=False)
    listbox.grid(row=0, column=0, sticky="nsew")
    sb = ttk.Scrollbar(list_holder, orient="vertical",
                       command=listbox.yview)
    sb.grid(row=0, column=1, sticky="ns")
    listbox.configure(yscrollcommand=sb.set)

    paths_state: list[str] = []  # mutated by _refresh; lives in closure
    current_set = {str(Path(p).resolve()) for p in (current_paths or [])}

    def _refresh():
        d = dir_var.get().strip()
        listbox.delete(0, "end")
        paths_state.clear()
        if not d:
            status_var.set("Pick a folder.")
            return
        if not Path(d).is_dir():
            status_var.set(f"Not a folder: {d}")
            return
        depth = max(0, int(depth_var.get()))
        try:
            paths = preprocessing.list_tiffs(d, max_depth=depth)
        except Exception as e:
            status_var.set(f"List error: {e}")
            return
        d_path = Path(d)
        for p in paths:
            try:
                rel = Path(p).relative_to(d_path).as_posix()
            except ValueError:
                rel = str(p)
            paths_state.append(str(p))
            listbox.insert("end", rel)
        # Pre-select rows whose absolute path matches current_paths.
        for i, abs_p in enumerate(paths_state):
            if str(Path(abs_p).resolve()) in current_set:
                listbox.selection_set(i)
        n_sel = len(listbox.curselection())
        status_var.set(
            f"{len(paths)} TIFFs"
            + (f"  ({n_sel} pre-selected)" if n_sel else ""))

    refresh_btn.configure(command=_refresh)

    foot = ttk.Frame(body)
    foot.pack(fill="x", pady=(8, 0))
    result: dict = {"paths": None}

    def _confirm():
        sel = listbox.curselection()
        result["paths"] = [paths_state[i] for i in sel]
        win.destroy()

    def _cancel():
        result["paths"] = None
        win.destroy()

    ttk.Button(foot, text="Cancel", width=10,
               command=_cancel).pack(side="right")
    ttk.Button(foot, text="Confirm",
               command=_confirm).pack(side="right", padx=(0, 6))

    _refresh()
    parent.wait_window(win)
    return result["paths"]


class BatchRow:
    """One recording in the queue. Owns its widgets, identifier,
    tiff list, parameter overrides, and status label.
    """

    def __init__(self, parent: ttk.Frame, tab: "BatchTab",
                 *, identifier: str = "",
                 tiffs: Optional[list[str]] = None,
                 params: Optional[dict] = None) -> None:
        self.tab = tab
        self.frame = ttk.Frame(parent, padding=(2, 4))
        self.frame.pack(fill="x", padx=4)

        self.id_var = tk.StringVar(value=identifier)
        self.tiff_var = tk.StringVar(
            value="; ".join(tiffs or []))
        self.params: dict = dict(params) if params is not None else {}
        self.status_var = tk.StringVar(value="ready")

        ttk.Entry(self.frame, textvariable=self.id_var, width=22) \
            .grid(row=0, column=0, padx=(0, 6))
        ttk.Entry(self.frame, textvariable=self.tiff_var, width=46) \
            .grid(row=0, column=1, padx=(0, 4), sticky="ew")
        ttk.Button(self.frame, text="Browse...", width=10,
                   command=self._on_browse) \
            .grid(row=0, column=2, padx=(0, 4))
        ttk.Button(self.frame, text="Edit params", width=12,
                   command=self._on_edit_params) \
            .grid(row=0, column=3, padx=(0, 4))
        ttk.Label(self.frame, textvariable=self.status_var,
                  width=18, anchor="w") \
            .grid(row=0, column=4, padx=(0, 4))
        ttk.Button(self.frame, text="x", width=2,
                   command=self._on_remove) \
            .grid(row=0, column=5)
        self.frame.columnconfigure(1, weight=1)

    @property
    def identifier(self) -> str:
        return self.id_var.get().strip() or "unnamed"

    def tiff_list(self) -> list[str]:
        raw = self.tiff_var.get().strip()
        if not raw:
            return []
        return [s.strip() for s in raw.split(";") if s.strip()]

    def to_dict(self) -> dict:
        return {
            "identifier": self.id_var.get(),
            "tiffs": self.tiff_list(),
            "params": dict(self.params),
        }

    def set_status(self, msg: str) -> None:
        self.status_var.set(msg)

    def merge_defaults(self, defaults: dict) -> None:
        """Overwrite per-row params with defaults; called by the
        top-level "Apply to all rows" button."""
        self.params = dict(defaults)

    def destroy(self) -> None:
        self.frame.destroy()

    def _on_browse(self) -> None:
        paths = _pick_tiffs_dialog(
            self.tab,
            title=f"TIFFs for '{self.identifier}'",
            initial_dir=self.tab.workdir_var.get().strip()
                        or self.tab.outdir_var.get().strip(),
            current_paths=self.tiff_list(),
            initial_depth=int(self.tab.depth_var.get() or 0),
        )
        if paths is not None:
            self.tiff_var.set("; ".join(paths))

    def _on_edit_params(self) -> None:
        # Start from row params -> tab defaults -> spec defaults.
        merged = dict(self.tab.default_params)
        merged.update(self.params)
        if open_advanced(
                self.tab,
                f"Edit parameters - '{self.identifier}'",
                self.tab.PARAM_SPEC, merged):
            self.params = dict(merged)

    def _on_remove(self) -> None:
        self.tab.remove_row(self)


class BatchTab(ttk.Frame):
    """Tab 0: queue of recordings + run-all worker."""

    POLL_MS = 100

    PARAM_SPEC = _build_batch_param_spec()

    def __init__(self, master, state: AppState) -> None:
        super().__init__(master, padding=10)
        self.state = state
        self._rows: list[BatchRow] = []
        self.default_params: dict = spec_defaults(self.PARAM_SPEC)

        self._log_queue: queue.Queue = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._abort_flag = threading.Event()
        self._build_ui()
        self.after(self.POLL_MS, self._drain_log_queue)

    # -- UI ----------------------------------------------------------------

    def _build_ui(self) -> None:
        top = ttk.LabelFrame(self, text="Batch run setup", padding=8)
        top.pack(fill="x", pady=(0, 6))

        row = ttk.Frame(top); row.pack(fill="x", pady=2)
        ttk.Label(row, text="Working dir:", width=16).pack(side="left")
        self.workdir_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.workdir_var) \
            .pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(row, text="Browse...", width=10,
                   command=self._on_pick_workdir).pack(side="left")
        ttk.Label(row, text="Depth:").pack(side="left", padx=(8, 2))
        self.depth_var = tk.IntVar(value=2)
        ttk.Spinbox(row, from_=0, to=6, width=4,
                    textvariable=self.depth_var).pack(side="left")
        ttk.Button(row, text="Scan", width=8,
                   command=self._on_scan).pack(side="left", padx=(8, 0))

        row = ttk.Frame(top); row.pack(fill="x", pady=2)
        ttk.Label(row, text="Output folder:", width=16).pack(side="left")
        self.outdir_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.outdir_var) \
            .pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(row, text="Browse...", width=10,
                   command=self._on_pick_outdir).pack(side="left")
        ttk.Button(row, text="Reload queue", width=14,
                   command=self._on_reload_queue) \
            .pack(side="left", padx=(8, 0))

        row = ttk.Frame(top); row.pack(fill="x", pady=(6, 2))
        ttk.Button(row, text="Apply defaults to all rows",
                   command=self._on_edit_defaults) \
            .pack(side="left")
        ttk.Button(row, text="+ Add row", width=12,
                   command=self.add_row) \
            .pack(side="left", padx=(8, 0))
        self.run_btn = ttk.Button(
            row, text="Run all", width=12,
            command=self._on_run_all)
        self.run_btn.pack(side="right")
        self.abort_btn = ttk.Button(
            row, text="Abort", width=10,
            command=self._on_abort, state="disabled")
        self.abort_btn.pack(side="right", padx=(0, 6))

        # Headers + scrollable rows -----------------------------------------
        headers = ttk.Frame(self, padding=(4, 0))
        headers.pack(fill="x")
        for txt, w in (("Identifier", 22), ("TIFF file(s) (semicolon-sep)", 50),
                       ("", 11), ("", 13), ("Status", 18), ("", 3)):
            ttk.Label(headers, text=txt, width=w, anchor="w") \
                .pack(side="left", padx=(0, 4))

        body = ttk.LabelFrame(self, text="Recordings", padding=4)
        body.pack(fill="both", expand=True, pady=(0, 6))
        canvas = tk.Canvas(body, highlightthickness=0)
        canvas.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(body, orient="vertical", command=canvas.yview)
        sb.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=sb.set)
        self._rows_frame = ttk.Frame(canvas)
        self._rows_canvas_id = canvas.create_window(
            (0, 0), window=self._rows_frame, anchor="nw")

        def _on_frame_resize(_e=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_resize(e):
            canvas.itemconfig(self._rows_canvas_id, width=e.width)

        self._rows_frame.bind("<Configure>", _on_frame_resize)
        canvas.bind("<Configure>", _on_canvas_resize)

        # Console -----------------------------------------------------------
        console_frame = ttk.LabelFrame(self, text="Run log", padding=4)
        console_frame.pack(fill="both", expand=True)
        self.log = tk.Text(console_frame, height=10, wrap="word",
                           state="disabled")
        log_sb = ttk.Scrollbar(console_frame, orient="vertical",
                               command=self.log.yview)
        self.log.configure(yscrollcommand=log_sb.set)
        log_sb.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)

    # -- Row management ---------------------------------------------------

    def add_row(self, *, identifier: str = "",
                tiffs: Optional[list[str]] = None,
                params: Optional[dict] = None) -> BatchRow:
        row = BatchRow(self._rows_frame, self,
                       identifier=identifier, tiffs=tiffs,
                       params=params if params is not None
                                       else dict(self.default_params))
        self._rows.append(row)
        return row

    def remove_row(self, row: BatchRow) -> None:
        if row in self._rows:
            self._rows.remove(row)
        row.destroy()

    def _clear_rows(self) -> None:
        for row in list(self._rows):
            row.destroy()
        self._rows.clear()

    # -- Top-bar actions ---------------------------------------------------

    def _on_pick_workdir(self) -> None:
        path = filedialog.askdirectory(title="Working directory")
        if path:
            self.workdir_var.set(path)

    def _on_pick_outdir(self) -> None:
        path = filedialog.askdirectory(title="Output folder")
        if path:
            self.outdir_var.set(path)

    def _on_scan(self) -> None:
        """Walk the working directory up to ``depth`` levels and add
        one row per folder that contains TIFFs.
        """
        root = self.workdir_var.get().strip()
        if not root:
            messagebox.showinfo("Pick working directory",
                                "Choose a folder to scan first.")
            return
        depth = max(0, int(self.depth_var.get()))
        root_path = Path(root)
        if not root_path.is_dir():
            messagebox.showerror("Bad path", f"{root} is not a folder.")
            return
        found: dict[Path, list[str]] = {}
        for path in self._walk_to_depth(root_path, depth):
            tiffs = self._tiffs_in(path)
            if tiffs:
                found[path] = tiffs
        if not found:
            messagebox.showinfo(
                "No TIFFs",
                f"Found no .tif/.tiff files under {root_path} "
                f"at depth {depth}.")
            return
        for folder, tiffs in sorted(found.items()):
            ident = folder.name
            self.add_row(identifier=ident, tiffs=tiffs)
        self._log(f"Scan: added {len(found)} rows from {root_path}")

    @staticmethod
    def _walk_to_depth(root: Path, depth: int):
        """Yield ``root`` and every subfolder up to ``depth`` levels
        deep (depth=0 means just ``root``).
        """
        yield root
        if depth <= 0:
            return
        stack = [(root, 0)]
        while stack:
            cur, d = stack.pop()
            if d >= depth:
                continue
            try:
                children = [c for c in cur.iterdir() if c.is_dir()]
            except OSError:
                continue
            for child in children:
                yield child
                stack.append((child, d + 1))

    @staticmethod
    def _tiffs_in(folder: Path) -> list[str]:
        out: list[str] = []
        for pat in TIFF_GLOBS:
            out.extend(str(p) for p in folder.glob(pat))
        return sorted(set(out))

    def _on_edit_defaults(self) -> None:
        snapshot = dict(self.default_params)
        if not open_advanced(
                self, "Default parameters (apply to all rows)",
                self.PARAM_SPEC, snapshot):
            return
        self.default_params = snapshot
        for row in self._rows:
            row.merge_defaults(self.default_params)
        self._log(f"Defaults updated and applied to {len(self._rows)} rows.")

    def _on_reload_queue(self) -> None:
        out = self.outdir_var.get().strip()
        if not out:
            messagebox.showinfo("Pick output folder",
                                "Set the output folder first.")
            return
        path = Path(out) / BATCH_JSON_NAME
        if not path.exists():
            messagebox.showinfo("No saved queue",
                                f"{path} not found.")
            return
        try:
            data = json.loads(path.read_text())
        except Exception as e:
            messagebox.showerror("Reload failed", str(e))
            return
        self._clear_rows()
        defaults = data.get("default_params")
        if isinstance(defaults, dict):
            self.default_params = defaults
        for entry in data.get("rows", []):
            self.add_row(
                identifier=entry.get("identifier", ""),
                tiffs=entry.get("tiffs", []),
                params=entry.get("params", {}),
            )
        self._log(f"Loaded {len(self._rows)} rows from {path}")

    # -- Persistence ------------------------------------------------------

    def _save_queue(self) -> None:
        out = self.outdir_var.get().strip()
        if not out:
            return
        out_path = Path(out)
        out_path.mkdir(parents=True, exist_ok=True)
        payload = {
            "default_params": self.default_params,
            "rows": [r.to_dict() for r in self._rows],
        }
        (out_path / BATCH_JSON_NAME).write_text(
            json.dumps(payload, indent=2, default=str))

    # -- Run all ----------------------------------------------------------

    def _on_run_all(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            messagebox.showinfo("Busy", "A run is already in progress.")
            return
        if not self._rows:
            messagebox.showinfo("Empty queue",
                                "Add at least one row before running.")
            return
        out = self.outdir_var.get().strip()
        if not out:
            messagebox.showinfo("Pick output folder",
                                "Set the output folder first.")
            return
        # Validate TIFFs.
        bad = [r for r in self._rows if not r.tiff_list()]
        if bad:
            messagebox.showerror(
                "Empty TIFFs",
                f"Row '{bad[0].identifier}' has no TIFF files.")
            return
        self._save_queue()

        self.run_btn.config(state="disabled")
        self.abort_btn.config(state="normal")
        self._abort_flag.clear()
        for row in self._rows:
            row.set_status("queued")

        rows_snapshot = list(self._rows)
        out_root = Path(out)

        def worker():
            try:
                self._run_pipeline_loop(rows_snapshot, out_root)
            except Exception as e:
                self._log_queue.put(
                    ("log",
                     f"[batch] worker crashed: {e}\n"
                     f"{traceback.format_exc()}"))
            finally:
                self._log_queue.put(("done", None))

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    def _on_abort(self) -> None:
        self._abort_flag.set()
        self._log("[batch] abort requested; will stop after current row.")

    def _run_pipeline_loop(self, rows: list[BatchRow],
                           out_root: Path) -> None:
        """Worker thread body. Iterate rows, run the full pipeline,
        write batch_report.csv at the end.
        """
        from ...core import batch_pipeline

        writer = QueueWriter(self._log_queue)
        results: list[dict] = []

        with contextlib.redirect_stdout(writer), \
                contextlib.redirect_stderr(writer):
            for row in rows:
                if self._abort_flag.is_set():
                    self._log_queue.put(("status", (row, "aborted")))
                    break
                ident = row.identifier
                self._log_queue.put(("status", (row, "running")))
                rec_save = out_root / ident
                rec_save.mkdir(parents=True, exist_ok=True)
                tiffs = row.tiff_list()
                if not tiffs:
                    self._log_queue.put(
                        ("log",
                         f"[{ident}] FAILED: no TIFFs specified."))
                    self._log_queue.put(("status", (row, "failed")))
                    results.append({
                        "recording_id": ident, "status": "failed",
                        "plane0": "", "total_s": 0.0,
                        "error": "no_tiffs_specified",
                    })
                    continue
                self._log_queue.put(("log", ""))
                if len(tiffs) == 1:
                    self._log_queue.put(
                        ("log",
                         f"--- Recording '{ident}' "
                         f"({Path(tiffs[0]).name}) ---"))
                else:
                    self._log_queue.put(
                        ("log",
                         f"--- Recording '{ident}' "
                         f"({len(tiffs)} TIFFs, in trailing-index "
                         f"order):"))
                    for s in tiffs:
                        self._log_queue.put(
                            ("log", f"      {Path(s).name}"))
                    self._log_queue.put(("log", "---"))
                self._log_queue.put(
                    ("log", f"  output -> {rec_save}"))
                params = dict(self.default_params)
                params.update(row.params)
                params.setdefault("cutoff_hz",
                                  params.get("default_lowpass_hz", 1.0))
                t0 = time.time()
                try:
                    result = batch_pipeline.run_recording(
                        src_tiffs=tiffs,
                        save_folder=str(rec_save),
                        params=params,
                        baseline_mode=str(
                            params.get("baseline_mode", "first_n")),
                        baseline_min=float(
                            params.get("baseline_min", 2.0)),
                        recording_id=ident,
                        progress_cb=lambda msg: self._log_queue.put(
                            ("log", msg)),
                    )
                    result["total_s"] = time.time() - t0
                    results.append(result)
                    self._log_queue.put(
                        ("status", (row, result.get("status", "ok"))))
                except Exception as e:
                    self._log_queue.put(
                        ("log",
                         f"[{ident}] worker exception: {e}\n"
                         f"{traceback.format_exc()}"))
                    self._log_queue.put(("status", (row, "failed")))
                    results.append({
                        "recording_id": ident, "status": "failed",
                        "plane0": "", "total_s": time.time() - t0,
                        "error": str(e),
                    })

        self._write_report(out_root, results)

    @staticmethod
    def _write_report(out_root: Path, results: list[dict]) -> None:
        """Flatten the per-row stage status dicts into a CSV."""
        if not results:
            return
        report_path = out_root / BATCH_REPORT_NAME
        all_stages: list[str] = []
        for r in results:
            for st in (r.get("stages") or {}):
                if st not in all_stages:
                    all_stages.append(st)
        header = ["recording_id", "status", "plane0", "total_s", "error"]
        for st in all_stages:
            header.append(f"{st}_status")
            header.append(f"{st}_duration_s")
        with report_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in results:
                row = [
                    r.get("recording_id", ""),
                    r.get("status", ""),
                    r.get("plane0", ""),
                    f"{r.get('total_s', 0.0):.2f}",
                    r.get("error", ""),
                ]
                stages = r.get("stages") or {}
                for st in all_stages:
                    info = stages.get(st) or {}
                    row.append(info.get("status", ""))
                    row.append(f"{info.get('duration_s', 0.0):.2f}")
                w.writerow(row)

    # -- Console drain ----------------------------------------------------

    def _drain_log_queue(self) -> None:
        try:
            while True:
                kind, payload = self._log_queue.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "status":
                    row, status = payload
                    if row in self._rows:
                        row.set_status(str(status))
                elif kind == "done":
                    self.run_btn.config(state="normal")
                    self.abort_btn.config(state="disabled")
                    self._append_log("[batch] all rows processed.")
        except queue.Empty:
            pass
        self.after(self.POLL_MS, self._drain_log_queue)

    def _append_log(self, text: str) -> None:
        if not text:
            return
        self.log.config(state="normal")
        self.log.insert("end", text.rstrip() + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def _log(self, msg: str) -> None:
        self._log_queue.put(("log", msg))
