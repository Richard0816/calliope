"""gui_common.py - shared state object and Tk helpers used by every
pipeline_gui tab module (preprocess_tab, qc_tab, suite2p_tab, lowpass_tab,
event_detection_tab, clustering_tab, crosscorrelation_tab).

Kept separate from pipeline_gui.py so each tab can be its own importable
module without pulling in the rest of the GUI.
"""

from __future__ import annotations

import io
import queue
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Optional

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk,
)

from .core.preprocessing import PreprocessResult


# ---------------------------------------------------------------------------
# Shared application state
# ---------------------------------------------------------------------------

class AppState:
    """Mutable state shared across tabs. Tabs subscribe via ``on_result``
    (preprocessing complete) or ``on_plane0`` (suite2p detection complete)."""

    def __init__(self) -> None:
        self.working_dir: Optional[Path] = None
        self.data_root: Optional[Path] = None
        self.selected_tiff: Optional[Path] = None
        self.result: Optional[PreprocessResult] = None
        self.plane0: Optional[Path] = None
        self.lowpass_plane0: Optional[Path] = None
        self._listeners: list = []
        self._plane0_listeners: list = []
        self._lowpass_listeners: list = []

    def subscribe(self, fn) -> None:
        self._listeners.append(fn)

    def set_result(self, result: PreprocessResult) -> None:
        self.result = result
        for fn in self._listeners:
            try:
                fn(result)
            except Exception as e:
                print(f"listener error: {e}")

    def subscribe_plane0(self, fn) -> None:
        self._plane0_listeners.append(fn)

    def set_plane0(self, plane0: Path) -> None:
        self.plane0 = Path(plane0)
        for fn in self._plane0_listeners:
            try:
                fn(self.plane0)
            except Exception as e:
                print(f"plane0 listener error: {e}")

    def subscribe_lowpass_ready(self, fn) -> None:
        self._lowpass_listeners.append(fn)

    def set_lowpass_ready(self, plane0: Path) -> None:
        self.lowpass_plane0 = Path(plane0)
        for fn in self._lowpass_listeners:
            try:
                fn(self.lowpass_plane0)
            except Exception as e:
                print(f"lowpass listener error: {e}")


# ---------------------------------------------------------------------------
# Advanced parameters dialog
# ---------------------------------------------------------------------------

class AdvancedDialog(tk.Toplevel):
    """Modal Toplevel that auto-builds an Entry/Combobox form from a
    list of parameter specs and writes accepted values into ``current``.

    Each spec is a dict with keys:
        name    : str - the dict key in ``current``
        label   : str - field label
        type    : 'float' | 'int' | 'str' | 'bool' | 'choice'
        group   : str - LabelFrame heading
        choices : list[str] - required iff type == 'choice'
        help    : str - optional one-line tooltip text shown next to label
    """

    def __init__(self, master, title: str, specs: list[dict],
                 current: dict) -> None:
        super().__init__(master)
        self.title(title)
        self.transient(master)
        self.resizable(True, True)
        self.geometry("560x600")
        self._specs = specs
        self._current = current
        self._defaults = {s["name"]: s["default"] for s in specs}
        self._vars: dict = {}
        self._accepted = False

        self._build_ui()
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self); outer.pack(fill="both", expand=True)
        canvas = tk.Canvas(outer, highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        inner = ttk.Frame(canvas, padding=8)
        win = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _resize(_e):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfigure(win, width=canvas.winfo_width())
        inner.bind("<Configure>", _resize)
        canvas.bind("<Configure>", _resize)

        groups: dict = {}
        for spec in self._specs:
            grp = spec.get("group", "Parameters")
            if grp not in groups:
                lf = ttk.LabelFrame(inner, text=grp, padding=8)
                lf.pack(fill="x", pady=(0, 8))
                lf.columnconfigure(1, weight=1)
                groups[grp] = (lf, 0)
            lf, row = groups[grp]
            self._add_field(lf, row, spec)
            groups[grp] = (lf, row + 1)

        btn_row = ttk.Frame(self, padding=(8, 4)); btn_row.pack(fill="x")
        ttk.Button(btn_row, text="Reset to defaults",
                   command=self._reset_defaults).pack(side="left")
        ttk.Button(btn_row, text="Cancel",
                   command=self._on_cancel).pack(side="right")
        ttk.Button(btn_row, text="OK",
                   command=self._on_ok).pack(side="right", padx=(0, 6))

    def _add_field(self, parent, row, spec) -> None:
        name = spec["name"]
        label = spec.get("label", name)
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        cur = self._current.get(name, spec["default"])

        if spec["type"] == "bool":
            var = tk.BooleanVar(value=bool(cur))
            ttk.Checkbutton(parent, variable=var).grid(
                row=row, column=1, sticky="w", pady=2)
        elif spec["type"] == "choice":
            var = tk.StringVar(value=str(cur))
            cb = ttk.Combobox(parent, textvariable=var, state="readonly",
                              values=list(spec.get("choices", [])))
            cb.grid(row=row, column=1, sticky="ew", pady=2)
        else:
            var = tk.StringVar(value=str(cur))
            ttk.Entry(parent, textvariable=var).grid(
                row=row, column=1, sticky="ew", pady=2)

        help_text = spec.get("help")
        if help_text:
            ttk.Label(parent, text=help_text, foreground="gray",
                      font=("", 8, "italic")).grid(
                row=row, column=2, sticky="w", padx=(8, 0), pady=2)

        self._vars[name] = (var, spec)

    def _reset_defaults(self) -> None:
        for name, (var, spec) in self._vars.items():
            d = self._defaults[name]
            if spec["type"] == "bool":
                var.set(bool(d))
            else:
                var.set(str(d))

    def _coerce(self, raw: str, spec: dict):
        t = spec["type"]
        if t == "bool":
            return bool(raw)
        if t == "int":
            return int(float(raw))
        if t == "float":
            return float(raw)
        return raw

    def _on_ok(self) -> None:
        try:
            new_values = {}
            for name, (var, spec) in self._vars.items():
                new_values[name] = self._coerce(var.get(), spec)
        except (ValueError, TypeError) as e:
            messagebox.showerror("Invalid value",
                                 f"Could not parse {name!r}: {e}")
            return
        self._current.update(new_values)
        self._accepted = True
        self.grab_release()
        self.destroy()

    def _on_cancel(self) -> None:
        self.grab_release()
        self.destroy()


def spec_defaults(specs: list[dict]) -> dict:
    """Build a {name: default} dict from a PARAM_SPEC list."""
    return {s["name"]: s["default"] for s in specs}


def open_advanced(master, title: str, specs: list[dict],
                  current: dict) -> bool:
    """Open AdvancedDialog modally; returns True if user clicked OK."""
    dlg = AdvancedDialog(master, title, specs, current)
    master.wait_window(dlg)
    return dlg._accepted


# ---------------------------------------------------------------------------
# Manual ROI spec parsing / formatting
# ---------------------------------------------------------------------------

def parse_manual_roi_spec(text: str, n_total: int) -> list[int]:
    """Parse a manual-ROI spec into a sorted unique list of original
    suite2p indices in [0, n_total). Accepts a single number ("4"), a
    comma list ("4,5,6"), a range ("1-5"), or any combination
    ("1-3,7,10-12"). Whitespace around tokens is allowed. Raises
    ValueError on empty/unparseable input or out-of-range indices.
    """
    if text is None:
        raise ValueError("empty input")
    indices: set[int] = set()
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a_str, b_str = token.split("-", 1)
            try:
                a = int(a_str.strip())
                b = int(b_str.strip())
            except ValueError:
                raise ValueError(f"bad range token {token!r}")
            if a > b:
                a, b = b, a
            indices.update(range(a, b + 1))
        else:
            try:
                indices.add(int(token))
            except ValueError:
                raise ValueError(f"bad token {token!r}")
    if not indices:
        raise ValueError("no indices parsed")
    out = sorted(indices)
    bad = [i for i in out if i < 0 or i >= n_total]
    if bad:
        raise ValueError(
            f"out of range [0, {n_total}): {bad[:5]}"
            + ("..." if len(bad) > 5 else ""))
    return out


def format_roi_indices(indices: list[int]) -> str:
    """Render a sorted list of ints as compact 'a-b' runs joined by commas
    (e.g. [1,2,3,7,10,11,12] -> '1-3,7,10-12')."""
    if not indices:
        return ""
    runs: list[str] = []
    s = indices[0]
    p = s
    for i in indices[1:]:
        if i == p + 1:
            p = i
        else:
            runs.append(f"{s}-{p}" if s != p else f"{s}")
            s = i
            p = i
    runs.append(f"{s}-{p}" if s != p else f"{s}")
    return ",".join(runs)


# ---------------------------------------------------------------------------
# Matplotlib navigation toolbar with "Save data..." button
# ---------------------------------------------------------------------------

def attach_fig_toolbar(canvas: FigureCanvasTkAgg,
                       parent_frame,
                       data_basename: str = "plot_data",
                       data_provider=None,
                       ) -> NavigationToolbar2Tk:
    """Attach a matplotlib navigation toolbar (pan / zoom / Save Figure)
    above ``canvas`` inside ``parent_frame``. Caller must pack/grid the
    canvas widget AFTER this call so the toolbar lands above it.

    The "Save data..." button next to it dumps the underlying
    Line2D / image data to CSV via ``plot_data_export.save_figure_data``;
    ``data_basename`` is the default filename suggested in the save
    dialog. Pass ``data_provider`` for tabs that need a custom export
    shape (callable taking a single ``parent`` argument).
    """
    from . import plot_data_export
    tb_frame = ttk.Frame(parent_frame)
    tb_frame.pack(side="top", fill="x")
    tb = NavigationToolbar2Tk(canvas, tb_frame, pack_toolbar=False)
    tb.update()
    tb.pack(side="left", fill="x")
    if data_provider is None:
        cmd = lambda c=canvas, p=tb_frame, b=data_basename: \
            plot_data_export.save_figure_data(c.figure, p, b)
    else:
        cmd = lambda p=tb_frame: data_provider(p)
    ttk.Button(
        tb_frame, text="Save data...", command=cmd,
    ).pack(side="left", padx=(8, 0))
    return tb


# ---------------------------------------------------------------------------
# Worker-thread stdout/stderr capture
# ---------------------------------------------------------------------------

class QueueWriter(io.TextIOBase):
    """File-like object that line-buffers writes onto a queue.Queue as
    ('log', line) tuples. Used to redirect a worker thread's stdout/stderr
    into the GUI log panel without touching upstream code."""

    def __init__(self, q: "queue.Queue") -> None:
        self.q = q
        self._buf = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self.q.put(("log", line))
        return len(s)

    def flush(self) -> None:
        if self._buf:
            self.q.put(("log", self._buf))
            self._buf = ""
