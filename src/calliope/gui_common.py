"""gui_common.py - shared state object and Tk helpers used by every
pipeline_gui tab module.

Why this file exists
--------------------
The eight tabs of CalLIOPE need to know about each other -- e.g. when
Tab 1 finishes preprocessing, Tab 2 should pick up the result; when
Tab 5 detects events, Tabs 7 and 8 should redraw with those events.
Rather than wiring every tab to every other tab, all of them subscribe
to a single ``AppState`` object and that object broadcasts state
changes. This is the same idea as a Redux store in a JavaScript app or
an event bus in a Shiny module.

Three other reusable Tk helpers also live here:

- ``AdvancedDialog`` / ``open_advanced``: an auto-built parameter form
  (one Entry / Checkbutton / Combobox per parameter spec) that any tab
  can pop up to let the user tweak its hyper-parameters.
- ``parse_manual_roi_spec`` / ``format_roi_indices``: convert between
  user-typed strings like "1-3,7,10-12" and Python lists of ints. Used
  by Tabs 4, 5 and 6 for the manual ROI subset feature.
- ``attach_fig_toolbar``: wraps every matplotlib figure with a
  pan / zoom / save toolbar plus a "Save data..." button that writes
  the underlying numerics to CSV.
- ``QueueWriter``: a file-like object that captures ``print(...)``
  output from a worker thread and puts each line onto a ``queue.Queue``
  so the main thread can fold it into the GUI's log panel.

Kept separate from ``pipeline_gui.py`` so each tab can be its own
importable module without pulling in the rest of the GUI.
"""

# See pipeline_gui.py for what ``from __future__ import annotations``
# does. Short version: turns type hints into strings, lets us forward-
# reference names that are defined later in the file.
from __future__ import annotations

# Standard-library bits we'll need.
import io        # Provides the ``TextIOBase`` base class for QueueWriter.
import queue     # Thread-safe FIFO queue used by QueueWriter.
import tkinter as tk
from pathlib import Path  # Modern object-oriented filesystem path API.
from tkinter import messagebox, ttk
from typing import Optional

# matplotlib's Tk-Agg backend bundles two helper classes:
# - ``FigureCanvasTkAgg``: turns a Figure into a Tk widget you can pack
#   / grid into a frame.
# - ``NavigationToolbar2Tk``: the familiar pan / zoom / save toolbar.
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk,
)

# Imported only so the type hint ``Optional[PreprocessResult]`` below
# works -- this file never instantiates PreprocessResult itself.
from .core.preprocessing import PreprocessResult


# ---------------------------------------------------------------------------
# Shared application state
# ---------------------------------------------------------------------------

class AppState:
    """Mutable state shared across tabs.

    Each "channel" (preprocess result, plane0 path, lowpass-ready flag,
    event-detection results) follows the same simple recipe:

        1. A public attribute holds the latest value (or ``None``).
        2. A private list holds the subscriber callbacks.
        3. ``subscribe_X(fn)`` appends ``fn`` to the list.
        4. ``set_X(value)`` updates the attribute and fans the new
           value out to every subscriber.

    Tabs publish via ``set_*`` and listen via ``subscribe_*``. They
    never call into each other directly. If you have a Shiny / R6
    background, this is essentially a hand-rolled reactive value.
    """

    def __init__(self) -> None:
        # Channels: each is the *latest* published value, or None until
        # the producer publishes one.
        self.working_dir: Optional[Path] = None
        self.data_root: Optional[Path] = None
        self.selected_tiff: Optional[Path] = None
        self.result: Optional[PreprocessResult] = None
        self.plane0: Optional[Path] = None
        self.lowpass_plane0: Optional[Path] = None
        self.event_results: Optional[dict] = None
        # Tab 6 + Tab 7 broadcast a "stage finished" signal that the
        # batch runner subscribes to so it can advance to the next
        # stage. Payloads carry the plane0 / output path the producer
        # tab just wrote.
        self.clusters_ready: Optional[Path] = None
        self.xcorr_ready: Optional[Path] = None
        # Subscriber lists. Underscore prefix is convention for
        # "internal -- don't poke this from outside the class". Python
        # does not actually enforce privacy.
        self._listeners: list = []
        self._plane0_listeners: list = []
        self._lowpass_listeners: list = []
        self._event_listeners: list = []
        self._clusters_listeners: list = []
        self._xcorr_listeners: list = []

    # ---- preprocess result channel ----

    def subscribe(self, fn) -> None:
        """Register ``fn`` to be called every time ``set_result`` runs."""
        self._listeners.append(fn)

    def set_result(self, result: PreprocessResult) -> None:
        """Publish a new ``PreprocessResult`` to every subscriber.

        Each subscriber is called inside its own try/except so one
        broken listener doesn't take the rest down with it.
        """
        self.result = result
        for fn in self._listeners:
            try:
                fn(result)
            except Exception as e:
                print(f"listener error: {e}")

    # ---- plane0 (Suite2p output folder) channel ----

    def subscribe_plane0(self, fn) -> None:
        self._plane0_listeners.append(fn)

    def set_plane0(self, plane0: Path) -> None:
        # Always normalise to a ``Path`` so subscribers can rely on the
        # type regardless of whether the producer passed a string.
        self.plane0 = Path(plane0)
        for fn in self._plane0_listeners:
            try:
                fn(self.plane0)
            except Exception as e:
                print(f"plane0 listener error: {e}")

    # ---- lowpass-ready channel ----

    def subscribe_lowpass_ready(self, fn) -> None:
        self._lowpass_listeners.append(fn)

    def set_lowpass_ready(self, plane0: Path) -> None:
        self.lowpass_plane0 = Path(plane0)
        for fn in self._lowpass_listeners:
            try:
                fn(self.lowpass_plane0)
            except Exception as e:
                print(f"lowpass listener error: {e}")

    # ---- event-detection results channel (Tab 5 -> Tabs 7 + 8) ----

    def subscribe_event_results(self, fn) -> None:
        """Subscribe to Tab 5 event-detection result publications.

        Listener receives a dict with keys: ``plane0``, ``event_windows``,
        ``A``, ``first_time``, ``kept_idx``, ``fps``, ``T``,
        ``onsets_by_roi``."""
        self._event_listeners.append(fn)

    def set_event_results(self, results: dict) -> None:
        self.event_results = results
        for fn in self._event_listeners:
            try:
                fn(results)
            except Exception as e:
                print(f"event_results listener error: {e}")

    # ---- clustering-ready channel (Tab 6 -> batch runner) ----

    def subscribe_clusters_ready(self, fn) -> None:
        """Subscribe to Tab 6's "clustering analysis finished" signal.

        Listener receives the plane0 path the clustering ran on so
        callers (batch runner) can move on to the next stage.
        """
        self._clusters_listeners.append(fn)

    def set_clusters_ready(self, plane0: Path) -> None:
        self.clusters_ready = Path(plane0)
        for fn in self._clusters_listeners:
            try:
                fn(self.clusters_ready)
            except Exception as e:
                print(f"clusters_ready listener error: {e}")

    # ---- crosscorrelation-ready channel (Tab 7 -> batch runner) ----

    def subscribe_xcorr_ready(self, fn) -> None:
        """Subscribe to Tab 7's "xcorr finished" signal.

        Listener receives the output directory Tab 7 wrote into.
        """
        self._xcorr_listeners.append(fn)

    def set_xcorr_ready(self, out_path: Path) -> None:
        self.xcorr_ready = Path(out_path)
        for fn in self._xcorr_listeners:
            try:
                fn(self.xcorr_ready)
            except Exception as e:
                print(f"xcorr_ready listener error: {e}")


# ---------------------------------------------------------------------------
# Advanced parameters dialog
# ---------------------------------------------------------------------------

class AdvancedDialog(tk.Toplevel):
    """Modal Toplevel that auto-builds an Entry/Combobox form from a
    list of parameter specs and writes accepted values into ``current``.

    A "spec list" is just a Python list of dicts, where each dict
    describes one parameter:

        [
            {"name": "z_enter", "label": "Hysteresis enter (z)",
             "type": "float", "default": 3.5,
             "group": "Per-ROI hysteresis",
             "help": "robust z threshold for entering an event"},
            ...
        ]

    Spec keys
    ---------
        name    : str              - the dict key in ``current``
        label   : str              - field label shown to the user
        type    : 'float' | 'int' | 'str' | 'bool' | 'choice'
        group   : str              - LabelFrame heading for grouping
        choices : list[str]        - required iff type == 'choice'
        help    : str              - optional one-line tooltip text

    Subclassing ``tk.Toplevel`` makes an instance behave like a modal
    pop-up window. The dialog mutates the caller's ``current`` dict in
    place when the user clicks OK; the caller checks the returned bool
    to know whether anything actually changed.
    """

    def __init__(self, master, title: str, specs: list[dict],
                 current: dict) -> None:
        # ``super().__init__(master)`` calls Toplevel's constructor and
        # gives us a real window. ``transient(master)`` ties it to the
        # parent so the OS treats it like a modal child window.
        super().__init__(master)
        self.title(title)
        self.transient(master)
        self.resizable(True, True)
        self.geometry("560x600")
        # Stash the inputs on the instance so the helper methods below
        # can read them. ``self._defaults`` is a dict comprehension --
        # the Python form of R's ``setNames(...)`` shortcut: build a
        # new dict by iterating over ``specs``.
        self._specs = specs
        self._current = current
        self._defaults = {s["name"]: s["default"] for s in specs}
        # Will be filled in ``_add_field``: maps spec name -> (Tk Var,
        # spec dict). We use a tuple so we don't have to look the spec
        # back up by name later.
        self._vars: dict = {}
        self._accepted = False

        self._build_ui()
        # ``grab_set()`` makes the dialog modal: pointer events are
        # captured here until it closes, so the user can't accidentally
        # click on a tab behind it.
        self.grab_set()
        # Catch the OS "X" close button so it cancels rather than just
        # destroying the window.
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

    def _build_ui(self) -> None:
        """Lay out the scrollable form, the per-group LabelFrames, and
        the OK / Cancel / Reset buttons. The actual fields are added by
        ``_add_field`` per spec."""
        # Outer frame holds the scroll canvas + scrollbar.
        outer = ttk.Frame(self); outer.pack(fill="both", expand=True)
        canvas = tk.Canvas(outer, highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # Tk's ``Canvas`` lets us host any widget by wrapping it in a
        # window item. The fields go inside ``inner``; the canvas
        # scrolls ``inner`` vertically.
        inner = ttk.Frame(canvas, padding=8)
        win = canvas.create_window((0, 0), window=inner, anchor="nw")

        # Resize handler: keep the inner frame's width matching the
        # canvas so the form fills horizontally, and update the
        # scrollable region to match its full height.
        def _resize(_e):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfigure(win, width=canvas.winfo_width())
        inner.bind("<Configure>", _resize)
        canvas.bind("<Configure>", _resize)

        # Group fields by their ``group`` key. ``groups`` maps group
        # name -> (LabelFrame, current row). We add a new LabelFrame
        # the first time each group is encountered.
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

        # Bottom button row: Reset / Cancel / OK.
        btn_row = ttk.Frame(self, padding=(8, 4)); btn_row.pack(fill="x")
        ttk.Button(btn_row, text="Reset to defaults",
                   command=self._reset_defaults).pack(side="left")
        ttk.Button(btn_row, text="Cancel",
                   command=self._on_cancel).pack(side="right")
        ttk.Button(btn_row, text="OK",
                   command=self._on_ok).pack(side="right", padx=(0, 6))

    def _add_field(self, parent, row, spec) -> None:
        """Create one row in the group's LabelFrame: label | widget |
        help text. The widget type depends on ``spec["type"]``."""
        name = spec["name"]
        label = spec.get("label", name)
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        # Pre-fill from the caller's ``current`` dict if a value is
        # already there, otherwise from the spec's default.
        cur = self._current.get(name, spec["default"])

        # Tk uses *Variable* objects (BooleanVar / StringVar / IntVar)
        # to wire form widgets to a stable Python-level handle. Setting
        # ``var.set(...)`` updates the widget; ``var.get()`` reads it.
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
            # ints, floats and free-form strings all use a plain Entry.
            # We store the value as a string and coerce in ``_coerce``.
            var = tk.StringVar(value=str(cur))
            ttk.Entry(parent, textvariable=var).grid(
                row=row, column=1, sticky="ew", pady=2)

        # Optional italic-grey help text shown to the right of the
        # input -- the cheapest possible tooltip.
        help_text = spec.get("help")
        if help_text:
            ttk.Label(parent, text=help_text, foreground="gray",
                      font=("", 8, "italic")).grid(
                row=row, column=2, sticky="w", padx=(8, 0), pady=2)

        # Save (var, spec) so the OK handler can read the value back
        # and coerce it to the right Python type.
        self._vars[name] = (var, spec)

    def _reset_defaults(self) -> None:
        """Snap every field back to the value declared in its spec."""
        # ``.items()`` iterates a dict as (key, value) pairs. The inner
        # tuple ``(var, spec)`` is destructured directly in the for
        # loop -- this is one of the niceties Python has over R.
        for name, (var, spec) in self._vars.items():
            d = self._defaults[name]
            if spec["type"] == "bool":
                var.set(bool(d))
            else:
                var.set(str(d))

    def _coerce(self, raw: str, spec: dict):
        """Turn the string sitting in a Tk var back into the right
        Python type for this spec. Raises ``ValueError`` if the user
        typed garbage in a numeric field."""
        t = spec["type"]
        if t == "bool":
            return bool(raw)
        if t == "int":
            # Accept things like "3.0" by going through float first.
            return int(float(raw))
        if t == "float":
            return float(raw)
        return raw

    def _on_ok(self) -> None:
        """OK handler: parse every field and, if all parse cleanly,
        copy the new values into the caller's ``current`` dict."""
        try:
            new_values = {}
            for name, (var, spec) in self._vars.items():
                new_values[name] = self._coerce(var.get(), spec)
        except (ValueError, TypeError) as e:
            # ``f"{name!r}"`` is f-string magic for ``repr(name)`` --
            # adds quotes around strings so the error message is
            # unambiguous about which field failed.
            messagebox.showerror("Invalid value",
                                 f"Could not parse {name!r}: {e}")
            return
        # ``dict.update(other)`` merges ``other`` into ``dict`` in place.
        self._current.update(new_values)
        self._accepted = True
        self.grab_release()
        self.destroy()

    def _on_cancel(self) -> None:
        self.grab_release()
        self.destroy()


def spec_defaults(specs: list[dict]) -> dict:
    """Build a {name: default} dict from a PARAM_SPEC list.

    Used by every tab that owns a PARAM_SPEC: ``self._params =
    spec_defaults(self.PARAM_SPEC)`` in ``__init__`` gives a fresh dict
    of defaults that the Advanced dialog can later mutate.
    """
    return {s["name"]: s["default"] for s in specs}


def open_advanced(master, title: str, specs: list[dict],
                  current: dict) -> bool:
    """Open AdvancedDialog modally; returns True if user clicked OK.

    Tabs treat the True return value as "the user changed something,
    rerender on the next button press". When False, ``current`` is
    untouched and there's nothing new to do.
    """
    dlg = AdvancedDialog(master, title, specs, current)
    # ``wait_window`` blocks the calling thread until the dialog is
    # destroyed. Tk re-enters its event loop while waiting so the
    # dialog stays responsive.
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

    The reverse direction (list -> compact string) is in
    ``format_roi_indices`` below.
    """
    if text is None:
        raise ValueError("empty input")
    # ``set`` is the unordered no-duplicates collection. We accumulate
    # ints into it so that ``"1,1,2"`` collapses to ``{1, 2}``.
    indices: set[int] = set()
    # Iterate over comma-separated tokens. ``split(",")`` returns a
    # plain list of substrings.
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            # ``"1-5"`` -> a (1) and b (5); allow reversed ranges by
            # swapping if needed.
            a_str, b_str = token.split("-", 1)
            try:
                a = int(a_str.strip())
                b = int(b_str.strip())
            except ValueError:
                raise ValueError(f"bad range token {token!r}")
            if a > b:
                a, b = b, a
            # ``range(a, b+1)`` is an iterator from a to b inclusive.
            indices.update(range(a, b + 1))
        else:
            try:
                indices.add(int(token))
            except ValueError:
                raise ValueError(f"bad token {token!r}")
    if not indices:
        raise ValueError("no indices parsed")
    out = sorted(indices)
    # List comprehension: "for every i in out, keep it if i is out of
    # range". Python's most idiomatic way to filter a list.
    bad = [i for i in out if i < 0 or i >= n_total]
    if bad:
        # Show the first few offenders to keep the error message short.
        raise ValueError(
            f"out of range [0, {n_total}): {bad[:5]}"
            + ("..." if len(bad) > 5 else ""))
    return out


def format_roi_indices(indices: list[int]) -> str:
    """Render a sorted list of ints as compact 'a-b' runs joined by commas
    (e.g. [1,2,3,7,10,11,12] -> '1-3,7,10-12').

    The Tab 6 "Copy ROIs" feature uses this so the clipboard contents
    can be pasted straight back into Tab 4's / Tab 5's manual-subset
    field.
    """
    if not indices:
        return ""
    runs: list[str] = []
    # Two-pointer walk: ``s`` is the start of the current run, ``p``
    # is the previous index seen.
    s = indices[0]
    p = s
    for i in indices[1:]:
        if i == p + 1:
            # Still extending the current run.
            p = i
        else:
            # Run broke; flush it and start a fresh run.
            runs.append(f"{s}-{p}" if s != p else f"{s}")
            s = i
            p = i
    # Don't forget the last run.
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
    # Lazy-import: only loaded when the toolbar is actually built. This
    # avoids a small import cost for tabs that never call this helper.
    from . import plot_data_export
    tb_frame = ttk.Frame(parent_frame)
    tb_frame.pack(side="top", fill="x")
    # ``pack_toolbar=False`` lets us pack it ourselves instead of
    # NavigationToolbar2Tk auto-packing into the parent.
    tb = NavigationToolbar2Tk(canvas, tb_frame, pack_toolbar=False)
    tb.update()
    tb.pack(side="left", fill="x")
    # ``lambda x=default: ...`` is Python's anonymous function (R: ``\(x)
    # ...`` or ``function(x) ...``). The ``c=canvas`` form captures the
    # current value of ``canvas`` at lambda creation time, dodging the
    # classic late-binding gotcha where loop variables would all
    # resolve to the last value.
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
    ('log', line) tuples.

    Tabs that run heavy work (e.g. Tab 3's Suite2p detection) launch a
    background thread for it. The library being called inside that
    thread happily ``print``s status updates -- but those would
    normally go to the terminal, not the GUI's log panel. Subclassing
    ``io.TextIOBase`` and overriding ``write`` lets us pretend to be a
    file: the worker temporarily swaps ``sys.stdout = QueueWriter(q)``,
    every print becomes a queue entry, and the main thread's polling
    loop drains the queue and writes each line into a Text widget.
    """

    def __init__(self, q: "queue.Queue") -> None:
        # ``q: "queue.Queue"`` is a string-form type hint; we quote it
        # so Python doesn't have to resolve ``queue.Queue`` at parse
        # time. (``from __future__ import annotations`` already does
        # this globally, but the explicit quotes make the intent
        # obvious here.)
        self.q = q
        # ``write`` may be called with a partial line (no trailing
        # newline). We buffer until we hit a newline, then ship a
        # complete line through the queue.
        self._buf = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buf += s
        # Flush every complete line. ``split("\n", 1)`` splits at most
        # once: it returns ``[before_first_newline, rest_of_string]``.
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self.q.put(("log", line))
        # ``write`` is supposed to return the number of characters
        # accepted; we always accept everything.
        return len(s)

    def flush(self) -> None:
        """Called by the worker (or the runtime) at shutdown so we
        don't drop a final partial line."""
        if self._buf:
            self.q.put(("log", self._buf))
            self._buf = ""
