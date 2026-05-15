"""gui_common.py - shared state object and Tk helpers used by every
pipeline_gui tab module.

Why this file exists
--------------------
The eight tabs of CalLIOPE need to know about each other -- e.g. when
Tab 1 finishes preprocessing, Tab 2 should pick up the result; when
Tab 5 detects events, Tabs 7 and 8 should redraw with those events.
Rather than wiring every tab to every other tab, all of them subscribe
to a single ``AppState`` object and that object broadcasts state
changes. Tabs publish via ``set_*`` methods and listen via
``subscribe_*`` methods; nobody calls into a sibling tab directly.

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

# Turns every type hint in this file into a string at parse time. Used
# here so the ``Optional[PreprocessResult]`` annotation on AppState
# stays cheap even if ``PreprocessResult`` import gets expensive later.
from __future__ import annotations

# Standard-library bits we'll need.
import io        # Provides the ``TextIOBase`` base class for QueueWriter.
import queue     # Thread-safe FIFO queue used by QueueWriter.
import tkinter as tk
from pathlib import Path  # Modern object-oriented filesystem path API.
from tkinter import filedialog, messagebox, ttk
from typing import Optional

# ``customtkinter`` ships its own theme manager. The palette helpers
# below pull whatever appearance is active so raw ``tk`` widgets
# (Text, Listbox, Canvas) -- which CTk does not skin -- can match.
import customtkinter as ctk

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
# Dark-palette helpers for raw tk widgets
# ---------------------------------------------------------------------------
#
# ``customtkinter`` only restyles widgets you create as ``CTk*``
# (CTkFrame, CTkButton, etc.). Raw ``tk`` widgets like ``tk.Text``,
# ``tk.Listbox``, and ``tk.Canvas`` keep their default system colors
# unless we hand-skin them. These two helpers grab the current CTk
# theme's frame colors and apply them to a tk widget so the log
# consoles and listboxes blend with the dark surround.

def _resolve_color(spec) -> str:
    """CTk theme entries are either a hex string or a 2-tuple
    ``(light_mode_color, dark_mode_color)``. Return the entry that
    matches the active appearance mode.
    """
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        return spec[1] if ctk.get_appearance_mode() == "Dark" else spec[0]
    return spec


def tk_dark_palette() -> dict:
    """Return a dict of colors keyed by role, sampled from the current
    CTk theme. Use the entries (``bg``, ``fg``, ``select_bg``,
    ``select_fg``, ``insert``) to skin raw tk widgets.
    """
    theme = ctk.ThemeManager.theme
    frame_bg = _resolve_color(theme["CTkFrame"]["fg_color"])
    text_fg = _resolve_color(theme["CTkLabel"]["text_color"])
    border = _resolve_color(theme["CTkFrame"]["border_color"])
    button_fg = _resolve_color(theme["CTkButton"]["fg_color"])
    return {
        "bg": frame_bg,
        "fg": text_fg,
        "select_bg": button_fg,
        "select_fg": "#ffffff",
        "insert": text_fg,
        "border": border,
        "trough": frame_bg,
    }


def apply_dark_to_tk_widget(widget) -> None:
    """Recolor a raw ``tk.Text`` / ``tk.Listbox`` / ``tk.Canvas`` so it
    matches the current CTk dark theme. Silently skips options the
    widget doesn't accept (a ``tk.Canvas`` has no ``insertbackground``
    etc.).
    """
    palette = tk_dark_palette()
    options = {
        "background": palette["bg"],
        "foreground": palette["fg"],
        "selectbackground": palette["select_bg"],
        "selectforeground": palette["select_fg"],
        "insertbackground": palette["insert"],
        "highlightthickness": 0,
        "borderwidth": 0,
        "relief": "flat",
    }
    for opt, val in options.items():
        try:
            widget.configure({opt: val})
        except tk.TclError:
            # Widget doesn't support this option; skip.
            pass


def attach_resize_handle(parent, target, *, min_height: int = 60,
                         initial_height: Optional[int] = None,
                         pad: tuple = (0, 4)):
    """Pack a thin draggable grip below ``target`` that lets the user
    resize ``target``'s height by click-and-drag.

    Unlike a ``PanedWindow`` sash, this grip does **not** redistribute
    space between two siblings -- it just grows / shrinks ``target``,
    and the surrounding scrollable wrapper handles overflow. Use it
    when only one panel in a stack should be resizable while the
    others stay at their natural height.

    ``parent`` is the container the grip is packed into (typically the
    same parent ``target`` was packed into, packed *after* ``target``).
    ``min_height`` clamps the drag floor so the panel can't be
    collapsed to invisibility.

    After each drag we walk up to the nearest ``CTkScrollableFrame``
    and bump its canvas-window height to match the new total content
    requirement -- without that step the wrapper holds its old fixed
    inner-frame height and the dragged panel just crushes its
    neighbours instead of extending the tab.
    """
    target.pack_propagate(False)
    target.update_idletasks()
    natural_h = target.winfo_reqheight()
    target.configure(height=initial_height
                     if initial_height is not None
                     else max(natural_h, min_height))

    palette = tk_dark_palette()
    handle = tk.Frame(parent, height=6, cursor="sb_v_double_arrow",
                      bg=palette["border"], highlightthickness=0)
    handle.pack(fill="x", pady=pad)

    state = {"y0": None, "h0": None}

    def _scrollable_ancestor(widget):
        w = widget.master
        while w is not None:
            if isinstance(w, ctk.CTkScrollableFrame):
                return w
            w = w.master
        return None

    def _bump_scrollable(sf):
        sf.update_idletasks()
        canvas = sf._parent_canvas
        window_id = sf._create_window_id
        canvas_h = canvas.winfo_height()
        natural = sf.winfo_reqheight()
        canvas.itemconfigure(window_id, height=max(natural, canvas_h))

    def _on_press(e):
        state["y0"] = e.y_root
        state["h0"] = target.winfo_height()

    def _on_drag(e):
        if state["y0"] is None:
            return
        dy = e.y_root - state["y0"]
        new_h = max(min_height, int(state["h0"] + dy))
        target.configure(height=new_h)
        sf = _scrollable_ancestor(target)
        if sf is not None:
            _bump_scrollable(sf)

    def _on_enter(_e):
        handle.configure(bg=palette["select_bg"])

    def _on_leave(_e):
        handle.configure(bg=palette["border"])

    handle.bind("<ButtonPress-1>", _on_press)
    handle.bind("<B1-Motion>", _on_drag)
    handle.bind("<Enter>", _on_enter)
    handle.bind("<Leave>", _on_leave)
    return handle


def install_scroll_router(
    toplevel,
    *,
    fallback_canvas=None,
    fallback_units_per_notch: int = 25,
    compact_canvas_units_per_notch: int = 3,
) -> None:
    """Route every ``<MouseWheel>`` event under ``toplevel`` to the
    widget the cursor is actually over.

    Dispatch priority for each wheel tick:

    1. Walk up from ``event.widget`` looking for the innermost
       scrollable inner widget the cursor sits inside -- a console
       (``CTkTextbox`` / ``tk.Text``), a listbox, a scrollable inner
       ``tk.Canvas``, or a nested ``CTkScrollableFrame``. If one is
       found, the wheel scrolls that widget. (For ``tk.Text`` /
       ``tk.Listbox`` / ``CTkTextbox`` the widget's class binding has
       already done the scroll by the time this handler runs -- the
       class tag fires before the toplevel tag in the bindtag chain
       -- so we just consume the event so the same notch doesn't
       trigger a second scroll downstream.)
    2. Otherwise -- and only when ``fallback_canvas`` is set -- the
       wheel scrolls the fallback. The tab wrapper passes its inner
       canvas here so the wheel falls through to scroll the whole tab
       when the cursor isn't over any inner panel. Popouts pass
       ``fallback_canvas=None`` since there's no outer scrollable to
       absorb the leftover.
    3. The handler always returns ``"break"`` so the global
       ``CTkScrollableFrame`` ``bind_all`` (on the ``"all"`` bindtag
       stage) is suppressed -- without that step the wheel would also
       scroll whichever ``CTkScrollableFrame`` was most recently
       constructed in the process, regardless of where the cursor is.

    The binding is installed on ``toplevel`` itself; because the
    toplevel's bindtag appears in every descendant's bindtag list
    (widget, class, toplevel, all), the handler fires for events
    anywhere inside the window -- no per-widget recursion needed.

    ``fallback_canvas`` may be a callable returning the canvas
    (used by the app root to point at whichever tab is currently
    visible).

    ``fallback_units_per_notch`` is the per-notch step for the
    fallback and for any nested ``CTkScrollableFrame`` -- both are
    "big region" scrolls that feel right when they move ~a fifth of
    a viewport per notch. ``compact_canvas_units_per_notch`` is the
    smaller step for raw ``tk.Canvas`` content (compact lists like
    Tab 0's recordings rows) where a big jump would skip past
    visible rows.
    """

    def _resolve_fallback():
        if fallback_canvas is None:
            return None
        if callable(fallback_canvas):
            try:
                return fallback_canvas()
            except Exception:
                return None
        return fallback_canvas

    def _is_canvas_scrollable(c) -> bool:
        # A ``tk.Canvas`` whose content fits in its viewport reports
        # ``yview() == (0.0, 1.0)`` -- nothing to scroll. Matplotlib's
        # ``FigureCanvasTkAgg`` widget is a tk.Canvas in that state,
        # so this check naturally keeps the wheel from doing anything
        # silly over a plot.
        try:
            return c.yview() != (0.0, 1.0)
        except tk.TclError:
            return False

    def _find_target(widget, stop_at):
        # Walk up from ``widget`` toward the toplevel looking for the
        # innermost widget that owns the wheel. ``stop_at`` (the
        # fallback canvas, when set) bounds the search so the wrapper
        # canvas is only reached via the fallback path, not as an
        # "inner" scrollable.
        w = widget
        while w is not None and w is not toplevel:
            if stop_at is not None and w is stop_at:
                return None
            # tk.Text / tk.Listbox each have a class binding for
            # <MouseWheel> that already scrolled the widget by the
            # time the toplevel-tag stage runs. Same goes for
            # CTkTextbox, which wraps a tk.Text. We flag those as
            # "self-handled" -- still consume the event so it doesn't
            # bubble to CTk's bind_all on the "all" stage.
            if isinstance(w, (tk.Text, tk.Listbox)):
                return ("self_handled", w)
            if isinstance(w, ctk.CTkTextbox):
                return ("self_handled", w)
            if isinstance(w, ctk.CTkScrollableFrame):
                return ("manual_big", w._parent_canvas)
            if isinstance(w, tk.Canvas) and _is_canvas_scrollable(w):
                return ("manual_compact", w)
            w = getattr(w, "master", None)
        return None

    def _do_scroll(canvas, delta, units):
        # Windows fires +/-120 per wheel notch; macOS / Linux fire
        # +/-1. Normalise to scroll-units before applying the
        # per-context multiplier (1 for inner scrollables, higher
        # for the tab fallback so long-distance scrolls feel snappy).
        if abs(delta) >= 120:
            n = -int(delta / 120)
        else:
            n = -1 if delta > 0 else 1
        try:
            canvas.yview_scroll(n * units, "units")
        except tk.TclError:
            pass

    def _handler(event):
        widget = event.widget
        if not isinstance(widget, tk.Misc):
            return "break"
        fallback = _resolve_fallback()
        match = _find_target(widget, stop_at=fallback)
        if match is not None:
            kind, target = match
            if kind == "manual_big":
                # CTkScrollableFrame's inner canvas -- always a big
                # scroll region (Advanced dialog form, future tab
                # subareas). Use the same multiplier as the fallback
                # so its feel matches the whole-tab scroll.
                _do_scroll(target, event.delta, fallback_units_per_notch)
            elif kind == "manual_compact":
                # Raw tk.Canvas with content -- typically a compact
                # list of widgets (the batch tab's recordings rows).
                # A smaller per-notch step keeps single-line lists
                # from blowing past the visible range.
                _do_scroll(target, event.delta,
                           compact_canvas_units_per_notch)
            # ``self_handled`` -> the widget already scrolled itself
            # via its class binding; nothing more to do here, just
            # consume the event below.
        elif fallback is not None and _is_canvas_scrollable(fallback):
            _do_scroll(fallback, event.delta, fallback_units_per_notch)
        return "break"

    toplevel.bind("<MouseWheel>", _handler, add="+")


def drain_queue(widget, q: "queue.Queue", handlers: dict,
                *, poll_ms: int = 100) -> None:
    """Pull every ``(kind, payload)`` tuple off ``q`` and dispatch each
    to ``handlers[kind]``, then re-arm via ``widget.after(poll_ms, ...)``.

    Typical use from a tab::

        drain_queue(self, self._log_queue,
                    {"log": self._append_log,
                     "done": self._on_done,
                     "error": self._on_error},
                    poll_ms=self.POLL_MS)

    The dispatch dict centralises the (kind -> handler) wiring so every
    tab doesn't reinvent a get_nowait/elif/elif loop. Unknown kinds are
    silently ignored so a handler dict can lag the producer without
    crashing. ``widget`` is any Tk widget that owns an ``.after`` method
    (the tab itself works).
    """

    def _tick() -> None:
        try:
            while True:
                kind, payload = q.get_nowait()
                handler = handlers.get(kind)
                if handler is not None:
                    handler(payload)
        except queue.Empty:
            pass
        widget.after(poll_ms, _tick)

    widget.after(poll_ms, _tick)


def apply_ttk_dark_theme(root) -> None:
    """Restyle every ``ttk`` widget class to blend with the CTk
    surround. Must be called once after the root window exists.

    ``ttk`` widgets (Frame, LabelFrame, Button, Entry, Scrollbar,
    Combobox, Checkbutton, Radiobutton, Progressbar, Separator,
    PanedWindow, Spinbox, Treeview) are not touched by customtkinter
    -- their default system theme renders bright on a dark root.
    Switching the ttk theme to ``clam`` and overriding the common
    widget classes here keeps the few ttk widgets we still rely on
    (mostly PanedWindow sashes and Spinbox arrows, where CTk has no
    equivalent) visually consistent with the CTk surround.
    """
    palette = tk_dark_palette()
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        # The clam theme is missing on some custom builds; fall back
        # to whichever default the user has.
        pass
    fg = palette["fg"]
    bg = palette["bg"]
    # Hover / pressed shade: CTk's typical button hover.
    hover = _resolve_color(
        ctk.ThemeManager.theme["CTkButton"]["hover_color"])
    accent = palette["select_bg"]

    style.configure(".", background=bg, foreground=fg, fieldbackground=bg,
                    bordercolor=palette["border"], lightcolor=bg,
                    darkcolor=bg, troughcolor=bg)
    style.configure("TFrame", background=bg)
    style.configure("TLabel", background=bg, foreground=fg)
    style.configure("TLabelframe", background=bg, foreground=fg,
                    bordercolor=palette["border"])
    style.configure("TLabelframe.Label", background=bg, foreground=fg)
    style.configure("TButton", background=accent, foreground="#ffffff",
                    bordercolor=accent, focusthickness=0, padding=4)
    style.map("TButton",
              background=[("active", hover), ("pressed", hover)])
    style.configure("TEntry", fieldbackground=bg, foreground=fg,
                    insertcolor=fg, bordercolor=palette["border"])
    style.configure("TCombobox", fieldbackground=bg, background=bg,
                    foreground=fg, arrowcolor=fg,
                    bordercolor=palette["border"])
    style.map("TCombobox",
              fieldbackground=[("readonly", bg)],
              foreground=[("readonly", fg)])
    style.configure("TCheckbutton", background=bg, foreground=fg,
                    indicatorbackground=bg, indicatorforeground=fg,
                    focuscolor=bg)
    style.map("TCheckbutton", background=[("active", bg)])
    style.configure("TRadiobutton", background=bg, foreground=fg,
                    indicatorbackground=bg, focuscolor=bg)
    style.map("TRadiobutton", background=[("active", bg)])
    style.configure("TProgressbar", background=accent, troughcolor=bg,
                    bordercolor=palette["border"])
    style.configure("TSeparator", background=palette["border"])
    style.configure("TPanedwindow", background=bg)
    style.configure("Sash", sashthickness=6, background=palette["border"])
    style.configure("Vertical.TScrollbar", background=bg, troughcolor=bg,
                    bordercolor=palette["border"], arrowcolor=fg)
    style.configure("Horizontal.TScrollbar", background=bg, troughcolor=bg,
                    bordercolor=palette["border"], arrowcolor=fg)
    style.map("Vertical.TScrollbar",
              background=[("active", hover), ("pressed", hover)])
    style.map("Horizontal.TScrollbar",
              background=[("active", hover), ("pressed", hover)])
    style.configure("TSpinbox", fieldbackground=bg, background=bg,
                    foreground=fg, arrowcolor=fg,
                    bordercolor=palette["border"])
    style.configure("TNotebook", background=bg, bordercolor=palette["border"])
    style.configure("TNotebook.Tab", background=bg, foreground=fg,
                    padding=(10, 4))
    style.map("TNotebook.Tab",
              background=[("selected", accent), ("active", hover)],
              foreground=[("selected", "#ffffff")])


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
    never call into each other directly -- this keeps the dependency
    graph between tabs flat and lets new tabs join the bus without
    every existing tab needing to know about them.
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

class AdvancedDialog(ctk.CTkToplevel):
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
        self.wm_minsize(440, 400)
        # Stash the inputs on the instance so the helper methods below
        # can read them. ``self._defaults`` is a dict comprehension --
        # one entry per spec, keyed by spec name -- so the Reset
        # button can snap fields back without re-walking ``specs``.
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
        # Route wheel events to the inner CTkScrollableFrame; no
        # fallback since the dialog itself isn't scrollable.
        install_scroll_router(self)

    def _build_ui(self) -> None:
        """Lay out the scrollable form, the per-group LabelFrames, and
        the OK / Cancel / Reset buttons. The actual fields are added by
        ``_add_field`` per spec.

        Uses ``CTkScrollableFrame`` for the form body so the dialog can
        host long PARAM_SPECs without forcing a tall fixed window.
        """
        inner = ctk.CTkScrollableFrame(self, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=8, pady=(8, 0))

        # Group fields by their ``group`` key. ``groups`` maps group
        # name -> (frame, current row). We add a new CTkFrame the first
        # time each group is encountered; the header label sits in row
        # 0 and fields stack below it.
        groups: dict = {}
        for spec in self._specs:
            grp = spec.get("group", "Parameters")
            if grp not in groups:
                lf = ctk.CTkFrame(inner)
                lf.pack(fill="x", pady=(0, 8))
                ctk.CTkLabel(lf, text=grp,
                             font=ctk.CTkFont(weight="bold")).grid(
                    row=0, column=0, columnspan=3, sticky="w",
                    padx=8, pady=(6, 4))
                lf.columnconfigure(1, weight=1)
                # Field rows start at index 1 so the header label at
                # row 0 stays above them.
                groups[grp] = (lf, 1)
            lf, row = groups[grp]
            self._add_field(lf, row, spec)
            groups[grp] = (lf, row + 1)

        # Bottom button row: Reset / Cancel / OK.
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=8, pady=6)
        ctk.CTkButton(btn_row, text="Reset to defaults", width=160,
                      command=self._reset_defaults).pack(side="left")
        ctk.CTkButton(btn_row, text="Cancel", width=100,
                      command=self._on_cancel).pack(side="right")
        ctk.CTkButton(btn_row, text="OK", width=100,
                      command=self._on_ok).pack(side="right", padx=(0, 6))

    def _add_field(self, parent, row, spec) -> None:
        """Create one row in the group's CTkFrame: label | widget |
        help text. The widget type depends on ``spec["type"]``."""
        name = spec["name"]
        label = spec.get("label", name)
        ctk.CTkLabel(parent, text=label, anchor="w").grid(
            row=row, column=0, sticky="w", padx=(8, 8), pady=2)
        # Pre-fill from the caller's ``current`` dict if a value is
        # already there, otherwise from the spec's default.
        cur = self._current.get(name, spec["default"])

        # Tk uses *Variable* objects (BooleanVar / StringVar / IntVar)
        # to wire form widgets to a stable Python-level handle. Setting
        # ``var.set(...)`` updates the widget; ``var.get()`` reads it.
        if spec["type"] == "bool":
            var = tk.BooleanVar(value=bool(cur))
            ctk.CTkCheckBox(parent, text="", variable=var).grid(
                row=row, column=1, sticky="w", pady=2)
        elif spec["type"] == "choice":
            var = tk.StringVar(value=str(cur))
            cb = ctk.CTkComboBox(parent, variable=var, state="readonly",
                                 values=list(spec.get("choices", [])))
            cb.grid(row=row, column=1, sticky="ew", pady=2)
        else:
            # ints, floats and free-form strings all use a plain Entry.
            # We store the value as a string and coerce in ``_coerce``.
            var = tk.StringVar(value=str(cur))
            ctk.CTkEntry(parent, textvariable=var).grid(
                row=row, column=1, sticky="ew", pady=2)

        # Optional italic-grey help text shown to the right of the
        # input -- the cheapest possible tooltip.
        help_text = spec.get("help")
        if help_text:
            ctk.CTkLabel(parent, text=help_text, text_color="gray",
                         font=ctk.CTkFont(size=11, slant="italic"),
                         anchor="w").grid(
                row=row, column=2, sticky="w", padx=(8, 8), pady=2)

        # Save (var, spec) so the OK handler can read the value back
        # and coerce it to the right Python type.
        self._vars[name] = (var, spec)

    def _reset_defaults(self) -> None:
        """Snap every field back to the value declared in its spec."""
        # ``.items()`` iterates a dict as (key, value) pairs; the
        # inner tuple ``(var, spec)`` is unpacked directly in the
        # for-loop header so we get both back without a second lookup.
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


def pick_tiffs_dialog(parent: tk.Misc, *, title: str,
                      initial_dir: str = "",
                      current_paths: Optional[list[str]] = None,
                      initial_depth: int = 0) -> Optional[list[str]]:
    """Modal Listbox-based TIFF picker (the same dialog Tab 1 and the
    batch row "Browse..." buttons open).

    Shows the TIFFs under ``initial_dir`` (recursing to
    ``initial_depth`` sub-levels) in a multi-select Listbox. Returns
    the list of absolute paths the user confirmed, or ``None`` if the
    dialog was cancelled.

    The user can change the search root + depth and hit "Refresh" to
    re-populate the list. Items already in ``current_paths`` are
    pre-selected.
    """
    from .core import preprocessing

    win = ctk.CTkToplevel(parent)
    win.title(title)
    win.transient(parent.winfo_toplevel())
    win.grab_set()
    win.geometry("700x520")
    win.wm_minsize(560, 380)

    body = ctk.CTkFrame(win, fg_color="transparent")
    body.pack(fill="both", expand=True, padx=8, pady=8)

    row = ctk.CTkFrame(body, fg_color="transparent")
    row.pack(fill="x", pady=(0, 6))
    ctk.CTkLabel(row, text="Folder:", width=70, anchor="w").pack(side="left")
    dir_var = tk.StringVar(value=initial_dir or "")
    ctk.CTkEntry(row, textvariable=dir_var).pack(
        side="left", fill="x", expand=True, padx=(0, 4))
    ctk.CTkButton(
        row, text="Browse...", width=90,
        command=lambda: dir_var.set(
            filedialog.askdirectory(title="Pick search folder",
                                    parent=win) or dir_var.get())
    ).pack(side="left")

    row = ctk.CTkFrame(body, fg_color="transparent")
    row.pack(fill="x", pady=(0, 6))
    ctk.CTkLabel(row, text="Depth:", width=70, anchor="w").pack(side="left")
    depth_var = tk.IntVar(value=max(0, int(initial_depth)))
    # ttk.Spinbox here because customtkinter has no spinbox widget;
    # gets themed via apply_ttk_dark_theme.
    ttk.Spinbox(row, from_=0, to=6, width=4,
                textvariable=depth_var).pack(side="left")
    refresh_btn = ctk.CTkButton(row, text="Refresh", width=90)
    refresh_btn.pack(side="left", padx=(8, 0))
    status_var = tk.StringVar(value="")
    ctk.CTkLabel(row, textvariable=status_var,
                 font=ctk.CTkFont(size=11, slant="italic")).pack(
        side="left", padx=(8, 0))

    list_holder = ctk.CTkFrame(body, fg_color="transparent")
    list_holder.pack(fill="both", expand=True)
    list_holder.columnconfigure(0, weight=1)
    list_holder.rowconfigure(0, weight=1)
    listbox = tk.Listbox(list_holder, selectmode="extended",
                         exportselection=False)
    listbox.grid(row=0, column=0, sticky="nsew")
    apply_dark_to_tk_widget(listbox)
    sb = ctk.CTkScrollbar(list_holder, orientation="vertical",
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

    foot = ctk.CTkFrame(body, fg_color="transparent")
    foot.pack(fill="x", pady=(8, 0))
    result: dict = {"paths": None}

    def _confirm():
        sel = listbox.curselection()
        result["paths"] = [paths_state[i] for i in sel]
        win.destroy()

    def _cancel():
        result["paths"] = None
        win.destroy()

    ctk.CTkButton(foot, text="Cancel", width=90,
                  command=_cancel).pack(side="right")
    ctk.CTkButton(foot, text="Confirm", width=100,
                  command=_confirm).pack(side="right", padx=(0, 6))

    # Route wheel events: cursor over the Listbox scrolls it, cursor
    # elsewhere in the dialog does nothing (no outer scrollable).
    install_scroll_router(win)

    _refresh()
    parent.wait_window(win)
    return result["paths"]


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
    tb_frame = ctk.CTkFrame(parent_frame, fg_color="transparent")
    tb_frame.pack(side="top", fill="x")
    # ``pack_toolbar=False`` lets us pack it ourselves instead of
    # NavigationToolbar2Tk auto-packing into the parent.
    tb = NavigationToolbar2Tk(canvas, tb_frame, pack_toolbar=False)
    tb.update()
    tb.pack(side="left", fill="x")
    # matplotlib's toolbar ships as a raw ``tk.Frame`` with system-grey
    # ``tk.Button``s and a coordinate ``tk.Label`` -- bright spots on
    # the dark surround. Recolour every child to the dark palette here.
    restyle_matplotlib_toolbar(tb)
    # The ``c=canvas`` / ``p=tb_frame`` / ``b=data_basename`` default
    # arguments on each lambda capture the *current* values at lambda
    # creation time. Without them the closures would all share the
    # same names from the enclosing scope, and a second call to
    # ``attach_fig_toolbar`` would silently rebind them.
    if data_provider is None:
        cmd = lambda c=canvas, p=tb_frame, b=data_basename: \
            plot_data_export.save_figure_data(c.figure, p, b)
    else:
        cmd = lambda p=tb_frame: data_provider(p)
    ctk.CTkButton(
        tb_frame, text="Save data...", command=cmd, width=110,
    ).pack(side="left", padx=(8, 0))
    return tb


def restyle_matplotlib_toolbar(tb) -> None:
    """Recolour every child of a matplotlib ``NavigationToolbar2Tk`` to
    blend with the dark theme. The toolbar's internal buttons are raw
    ``tk.Button`` widgets and its message Label is a ``tk.Label``; both
    keep system colours unless we touch them.
    """
    palette = tk_dark_palette()
    try:
        tb.configure(background=palette["bg"])
    except tk.TclError:
        pass
    hover = _resolve_color(
        ctk.ThemeManager.theme["CTkButton"]["hover_color"])
    for child in tb.winfo_children():
        try:
            cls = child.winfo_class()
        except tk.TclError:
            continue
        if cls == "Button":
            try:
                child.configure(
                    background=palette["bg"], foreground=palette["fg"],
                    activebackground=hover, activeforeground=palette["fg"],
                    relief="flat", borderwidth=0, highlightthickness=0,
                )
            except tk.TclError:
                pass
        elif cls in ("Label", "Frame"):
            try:
                child.configure(background=palette["bg"],
                                foreground=palette["fg"])
            except tk.TclError:
                # tk.Frame has no ``foreground``.
                try:
                    child.configure(background=palette["bg"])
                except tk.TclError:
                    pass
        elif cls == "Checkbutton":
            try:
                child.configure(background=palette["bg"],
                                foreground=palette["fg"],
                                activebackground=hover,
                                selectcolor=palette["bg"],
                                highlightthickness=0)
            except tk.TclError:
                pass


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
