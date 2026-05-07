"""calliope.pipeline_gui - top-level CalLIOPE GUI for the calcium imaging
pipeline.

This is the *application root*: the file that creates the main window
and stitches all eight tabs into a single Tk notebook. ``main()`` at
the bottom is the entrypoint -- it is what ``python -m calliope`` ends
up calling.

If you have an R background, the closest equivalent is the top-level
script that launches a ``shiny::shinyApp(ui, server)``: the UI scaffold
plus the wiring that hands a shared state object to each tab.

Tabs (each lives in its own subfolder under ``calliope.tabs``):
    1. Input & Preprocess     -> calliope.tabs.preprocess.tab.PreprocessTab
    2. QC Preview             -> calliope.tabs.qc.tab.QcTab
    3. Suite2p Detection      -> calliope.tabs.suite2p.tab.Suite2pTab
    4. Low-pass filter        -> calliope.tabs.lowpass.tab.LowpassTab
    5. Event detection        -> calliope.tabs.event_detection.tab.EventDetectionTab
    6. Clustering             -> calliope.tabs.clustering.tab.ClusteringTab
    7. Cross-correlation      -> calliope.tabs.crosscorrelation.tab.CrossCorrelationTab
    8. Spatial propagation    -> calliope.tabs.spatial_propagation.tab.SpatialPropagationTab

Shared state (``AppState``) and Tk helpers (advanced-params dialog,
matplotlib toolbar wrapper, manual-ROI parsing, queue-backed log writer)
live in ``calliope.gui_common``. Each tab is wrapped by
``_make_scrollable_tab`` so its contents stay reachable on smaller windows.
"""

# ``from __future__ import annotations`` switches every type hint in
# this file (e.g. ``Optional[ImageTk.PhotoImage]``) into a string at
# parse time. That has two practical benefits: (1) you can reference
# names that are defined later in the file without import gymnastics,
# and (2) very large class hierarchies do not pay an import cost just
# to evaluate annotations. R has no equivalent -- type hints are
# entirely opt-in in Python and are usually only consulted by editors
# / linters, not at runtime.
from __future__ import annotations

# Standard-library imports come first by convention. ``sys`` exposes
# things like ``sys.platform`` (used below to detect Windows).
import sys

# Tkinter is the GUI toolkit shipped with Python's stdlib. ``tk`` is
# the legacy widget set; ``ttk`` is the modern themed-widgets layer
# that gives buttons / frames / notebooks a native look. Convention is
# to import ``tk`` for primitives like ``Tk`` (the root window) and
# ``ttk`` for everything user-facing.
import tkinter as tk
from tkinter import ttk

# ``Optional[X]`` is shorthand for "X or None". Equivalent to writing
# ``X | None`` in Python 3.10+. Hints only -- they don't affect
# runtime behaviour.
from typing import Optional

# Pillow (PIL) is the image-processing library we use to render the
# headphones emoji into a PhotoImage Tk can show as the window icon.
from PIL import Image, ImageDraw, ImageFont, ImageTk

# Single dot in front of a name = "import from a sibling module in
# this same package". Two dots = parent package, three dots = parent's
# parent, etc. ``AppState`` is the shared event-bus / state container
# every tab subscribes to.
from .gui_common import AppState

# Each tab class lives in its own sub-package. We import them all up
# front so that PipelineApp.__init__ can wire them into the notebook.
from .tabs.preprocess.tab import PreprocessTab
from .tabs.qc.tab import QcTab
from .tabs.suite2p.tab import Suite2pTab
from .tabs.lowpass.tab import LowpassTab
from .tabs.event_detection.tab import EventDetectionTab
from .tabs.clustering.tab import ClusteringTab
from .tabs.crosscorrelation.tab import CrossCorrelationTab
from .tabs.spatial_propagation.tab import SpatialPropagationTab


# ---------------------------------------------------------------------------
# Window icon + scrollable tab wrapper
# ---------------------------------------------------------------------------

def _render_emoji_icon(emoji: str, size: int = 128) -> Optional[ImageTk.PhotoImage]:
    """Render ``emoji`` to an RGBA PhotoImage for use with ``Tk.iconphoto``.

    Uses Segoe UI Emoji on Windows (color bitmap glyphs, requires
    Pillow >= 9 for ``embedded_color=True``). Returns None on failure.

    Notation note: ``-> Optional[ImageTk.PhotoImage]`` is a return-type
    hint. The leading underscore on the function name is just a Python
    convention meaning "private to this module"; it doesn't enforce
    anything, but lints and IDEs treat it as a soft "do not import this
    elsewhere".
    """
    # ``try / except`` is Python's exception handler. Anything inside
    # ``try`` that raises an error gets caught by the matching
    # ``except`` block instead of crashing the program. Equivalent to R's
    # ``tryCatch``.
    try:
        # Build a transparent square at the requested size. RGBA ==
        # red/green/blue/alpha; the (0,0,0,0) is fully-transparent
        # black.
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Try a few platform-specific colour-emoji fonts in priority
        # order. The first one that loads wins. ``ImageFont.truetype``
        # raises if the font file isn't on the system.
        font = None
        for candidate in ("seguiemj.ttf", "AppleColorEmoji.ttf",
                          "NotoColorEmoji.ttf"):
            try:
                font = ImageFont.truetype(candidate, int(size * 0.8))
                break
            except Exception:
                continue
        # Fallback: Pillow's built-in monochrome bitmap font. Better
        # than crashing; the user just sees a black-and-white glyph.
        if font is None:
            font = ImageFont.load_default()

        # Newer Pillow accepts ``embedded_color=True`` for COLR/CPAL
        # colour glyphs. Older versions raise ``TypeError`` on that
        # keyword -- in that case fall back to a monochrome draw.
        try:
            draw.text((size // 2, size // 2), emoji, font=font,
                      anchor="mm", embedded_color=True)
        except TypeError:
            draw.text((size // 2, size // 2), emoji, font=font,
                      anchor="mm", fill=(0, 0, 0, 255))

        # Wrap the Pillow image in a Tk-compatible PhotoImage so that
        # ``Tk.iconphoto`` can use it.
        return ImageTk.PhotoImage(img)
    except Exception as e:
        # f-string: an inline string template. ``f"...{x}..."`` evaluates
        # ``x`` and pastes it into the string. R equivalent is
        # ``sprintf`` or ``glue::glue``.
        print(f"icon render failed: {e}")
        return None


def _make_scrollable_tab(nb, tab_class, *args, **kwargs):
    """Wrap a tab class instance in a vertically-scrollable canvas.

    Returns (container, tab). Add ``container`` to the notebook; ``tab`` is
    the original tab instance and exposes its full API unchanged. The tab
    is sized to the canvas width (so ``pack(fill='x')`` still works) and
    its natural height drives the scroll region.

    ``*args`` and ``**kwargs`` are Python's catch-all argument forms:
    they collect *every* extra positional / keyword argument into a tuple
    / dict and pass them straight through to ``tab_class(...)``. Equivalent
    to R's ``...``. Here it lets us forward ``state_obj`` through without
    naming it explicitly.
    """
    # Frame that holds the scroll canvas + vertical scrollbar.
    container = ttk.Frame(nb)
    canvas = tk.Canvas(container, highlightthickness=0, borderwidth=0)
    vsb = ttk.Scrollbar(container, orient="vertical",
                        command=canvas.yview)
    # Tell the canvas to update the scrollbar position whenever it
    # scrolls.
    canvas.configure(yscrollcommand=vsb.set)
    # Tk's "grid" is a 2D layout manager -- think CSS grid. ``sticky``
    # = which edges the widget should stretch to (n/s/e/w).
    canvas.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    # Make the canvas grow when the container is resized.
    container.rowconfigure(0, weight=1)
    container.columnconfigure(0, weight=1)

    # Instantiate the tab with the canvas as its parent widget. The
    # tab subclasses ``ttk.Frame`` so it can be added straight into a
    # canvas window.
    tab = tab_class(canvas, *args, **kwargs)
    win_id = canvas.create_window((0, 0), window=tab, anchor="nw")

    # The two helper closures below (``_on_tab_config`` /
    # ``_on_canvas_config``) keep the tab's apparent height and width
    # in sync with the canvas's scroll region as the user resizes the
    # window. Closures inherit the surrounding ``canvas`` / ``win_id``
    # variables, much like R's lexical scoping.
    def _on_tab_config(_e):
        canvas.configure(scrollregion=canvas.bbox("all"))
    tab.bind("<Configure>", _on_tab_config)

    def _on_canvas_config(event):
        canvas.itemconfigure(win_id, width=event.width)
    canvas.bind("<Configure>", _on_canvas_config)

    # Mouse-wheel handling: Tk fires <MouseWheel> with ``event.delta``
    # in units of 120 on Windows (one notch) or +/-1 on other
    # platforms. We normalise to one scroll step.
    def _on_wheel(event):
        if abs(event.delta) >= 120:
            delta = -int(event.delta / 120)
        else:
            delta = -1 if event.delta > 0 else 1
        canvas.yview_scroll(delta, "units")

    # Bind mouse-wheel scrolling globally only while the cursor is
    # inside this container, so other widgets don't fight over the
    # wheel event.
    def _on_enter(_e):
        canvas.bind_all("<MouseWheel>", _on_wheel)

    def _on_leave(_e):
        canvas.unbind_all("<MouseWheel>")

    container.bind("<Enter>", _on_enter)
    container.bind("<Leave>", _on_leave)

    # A function can return more than one value as a tuple -- here we
    # ship back both the outer ``container`` (added to the notebook)
    # and the inner ``tab`` instance (so the caller can poke at its
    # methods).
    return container, tab


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

class PipelineApp(tk.Tk):
    """The root window itself.

    ``class Foo(Bar):`` means "Foo subclasses Bar" -- it inherits all
    of Bar's behaviour and can override or add bits. Here we extend
    ``tk.Tk`` (the top-level window) so an instance *is* a window.
    Nothing fancier than that. R8s S4 / R6 reference classes have a
    similar feel.
    """

    # ``APP_NAME`` etc. are *class attributes*: shared by every
    # instance. They behave like constants. (Lowercase names defined
    # in ``__init__`` would be *instance* attributes, set per object.)
    APP_NAME = "CalLIOPE"
    APP_SUBTITLE = "Calcium Live-imaging Output Pipeline for Epileptiform-recordings"
    APP_EMOJI = "\U0001F3A7"  # headphones unicode codepoint

    # Windows-specific app-id used so the taskbar groups our windows
    # under their own icon instead of bundling them into python.exe.
    APP_USER_MODEL_ID = "calliope.pipeline.1"

    def __init__(self) -> None:
        """Constructor. Runs once when ``PipelineApp()`` is called.

        ``self`` is the implicit first argument every method receives
        and refers to "this instance". Equivalent to R6's ``self`` or
        S4's first dispatch slot.
        """
        # Windows groups taskbar icons by AppUserModelID; without this, the
        # process inherits python.exe's AppID and the taskbar shows the
        # Python icon regardless of iconphoto. Must run before any Tk window
        # is created.
        if sys.platform == "win32":
            try:
                # ``ctypes`` lets Python call C functions in Windows
                # DLLs. We lazy-import it inside the if-block so non-
                # Windows installs never have to touch it.
                import ctypes
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                    self.APP_USER_MODEL_ID)
            except Exception as e:
                print(f"SetCurrentProcessExplicitAppUserModelID failed: {e}")

        # ``super().__init__()`` calls the parent class's
        # constructor -- here that means actually creating the Tk
        # window. Roughly the equivalent of ``callNextMethod()`` in S4.
        super().__init__()
        self.title(f"{self.APP_EMOJI} {self.APP_NAME} - {self.APP_SUBTITLE}")
        self.geometry("1100x720")

        # Keep a reference to the icon PhotoImage so the garbage
        # collector doesn't reclaim it while Tk still needs it.
        self._icon_photo = _render_emoji_icon(self.APP_EMOJI, size=128)
        if self._icon_photo is not None:
            try:
                # ``True`` => set the default icon for *every* new Tk
                # window in this process, not just this root window.
                self.iconphoto(True, self._icon_photo)
            except Exception as e:
                print(f"iconphoto failed: {e}")

        # The shared state object every tab will subscribe to. Tabs
        # never talk to each other directly; they read / publish via
        # ``state_obj``. See ``calliope.gui_common.AppState`` for the
        # full publish/subscribe surface.
        self.state_obj = AppState()

        # ``ttk.Notebook`` is the tabbed-pane widget. ``pack(...)``
        # places it inside ``self`` (the window) and tells it to fill
        # the available space.
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        # Local helper closure: build a scrollable tab and register it
        # in the notebook with a friendly label. Captures ``nb`` and
        # ``self.state_obj`` from the enclosing scope.
        def add(tab_class, text):
            container, tab = _make_scrollable_tab(
                nb, tab_class, self.state_obj)
            nb.add(container, text=text)
            return tab

        # Build all eight tabs. Each ``add(...)`` call returns the tab
        # *instance*, which we keep on ``self`` so that other code (or
        # tests) can poke at it later.
        self.preprocess_tab = add(PreprocessTab, "1. Input & Preprocess")
        self.qc_tab = add(QcTab, "2. QC Preview")
        self.detection_tab = add(Suite2pTab, "3. Suite2p Detection")
        self.lowpass_tab = add(LowpassTab, "4. Low-pass filter")
        self.event_tab = add(EventDetectionTab, "5. Event detection")
        self.clustering_tab = add(ClusteringTab, "6. Clustering")
        self.xcorr_tab = add(CrossCorrelationTab, "7. Cross-correlation")
        self.spatial_tab = add(SpatialPropagationTab, "8. Spatial propagation")

        # Dispatch ``on_tab_shown`` / ``on_tab_hidden`` hooks so individual
        # tabs can free heavy resources (e.g. the QC tab's PhotoImage frame
        # buffer) when the user switches away. Tabs without these methods
        # are simply skipped.
        self._tabs_in_order = [
            self.preprocess_tab, self.qc_tab, self.detection_tab,
            self.lowpass_tab, self.event_tab, self.clustering_tab,
            self.xcorr_tab, self.spatial_tab,
        ]
        self._nb = nb
        self._current_tab_index = nb.index("current")
        # Tk events are bound by name with ``bind``. The handler will
        # be called with an Event object that we choose to ignore.
        nb.bind("<<NotebookTabChanged>>", self._on_notebook_tab_changed)

    def _on_notebook_tab_changed(self, _event) -> None:
        """Notebook callback: dispatch ``on_tab_hidden`` on the tab the
        user is leaving and ``on_tab_shown`` on the tab they are
        landing on, *if* either method exists. ``getattr(obj, name,
        None)`` returns the attribute or None if the attribute is
        missing -- a graceful way to call optional hooks.
        """
        try:
            new_index = self._nb.index("current")
        except tk.TclError:
            # Window is being destroyed; ignore.
            return
        if new_index == self._current_tab_index:
            return
        old = self._tabs_in_order[self._current_tab_index]
        new = self._tabs_in_order[new_index]
        on_hidden = getattr(old, "on_tab_hidden", None)
        if callable(on_hidden):
            try:
                on_hidden()
            except Exception as e:
                print(f"on_tab_hidden({type(old).__name__}) failed: {e}")
        on_shown = getattr(new, "on_tab_shown", None)
        if callable(on_shown):
            try:
                on_shown()
            except Exception as e:
                print(f"on_tab_shown({type(new).__name__}) failed: {e}")
        self._current_tab_index = new_index


def main() -> None:
    """Entrypoint: instantiate the app and start the event loop.

    ``mainloop()`` blocks until the user closes the window. Until
    then, Tk dispatches mouse / keyboard / timer events to whichever
    widget callbacks were registered above. R's analogue is
    ``shiny::runApp()``.
    """
    PipelineApp().mainloop()


# Safety belt: if someone runs this file directly (``python
# pipeline_gui.py``) instead of ``python -m calliope``, this still
# launches the GUI rather than doing nothing.
if __name__ == "__main__":
    main()
