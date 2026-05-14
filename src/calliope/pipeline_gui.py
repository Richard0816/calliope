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

# ``customtkinter`` is a third-party widget set that re-skins Tk with a
# modern look and built-in dark mode. ``ctk.CTk`` is its themed root
# window (still inherits ``tk.Tk`` under the hood, so ``iconphoto``,
# ``geometry``, ``mainloop``, etc. still work).
import customtkinter as ctk

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
from .gui_common import AppState, apply_ttk_dark_theme

# Each tab class lives in its own sub-package. We import them all up
# front so that PipelineApp.__init__ can wire them into the notebook.
from .tabs.preprocess.tab import PreprocessTab
from .tabs.qc.tab import QcTab
from .tabs.batch.tab import BatchTab
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


def _make_scrollable_tab(parent, tab_class, *args, **kwargs):
    """Wrap a tab class instance in a scrollable CTk container.

    Returns ``(container, tab)``. Mount ``container`` in the content
    host; ``tab`` is the original tab instance and exposes its full
    API unchanged.

    Stock ``CTkScrollableFrame`` only stretches its inner frame
    horizontally (width-to-canvas). When the canvas is taller than the
    content's natural height the inner frame stays at natural height,
    leaving a gap below and starving any ``ttk.PanedWindow`` inside
    that wants to fill the tab. We patch that here: bind the canvas's
    ``<Configure>`` event to also stretch the inner frame's *height* to
    ``max(natural, canvas_height)``. Result: PanedWindows fill the
    viewport when there's room, and scrolling kicks in only when the
    content's natural height exceeds the viewport.
    """
    container = ctk.CTkScrollableFrame(parent, fg_color="transparent")
    tab = tab_class(container, *args, **kwargs)
    tab.pack(fill="both", expand=True)

    canvas = container._parent_canvas
    window_id = container._create_window_id

    def _stretch_inner_to_canvas(_event=None):
        # Fire from canvas resize AND from inner-frame resize (e.g. a
        # ``attach_resize_handle`` drag growing one child) so the canvas
        # window grows to absorb the new requested height. Without the
        # second binding the canvas keeps its old fixed window height
        # and the extra child just crushes its siblings.
        canvas_h = canvas.winfo_height()
        natural_h = container.winfo_reqheight()
        canvas.itemconfigure(window_id, height=max(natural_h, canvas_h))

    canvas.bind("<Configure>", _stretch_inner_to_canvas, add="+")
    container.bind("<Configure>", _stretch_inner_to_canvas, add="+")

    # Catch-all wheel handler. ``CTkScrollableFrame`` already does
    # ``bind_all("<MouseWheel>")`` but the path is fragile: some widget
    # class bindings (Listbox, Text) can interfere, and raw ``tk.Frame``
    # regions (resize-handle grips, gaps between panels) sometimes
    # don't trigger CTk's master-walk check. Binding the wheel on every
    # existing descendant of the tab guarantees scrolling works
    # wherever the cursor is, without forcing the user to aim at the
    # scrollbar.
    def _wheel_scroll(event):
        if canvas.yview() == (0.0, 1.0):
            return
        if abs(event.delta) >= 120:
            delta = -int(event.delta / 120)
        else:
            delta = -1 if event.delta > 0 else 1
        canvas.yview_scroll(delta, "units")

    def _bind_wheel_recursive(widget):
        # ``add="+"`` so we don't clobber existing bindings (the
        # Listbox / Text class bindings that scroll those widgets'
        # internal contents fire alongside this one).
        try:
            widget.bind("<MouseWheel>", _wheel_scroll, add="+")
        except tk.TclError:
            pass
        for child in widget.winfo_children():
            _bind_wheel_recursive(child)

    tab.after_idle(lambda: _bind_wheel_recursive(tab))

    return container, tab


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

class _Sidebar(ctk.CTkFrame):
    """Vertical navigation rail of CTkButtons -- one per tab.

    Replaces the old ``ttk.Notebook`` tab bar. The active tab's button is
    highlighted via ``fg_color``; inactive buttons render flat against
    the sidebar background. Selection callback receives the integer
    index of the clicked tab.
    """

    def __init__(self, master, labels, on_select, width=210):
        super().__init__(master, width=width, corner_radius=0)
        self._on_select = on_select
        # Capture the theme's accent color so the active state matches
        # whatever CTk theme is loaded. ``ThemeManager.theme`` is a
        # nested dict keyed by widget class.
        self._active_color = ctk.ThemeManager.theme["CTkButton"]["fg_color"]
        self._buttons: list[ctk.CTkButton] = []

        title = ctk.CTkLabel(
            self, text="CalLIOPE",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        title.pack(pady=(16, 4), padx=12)
        subtitle = ctk.CTkLabel(
            self, text="pipeline", text_color=("gray60", "gray60"),
            font=ctk.CTkFont(size=11),
        )
        subtitle.pack(pady=(0, 14), padx=12)

        for idx, text in enumerate(labels):
            btn = ctk.CTkButton(
                self, text=text, anchor="w", height=34,
                corner_radius=6, fg_color="transparent",
                text_color=("gray10", "gray90"),
                hover_color=("gray80", "gray25"),
                command=lambda i=idx: self._on_select(i),
            )
            btn.pack(fill="x", padx=8, pady=2)
            self._buttons.append(btn)

        # Keep the sidebar from shrinking below its requested width when
        # a tab body asks for more space.
        self.pack_propagate(False)
        self.grid_propagate(False)

    def set_active(self, idx: int) -> None:
        """Highlight the active tab's button; flatten the rest."""
        for i, btn in enumerate(self._buttons):
            if i == idx:
                btn.configure(fg_color=self._active_color)
            else:
                btn.configure(fg_color="transparent")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

class PipelineApp(ctk.CTk):
    """The root window itself.

    ``class Foo(Bar):`` means "Foo subclasses Bar" -- it inherits all
    of Bar's behaviour and can override or add bits. Here we extend
    ``ctk.CTk`` (customtkinter's themed root window, which itself
    inherits from ``tk.Tk``) so an instance *is* a window with dark
    mode baked in. R's S4 / R6 reference classes have a similar feel.
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
        # constructor -- here that means actually creating the CTk
        # window. Roughly the equivalent of ``callNextMethod()`` in S4.
        super().__init__()
        self.title(f"{self.APP_EMOJI} {self.APP_NAME} - {self.APP_SUBTITLE}")
        self.geometry("1280x800")
        self.minsize(960, 640)

        # Keep a reference to the icon PhotoImage so the garbage
        # collector doesn't reclaim it while Tk still needs it.
        self._icon_photo = _render_emoji_icon(self.APP_EMOJI, size=128)
        if self._icon_photo is not None:
            try:
                # ``True`` => set the default icon for *every* new Tk
                # window in this process, not just this root window.
                self.iconphoto(True, self._icon_photo)
                # ``customtkinter.CTk`` checks a private flag and, if
                # ``iconbitmap`` was never called, force-overrides the
                # window icon with its own bundled ``.ico``. ``iconphoto``
                # doesn't flip that flag, so flip it manually to keep
                # the headphones emoji on the titlebar.
                self._iconbitmap_method_called = True
            except Exception as e:
                print(f"iconphoto failed: {e}")

        # The shared state object every tab will subscribe to. Tabs
        # never talk to each other directly; they read / publish via
        # ``state_obj``. See ``calliope.gui_common.AppState`` for the
        # full publish/subscribe surface.
        self.state_obj = AppState()

        # Re-skin every ``ttk`` widget class so legacy widgets inside
        # tabs (not yet migrated to CTk in Phase 3) blend with the
        # dark surround instead of glowing white.
        apply_ttk_dark_theme(self)

        # Two-pane root: sidebar on the left, content host on the
        # right. ``grid`` weights ensure only the content column grows
        # on window resize -- the sidebar keeps its fixed width.
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)

        # Build sidebar with one button per tab (same order as the old
        # notebook). ``_show_tab`` receives the integer index when the
        # user clicks.
        tab_labels = [
            "0. Batch runner",
            "1. Input & Preprocess",
            "2. QC Preview",
            "3. Suite2p Detection",
            "4. Low-pass filter",
            "5. Event detection",
            "6. Clustering",
            "7. Cross-correlation",
            "8. Spatial propagation",
        ]
        self._sidebar = _Sidebar(self, labels=tab_labels,
                                 on_select=self._show_tab)
        self._sidebar.grid(row=0, column=0, sticky="nsw")

        # Content host: every tab body is grided into row 0 / col 0 of
        # this frame. ``_show_tab`` hides the outgoing one and reveals
        # the incoming one. Only one body is mapped at a time -- which
        # is what fires the ``on_tab_shown`` / ``on_tab_hidden`` hooks.
        self._content = ctk.CTkFrame(self, corner_radius=0,
                                     fg_color="transparent")
        self._content.grid(row=0, column=1, sticky="nsew")
        self._content.grid_rowconfigure(0, weight=1)
        self._content.grid_columnconfigure(0, weight=1)

        # Instantiate every tab eagerly so the publish/subscribe wiring
        # in ``AppState`` is connected from the start. ``_make_scrollable_tab``
        # returns the (container, tab) pair; we hold both so the
        # container can be grided/hidden and the tab can receive
        # lifecycle hooks.
        # Every tab is scrollable; ``_make_scrollable_tab`` patches the
        # underlying CTkScrollableFrame so its inner frame stretches to
        # canvas height when content fits (top-level PanedWindow tabs
        # then redistribute within the viewport), and scrolling only
        # activates when content's natural height exceeds the viewport.
        tab_classes = [
            BatchTab, PreprocessTab, QcTab, Suite2pTab, LowpassTab,
            EventDetectionTab, ClusteringTab, CrossCorrelationTab,
            SpatialPropagationTab,
        ]
        self._tabs_in_order: list = []
        self._tab_containers: list = []
        for cls in tab_classes:
            container, tab = _make_scrollable_tab(
                self._content, cls, self.state_obj)
            self._tabs_in_order.append(tab)
            self._tab_containers.append(container)

        # Convenience attributes preserved from the old layout so
        # external code (tests, batch tab) can still poke individual
        # tabs by name.
        (self.batch_tab, self.preprocess_tab, self.qc_tab,
         self.detection_tab, self.lowpass_tab, self.event_tab,
         self.clustering_tab, self.xcorr_tab, self.spatial_tab) = \
            self._tabs_in_order

        # ``-1`` is a sentinel meaning "no tab is shown yet"; the
        # first ``_show_tab(0)`` call below promotes Tab 0.
        self._current_tab_index = -1
        self._show_tab(0)

    def _show_tab(self, idx_or_widget) -> None:
        """Activate the tab at ``idx_or_widget`` (int index OR tab
        instance / container).

        Fires ``on_tab_hidden`` on the outgoing tab and ``on_tab_shown``
        on the incoming one (both optional). Equivalent to the old
        ``ttk.Notebook`` selection event.
        """
        if isinstance(idx_or_widget, int):
            idx = idx_or_widget
        else:
            try:
                idx = self._tabs_in_order.index(idx_or_widget)
            except ValueError:
                idx = self._tab_containers.index(idx_or_widget)
        if not 0 <= idx < len(self._tab_containers):
            return
        if idx == self._current_tab_index:
            return
        # Hide outgoing.
        if 0 <= self._current_tab_index < len(self._tab_containers):
            old_container = self._tab_containers[self._current_tab_index]
            old_tab = self._tabs_in_order[self._current_tab_index]
            old_container.grid_forget()
            on_hidden = getattr(old_tab, "on_tab_hidden", None)
            if callable(on_hidden):
                try:
                    on_hidden()
                except Exception as e:
                    print(
                        f"on_tab_hidden({type(old_tab).__name__}) failed: {e}")
        # Show incoming.
        new_container = self._tab_containers[idx]
        new_tab = self._tabs_in_order[idx]
        new_container.grid(row=0, column=0, sticky="nsew")
        self._sidebar.set_active(idx)
        self._current_tab_index = idx
        on_shown = getattr(new_tab, "on_tab_shown", None)
        if callable(on_shown):
            try:
                on_shown()
            except Exception as e:
                print(f"on_tab_shown({type(new_tab).__name__}) failed: {e}")


def main() -> None:
    """Entrypoint: instantiate the app and start the event loop.

    ``mainloop()`` blocks until the user closes the window. Until
    then, Tk dispatches mouse / keyboard / timer events to whichever
    widget callbacks were registered above. R's analogue is
    ``shiny::runApp()``.

    The two ``customtkinter`` calls below must run *before* any CTk
    widget is created. ``"dark"`` flips the global appearance mode;
    ``"blue"`` picks the accent palette used by buttons and active
    sidebar items.
    """
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    PipelineApp().mainloop()


# Safety belt: if someone runs this file directly (``python
# pipeline_gui.py``) instead of ``python -m calliope``, this still
# launches the GUI rather than doing nothing.
if __name__ == "__main__":
    main()
