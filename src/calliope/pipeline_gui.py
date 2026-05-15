"""calliope.pipeline_gui - top-level CalLIOPE GUI for the calcium imaging
pipeline.

Application root: creates the main window and stitches all nine tabs
into a single content host. ``main()`` at the bottom is the entrypoint
that ``python -m calliope`` calls.

Tabs (each lives in its own subfolder under ``calliope.tabs``):
    0. Batch runner           -> calliope.tabs.batch.tab.BatchTab
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

# ``from __future__ import annotations`` makes every type hint in this
# file a string at parse time instead of being evaluated immediately.
# Why we want that here: (1) we can reference names that aren't defined
# yet (e.g. forward references to tab classes), and (2) annotations
# stay cheap even when the file imports a lot of heavy modules.
from __future__ import annotations

# ``sys`` is only used for ``sys.platform`` -- we sniff it to apply a
# Windows-only taskbar workaround below.
import sys

# Vanilla tkinter. We only touch it directly when CTk doesn't cover
# a primitive we need (e.g. ``tk.TclError`` for the wheel-binding
# fallback) -- everything user-facing is built with customtkinter.
import tkinter as tk

# customtkinter is a third-party re-skin of Tk. ``ctk.CTk`` is its
# themed root window; we subclass it for ``PipelineApp`` so the whole
# app inherits CTk's widget styling without extra plumbing.
import customtkinter as ctk

# ``Optional[X]`` is shorthand for ``X | None`` -- we use it on the
# icon helper that may fail to load a font and return None.
from typing import Optional

# Pillow lets us rasterise an emoji into an in-memory image we can hand
# to Tk as the window icon. ``ImageTk.PhotoImage`` is the Tk-compatible
# wrapper.
from PIL import Image, ImageDraw, ImageFont, ImageTk

# A leading dot means "import from this same package". ``AppState`` is
# the shared pub/sub container every tab subscribes to;
# ``apply_ttk_dark_theme`` re-skins the legacy ttk widgets that some
# tabs still use.
from .gui_common import AppState, apply_ttk_dark_theme, install_scroll_router

# One tab class per sub-package. Importing them at the top lets
# ``PipelineApp.__init__`` wire them all in one pass.
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
    Pillow >= 9 for ``embedded_color=True``). Returns None on failure so
    the caller can fall back to Tk's default icon without crashing.

    The leading underscore is a Python convention for "module-private":
    nothing enforces it, but linters / IDEs treat it as a soft "don't
    import this from outside".
    """
    # The whole render is wrapped in try/except because any of the
    # Pillow calls can raise (missing font, unsupported keyword on
    # older Pillow). A missing window icon isn't worth crashing for, so
    # we swallow the error and return None.
    try:
        # RGBA = red/green/blue/alpha; ``(0, 0, 0, 0)`` is fully
        # transparent. We draw the glyph onto this transparent square
        # so the icon has clean edges instead of a coloured backplate.
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Try platform-specific colour-emoji fonts in priority order;
        # first one that loads wins. ``ImageFont.truetype`` raises if
        # the font file isn't installed, so the loop is the cheapest
        # way to probe for whichever font this OS happens to ship.
        font = None
        for candidate in ("seguiemj.ttf", "AppleColorEmoji.ttf",
                          "NotoColorEmoji.ttf"):
            try:
                font = ImageFont.truetype(candidate, int(size * 0.8))
                break
            except Exception:
                continue
        # Pillow's built-in monochrome bitmap font as a last resort --
        # the user just sees a black-and-white glyph but the app still
        # gets an icon.
        if font is None:
            font = ImageFont.load_default()

        # ``embedded_color=True`` lights up COLR/CPAL colour glyphs on
        # newer Pillow; older versions raise TypeError on that kwarg,
        # so we catch and fall back to a flat monochrome draw.
        try:
            draw.text((size // 2, size // 2), emoji, font=font,
                      anchor="mm", embedded_color=True)
        except TypeError:
            draw.text((size // 2, size // 2), emoji, font=font,
                      anchor="mm", fill=(0, 0, 0, 255))

        # Wrap the Pillow image in the Tk-compatible PhotoImage class
        # so ``Tk.iconphoto`` will accept it.
        return ImageTk.PhotoImage(img)
    except Exception as e:
        # f-string: text with ``{expr}`` placeholders that get evaluated
        # and inlined. Used here to format the error message in one
        # line.
        print(f"icon render failed: {e}")
        return None


def _make_scrollable_tab(parent, tab_class, *args, **kwargs):
    """Wrap a tab class instance in a scrollable CTk container.

    Returns ``(container, tab)``. Mount ``container`` in the content
    host; ``tab`` is the original tab instance and exposes its full
    API unchanged.

    ``*args, **kwargs`` are the same forwarding pattern used in lots of
    factory functions -- whatever positional / keyword arguments the
    caller passes get forwarded straight to ``tab_class``. Lets the
    factory stay agnostic about what each tab class wants in its
    constructor.

    Stock ``CTkScrollableFrame`` only stretches its inner frame
    horizontally. When the canvas is taller than the content's natural
    height, the inner frame stays at natural height -- which leaves a
    gap below and starves any ``ttk.PanedWindow`` inside that wants to
    fill the tab. We patch around that here by binding ``<Configure>``
    and stretching the inner frame's *height* to
    ``max(natural, canvas_height)``. PanedWindows then fill the viewport
    when there's room, and scrolling kicks in only when content's
    natural height exceeds the viewport.
    """
    container = ctk.CTkScrollableFrame(parent, fg_color="transparent")
    tab = tab_class(container, *args, **kwargs)
    tab.pack(fill="both", expand=True)

    # ``_parent_canvas`` and ``_create_window_id`` are CTk-internal
    # attributes. Tapping private attrs is risky, but it's the only way
    # to reach the underlying Tk canvas + canvas-window we need to
    # resize. If CTk renames them, this is what would break first.
    canvas = container._parent_canvas
    window_id = container._create_window_id

    def _stretch_inner_to_canvas(_event=None):
        # Closure: this nested function captures ``canvas`` and
        # ``window_id`` from the enclosing function's scope, so the
        # Tk event callback has access to them at fire time.
        #
        # Fires on canvas resize AND on inner-frame resize (e.g. a
        # resize-handle drag growing one child), so the canvas window
        # grows to absorb the new requested height. Without the second
        # binding the canvas keeps its old fixed height and the new
        # child just crushes its siblings.
        canvas_h = canvas.winfo_height()
        natural_h = container.winfo_reqheight()
        canvas.itemconfigure(window_id, height=max(natural_h, canvas_h))

    # ``add="+"`` chains a new binding alongside whatever's already
    # registered for this event. Without it we'd overwrite CTk's own
    # ``<Configure>`` handler and break its width-stretch.
    canvas.bind("<Configure>", _stretch_inner_to_canvas, add="+")
    container.bind("<Configure>", _stretch_inner_to_canvas, add="+")

    # Wheel routing isn't installed here: the app root owns a single
    # ``install_scroll_router`` that resolves the active tab's wrapper
    # canvas dynamically, so each tab is automatically covered.
    return container, tab


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

class _Sidebar(ctk.CTkFrame):
    """Vertical navigation rail of CTkButtons -- one per tab.

    The active tab's button is highlighted via ``fg_color``; inactive
    buttons render flat against the sidebar background. Selection
    callback receives the integer index of the clicked tab.
    """

    def __init__(self, master, labels, on_select, width=210):
        # ``super()`` reaches the parent class (``ctk.CTkFrame``);
        # forwarding the relevant kwargs lets it set up its own widget
        # state before we add ours.
        super().__init__(master, width=width, corner_radius=0)
        self._on_select = on_select
        # ``ThemeManager.theme`` is a nested dict keyed by CTk widget
        # class -- we pull the active accent so our highlight matches
        # whichever theme is loaded instead of hard-coding a colour.
        self._active_color = ctk.ThemeManager.theme["CTkButton"]["fg_color"]
        # Type hint on a local attribute. Pure documentation -- Python
        # doesn't enforce list[ctk.CTkButton] at runtime, but the
        # annotation helps editors autocomplete inside the class.
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
                # Default-argument trick: ``i=idx`` captures the
                # current value of ``idx`` at lambda-definition time.
                # Without it, every lambda would close over the final
                # ``idx`` and every button would fire the same index.
                command=lambda i=idx: self._on_select(i),
            )
            btn.pack(fill="x", padx=8, pady=2)
            self._buttons.append(btn)

        # Both ``pack`` and ``grid`` geometry managers normally shrink
        # a container to fit its contents. Turning propagation off here
        # keeps the sidebar at its requested width even when a tab body
        # asks for more space.
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
    """The root window.

    Subclassing ``ctk.CTk`` (which itself inherits from ``tk.Tk``) means
    an instance of this class *is* a window -- ``self.title(...)``,
    ``self.geometry(...)``, ``self.mainloop()`` all work because they're
    inherited from up the chain.
    """

    # Class attributes (above ``__init__``) are shared by every
    # instance, so they behave like constants. ``APP_NAME`` etc. live
    # here instead of in ``__init__`` because they never need to vary
    # per-instance.
    APP_NAME = "CalLIOPE"
    APP_SUBTITLE = "Calcium Live-imaging Output Pipeline for Epileptiform-recordings"
    APP_EMOJI = "\U0001F3A7"  # headphones codepoint

    # Windows-specific app-id so the taskbar groups our windows under
    # their own icon instead of folding them into python.exe.
    APP_USER_MODEL_ID = "calliope.pipeline.1"

    def __init__(self) -> None:
        """Constructor. ``self`` is the implicit first argument and
        refers to "this instance" -- every attribute we set on ``self``
        becomes part of the live window's state.
        """
        # Windows groups taskbar icons by AppUserModelID; without this
        # call the process inherits python.exe's AppID and the taskbar
        # shows the Python icon regardless of iconphoto. Must run
        # before any Tk window is created, which is why it sits ahead
        # of ``super().__init__()``.
        if sys.platform == "win32":
            try:
                # ``ctypes`` lets Python call into Windows DLLs. We
                # import it inside the if-block so non-Windows installs
                # never have to touch it.
                import ctypes
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                    self.APP_USER_MODEL_ID)
            except Exception as e:
                print(f"SetCurrentProcessExplicitAppUserModelID failed: {e}")

        # ``super().__init__()`` runs the CTk root-window constructor,
        # which is what actually creates the underlying Tk window.
        # Until this line, ``self`` exists as a Python object but has
        # no Tk window behind it.
        super().__init__()
        self.title(f"{self.APP_EMOJI} {self.APP_NAME} - {self.APP_SUBTITLE}")
        self.geometry("1280x800")
        self.minsize(960, 640)

        # Hold a reference to the PhotoImage on ``self`` so the garbage
        # collector doesn't reclaim it while Tk still needs it. (Tk
        # only weakly references PhotoImages; if Python drops the last
        # strong ref, the icon vanishes mid-render.)
        self._icon_photo = _render_emoji_icon(self.APP_EMOJI, size=128)
        if self._icon_photo is not None:
            try:
                # ``True`` => set this as the default for *every* new Tk
                # window in this process, not just the root.
                self.iconphoto(True, self._icon_photo)
                # CTk checks a private flag and, if ``iconbitmap`` was
                # never called, force-overrides the window icon with
                # its bundled ``.ico``. ``iconphoto`` doesn't flip that
                # flag, so flip it manually here to keep our headphones
                # emoji on the titlebar.
                self._iconbitmap_method_called = True
            except Exception as e:
                print(f"iconphoto failed: {e}")

        # Shared pub/sub container. Tabs never talk to each other
        # directly: they publish updates through ``state_obj`` and
        # subscribe to the channels they care about. Full surface in
        # ``calliope.gui_common.AppState``.
        self.state_obj = AppState()

        # Re-skin every ttk widget class so legacy ttk widgets inside
        # tabs (not yet migrated to CTk) blend with the dark surround
        # instead of glowing white.
        apply_ttk_dark_theme(self)

        # Two-pane root: sidebar on the left, content host on the
        # right. ``grid_columnconfigure`` with ``weight=1`` is what
        # tells Tk "this column absorbs extra space on resize" -- only
        # column 1 (content) grows; the sidebar (column 0) keeps its
        # fixed width.
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)

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
        # the incoming one -- only one body is mapped at a time, which
        # is what fires the ``on_tab_shown`` / ``on_tab_hidden`` hooks.
        self._content = ctk.CTkFrame(self, corner_radius=0,
                                     fg_color="transparent")
        self._content.grid(row=0, column=1, sticky="nsew")
        self._content.grid_rowconfigure(0, weight=1)
        self._content.grid_columnconfigure(0, weight=1)

        # Tabs are instantiated eagerly (not lazily on first click) so
        # the publish/subscribe wiring in ``AppState`` is connected
        # from the start. ``_make_scrollable_tab`` returns the
        # (container, tab) pair: ``container`` is what we grid /
        # ungrid; ``tab`` is the original tab instance and still
        # exposes its full API.
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

        # Named handles so external code (tests, the batch tab) can
        # poke individual tabs by name without indexing into the list.
        # Tuple unpacking on the left mirrors the order in
        # ``tab_classes`` above.
        (self.batch_tab, self.preprocess_tab, self.qc_tab,
         self.detection_tab, self.lowpass_tab, self.event_tab,
         self.clustering_tab, self.xcorr_tab, self.spatial_tab) = \
            self._tabs_in_order

        # ``-1`` is the "no tab is shown yet" sentinel; the first
        # ``_show_tab(0)`` call below promotes Tab 0.
        self._current_tab_index = -1
        self._show_tab(0)

        # Single mouse-wheel router for the whole app. The fallback
        # canvas is resolved dynamically per event to the currently
        # visible tab's wrapper canvas -- so the wheel scrolls inner
        # consoles / listboxes / row canvases when the cursor is over
        # them, and falls through to scrolling the whole tab otherwise.
        # Popouts install their own routers (no fallback) so they
        # don't accidentally scroll the main window behind them.
        install_scroll_router(
            self, fallback_canvas=self._current_tab_canvas)

    def _current_tab_canvas(self):
        """Return the visible tab's wrapper canvas (the scrollable
        target for the app-wide wheel router's fallback path), or
        ``None`` if nothing is mapped yet."""
        idx = self._current_tab_index
        if 0 <= idx < len(self._tab_containers):
            return self._tab_containers[idx]._parent_canvas
        return None

    def _show_tab(self, idx_or_widget) -> None:
        """Activate the tab at ``idx_or_widget`` (int index OR tab
        instance / container).

        Fires ``on_tab_hidden`` on the outgoing tab and ``on_tab_shown``
        on the incoming one (both optional). Tabs use those hooks for
        cheap on-show refreshes (e.g. re-reading the working directory
        if Tab 1 might have changed it).
        """
        # The dispatch on input type lets callers pass either an index
        # (sidebar button click) or the actual widget/container (when
        # another tab wants to push the user to a specific tab).
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
            # ``getattr(obj, "name", None)`` is "give me obj.name or
            # None if it doesn't exist" -- handy for optional hooks
            # that not every tab implements.
            on_hidden = getattr(old_tab, "on_tab_hidden", None)
            if callable(on_hidden):
                try:
                    on_hidden()
                except Exception as e:
                    # Swallow hook errors so one misbehaving tab can't
                    # wedge the whole app.
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

    The two ``customtkinter`` calls below must run *before* any CTk
    widget is created. ``"dark"`` flips the global appearance mode;
    ``"blue"`` picks the accent palette used by buttons and active
    sidebar items.

    ``mainloop()`` blocks until the user closes the window -- Tk uses
    that time to dispatch mouse / keyboard / timer events to the
    callbacks we wired up above.
    """
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    PipelineApp().mainloop()


# ``if __name__ == "__main__"`` is true when this file is run directly
# (``python pipeline_gui.py``) and false when it's imported as a
# module. Keeping ``main()`` behind this guard lets us import the file
# without launching the GUI as a side effect.
if __name__ == "__main__":
    main()
