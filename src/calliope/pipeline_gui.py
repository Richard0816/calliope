"""calliope.pipeline_gui - top-level CalLIOPE GUI for the calcium imaging
pipeline.

Tabs (each lives in its own subfolder under ``calliope.tabs``):
    1. Input & Preprocess     -> calliope.tabs.preprocess.tab.PreprocessTab
    2. QC Preview             -> calliope.tabs.qc.tab.QcTab
    3. Suite2p Detection      -> calliope.tabs.suite2p.tab.Suite2pTab
    4. Low-pass filter        -> calliope.tabs.lowpass.tab.LowpassTab
    5. Event detection        -> calliope.tabs.event_detection.tab.EventDetectionTab
    6. Clustering             -> calliope.tabs.clustering.tab.ClusteringTab
    7. Cross-correlation      -> calliope.tabs.crosscorrelation.tab.CrossCorrelationTab

Shared state (``AppState``) and Tk helpers (advanced-params dialog,
matplotlib toolbar wrapper, manual-ROI parsing, queue-backed log writer)
live in ``calliope.gui_common``. Each tab is wrapped by
``_make_scrollable_tab`` so its contents stay reachable on smaller windows.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional

from PIL import Image, ImageDraw, ImageFont, ImageTk

from .gui_common import AppState

from .tabs.preprocess.tab import PreprocessTab
from .tabs.qc.tab import QcTab
from .tabs.suite2p.tab import Suite2pTab
from .tabs.lowpass.tab import LowpassTab
from .tabs.event_detection.tab import EventDetectionTab
from .tabs.clustering.tab import ClusteringTab
from .tabs.crosscorrelation.tab import CrossCorrelationTab


# ---------------------------------------------------------------------------
# Window icon + scrollable tab wrapper
# ---------------------------------------------------------------------------

def _render_emoji_icon(emoji: str, size: int = 128) -> Optional[ImageTk.PhotoImage]:
    """Render ``emoji`` to an RGBA PhotoImage for use with ``Tk.iconphoto``.

    Uses Segoe UI Emoji on Windows (color bitmap glyphs, requires
    Pillow >= 9 for ``embedded_color=True``). Returns None on failure.
    """
    try:
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        font = None
        for candidate in ("seguiemj.ttf", "AppleColorEmoji.ttf",
                          "NotoColorEmoji.ttf"):
            try:
                font = ImageFont.truetype(candidate, int(size * 0.8))
                break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()

        try:
            draw.text((size // 2, size // 2), emoji, font=font,
                      anchor="mm", embedded_color=True)
        except TypeError:
            draw.text((size // 2, size // 2), emoji, font=font,
                      anchor="mm", fill=(0, 0, 0, 255))

        return ImageTk.PhotoImage(img)
    except Exception as e:
        print(f"icon render failed: {e}")
        return None


def _make_scrollable_tab(nb, tab_class, *args, **kwargs):
    """Wrap a tab class instance in a vertically-scrollable canvas.

    Returns (container, tab). Add ``container`` to the notebook; ``tab`` is
    the original tab instance and exposes its full API unchanged. The tab
    is sized to the canvas width (so ``pack(fill='x')`` still works) and
    its natural height drives the scroll region.
    """
    container = ttk.Frame(nb)
    canvas = tk.Canvas(container, highlightthickness=0, borderwidth=0)
    vsb = ttk.Scrollbar(container, orient="vertical",
                        command=canvas.yview)
    canvas.configure(yscrollcommand=vsb.set)
    canvas.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    container.rowconfigure(0, weight=1)
    container.columnconfigure(0, weight=1)

    tab = tab_class(canvas, *args, **kwargs)
    win_id = canvas.create_window((0, 0), window=tab, anchor="nw")

    def _on_tab_config(_e):
        canvas.configure(scrollregion=canvas.bbox("all"))
    tab.bind("<Configure>", _on_tab_config)

    def _on_canvas_config(event):
        canvas.itemconfigure(win_id, width=event.width)
    canvas.bind("<Configure>", _on_canvas_config)

    def _on_wheel(event):
        if abs(event.delta) >= 120:
            delta = -int(event.delta / 120)
        else:
            delta = -1 if event.delta > 0 else 1
        canvas.yview_scroll(delta, "units")

    def _on_enter(_e):
        canvas.bind_all("<MouseWheel>", _on_wheel)

    def _on_leave(_e):
        canvas.unbind_all("<MouseWheel>")

    container.bind("<Enter>", _on_enter)
    container.bind("<Leave>", _on_leave)

    return container, tab


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

class PipelineApp(tk.Tk):
    APP_NAME = "CalLIOPE"
    APP_SUBTITLE = "Calcium Live-imaging Output Pipeline for Epileptiform-recordings"
    APP_EMOJI = "\U0001F3A7"  # headphones

    def __init__(self) -> None:
        super().__init__()
        self.title(f"{self.APP_EMOJI} {self.APP_NAME} - {self.APP_SUBTITLE}")
        self.geometry("1100x720")

        self._icon_photo = _render_emoji_icon(self.APP_EMOJI, size=128)
        if self._icon_photo is not None:
            try:
                self.iconphoto(True, self._icon_photo)
            except Exception as e:
                print(f"iconphoto failed: {e}")

        self.state_obj = AppState()

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        def add(tab_class, text):
            container, tab = _make_scrollable_tab(
                nb, tab_class, self.state_obj)
            nb.add(container, text=text)
            return tab

        self.preprocess_tab = add(PreprocessTab, "1. Input & Preprocess")
        self.qc_tab = add(QcTab, "2. QC Preview")
        self.detection_tab = add(Suite2pTab, "3. Suite2p    Detection")
        self.lowpass_tab = add(LowpassTab, "4. Low-pass filter")
        self.event_tab = add(EventDetectionTab, "5. Event detection")
        self.clustering_tab = add(ClusteringTab, "6. Clustering")
        self.xcorr_tab = add(CrossCorrelationTab, "7. Cross-correlation")

        # Dispatch ``on_tab_shown`` / ``on_tab_hidden`` hooks so individual
        # tabs can free heavy resources (e.g. the QC tab's PhotoImage frame
        # buffer) when the user switches away. Tabs without these methods
        # are simply skipped.
        self._tabs_in_order = [
            self.preprocess_tab, self.qc_tab, self.detection_tab,
            self.lowpass_tab, self.event_tab, self.clustering_tab,
            self.xcorr_tab,
        ]
        self._nb = nb
        self._current_tab_index = nb.index("current")
        nb.bind("<<NotebookTabChanged>>", self._on_notebook_tab_changed)

    def _on_notebook_tab_changed(self, _event) -> None:
        try:
            new_index = self._nb.index("current")
        except tk.TclError:
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
    PipelineApp().mainloop()


if __name__ == "__main__":
    main()
