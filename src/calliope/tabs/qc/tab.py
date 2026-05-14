"""calliope.tabs.qc.tab - Tab 2: QC preview.

What the user sees
------------------
Two side-by-side panels: an animated GIF of the shifted movie on the
left, and the per-pixel mean image on the right. There's a checkbox
to pause the animation (which also frees the frame buffer for
memory-strapped machines) and a "Reload from folder..." button to
re-load an already-preprocessed recording from disk.

How this tab is driven
----------------------
Tab 1 publishes a ``PreprocessResult`` on the shared
``AppState`` whenever it finishes a preprocess run. Tab 2 subscribes
to that channel in ``__init__`` -- ``state.subscribe(self._on_result)``
registers ``_on_result`` as a callback that will be invoked from
Tab 1's worker. ``_on_result`` then materialises the GIF frames into
``ImageTk.PhotoImage`` objects and displays the mean image.

Memory note
-----------
A 5-minute GIF can occupy ~50 MB once each frame has been turned
into a Tk PhotoImage. The "Animate" checkbox flips between
"playing animation" (frame buffer in RAM) and "still image only"
(buffer freed); the ``on_tab_hidden`` hook also unloads the buffer
when the user switches away.

Class layout
------------
``QcTab(ttk.Frame)``
    The whole tab. Lifecycle:
        __init__               build widgets, subscribe to AppState.
        _on_result             callback: arrange the GIF + mean panels.
        _start_animation       schedule successive ``self.after`` calls
                                that flip frames in the Label widget.
        _reload_from_folder    "Reload..." button handler.
        on_tab_hidden / on_tab_shown
                                free / re-materialise the frame buffer
                                when the user navigates between tabs.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import customtkinter as ctk

import matplotlib
# Tell matplotlib to use the Tk-Agg backend BEFORE pyplot is
# imported; needed so the figure canvas integrates cleanly into Tk.
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# ``ImageTk.PhotoImage`` is Pillow's bridge into a Tk-displayable
# image. Each animation frame becomes one PhotoImage.
from PIL import Image, ImageTk

from . import logic as preprocessing
from .logic import PreprocessResult

from ...gui_common import AppState, attach_fig_toolbar


class QcTab(ttk.Frame):
    """QC preview: GIF playback on the left, mean image on the right."""

    # Inter-frame delay in milliseconds. 66 ms ≈ 15 fps.
    FRAME_MS = 66

    def __init__(self, master, state: AppState) -> None:
        super().__init__(master, padding=10)
        self.state = state
        # Pre-materialised list of Tk PhotoImage frames; populated
        # when the GIF loads, cleared when the user pauses animation
        # or hides the tab. List is cheap; PhotoImage objects are
        # not.
        self._gif_frames: list[ImageTk.PhotoImage] = []
        # Index of the next frame to display.
        self._gif_index = 0
        # Handle returned by ``self.after(...)`` for the next frame
        # tick. We hold onto it so we can cancel the pending tick if
        # the user pauses or hides the tab.
        self._gif_job: Optional[str] = None
        # Remember the GIF source so we can re-materialise frames after
        # the user switches away from this tab and comes back, or after
        # they manually toggle the "Animate" checkbox back on. Also tracks
        # whether we're currently in "frozen" (single-frame) mode.
        self._gif_path: Optional[Path] = None
        self._gif_frozen = False
        # User-driven master switch. Default ON; flipping OFF drops the
        # frame buffer to a single still image, flipping back ON reloads.
        self._gif_animate_var = tk.BooleanVar(value=True)

        # Build the widgets...
        self._build_ui()
        # ...and subscribe to the preprocess result channel. The
        # callback will fire when Tab 1 publishes its result, even
        # if that happens long before the user clicks over to this
        # tab.
        state.subscribe(self._on_result)

    # -- UI -----------------------------------------------------------------

    def _build_ui(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", pady=(0, 6))
        self.header_var = tk.StringVar(
            value="No preprocessing result yet. Run it in the first tab.")
        # ``ttk.Label`` retained where ``textvariable=`` is needed -- CTkLabel
        # has no equivalent. The global dark ttk style keeps it on-theme.
        ttk.Label(header, textvariable=self.header_var,
                  font=("", 10, "italic")).pack(side="left")
        ctk.CTkButton(header, text="Reload from folder...",
                      command=self._reload_from_folder).pack(side="right")

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1, uniform="cols")
        body.columnconfigure(1, weight=1, uniform="cols")
        body.rowconfigure(0, weight=1)

        gif_frame = ttk.LabelFrame(body, text="QC movie (GIF)", padding=6)
        gif_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        # Animate-toggle row at the top: unchecking drops the frame buffer
        # to a single still (saves ~1 MB per 512x512 frame); rechecking
        # reloads the GIF from disk and resumes playback.
        toggle_row = ctk.CTkFrame(gif_frame, fg_color="transparent")
        toggle_row.pack(anchor="w", pady=(0, 4))
        ctk.CTkCheckBox(
            toggle_row,
            text="Animate (uncheck to free frame buffer; still image stays)",
            variable=self._gif_animate_var,
            command=self._on_animate_toggle,
        ).pack(side="left")
        # gif_label holds a Pillow ImageTk.PhotoImage -- keep ttk.Label.
        # CTkLabel only accepts CTkImage which can't wrap a Tk PhotoImage.
        self.gif_label = ttk.Label(gif_frame, anchor="center")
        self.gif_label.pack(fill="both", expand=True)
        self.gif_status = tk.StringVar(value="")
        ttk.Label(gif_frame, textvariable=self.gif_status).pack(anchor="w",
                                                                pady=(4, 0))

        mean_frame = ttk.LabelFrame(body, text="Mean image", padding=6)
        mean_frame.grid(row=0, column=1, sticky="nsew")

        self.fig = plt.Figure(figsize=(5, 5), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        self.ax.text(0.5, 0.5, "No data", ha="center", va="center",
                     transform=self.ax.transAxes)
        self.canvas = FigureCanvasTkAgg(self.fig, master=mean_frame)
        attach_fig_toolbar(self.canvas, mean_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # -- Reload helper ------------------------------------------------------

    def _reload_from_folder(self) -> None:
        path = filedialog.askdirectory(
            title="Select an already-preprocessed recording folder")
        if not path:
            return
        result = preprocessing.load_existing_preprocess(path)
        if result is None:
            messagebox.showwarning(
                "No outputs",
                f"Folder {path} doesn't contain the expected "
                "shifted tiff / qc.gif / mean.npy.")
            return
        self.state.set_result(result)

    # -- AppState listener --------------------------------------------------

    def _on_result(self, result: PreprocessResult) -> None:
        self.header_var.set(
            f"Recording: {result.out_dir.name}   "
            f"({result.shape_yx[0]} x {result.shape_yx[1]})")
        self._load_gif(result.qc_gif)
        self._draw_mean_image(result)

    # -- GIF playback -------------------------------------------------------

    def _load_gif(self, gif_path: Path) -> None:
        if self._gif_job is not None:
            self.after_cancel(self._gif_job)
            self._gif_job = None
        self._gif_frames = []
        self._gif_index = 0
        self._gif_path = gif_path
        self._gif_frozen = False

        if not gif_path.exists():
            self.gif_status.set("GIF missing.")
            return

        # If the user has the Animate toggle off, only decode the first
        # frame -- enough to display a still without paying the full
        # PhotoImage frame-buffer cost.
        animate = self._gif_animate_var.get()
        try:
            im = Image.open(str(gif_path))
            if not animate:
                self._gif_frames.append(ImageTk.PhotoImage(im.copy()))
            else:
                while True:
                    self._gif_frames.append(ImageTk.PhotoImage(im.copy()))
                    im.seek(im.tell() + 1)
        except EOFError:
            pass
        except Exception as e:
            self.gif_status.set(f"GIF load error: {e}")
            return

        if not self._gif_frames:
            self.gif_status.set("GIF has no frames.")
            return

        if not animate:
            # Show the single still and stay frozen; toggling Animate on
            # later will reload the full GIF.
            self._gif_frozen = True
            frame = self._gif_frames[0]
            self.gif_label.configure(image=frame)
            self.gif_label.image = frame
            self.gif_status.set("Animate off (single frame)")
            return

        self.gif_status.set(f"{len(self._gif_frames)} frames")
        self._advance_gif()

    def _advance_gif(self) -> None:
        if not self._gif_frames:
            return
        frame = self._gif_frames[self._gif_index]
        self.gif_label.configure(image=frame)
        self.gif_label.image = frame
        self._gif_index = (self._gif_index + 1) % len(self._gif_frames)
        self._gif_job = self.after(self.FRAME_MS, self._advance_gif)

    # -- Animate toggle + tab visibility hooks -----------------------------

    def _freeze_to_still(self, status: str) -> None:
        """Stop the animation and drop every cached frame except the one
        currently on screen. ``self.gif_label.image`` holds the kept
        frame, so the still stays visible without a full frame buffer.
        """
        if self._gif_job is not None:
            self.after_cancel(self._gif_job)
            self._gif_job = None
        if len(self._gif_frames) <= 1:
            return
        kept = self._gif_frames[self._gif_index]
        self._gif_frames = [kept]
        self._gif_index = 0
        self._gif_frozen = True
        import gc
        gc.collect()
        self.gif_status.set(status)

    def _on_animate_toggle(self) -> None:
        """Master switch driven by the ``Animate`` checkbox.

        OFF -> freeze to a single still image and free the frame buffer.
        ON  -> reload the GIF from disk (if we know its path) and resume
        playback.
        """
        if self._gif_animate_var.get():
            if self._gif_path is not None and self._gif_path.exists():
                self._load_gif(self._gif_path)
            elif len(self._gif_frames) > 1 and self._gif_job is None:
                self._advance_gif()
        else:
            self._freeze_to_still(
                "Paused by user (single frame held to save memory)")

    def on_tab_hidden(self) -> None:
        """Called by ``PipelineApp`` when the user switches off this tab.

        Always freezes the animation so PhotoImage memory is released;
        if the user has the ``Animate`` toggle on, the next ``on_tab_shown``
        re-materialises the buffer.
        """
        self._freeze_to_still(
            "Paused (single frame held to save memory)")

    def on_tab_shown(self) -> None:
        """Called when the user switches back to this tab.

        Honours the user's ``Animate`` toggle: if it's off, we leave the
        still image in place. If it's on and we previously dropped the
        frame buffer, re-materialise it from ``self._gif_path``.
        """
        if not self._gif_animate_var.get():
            return
        if self._gif_frozen and self._gif_path is not None and self._gif_path.exists():
            self._load_gif(self._gif_path)
            return
        if len(self._gif_frames) > 1 and self._gif_job is None:
            self._advance_gif()

    # -- Mean image plot ----------------------------------------------------

    def _draw_mean_image(self, result: PreprocessResult) -> None:
        mean = np.load(str(result.mean_image_path))

        self.ax.clear()
        self.ax.set_axis_off()
        vmax = float(np.quantile(mean, 0.995))
        self.ax.imshow(mean, cmap="gray", vmax=vmax)
        self.ax.set_title("Mean image")
        self.canvas.draw_idle()
