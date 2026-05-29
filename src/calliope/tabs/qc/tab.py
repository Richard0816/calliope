"""calliope.tabs.qc.tab - Tab 2: QC preview.

What the user sees
------------------
Two side-by-side panels: an animated GIF of the shifted movie on the
left, and the per-pixel mean image on the right. There's a checkbox
to pause/resume the animation and a "Reload from folder..." button to
re-load an already-preprocessed recording from disk.

How this tab is driven
----------------------
Tab 1 publishes a ``PreprocessResult`` on the shared
``AppState`` whenever it finishes a preprocess run. Tab 2 subscribes
to that channel in ``__init__`` -- ``state.subscribe(self._on_result)``
registers ``_on_result`` as a callback that runs on the Tk main thread
(Tab 1's worker hands the result back through a queue drained by an
``after`` poll). ``_on_result`` opens the GIF and displays the mean
image.

Playback model
--------------
The movie is *streamed* one frame at a time straight from an open
``PIL.Image`` handle: each tick seeks to the next frame and builds a
single ``ImageTk.PhotoImage`` for it. Only one frame is ever in memory,
so playback costs the same whether the clip is 5 seconds or 5 minutes --
there is no multi-megabyte frame buffer to build, free, or rebuild.

That single decision removes a whole class of bugs the old buffered
design suffered from: pausing only stops the clock (it never frees a
buffer), resuming never re-reads the file, and there is no bulk decode
to freeze the UI. The ``on_tab_hidden`` hook closes the file handle to
drop the OS file lock; ``on_tab_shown`` reopens it lazily.

Class layout
------------
``QcTab(ctk.CTkFrame)``
    The whole tab. Lifecycle:
        __init__               build widgets, subscribe to AppState.
        _on_result             callback: open the GIF + draw the mean.
        _load_gif              open the handle and arm/start playback.
        _advance_gif           one frame tick; re-arms via ``self.after``.
        _start_playback /      guarded scheduling -- only ever one clock.
        _stop_playback
        _on_animate_toggle     play/pause checkbox handler.
        _reload_from_folder    "Reload..." button handler.
        on_tab_hidden /        stop the clock + release / reopen the
        on_tab_shown           file handle as the user navigates tabs.
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


class QcTab(ctk.CTkFrame):
    """QC preview: GIF playback on the left, mean image on the right."""

    # Inter-frame delay in milliseconds. 66 ms ≈ 15 fps.
    FRAME_MS = 66

    def __init__(self, master, state: AppState) -> None:
        super().__init__(master, fg_color="transparent")
        self.state = state
        # The movie is streamed straight from this open Pillow handle:
        # we hold exactly one decoded frame on screen at a time and seek
        # to the next one on each tick. Keeping memory flat at ~one frame
        # means pausing never has to free a buffer and resuming never has
        # to re-read the file. ``None`` when no GIF is open.
        self._gif_im: Optional[Image.Image] = None
        # The PhotoImage currently shown. Held as an attribute so Tk does
        # not garbage-collect the image out from under the label.
        self._gif_photo: Optional[ImageTk.PhotoImage] = None
        # Index of the frame currently on screen.
        self._gif_index = 0
        # Total frame count, learned lazily the first time playback wraps
        # past the end (0 = not known yet).
        self._gif_n_frames = 0
        # Handle returned by ``self.after(...)`` for the next frame tick.
        # Held so we can cancel a pending tick when the user pauses, hides
        # the tab, or loads a different recording. All scheduling goes
        # through _start_playback / _stop_playback so two clocks can never
        # run at once.
        self._gif_job: Optional[str] = None
        # Remember the GIF source so we can reopen it after the handle is
        # released on tab-hide, or after the user re-checks Animate.
        self._gif_path: Optional[Path] = None
        # User-driven play/pause switch. Default ON (playing).
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
        header.pack(fill="x", padx=10, pady=(10, 6))
        self.header_var = tk.StringVar(
            value="No preprocessing result yet. Run it in the first tab.")
        ctk.CTkLabel(header, textvariable=self.header_var,
                     font=ctk.CTkFont(size=12, slant="italic")).pack(side="left")
        ctk.CTkButton(header, text="Reload from folder...",
                      command=self._reload_from_folder).pack(side="right")

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        body.columnconfigure(0, weight=1, uniform="cols")
        body.columnconfigure(1, weight=1, uniform="cols")
        body.rowconfigure(0, weight=1)

        gif_frame = ctk.CTkFrame(body)
        gif_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        ctk.CTkLabel(gif_frame, text="QC movie (GIF)",
                     font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(6, 0))
        # Play/pause row. Playback streams one frame at a time from an
        # open handle, so unchecking only stops the clock (the current
        # frame stays on screen) and rechecking resumes instantly --
        # neither touches the disk.
        toggle_row = ctk.CTkFrame(gif_frame, fg_color="transparent")
        toggle_row.pack(anchor="w", padx=6, pady=(4, 4))
        ctk.CTkCheckBox(
            toggle_row,
            text="Animate (uncheck to pause playback)",
            variable=self._gif_animate_var,
            command=self._on_animate_toggle,
        ).pack(side="left")
        # gif_label holds a Pillow ImageTk.PhotoImage -- keep ttk.Label.
        # CTkLabel only accepts CTkImage which can't wrap a Tk PhotoImage.
        self.gif_label = ttk.Label(gif_frame, anchor="center")
        self.gif_label.pack(fill="both", expand=True, padx=6)
        self.gif_status = tk.StringVar(value="")
        ctk.CTkLabel(gif_frame, textvariable=self.gif_status).pack(
            anchor="w", padx=6, pady=(4, 6))

        mean_frame = ctk.CTkFrame(body)
        mean_frame.grid(row=0, column=1, sticky="nsew")
        ctk.CTkLabel(mean_frame, text="Mean image",
                     font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(6, 0))

        self.fig = plt.Figure(figsize=(5, 5), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        self.ax.text(0.5, 0.5, "No data", ha="center", va="center",
                     transform=self.ax.transAxes)
        self.canvas = FigureCanvasTkAgg(self.fig, master=mean_frame)
        attach_fig_toolbar(self.canvas, mean_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True,
                                         padx=6, pady=(4, 6))

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
        """Open ``gif_path`` and show its first frame, then start playback
        if Animate is on. Streams frames from the open handle -- nothing
        is bulk-decoded, so this returns immediately even for a long clip.
        """
        self._stop_playback()
        self._close_gif()
        self._gif_index = 0
        self._gif_n_frames = 0
        self._gif_path = gif_path

        if not gif_path.exists():
            self.gif_status.set("GIF missing.")
            return

        try:
            self._gif_im = Image.open(str(gif_path))
        except Exception as e:
            self._gif_im = None
            self.gif_status.set(f"GIF load error: {e}")
            return

        # Paint frame 0 up front so something is visible even while paused.
        if not self._show_frame(0):
            self.gif_status.set("GIF has no frames.")
            self._close_gif()
            return

        if self._gif_animate_var.get():
            self._start_playback()
        else:
            self.gif_status.set("Paused (frame 1)")

    def _close_gif(self) -> None:
        """Release the open Pillow handle so the OS file lock is dropped.
        The last painted frame stays on screen via ``self._gif_photo``."""
        if self._gif_im is not None:
            try:
                self._gif_im.close()
            except Exception:
                pass
            self._gif_im = None

    def _show_frame(self, index: int) -> bool:
        """Seek to ``index`` and paint it into the label.

        Seeks only ever advance by one or wrap to 0, so Pillow composites
        palette/disposal frames correctly. Returns ``False`` if the frame
        can't be decoded (e.g. an empty GIF). Updates ``self._gif_index``
        and learns ``self._gif_n_frames`` the first time we run off the end.
        """
        if self._gif_im is None:
            return False
        try:
            self._gif_im.seek(index)
        except EOFError:
            # Ran past the last frame: remember the length and wrap to 0.
            if index == 0:
                return False
            self._gif_n_frames = index
            index = 0
            try:
                self._gif_im.seek(0)
            except (EOFError, OSError):
                return False
        except OSError as e:
            self.gif_status.set(f"GIF decode error: {e}")
            return False
        try:
            photo = ImageTk.PhotoImage(self._gif_im.copy())
        except Exception as e:
            self.gif_status.set(f"GIF render error: {e}")
            return False
        self._gif_index = index
        self._gif_photo = photo
        self.gif_label.configure(image=photo)
        # The .image assignment is required so Python doesn't garbage-
        # collect the PhotoImage while it's still on screen.
        self.gif_label.image = photo
        return True

    def _advance_gif(self) -> None:
        # One playback tick: show the next frame and re-arm the clock.
        if self._gif_im is None:
            self._gif_job = None
            return
        self._show_frame(self._gif_index + 1)
        total = self._gif_n_frames or "?"
        self.gif_status.set(f"Playing - frame {self._gif_index + 1}/{total}")
        self._gif_job = self.after(self.FRAME_MS, self._advance_gif)

    # -- Playback scheduling + toggle + tab visibility hooks ---------------

    def _start_playback(self) -> None:
        """Arm the frame clock. Idempotent -- cancels any pending tick
        first, so on_tab_shown + a toggle can never spawn two loops."""
        if self._gif_im is None:
            return
        self._stop_playback()
        total = self._gif_n_frames or "?"
        self.gif_status.set(f"Playing - frame {self._gif_index + 1}/{total}")
        self._gif_job = self.after(self.FRAME_MS, self._advance_gif)

    def _stop_playback(self) -> None:
        """Cancel any pending frame tick. Safe to call repeatedly."""
        if self._gif_job is not None:
            self.after_cancel(self._gif_job)
            self._gif_job = None

    def _on_animate_toggle(self) -> None:
        """Play/pause switch driven by the ``Animate`` checkbox.

        ON  -> resume playback; reopen the file only if the handle was
               released (e.g. after the tab was hidden).
        OFF -> stop the clock; the current frame stays on screen.

        Never a silent no-op: every branch either plays or says why it
        can't.
        """
        if self._gif_animate_var.get():
            if self._gif_im is not None:
                self._start_playback()
            elif self._gif_path is not None and self._gif_path.exists():
                self._load_gif(self._gif_path)
            else:
                self._stop_playback()
                self.gif_status.set(
                    "Can't play - no QC movie loaded. Run or reload a "
                    "recording first.")
        else:
            self._stop_playback()
            self.gif_status.set(f"Paused (frame {self._gif_index + 1})")

    def on_tab_hidden(self) -> None:
        """Called by ``PipelineApp`` when the user switches off this tab.

        Stops the clock and releases the file handle (drops the OS lock).
        The last frame stays on screen; ``on_tab_shown`` reopens lazily.
        """
        self._stop_playback()
        self._close_gif()

    def on_tab_shown(self) -> None:
        """Called when the user switches back to this tab.

        Honours the ``Animate`` toggle: if it's off, leave the still in
        place; if it's on, reopen the GIF from ``self._gif_path`` and
        resume playback.
        """
        if not self._gif_animate_var.get():
            return
        if self._gif_im is not None:
            self._start_playback()
        elif self._gif_path is not None and self._gif_path.exists():
            self._load_gif(self._gif_path)

    # -- Mean image plot ----------------------------------------------------

    def _draw_mean_image(self, result: PreprocessResult) -> None:
        mean = np.load(str(result.mean_image_path))

        self.ax.clear()
        self.ax.set_axis_off()
        vmax = float(np.quantile(mean, 0.995))
        self.ax.imshow(mean, cmap="gray", vmax=vmax)
        self.ax.set_title("Mean image")
        self.canvas.draw_idle()
