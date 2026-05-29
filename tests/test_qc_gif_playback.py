"""Regression tests for QcTab GIF playback (Tab 2).

These guard the bugs the LLM-council audit (2026-05-28) found in the old
buffered design, where re-checking the "Animate" toggle could silently
do nothing:

  * pausing must not destroy state, so resuming never re-reads the file;
  * resuming with no recording loaded must surface a message, never a
    silent no-op;
  * a single-frame GIF must wrap cleanly instead of dead-ending;
  * hiding the tab releases the file handle and showing it reopens.

The tab streams one frame at a time from an open ``PIL.Image`` handle, so
we exercise the real ``QcTab`` methods against a lightweight stub whose Tk
surface (``after``/``after_cancel``/label/status) is faked. ``ImageTk``
still needs a live Tk root, so the whole module skips if Tk can't start
(e.g. a headless CI box with no display).
"""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

import pytest
from PIL import Image

# A Tk root is required for ImageTk.PhotoImage; skip cleanly if unavailable.
try:
    _ROOT = tk.Tk()
    _ROOT.withdraw()
except tk.TclError:  # pragma: no cover - headless environment
    pytest.skip("Tk display unavailable", allow_module_level=True)

from calliope.tabs.qc.tab import QcTab


# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------

def _write_gif(path: Path, n_frames: int) -> Path:
    """Write an ``n_frames``-frame greyscale GIF to ``path``."""
    frames = [Image.new("L", (8, 8), color=(i * 30) % 256) for i in range(n_frames)]
    if n_frames == 1:
        frames[0].save(str(path))
    else:
        frames[0].save(str(path), save_all=True, append_images=frames[1:],
                       duration=66, loop=0)
    return path


class _FakeAfter:
    """Records after_cancel calls and hands out fake job ids without ever
    actually scheduling (so playback never recurses during a test)."""

    def __init__(self) -> None:
        self.cancelled: list[str] = []
        self._n = 0

    def after(self, _ms, _cb):
        self._n += 1
        return f"job{self._n}"

    def after_cancel(self, job):
        self.cancelled.append(job)


def _make_tab(animate: bool = True) -> tuple[QcTab, _FakeAfter]:
    """Build a QcTab with its real methods but a faked Tk surface."""
    obj = QcTab.__new__(QcTab)
    clock = _FakeAfter()
    obj.after = clock.after            # type: ignore[method-assign]
    obj.after_cancel = clock.after_cancel  # type: ignore[method-assign]
    obj._gif_im = None
    obj._gif_photo = None
    obj._gif_index = 0
    obj._gif_n_frames = 0
    obj._gif_job = None
    obj._gif_path = None
    obj._gif_animate_var = tk.BooleanVar(master=_ROOT, value=animate)
    obj.gif_status = tk.StringVar(master=_ROOT, value="")
    obj.gif_label = ttk.Label(_ROOT)
    return obj, clock


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_load_multiframe_starts_playing(tmp_path):
    gif = _write_gif(tmp_path / "qc.gif", 5)
    tab, clock = _make_tab(animate=True)

    tab._load_gif(gif)

    assert tab._gif_im is not None                  # handle held open
    assert tab._gif_job is not None                 # clock armed
    assert "Playing" in tab.gif_status.get()


def test_load_with_animate_off_shows_still_no_clock(tmp_path):
    gif = _write_gif(tmp_path / "qc.gif", 5)
    tab, clock = _make_tab(animate=False)

    tab._load_gif(gif)

    assert tab._gif_im is not None
    assert tab._gif_job is None                      # paused: no clock
    assert "Paused" in tab.gif_status.get()


def test_pause_then_resume_does_not_reload_file(tmp_path):
    gif = _write_gif(tmp_path / "qc.gif", 5)
    tab, clock = _make_tab(animate=True)
    tab._load_gif(gif)
    handle_before = tab._gif_im

    # Uncheck -> pause.
    tab._gif_animate_var.set(False)
    tab._on_animate_toggle()
    assert tab._gif_job is None
    assert "Paused" in tab.gif_status.get()
    assert tab._gif_im is handle_before             # NOT closed/reloaded

    # Re-check -> resume from the SAME open handle.
    tab._gif_animate_var.set(True)
    tab._on_animate_toggle()
    assert tab._gif_job is not None
    assert tab._gif_im is handle_before             # still the same handle


def test_resume_with_no_recording_is_never_silent():
    tab, clock = _make_tab(animate=False)            # nothing ever loaded
    tab._gif_animate_var.set(True)

    tab._on_animate_toggle()

    assert tab._gif_job is None
    assert "Can't play" in tab.gif_status.get()      # explains, never silent


def test_resume_after_file_deleted_is_never_silent(tmp_path):
    gif = _write_gif(tmp_path / "qc.gif", 5)
    tab, clock = _make_tab(animate=True)
    tab._load_gif(gif)

    # Simulate leaving the tab (handle released) then the folder vanishing.
    tab.on_tab_hidden()
    assert tab._gif_im is None
    gif.unlink()

    tab._gif_animate_var.set(True)
    tab._on_animate_toggle()

    assert "Can't play" in tab.gif_status.get()


def test_missing_path_reports_missing(tmp_path):
    tab, clock = _make_tab(animate=True)

    tab._load_gif(tmp_path / "nope.gif")

    assert tab._gif_im is None
    assert "missing" in tab.gif_status.get().lower()


def test_single_frame_gif_wraps_without_crashing(tmp_path):
    gif = _write_gif(tmp_path / "one.gif", 1)
    tab, clock = _make_tab(animate=True)
    tab._load_gif(gif)

    # Advancing past the only frame must wrap to 0, not raise.
    tab._advance_gif()

    assert tab._gif_index == 0
    assert tab._gif_n_frames == 1
    assert tab._gif_job is not None


def test_advance_cycles_and_learns_frame_count(tmp_path):
    gif = _write_gif(tmp_path / "qc.gif", 3)
    tab, clock = _make_tab(animate=True)
    tab._load_gif(gif)
    assert tab._gif_index == 0

    tab._advance_gif()
    assert tab._gif_index == 1
    tab._advance_gif()
    assert tab._gif_index == 2
    tab._advance_gif()                               # wraps 3 -> 0
    assert tab._gif_index == 0
    assert tab._gif_n_frames == 3


def test_start_playback_is_idempotent(tmp_path):
    gif = _write_gif(tmp_path / "qc.gif", 5)
    tab, clock = _make_tab(animate=True)
    tab._load_gif(gif)
    existing = tab._gif_job

    # A second start must cancel the live clock before arming a new one,
    # so on_tab_shown + a toggle can never leave two loops running.
    tab._start_playback()

    assert existing in clock.cancelled
    assert tab._gif_job is not None
    assert tab._gif_job != existing


def test_hide_releases_handle_show_reopens(tmp_path):
    gif = _write_gif(tmp_path / "qc.gif", 5)
    tab, clock = _make_tab(animate=True)
    tab._load_gif(gif)

    tab.on_tab_hidden()
    assert tab._gif_im is None                       # file lock dropped
    assert tab._gif_job is None
    assert tab._gif_path == gif                      # remembered for reopen

    tab.on_tab_shown()
    assert tab._gif_im is not None                   # reopened lazily
    assert tab._gif_job is not None
