"""calliope.core.offload - run heavy compute in a child process.

Why this exists
---------------
The GUI is single-threaded Tk (customtkinter). Moving CPU-bound work onto
a ``threading.Thread`` does **not** keep the window responsive: tight
Python loops hold the GIL and starve Tk's event loop, so Windows paints
the window "Not Responding" and can't even repaint it on minimise/restore.
The only true fix is to run the heavy work in a *separate process* so the
GUI process keeps its own free GIL.

This module offers a small :class:`ProcessJob` (plus the
:func:`run_offloaded` convenience) that mirrors the existing
``threading.Thread`` + ``queue.Queue`` + :func:`calliope.gui_common.
drain_queue` ergonomics the tabs already use, so converting a tab is a
near-mechanical swap.

Contract for offload targets
----------------------------
* Must be a **module-level** function (picklable by reference) -- ideally
  living in ``core`` so the child imports it cheaply (``calliope`` and
  ``calliope.core`` ``__init__`` files are deliberately import-light).
* Receives only **picklable** arguments: path strings, param dicts,
  scalars. Never a live ``np.memmap`` / open file / Tk object -- re-open
  memmaps *by path* inside the child, anchoring filtered memmaps via
  ``utils.resolve_filtered_mask`` (see the WinError-8 note there).
* Accepts a ``progress_q`` keyword. Build a local progress callback from
  it with :func:`make_progress_cb` (returns ``None`` when progress is
  off). The callback's positional args are forwarded verbatim to the
  tab's progress handler, so any ``progress_cb`` signature works
  (``cb(msg)`` for event detection, ``cb(done, total, label)`` for
  cross-correlation, ...).
* Returns a **picklable** payload (ndarrays / plain dicts / path
  strings). The child wrapper -- not the target -- publishes done/error.

Threading / process model
--------------------------
* One ``multiprocessing.Process`` per job, ``spawn`` context. Cheap
  enough here because the package import is light, and it buys true GIL
  isolation plus a real ``.cancel()`` (terminate) for cancellable runs
  (e.g. cross-correlation). No ``ProcessPoolExecutor`` -- that can't be
  handed a plain ``mp.Queue`` for live progress and goes
  ``BrokenProcessPool`` if a worker dies.
* A single inherited ``mp.Queue`` carries tagged messages back:
  ``(_PROGRESS, args)`` during the run, then exactly one
  ``(_DONE, result)`` or ``(_ERROR, text)``.
* The parent drains that queue on the **Tk main thread** via
  ``widget.after`` and re-posts onto the tab's own ``queue.Queue`` using
  the tab's existing ``drain_queue`` *kind* strings. So result handlers
  -- and the AppState / batch ``set_*`` calls they make -- still run on
  the main thread exactly as before, and batch mode is unaffected.
"""

from __future__ import annotations

import multiprocessing as mp
import queue as _queue
import traceback
from typing import Any, Callable, Optional, Sequence

# Always spawn: it is the Windows default, and being explicit keeps the
# behaviour identical on macOS/Linux (no inherited Tk / fork-after-import
# surprises). The context is module-level so every job shares it.
_CTX = mp.get_context("spawn")

# Internal message tags carried over the child->parent queue. They are
# plain strings (picklable, identity-stable across the process boundary).
_PROGRESS = "__offload_progress__"
_DONE = "__offload_done__"
_ERROR = "__offload_error__"

# Number of consecutive empty polls after the child is observed dead
# before we conclude it exited without a terminal message. This grace
# window lets the mp.Queue feeder thread flush an in-transit ``_DONE``
# that was put just before the process exited (avoids a false
# "exited unexpectedly").
_DEAD_GRACE_POLLS = 3


def make_progress_cb(progress_q) -> Optional[Callable[..., None]]:
    """Build a progress callback that ships its args back to the parent.

    Returns ``None`` when ``progress_q`` is ``None`` so callers can pass
    the result straight through as ``progress_cb=`` (matching the
    ``Optional[Callable]`` that the compute functions already accept).
    The callback never raises -- a dead/closed queue is swallowed so a
    progress hiccup can't crash a long compute.
    """
    if progress_q is None:
        return None

    def _cb(*args: Any) -> None:
        # Forward the bare value for the common single-arg callback
        # (``progress_cb(message)``) so existing string-only status
        # handlers work unchanged; multi-arg callbacks
        # (``progress_cb(done, total, label)``) get the full tuple.
        payload = args[0] if len(args) == 1 else args
        try:
            progress_q.put((_PROGRESS, payload))
        except Exception:
            pass

    return _cb


def make_abort_event():
    """A spawn-context ``Event`` for cooperative cancellation.

    Pass it to an offload target in ``kwargs`` (it is inherited by the
    child at spawn). Set it from the GUI to request an abort; the
    target's progress callback checks it and raises at the next
    checkpoint, so the child stops cleanly *between* writes rather than
    being killed mid-write (matches the old in-thread ``RunAborted``
    behaviour). Use this instead of :meth:`ProcessJob.cancel` when the
    target writes output incrementally.
    """
    return _CTX.Event()


def _run_child(q, fn, args, kwargs) -> None:
    """Child-process entry: run ``fn`` and publish its outcome on ``q``.

    Runs in the spawned process. ``progress_q`` has already been injected
    into ``kwargs`` by the parent so ``fn`` can build its own progress
    callback. Catches ``BaseException`` so even a ``SystemExit`` / odd
    error still reports back instead of leaving the parent hanging.
    """
    try:
        result = fn(*args, **kwargs)
        q.put((_DONE, result))
    except BaseException:  # noqa: BLE001 - report everything to the parent
        q.put((_ERROR, traceback.format_exc()))
    finally:
        try:
            q.close()
        except Exception:
            pass


class ProcessJob:
    """A single heavy compute running in a child process.

    Mirrors the thread+queue pattern: construct, :meth:`start`, and the
    tagged messages land on ``out_queue`` (the tab's existing
    ``queue.Queue``) where ``drain_queue`` dispatches them on the main
    thread. ``running()`` replaces ``thread.is_alive()`` busy-guards;
    ``cancel()`` terminates the child (used by tabs with an Abort button).
    """

    def __init__(self, widget, out_queue, fn: Callable, args: Sequence = (),
                 kwargs: Optional[dict] = None, *,
                 done_kind: str = "done", error_kind: str = "error",
                 progress_kind: str = "progress",
                 aborted_kind: Optional[str] = None,
                 poll_ms: int = 80) -> None:
        self._widget = widget
        self._out = out_queue
        self._done_kind = done_kind
        self._error_kind = error_kind
        self._progress_kind = progress_kind
        self._aborted_kind = aborted_kind
        self._poll_ms = int(poll_ms)

        kwargs = dict(kwargs or {})
        self._q = _CTX.Queue()
        # The child uses the same queue for progress AND done/error; the
        # target reads ``progress_q`` to build its callback.
        kwargs["progress_q"] = self._q
        self._proc = _CTX.Process(
            target=_run_child,
            args=(self._q, fn, tuple(args), kwargs),
            daemon=True,
        )
        self._after_id = None
        self._terminal_seen = False
        self._cancelled = False
        self._dead_polls = 0

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "ProcessJob":
        self._proc.start()
        self._after_id = self._widget.after(self._poll_ms, self._tick)
        return self

    def running(self) -> bool:
        """True while the child is still working (busy-guard helper)."""
        try:
            return self._proc.is_alive() and not self._terminal_seen
        except Exception:
            return False

    def cancel(self) -> None:
        """Terminate the child. Safe to call when already finished."""
        self._cancelled = True
        try:
            if self._proc.is_alive():
                self._proc.terminate()
        except Exception:
            pass

    # -- main-thread polling ----------------------------------------------

    def _tick(self) -> None:
        # 1) Forward everything queued right now.
        try:
            while True:
                tag, payload = self._q.get_nowait()
                if tag == _PROGRESS:
                    self._out.put((self._progress_kind, payload))
                elif tag == _DONE:
                    self._terminal_seen = True
                    self._out.put((self._done_kind, payload))
                elif tag == _ERROR:
                    self._terminal_seen = True
                    self._out.put((self._error_kind, payload))
        except _queue.Empty:
            pass

        # 2) Terminal message delivered -> done.
        if self._terminal_seen:
            self._join()
            return

        # 3) Child gone but no terminal yet: allow a short grace for an
        #    in-transit message, then conclude (cancelled -> aborted,
        #    else an unexpected-exit error so the tab never hangs).
        if not self._proc.is_alive():
            self._dead_polls += 1
            if self._dead_polls < _DEAD_GRACE_POLLS:
                self._after_id = self._widget.after(self._poll_ms, self._tick)
                return
            self._terminal_seen = True
            if self._cancelled:
                if self._aborted_kind is not None:
                    self._out.put((self._aborted_kind, None))
            else:
                self._out.put(
                    (self._error_kind,
                     "Worker process exited unexpectedly (no result). "
                     "It may have run out of memory or crashed."))
            self._join()
            return

        # 4) Still running -> re-arm.
        self._after_id = self._widget.after(self._poll_ms, self._tick)

    def _join(self) -> None:
        try:
            self._proc.join(timeout=0)
        except Exception:
            pass


def run_offloaded(widget, out_queue, fn: Callable, args: Sequence = (),
                  kwargs: Optional[dict] = None, **opts) -> ProcessJob:
    """Construct, start, and return a :class:`ProcessJob`.

    Thin convenience so a tab call site reads like the old
    ``threading.Thread(target=worker, daemon=True).start()`` it replaces.
    ``opts`` are forwarded to :class:`ProcessJob` (``done_kind``,
    ``error_kind``, ``progress_kind``, ``aborted_kind``, ``poll_ms``).
    """
    return ProcessJob(widget, out_queue, fn, args=args, kwargs=kwargs,
                      **opts).start()
