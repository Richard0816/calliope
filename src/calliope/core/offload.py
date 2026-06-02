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
* By default :func:`run_offloaded` uses a reused **warm worker** (a single
  long-lived child process) so the heavy import cost (~3-4 s on Windows
  spawn) is paid once per session, not per run -- crucial for batches. It
  runs one job at a time; a job submitted while it's busy falls back to a
  fresh per-run process. See the "Warm reused worker" section below.
* ``ProcessJob`` is the per-run path (a fresh ``multiprocessing.Process``,
  ``spawn`` context). It is used as the warm-worker overflow fallback and
  for cancellable runs (``warm=False``) that need a live abort ``Event``
  inherited at spawn (e.g. cross-correlation). No ``ProcessPoolExecutor``
  -- that can't be handed a plain ``mp.Queue`` for live progress and goes
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

import atexit
import itertools
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


# ===========================================================================
# Warm reused worker
# ===========================================================================
# A single long-lived child process reused across runs so the heavy import
# cost (numpy/scipy/pandas/matplotlib, ~3-4 s on Windows spawn) is paid ONCE
# per session instead of per run -- the difference between a 134-recording
# batch spending ~30 min just re-importing and spending it once.
#
# Constraints that shape this design:
#   * A plain spawn ``Event``/``Queue`` can only be shared by INHERITANCE
#     (passed to ``Process`` at spawn), not pickled through a queue to an
#     already-running worker. So progress is delivered via a job-id-tagging
#     adapter the worker injects locally, and cancellable jobs (which need a
#     live abort ``Event``) keep using a per-run :class:`ProcessJob`
#     (``warm=False``) where the Event is inherited at spawn.
#   * The worker runs ONE job at a time. Offload jobs are serial in practice
#     (Tab 0's batch driver runs one stage at a time). If a second job is
#     submitted while the worker is busy, the caller falls back to a per-run
#     ``ProcessJob`` -- no internal scheduler/backlog to get wrong.
#   * If a job crashes the worker (OOM kill, segfault), the dispatcher reports
#     an error for that job and respawns a fresh worker for the next one.


class _TaggedPut:
    """``progress_q`` adapter handed to offload targets inside the warm
    worker. ``make_progress_cb`` calls ``put((_PROGRESS, payload))``; this
    re-tags it with the job id so the shared result queue can be
    demultiplexed by the parent. Constructed in the child, so it need not
    be picklable.
    """

    __slots__ = ("_q", "_jid")

    def __init__(self, result_q, job_id: int) -> None:
        self._q = result_q
        self._jid = job_id

    def put(self, item) -> None:
        try:
            tag, payload = item
        except Exception:
            tag, payload = _PROGRESS, item
        try:
            self._q.put((self._jid, tag, payload))
        except Exception:
            pass


def _warm_loop(task_q, result_q) -> None:
    """Warm-worker entry: run submitted jobs until a ``None`` sentinel.

    Each task is ``(job_id, fn, args, kwargs)``. The worker injects a
    job-id-tagging ``progress_q`` so the target's progress callback routes
    back over the shared ``result_q`` as ``(job_id, tag, payload)``.
    """
    while True:
        try:
            job = task_q.get()
        except (EOFError, OSError):
            break
        if job is None:
            break
        job_id, fn, args, kwargs = job
        kwargs = dict(kwargs)
        kwargs["progress_q"] = _TaggedPut(result_q, job_id)
        try:
            result = fn(*args, **kwargs)
            result_q.put((job_id, _DONE, result))
        except BaseException:  # noqa: BLE001 - report everything back
            result_q.put((job_id, _ERROR, traceback.format_exc()))


class _WarmJobHandle:
    """Returned by the warm pool; mirrors :class:`ProcessJob`'s public API
    (``running``/``cancel``) so tab busy-guards work the same either way."""

    def __init__(self) -> None:
        self._terminal = False

    def running(self) -> bool:
        return not self._terminal

    def cancel(self) -> None:
        # The warm worker can't be force-cancelled without killing the
        # shared process; cancellable tabs use ``warm=False`` + a
        # cooperative abort Event instead.
        pass


class _WarmPool:
    """Single-slot manager for the long-lived warm worker (main-thread only;
    Tk is single-threaded so no locking is needed on the parent side)."""

    def __init__(self) -> None:
        self._proc = None
        self._task_q = None
        self._result_q = None
        self._busy = False
        self._cur: Optional[dict] = None
        self._counter = itertools.count(1)
        self._armed = False
        self._poll_ms = 80

    def busy(self) -> bool:
        return self._busy

    def _ensure_worker(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return
        self._task_q = _CTX.Queue()
        self._result_q = _CTX.Queue()
        self._proc = _CTX.Process(
            target=_warm_loop, args=(self._task_q, self._result_q),
            daemon=True)
        self._proc.start()

    def submit(self, widget, out_queue, fn, args, kwargs, *,
               done_kind: str, error_kind: str, progress_kind: str,
               poll_ms: int) -> _WarmJobHandle:
        """Hand a job to the warm worker. Caller guarantees ``not busy()``."""
        self._ensure_worker()
        self._poll_ms = int(poll_ms)
        jid = next(self._counter)
        handle = _WarmJobHandle()
        root = widget.winfo_toplevel()
        self._busy = True
        self._cur = {"id": jid, "out": out_queue, "done": done_kind,
                     "error": error_kind, "progress": progress_kind,
                     "handle": handle}
        try:
            self._task_q.put((jid, fn, tuple(args), dict(kwargs or {})))
        except Exception as e:
            # e.g. an arg that can't be pickled through the task queue.
            self._busy = False
            self._cur = None
            handle._terminal = True
            out_queue.put((error_kind,
                           f"Could not submit job to warm worker: {e}"))
            return handle
        if not self._armed:
            self._armed = True
            root.after(self._poll_ms, lambda: self._pump(root))
        return handle

    def _pump(self, root) -> None:
        try:
            while True:
                jid, tag, payload = self._result_q.get_nowait()
                cur = self._cur
                if cur is None or jid != cur["id"]:
                    continue  # stale message from a finished/dead job
                if tag == _PROGRESS:
                    cur["out"].put((cur["progress"], payload))
                elif tag == _DONE:
                    cur["out"].put((cur["done"], payload))
                    cur["handle"]._terminal = True
                    self._busy = False
                    self._cur = None
                elif tag == _ERROR:
                    cur["out"].put((cur["error"], payload))
                    cur["handle"]._terminal = True
                    self._busy = False
                    self._cur = None
        except _queue.Empty:
            pass

        # Worker died mid-job (OOM kill / crash): report + respawn next time.
        if self._busy and (self._proc is None or not self._proc.is_alive()):
            cur = self._cur
            if cur is not None:
                cur["out"].put((cur["error"],
                                "Warm worker process died (likely out of "
                                "memory or a crash). It will be respawned "
                                "for the next run."))
                cur["handle"]._terminal = True
            self._busy = False
            self._cur = None
            self._proc = None  # _ensure_worker respawns on next submit

        if self._busy:
            root.after(self._poll_ms, lambda: self._pump(root))
        else:
            self._armed = False  # idle: stop polling until the next submit

    def shutdown(self) -> None:
        try:
            if self._task_q is not None:
                self._task_q.put(None)
        except Exception:
            pass
        try:
            if self._proc is not None and self._proc.is_alive():
                self._proc.join(timeout=0.5)
        except Exception:
            pass


_POOL = _WarmPool()
atexit.register(_POOL.shutdown)


def run_offloaded(widget, out_queue, fn: Callable, args: Sequence = (),
                  kwargs: Optional[dict] = None, *,
                  done_kind: str = "done", error_kind: str = "error",
                  progress_kind: str = "progress",
                  aborted_kind: Optional[str] = None,
                  poll_ms: int = 80, warm: bool = True):
    """Run ``fn`` in a child process; return a handle with ``running()``.

    Thin convenience so a tab call site reads like the old
    ``threading.Thread(target=worker, daemon=True).start()`` it replaces.

    By default the work goes to the reused **warm worker** (amortising the
    ~3-4 s spawn import cost across the session). Falls back to a fresh
    per-run :class:`ProcessJob` when the warm worker is busy. Pass
    ``warm=False`` for jobs that need a live abort ``Event`` (it must be
    inherited at spawn, so they require a dedicated process) -- those always
    use a :class:`ProcessJob`.
    """
    if warm and aborted_kind is None and not _POOL.busy():
        return _POOL.submit(
            widget, out_queue, fn, args, kwargs,
            done_kind=done_kind, error_kind=error_kind,
            progress_kind=progress_kind, poll_ms=poll_ms)
    return ProcessJob(
        widget, out_queue, fn, args=args, kwargs=kwargs,
        done_kind=done_kind, error_kind=error_kind,
        progress_kind=progress_kind, aborted_kind=aborted_kind,
        poll_ms=poll_ms).start()
