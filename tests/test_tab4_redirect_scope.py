"""Tab 4's log capture is scoped to its own worker thread.

Tab 4 used ``contextlib.redirect_stdout/stderr`` in its worker thread. Those
swap the process-wide ``sys.stdout``/``sys.stderr``, so during detection every
print from any other GUI thread (the batch runner's status lines, the Tk main
thread) went into Tab 4's log instead of the console (bug 8 run 8: the driver's
status lines vanished for the whole Suite2p window).

Calls the real unbound ``Suite2pTab._run_pipeline`` with a stub ``self`` and a
fake ``call_in_child``, as test_suite2p_tab_wiring.py does; no Tk root needed.
"""
from __future__ import annotations

import queue
import sys
import threading
import types
from types import SimpleNamespace

import pytest

from calliope.gui_common import spec_defaults
from calliope.tabs.suite2p.tab import Suite2pTab


class _Stop(Exception):
    pass


def _drain(q):
    lines = []
    while True:
        try:
            tag, payload = q.get_nowait()
        except queue.Empty:
            return lines
        if tag == "log":
            lines.append(payload)


def _run_tab4_in_worker(monkeypatch, tmp_path, during_child):
    """Run _run_pipeline on a worker thread; ``during_child(on_progress)`` runs
    while the fake child is "running". Returns Tab 4's log lines."""
    fake_spc = types.ModuleType("calliope.core.sparse_plus_cellpose")
    fake_spc.run = fake_spc.run_offload = lambda **kw: None
    import calliope.core as core_pkg
    from calliope.core import offload

    def fake_call_in_child(fn, kwargs, *a, on_progress=None, **k):
        during_child(on_progress)
        raise _Stop

    monkeypatch.setattr(offload, "call_in_child", fake_call_in_child)
    monkeypatch.setitem(sys.modules, "calliope.core.sparse_plus_cellpose", fake_spc)
    monkeypatch.setattr(core_pkg, "sparse_plus_cellpose", fake_spc, raising=False)
    stub = SimpleNamespace(_params=spec_defaults(Suite2pTab.PARAM_SPEC),
                           _settings_override={}, _log_queue=queue.Queue())
    err = []

    def worker():
        try:
            Suite2pTab._run_pipeline(
                stub, tiff_folder=tmp_path, save_folder=tmp_path / "det",
                ckpt_path=None, rec_id="rec", baseline_mode="first_n",
                baseline_min=2.0, tau_override=0.25)
        except BaseException as e:  # noqa: BLE001 - reported below
            err.append(e)

    t = threading.Thread(target=worker)
    t.start()
    t.join(30)
    assert not t.is_alive()
    assert len(err) == 1 and isinstance(err[0], _Stop), err
    return _drain(stub._log_queue)


def test_other_thread_prints_stay_out_of_tab4_log(monkeypatch, tmp_path, capsys):
    def during_child(on_progress):
        def other():
            print("OTHER-THREAD-STDOUT")
            print("OTHER-THREAD-STDERR", file=sys.stderr)
        o = threading.Thread(target=other)
        o.start()
        o.join()
        on_progress("CHILD-PROGRESS-LINE")
        print("WORKER-THREAD-PRINT")

    lines = _run_tab4_in_worker(monkeypatch, tmp_path, during_child)
    # Tab 4's own output still reaches its log ...
    assert any("running sparse_plus_cellpose" in ln for ln in lines), lines
    assert "CHILD-PROGRESS-LINE" in lines
    assert "WORKER-THREAD-PRINT" in lines
    # ... but another thread's prints do not, and still reach the console.
    assert not any("OTHER-THREAD" in ln for ln in lines), lines
    out, errout = capsys.readouterr()
    assert "OTHER-THREAD-STDOUT" in out
    assert "OTHER-THREAD-STDERR" in errout


def test_concurrent_and_nested_routes_restore_cleanly(capsys):
    from calliope.gui_common import QueueWriter, redirect_thread_output
    qa, qb, qn = queue.Queue(), queue.Queue(), queue.Queue()
    before = (sys.stdout, sys.stderr)
    a_in, b_done = threading.Event(), threading.Event()

    def a():
        with redirect_thread_output(QueueWriter(qa)):
            a_in.set()
            b_done.wait(10)      # B enters and leaves while A is active
            print("A1")
            with redirect_thread_output(QueueWriter(qn)):
                print("NESTED")
            print("A2", file=sys.stderr)

    def b():
        a_in.wait(10)
        with redirect_thread_output(QueueWriter(qb)):
            print("B1")
        b_done.set()

    ta, tb = threading.Thread(target=a), threading.Thread(target=b)
    ta.start(); tb.start(); ta.join(10); tb.join(10)
    print("MAIN")
    assert _drain(qa) == ["A1", "A2"]
    assert _drain(qb) == ["B1"]
    assert _drain(qn) == ["NESTED"]
    assert (sys.stdout, sys.stderr) == before
    assert "MAIN" in capsys.readouterr().out


def test_no_console_pythonw_other_threads_stay_silent(monkeypatch):
    # Under pythonw sys.stdout/stderr are None and print() is a no-op; the
    # router must keep it a no-op for other threads instead of raising.
    from calliope.gui_common import QueueWriter, redirect_thread_output
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    q, errs, inside, done = queue.Queue(), [], threading.Event(), threading.Event()

    def worker():
        with redirect_thread_output(QueueWriter(q)):
            inside.set()
            done.wait(10)
            print("WORKER")

    def other():
        inside.wait(10)
        try:
            print("OTHER")
            print("OTHER", file=sys.stderr)
            sys.stdout.flush()
        except Exception as e:  # noqa: BLE001 - the failure under test
            errs.append(e)
        done.set()

    ts = [threading.Thread(target=worker), threading.Thread(target=other)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(10)
    assert errs == []
    assert _drain(q) == ["WORKER"]
    assert sys.stdout is None and sys.stderr is None


def test_other_threads_keep_the_stream_interface(monkeypatch):
    # While a route is active, unrelated threads still see their real
    # stream's interface (buffer, encoding), not a bare TextIOBase.
    import io as _io
    from calliope.gui_common import QueueWriter, redirect_thread_output
    raw = _io.BytesIO()
    real = _io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
    monkeypatch.setattr(sys, "stdout", real)
    seen, inside, done = {}, threading.Event(), threading.Event()

    def worker():
        with redirect_thread_output(QueueWriter(queue.Queue())):
            inside.set()
            done.wait(10)

    def other():
        inside.wait(10)
        seen["enc"] = sys.stdout.encoding
        seen["err"] = sys.stdout.errors
        seen["writable"] = sys.stdout.writable()
        sys.stdout.buffer.write(b"RAW-BYTES")
        sys.stdout.flush()
        done.set()

    ts = [threading.Thread(target=worker), threading.Thread(target=other)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(10)
    assert seen == {"enc": "utf-8", "err": "replace", "writable": True}
    assert b"RAW-BYTES" in raw.getvalue()
    assert sys.stdout is real


def test_streams_restored_after_run(monkeypatch, tmp_path):
    before = (sys.stdout, sys.stderr)
    _run_tab4_in_worker(monkeypatch, tmp_path, lambda on_progress: None)
    print("AFTER-RUN")  # must not raise or vanish into a dead writer
    assert (sys.stdout, sys.stderr) == before
