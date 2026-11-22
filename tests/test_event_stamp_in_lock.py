"""Tracker #33 item 4: the Tab 5 event stamp is committed under the same
workbook lock as the sheets it describes, and readers (NWB export, Tab 7)
check it and read the sheets under that lock too."""
from __future__ import annotations

import contextlib
import multiprocessing as mp
import time

import numpy as np
import pytest

from calliope.core import detection_run, generation, summary_writer

from test_nwb_cell_columns import N, T, _plane0 as _raw_plane0  # noqa: E402


def _plane0(tmp_path):
    p = _raw_plane0(tmp_path)
    mm = np.memmap(p / "r0p7_dff.memmap.float32", np.float32, "w+", shape=(T, N))
    mm[:] = 0.5
    mm.flush()
    del mm
    generation.finish_dff_build(p, generation.new_dff_build(p))
    detection_run.compute_filtered_dff(p)
    generation.stamp_lowpass(p, generation.check_filtered(p))
    return p


def _write(p, windows=((0.5, 1.5),)):
    return summary_writer.write_events_sheets(
        p, event_windows=list(windows), onsets_by_roi=[np.array([1.0])] * N,
        roi_ids=list(range(N)), generation=generation.check_lowpass(p))


def test_stamp_is_written_inside_the_workbook_lock(tmp_path, monkeypatch):
    p = _plane0(tmp_path)
    seq = []
    real_lock = summary_writer._workbook_lock

    @contextlib.contextmanager
    def spy_lock(path, timeout_s=300.0):
        with real_lock(path, timeout_s):
            seq.append("lock")
            yield
            seq.append("unlock")

    real_begin, real_stamp = generation.begin_events, generation.stamp_events
    monkeypatch.setattr(summary_writer, "_workbook_lock", spy_lock)
    monkeypatch.setattr(generation, "begin_events",
                        lambda q: (seq.append("pending"), real_begin(q)))
    monkeypatch.setattr(generation, "stamp_events",
                        lambda q, t: (seq.append("stamp"), real_stamp(q, t)))
    _write(p)
    assert seq == ["lock", "pending", "stamp", "unlock"]
    assert generation.check_events(p) == generation.check_lowpass(p)


def test_unfinished_event_write_is_refused(tmp_path):
    p = _plane0(tmp_path)
    _write(p)
    generation.begin_events(p)          # a writer died mid-write
    with pytest.raises(generation.StaleOutputError, match="did not finish"):
        generation.check_events(p)
    _write(p)                           # re-running Tab 5 repairs it
    generation.check_events(p)


def test_failed_sheet_write_leaves_the_stamp_pending(tmp_path, monkeypatch):
    p = _plane0(tmp_path)
    _write(p)

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(summary_writer.pd.DataFrame, "to_excel", boom)
    with pytest.raises(OSError):
        _write(p, windows=((2.0, 3.0),))
    with pytest.raises(generation.StaleOutputError, match="did not finish"):
        generation.check_events(p)


def test_nwb_checks_and_reads_events_under_the_lock(tmp_path, monkeypatch):
    nwb_export = pytest.importorskip("calliope.core.nwb_export")
    if not nwb_export.nwb_available():
        pytest.skip("pynwb not installed")
    p = _plane0(tmp_path)
    _write(p)
    held = []
    real_locked = summary_writer.locked

    @contextlib.contextmanager
    def spy_locked(plane0, *a, **k):
        with real_locked(plane0, *a, **k) as path:
            held.append(True)
            yield path
            held.append(False)

    real_check, real_read = generation.check_events, nwb_export._read_event_windows
    calls = []
    monkeypatch.setattr(summary_writer, "locked", spy_locked)
    monkeypatch.setattr(generation, "check_events",
                        lambda q: (calls.append(("check", held[-1:])), real_check(q))[1])
    monkeypatch.setattr(nwb_export, "_read_event_windows",
                        lambda q: (calls.append(("read", held[-1:])), real_read(q))[1])
    nwb_export.export_recording_to_nwb(p, tmp_path / "a.nwb")
    assert calls == [("read", [True]), ("check", [True])]


def test_nwb_reports_a_lock_timeout(tmp_path, monkeypatch):
    nwb_export = pytest.importorskip("calliope.core.nwb_export")
    if not nwb_export.nwb_available():
        pytest.skip("pynwb not installed")
    p = _plane0(tmp_path)
    _write(p)

    @contextlib.contextmanager
    def stuck(*a, **k):
        raise TimeoutError("locked")
        yield  # pragma: no cover

    monkeypatch.setattr(summary_writer, "locked", stuck)
    notes = []
    nwb_export.export_recording_to_nwb(p, tmp_path / "a.nwb", notes=notes)
    assert notes and "stayed locked" in notes[0]


def _hold_lock(path, ready, release):
    from calliope.core import summary_writer as sw
    with sw._workbook_lock(path):
        ready.set()
        release.wait(30)


def test_reader_waits_for_a_writer_in_another_process(tmp_path):
    # A writer holding the lock (another process, as Tab 5 is) blocks a
    # reader until it releases; a short reader timeout gives TimeoutError.
    p = _plane0(tmp_path)
    _write(p)
    ctx = mp.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    proc = ctx.Process(target=_hold_lock,
                       args=(summary_writer.summary_path(p), ready, release))
    proc.start()
    try:
        assert ready.wait(60)
        t0 = time.monotonic()
        with pytest.raises(TimeoutError):
            with summary_writer.locked(p, timeout_s=0.5):
                pass
        assert time.monotonic() - t0 >= 0.5
    finally:
        release.set()
        proc.join(30)
    with summary_writer.locked(p, timeout_s=5):
        generation.check_events(p)


def test_tab7_cold_start_reads_events_under_the_lock(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from calliope.tabs.crosscorrelation import tab as t7
    p = _plane0(tmp_path)
    _write(p)
    held, seen = [], {}
    real_locked = summary_writer.locked

    @contextlib.contextmanager
    def spy_locked(plane0, *a, **k):
        with real_locked(plane0, *a, **k) as path:
            held.append(True)
            yield path
            held.append(False)

    real_read = t7._read_event_windows_from_summary
    monkeypatch.setattr(summary_writer, "locked", spy_locked)
    monkeypatch.setattr(t7, "_read_event_windows_from_summary",
                        lambda q: (seen.setdefault("held", held[-1:]), real_read(q))[1])
    me = SimpleNamespace(_plane0=p, state=None,
                         status_var=SimpleNamespace(set=lambda *a: None),
                         _event_windows_from_state=lambda cur: None,
                         _apply_event_windows=lambda ev, token=None: seen.update(ev=ev))
    me._events_status = lambda m: (setattr(me, "_events_note", m), me.status_var.set(m))
    t7.CrossCorrelationTab._on_refresh_events(me)
    assert seen["held"] == [True] and seen["ev"] == [(0.5, 1.5)]


def test_two_writers_interleaving_keeps_sheets_and_stamp_together(tmp_path, monkeypatch):
    # The reported schedule: B (current low-pass) writes and pauses inside
    # its commit; A (older events) tries to write meanwhile. A must wait, so
    # the workbook and stamp on disk always belong to the same writer.
    import threading
    p = _plane0(tmp_path)
    cur = generation.check_lowpass(p)
    in_b, go_b = threading.Event(), threading.Event()
    real_stamp = generation.stamp_events

    def slow_stamp(q, token):
        if token == cur and not go_b.is_set():
            in_b.set()
            go_b.wait(30)
        return real_stamp(q, token)

    monkeypatch.setattr(generation, "stamp_events", slow_stamp)

    def write(windows, token):
        summary_writer.write_events_sheets(
            p, event_windows=windows, onsets_by_roi=[np.array([1.0])] * N,
            roi_ids=list(range(N)), generation=token)

    b = threading.Thread(target=write, args=([(0.5, 1.5)], cur))
    a = threading.Thread(target=write, args=([(9.0, 9.5)], "older-lowpass"))
    b.start()
    assert in_b.wait(30)
    a.start()
    time.sleep(1.0)
    assert a.is_alive()                 # A is held off while B commits
    go_b.set()
    b.join(30)
    a.join(30)
    # A wrote last: its sheets and its (stale) stamp together, so the
    # workbook is refused; never A's sheets under B's stamp.
    import pandas as pd
    w = pd.read_excel(summary_writer.summary_path(p), "EventWindows")
    assert list(w["start_s"]) == [9.0]
    with pytest.raises(generation.StaleOutputError):
        generation.check_events(p)


def test_read_only_folder_reads_without_the_lock(tmp_path, monkeypatch):
    import builtins
    import errno
    p = _plane0(tmp_path)
    _write(p)

    def no_lock_file(path, *a, **k):
        if str(path).endswith(".lock"):
            raise PermissionError(errno.EACCES, "read-only", str(path))
        return builtins.open(path, *a, **k)

    monkeypatch.setattr(summary_writer, "open", no_lock_file, raising=False)
    with summary_writer.locked(p) as held:   # readers carry on, flagged
        assert held is False
        generation.check_events(p)
    with pytest.raises(PermissionError):  # writers still fail loudly
        _write(p)


def test_tab7_ignores_other_summary_workbooks(tmp_path):
    from types import SimpleNamespace
    from calliope.tabs.crosscorrelation import tab as t7
    p = _plane0(tmp_path)
    _write(p, windows=())               # default workbook: no events
    summary_writer.write_events_sheets(
        p, event_windows=[(3.0, 4.0)], onsets_by_roi=[np.array([3.5])] * N,
        roi_ids=list(range(N)), filename="old_summary.xlsx")
    assert t7._read_event_windows_from_summary(p) == []
    status = []
    me = SimpleNamespace(_plane0=p, state=None,
                         status_var=SimpleNamespace(set=status.append),
                         _event_windows_from_state=lambda cur: None,
                         _apply_event_windows=lambda ev, token=None: None)
    me._events_status = lambda m: (setattr(me, "_events_note", m), me.status_var.set(m))
    t7.CrossCorrelationTab._on_refresh_events(me)
    assert "old_summary.xlsx not used" in status[-1]


def test_tab7_waits_briefly_then_asks_for_a_refresh(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from calliope.tabs.crosscorrelation import tab as t7
    p = _plane0(tmp_path)
    _write(p)
    asked = {}

    @contextlib.contextmanager
    def busy(plane0, filename=None, timeout_s=30.0):
        asked["timeout"] = timeout_s
        raise TimeoutError("locked")
        yield  # pragma: no cover

    monkeypatch.setattr(summary_writer, "locked", busy)
    status, applied = [], []
    me = SimpleNamespace(_plane0=p, state=None,
                         status_var=SimpleNamespace(set=status.append),
                         _event_windows_from_state=lambda cur: None,
                         _apply_event_windows=lambda ev, token=None: applied.append(ev))
    me._events_status = lambda m: (setattr(me, "_events_note", m), me.status_var.set(m))
    t7.CrossCorrelationTab._on_refresh_events(me)
    assert asked["timeout"] <= 2.0
    assert applied == [[]] and "refresh again" in status[-1]


def test_nwb_flags_events_read_without_the_lock(tmp_path, monkeypatch):
    import builtins
    import errno
    nwb_export = pytest.importorskip("calliope.core.nwb_export")
    if not nwb_export.nwb_available():
        pytest.skip("pynwb not installed")
    p = _plane0(tmp_path)
    _write(p)

    def no_lock_file(path, *a, **k):
        if str(path).endswith(".lock"):
            raise PermissionError(errno.EACCES, "read-only", str(path))
        return builtins.open(path, *a, **k)

    monkeypatch.setattr(summary_writer, "open", no_lock_file, raising=False)
    notes = []
    nwb_export.export_recording_to_nwb(p, tmp_path / "a.nwb", notes=notes)
    assert len(notes) == 1 and "without the workbook lock" in notes[0]


def test_selecting_a_folder_keeps_the_event_message(tmp_path, monkeypatch):
    # _set_plane0 refreshes events, then sets "Ready." - the event-loading
    # message (here: alternate workbook skipped) must still be visible.
    from types import SimpleNamespace
    from calliope.tabs.crosscorrelation import tab as t7
    p = _plane0(tmp_path)
    _write(p, windows=())
    summary_writer.write_events_sheets(
        p, event_windows=[(3.0, 4.0)], onsets_by_roi=[np.array([3.5])] * N,
        roi_ids=list(range(N)), filename="old_summary.xlsx")
    status = []
    btn = SimpleNamespace(configure=lambda **k: None)
    var = lambda v: SimpleNamespace(get=lambda: v, set=lambda *a: None)
    me = SimpleNamespace(
        state=None, status_var=SimpleNamespace(set=status.append),
        fps_var=var(""), prefix_var=var("r0p7_filtered_"), cfolder_var=var(""),
        full_btn=btn, per_event_btn=btn, sp_plot_btn=btn, violin_btn=btn,
        export_figs_btn=btn, _event_windows=[],
        _event_windows_from_state=lambda cur: None,
        _apply_event_windows=lambda ev, token=None: None,
        _inputs_ready=lambda: True)
    me._events_status = lambda m: t7.CrossCorrelationTab._events_status(me, m)
    me._on_refresh_events = lambda: t7.CrossCorrelationTab._on_refresh_events(me)
    t7.CrossCorrelationTab._set_plane0(me, p)
    assert status[-1].startswith("Ready.") and "old_summary.xlsx not used" in status[-1]
