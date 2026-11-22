"""Tracker #33 item 5: only one process at a time may rebuild a recording's
ROIs / dF/F, and finishing one build never marks another complete."""
from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pytest

from calliope.core import detection_run as dr
from calliope.core import generation

CTX = mp.get_context("spawn")


def _hold_build(plane0, ready, release, finish):
    from calliope.core import generation as g
    b = g.new_dff_build(plane0)
    ready.set()
    release.wait(60)
    if finish:
        g.finish_dff_build(plane0, b)


def _try_build(plane0, q):
    from calliope.core import generation as g
    try:
        g.finish_dff_build(plane0, g.new_dff_build(plane0))
        q.put("built")
    except g.DffBuildBusy:
        q.put("busy")


def _child(target, *args):
    p = CTX.Process(target=target, args=args)
    p.start()
    return p


def test_second_builder_is_refused_while_the_first_runs(tmp_path):
    ready, release = CTX.Event(), CTX.Event()
    a = _child(_hold_build, tmp_path, ready, release, True)
    try:
        assert ready.wait(60)
        # this process wants to rebuild the same recording meanwhile
        with pytest.raises(generation.DffBuildBusy, match="another process"):
            generation.new_dff_build(tmp_path)
        # A's pending build is still the one on disk, untouched
        with pytest.raises(generation.StaleOutputError, match="did not finish"):
            generation.dff_build_id(tmp_path)
    finally:
        release.set()
        a.join(60)
    b = generation.new_dff_build(tmp_path)        # A done: now allowed
    generation.finish_dff_build(tmp_path, b)
    assert generation.dff_build_id(tmp_path) == b


def test_a_crashed_builder_frees_the_recording(tmp_path):
    ready, release = CTX.Event(), CTX.Event()
    a = _child(_hold_build, tmp_path, ready, release, False)
    assert ready.wait(60)
    a.terminate()                                  # dies mid-build
    a.join(60)
    b = generation.new_dff_build(tmp_path)         # lock freed by the OS
    generation.finish_dff_build(tmp_path, b)
    assert generation.dff_build_id(tmp_path) == b


def test_finish_and_abort_release_the_recording(tmp_path):
    q = CTX.Queue()
    a = generation.new_dff_build(tmp_path)
    generation.new_dff_build(tmp_path)             # re-entrant in-process
    p = _child(_try_build, tmp_path, q)
    p.join(60)
    assert q.get(timeout=10) == "busy"
    generation.abort_dff_build(tmp_path)           # failed build
    p = _child(_try_build, tmp_path, q)
    p.join(60)
    assert q.get(timeout=10) == "built"
    generation.finish_dff_build(tmp_path, a)       # stale id: no-op, no hang


def test_failed_dff_build_releases_the_recording(tmp_path, monkeypatch):
    F = np.full((2, 100), 300, np.float32)
    np.save(tmp_path / "F.npy", F)
    np.save(tmp_path / "Fneu.npy", np.zeros_like(F))

    def boom(*a, **k):
        raise MemoryError("out of RAM")

    monkeypatch.setattr(dr, "_build_dff", boom)
    with pytest.raises(MemoryError):
        dr.compute_dff_memmaps(tmp_path, {"fps_override": 10.0,
                                          "use_gpu_dff": False},
                               baseline_mode="first_n", baseline_min=0.1)
    with pytest.raises(generation.StaleOutputError, match="did not finish"):
        generation.dff_build_id(tmp_path)          # downstream refused
    q = CTX.Queue()
    p = _child(_try_build, tmp_path, q)
    p.join(60)
    assert q.get(timeout=10) == "built"            # others may rebuild


def test_finish_marks_only_its_own_build_complete(tmp_path):
    a = generation.new_dff_build(tmp_path)
    generation.finish_dff_build(tmp_path, a)
    assert generation.dff_build_id(tmp_path) == a
    b = generation.new_dff_build(tmp_path)
    generation.finish_dff_build(tmp_path, a)        # stale id: no effect
    with pytest.raises(generation.StaleOutputError):
        generation.dff_build_id(tmp_path)


def test_finished_build_frees_the_recording_for_other_processes(tmp_path):
    # e.g. the GUI finishes a rebuild and stays open; a batch run in
    # another process must then be able to rebuild the same recording.
    q = CTX.Queue()
    a = generation.new_dff_build(tmp_path)
    generation.finish_dff_build(tmp_path, a)
    p = _child(_try_build, tmp_path, q)
    p.join(60)
    assert q.get(timeout=10) == "built"


def _busy_from_child(plane0):
    q = CTX.Queue()
    p = _child(_try_build, plane0, q)
    p.join(60)
    return q.get(timeout=10)


def test_stale_finish_keeps_the_live_build_claimed(tmp_path):
    a = generation.new_dff_build(tmp_path)
    b = generation.new_dff_build(tmp_path)        # supersedes a
    generation.finish_dff_build(tmp_path, a)       # stale id
    assert _busy_from_child(tmp_path) == "busy"    # b still being built
    generation.finish_dff_build(tmp_path, b)
    assert _busy_from_child(tmp_path) == "built"


def test_merge_then_dff_in_one_process_frees_the_recording(tmp_path):
    F = np.full((2, 100), 300, np.float32)
    np.save(tmp_path / "F.npy", F)
    np.save(tmp_path / "Fneu.npy", np.zeros_like(F))
    generation.new_dff_build(tmp_path)              # ROI merge (batch)
    dr.compute_dff_memmaps(tmp_path, {"fps_override": 10.0,
                                      "use_gpu_dff": False},
                           baseline_mode="first_n", baseline_min=0.1)
    assert generation.dff_build_id(tmp_path)        # complete
    assert _busy_from_child(tmp_path) == "built"


def test_failed_pending_stamp_does_not_leak_the_claim(tmp_path, monkeypatch):
    def full_disk(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(generation, "_write", full_disk)
    with pytest.raises(OSError):
        generation.new_dff_build(tmp_path)
    monkeypatch.undo()
    assert _busy_from_child(tmp_path) == "built"


def test_dff_build_claims_before_reading_its_inputs(tmp_path):
    # While another process holds the recording, compute_dff_memmaps must
    # refuse before touching F/Fneu (here they don't even exist yet).
    ready, release = CTX.Event(), CTX.Event()
    a = _child(_hold_build, tmp_path, ready, release, True)
    try:
        assert ready.wait(60)
        with pytest.raises(generation.DffBuildBusy):
            dr.compute_dff_memmaps(tmp_path, {"fps_override": 10.0})
    finally:
        release.set()
        a.join(60)
