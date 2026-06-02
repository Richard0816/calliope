"""Low-data edge-case guards across the pipeline.

Covers the audit-driven hardening:
  * AppState stage-error channel + ``report_stage_error`` helper.
  * The batch driver's ``_arm_completion`` failing-and-continuing (instead
    of stalling) when a stage worker errors before its ready signal.
  * The clustering linkage family rejecting <2 ROIs / degenerate trees
    with a clear message instead of scipy's "empty distance matrix".
  * ROI-count resolution + the <2-ROI clustering skip in both batch
    runners.

All exercised against lightweight stubs / temp dirs -- no Tk, no suite2p.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from calliope.gui_common import AppState, report_stage_error
from calliope.core import clustering as core_clustering
from calliope.core import batch_pipeline
from calliope.tabs.batch.tab import BatchTab


# --- AppState stage-error channel + report_stage_error --------------------

def test_appstate_stage_error_notifies_subscribers():
    state = AppState()
    seen = []
    state.subscribe_stage_error(seen.append)
    state.set_stage_error("boom")
    assert seen == ["boom"]
    assert state.stage_error == "boom"


def test_report_stage_error_publishes_and_reports_batch_flag():
    state = AppState()
    seen = []
    state.subscribe_stage_error(seen.append)

    # Interactive: publishes, but tells caller to show its own dialog.
    assert report_stage_error(state, "err1") is False
    # Batch active: publishes AND tells caller to suppress its modal.
    state.batch_active = True
    assert report_stage_error(state, "err2") is True
    assert seen == ["err1", "err2"]

    # None state is tolerated (returns False, no raise).
    assert report_stage_error(None, "err3") is False


# --- batch _arm_completion: worker error fails-and-continues --------------

class _ArmStub:
    def __init__(self, state):
        self.state = state
        self._current_stage = "lowpass"
        self.done = []
        self.failed = []

    def after(self, delay, fn):
        fn()  # run inline

    def _on_stage_done(self, stage, hint):
        self.done.append((stage, hint))

    def _fail_stage(self, stage, exc):
        self.failed.append((stage, str(exc)))


def test_arm_completion_fails_stage_on_worker_error():
    state = AppState()
    stub = _ArmStub(state)
    BatchTab._arm_completion(stub, state.subscribe_lowpass_ready, "lowpass",
                             path_extractor=lambda p: p)
    # Worker errors before publishing its ready signal.
    state.set_stage_error("kaboom\ntraceback...")
    assert stub.failed == [("lowpass", "kaboom")]
    assert stub.done == []
    # A late ready publish is ignored (shared one-shot guard).
    state.set_lowpass_ready("/tmp/plane0")
    assert stub.done == []


def test_arm_completion_ready_wins_over_late_error():
    state = AppState()
    stub = _ArmStub(state)
    BatchTab._arm_completion(stub, state.subscribe_lowpass_ready, "lowpass",
                             path_extractor=lambda p: p)
    state.set_lowpass_ready("/tmp/plane0")
    assert len(stub.done) == 1 and stub.done[0][0] == "lowpass"
    # A later error for the same stage is ignored.
    state.set_stage_error("too late")
    assert stub.failed == []


def test_arm_completion_ignores_error_for_other_stage():
    state = AppState()
    stub = _ArmStub(state)
    stub._current_stage = "detection"   # we armed lowpass, but moved on
    BatchTab._arm_completion(stub, state.subscribe_lowpass_ready, "lowpass",
                             path_extractor=lambda p: p)
    state.set_stage_error("stale")
    assert stub.failed == []


# --- clustering linkage guards --------------------------------------------

def test_ward_linkage_rejects_single_roi():
    dff = np.random.RandomState(0).rand(100, 1).astype(np.float32)  # (T, 1)
    with pytest.raises(ValueError, match="at least 2 ROIs"):
        core_clustering._ward_linkage(dff)


def test_ward_linkage_works_with_two_rois():
    dff = np.random.RandomState(1).rand(100, 2).astype(np.float32)
    Z = core_clustering._ward_linkage(dff)
    assert Z.shape == (1, 4)  # one merge for two leaves


def test_gui_ward_linkage_rejects_single_roi():
    from calliope.tabs.clustering import logic as cl_logic
    dff = np.random.RandomState(2).rand(50, 1).astype(np.float32)
    with pytest.raises(ValueError, match="at least 2 ROIs"):
        cl_logic.ward_linkage(dff)


def test_count_clusters_degenerate_tree_returns_one():
    assert core_clustering.count_clusters(np.empty((0, 4)), 0.5) == 1
    assert core_clustering.count_clusters(None, 0.5) == 1


# --- ROI-count helpers + <2-ROI skip --------------------------------------

def _write_mask(plane0, keep, total=5):
    m = np.zeros(total, dtype=bool)
    m[:keep] = True
    np.save(plane0 / "predicted_cell_mask.npy", m)


def test_batch_pipeline_kept_roi_count(tmp_path):
    _write_mask(tmp_path, keep=1)
    assert batch_pipeline._kept_roi_count(tmp_path) == 1
    _write_mask(tmp_path, keep=3)
    assert batch_pipeline._kept_roi_count(tmp_path) == 3


class _SkipStub:
    STAGES = BatchTab.STAGES

    def __init__(self):
        self.logs = []
        self._row_results = {"status": "running", "stages": {}}
        self.begun = []

    def _append_log(self, m):
        self.logs.append(m)

    def _begin_stage(self, stage):
        self.begun.append(stage)


def test_batch_kept_roi_count_helper(tmp_path):
    _write_mask(tmp_path, keep=1)
    assert BatchTab._kept_roi_count(_SkipStub(), tmp_path) == 1


def test_skip_clustering_xcorr_few_rois_marks_and_jumps_to_spatial():
    stub = _SkipStub()
    BatchTab._skip_clustering_xcorr_few_rois(stub, 1)
    assert stub._row_results["stages"]["clustering"]["status"] == "skipped"
    assert stub._row_results["stages"]["crosscorrelation"]["status"] == "skipped"
    # Continues to spatial so the row still finishes with event/spatial output.
    assert stub.begun == ["spatial_propagation"]
    # Event detection (which ran) is NOT marked skipped.
    assert "event_detection" not in stub._row_results["stages"]
