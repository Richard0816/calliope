"""Tests for the zero-ROI ("genuinely noise") skip path in the batch
runners.

A recording that is just noise can finish detection with no ROIs
surviving the cell filter. ``detection_run.compute_filtered_dff`` then
writes no ``r0p7_filtered_dff.memmap.float32`` and the lowpass stage --
plus everything downstream -- has nothing to read. Both batch runners
must detect that and skip the remaining stages instead of stalling /
raising at the lowpass stage:

  * the headless orchestrator ``core.batch_pipeline._no_rois_survive``
  * the GUI driver ``tabs.batch.tab.BatchTab._detection_has_no_rois``
    / ``_skip_row_no_rois`` (exercised unbound against a stub, no Tk).
"""

from __future__ import annotations

import numpy as np

from calliope.core import batch_pipeline
from calliope.tabs.batch.tab import BatchTab


# --- shared plane0 builders ------------------------------------------------

def _write_filtered_memmap(plane0):
    p = plane0 / "r0p7_filtered_dff.memmap.float32"
    np.memmap(str(p), dtype="float32", mode="w+", shape=(4, 2)).flush()


def _write_mask(plane0, keep: int, total: int = 3):
    m = np.zeros(total, dtype=bool)
    m[:keep] = True
    np.save(plane0 / "predicted_cell_mask.npy", m)


def _write_iscell(plane0, keep: int, total: int = 3):
    ic = np.zeros((total, 2), dtype=np.float32)
    ic[:keep, 0] = 1.0
    np.save(plane0 / "iscell.npy", ic)


# --- batch_pipeline._no_rois_survive --------------------------------------

def test_no_rois_survive_false_when_filtered_memmap_present(tmp_path):
    _write_filtered_memmap(tmp_path)
    # Even with an all-False mask on disk, the memmap's presence wins:
    # it's only ever written when N_kept > 0.
    _write_mask(tmp_path, keep=0)
    assert batch_pipeline._no_rois_survive(tmp_path) is False


def test_no_rois_survive_true_when_mask_empty_and_no_memmap(tmp_path):
    _write_mask(tmp_path, keep=0)
    assert batch_pipeline._no_rois_survive(tmp_path) is True


def test_no_rois_survive_false_when_mask_has_kept(tmp_path):
    _write_mask(tmp_path, keep=2)
    assert batch_pipeline._no_rois_survive(tmp_path) is False


def test_no_rois_survive_iscell_fallback(tmp_path):
    _write_iscell(tmp_path, keep=0)
    assert batch_pipeline._no_rois_survive(tmp_path) is True
    # Inverse: at least one cell -> not noise.
    np.save(tmp_path / "iscell.npy",
            np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32))
    assert batch_pipeline._no_rois_survive(tmp_path) is False


def test_no_rois_survive_true_when_nothing_on_disk(tmp_path):
    # No filtered memmap, no mask files at all -> nothing for lowpass.
    assert batch_pipeline._no_rois_survive(tmp_path) is True


# --- BatchTab._detection_has_no_rois (unbound) ----------------------------

class _Stub:
    """Minimal stand-in exposing only what the methods read/write."""

    STAGES = BatchTab.STAGES

    def __init__(self):
        self.logs: list[str] = []
        self._row_results: dict = {"status": "running", "stages": {}}
        self._row_done_called = False

    def _append_log(self, msg):
        self.logs.append(msg)

    def _on_row_done(self):
        self._row_done_called = True


def test_detection_has_no_rois_none_plane0():
    assert BatchTab._detection_has_no_rois(_Stub(), None) is False


def test_detection_has_no_rois_memmap_present(tmp_path):
    _write_filtered_memmap(tmp_path)
    assert BatchTab._detection_has_no_rois(_Stub(), tmp_path) is False


def test_detection_has_no_rois_empty_mask(tmp_path):
    _write_mask(tmp_path, keep=0)
    assert BatchTab._detection_has_no_rois(_Stub(), tmp_path) is True


def test_detection_has_no_rois_kept_mask(tmp_path):
    _write_mask(tmp_path, keep=1)
    assert BatchTab._detection_has_no_rois(_Stub(), tmp_path) is False


# --- BatchTab._skip_row_no_rois (unbound) ---------------------------------

def test_skip_row_no_rois_marks_downstream_and_finishes():
    stub = _Stub()
    BatchTab._skip_row_no_rois(stub)

    assert stub._row_results["status"] == "no_rois"
    assert stub._row_results["error"]
    assert stub._row_done_called is True

    # Every stage strictly after detection is marked skipped; detection
    # and preprocess are left untouched.
    det_idx = BatchTab.STAGES.index("detection")
    for st in BatchTab.STAGES[det_idx + 1:]:
        assert stub._row_results["stages"][st]["status"] == "skipped"
    assert "detection" not in stub._row_results["stages"]
    assert "preprocess" not in stub._row_results["stages"]
