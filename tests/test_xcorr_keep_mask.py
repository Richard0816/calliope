"""Tab 7 cross-correlation: Suite2p id -> dF/F column translation.

The filtered memmap is column-sliced with the keep mask frozen at write time
(``r0p7_cell_mask_bool.npy``). Curation can later change the live
``predicted_cell_mask.npy``; xcorr must keep reading the columns the memmap
was actually written with.
"""
from __future__ import annotations

import inspect
import types

import numpy as np
import pandas as pd
import pytest

from calliope.core import crosscorrelation as xc

PREFIX = "r0p7_filtered_"
T = 400


def _make_plane0(tmp_path, frozen, live, columns, clusters):
    plane0 = tmp_path / "plane0"
    plane0.mkdir()
    n_total = len(frozen)
    np.save(plane0 / "F.npy", np.zeros((n_total, T), dtype=np.float32))
    np.save(plane0 / "r0p7_cell_mask_bool.npy", np.asarray(frozen, bool))
    np.save(plane0 / "predicted_cell_mask.npy", np.asarray(live, bool))
    dff = np.column_stack(columns).astype(np.float32)
    assert dff.shape == (T, int(np.sum(frozen)))
    mm = np.memmap(plane0 / f"{PREFIX}dff.memmap.float32", dtype="float32",
                   mode="w+", shape=dff.shape)
    mm[:] = dff
    mm.flush()
    del mm
    cdir = plane0 / f"{PREFIX}cluster_results"
    cdir.mkdir()
    (cdir / "_indices_are_suite2p").touch()
    for name, rois in clusters.items():
        np.save(cdir / f"{name}_rois.npy", np.asarray(rois, dtype=np.int64))
    return plane0


def _drifted_plane0(tmp_path, clusters):
    # Frozen mask keeps ROIs 1-4 -> columns 0..3. The live mask has drifted
    # to ROIs 0-3 (same kept count), which would put ROI 1 at column 1 and
    # ROI 3 at column 3.
    frozen = [0, 1, 1, 1, 1, 0]
    live = [1, 1, 1, 1, 0, 0]
    rng = np.random.default_rng(0)
    t = np.arange(T)
    sig = np.sin(2 * np.pi * t / 37.0).astype(np.float32)
    cols = [sig,                                  # col 0 = ROI 1
            rng.standard_normal(T),               # col 1 = ROI 2
            sig.copy(),                           # col 2 = ROI 3
            rng.standard_normal(T)]               # col 3 = ROI 4
    return _make_plane0(tmp_path, frozen, live, cols, clusters)


def test_load_keep_mask_prefers_frozen_mask(tmp_path):
    plane0 = _drifted_plane0(tmp_path, {"C1": [1], "C2": [3]})
    mask = xc._load_keep_mask(plane0, PREFIX)
    np.testing.assert_array_equal(mask, [0, 1, 1, 1, 1, 0])
    assert xc._load_keep_mask(plane0, "r0p7_") is None


def test_full_run_reads_frozen_columns_after_live_mask_drift(tmp_path):
    plane0 = _drifted_plane0(tmp_path, {"C1": [1], "C2": [3]})
    out = xc.run_cluster_xcorr_full_fast(
        plane0, PREFIX, 15.0, "", max_lag_seconds=0.5, use_gpu=False,
        n_shuffles=0)
    df = pd.read_csv(out / "C1xC2" / "C1xC2_summary.csv")
    assert list(df["roiA"]) == [1] and list(df["roiB"]) == [3]
    # ROI 1 and ROI 3 carry the same trace in the frozen layout; the live
    # mask would have paired two independent noise columns instead.
    assert df["zero_lag_corr"].iloc[0] == pytest.approx(1.0, abs=1e-4)


def test_per_event_run_reads_frozen_columns(tmp_path):
    plane0 = _drifted_plane0(tmp_path, {"C1": [1], "C2": [3]})
    out = xc.run_cluster_xcorr_per_event_fast(
        plane0, PREFIX, 15.0, "", [(1.0, 20.0)], max_lag_seconds=0.5,
        use_gpu=False)
    csvs = list(out.glob("event*/C1xC2/C1xC2_summary.csv"))
    assert len(csvs) == 1
    df = pd.read_csv(csvs[0])
    assert df["zero_lag_corr"].iloc[0] == pytest.approx(1.0, abs=1e-4)


def test_out_of_range_roi_raises_clear_error(tmp_path):
    plane0 = _drifted_plane0(tmp_path, {"C1": [1], "C2": [9]})
    with pytest.raises(ValueError, match="out of range"):
        xc.run_cluster_xcorr_full_fast(
            plane0, PREFIX, 15.0, "", use_gpu=False, n_shuffles=0)


def test_per_event_runner_has_no_dead_shuffle_kwargs():
    params = inspect.signature(xc.run_cluster_xcorr_per_event_fast).parameters
    assert "n_shuffles" not in params
    assert "shuffle_seed" not in params


class _Var:
    def __init__(self, v):
        self.v = v

    def get(self):
        return self.v

    def set(self, v):
        self.v = v


def _fake_tab():
    return types.SimpleNamespace(
        maxlag_var=_Var("2.0"), zero_lag_var=_Var(True), gpu_var=_Var(True),
        n_shuffles_var=_Var("500"), partial_corr_var=_Var(False))


def test_apply_batch_row_honours_shuffles_and_partial():
    from calliope.tabs.crosscorrelation.tab import CrossCorrelationTab
    tab = _fake_tab()
    CrossCorrelationTab.apply_batch_row(
        tab, {"n_shuffles": 0, "compute_partial": "true"})
    assert tab.n_shuffles_var.get() == "0"
    assert tab.partial_corr_var.get() is True

    tab = _fake_tab()
    CrossCorrelationTab.apply_batch_row(tab, {"max_lag_seconds": 1.5})
    # Keys the row doesn't carry leave the widgets alone.
    assert tab.n_shuffles_var.get() == "500"
    assert tab.partial_corr_var.get() is False


def test_run_crosscorrelation_forwards_shuffle_seed(tmp_path, monkeypatch):
    seen = {}

    def fake_full(plane0, **kw):
        seen.update(kw)

    monkeypatch.setattr(xc, "run_cluster_xcorr_full_fast", fake_full)
    xc.run_crosscorrelation(tmp_path, {"fps": 15.0, "shuffle_seed": 7})
    assert seen["shuffle_seed"] == 7
