"""Follow-ups from the v1.0.1 review: stale Tab 4 outputs after a cell-set
change, the Retrain re-apply keep rule / failure handling, and the
Cellpose provenance + model use of both Cellpose passes."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from calliope.core import detection_run as dr
from calliope.core import suite2p_pipeline as sp

LP = "r0p7_filtered_dff_lowpass.memmap.float32"
DT = "r0p7_filtered_dff_dt.memmap.float32"
FILT = "r0p7_filtered_dff.memmap.float32"


def _plane0(tmp_path, live, frozen, T=6):
    """A finished plane0: full dF/F, the live keep mask, the filtered memmap
    and Tab 4 outputs written for the ``frozen`` mask."""
    p = tmp_path / "plane0"
    p.mkdir()
    N = len(live)
    np.save(p / "F.npy", np.zeros((N, T), np.float32))
    full = np.memmap(p / "r0p7_dff.memmap.float32", np.float32, "w+",
                     shape=(T, N))
    full[:] = np.arange(N, dtype=np.float32)       # column k == k
    full.flush()
    del full
    np.save(p / "predicted_cell_mask.npy", np.asarray(live, bool))
    frozen = np.asarray(frozen, bool)
    np.save(p / "r0p7_cell_mask_bool.npy", frozen)
    for name in (FILT, LP, DT):
        mm = np.memmap(p / name, np.float32, "w+", shape=(T, int(frozen.sum())))
        mm[:] = 1.0
        mm.flush()
        del mm
    return p


def test_changed_cell_set_drops_tab4_outputs_even_at_same_count(tmp_path):
    # keep {0,1} -> {0,2}: same width, so Tab 5 would read ROI 1's
    # low-pass trace as ROI 2.
    p = _plane0(tmp_path, live=[1, 0, 1], frozen=[1, 1, 0])
    assert dr.compute_filtered_dff(p) is not None
    assert not (p / LP).exists() and not (p / DT).exists()
    filt = np.memmap(p / FILT, np.float32, "r", shape=(6, 2))
    assert filt[0].tolist() == [0.0, 2.0]
    del filt
    assert np.load(p / "r0p7_cell_mask_bool.npy").tolist() == [1, 0, 1]


def test_unchanged_cell_set_keeps_tab4_outputs(tmp_path):
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0])
    dr.compute_filtered_dff(p)
    assert (p / LP).exists() and (p / DT).exists()


def test_zero_cells_invalidates_old_filtered_data(tmp_path):
    p = _plane0(tmp_path, live=[0, 0, 0], frozen=[1, 1, 0])
    assert dr.compute_filtered_dff(p) is None
    for name in (FILT, LP, DT):
        assert not (p / name).exists()
    assert not np.load(p / "r0p7_cell_mask_bool.npy").any()


@pytest.mark.skipif(sys.platform != "win32",
                    reason="only Windows refuses to delete a mapped file")
def test_mapped_tab4_output_stops_before_anything_is_rewritten(tmp_path):
    p = _plane0(tmp_path, live=[1, 0, 1], frozen=[1, 1, 0])
    held = np.memmap(p / DT, np.float32, "r", shape=(6, 2))
    try:
        with pytest.raises(OSError):
            dr.compute_filtered_dff(p)
        # Nothing rewritten: filtered memmap and frozen mask still agree.
        assert np.load(p / "r0p7_cell_mask_bool.npy").tolist() == [1, 1, 0]
        filt = np.memmap(p / FILT, np.float32, "r", shape=(6, 2))
        assert filt[0].tolist() == [1.0, 1.0]
        del filt
    finally:
        del held


# --- Retrain re-apply ----------------------------------------------------

def test_reapply_uses_keep_rule_and_failures_keep_current_result(
        tmp_path, monkeypatch):
    from calliope.tabs.suite2p import curation_popout as cp
    from calliope.core.cellfilter import predict as pr

    good, bad = tmp_path / "a", tmp_path / "b"
    monkeypatch.setattr(cp, "_find_plane0s", lambda root, ex: [good, bad])
    monkeypatch.setattr(cp, "_apply_label_overrides", lambda p0, lab: 0)
    calls = []

    def fake_predict(rec, model, device, *, plane0, size_outlier_z, **rule):
        calls.append(rule)
        if plane0 == bad:
            raise RuntimeError("disk full")

    monkeypatch.setattr(pr, "predict_recording", fake_predict)
    me = SimpleNamespace(_train_result=("ok", "Training complete."),
                         _plane0=tmp_path / "cur", _reapply_failed=[],
                         _describe_keep_rule=cp.CurationPopout._describe_keep_rule)
    rule = {"threshold": 0.9, "use_iscell": False}
    cp.CurationPopout._reapply(me, tmp_path, None, None, [], 0.0, rule)
    assert calls == [rule, rule]
    kind, msg = me._train_result
    assert kind == "ok"                      # current recording still refreshes
    assert "cell prob >= 0.9" in msg and "1/2" in msg
    assert len(me._reapply_failed) == 1 and "disk full" in me._reapply_failed[0]


def test_keep_rule_comes_from_parent_or_defaults():
    from calliope.tabs.suite2p import curation_popout as cp
    me = SimpleNamespace(_get_keep_rule=None)
    assert cp.CurationPopout._keep_rule(me) == {"threshold": 0.5,
                                                "use_iscell": False}
    me._get_keep_rule = lambda: dr.keep_rule_kwargs(
        {"cellfilter_threshold": 0.8, "cellfilter_source": "suite2p iscell"})
    assert cp.CurationPopout._keep_rule(me) == {"threshold": 0.8,
                                                "use_iscell": True}


# --- Cellpose provenance and the SAM-on-Vcorr pass -----------------------

def test_preflight_runs_when_only_the_sam_pass_is_on(tmp_path, monkeypatch):
    monkeypatch.setattr(sp._cp_models, "MODEL_DIR", tmp_path)
    assert sp.cellpose_model_problem({"cellpose_enabled": False}) is None
    assert "not downloaded" in sp.cellpose_model_problem(
        {"cellpose_enabled": False, "enable_sam_vcorr_pass": True})


def test_final_ops_records_both_cellpose_passes(tmp_path, monkeypatch):
    from calliope.core import sparse_plus_cellpose as spc
    from calliope.core import correlation_image

    shared = tmp_path / "shared"
    shared.mkdir()
    final = tmp_path / "out" / "final" / "plane0"
    seen_cfgs = []

    def fake_cp_pass(save_dir, shared_plane0, cfg, verbose=True):
        seen_cfgs.append(dict(cfg))
        name = cfg.get("cellpose_channel_input", "meanImg")
        return [], {"cellpose_config": {"cellpose_version": "4.2.1.1",
                                        "model_name": "w.pt",
                                        "channel_input": name}}, save_dir

    def fake_merge(*a, **k):
        final.mkdir(parents=True)
        ops = {"Ly": 4, "Lx": 4}
        np.save(final / "ops.npy", ops, allow_pickle=True)
        return np.array([]), ops, final

    for name, fn in {
        "Suite2pPipelineConfig": lambda **k: SimpleNamespace(**k),
        "load_base_settings": lambda cfg: ({}, {}),
        "_get_or_create_shared_registration": lambda *a: shared,
        "check_detection_memory": lambda *a, **k: None,
        "run_one_pass": lambda *a, **k: ([], {}, tmp_path / "nope"),
        "run_cellpose_pass": fake_cp_pass,
        "merge_and_extract": fake_merge,
    }.items():
        monkeypatch.setattr(spc, name, fn)
    monkeypatch.setattr(spc.utils, "load_plane_view",
                        lambda p: {"Ly": 4, "Lx": 4, "nframes": 2})
    monkeypatch.setattr(correlation_image, "load_or_compute_vcorr",
                        lambda *a, **k: np.zeros((4, 4), np.float32))
    monkeypatch.setattr(sp, "cellpose_sam_available", lambda: True)

    cp_cfg = dict(spc.CELLPOSE_CFG, cellpose_model_path=str(tmp_path / "w.pt"))
    spc.run(tiff_folder=tmp_path, save_folder=tmp_path / "out",
            cellpose_cfg=cp_cfg, enable_sam_vcorr_pass=True,
            sam_vcorr_cfg={}, verbose=False)

    # The SAM pass loads the configured model file, never a fresh default.
    assert [c.get("cellpose_model_path") for c in seen_cfgs] == \
        [str(tmp_path / "w.pt")] * 2
    passes = np.load(final / "ops.npy", allow_pickle=True).item()[
        "cellpose_passes"]
    assert set(passes) == {"meanImg", "Vcorr"}
    assert passes["Vcorr"]["cellpose_version"] == "4.2.1.1"
