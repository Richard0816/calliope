"""Follow-ups from the v1.0.1 review: generation stamps that refuse outputs
made for an earlier cell set or dF/F, the Retrain re-apply keep rule / failure handling, and the
Cellpose provenance + model use of both Cellpose passes."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from calliope.core import detection_run as dr
from calliope.core import generation as gen
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


def _stamp_tab4(p):
    """Tab 4 output made from the filtered dF/F as it is now."""
    gen.stamp_lowpass(p, gen.check_filtered(p))


def test_same_count_cell_swap_refuses_old_lowpass(tmp_path):
    # keep {0,1} -> {0,2}: same width, so Tab 5 would read ROI 1's
    # low-pass trace as ROI 2. Nothing is deleted; the stamp refuses it.
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0])
    dr.compute_filtered_dff(p)
    _stamp_tab4(p)
    assert gen.check_lowpass(p) is not None
    np.save(p / "predicted_cell_mask.npy", np.array([1, 0, 1], bool))
    dr.compute_filtered_dff(p)
    with pytest.raises(gen.StaleOutputError, match="re-run Tab 4"):
        gen.check_lowpass(p)
    assert (p / LP).exists() and (p / DT).exists()
    filt = np.memmap(p / FILT, np.float32, "r", shape=(6, 2))
    assert filt[0].tolist() == [0.0, 2.0]
    del filt


def test_unchanged_cell_set_keeps_lowpass_valid(tmp_path):
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0])
    dr.compute_filtered_dff(p)
    _stamp_tab4(p)
    dr.compute_filtered_dff(p)                   # Promote with no change
    assert gen.check_lowpass(p) is not None


def test_pre_stamp_run_accepted_until_its_cells_change(tmp_path):
    # Runs from before v1.0.2 carry no stamps and behave as before ...
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0])
    assert gen.check_lowpass(p) is None
    dr.compute_filtered_dff(p)                   # same cells: still fine
    assert gen.check_lowpass(p) is not None
    # ... until a stamped rewrite changes the cell set.
    np.save(p / "predicted_cell_mask.npy", np.array([1, 0, 1], bool))
    dr.compute_filtered_dff(p)
    with pytest.raises(gen.StaleOutputError):
        gen.check_lowpass(p)


def test_pre_stamp_run_refused_when_its_cells_already_differ(tmp_path):
    p = _plane0(tmp_path, live=[1, 0, 1], frozen=[1, 1, 0])
    dr.compute_filtered_dff(p)
    with pytest.raises(gen.StaleOutputError):
        gen.check_lowpass(p)


def test_dff_rebuild_makes_filtered_and_lowpass_stale(tmp_path):
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0])
    dr.compute_filtered_dff(p)
    _stamp_tab4(p)
    gen.new_dff_build(p)                         # full dF/F rebuilt
    with pytest.raises(gen.StaleOutputError, match="re-run Tab 3"):
        gen.check_filtered(p)
    dr.compute_filtered_dff(p)                   # same mask, new dF/F
    gen.check_filtered(p)
    with pytest.raises(gen.StaleOutputError, match="re-run Tab 4"):
        gen.check_lowpass(p)


def test_zero_cells_then_cells_again(tmp_path):
    p = _plane0(tmp_path, live=[0, 0, 0], frozen=[1, 1, 0])
    assert dr.compute_filtered_dff(p) is None
    assert not np.load(p / "r0p7_cell_mask_bool.npy").any()
    assert dr.has_filtered_outputs(p)
    with pytest.raises(gen.StaleOutputError, match="no ROI survives"):
        gen.check_filtered(p)
    with pytest.raises(gen.StaleOutputError):
        gen.check_lowpass(p)
    np.save(p / "predicted_cell_mask.npy", np.array([0, 1, 1], bool))
    assert dr.compute_filtered_dff(p) is not None
    assert gen.check_filtered(p) is not None
    filt = np.memmap(p / FILT, np.float32, "r", shape=(6, 2))
    assert filt[0].tolist() == [1.0, 2.0]
    del filt


@pytest.mark.skipif(sys.platform != "win32",
                    reason="only Windows refuses to truncate a mapped file")
def test_locked_filtered_memmap_leaves_stamps_consistent(tmp_path):
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0])
    dr.compute_filtered_dff(p)
    _stamp_tab4(p)
    np.save(p / "predicted_cell_mask.npy", np.array([1, 0, 1], bool))
    held = np.asarray(np.memmap(p / FILT, np.float32, "r", shape=(6, 2))[:, 0])
    try:
        with pytest.raises(OSError):
            dr.compute_filtered_dff(p)
        # The failed rewrite changed nothing: old generation still current.
        assert gen.check_lowpass(p) is not None
        assert np.load(p / "r0p7_cell_mask_bool.npy").tolist() == [1, 1, 0]
    finally:
        del held


def test_lowpass_compute_refuses_stale_filtered_and_stamps(tmp_path):
    from calliope.core import lowpass_run
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0], T=200)
    dr.compute_filtered_dff(p)
    gen.new_dff_build(p)
    with pytest.raises(gen.StaleOutputError):
        lowpass_run.compute_lowpass_and_dt(p, fps=10.0, cutoff_hz=1.0)
    dr.compute_filtered_dff(p)
    lowpass_run.compute_lowpass_and_dt(p, fps=10.0, cutoff_hz=1.0)
    assert gen.check_lowpass(p) == gen.check_filtered(p)


def test_event_run_refuses_stale_lowpass(tmp_path):
    from calliope.core import event_detection_run as edr
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0])
    dr.compute_filtered_dff(p)
    _stamp_tab4(p)
    np.save(p / "predicted_cell_mask.npy", np.array([1, 0, 1], bool))
    dr.compute_filtered_dff(p)
    with pytest.raises(gen.StaleOutputError, match="re-run Tab 4"):
        edr.run_event_detection(p, {}, write_summary=False)


def test_event_sheets_and_nwb_follow_the_generation(tmp_path):
    from calliope.core import summary_writer
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0])
    gen.new_dff_build(p)            # a stamped run: no pre-stamp leniency
    dr.compute_filtered_dff(p)
    _stamp_tab4(p)
    summary_writer.write_events_sheets(
        p, event_windows=[(0.0, 0.3)], onsets_by_roi=[np.array([0.1])] * 2,
        fps=10.0, roi_ids=[0, 1], generation=gen.check_lowpass(p))
    assert gen.check_events(p) is not None
    np.save(p / "predicted_cell_mask.npy", np.array([1, 0, 1], bool))
    dr.compute_filtered_dff(p)
    with pytest.raises(gen.StaleOutputError):
        gen.check_events(p)


def test_clusters_follow_the_generation(tmp_path):
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0])
    dr.compute_filtered_dff(p)
    cdir = p / "r0p7_filtered_cluster_results"
    cdir.mkdir()
    gen.stamp_clusters(cdir, gen.cluster_input(p, True))
    gen.check_clusters(p, cdir, True)
    np.save(p / "predicted_cell_mask.npy", np.array([1, 0, 1], bool))
    dr.compute_filtered_dff(p)
    with pytest.raises(gen.StaleOutputError, match="Tab 6"):
        gen.check_clusters(p, cdir, True)
    # Clusters on the full dF/F only care about dF/F rebuilds.
    udir = p / "r0p7_cluster_results"
    udir.mkdir()
    gen.stamp_clusters(udir, gen.cluster_input(p, False))
    gen.check_clusters(p, udir, False)
    gen.new_dff_build(p)
    with pytest.raises(gen.StaleOutputError):
        gen.check_clusters(p, udir, False)


def test_open_run_reports_stale_lowpass(tmp_path):
    from calliope.core import run_loader
    run = tmp_path / "run"
    p = run_loader.plane0_for(run)
    p.parent.mkdir(parents=True)
    src = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0])
    src.rename(p)
    np.save(p / "stat.npy", np.array([{}, {}, {}], dtype=object))
    dr.compute_filtered_dff(p)
    _stamp_tab4(p)
    assert run_loader.scan_run(run).has_lowpass
    np.save(p / "predicted_cell_mask.npy", np.array([1, 0, 1], bool))
    dr.compute_filtered_dff(p)
    inv = run_loader.scan_run(run)
    assert not inv.has_lowpass and "re-run Tab 4" in inv.lowpass_stale


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


def test_reapply_regenerates_a_former_zero_cell_recording(tmp_path,
                                                          monkeypatch):
    from calliope.tabs.suite2p import curation_popout as cp
    from calliope.core.cellfilter import predict as pr

    p = _plane0(tmp_path, live=[0, 0, 0], frozen=[1, 1, 0])
    dr.compute_filtered_dff(p)                   # now only the frozen mask
    monkeypatch.setattr(cp, "_find_plane0s", lambda root, ex: [p])
    monkeypatch.setattr(cp, "_apply_label_overrides", lambda p0, lab: 0)
    monkeypatch.setattr(
        pr, "predict_recording",
        lambda *a, plane0, **k: np.save(plane0 / "predicted_cell_mask.npy",
                                        np.array([1, 0, 1], bool)))
    me = SimpleNamespace(_train_result=("ok", "done."), _plane0=tmp_path,
                         _reapply_failed=[],
                         _describe_keep_rule=cp.CurationPopout._describe_keep_rule)
    cp.CurationPopout._reapply(me, tmp_path, None, None, [], 0.0,
                               {"threshold": 0.5, "use_iscell": False})
    assert (p / FILT).exists()
    assert np.load(p / "r0p7_cell_mask_bool.npy").tolist() == [1, 0, 1]


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


def test_real_dff_rebuild_marks_downstream_stale(tmp_path):
    # Through compute_dff_memmaps itself (CPU path), not the helper.
    rng = np.random.default_rng(0)
    F = (100 + rng.random((3, 200))).astype(np.float32)
    np.save(tmp_path / "F.npy", F)
    np.save(tmp_path / "Fneu.npy", (F * 0.1).astype(np.float32))
    np.save(tmp_path / "predicted_cell_mask.npy", np.array([1, 1, 0], bool))
    params = {"fps_override": 10.0, "use_gpu_dff": False}
    dr.compute_dff_memmaps(tmp_path, params, baseline_min=0.1)
    dr.compute_filtered_dff(tmp_path)
    _stamp_tab4(tmp_path)
    assert gen.check_lowpass(tmp_path) is not None
    dr.compute_dff_memmaps(tmp_path, params, baseline_min=0.1)
    with pytest.raises(gen.StaleOutputError, match="re-run Tab 3"):
        gen.check_lowpass(tmp_path)


def test_clustering_refuses_stale_input_and_stamps_its_export(tmp_path):
    from calliope.core import clustering
    rng = np.random.default_rng(1)
    p = _plane0(tmp_path, live=[1, 1, 1, 1, 0], frozen=[1, 1, 1, 1, 0], T=40)
    full = np.memmap(p / "r0p7_dff.memmap.float32", np.float32, "r+",
                     shape=(40, 5))
    full[:] = rng.random((40, 5))
    full.flush()
    del full
    gen.new_dff_build(p)
    with pytest.raises(gen.StaleOutputError):
        clustering.load_filtered_dff(p, "r0p7_filtered_")
    dr.compute_filtered_dff(p)
    stat = [{"ypix": np.array([i]), "xpix": np.array([i]),
             "med": [i, i]} for i in range(5)]
    np.save(p / "stat.npy", np.array(stat, dtype=object), allow_pickle=True)
    np.save(p / "ops.npy", {"Ly": 8, "Lx": 8}, allow_pickle=True)
    out = clustering.run_clustering(p, {"prefix": "r0p7_filtered_",
                                        "threshold": 1.0},
                                    figures_dir=None, write_summary=False)
    cdir = Path(out["export_dir"])
    gen.check_clusters(p, cdir, True)
    np.save(p / "predicted_cell_mask.npy", np.array([1, 1, 0, 1, 1], bool))
    dr.compute_filtered_dff(p)
    with pytest.raises(gen.StaleOutputError):
        gen.check_clusters(p, cdir, True)


def test_tab6_reload_refuses_stale_clusters(tmp_path, monkeypatch):
    from calliope.tabs.clustering import tab as ct
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0])
    gen.new_dff_build(p)
    dr.compute_filtered_dff(p)
    cdir = p / "r0p7_filtered_cluster_results"
    cdir.mkdir()
    gen.stamp_clusters(cdir, gen.cluster_input(p, True))
    np.save(p / "predicted_cell_mask.npy", np.array([1, 0, 1], bool))
    dr.compute_filtered_dff(p)
    shown = []
    monkeypatch.setattr(ct.messagebox, "showerror",
                        lambda *a, **k: shown.append(a))

    class _Var:
        def __init__(self, v=""):
            self.v = v

        def get(self):
            return self.v

        def set(self, v):
            self.v = v

    me = SimpleNamespace(
        _job=None, path_var=_Var(str(p)), prefix_var=_Var("r0p7_filtered_"),
        status_var=_Var(), _inputs_ready=lambda *a: True,
        _has_existing_clusters=lambda *a: True,
        _existing_cluster_dir=lambda pl, pr: pl / f"{pr}cluster_results")
    assert ct.ClusteringTab._on_reload_clusters(me) is False
    assert shown and "out of date" in shown[0][0]


def test_tab6_gui_export_stamps_so_its_reload_is_accepted(tmp_path):
    # A missing stamp would make the GUI's own export refuse to reload.
    from scipy.cluster.hierarchy import linkage
    from calliope.tabs.clustering import tab as ct
    p = _plane0(tmp_path, live=[1, 1, 1, 0], frozen=[1, 1, 1, 0])
    gen.new_dff_build(p)
    dr.compute_filtered_dff(p)
    Z = linkage(np.random.default_rng(0).random((3, 4)), "ward")
    me = SimpleNamespace(
        _Z=Z, _plane0=p, _prefix="r0p7_filtered_",
        _current_threshold=lambda: 1e9, _visual_to_label={},
        _compute_visual_cluster_map=lambda T: {1: 1},
        _linkage_ids_current=lambda: np.array([0, 1, 2]),
        _existing_cluster_dir=lambda pl, pr: pl / f"{pr}cluster_results")
    out = ct.ClusteringTab._export_clusters(me)
    gen.check_clusters(p, out, True)


def test_batch_asks_once_before_downloading_the_cellpose_model(
        tmp_path, monkeypatch):
    from calliope.tabs.batch import tab as bt
    monkeypatch.setattr(sp._cp_models, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(sp, "cellpose_sam_available", lambda: True)
    asked, errors = [], []
    answer = {"yes": False}
    monkeypatch.setattr(bt.messagebox, "askyesno",
                        lambda *a, **k: asked.append(a) or answer["yes"])
    monkeypatch.setattr(bt.messagebox, "showerror",
                        lambda *a, **k: errors.append(a))
    rows = [SimpleNamespace(params={"cellpose_enabled": True}),
            SimpleNamespace(params={"cellpose_enabled": False,
                                    "enable_sam_vcorr_pass": True})]
    me = SimpleNamespace(_rows=rows,
                         _effective_row_params=lambda r: dict(r.params))
    assert bt.BatchTab._confirm_cellpose_model(me) is False   # No = cancel
    assert len(asked) == 1                                     # one question
    answer["yes"] = True
    assert bt.BatchTab._confirm_cellpose_model(me) is True
    (tmp_path / sp.cellpose_default_model()).write_bytes(b"x")
    asked.clear()
    assert bt.BatchTab._confirm_cellpose_model(me) is True and not asked
    rows.append(SimpleNamespace(params={"cellpose_model_path":
                                        str(tmp_path / "nope.pt")}))
    assert bt.BatchTab._confirm_cellpose_model(me) is False and errors
    # Sparsery-only rows never ask.
    me._rows = [SimpleNamespace(params={"cellpose_enabled": False})]
    (tmp_path / sp.cellpose_default_model()).unlink()
    asked.clear()
    assert bt.BatchTab._confirm_cellpose_model(me) is True and not asked
