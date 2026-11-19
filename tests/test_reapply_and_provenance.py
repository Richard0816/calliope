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


def _rebuild(p):
    """A completed full dF/F rebuild."""
    gen.finish_dff_build(p, gen.new_dff_build(p))


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
    gen.check_lowpass(p)
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
    _rebuild(p)                         # full dF/F rebuilt
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
        # The truncating open failed, so nothing changed: the previous
        # generation still holds (no false refusal), data and mask intact.
        assert gen.check_lowpass(p) is not None
        assert np.load(p / "r0p7_cell_mask_bool.npy").tolist() == [1, 1, 0]
    finally:
        del held
    dr.compute_filtered_dff(p)                  # rerun once released
    with pytest.raises(gen.StaleOutputError, match="re-run Tab 4"):
        gen.check_lowpass(p)                    # new cells: Tab 4 again


def test_lowpass_compute_refuses_stale_filtered_and_stamps(tmp_path):
    from calliope.core import lowpass_run
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0], T=200)
    dr.compute_filtered_dff(p)
    _rebuild(p)
    with pytest.raises(gen.StaleOutputError):
        lowpass_run.compute_lowpass_and_dt(p, fps=10.0, cutoff_hz=1.0)
    dr.compute_filtered_dff(p)
    lowpass_run.compute_lowpass_and_dt(p, fps=10.0, cutoff_hz=1.0)
    assert gen.check_lowpass(p).startswith(gen.check_filtered(p) + "|")


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
    _rebuild(p)            # a stamped run: no pre-stamp leniency
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
    _rebuild(p)
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
    _rebuild(p)
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
    _rebuild(p)
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
    _rebuild(p)
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


# --- 4th review --------------------------------------------------------------

def test_gui_render_sends_the_token_to_the_summary_write(tmp_path,
                                                         monkeypatch):
    # The GUI Render path built its own payload and dropped the token, so
    # its workbook was stamped null and NWB left the events out.
    from unittest.mock import MagicMock
    from calliope.core import offload
    from calliope.core import event_detection_run as edr
    from calliope.tabs.event_detection.tab import EventDetectionTab
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0])
    _rebuild(p)
    dr.compute_filtered_dff(p)
    token = gen.stamp_lowpass(p, gen.check_filtered(p))
    sent = {}
    monkeypatch.setattr(offload, "run_offloaded",
                        lambda *a, **k: sent.update(args=k["args"]))
    me = MagicMock()
    me._plane0 = p
    me.state.batch_active = False
    data = {"heatmap": np.zeros((2, 4)), "raster": np.zeros((2, 4)),
            "N_kept": 2, "T": 6, "downsample": 1, "fps": 10.0,
            "event_windows": [(0.1, 0.3)],
            "onsets_by_roi": [np.array([0.2])] * 2, "kept_idx": [0, 1],
            "event_class": ["x"], "event_monotonicity": [],
            "diagnostics": {}, "event_counts": np.zeros(2),
            "generation": token}
    EventDetectionTab._on_render_done(me, data)
    payload = sent["args"][1]
    assert payload["generation"] == token
    edr.write_event_summary_offload(str(p), payload)
    assert gen.check_events(p) == token


def test_pre_stamp_run_tab4_rerun_then_unchanged_promote(tmp_path):
    # Tab 4 recomputed on a pre-stamp run, then Promote with no change:
    # still current (it was refused before).
    from calliope.core import lowpass_run
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0], T=200)
    lowpass_run.compute_lowpass_and_dt(p, fps=10.0, cutoff_hz=1.0)
    gen.check_lowpass(p)
    dr.compute_filtered_dff(p)
    gen.check_lowpass(p)
    np.save(p / "predicted_cell_mask.npy", np.array([1, 0, 1], bool))
    dr.compute_filtered_dff(p)
    with pytest.raises(gen.StaleOutputError):
        gen.check_lowpass(p)


def test_zero_cell_rewrite_is_skipped_by_batch(tmp_path):
    from calliope.core import utils
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0])
    dr.compute_filtered_dff(p)
    assert not utils.no_rois_survive(p)
    np.save(p / "predicted_cell_mask.npy", np.zeros(3, bool))
    dr.compute_filtered_dff(p)
    assert (p / FILT).exists()            # old memmap kept on disk ...
    assert utils.no_rois_survive(p)       # ... but the stamp says 0 cells


def test_interrupted_writes_are_refused(tmp_path):
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0])
    dr.compute_filtered_dff(p)
    _stamp_tab4(p)
    gen.begin_lowpass(p)                   # Tab 4 crashed mid-write
    with pytest.raises(gen.StaleOutputError, match="did not finish"):
        gen.check_lowpass(p)
    _stamp_tab4(p)
    gen.begin_filtered(p)                  # Tab 3 crashed mid-write
    with pytest.raises(gen.StaleOutputError, match="did not finish"):
        gen.check_filtered(p)
    dr.compute_filtered_dff(p)             # re-running Tab 3 recovers
    gen.check_filtered(p)
    gen.new_dff_build(p)                   # detection / dF/F crashed
    with pytest.raises(gen.StaleOutputError, match="did not finish"):
        gen.check_filtered(p)
    with pytest.raises(gen.StaleOutputError):
        dr.compute_filtered_dff(p)         # won't stamp a half-built dF/F


def test_redetection_marks_the_build_pending_before_new_rois(tmp_path,
                                                            monkeypatch):
    from calliope.core import sparse_plus_cellpose as spc
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0])
    dr.compute_filtered_dff(p)
    seen = {}
    real_save = np.save

    def spy(path, *a, **k):
        if str(path).endswith("stat.npy") and "build" not in seen:
            try:
                gen.check_filtered(p)
                seen["build"] = "accepted"
            except gen.StaleOutputError:
                seen["build"] = "refused"
        return real_save(path, *a, **k)

    monkeypatch.setattr(spc.np, "save", spy)
    import inspect
    src = inspect.getsource(spc.merge_and_extract)
    assert src.index("generation.new_dff_build(final_plane0)") < \
        src.index("np.save(final_plane0 / 'stat.npy'")
    gen.new_dff_build(p)                   # what merge_and_extract does first
    spy(p / "stat.npy", np.array([], dtype=object), allow_pickle=True)
    assert seen["build"] == "refused"


def test_tab4_rerun_makes_old_events_stale(tmp_path):
    from calliope.core import summary_writer
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0])
    _rebuild(p)
    dr.compute_filtered_dff(p)
    _stamp_tab4(p)
    summary_writer.write_events_sheets(
        p, event_windows=[(0.0, 0.3)], onsets_by_roi=[np.array([0.1])] * 2,
        fps=10.0, roi_ids=[0, 1], generation=gen.check_lowpass(p))
    gen.check_events(p)
    _stamp_tab4(p)                         # Tab 4 again (other cutoff)
    with pytest.raises(gen.StaleOutputError, match="re-run Tab 5"):
        gen.check_events(p)


def test_events_to_another_workbook_do_not_restamp(tmp_path):
    from calliope.core import summary_writer
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0])
    _rebuild(p)
    dr.compute_filtered_dff(p)
    _stamp_tab4(p)
    summary_writer.write_events_sheets(
        p, event_windows=[(0.0, 0.3)], onsets_by_roi=[np.array([0.1])] * 2,
        fps=10.0, roi_ids=[0, 1], generation="something else",
        filename="other.xlsx")
    assert not (p / gen.EVENTS_FILE).exists()


def test_unstamped_full_dff_clusters_stale_after_rebuild(tmp_path):
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0])
    udir = p / "r0p7_cluster_results"
    udir.mkdir()
    gen.check_clusters(p, udir, False)     # pre-stamp run: accepted
    _rebuild(p)
    with pytest.raises(gen.StaleOutputError):
        gen.check_clusters(p, udir, False)


def test_unreadable_stamp_is_refused_not_taken_for_legacy(tmp_path):
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0])
    dr.compute_filtered_dff(p)
    (p / gen.FILTERED_FILE).write_text("{not json", encoding="utf-8")
    with pytest.raises(gen.StaleOutputError, match="unreadable"):
        gen.check_lowpass(p)


def test_stamp_writes_leave_no_temp_files(tmp_path):
    for _ in range(3):
        gen.stamp_lowpass(tmp_path, "g")
    assert [f.name for f in tmp_path.iterdir()] == [gen.LOWPASS_FILE]


def test_tab4_preview_refuses_out_of_date_filtered(tmp_path):
    from unittest.mock import MagicMock
    from calliope.tabs.lowpass.tab import LowpassTab
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0])
    dr.compute_filtered_dff(p)
    _rebuild(p)                            # filtered not rewritten
    with pytest.raises(gen.StaleOutputError, match="re-run Tab 3"):
        LowpassTab._load_data(MagicMock(), p)


def test_tab7_checks_its_cluster_inputs(tmp_path):
    from calliope.core import crosscorrelation as xc
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0])
    _rebuild(p)
    dr.compute_filtered_dff(p)
    cdir = p / "r0p7_filtered_cluster_results"
    cdir.mkdir()
    gen.stamp_clusters(cdir, gen.cluster_input(p, True))
    xc._check_xcorr_inputs(p, "r0p7_filtered_", cdir)
    np.save(p / "predicted_cell_mask.npy", np.array([1, 0, 1], bool))
    dr.compute_filtered_dff(p)
    with pytest.raises(gen.StaleOutputError):
        xc._check_xcorr_inputs(p, "r0p7_filtered_", cdir)
    with pytest.raises(gen.StaleOutputError):    # and the real runner
        xc.run_cluster_xcorr_full_fast(p, prefix="r0p7_filtered_", fps=10.0,
                                       cluster_folder="")


def test_tab7_ignores_events_from_other_lowpass(tmp_path):
    from calliope.tabs.crosscorrelation.tab import CrossCorrelationTab
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0])
    me = SimpleNamespace(_plane0=p, state=SimpleNamespace(event_results={
        "plane0": str(p), "event_windows": [(0.0, 1.0)],
        "generation": "old|token"}))
    assert CrossCorrelationTab._event_windows_from_state(me, "new|t") is None
    assert CrossCorrelationTab._event_windows_from_state(me, "old|token") \
        == [(0.0, 1.0)]


def test_tab6_export_refuses_linkage_from_an_earlier_dff(tmp_path):
    from scipy.cluster.hierarchy import linkage
    from calliope.tabs.clustering import tab as ct
    p = _plane0(tmp_path, live=[1, 1, 1, 0], frozen=[1, 1, 1, 0])
    _rebuild(p)
    dr.compute_filtered_dff(p)
    made_from = gen.cluster_input(p, True)
    _rebuild(p)                            # same cells, new dF/F
    dr.compute_filtered_dff(p)
    me = SimpleNamespace(
        _Z=linkage(np.random.default_rng(0).random((3, 4)), "ward"),
        _plane0=p, _prefix="r0p7_filtered_", _cluster_inputs=made_from,
        _current_threshold=lambda: 1e9, _visual_to_label={},
        _compute_visual_cluster_map=lambda T: {1: 1},
        _linkage_ids_current=lambda: np.array([0, 1, 2]),
        _existing_cluster_dir=lambda pl, pr: pl / f"{pr}cluster_results")
    with pytest.raises(ValueError, match="Re-run clustering"):
        ct.ClusteringTab._export_clusters(me)


def test_tab4_failing_mid_write_is_refused(tmp_path, monkeypatch):
    from calliope.core import lowpass_run
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0], T=200)
    dr.compute_filtered_dff(p)
    lowpass_run.compute_lowpass_and_dt(p, fps=10.0, cutoff_hz=1.0)
    gen.check_lowpass(p)

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(lowpass_run.utils, "lowpass_causal_1d", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        lowpass_run.compute_lowpass_and_dt(p, fps=10.0, cutoff_hz=1.0)
    with pytest.raises(gen.StaleOutputError, match="did not finish"):
        gen.check_lowpass(p)


def test_filtered_failing_mid_write_is_refused(tmp_path, monkeypatch):
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0])
    dr.compute_filtered_dff(p)
    np.save(p / "predicted_cell_mask.npy", np.array([1, 0, 1], bool))
    real = np.save

    def save(path, *a, **k):
        if str(path).endswith("r0p7_cell_mask_bool.npy"):
            raise OSError("disk full")
        return real(path, *a, **k)

    monkeypatch.setattr(dr.np, "save", save)
    with pytest.raises(OSError):
        dr.compute_filtered_dff(p)       # memmap rewritten, mask not saved
    monkeypatch.setattr(dr.np, "save", real)
    with pytest.raises(gen.StaleOutputError, match="did not finish"):
        gen.check_filtered(p)


def test_unfinished_dff_rebuild_stops_before_rewriting(tmp_path):
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0])
    np.save(p / "predicted_cell_mask.npy", np.array([1, 0, 1], bool))
    gen.new_dff_build(p)                 # never finished
    with pytest.raises(gen.StaleOutputError, match="did not finish"):
        dr.compute_filtered_dff(p)
    filt = np.memmap(p / FILT, np.float32, "r", shape=(6, 2))
    assert filt[0].tolist() == [1.0, 1.0]          # untouched
    del filt
    assert np.load(p / "r0p7_cell_mask_bool.npy").tolist() == [1, 1, 0]


# --- final review --------------------------------------------------------------

def test_tab5_publish_carries_the_token_tab7_needs(tmp_path):
    # Tab 5 published its events without the token, so Tab 7 refused
    # fresh events after an ordinary Render.
    from unittest.mock import MagicMock
    from calliope.tabs.event_detection.tab import EventDetectionTab
    from calliope.tabs.crosscorrelation.tab import CrossCorrelationTab
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0])
    _rebuild(p)
    dr.compute_filtered_dff(p)
    token = gen.stamp_lowpass(p, gen.check_filtered(p))
    me = MagicMock()
    me._plane0 = p
    EventDetectionTab._publish_event_results(
        me, {"event_windows": [(0.1, 0.4)], "generation": token})
    published = me.state.set_event_results.call_args[0][0]
    tab7 = SimpleNamespace(_plane0=p,
                           state=SimpleNamespace(event_results=published))
    assert CrossCorrelationTab._event_windows_from_state(
        tab7, gen.check_lowpass(p)) == [(0.1, 0.4)]


def test_retry_after_interrupted_rewrite_keeps_old_outputs_stale(tmp_path,
                                                                 monkeypatch):
    # Pre-stamp run, Tab 4 output for mask A; Promote a same-count mask B
    # that is interrupted after the new mask is saved; the retry must not
    # declare the old (unstamped) Tab 4 output current again.
    p = _plane0(tmp_path, live=[1, 0, 1], frozen=[1, 1, 0])
    real = gen.stamp_filtered

    def crash(*a, **k):
        raise OSError("power cut")

    monkeypatch.setattr(gen, "stamp_filtered", crash)
    with pytest.raises(OSError):
        dr.compute_filtered_dff(p)         # mask B saved, stamp pending
    monkeypatch.setattr(gen, "stamp_filtered", real)
    assert np.load(p / "r0p7_cell_mask_bool.npy").tolist() == [1, 0, 1]
    dr.compute_filtered_dff(p)             # retry
    with pytest.raises(gen.StaleOutputError):
        gen.check_lowpass(p)


def test_locked_tab4_output_leaves_it_pending_not_half_valid(tmp_path):
    # begin_lowpass runs before the w+ opens: if the second file can't be
    # opened after the first was truncated, the run is refused.
    import sys as _sys
    from calliope.core import lowpass_run
    if _sys.platform != "win32":
        pytest.skip("only Windows refuses to truncate a mapped file")
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0], T=200)
    dr.compute_filtered_dff(p)
    lowpass_run.compute_lowpass_and_dt(p, fps=10.0, cutoff_hz=1.0)
    held = np.asarray(np.memmap(p / DT, np.float32, "r", shape=(200, 2))[:, 0])
    try:
        with pytest.raises(OSError):
            lowpass_run.compute_lowpass_and_dt(p, fps=10.0, cutoff_hz=1.0)
        with pytest.raises(gen.StaleOutputError, match="did not finish"):
            gen.check_lowpass(p)
    finally:
        del held


def test_cluster_export_pending_while_rewriting(tmp_path):
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0])
    dr.compute_filtered_dff(p)
    cdir = p / "r0p7_filtered_cluster_results"
    cdir.mkdir()
    gen.stamp_clusters(cdir, gen.cluster_input(p, True))
    gen.begin_clusters(cdir)               # export interrupted
    with pytest.raises(gen.StaleOutputError, match="did not finish"):
        gen.check_clusters(p, cdir, True)


def test_finish_only_completes_its_own_build(tmp_path):
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0])
    a = gen.new_dff_build(p)
    gen.new_dff_build(p)                   # another writer started b
    gen.finish_dff_build(p, a)             # a finishing must not complete b
    with pytest.raises(gen.StaleOutputError, match="did not finish"):
        gen.dff_build_id(p)


def test_nwb_refuses_an_unfinished_rebuild(tmp_path):
    from calliope.core import nwb_export
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0])
    np.save(p / "stat.npy", np.array([{"ypix": np.array([1]),
                                       "xpix": np.array([1]),
                                       "lam": np.array([1.0])}] * 3,
                                     dtype=object), allow_pickle=True)
    np.save(p / "ops.npy", {"Ly": 4, "Lx": 4, "fs": 10.0}, allow_pickle=True)
    gen.new_dff_build(p)                   # detection stopped half-way
    with pytest.raises(gen.StaleOutputError, match="did not finish"):
        nwb_export.export_recording_to_nwb(p, tmp_path / "x.nwb")


def test_reload_keeps_the_linkage_inputs(tmp_path):
    from scipy.cluster.hierarchy import linkage
    from calliope.tabs.clustering.logic import reload_clustering_offload
    p = _plane0(tmp_path, live=[1, 1, 1, 0], frozen=[1, 1, 1, 0])
    _rebuild(p)
    dr.compute_filtered_dff(p)
    np.save(p / "ops.npy", {"Ly": 8, "Lx": 8}, allow_pickle=True)
    cdir = p / "r0p7_filtered_cluster_results"
    cdir.mkdir()
    np.save(cdir / "linkage.npy",
            linkage(np.random.default_rng(0).random((3, 4)), "ward"))
    np.save(cdir / "threshold_used.npy", np.array([1.0]))
    made_from = gen.cluster_input(p, True)
    gen.stamp_clusters(cdir, made_from)
    _rebuild(p)                            # newer dF/F, same cells
    dr.compute_filtered_dff(p)
    out = reload_clustering_offload(str(p), "r0p7_filtered_", str(cdir))
    assert out["inputs"] == made_from     # what the linkage was built on
    assert out["inputs"] != gen.cluster_input(p, True)


def test_unreadable_stamp_is_repaired_by_rerunning_tab3(tmp_path):
    # Pre-stamp Tab 4 output, then a stamp that became unreadable.
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0])
    dr.compute_filtered_dff(p)
    (p / gen.FILTERED_FILE).write_text("{oops", encoding="utf-8")
    dr.compute_filtered_dff(p)             # the instructed rerun works
    gen.check_filtered(p)
    with pytest.raises(gen.StaleOutputError):   # but nothing older trusted
        gen.check_lowpass(p)


def test_tab4_metadata_failure_leaves_the_run_pending(tmp_path, monkeypatch):
    from calliope.core import lowpass_run, recording_meta
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0], T=200)
    dr.compute_filtered_dff(p)

    def locked(*a, **k):
        raise PermissionError("meta.json locked")

    monkeypatch.setattr(recording_meta, "update_meta", locked)
    with pytest.raises(RuntimeError, match="re-run Tab 4"):
        lowpass_run.compute_lowpass_and_dt(p, fps=10.0, cutoff_hz=1.0)
    with pytest.raises(gen.StaleOutputError, match="did not finish"):
        gen.check_lowpass(p)


def test_v101_style_event_summary_call_still_counts(tmp_path):
    # Scripts using the v1.0.1 signature (no generation argument) write
    # events from the current low-pass run; NWB must keep them.
    from calliope.core import summary_writer
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0])
    _rebuild(p)
    dr.compute_filtered_dff(p)
    _stamp_tab4(p)
    summary_writer.write_events_sheets(
        p, event_windows=[(0.0, 0.3)], onsets_by_roi=[np.array([0.1])] * 2,
        fps=10.0, roi_ids=[0, 1])
    assert gen.check_events(p) == gen.check_lowpass(p)


def test_locked_first_tab4_file_keeps_the_previous_run(tmp_path):
    import sys as _sys
    from calliope.core import lowpass_run
    if _sys.platform != "win32":
        pytest.skip("only Windows refuses to truncate a mapped file")
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0], T=200)
    dr.compute_filtered_dff(p)
    lowpass_run.compute_lowpass_and_dt(p, fps=10.0, cutoff_hz=1.0)
    before = gen.check_lowpass(p)
    held = np.asarray(np.memmap(p / LP, np.float32, "r", shape=(200, 2))[:, 0])
    try:
        with pytest.raises(OSError):
            lowpass_run.compute_lowpass_and_dt(p, fps=10.0, cutoff_hz=2.0)
        assert gen.check_lowpass(p) == before   # nothing was truncated
    finally:
        del held


def test_explicit_none_token_is_kept(tmp_path):
    # A pre-stamp render (token None) saved after Tab 4 was re-run must not
    # be stamped with the new low-pass token.
    from calliope.core import summary_writer
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0])
    dr.compute_filtered_dff(p)
    _stamp_tab4(p)                          # Tab 4 re-run after the render
    summary_writer.write_events_sheets(
        p, event_windows=[(0.0, 0.3)], onsets_by_roi=[np.array([0.1])] * 2,
        fps=10.0, roi_ids=[0, 1], generation=None)
    with pytest.raises(gen.StaleOutputError):
        gen.check_events(p)


def test_tab7_refuses_cached_windows_after_tab4_rerun(tmp_path, monkeypatch):
    from calliope.tabs.crosscorrelation import tab as xt
    p = _plane0(tmp_path, live=[1, 1, 0], frozen=[1, 1, 0])
    dr.compute_filtered_dff(p)
    old = gen.stamp_lowpass(p, gen.check_filtered(p))
    errors = []
    monkeypatch.setattr(xt.messagebox, "showerror",
                        lambda *a, **k: errors.append(a))
    me = SimpleNamespace(_plane0=p, _inputs_ready=lambda: True,
                         _event_windows=[(0.0, 1.0)], _event_token=old)
    _stamp_tab4(p)                          # Tab 4 re-run
    xt.CrossCorrelationTab._on_run_per_event(me)
    assert errors and "out of date" in errors[0][0]
    # A broadcast from the old render is ignored too.
    applied = []
    me._apply_event_windows = lambda e, token=None: applied.append(e)
    xt.CrossCorrelationTab._on_event_results_broadcast(
        me, {"plane0": str(p), "event_windows": [(0.0, 1.0)],
             "generation": old})
    assert applied == []



def _reload_and_export(p, cdir):
    """Reload a saved clustering, then export it, as Tab 6 does."""
    from scipy.cluster.hierarchy import linkage
    from calliope.tabs.clustering import tab as ct
    from calliope.tabs.clustering.logic import reload_clustering_offload
    out = reload_clustering_offload(str(p), "r0p7_filtered_", str(cdir))
    me = SimpleNamespace(
        _Z=linkage(np.random.default_rng(0).random((3, 4)), "ward"),
        _plane0=p, _prefix="r0p7_filtered_", _cluster_inputs=out["inputs"],
        _current_threshold=lambda: 1e9, _visual_to_label={},
        _compute_visual_cluster_map=lambda T: {1: 1},
        _linkage_ids_current=lambda: np.array([0, 1, 2]),
        _existing_cluster_dir=lambda pl, pr: pl / f"{pr}cluster_results")
    return out, lambda: ct.ClusteringTab._export_clusters(me)


def _saved_clusters(p):
    from scipy.cluster.hierarchy import linkage
    np.save(p / "ops.npy", {"Ly": 8, "Lx": 8}, allow_pickle=True)
    cdir = p / "r0p7_filtered_cluster_results"
    cdir.mkdir(exist_ok=True)
    np.save(cdir / "linkage.npy",
            linkage(np.random.default_rng(0).random((3, 4)), "ward"))
    np.save(cdir / "threshold_used.npy", np.array([1.0]))
    return cdir


def test_pre_stamp_clustering_reload_then_rebuild_is_refused(tmp_path):
    # Unstamped (pre-v1.0.2) clustering, reloaded, then the dF/F is
    # rebuilt with the same cells: exporting the old linkage must fail.
    p = _plane0(tmp_path, live=[1, 1, 1, 0], frozen=[1, 1, 1, 0])
    cdir = _saved_clusters(p)
    out, export = _reload_and_export(p, cdir)
    _rebuild(p)
    dr.compute_filtered_dff(p)
    with pytest.raises(ValueError, match="Re-run clustering"):
        export()


def test_null_stamped_clustering_after_unchanged_rewrite_exports(tmp_path):
    # A clustering stamped against a pre-stamp filtered dF/F ({"filtered":
    # null}), then an unchanged rewrite: still valid, so export works.
    p = _plane0(tmp_path, live=[1, 1, 1, 0], frozen=[1, 1, 1, 0])
    cdir = _saved_clusters(p)
    gen.stamp_clusters(cdir, gen.cluster_input(p, True))   # {"filtered": None}
    dr.compute_filtered_dff(p)                             # unchanged rewrite
    gen.check_clusters(p, cdir, True)
    out, export = _reload_and_export(p, cdir)
    gen.check_clusters(p, export(), True)
