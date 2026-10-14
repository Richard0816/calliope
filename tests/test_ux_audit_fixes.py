"""Regression tests for the 2026-10-14 GUI walkthrough audit (docs/ux_audit).

Headless: no suite2p run, no GPU, no real recording.
"""
from __future__ import annotations

import numpy as np
import pytest

from calliope.core import suite2p_pipeline as sp


def _cfg(tmp_path, monkeypatch, **kw):
    class _Sizer:
        def pick(self, _folder):
            return 777

    monkeypatch.setattr(sp.utils, "get_adaptive_sizer", lambda: _Sizer())
    monkeypatch.setattr(sp.utils, "change_nbinned_according_to_free_ram",
                        lambda _f: 5000)
    monkeypatch.setattr(sp.utils, "_peek_tiff_frame_shape", lambda _f: None)
    return sp.Suite2pPipelineConfig(tiff_folder=str(tmp_path),
                                    save_folder=str(tmp_path),
                                    verbose=False, **kw)


# --- finding 2: a user-set batch_size is no longer overwritten ---

@pytest.mark.parametrize("override, expected", [
    (None, 777),                                    # unset -> auto-pick
    ({"registration": {"batch_size": 200}}, 200),   # popout edit wins
])
def test_batch_size_override_respected(tmp_path, monkeypatch, override,
                                       expected):
    cfg = _cfg(tmp_path, monkeypatch, settings_override=override)
    _db, settings = sp.load_base_settings(cfg)
    assert sp.get_setting(settings, "batch_size") == expected


# --- finding 1: budget follows available RAM; an OOM kill spares the GUI ---

def test_batch_budget_uses_available_ram(monkeypatch):
    from types import SimpleNamespace
    from calliope.core import utils
    gib = 1024 ** 3
    monkeypatch.setattr(utils.psutil, "virtual_memory",
                        lambda: SimpleNamespace(total=32 * gib,
                                                available=4 * gib))
    monkeypatch.setattr(utils, "_peek_tiff_frame_shape", lambda _f: (256, 256))
    monkeypatch.setattr(utils, "pick_batch_size_gpu", lambda *a, **k: None)
    # 0.8 * 4 GiB available, not 0.8 * 32 - 4 GiB total.
    assert utils._ram_budget_bytes(0.8, 4.0) == pytest.approx(3.2 * gib)
    per_frame = 256 * 256 * 2 * utils.RAM_OVERHEAD_MULTIPLIER
    assert utils.pick_batch_size("x", floor=1) == int(3.2 * gib // per_frame)


def _child_ok(progress_q=None):
    from calliope.core.offload import make_progress_cb
    make_progress_cb(progress_q)("hello")
    return 42


def _child_killed(progress_q=None):
    import os
    os._exit(137)  # what an OS out-of-memory kill looks like to the parent


def test_call_in_child_returns_and_streams_progress():
    from calliope.core.offload import call_in_child
    seen = []
    assert call_in_child(_child_ok, on_progress=seen.append) == 42
    assert seen == ["hello"]


# --- finding 3: frame rate / pixel size come from the TIFF metadata ---

def test_imagej_and_ome_metadata(tmp_path):
    import numpy as np
    import tifffile
    from calliope.core import acquisition
    mov = np.zeros((4, 8, 8), np.uint16)
    ij = tmp_path / "ij.tif"
    tifffile.imwrite(ij, mov, imagej=True, resolution=(2.0, 2.0),
                     metadata={"axes": "TYX", "finterval": 1 / 30,
                               "unit": "micron"})
    md = acquisition.read_tiff_metadata(ij)
    assert md["fps"] == pytest.approx(30.0)
    assert md["um_per_px"] == pytest.approx(0.5)
    assert "ImageJ" in md["source"]
    ome = tmp_path / "ome.ome.tif"
    tifffile.imwrite(ome, mov, ome=True,
                     metadata={"axes": "TYX", "TimeIncrement": 0.05,
                               "PhysicalSizeX": 1.25})
    md = acquisition.read_tiff_metadata(ome)
    assert md["fps"] == pytest.approx(20.0)
    assert md["um_per_px"] == pytest.approx(1.25)
    assert acquisition.read_tiff_metadata(tmp_path / "missing.tif") == {}


def test_scanimage_rate_divides_by_averaging():
    from types import SimpleNamespace
    from calliope.core import acquisition
    tf = SimpleNamespace(scanimage_metadata={"FrameData": {
        "SI.hRoiManager.scanFrameRate": 30.0,
        "SI.hScan2D.logAverageFactor": 2}})
    assert acquisition._scanimage(tf) == {"fps": 15.0}


def test_fill_prefers_raw_tiff_and_keeps_user_value(tmp_path):
    import numpy as np
    import tifffile
    from calliope.core import acquisition, preprocessing
    raw = tmp_path / "raw" / "rec.tif"
    raw.parent.mkdir()
    tifffile.imwrite(raw, np.zeros((4, 8, 8), np.int16), imagej=True,
                     metadata={"axes": "TYX", "finterval": 1 / 30})
    rec = tmp_path / "rec"
    rec.mkdir()
    # The shifted TIFF carries no acquisition metadata.
    tifffile.imwrite(rec / "rec_shifted.tif", np.zeros((4, 8, 8), np.uint16))
    preprocessing.write_raw_paths_sidecar(rec, [raw])
    logs = []
    out = acquisition.fill_from_metadata({"fps_override": 0.0}, rec,
                                         log=logs.append)
    assert out["fps_override"] == pytest.approx(30.0)
    assert any("30 Hz" in m for m in logs)
    kept = acquisition.fill_from_metadata({"fps_override": 12.5}, rec,
                                          log=logs.append)
    assert kept["fps_override"] == 12.5


# --- finding 6: within-cluster xcorr uses distinct pairs only ---

def test_within_cluster_xcorr_has_no_self_or_duplicate_pairs(tmp_path):
    import numpy as np
    import pandas as pd
    from calliope.core import crosscorrelation as xc
    prefix, T, N = "r0p7_filtered_", 300, 5
    plane0 = tmp_path / "plane0"
    plane0.mkdir()
    np.save(plane0 / "F.npy", np.zeros((N, T), np.float32))
    np.save(plane0 / "r0p7_cell_mask_bool.npy", np.ones(N, bool))
    mm = np.memmap(plane0 / f"{prefix}dff.memmap.float32", dtype="float32",
                   mode="w+", shape=(T, N))
    mm[:] = np.random.default_rng(0).standard_normal((T, N))
    mm.flush()
    del mm
    cdir = plane0 / f"{prefix}cluster_results"
    cdir.mkdir()
    (cdir / "_indices_are_suite2p").touch()
    np.save(cdir / "C1_rois.npy", np.array([0, 1, 2]))
    np.save(cdir / "C2_rois.npy", np.array([3]))        # singleton
    out = xc.run_cluster_xcorr_full_fast(
        plane0, prefix, 15.0, "", max_lag_seconds=0.5, use_gpu=False,
        n_shuffles=20)
    df = pd.read_csv(out / "C1xC1" / "C1xC1_summary.csv")
    assert sorted(zip(df["roiA"], df["roiB"])) == [(0, 1), (0, 2), (1, 2)]
    assert df["p_value_fdr"].notna().all()
    assert not (out / "C1xC2").exists() and not (out / "C2xC2").exists()


def test_lag_significance_is_roi_level():
    import pandas as pd
    from calliope.tabs.crosscorrelation.tab import _roi_level_lag_pvalue
    within = pd.DataFrame({"roiA": [0, 0, 1], "roiB": [1, 2, 2],
                           "best_lag_sec": [0.2, 0.2, 0.2]})
    assert np.isnan(_roi_level_lag_pvalue("C1xC1", within))
    # One ROI of A paired with 26 ROIs of B is one observation, not 26.
    one_cell = pd.DataFrame({"roiA": [7] * 26, "roiB": range(26),
                             "best_lag_sec": [0.3] * 26})
    assert np.isnan(_roi_level_lag_pvalue("C1xC2", one_cell))


# --- finding 7: the auto cut ignores tiny clusters when counting ---

def test_auto_cut_does_not_count_singletons():
    from scipy.cluster.hierarchy import fcluster, linkage
    from calliope.core.clustering import auto_choose_threshold
    rng = np.random.default_rng(0)
    centres = [(0, 0), (10, 0), (0, 10), (10, 10)]
    X = np.vstack([rng.normal(c, 0.5, (10, 2)) for c in centres]
                  + [[[100.0, 100.0]]])                # one far outlier
    Z = linkage(X, "ward")

    def groups(frac):
        labels = fcluster(Z, t=frac * Z[:, 2].max(), criterion="distance")
        return [len(set(labels[i * 10:(i + 1) * 10])) for i in range(4)], \
            len(set(labels[:40]))

    # Old behaviour: stop at 4 clusters = outlier + 3 (two groups merged).
    _, n_real_old = groups(auto_choose_threshold(Z, min_cluster_size=1))
    assert n_real_old == 3
    per_group, n_real = groups(auto_choose_threshold(Z))
    assert n_real == 4 and per_group == [1, 1, 1, 1]


# --- finding 12: zero merged ROIs is no_rois, not failed ---

def test_headless_batch_zero_merged_rois_is_no_rois(tmp_path, monkeypatch):
    import types
    from calliope.core import batch_pipeline as bp
    from calliope.core.sparse_plus_cellpose import NoRoisDetected
    monkeypatch.setattr(
        bp.preprocessing, "run_preprocess",
        lambda **kw: types.SimpleNamespace(shifted_folder=tmp_path,
                                           raw_paths=[]))

    def no_rois(**kw):
        raise NoRoisDetected("Sparsery and Cellpose found no ROIs")

    monkeypatch.setattr(bp.detection_run, "run_detection", no_rois)
    out = bp.run_recording([tmp_path / "a.tif"], str(tmp_path / "rec"), {})
    assert out["status"] == "no_rois"
    assert out["stages"]["detection"]["status"] == "no_rois"
    assert out["stages"]["lowpass"]["status"] == "skipped"


def test_gui_batch_zero_merged_rois_is_no_rois():
    import time
    from types import SimpleNamespace
    from calliope.tabs.batch.tab import BatchTab
    s = SimpleNamespace(STAGES=BatchTab.STAGES, logs=[], done=False,
                        _stage_t0=time.time(),
                        _row_results={"stages": {}, "status": "running"})
    s._append_log = s.logs.append
    s._on_row_done = lambda: setattr(s, "done", True)
    s._skip_row_no_rois = lambda reason: BatchTab._skip_row_no_rois(s, reason)
    # What Tab 3 reports when the detection child raises NoRoisDetected.
    BatchTab._fail_stage(s, "detection", RuntimeError(
        "calliope.core.sparse_plus_cellpose.NoRoisDetected: Sparsery and "
        "Cellpose found no ROIs"))
    assert s._row_results["status"] == "no_rois" and s.done
    assert s._row_results["stages"]["detection"]["status"] == "no_rois"
    assert any("found no ROIs" in m for m in s.logs)


def test_child_error_message_leads_with_exception_line():
    from calliope.core.offload import call_in_child
    with pytest.raises(RuntimeError) as ei:
        call_in_child(_child_raises)
    assert str(ei.value).splitlines()[0] == "ValueError: bad input"


def _child_raises(progress_q=None):
    raise ValueError("bad input")


# --- finding 10: Tab 6 Export tells Tab 7 the cluster files exist ---

def test_cluster_export_publishes_clusters_ready(tmp_path):
    from types import SimpleNamespace
    from calliope.gui_common import AppState
    from calliope.tabs.clustering.tab import ClusteringTab
    state, seen = AppState(), []
    state.subscribe_clusters_ready(seen.append)
    stub = SimpleNamespace(
        state=state, _plane0=tmp_path,
        _export_clusters=lambda: tmp_path / "out",
        _write_summary=lambda: None,
        status_var=SimpleNamespace(set=lambda v: None))
    ClusteringTab._on_export(stub)
    assert seen == [tmp_path]


# --- finding 11: "Open run folder" starts every analysis tab fresh ---

def test_rebuild_analysis_tabs_replaces_tabs_and_listeners():
    import traceback
    import customtkinter as ctk
    from calliope.pipeline_gui import PipelineApp
    ctk.set_appearance_mode("dark")
    app = PipelineApp()
    app.withdraw()
    errors = []
    app.report_callback_exception = lambda *e: errors.append(
        "".join(traceback.format_exception(*e)))
    try:
        app._show_tab(8)
        st = app.state_obj
        counts = {k: len(v) for k, v in vars(st).items()
                  if k.endswith("_listeners")}
        old_tabs = list(app._tabs_in_order)
        app.qc_tab.status_marker = "previous recording"
        app._rebuild_analysis_tabs()
        app._show_tab(2)
        for _ in range(5):
            app.update()
        assert app.batch_tab is old_tabs[0]
        assert all(new is not old for new, old in
                   zip(app._tabs_in_order[1:], old_tabs[1:]))
        assert not hasattr(app.qc_tab, "status_marker")
        assert not old_tabs[8].winfo_exists()
        # Fresh tabs re-subscribed exactly once; no stale callbacks remain.
        assert {k: len(v) for k, v in vars(st).items()
                if k.endswith("_listeners")} == counts
        assert errors == []
    finally:
        app.destroy()


# --- finding 13: batch Scan groups multi-part recordings ---

def test_group_multipart_tiffs(tmp_path):
    from pathlib import Path
    from calliope.tabs.batch.logic import group_multipart_tiffs
    a, b = tmp_path / "a", tmp_path / "b"
    paths = ([a / f"s75_PTZexp1_00001_{i:05d}.tif" for i in (2, 1, 10)]
             + [a / "rec_split_part002.tif", a / "rec_split_part001.tif",
                a / "rec_00018.tif", a / "rec_00019.tif",   # separate acqs
                b / "s75_PTZexp1_00001_00003.tif"])          # other folder
    got = {ident: [Path(p).name for p in parts]
           for ident, parts in group_multipart_tiffs(paths)}
    assert got["s75_PTZexp1_00001"] == [
        "s75_PTZexp1_00001_00001.tif", "s75_PTZexp1_00001_00002.tif",
        "s75_PTZexp1_00001_00010.tif"]
    assert got["rec_split"] == ["rec_split_part001.tif",
                                "rec_split_part002.tif"]
    assert got["rec_00018"] == ["rec_00018.tif"]
    assert got["rec_00019"] == ["rec_00019.tif"]
    # A lone part in another folder stays its own row.
    assert got["s75_PTZexp1_00001_00003"] == ["s75_PTZexp1_00001_00003.tif"]


# --- finding 14: import an existing Suite2p result ---

def _fake_suite2p(n=4, T=600, fs=30.0):
    rng = np.random.default_rng(0)
    stat = [{"ypix": np.arange(3) + 5 * i, "xpix": np.arange(3) + 5 * i,
             "lam": np.ones(3, np.float32), "med": [5 * i + 1, 5 * i + 1],
             "npix": 3, "radius": 1.0} for i in range(n)]
    F = (100 + rng.standard_normal((n, T))).astype(np.float32)
    ops = {"fs": fs, "Ly": 32, "Lx": 32, "meanImg": np.ones((32, 32))}
    return {"F": F, "Fneu": F * 0.5, "spks": np.zeros_like(F),
            "iscell": np.c_[np.ones(n), np.ones(n)], "stat": stat, "ops": ops}


@pytest.mark.parametrize("kind", ["plane0", "fall_mat"])
def test_import_suite2p_builds_a_loadable_run(tmp_path, kind):
    import scipy.io as sio
    from calliope.core import run_loader, suite2p_import
    d = _fake_suite2p()
    if kind == "plane0":
        src = tmp_path / "s2p" / "suite2p" / "plane0"
        src.mkdir(parents=True)
        for k in ("F", "Fneu", "spks", "iscell"):
            np.save(src / f"{k}.npy", d[k])
        np.save(src / "stat.npy", np.array(d["stat"], dtype=object),
                allow_pickle=True)
        np.save(src / "ops.npy", d["ops"], allow_pickle=True)
    else:
        src = tmp_path / "Fall.mat"
        sio.savemat(src, {**{k: d[k] for k in ("F", "Fneu", "spks",
                                                "iscell", "ops")},
                          "stat": np.array(d["stat"], dtype=object)})
    run = tmp_path / "run"
    suite2p_import.import_suite2p(str(src), str(run), ckpt_path="",
                                  progress_q=None)
    inv = run_loader.scan_run(run)
    assert inv.has_detection
    plane0 = run_loader.plane0_for(run)
    assert (plane0 / "r0p7_filtered_dff.memmap.float32").is_file()
    assert np.load(plane0 / "ops.npy", allow_pickle=True).item()["fs"] == 30


# --- finding 9: Cellpose model preflight, skip option, local model ---

def test_cellpose_model_problem(tmp_path, monkeypatch):
    monkeypatch.setattr(sp._cp_models, "MODEL_DIR", tmp_path)
    assert sp.cellpose_model_problem({"cellpose_enabled": False}) is None
    assert "not downloaded" in sp.cellpose_model_problem({})
    assert "not found" in sp.cellpose_model_problem(
        {"cellpose_model_path": str(tmp_path / "nope")})
    (tmp_path / "cpsam").write_bytes(b"x")
    assert sp.cellpose_model_problem({}) is None
    assert sp.cellpose_model_problem(
        {"cellpose_model_path": str(tmp_path / "cpsam")}) is None


def test_cellpose_load_failure_explains_itself(tmp_path, monkeypatch):
    plane0 = tmp_path / "shared"
    plane0.mkdir()
    np.save(plane0 / "ops.npy", {"meanImg": np.random.rand(32, 32),
                                 "Ly": 32, "Lx": 32}, allow_pickle=True)

    def blocked(*a, **k):
        raise OSError("<urlopen error Tunnel connection failed: 403>")

    monkeypatch.setattr(sp._cp_models, "CellposeModel", blocked)
    monkeypatch.setattr(sp, "cellpose_model_problem", lambda cfg: None)
    with pytest.raises(RuntimeError, match="huggingface.co.*Sparsery only"):
        sp.run_cellpose_pass(tmp_path / "cp", plane0, {}, verbose=False)


# --- finding 16: missing tkinter gives an install hint, not a traceback ---

def test_missing_tkinter_exits_with_hint():
    import subprocess
    import sys
    r = subprocess.run(
        [sys.executable, "-c", "import sys; sys.modules['tkinter'] = None; "
                               "import calliope.pipeline_gui"],
        capture_output=True, text=True, timeout=120)
    assert r.returncode == 1
    assert "needs tkinter" in r.stderr and "Traceback" not in r.stderr


def test_call_in_child_survives_killed_child():
    from calliope.core.offload import call_in_child
    with pytest.raises(RuntimeError, match=r"exit code 137.*out of memory"):
        call_in_child(_child_killed, died_hint="ran out of memory")
