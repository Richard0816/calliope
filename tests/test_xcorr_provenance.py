"""Tracker #33 item 8: saved Tab 7 outputs carry the identity of the
clusters, dF/F and event windows they were computed from, and the violin
window / figure export refuse outputs made from other inputs."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from calliope.core import crosscorrelation as xc
from calliope.core import detection_run as dr
from calliope.core import generation as gen
from test_reapply_and_provenance import _plane0, _rebuild  # noqa: E402

PFX = "r0p7_filtered_"


def _recording(tmp_path):
    p = _plane0(tmp_path, live=[1, 1, 1, 1], frozen=[1, 1, 1, 1], T=200)
    _rebuild(p)
    rng = np.random.default_rng(0)
    full = np.memmap(p / "r0p7_dff.memmap.float32", np.float32, "r+",
                     shape=(200, 4))
    full[:] = rng.normal(size=(200, 4)).astype(np.float32)
    full.flush()
    del full
    dr.compute_filtered_dff(p)
    cdir = p / f"{PFX}cluster_results"
    cdir.mkdir()
    (cdir / "_indices_are_suite2p").touch()
    np.save(cdir / "C1_rois.npy", np.array([0, 1]))
    np.save(cdir / "C2_rois.npy", np.array([2, 3]))
    gen.stamp_clusters(cdir, gen.cluster_input(p, True))
    return p, cdir


def _full(p):
    return xc.run_cluster_xcorr_full_fast(
        p, prefix=PFX, fps=10.0, cluster_folder="", max_lag_seconds=0.5,
        use_gpu=False, n_shuffles=0)


def test_full_run_is_stamped_and_checked(tmp_path):
    p, cdir = _recording(tmp_path)
    out = _full(p)
    assert xc.check_xcorr_output(p, PFX, cdir, out) is True
    # another cluster assignment (e.g. a new Tab 6 cut): refused
    np.save(cdir / "C1_rois.npy", np.array([0, 2]))
    np.save(cdir / "C2_rois.npy", np.array([1, 3]))
    with pytest.raises(gen.StaleOutputError, match="other clusters"):
        xc.check_xcorr_output(p, PFX, cdir, out)


def test_other_dff_is_refused(tmp_path):
    p, cdir = _recording(tmp_path)
    out = _full(p)
    _rebuild(p)                       # re-detection: new dF/F build
    dr.compute_filtered_dff(p)
    with pytest.raises(gen.StaleOutputError):
        xc.check_xcorr_output(p, PFX, cdir, out)


def test_unfinished_run_is_refused_and_legacy_is_flagged(tmp_path):
    p, cdir = _recording(tmp_path)
    out = _full(p)
    xc._begin_xcorr(out)              # a run that died part-way
    with pytest.raises(gen.StaleOutputError, match="did not finish"):
        xc.check_xcorr_output(p, PFX, cdir, out)
    (out / xc.XCORR_STAMP).unlink()   # made before these stamps
    assert xc.check_xcorr_output(p, PFX, cdir, out) is False


def test_per_event_run_records_its_windows(tmp_path):
    p, cdir = _recording(tmp_path)
    w = [(2.0, 6.0), (10.0, 14.0)]
    out = xc.run_cluster_xcorr_per_event_fast(
        p, PFX, 10.0, "", w, max_lag_seconds=0.5, use_gpu=False)
    assert xc.check_xcorr_output(p, PFX, cdir, out, event_windows=w) is True
    with pytest.raises(gen.StaleOutputError, match="other event windows"):
        xc.check_xcorr_output(p, PFX, cdir, out, event_windows=[(2.0, 6.0)])


def _tab(p, monkeypatch):
    from calliope.tabs.crosscorrelation import tab as t7
    errors, status = [], []
    monkeypatch.setattr(t7.messagebox, "showerror", lambda *a, **k: errors.append(a))
    monkeypatch.setattr(t7.messagebox, "showinfo", lambda *a, **k: None)
    opened = []
    monkeypatch.setattr(t7, "ViolinWindow", lambda *a, **k: opened.append(k))
    me = SimpleNamespace(
        _plane0=p, prefix_var=SimpleNamespace(get=lambda: PFX),
        cfolder_var=SimpleNamespace(get=lambda: ""),
        status_var=SimpleNamespace(set=status.append),
        winfo_toplevel=lambda: None)
    me._xcorr_output_ok = lambda *a, **k: t7.CrossCorrelationTab._xcorr_output_ok(me, *a, **k)
    return t7, me, errors, status, opened


def test_violin_refuses_outputs_from_other_clusters(tmp_path, monkeypatch):
    p, cdir = _recording(tmp_path)
    _full(p)
    t7, me, errors, status, opened = _tab(p, monkeypatch)
    t7.CrossCorrelationTab._on_violin(me)
    assert opened and not errors                 # current: opens
    np.save(cdir / "C1_rois.npy", np.array([0, 2]))
    opened.clear()
    t7.CrossCorrelationTab._on_violin(me)
    assert not opened and errors                 # stale: refused


def test_figure_export_refuses_outputs_from_other_clusters(tmp_path, monkeypatch):
    p, cdir = _recording(tmp_path)
    _full(p)
    t7, me, errors, status, opened = _tab(p, monkeypatch)
    saved = []
    monkeypatch.setattr(xc, "_save_violin_pair", lambda *a, **k: saved.append(k))
    np.save(cdir / "C1_rois.npy", np.array([0, 2]))
    t7.CrossCorrelationTab._on_export_figures(me, quiet=True)
    assert not saved and "out of date" in status[-1]


def test_run_that_fails_part_way_is_refused(tmp_path, monkeypatch):
    p, cdir = _recording(tmp_path)
    out = _full(p)                    # a complete earlier run on disk

    def boom(*a, **k):
        raise RuntimeError("GPU fell over")

    monkeypatch.setattr(xc, "batch_xcorr_clusters", boom)
    with pytest.raises(RuntimeError):
        _full(p)
    # the half-overwritten folder is not passed off as the earlier run
    with pytest.raises(gen.StaleOutputError, match="did not finish"):
        xc.check_xcorr_output(p, PFX, cdir, out)


def test_rerun_with_fewer_clusters_drops_old_pairs(tmp_path):
    p, cdir = _recording(tmp_path)
    np.save(cdir / "C3_rois.npy", np.array([0, 2]))
    out = _full(p)
    assert any("C3" in d.name for d in out.iterdir() if d.is_dir())
    (cdir / "C3_rois.npy").unlink()           # new cut: no C3
    out = _full(p)
    assert not any("C3" in d.name for d in out.iterdir() if d.is_dir())
    assert xc.check_xcorr_output(p, PFX, cdir, out) is True


def test_per_event_rerun_drops_old_event_folders(tmp_path):
    p, cdir = _recording(tmp_path)
    w3 = [(2.0, 6.0), (8.0, 12.0), (14.0, 18.0)]
    out = xc.run_cluster_xcorr_per_event_fast(
        p, PFX, 10.0, "", w3, max_lag_seconds=0.5, use_gpu=False)
    n3 = len([d for d in out.iterdir() if d.is_dir()])
    out = xc.run_cluster_xcorr_per_event_fast(
        p, PFX, 10.0, "", w3[:1], max_lag_seconds=0.5, use_gpu=False)
    n1 = len([d for d in out.iterdir() if d.is_dir()])
    assert n3 > n1 == 1


def test_export_leaves_out_per_event_for_other_windows(tmp_path, monkeypatch):
    p, cdir = _recording(tmp_path)
    _full(p)
    w = [(2.0, 6.0), (10.0, 14.0)]
    xc.run_cluster_xcorr_per_event_fast(p, PFX, 10.0, "", w,
                                        max_lag_seconds=0.5, use_gpu=False)
    from calliope.core import utils as cu
    monkeypatch.setattr(cu, "save_folder_for_plane0", lambda q: tmp_path / "out")
    xc_dir = tmp_path / "out" / "calliope_figures" / "crosscorrelation"
    xc_dir.mkdir(parents=True)
    for ext in (".png", ".svg"):
        (xc_dir / f"per_event{ext}").write_text("old")
    t7, me, errors, status, opened = _tab(p, monkeypatch)
    saved = []
    monkeypatch.setattr(xc, "_save_violin_pair",
                        lambda pair, save_path, title_suffix: saved.append(save_path.name))
    me._event_windows = [(2.0, 6.0)]          # Tab 5 changed the events
    t7.CrossCorrelationTab._on_export_figures(me, quiet=True)
    assert saved == ["full.png"]
    assert not (xc_dir / "per_event.png").exists()
    assert "per-event figure left out" in status[-1]
    me._event_windows = w                      # matching windows: exported
    saved.clear()
    t7.CrossCorrelationTab._on_export_figures(me, quiet=True)
    assert saved == ["full.png", "per_event.png"]


def test_unverified_warning_survives_export(tmp_path, monkeypatch):
    p, cdir = _recording(tmp_path)
    out = _full(p)
    (out / xc.XCORR_STAMP).unlink()            # made before these stamps
    from calliope.core import utils as cu
    monkeypatch.setattr(cu, "save_folder_for_plane0", lambda q: tmp_path / "out")
    t7, me, errors, status, opened = _tab(p, monkeypatch)
    monkeypatch.setattr(xc, "_save_violin_pair", lambda *a, **k: None)
    me._event_windows = []
    t7.CrossCorrelationTab._on_export_figures(me, quiet=True)
    assert status[-1].startswith("Exported") and "not verified" in status[-1]


def test_per_event_stamp_without_windows_is_refused(tmp_path):
    # A per-event folder must say which windows it was computed for;
    # one that doesn't can't be passed off as matching the current events.
    p, cdir = _recording(tmp_path)
    w = [(2.0, 6.0)]
    out = xc.run_cluster_xcorr_per_event_fast(
        p, PFX, 10.0, "", w, max_lag_seconds=0.5, use_gpu=False)
    xc._stamp_xcorr(out, xc._xcorr_inputs_id(p, PFX, cdir))   # no "events"
    with pytest.raises(gen.StaleOutputError, match="other event windows"):
        xc.check_xcorr_output(p, PFX, cdir, out, event_windows=w)


def test_rerun_keeps_folders_it_did_not_write(tmp_path):
    p, cdir = _recording(tmp_path)
    out = _full(p)
    mine = out / "my_notes"
    mine.mkdir()
    (mine / "plot.png").write_text("keep me")
    out = _full(p)
    assert (mine / "plot.png").read_text() == "keep me"


def test_backup_folders_are_kept_but_never_read(tmp_path):
    from calliope.tabs.crosscorrelation.tab import _load_pair_data
    p, cdir = _recording(tmp_path)
    out = _full(p)
    backup = out / "backup"
    backup.mkdir()
    (backup / "C1xC1_summary.csv").write_text("zero_lag_corr,best_lag_sec\n9,9\n")
    out = _full(p)
    assert (backup / "C1xC1_summary.csv").exists()          # kept
    assert "backup" not in _load_pair_data(out)             # not read
    assert set(_load_pair_data(out)) == xc.generated_dirs(out)


def test_representative_event_ignores_other_folders(tmp_path):
    p, cdir = _recording(tmp_path)
    w = [(2.0, 6.0)]
    out = xc.run_cluster_xcorr_per_event_fast(
        p, PFX, 10.0, "", w, max_lag_seconds=0.5, use_gpu=False)
    big = out / "zz_backup" / "C1xC2"                       # more CSVs
    big.mkdir(parents=True)
    for k in range(5):
        (big / f"x{k}_summary.csv").write_text("a\n1\n")
    rep = xc._pick_representative_event_dir(out)
    assert rep is not None and rep.name.startswith("event0000_")
