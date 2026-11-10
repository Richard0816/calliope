"""Regression tests for two output-figure bugs found 2026-06-15:

1. The lowpass population trace blew up to ~1e9 because it averaged dF/F
   across ROIs (mean-of-ratios dominated by near-zero-baseline ROIs).
   ``_population_median_filtered_dff`` must use the median and stay bounded.
2. ``crosscorrelation/per_event.png`` rendered "No pair data" because the
   per-event CSVs nest one level deeper than the full run, and the loader was
   pointed at the per-event root. ``_pick_representative_event_dir`` must find
   the right event subfolder.
"""
import numpy as np


def test_population_trace_is_median_not_mean(tmp_path):
    from calliope.core.lowpass_run import _population_median_filtered_dff

    T, N = 200, 12
    rng = np.random.default_rng(0)
    arr = rng.normal(0.5, 0.1, size=(T, N)).astype(np.float32)
    arr[:, 0] = 1e9  # one ROI with a near-zero baseline -> dF/F blows up
    arr.tofile(tmp_path / "r0p7_filtered_dff.memmap.float32")

    out = _population_median_filtered_dff(tmp_path, T, N)

    assert out.shape == (T,)
    assert np.all(np.isfinite(out))
    # median ignores the single 1e9 ROI; the plain mean would not.
    assert out.max() < 5.0
    assert np.mean(arr, axis=1).max() > 1e7


def test_pick_representative_event_dir(tmp_path):
    from calliope.core.crosscorrelation import _pick_representative_event_dir

    root = tmp_path / "cross_correlation_per_event"
    # Event with one cluster pair.
    (root / "event0000_a" / "C1xC1").mkdir(parents=True)
    (root / "event0000_a" / "C1xC1" / "C1xC1_summary.csv").write_text(
        "max_corr,best_lag_sec\n0.5,0.1\n")
    # Event with two cluster pairs -> should be picked.
    (root / "event0001_b" / "C1xC1").mkdir(parents=True)
    (root / "event0001_b" / "C1xC1" / "C1xC1_summary.csv").write_text(
        "max_corr,best_lag_sec\n0.5,0.1\n0.6,0.2\n")
    (root / "event0001_b" / "C1xC2").mkdir(parents=True)
    (root / "event0001_b" / "C1xC2" / "C1xC2_summary.csv").write_text(
        "max_corr,best_lag_sec\n0.4,-0.1\n")

    best = _pick_representative_event_dir(root)
    assert best is not None
    assert best.name == "event0001_b"

    # and the pair loader finds the nested pair CSVs from that event dir.
    from calliope.tabs.crosscorrelation.tab import _load_pair_data
    pair_data = _load_pair_data(best)
    assert set(pair_data.keys()) == {"C1xC1", "C1xC2"}


def test_pick_representative_event_dir_empty(tmp_path):
    from calliope.core.crosscorrelation import _pick_representative_event_dir

    root = tmp_path / "cross_correlation_per_event"
    root.mkdir(parents=True)
    assert _pick_representative_event_dir(root) is None


def test_tab7_export_plots_a_real_event(tmp_path, monkeypatch):
    """Tab 7's export (what Tab 0 batch calls) must plot an event subfolder,
    not the per-event root (which renders "No pair data")."""
    from types import SimpleNamespace
    import calliope.core.crosscorrelation as cc
    import calliope.core.utils as cu
    import calliope.core.export_manifest as em
    from calliope.tabs.crosscorrelation import tab as t7

    plane0 = tmp_path / "plane0"
    cr = plane0 / "r0p7_filtered_cluster_results"
    for sub in ("cross_correlation_full", "cross_correlation_per_event/event0003_x"):
        (cr / sub / "C1xC1").mkdir(parents=True)
        (cr / sub / "C1xC1" / "C1xC1_summary.csv").write_text(
            "zero_lag_corr,best_lag_sec\n0.5,0.1\n0.4,0.2\n")
    seen = {}
    monkeypatch.setattr(cc, "_save_violin_pair",
                        lambda pd_, save_path, title_suffix: seen.__setitem__(save_path.name, (pd_, title_suffix)))
    monkeypatch.setattr(cu, "save_folder_for_plane0", lambda p: tmp_path / "out")
    monkeypatch.setattr(em, "write_export_manifest", lambda *a, **k: tmp_path / "m.json")
    var = lambda v: SimpleNamespace(get=lambda: v, set=lambda *_: None)
    stub = SimpleNamespace(_plane0=plane0, prefix_var=var("r0p7_filtered_"),
                           cfolder_var=var(""), status_var=var(""))
    t7.CrossCorrelationTab._on_export_figures(stub, quiet=True)
    pe, title = seen["per_event.png"]
    assert set(pe) == {"C1xC1"} and "event0003_x" in title
