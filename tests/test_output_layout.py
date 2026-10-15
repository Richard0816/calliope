"""output/ layout: toggles, trace/table writers, README, old-run migration."""
import numpy as np
import pandas as pd

from calliope.core import output_layout as ol


def test_settings_defaults_roundtrip_and_unknown_keys(tmp_path):
    f = tmp_path / "exports.json"
    s = ol.load_settings(f)
    assert s["event_windows"] and not s["trace_dff"] and not s[ol.COMPACT_KEY]
    s["trace_dff"] = True
    s["bogus"] = True
    ol.save_settings(s, f)
    again = ol.load_settings(f)
    assert again["trace_dff"] and "bogus" not in again


def test_write_trace_csv_time_down_and_compact(tmp_path):
    tr = np.arange(12, dtype=np.float32).reshape(6, 2)
    p = ol.write_trace(tmp_path, 3, "dff", tr, fps=2.0, roi_ids=[7, 31])
    df = pd.read_csv(p)
    assert list(df.columns) == ["time_s", "roi_7", "roi_31"]
    assert df["time_s"].tolist() == [0, 0.5, 1, 1.5, 2, 2.5]
    assert df["roi_31"].tolist() == [1, 3, 5, 7, 9, 11]
    assert p.parent == tmp_path / "output" / "data" / "3_detection"
    q = ol.write_trace(tmp_path, 3, "dff", tr, 2.0, [7, 31], compact=True)
    assert np.array_equal(np.load(q), tr)
    assert np.load(q.with_name("dff_roi_ids.npy")).tolist() == [7, 31]


def test_migrate_old_run_moves_once_and_writes_readme(tmp_path):
    p0 = tmp_path / "detection" / "final" / "suite2p" / "plane0"
    p0.mkdir(parents=True)
    (p0 / "calliope_summary.xlsx").write_bytes(b"x")
    ev = tmp_path / "calliope_figures" / "event_detection"
    ev.mkdir(parents=True)
    (ev / "raster.png").write_bytes(b"png")
    (tmp_path / "calliope_figures" / "manifest.json").write_text("{}")
    moves = ol.migrate_run(tmp_path, log=lambda *_: None)
    out = tmp_path / "output"
    assert len(moves) == 3
    assert (out / "calliope_summary.xlsx").exists()
    assert (out / "figures" / "5_events" / "raster.png").exists()
    assert (out / "manifest.json").exists()
    assert not (tmp_path / "calliope_figures").exists()
    assert "figures/5_events/raster.png" in (out / "README.txt").read_text()
    assert ol.migrate_run(tmp_path, log=lambda *_: None) == []  # idempotent


def test_summary_workbook_goes_to_output_and_adopts_old_copy(tmp_path):
    from calliope.core import summary_writer as sw
    p0 = tmp_path / "rec" / "detection" / "final" / "suite2p" / "plane0"
    p0.mkdir(parents=True)
    sw.write_sheets(p0.parent, {"Old": {"a": [1]}})       # bare folder: in place
    assert (p0.parent / "calliope_summary.xlsx").exists()
    old = p0 / "calliope_summary.xlsx"
    (p0.parent / "calliope_summary.xlsx").rename(old)        # a pre-output/ run
    assert sw.existing_summary_path(p0) == old
    path = sw.write_sheet(p0, "New", {"b": [2]})
    assert path == tmp_path / "rec" / "output" / "calliope_summary.xlsx"
    assert not old.exists()
    assert {"Old", "New"} <= set(pd.read_excel(path, sheet_name=None))
