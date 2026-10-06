"""Headless run_recording must accept crosscorrelation's 3-arg progress
callback ``progress_cb(done, total, label)``.

The lambda it passed took only ``(done, total)``; the real
crosscorrelation swallows the resulting TypeError, so progress lines
were silently lost. Every stage is stubbed here so only the callback
wiring is exercised.
"""

from __future__ import annotations

import types

from calliope.core import batch_pipeline as bp


def test_xcorr_progress_callback_takes_label(tmp_path, monkeypatch):
    plane0 = tmp_path / "plane0"
    monkeypatch.setattr(
        bp.preprocessing, "run_preprocess",
        lambda **kw: types.SimpleNamespace(shifted_folder=tmp_path,
                                           raw_paths=[]))
    monkeypatch.setattr(bp.detection_run, "run_detection",
                        lambda **kw: {"plane0": plane0})
    monkeypatch.setattr(bp, "write_export_manifest",
                        lambda *a, **kw: tmp_path / "manifest.json")
    monkeypatch.setattr(bp, "_no_rois_survive", lambda p: False)
    monkeypatch.setattr(bp, "_kept_roi_count", lambda p: 5)
    monkeypatch.setattr(bp.lowpass_run, "run_lowpass",
                        lambda *a, **kw: {"figures": []})
    monkeypatch.setattr(
        bp.event_detection_run, "run_event_detection",
        lambda *a, **kw: {"figures": [], "event_windows": [], "fps": 10.0})
    monkeypatch.setattr(bp.clustering, "run_clustering",
                        lambda *a, **kw: None)

    def fake_xcorr(plane0, params, *, event_windows, figures_dir,
                   progress_cb):
        progress_cb(1, 2, "C1xC2")

    monkeypatch.setattr(bp.crosscorrelation, "run_crosscorrelation",
                        fake_xcorr)
    monkeypatch.setattr(bp.spatial, "render_spatial_event_figures",
                        lambda *a, **kw: [])

    msgs: list[str] = []
    out = bp.run_recording([tmp_path / "a.tif"], str(tmp_path / "rec"),
                           {}, progress_cb=msgs.append)

    assert out["stages"]["crosscorrelation"]["status"] == "ok", \
        out["stages"]["crosscorrelation"]["error"]
    assert any("crosscorrelation: 1/2" in m for m in msgs)
