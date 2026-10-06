"""Regression tests for Tab 3 (Suite2pTab) worker wiring.

Covers the GUI-only drift fixed in the 2026-10-05 legacy audit:
- the FPS override reaches suite2p (``spc.run(fps_override=...)``),
- the GUI dF/F stamps meta.json ``dff`` provenance (shared core path),
- spatial_scale + GCaMP label are stamped into meta.json,
- the NWB export worker never touches Tk,
- zdrift_qc / keep_registration_cache are settable PARAM_SPEC fields.

Every test calls the real unbound ``Suite2pTab`` method with a stub
``self``, so no Tk root is needed.
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import types
from types import SimpleNamespace

import numpy as np
import pytest

from calliope.gui_common import spec_defaults
from calliope.tabs.suite2p import tab as tab_mod
from calliope.tabs.suite2p.tab import Suite2pTab


class _StopAfterSpcRun(Exception):
    pass


def _run_pipeline_capture_spc_kwargs(monkeypatch, tmp_path, fps_override):
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        raise _StopAfterSpcRun

    fake_spc = types.ModuleType("calliope.core.sparse_plus_cellpose")
    fake_spc.run = fake_run
    import calliope.core as core_pkg
    monkeypatch.setitem(sys.modules, "calliope.core.sparse_plus_cellpose",
                        fake_spc)
    monkeypatch.setattr(core_pkg, "sparse_plus_cellpose", fake_spc,
                        raising=False)

    params = spec_defaults(Suite2pTab.PARAM_SPEC)
    params["fps_override"] = fps_override
    stub = SimpleNamespace(_params=params, _settings_override={},
                           _log_queue=queue.Queue())
    with pytest.raises(_StopAfterSpcRun):
        Suite2pTab._run_pipeline(
            stub, tiff_folder=tmp_path, save_folder=tmp_path / "det",
            ckpt_path=None, rec_id="rec", baseline_mode="first_n",
            baseline_min=2.0, tau_override=0.25)
    return captured


def test_gui_forwards_fps_override_to_suite2p(monkeypatch, tmp_path):
    kw = _run_pipeline_capture_spc_kwargs(monkeypatch, tmp_path, 30.0)
    assert kw["fps_override"] == 30.0


def test_gui_zero_fps_override_means_none(monkeypatch, tmp_path):
    kw = _run_pipeline_capture_spc_kwargs(monkeypatch, tmp_path, 0.0)
    assert kw["fps_override"] is None


def _tiny_plane0(tmp_path, N=4, T=120):
    plane0 = tmp_path / "plane0"
    plane0.mkdir()
    rng = np.random.default_rng(0)
    F = (100 + rng.normal(0, 1, (N, T))).astype(np.float32)
    np.save(plane0 / "F.npy", F)
    np.save(plane0 / "Fneu.npy", (F * 0.1).astype(np.float32))
    return plane0, N, T


def test_gui_dff_stamps_meta_provenance(tmp_path):
    plane0, N, T = _tiny_plane0(tmp_path)
    params = spec_defaults(Suite2pTab.PARAM_SPEC)
    params.update(fps_override=30.0, use_gpu_dff=False, neuropil_coef=0.6)
    stub = SimpleNamespace(_params={}, _gpu_dff_eligible=lambda p, m: False)
    Suite2pTab._run_dff(stub, plane0, "first_n", 0.05, params=params)

    meta = json.loads((plane0 / "meta.json").read_text(encoding="utf-8"))
    assert meta["dff"]["fps"] == 30.0
    assert meta["dff"]["baseline_mode"] == "first_n"
    assert meta["extraction"]["neuropil_coef"] == 0.6
    dff = np.memmap(plane0 / "r0p7_dff.memmap.float32", dtype="float32",
                    mode="r", shape=(T, N))
    assert np.isfinite(np.asarray(dff)).all()


def test_stamp_run_meta_records_spatial_scale_and_label(tmp_path):
    Suite2pTab._stamp_run_meta(tmp_path, {"spatial_scale": 2},
                               "jGCaMP8m (tau = 0.25 s)")
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["detection"]["spatial_scale"] == 2
    assert meta["extraction"]["gcamp_variant_label"].startswith("jGCaMP8m")


def test_cellfilter_delegates_to_core(monkeypatch, tmp_path):
    from calliope.core import detection_run
    seen = {}

    def fake_predict(plane0, ckpt_path, rec_id, size_outlier_z):
        seen.update(plane0=plane0, ckpt=ckpt_path, rec=rec_id,
                    z=size_outlier_z)

    monkeypatch.setattr(detection_run, "predict_cell_filter", fake_predict)
    Suite2pTab._run_cellfilter(SimpleNamespace(), tmp_path, "c.pt", "rec")
    from calliope.core.cellfilter.config import SIZE_OUTLIER_Z
    assert seen == dict(plane0=tmp_path, ckpt="c.pt", rec="rec",
                        z=SIZE_OUTLIER_Z)


class _TkRecorder:
    """Stands in for a Tk widget/var and records the calling thread."""

    def __init__(self):
        self.threads = []

    def set(self, *_a, **_k):
        self.threads.append(threading.current_thread())

    configure = set


def test_nwb_worker_does_not_touch_tk(monkeypatch, tmp_path):
    from calliope.core import nwb_export
    monkeypatch.setattr(nwb_export, "nwb_available", lambda: True)
    monkeypatch.setattr(nwb_export, "export_recording_to_nwb",
                        lambda plane0, out: out)
    monkeypatch.setattr(tab_mod.filedialog, "asksaveasfilename",
                        lambda **_k: str(tmp_path / "x.nwb"))
    started = []
    real_thread = threading.Thread

    def thread_factory(*a, **k):
        t = real_thread(*a, **k)
        started.append(t)
        return t

    monkeypatch.setattr(tab_mod.threading, "Thread", thread_factory)
    btn, status = _TkRecorder(), _TkRecorder()
    stub = SimpleNamespace(_final_plane0=tmp_path / "a" / "b" / "c" / "d",
                           nwb_btn=btn, status_var=status,
                           _log_queue=queue.Queue())
    Suite2pTab._on_export_nwb(stub)
    for t in started:
        t.join(timeout=10)
    main = threading.main_thread()
    assert all(t is main for t in btn.threads + status.threads)
    kind, payload = stub._log_queue.get_nowait()
    assert kind == "nwb_done" and payload.endswith("x.nwb")


def test_zdrift_and_cache_toggles_are_param_spec_fields():
    d = spec_defaults(Suite2pTab.PARAM_SPEC)
    assert d["zdrift_qc"] is True
    assert d["keep_registration_cache"] is False
    groups = {e["name"]: e["group"] for e in Suite2pTab.PARAM_SPEC}
    assert groups["keep_registration_cache"] == "Re-detect"
