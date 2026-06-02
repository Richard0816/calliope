"""RAM out-of-memory auto-retry in the batch runner.

When a recording's stage worker dies with a memory-allocation error, the
batch driver re-runs the WHOLE recording with progressively more
conservative resources (smaller suite2p batch_size via the adaptive
sizer's conservative_scale, GPU off, small nbins, coarser QC downsample)
before giving up. These tests exercise the pure pieces against stubs --
no Tk, no real OOM, no suite2p.
"""

from __future__ import annotations

import threading
import types

from calliope.core.utils import AdaptiveBatchSizer
from calliope.tabs.batch.tab import BatchTab


# --- OOM classifier -------------------------------------------------------

def test_is_oom_error_matches_ram_signatures():
    f = BatchTab._is_oom_error
    stub = types.SimpleNamespace(_OOM_SIGNATURES=BatchTab._OOM_SIGNATURES)
    for msg in [
        "MemoryError",
        "numpy.core._exceptions._ArrayMemoryError: Unable to allocate 4.50 GiB",
        "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB",
        "cudaErrorMemoryAllocation",
        "std::bad_alloc",
    ]:
        assert f(stub, msg) is True, msg


def test_is_oom_error_ignores_disk_and_generic():
    f = BatchTab._is_oom_error
    stub = types.SimpleNamespace(_OOM_SIGNATURES=BatchTab._OOM_SIGNATURES)
    for msg in [
        "OSError: [Errno 28] No space left on device",
        "[WinError 112] There is not enough space on the disk",
        "ValueError: something unrelated",
        "",
        None,
    ]:
        assert f(stub, msg) is False, msg


# --- AdaptiveBatchSizer.conservative_scale --------------------------------

def test_apply_conservative_scales_and_floors():
    s = AdaptiveBatchSizer(floor=100, ceiling=8000)
    s.conservative_scale = 1.0
    assert s._apply_conservative(1000) == 1000
    s.conservative_scale = 0.25
    assert s._apply_conservative(1000) == 250
    s.conservative_scale = 0.001
    assert s._apply_conservative(1000) == 100  # floored


def test_pick_applies_conservative_scale():
    base = AdaptiveBatchSizer().pick(None)        # scale 1.0
    s = AdaptiveBatchSizer()
    s.conservative_scale = 0.5
    scaled = s.pick(None)
    assert scaled == max(s.floor, round(base * 0.5))
    assert scaled <= base


# --- _apply_resource_profile ----------------------------------------------

class _Tk:
    def __init__(self, v=True):
        self._v = v

    def get(self):
        return self._v

    def set(self, v):
        self._v = v


class _ParamsTab:
    def __init__(self):
        self._params: dict = {}


class _DetTab:
    def __init__(self):
        self._params: dict = {"use_gpu_dff": True}
        self._settings_override: dict = {}


class _XcTab:
    def __init__(self):
        self.gpu_var = _Tk(True)


class _Row:
    def __init__(self, lvl, params=None, ident="rec"):
        self._oom_attempt = lvl
        self.params = params or {}
        self.identifier = ident


class _ProfStub:
    _OOM_RETRY_SCALES = BatchTab._OOM_RETRY_SCALES

    def __init__(self):
        self._app = types.SimpleNamespace(
            preprocess_tab=_ParamsTab(),
            detection_tab=_DetTab(),
            xcorr_tab=_XcTab())
        self.logs: list = []
        # Batch-start baselines captured in _on_run_all.
        self._baseline_use_gpu_dff = True
        self._baseline_settings_override: dict = {}
        self._baseline_xc_gpu = True
        self._baseline_downsample_t = 4

    def _append_log(self, m):
        self.logs.append(m)


def test_resource_profile_conservative_then_restore():
    from calliope.core import utils as u
    from calliope.core.suite2p_pipeline import get_setting
    stub = _ProfStub()
    det = stub._app.detection_tab
    xc = stub._app.xcorr_tab
    prep = stub._app.preprocess_tab

    # Level 1 -> conservative knobs applied. nbins rides _settings_override
    # (NOT _params -- suite2p never reads _params for nbins).
    BatchTab._apply_resource_profile(stub, _Row(1), "preprocess")
    assert det._params["use_gpu_dff"] is False
    assert get_setting(det._settings_override, "nbins") == 500
    assert "nbins" not in det._params
    assert xc.gpu_var.get() is False
    assert prep._params["downsample_t"] == 8        # base 4 * 2*1
    assert u.get_adaptive_sizer().conservative_scale == 0.5

    # Level 2 -> more aggressive scale + coarser downsample.
    BatchTab._apply_resource_profile(stub, _Row(2), "detection")
    assert u.get_adaptive_sizer().conservative_scale == 0.25
    assert prep._params["downsample_t"] == 16       # base 4 * 2*2

    # Back to a normal row (level 0) -> restores baselines + scale.
    BatchTab._apply_resource_profile(stub, _Row(0), "preprocess")
    assert det._params["use_gpu_dff"] is True
    assert det._settings_override == {}             # nbins override gone
    assert xc.gpu_var.get() is True
    assert prep._params["downsample_t"] == 4
    assert u.get_adaptive_sizer().conservative_scale == 1.0


# --- _fail_stage OOM routing ----------------------------------------------

class _FailStub:
    _MAX_OOM_RETRIES = BatchTab._MAX_OOM_RETRIES
    _OOM_SIGNATURES = BatchTab._OOM_SIGNATURES
    _is_oom_error = BatchTab._is_oom_error

    def __init__(self, oom_attempt=0):
        self._current_row = _Row(oom_attempt)
        self._abort_flag = threading.Event()
        self._stage_t0 = 0.0
        self._row_results = {"stages": {}, "status": "running"}
        self.retried: list = []
        self.row_done = 0
        self.logs: list = []

    def _append_log(self, m):
        self.logs.append(m)

    def _retry_row_conservative(self, stage, msg):
        self.retried.append((stage, msg))

    def _on_row_done(self):
        self.row_done += 1


def test_fail_stage_oom_triggers_retry():
    stub = _FailStub(oom_attempt=0)
    BatchTab._fail_stage(stub, "detection",
                         MemoryError("Unable to allocate 4 GiB"))
    assert len(stub.retried) == 1
    assert stub.row_done == 0          # not finalized; will re-run


def test_fail_stage_non_oom_fails_normally():
    stub = _FailStub(oom_attempt=0)
    BatchTab._fail_stage(stub, "detection", ValueError("boom"))
    assert stub.retried == []
    assert stub.row_done == 1
    assert stub._row_results["status"] == "failed"


def test_fail_stage_oom_exhausted_gives_up():
    stub = _FailStub(oom_attempt=BatchTab._MAX_OOM_RETRIES)
    BatchTab._fail_stage(stub, "detection",
                         MemoryError("Unable to allocate 4 GiB"))
    assert stub.retried == []
    assert stub.row_done == 1
    assert stub._row_results["status"] == "failed"
    assert "gave up" in stub._row_results["stages"]["detection"]["error"]


def test_fail_stage_oom_not_retried_when_aborting():
    stub = _FailStub(oom_attempt=0)
    stub._abort_flag.set()
    BatchTab._fail_stage(stub, "detection",
                         MemoryError("Unable to allocate 4 GiB"))
    assert stub.retried == []
    assert stub.row_done == 1
