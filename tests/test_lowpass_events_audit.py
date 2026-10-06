"""Regression tests for the Tab 4 -> Tab 5 hand-off (lowpass cutoff / fps
persistence, export-figures path, missing-input ordering).

Builds a tiny synthetic ``plane0`` (F.npy + r0p7_filtered_dff memmap +
persisted keep mask) so the real ``compute_lowpass_and_dt`` and
``run_event_detection`` run end to end without suite2p or a GUI.
"""
from pathlib import Path

import numpy as np
import pytest

from calliope.core import event_detection_run as edr
from calliope.core import lowpass_run, recording_meta

T, N = 600, 6
FPS = 20.0


def _make_plane0(tmp_path: Path) -> Path:
    plane0 = tmp_path / "suite2p" / "plane0"
    plane0.mkdir(parents=True)
    rng = np.random.default_rng(0)
    F = rng.normal(100.0, 1.0, size=(N, T)).astype(np.float32)
    np.save(plane0 / "F.npy", F)
    np.save(plane0 / "r0p7_cell_mask_bool.npy", np.ones(N, dtype=bool))
    dff = rng.normal(0.0, 0.05, size=(T, N)).astype(np.float32)
    # a few shared transients so detection has something to find
    for f0 in (100, 300, 500):
        dff[f0:f0 + 20, :] += np.linspace(1.0, 0.0, 20)[:, None]
    mm = np.memmap(str(plane0 / "r0p7_filtered_dff.memmap.float32"),
                   dtype="float32", mode="w+", shape=(T, N))
    mm[:] = dff
    mm.flush()
    del mm
    return plane0


def test_compute_lowpass_stamps_meta(tmp_path):
    plane0 = _make_plane0(tmp_path)
    lowpass_run.compute_lowpass_and_dt(
        plane0, fps=FPS, cutoff_hz=0.5, filter_order=3)
    sec = recording_meta.read_meta(plane0)["lowpass_filtered"]
    assert sec["cutoff_hz"] == 0.5
    assert sec["order"] == 3
    assert sec["fps"] == FPS
    # the detection-time 'lowpass' section is not touched
    assert recording_meta.read_meta(plane0)["lowpass"]["cutoff_hz"] is None


def test_auto_backtrack_uses_stamped_cutoff(tmp_path):
    """GUI path: params carry no cutoff_hz, so the Tab 4 stamp must drive
    the auto window (0.8 / 0.5 Hz = 1.6 s, not 0.8 s)."""
    recording_meta.update_meta(tmp_path, lowpass_filtered={"cutoff_hz": 0.5})
    assert edr._resolve_backtrack_s(0.0, tmp_path, {}) == pytest.approx(1.6)
    # explicit batch param still wins; explicit override wins over both
    assert edr._resolve_backtrack_s(
        0.0, tmp_path, {"cutoff_hz": 2.0}) == pytest.approx(0.4)
    assert edr._resolve_backtrack_s(0.3, tmp_path, {}) == 0.3


def test_auto_backtrack_defaults_to_1hz_without_stamp(tmp_path):
    assert edr._resolve_backtrack_s(0.0, tmp_path, {}) == pytest.approx(0.8)
    # junk values are ignored rather than dividing by zero
    assert edr._resolve_backtrack_s(
        0.0, tmp_path, {"cutoff_hz": 0}) == pytest.approx(0.8)


def test_resolve_fps_precedence(tmp_path):
    # nothing on disk -> get_fps_from_notes fallback (DEFAULT_FPS)
    from calliope.core.utils import DEFAULT_FPS
    assert edr._resolve_fps(tmp_path, {}) == DEFAULT_FPS
    recording_meta.update_meta(tmp_path, lowpass_filtered={"fps": 24.0})
    assert edr._resolve_fps(tmp_path, {}) == 24.0
    assert edr._resolve_fps(tmp_path, {"fps_override": 0}) == 24.0
    assert edr._resolve_fps(tmp_path, {"fps_override": 30.0}) == 30.0


def test_event_detection_uses_tab4_rate_and_can_skip_monotonicity(tmp_path):
    plane0 = _make_plane0(tmp_path)
    lowpass_run.compute_lowpass_and_dt(plane0, fps=FPS, cutoff_hz=1.0)
    data = edr.run_event_detection(
        plane0, {}, write_summary=False, compute_monotonicity=False)
    assert data["fps"] == FPS          # not ops/DEFAULT_FPS
    assert data["event_monotonicity"] == []


def test_export_event_figures_offload_writes_figures(tmp_path):
    plane0 = _make_plane0(tmp_path)
    lowpass_run.compute_lowpass_and_dt(plane0, fps=FPS, cutoff_hz=1.0)
    root = tmp_path / "calliope_figures"
    out = edr.export_event_figures_offload(str(plane0), {}, str(root))
    names = {Path(p).name for p in out["figures"]}
    assert names == {"heatmap.png", "raster.png", "event_detection.png"}
    assert Path(out["manifest"]).exists()
    assert Path(out["events_dir"]) == root / "event_detection"


def test_compute_lowpass_missing_source_raises_clear_error(tmp_path):
    plane0 = tmp_path / "plane0"
    plane0.mkdir()
    np.save(plane0 / "F.npy", np.zeros((N, T), dtype=np.float32))
    with pytest.raises(FileNotFoundError, match="r0p7_filtered_dff"):
        lowpass_run.compute_lowpass_and_dt(plane0, fps=FPS, cutoff_hz=1.0)


# ---- Tab 5 "Export figures" dispatch (#111) -------------------------------

class _Var:
    def __init__(self):
        self.value = None

    def set(self, v):
        self.value = v


class _Btn:
    def configure(self, **kw):
        self.kw = kw


class _StubTab:
    """Just enough of EventDetectionTab for _on_export_figures."""

    def __init__(self, plane0):
        self._plane0 = plane0
        self._params = {}
        self._render_job = None
        self._export_job = None
        self._render_queue = None
        self.POLL_MS = 80
        self.status_var = _Var()
        self.export_btn = _Btn()

    def update_idletasks(self):
        pass


def test_tab5_interactive_export_is_offloaded(tmp_path, monkeypatch):
    """A click must hand the work to run_offloaded (child process), not run
    detection on the Tk thread."""
    from calliope.core import offload
    from calliope.tabs.event_detection.tab import EventDetectionTab

    calls = []
    monkeypatch.setattr(offload, "run_offloaded",
                        lambda *a, **kw: calls.append((a, kw)) or "job")
    monkeypatch.setattr(edr, "run_event_detection",
                        lambda *a, **kw: pytest.fail("ran inline"))
    stub = _StubTab(tmp_path / "suite2p" / "plane0")
    EventDetectionTab._on_export_figures(stub, quiet=False)
    assert len(calls) == 1
    args, kw = calls[0]
    assert args[2] is edr.export_event_figures_offload
    assert kw["done_kind"] == "export_done"
    assert kw["error_kind"] == "export_error"
    assert stub._export_job == "job"


def test_tab5_quiet_export_runs_inline_without_monotonicity(tmp_path):
    from calliope.tabs.event_detection.tab import EventDetectionTab

    plane0 = _make_plane0(tmp_path)
    lowpass_run.compute_lowpass_and_dt(plane0, fps=FPS, cutoff_hz=1.0)
    stub = _StubTab(plane0)
    EventDetectionTab._on_export_figures(stub, quiet=True)
    assert stub.status_var.value.startswith("Exported event figures")


# ---- Tab 4 (#137 preview-only fallback, #132 header order) ----------------

class _LpStub:
    """Just enough of LowpassTab for _load_data / _on_plane0."""

    def __init__(self):
        from calliope.tabs.lowpass.tab import LowpassTab
        self._params = {"filter_order": 2}
        self._compute_ready = False
        self.compute_btn = _Btn()
        self.export_btn = _Btn()
        self.status_var = _Var()
        self.recording_var = _Var()
        self.cutoff_var = type("V", (), {"get": lambda self: 1.0})()
        self._load_data = lambda p: LowpassTab._load_data(self, p)
        self._draw_fft = self._draw_raw = self._draw_lowpass = lambda: None

    def update_idletasks(self):
        pass


def _make_unfiltered_only_plane0(tmp_path: Path) -> Path:
    plane0 = tmp_path / "plane0"
    plane0.mkdir()
    np.save(plane0 / "F.npy", np.zeros((N, T), dtype=np.float32))
    mm = np.memmap(str(plane0 / "r0p7_dff.memmap.float32"),
                   dtype="float32", mode="w+", shape=(T, N))
    mm[:] = 0.1
    mm.flush()
    del mm
    return plane0


def test_tab4_unfiltered_fallback_is_preview_only(tmp_path):
    from calliope.tabs.lowpass.tab import LowpassTab
    stub = _LpStub()
    LowpassTab._on_plane0(stub, _make_unfiltered_only_plane0(tmp_path))
    assert stub._compute_ready is False
    assert stub.compute_btn.kw == {"state": "disabled"}
    assert "preview only" in stub.status_var.value


def test_tab4_filtered_memmap_enables_compute(tmp_path):
    from calliope.tabs.lowpass.tab import LowpassTab
    stub = _LpStub()
    LowpassTab._on_plane0(stub, _make_plane0(tmp_path))
    assert stub._compute_ready is True
    assert stub.compute_btn.kw == {"state": "normal"}


def test_tab4_header_reflects_filter_order():
    from calliope.tabs.lowpass.tab import LowpassTab
    stub = type("S", (), {"_params": {"filter_order": 4}})()
    assert LowpassTab._header_text(stub).endswith("order 4)")
