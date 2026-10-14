"""Regression tests for the 2026-10-14 GUI walkthrough audit (docs/ux_audit).

Headless: no suite2p run, no GPU, no real recording.
"""
from __future__ import annotations

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


def test_call_in_child_survives_killed_child():
    from calliope.core.offload import call_in_child
    with pytest.raises(RuntimeError, match=r"exit code 137.*out of memory"):
        call_in_child(_child_killed, died_hint="ran out of memory")
