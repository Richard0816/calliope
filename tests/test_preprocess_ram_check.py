"""Oversized TIFFs must go straight to the streaming paths instead of an
in-RAM read that swaps the machine to death (Windows never raises MemoryError
promptly for a 23 GB stack on 34 GB RAM)."""
from types import SimpleNamespace

import numpy as np
import tifffile

from calliope.core import preprocessing as pp


def test_low_ram_takes_streaming_paths(tmp_path, monkeypatch):
    src = tmp_path / "raw.tif"
    data = np.arange(4 * 8 * 8, dtype=np.int16).reshape(4, 8, 8) - 50
    with tifffile.TiffWriter(src) as tw:  # one page per frame, like a ScanImage raw
        for frame in data:
            tw.write(frame, contiguous=False)
    monkeypatch.setattr(pp.psutil, "virtual_memory", lambda: SimpleNamespace(available=10))
    monkeypatch.setattr(pp.tifffile, "imread", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("in-RAM read attempted")))
    logs = []

    res = pp.shift_tiff_to_uint16(src, tmp_path / "shifted.tif", progress_cb=logs.append)
    assert any("streaming" in m for m in logs)
    assert res.n_frames == 4

    pp.compress_raw_tiff(src, tmp_path / "raw_zstd.tif", progress_cb=logs.append)
    assert any("falling back to streaming" in m for m in logs)

    monkeypatch.undo()
    np.testing.assert_array_equal(tifffile.imread(tmp_path / "shifted.tif", key=range(4)), (data + 50).astype(np.uint16))
    np.testing.assert_array_equal(tifffile.imread(tmp_path / "raw_zstd.tif", key=range(4)), data)
