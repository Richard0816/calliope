"""Tests for archive_recording_post_detection raw compression.

Guards the two defects behind "the compressed raw TIFF never appears":
1. The archive re-read the raw-paths sidecar from the wrong directory in
   batch mode (one level above where preprocess wrote it), so it found no
   sources and silently skipped compression. Passing ``raw_paths``
   explicitly bypasses the sidecar lookup.
2. When compression produced nothing, the shifted TIFFs were still
   deleted -- leaving the folder with neither a raw nor a compressed copy.
   The guard now keeps them.
"""
from __future__ import annotations

import numpy as np
import tifffile

from calliope.core.detection_run import archive_recording_post_detection


def _write_tiff(path, frames: int = 3, h: int = 8, w: int = 8):
    arr = np.arange(frames * h * w, dtype=np.uint16).reshape(frames, h, w)
    tifffile.imwrite(str(path), arr)
    return path


def test_compresses_via_raw_paths_into_recording_folder(tmp_path):
    """raw_paths threaded from the caller -> compressed TIFF lands in the
    recording folder, shifted TIFF cleaned up, count reported."""
    rec = tmp_path / "rec"; rec.mkdir()
    ext = tmp_path / "external"; ext.mkdir()
    raw = _write_tiff(ext / "rec_00001.tif")          # external raw
    shifted = _write_tiff(rec / "shifted_rec_00001.tif")

    summary = archive_recording_post_detection(
        rec, raw_paths=[raw], compress_raw=True,
        delete_shifted=True, delete_final_bin=False,
    )

    assert summary["compressed"] == 1
    assert (rec / "rec_00001.tif").exists()   # compressed copy landed in rec
    assert not shifted.exists()               # shifted cleaned up
    assert raw.exists()                       # external original untouched


def test_dest_root_decouples_read_from_write(tmp_path):
    """The compressed copy honors dest_root, independent of where the raw
    is read from (the compress-at-destination seam)."""
    rec = tmp_path / "rec"; rec.mkdir()
    final = tmp_path / "final_rec"; final.mkdir()
    ext = tmp_path / "external"; ext.mkdir()
    raw = _write_tiff(ext / "rec_00001.tif")

    summary = archive_recording_post_detection(
        rec, raw_paths=[raw], dest_root=final, compress_raw=True,
        delete_shifted=False, delete_final_bin=False,
    )

    assert summary["compressed"] == 1
    assert (final / "rec_00001.tif").exists()      # written to dest_root
    assert not (rec / "rec_00001.tif").exists()    # not to rec_root


def test_guard_keeps_shifted_when_nothing_compressed(tmp_path):
    """If compression was requested but produced 0 files, shifted TIFFs
    must NOT be deleted (no data-stranding)."""
    rec = tmp_path / "rec"; rec.mkdir()
    shifted = _write_tiff(rec / "shifted_rec_00001.tif")

    summary = archive_recording_post_detection(
        rec, raw_paths=[tmp_path / "does_not_exist.tif"],
        compress_raw=True, delete_shifted=True, delete_final_bin=False,
    )

    assert summary["compressed"] == 0
    assert shifted.exists()   # GUARD: kept despite delete_shifted=True


def test_delete_final_bin_uses_explicit_plane0(tmp_path):
    """data.bin cleanup follows the real plane0 the caller passes, not a
    reconstructed path (GUI nests detection/, batch doesn't)."""
    rec = tmp_path / "rec"; rec.mkdir()
    ext = tmp_path / "external"; ext.mkdir()
    raw = _write_tiff(ext / "rec_00001.tif")
    # Batch-style layout: plane0 directly under rec/final, no detection/.
    plane0 = rec / "final" / "suite2p" / "plane0"
    plane0.mkdir(parents=True)
    data_bin = plane0 / "data.bin"
    data_bin.write_bytes(b"x" * 4096)

    summary = archive_recording_post_detection(
        rec, raw_paths=[raw], plane0=plane0, compress_raw=True,
        delete_shifted=False, delete_final_bin=True,
    )

    assert summary["bin_deleted"] is True
    assert not data_bin.exists()


def test_compress_disabled_still_allows_shifted_cleanup(tmp_path):
    """compress_raw=False is an explicit opt-out; the guard must not block
    the user's requested shifted cleanup."""
    rec = tmp_path / "rec"; rec.mkdir()
    shifted = _write_tiff(rec / "shifted_rec_00001.tif")

    summary = archive_recording_post_detection(
        rec, compress_raw=False, delete_shifted=True, delete_final_bin=False,
    )

    assert summary["compressed"] == 0
    assert not shifted.exists()   # opt-out path still cleans up
