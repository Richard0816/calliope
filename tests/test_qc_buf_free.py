"""The QC-GIF frames go into one preallocated stack.

The streaming shift and grouped preprocess paths used to collect sampled QC
frames in a list and then ``np.stack`` it, which held two full copies at once
(about 7.9 GiB on E054 2025-10-28 00003+00004 at downsample_t=4). Frames now
go straight into one stack allocated at its final size (``_qc_put``).

* The GIF bytes match the old list + ``np.stack`` path exactly (group and
  streaming paths, including strides that cross the file boundary).
* Peak traced memory on the streaming path is about one QC stack, not two.
"""
from __future__ import annotations

import tracemalloc
from pathlib import Path

import numpy as np
import pytest
import tifffile

from calliope.core import preprocessing as pp

QC = {"max_size_px": 16, "playback_fps": 5, "clip_low": 1.0, "clip_high": 99.0}


def _raw(path: Path, T: int, seed: int, Y: int = 8, X: int = 10) -> Path:
    rng = np.random.default_rng(seed)
    tifffile.imwrite(str(path),
                     rng.integers(-50, 500, size=(T, Y, X)).astype(np.int16))
    return path


def _old_gif(frames, path: Path) -> bytes:
    """The pre-fix encode: ``np.stack`` a list of frames, then make_qc_gif."""
    pp.make_qc_gif(np.stack(list(frames), axis=0), path, downsample_t=1, **QC)
    return path.read_bytes()


def _capture_movies(monkeypatch):
    """Record every stack handed to make_qc_gif (still encoding it)."""
    seen, real = [], pp.make_qc_gif
    monkeypatch.setattr(pp, "make_qc_gif",
                        lambda movie, *a, **k: (seen.append(movie.copy()),
                                                real(movie, *a, **k)))
    return seen


@pytest.mark.parametrize("ds", [1, 3, 4, 20])
def test_group_gif_bytes_match_np_stack(tmp_path, monkeypatch, ds):
    # 7 + 6 frames: with ds=3 or 4 the global sample stride crosses the file
    # boundary, so the order across files is part of what's checked.
    srcs = [_raw(tmp_path / "rec_00001.tif", 7, 0),
            _raw(tmp_path / "rec_00002.tif", 6, 1)]
    movies = _capture_movies(monkeypatch)
    res = pp.preprocess_tiff_group(srcs, tmp_path / "data",
                                   qc_params=dict(QC, downsample_t=ds))
    whole = np.concatenate([tifffile.imread(str(p)) for p in res.shifted_paths])
    # The sampled array itself, not just the GIF (whose normalisation would
    # hide a constant offset).
    assert movies[0].dtype == np.uint16
    np.testing.assert_array_equal(movies[0], whole[::ds])
    assert Path(res.qc_gif).read_bytes() == _old_gif(whole[::ds],
                                                     tmp_path / "ref.gif")
    # Rerun reusing the shifted files (the scan-only branch): same bytes.
    res2 = pp.preprocess_tiff_group(srcs, tmp_path / "data",
                                    qc_params=dict(QC, downsample_t=ds))
    assert Path(res2.qc_gif).read_bytes() == (tmp_path / "ref.gif").read_bytes()
    np.testing.assert_array_equal(movies[1], whole[::ds])


def test_group_input_shrunk_between_passes_has_no_unwritten_slots(
        tmp_path, monkeypatch):
    # The second input is also the first one's output path, so writing the
    # first output cuts it from 6 pages to 2 after pass 1 counted 8 in all.
    rec = tmp_path / "data" / "rec"
    rec.mkdir(parents=True)
    a = _raw(rec / "raw.tif", 2, 0)
    b = _raw(rec / "00_shifted_raw.tif", 6, 1)
    movies = _capture_movies(monkeypatch)
    res = pp.preprocess_tiff_group([a, b], tmp_path / "data",
                                   recording_name="rec",
                                   qc_params=dict(QC, downsample_t=1))
    whole = np.concatenate([tifffile.imread(str(p)) for p in res.shifted_paths])
    assert len(whole) == 4
    np.testing.assert_array_equal(movies[0], whole)


def test_qc_put_refuses_a_frame_past_the_counted_stack():
    qc = pp._qc_put(None, 0, np.zeros((8, 10), np.uint16), 1)
    with pytest.raises(ValueError, match="change between passes"):
        pp._qc_put(qc, 1, np.zeros((8, 10), np.uint16), 1)


def test_streaming_shift_gif_bytes_match_np_stack(tmp_path):
    src = _raw(tmp_path / "rec.tif", 11, 2)
    mean, qc, n = pp._shift_tiff_streaming(src, tmp_path / "out.tif",
                                           qc_downsample_t=3)
    shifted = tifffile.imread(str(tmp_path / "out.tif"))
    assert n == 11 and qc.dtype == np.uint16
    np.testing.assert_array_equal(qc, shifted[::3])
    pp.make_qc_gif(qc, tmp_path / "new.gif", downsample_t=1, **QC)
    assert (tmp_path / "new.gif").read_bytes() == _old_gif(
        shifted[::3], tmp_path / "ref.gif")


def test_empty_and_mismatched_frames():
    e = pp._qc_finish(None, 0, 1, (8, 10))
    assert e.shape == (0, 8, 10) and e.dtype == np.uint16
    qc = pp._qc_put(None, 0, np.zeros((8, 10), np.uint16), 2)
    with pytest.raises(ValueError):
        pp._qc_put(qc, 1, np.zeros((1, 10), np.uint16), 2)


@pytest.mark.skipif(tracemalloc.is_tracing(),
                    reason="needs its own tracemalloc session (-X tracemalloc)")
def test_streaming_peak_memory_is_one_qc_stack(tmp_path):
    # 300 frames of 128x128 at stride 1: the QC stack (9.8 MB) dominates
    # everything else the streaming shift holds (one frame, a float64 sum).
    T, Y, X = 300, 128, 128
    src = _raw(tmp_path / "big.tif", T, 3, Y, X)
    stack = T * Y * X * 2
    tracemalloc.start()
    _, qc, _ = pp._shift_tiff_streaming(src, tmp_path / "out.tif",
                                        qc_downsample_t=1)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert qc.nbytes == stack
    # The old list + np.stack path peaks at about two stacks.
    assert peak < 1.3 * stack, f"peak {peak / stack:.2f}x the QC stack"
