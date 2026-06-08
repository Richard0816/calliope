"""preprocessing.py - raw TIFF -> shifted TIFF + QC GIF + mean image.

What this file does, in one paragraph
-------------------------------------
Two-photon microscopes save raw movies as multi-page TIFF files.
Those frames may have negative pixel values (from dark-current
correction) and inconsistent dynamic ranges across files. This module
streams every frame through a "shift to non-negative + cast to
uint16" pipeline so Suite2p can read them, and at the same time
produces two QC artefacts: a downsampled animated GIF and the
per-pixel mean image.

Public surface
--------------
``run_preprocess(data_root, recording_name, ...)``
    The top-level entry point Tab 1 calls. Returns a
    ``PreprocessResult`` dataclass describing the on-disk artefacts
    written under ``<data_root>/<recording_name>/``:
        shifted_<original>.tif         (single-TIFF case)
        NNN_shifted_<original>.tif     (multi-TIFF / group case)
        qc.gif
        mean.npy

History
-------
Tab 2's right-hand panel used to overlay LoG soma-candidate
circles on the mean image, fed by a ``detect_blobs_on_mean`` pass
that wrote ``blobs.npy``. That overlay was a holdover from before
Tab 3's Sparsery + Cellpose detection came online; nothing
downstream ever consumed ``blobs.npy``, and the LoG pass was just
paying skimage import cost on every preprocess. Removed
2026-05-12 along with the ``blobs_path`` / ``n_blobs`` fields.

Where to look in the source if you've never read NumPy / Pillow
---------------------------------------------------------------
* ``np.memmap`` lets us treat a file on disk as an ndarray without
  loading the whole movie into RAM. ``arr[t, :, :]`` then reads only
  one frame's bytes.
* ``tifffile.TiffFile / TiffWriter`` are the read / write APIs for
  multi-page TIFF; the convention is one page per frame.
* ``PIL.Image`` is the GIF writer used for the QC animation.

For an R reader: think of this file as the equivalent of a
``preprocess.R`` script that loops over a folder of TIFFs, fixes the
intensities, and saves rda-equivalent (.npy) files for downstream
analysis.
"""

# Type-hint forward-reference shim (see pipeline_gui.py).
from __future__ import annotations

# ``dataclass`` decorator and ``asdict`` helper -- see the comment on
# ``EventDetectionParams`` in core/utils.py for the basics.
from dataclasses import dataclass, field, asdict
import json
import os
from pathlib import Path
import shutil
# ``Callable`` is the type-hint name for "anything you can call like a
# function" -- used below for the optional progress callback.
from typing import Callable, Optional, Sequence

import numpy as np
# ``tifffile`` is the de-facto Python library for scientific TIFFs
# (multi-page, big, OME-XML, etc). ``imread`` returns an ndarray;
# ``TiffWriter`` lets us stream one page at a time.
import tifffile
# Pillow gives us GIF encoding via ``Image.save(..., save_all=True)``.
from PIL import Image


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PreprocessResult:
    """Plain "record" describing everything Tab 1 produced.

    Tab 1 publishes one of these onto ``AppState`` so subsequent tabs
    (especially Tab 2 - QC Preview) can locate the artefacts on disk.
    Subscribers do e.g. ``result.mean_image_path`` to get the path,
    no further bookkeeping required.

    Fields
    ------
    out_dir        : recording folder root.
    shifted_tiff   : path to the uint16-shifted TIFF.
    qc_gif         : path to the QC animation.
    mean_image_path: path to ``mean.npy`` (per-pixel time average).
    n_frames       : number of frames in the movie.
    shape_yx       : (height, width) in pixels.
    raw_paths      : original raw TIFF source paths (absolute). Used
                     by the post-detection archive step to locate the
                     source of truth for re-encoding into the
                     recording folder as compressed TIFFs.
    shifted_paths  : every shifted TIFF written for this recording
                     (single-element for ``preprocess_tiff``; one per
                     group member for ``preprocess_tiff_group``).
    """
    out_dir: Path
    shifted_tiff: Path
    qc_gif: Path
    mean_image_path: Path
    n_frames: int
    shape_yx: tuple
    raw_paths: list = field(default_factory=list)
    shifted_paths: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise to a plain JSON-friendly dict (every Path -> str).

        Used when we want to write the result alongside the artefacts
        as part of a sidecar JSON for reproducibility.
        """
        # ``asdict`` is the dataclass helper that turns a dataclass
        # instance into a plain dict.
        d = asdict(self)
        # Replace any ``Path`` value with its string form.
        # ``isinstance`` is Python's ``inherits()`` analogue.
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, list):
                d[k] = [str(x) if isinstance(x, Path) else x for x in v]
        return d


# ---------------------------------------------------------------------------
# Shift  (make minimum zero, save as uint16)
# ---------------------------------------------------------------------------

@dataclass
class ShiftResult:
    """Outputs of ``shift_tiff_to_uint16``.

    Mean image and QC frame samples are accumulated *during* the same
    read pass that produces the shifted output, so callers do not have
    to re-read the shifted TIFF from disk. On spinning disks that's the
    difference between three full stack reads and one.

    Fields
    ------
    array       : full shifted stack if it fit in RAM, else ``None``.
    mean        : (Y, X) float32 per-pixel time average of the shifted
                  stack.
    qc_frames   : (T // qc_downsample_t, Y, X) uint16 every-Nth shifted
                  frame, ready to hand to ``make_qc_gif`` with
                  ``downsample_t=1``.
    n_frames    : exact T (number of pages in the shifted stack).
    """
    array: Optional[np.ndarray]
    mean: np.ndarray
    qc_frames: np.ndarray
    n_frames: int


def shift_tiff_to_uint16(
    src_tiff: Path,
    dst_tiff: Path,
    progress_cb: Optional[Callable[[str], None]] = None,
    qc_downsample_t: int = 4,
) -> ShiftResult:
    """Shift TIFF so min=0 and rewrite as uint16; also return mean + QC samples.

    Tries the fast in-RAM path first (load whole stack as int32, shift,
    cast). On ``MemoryError`` falls back to the two-pass streaming
    implementation that never holds more than a single page in RAM.

    Both paths produce the per-pixel mean image and every-Nth QC frame
    in the same read loop they were already running, so downstream
    mean.npy / qc.gif steps don't have to re-read the shifted output.
    """
    def _log(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    qc_step = max(1, int(qc_downsample_t))

    try:
        # Fast path: read the whole stack into RAM as int32 (so we
        # can hold negative values), shift so min becomes 0, then
        # cast down to uint16 (Suite2p's expected dtype).
        _log(f"[shift] reading {src_tiff.name} (in-RAM)")
        data = tifffile.imread(str(src_tiff)).astype(np.int32)
        gmin = int(data.min()); gmax = int(data.max())
        _log(f"[shift] read shape={tuple(data.shape)} dtype=int32 "
             f"min={gmin} max={gmax}")
        # ``+= int(abs(data.min()))`` is in-place: if min was -10,
        # we add 10 to every pixel.
        shift = int(abs(gmin)) if gmin < 0 else 0
        if shift:
            data += shift
        shifted_max = gmax + shift
        _log(f"[shift] global min={gmin} max={gmax}  -> shift+={shift}  "
             f"(shifted max={shifted_max})")
        if data.max() > 65535:
            # uint16's max is 65535 (a valid value). Only bail when the
            # shifted max EXCEEDS it, where the cast would alias high
            # values back down and silently corrupt the data. Matches the
            # ``shifted_max > 65535`` guard on the streaming/group paths.
            raise ValueError(
                f"Shifted max {data.max()} > 65535; uint16 overflow."
            )
        data = data.astype(np.uint16)
        # Free: mean over already-in-RAM stack; ``data[::N]`` is a
        # stride view (no copy) -- ``ascontiguousarray`` makes a small
        # contiguous block that we can hand to Pillow later.
        mean = data.mean(axis=0, dtype=np.float64).astype(np.float32)
        qc_frames = np.ascontiguousarray(data[::qc_step])
        _log(f"[shift] writing {dst_tiff.name} ({data.shape[0]} pages)")
        tifffile.imwrite(str(dst_tiff), data)
        _log(f"[shift] done -> {dst_tiff.name}")
        return ShiftResult(
            array=data, mean=mean, qc_frames=qc_frames,
            n_frames=int(data.shape[0]),
        )
    except MemoryError:
        # Slow path: very large recordings (10+ GB raw) won't fit in
        # RAM. Drop down to a two-pass streaming version that scans
        # for the global min in pass 1 and writes the shifted output
        # one TIFF page at a time in pass 2. Mean + QC samples are
        # accumulated during pass 2 so we don't re-read the output.
        _log("[shift] MemoryError on in-RAM path; "
             "retrying with streaming shift")
        mean, qc_frames, n_pages = _shift_tiff_streaming(
            src_tiff, dst_tiff,
            progress_cb=_log,
            qc_downsample_t=qc_step,
        )
        return ShiftResult(
            array=None, mean=mean, qc_frames=qc_frames,
            n_frames=int(n_pages),
        )


def _shift_tiff_streaming(
    src_tiff: Path,
    dst_tiff: Path,
    progress_cb: Optional[Callable[[str], None]] = None,
    qc_downsample_t: int = 4,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Page-by-page shift + uint16 rewrite for memory-strapped runs.

    Two passes:
        Pass 1 - scan every TIFF page for global min/max without
                 keeping pages in memory.
        Pass 2 - re-open the source and the output, write each page
                 shifted into the new file, AND on the same iteration
                 accumulate a ``float64`` per-pixel sum (for the mean
                 image) and copy every ``qc_downsample_t``-th shifted
                 frame into a small list (for the QC gif).

    Returns ``(mean, qc_frames, n_pages)`` so the caller does not need
    to read the shifted output back from disk. Never holds more than
    one full frame plus the QC sample buffer in RAM.
    """
    def _log(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    qc_step = max(1, int(qc_downsample_t))

    # Pass 1: scan for global min/max without materializing the stack.
    _log(f"[shift] streaming scan {src_tiff.name}")
    gmin = np.iinfo(np.int64).max
    gmax = np.iinfo(np.int64).min
    with tifffile.TiffFile(str(src_tiff)) as tf:
        n_pages = len(tf.pages)
        for i, page in enumerate(tf.pages):
            frame = page.asarray()
            pmin = int(frame.min()); pmax = int(frame.max())
            if pmin < gmin: gmin = pmin
            if pmax > gmax: gmax = pmax
            if (i + 1) % 2000 == 0:
                _log(f"[shift]   scan {i + 1}/{n_pages}  "
                     f"min={gmin} max={gmax}")

    shift = -gmin if gmin < 0 else 0
    shifted_max = gmax + shift
    _log(f"[shift] global min={gmin} max={gmax}  -> shift+={shift}  "
         f"(shifted max={shifted_max})")
    if shifted_max > 65535:
        raise ValueError(
            f"Shifted max {shifted_max} > 65535; uint16 overflow."
        )

    # Pass 2: write page-by-page as uint16 (BigTIFF for >4 GB output),
    # folding mean accumulator + QC frame sampling into the same loop.
    _log(f"[shift] streaming write {dst_tiff.name}")
    sum_arr: Optional[np.ndarray] = None
    qc_buf: list[np.ndarray] = []
    with tifffile.TiffFile(str(src_tiff)) as tf, \
            tifffile.TiffWriter(str(dst_tiff), bigtiff=True) as tw:
        for i, page in enumerate(tf.pages):
            frame = page.asarray()
            if shift != 0:
                frame = (frame.astype(np.int32) + shift).astype(np.uint16)
            elif frame.dtype != np.uint16:
                frame = frame.astype(np.uint16)
            tw.write(frame, contiguous=True)
            # Mean: ``float64`` sum; allocate lazily so we know (Y, X).
            if sum_arr is None:
                sum_arr = np.zeros(frame.shape, dtype=np.float64)
            sum_arr += frame
            # QC: ``.copy()`` because ``frame`` is reassigned next loop.
            if i % qc_step == 0:
                qc_buf.append(frame.copy())
            if (i + 1) % 2000 == 0:
                _log(f"[shift]   write {i + 1}/{n_pages}")

    if sum_arr is None:
        raise ValueError(f"shift: source TIFF {src_tiff} has no pages")
    mean = (sum_arr / max(1, n_pages)).astype(np.float32)
    qc_frames = (
        np.stack(qc_buf, axis=0) if qc_buf
        else np.empty((0, *sum_arr.shape), dtype=np.uint16)
    )
    return mean, qc_frames, n_pages


# ---------------------------------------------------------------------------
# Raw TIFF compression (post-detection archive)
# ---------------------------------------------------------------------------

# Suffix used for the in-progress compressed TIFF before the atomic
# rename. Easy to spot + clean up if a run was killed mid-write.
_COMPRESS_TMP_SUFFIX = ".compressing"

# Sidecar file recording the raw source paths for a recording. Written
# by ``run_preprocess`` so the post-detection archive step can locate
# the originals without needing PreprocessResult passed through to
# detection_run.
RAW_PATHS_SIDECAR = "_calliope_raw_paths.json"


def write_raw_paths_sidecar(out_dir: Path, raw_paths: Sequence[Path]) -> Path:
    """Write the recording's raw source paths to ``_calliope_raw_paths.json``.

    Detection's archive step reads this back to know which originals
    to compress into the recording folder. Stored as absolute strings
    so the file is portable across CWDs but not across machines (the
    archive step verifies existence before acting).
    """
    out_dir = Path(out_dir)
    sidecar = out_dir / RAW_PATHS_SIDECAR
    payload = {"raw_paths": [str(Path(p).resolve()) for p in raw_paths]}
    sidecar.write_text(json.dumps(payload, indent=2))
    return sidecar


def read_raw_paths_sidecar(out_dir: Path) -> list[Path]:
    """Return the recording's raw source paths, or an empty list if missing."""
    out_dir = Path(out_dir)
    sidecar = out_dir / RAW_PATHS_SIDECAR
    if not sidecar.is_file():
        return []
    try:
        data = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    raws = data.get("raw_paths", []) or []
    return [Path(p) for p in raws]


def compress_raw_tiff(
    src: Path,
    dst: Path,
    *,
    level: int = 19,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> tuple[int, int]:
    """Rewrite a raw TIFF as Zstd-compressed (lossless) at ``dst``.

    Verifies byte-equality of the decompressed payload against the
    original array before swapping into place. If ``src == dst`` the
    swap is in-place (write tmp, verify, ``os.replace``). If they
    differ, ``src`` is left untouched and the compressed copy lands
    at ``dst``.

    Returns ``(original_bytes, compressed_bytes)``.

    Raises on any verification failure -- ``dst`` is removed and
    ``src`` is never overwritten.

    Compression: ``tifffile.imwrite`` with ``compression=('zstd',
    level)`` and ``predictor=2`` (horizontal differencing). The
    predictor encodes pixel-to-pixel differences before zstd; for
    spatially correlated fluorescence pixels this roughly doubles the
    ratio over raw-bytes zstd.
    """
    def _log(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    src = Path(src).resolve()
    dst = Path(dst).resolve()
    if not src.exists():
        raise FileNotFoundError(src)

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + _COMPRESS_TMP_SUFFIX)
    # Defensive cleanup of stale tmp from a prior killed run.
    if tmp.exists():
        try:
            tmp.unlink()
        except OSError:
            pass

    original_bytes = src.stat().st_size

    try:
        _log(f"[compress] reading {src.name} (in-RAM)")
        try:
            data = tifffile.imread(str(src))
        except MemoryError:
            _log("[compress] MemoryError on in-RAM read; falling back to streaming")
            return _compress_raw_tiff_streaming(
                src, dst, level=level, progress_cb=progress_cb,
            )

        # Try predictor=2 (horizontal differencing -- ~2x better ratio on
        # spatially correlated integer fluorescence) first, then fall back
        # to predictor=1 (no prefilter). zstd is lossless either way; the
        # predictor is only a pre-filter. predictor=2 RAISES on float dtypes
        # (tifffile only defines HORIZONTAL for integers) and, very rarely,
        # can round-trip wrong -- in both cases predictor=1 still gives a
        # byte-exact lossless rewrite, so we recover instead of aborting the
        # whole archive. Only if predictor=1 ALSO fails do we raise, and then
        # with a diagnostic so a genuine structural mismatch is visible.
        last_detail = ""
        ok = False
        for predictor in (2, 1):
            _log(
                f"[compress] writing {dst.name} (zstd-{level}, "
                f"predictor={predictor}, shape={tuple(data.shape)}, "
                f"dtype={data.dtype})"
            )
            try:
                # BigTIFF unconditionally -- raws over 4 GB are common and
                # the overhead on small files is negligible.
                tifffile.imwrite(
                    str(tmp),
                    data,
                    compression="zstd",
                    compressionargs={"level": int(level)},
                    predictor=predictor,
                    bigtiff=True,
                )
            except ValueError as e:
                # e.g. "cannot use PREDICTOR.HORIZONTAL with dtype('<f4')".
                last_detail = f"predictor={predictor} imwrite: {e}"
                _log(f"[compress] {last_detail}; trying next predictor")
                if tmp.exists():
                    tmp.unlink()
                continue

            _log("[compress] verifying byte-equality after decompression")
            verify = tifffile.imread(str(tmp))
            if (verify.shape == data.shape
                    and verify.dtype == data.dtype
                    and np.array_equal(verify, data)):
                # Free the verification buffer before the rename so we don't
                # hold double the RAM during the swap.
                del verify
                ok = True
                break
            last_detail = (
                f"predictor={predictor}: decompressed "
                f"shape={verify.shape} dtype={verify.dtype} != original "
                f"shape={data.shape} dtype={data.dtype}"
            )
            _log(f"[compress] verify mismatch ({last_detail}); "
                 f"trying next predictor")
            del verify
            if tmp.exists():
                tmp.unlink()

        if not ok:
            raise RuntimeError(
                f"compress_raw_tiff: verification failed for {src} "
                f"(decompressed payload does not match original). "
                f"Last attempt: {last_detail}"
            )

        compressed_bytes = tmp.stat().st_size
        os.replace(str(tmp), str(dst))
        _log(
            f"[compress] done -> {dst.name}  "
            f"{original_bytes / 1e9:.2f} GB -> "
            f"{compressed_bytes / 1e9:.2f} GB "
            f"({compressed_bytes / max(1, original_bytes) * 100:.1f}%)"
        )
        return original_bytes, compressed_bytes

    except BaseException:
        # Make best-effort to leave the original raw untouched.
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def _compress_raw_tiff_streaming(
    src: Path,
    dst: Path,
    *,
    level: int,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> tuple[int, int]:
    """Page-by-page compressed rewrite for raws too large to fit in RAM.

    Two passes:
        Pass 1 -- write each page through tifffile.TiffWriter with the
                  zstd codec + predictor.
        Pass 2 -- re-open both src and dst, byte-verify each page.
    """
    def _log(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    src = Path(src).resolve()
    dst = Path(dst).resolve()
    tmp = dst.with_suffix(dst.suffix + _COMPRESS_TMP_SUFFIX)
    if tmp.exists():
        try:
            tmp.unlink()
        except OSError:
            pass

    original_bytes = src.stat().st_size

    try:
        # predictor=2 (best ratio for integer raws) then predictor=1
        # (lossless fallback for float dtypes / rare round-trip mismatches);
        # see compress_raw_tiff for the rationale.
        last_detail = ""
        ok = False
        for predictor in (2, 1):
            _log(
                f"[compress] streaming write {src.name} -> {dst.name} "
                f"(predictor={predictor})"
            )
            try:
                with tifffile.TiffFile(str(src)) as tf, \
                        tifffile.TiffWriter(str(tmp), bigtiff=True) as tw:
                    n_pages = len(tf.pages)
                    for i, page in enumerate(tf.pages):
                        frame = page.asarray()
                        tw.write(
                            frame,
                            contiguous=False,
                            compression="zstd",
                            compressionargs={"level": int(level)},
                            predictor=predictor,
                        )
                        if (i + 1) % 2000 == 0:
                            _log(f"[compress]   write {i + 1}/{n_pages}")
            except ValueError as e:
                last_detail = f"predictor={predictor} write: {e}"
                _log(f"[compress] {last_detail}; trying next predictor")
                if tmp.exists():
                    tmp.unlink()
                continue

            _log(f"[compress] streaming verify (page-by-page, "
                 f"predictor={predictor})")
            mismatch = None
            with tifffile.TiffFile(str(src)) as tf_src, \
                    tifffile.TiffFile(str(tmp)) as tf_dst:
                if len(tf_src.pages) != len(tf_dst.pages):
                    mismatch = (f"page count {len(tf_src.pages)} != "
                                f"{len(tf_dst.pages)}")
                else:
                    for i, (sp, dp) in enumerate(
                            zip(tf_src.pages, tf_dst.pages)):
                        if not np.array_equal(sp.asarray(), dp.asarray()):
                            mismatch = f"page {i} differs"
                            break
                        if (i + 1) % 2000 == 0:
                            _log(f"[compress]   verify {i + 1}/"
                                 f"{len(tf_src.pages)}")
            if mismatch is None:
                ok = True
                break
            last_detail = f"predictor={predictor}: {mismatch}"
            _log(f"[compress] verify mismatch ({last_detail}); "
                 f"trying next predictor")
            if tmp.exists():
                tmp.unlink()

        if not ok:
            raise RuntimeError(
                f"compress_raw_tiff: streaming verification failed for "
                f"{src}. Last attempt: {last_detail}"
            )

        compressed_bytes = tmp.stat().st_size
        os.replace(str(tmp), str(dst))
        _log(
            f"[compress] streaming done -> {dst.name}  "
            f"{original_bytes / 1e9:.2f} GB -> "
            f"{compressed_bytes / 1e9:.2f} GB "
            f"({compressed_bytes / max(1, original_bytes) * 100:.1f}%)"
        )
        return original_bytes, compressed_bytes

    except BaseException:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


# ---------------------------------------------------------------------------
# QC gif
# ---------------------------------------------------------------------------

def make_qc_gif(
    movie: np.ndarray,
    gif_path: Path,
    downsample_t: int = 4,
    max_size_px: int = 512,
    playback_fps: int = 15,
    clip_low: float = 1.0,
    clip_high: float = 99.5,
) -> None:
    """Write a compact grayscale animated GIF for visual QC.

    This is what Tab 2 plays back so the user can eyeball the
    recording for motion / drift / focus issues. We need to:
        1. Drop most frames -- a 10-minute @ 15 fps movie is 9000
           frames, way too many for a smooth GIF. Keep every
           ``downsample_t``th frame instead.
        2. Auto-window: pick percentile-based clip points so the
           GIF isn't wasted on bright outliers.
        3. Down-resolution if any side is over ``max_size_px``.
        4. Pillow ``Image.save(..., save_all=True, ...)`` writes a
           multi-frame GIF in one call.
    """
    if movie.ndim != 3:
        raise ValueError(f"Expected (T, Y, X) stack, got shape {movie.shape}")

    # ``movie[::N]`` is "every Nth element along the first axis" --
    # cheapest possible time-downsample.
    frames = movie[::max(1, int(downsample_t))]
    # Percentile clip points. Computing this on the full stack would
    # flatten/partition a multi-GiB array; a strided subsample of frames
    # gives an effectively identical window at a fraction of the memory.
    stride = max(1, frames.shape[0] // 200)
    sample = frames[::stride]
    lo = float(np.percentile(sample, clip_low))
    hi = float(np.percentile(sample, clip_high))
    span = max(hi - lo, 1e-6)

    # Convert each frame to a Pillow ``Image`` and resize if too big.
    # Normalise one frame at a time rather than materialising a single
    # float32 copy of the whole stack -- a 6000-frame 512x512 movie
    # would need ~6 GiB as float32 and blow past the offload worker's
    # RAM. Per-frame keeps the working set at ~1 MB.
    pil_frames = []
    for raw in frames:
        # Vectorised normalisation: rescale ``[lo, hi]`` -> ``[0, 1]``
        # then clip outliers, per frame.
        norm = np.clip((raw.astype(np.float32) - lo) / span, 0.0, 1.0)
        f = (norm * 255.0).astype(np.uint8)
        # mode="L" is single-channel grayscale.
        im = Image.fromarray(f, mode="L")
        if max(im.size) > max_size_px:
            scale = max_size_px / max(im.size)
            new_size = (max(1, int(im.size[0] * scale)),
                        max(1, int(im.size[1] * scale)))
            # Bilinear interpolation: smoother than nearest, faster
            # than bicubic; fine for QC-quality output.
            im = im.resize(new_size, Image.BILINEAR)
        pil_frames.append(im)

    # Frame duration in ms (e.g. 15 fps -> 67 ms).
    duration_ms = max(1, int(round(1000.0 / playback_fps)))
    # ``save_all=True`` + ``append_images=...`` is Pillow's multi-
    # frame-GIF API: pass the first frame as ``self`` and the rest
    # as ``append_images``. ``loop=0`` means "loop forever";
    # ``optimize=True`` shrinks the file with a frame-diff encoding.
    pil_frames[0].save(
        str(gif_path),
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess_tiff(
    src_tiff: str | Path,
    data_root: str | Path,
    recording_name: Optional[str] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
    qc_params: Optional[dict] = None,
) -> PreprocessResult:
    """
    End-to-end preprocessing for a single raw TIFF.

    Parameters
    ----------
    src_tiff
        Path to the raw (un-shifted) TIFF stack.
    data_root
        Root under which per-recording folders live (e.g. ``data/``).
    recording_name
        Folder name for the recording. Defaults to the TIFF stem.
    progress_cb
        Optional callback receiving status strings.
    qc_params
        Optional overrides for ``make_qc_gif`` (keys: ``downsample_t``,
        ``max_size_px``, ``playback_fps``, ``clip_low``, ``clip_high``).
    """
    # Resolve to an absolute path; raise if the file doesn't exist
    # so the user gets a clean error instead of a cryptic "file not
    # found" 30 seconds in.
    src = Path(src_tiff).resolve()
    if not src.exists():
        raise FileNotFoundError(src)

    # Default the recording folder name to the TIFF stem
    # (``foo.tif`` -> ``foo``).
    name = recording_name or src.stem
    out_dir = Path(data_root).resolve() / name
    # ``parents=True`` creates any missing parent dirs;
    # ``exist_ok=True`` doesn't raise if the folder already exists.
    out_dir.mkdir(parents=True, exist_ok=True)

    shifted_path = out_dir / f"shifted_{src.name}"
    gif_path = out_dir / "qc.gif"
    mean_path = out_dir / "mean.npy"

    # Local closure for the progress callback. Captures
    # ``progress_cb`` from the enclosing scope.
    def log(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    # ---- Step 1: shift to non-negative + cast to uint16 ----
    # Mean image + QC frame samples are accumulated inside the shift's
    # read loop so we don't pay another full read of the shifted output.
    qc_kwargs = dict(qc_params or {})
    qc_downsample_t = max(1, int(qc_kwargs.pop("downsample_t", 4)))
    log(f"[1/3] Shifting {src.name} -> {shifted_path.name}")
    shifted = shift_tiff_to_uint16(
        src, shifted_path,
        progress_cb=log,
        qc_downsample_t=qc_downsample_t,
    )
    Y, X = shifted.mean.shape
    T = int(shifted.n_frames)

    # ---- Step 2: per-pixel mean image (no re-read; already accumulated) ----
    log(f"[2/3] Saving mean image ({T} frames, {Y}x{X})")
    np.save(str(mean_path), shifted.mean)

    # ---- Step 3: animated QC GIF (no re-read; frames already sampled) ----
    log("[3/3] Writing QC gif")
    make_qc_gif(shifted.qc_frames, gif_path, downsample_t=1, **qc_kwargs)

    log(f"Done. Output -> {out_dir}")

    # Wrap the artefacts in the dataclass so callers don't have to
    # remember the file naming conventions.
    return PreprocessResult(
        out_dir=out_dir,
        shifted_tiff=shifted_path,
        qc_gif=gif_path,
        mean_image_path=mean_path,
        n_frames=int(T),
        shape_yx=(int(Y), int(X)),
        raw_paths=[src],
        shifted_paths=[shifted_path],
    )


def preprocess_tiff_group(
    src_tiffs: list[str | Path],
    data_root: str | Path,
    recording_name: Optional[str] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
    qc_params: Optional[dict] = None,
) -> PreprocessResult:
    """End-to-end preprocessing for a group of raw TIFFs treated as one
    continuous recording.

    A single shift constant (derived from the global min across ALL inputs) is
    applied to every file, so the concatenated movie has consistent intensity.
    Each shifted output is written into
    ``<data_root>/<recording_name>/NNN_shifted_<orig>.tif`` with a zero-padded
    order prefix (suite2p uses natsort on data_path, so the prefix preserves
    input order). Mean image and QC gif are computed across the full
    concatenated stack.

    If ``recording_name`` is omitted, defaults to the ``+``-joined stems of
    ``src_tiffs`` (in the order given).
    """
    if not src_tiffs:
        raise ValueError("preprocess_tiff_group: empty src_tiffs")

    srcs = [Path(p).resolve() for p in src_tiffs]
    for s in srcs:
        if not s.exists():
            raise FileNotFoundError(s)

    name = recording_name or "+".join(s.stem for s in srcs)
    out_dir = Path(data_root).resolve() / name
    out_dir.mkdir(parents=True, exist_ok=True)

    pad = max(2, len(str(len(srcs) - 1)))
    shifted_paths = [
        out_dir / f"{i:0{pad}d}_shifted_{src.name}"
        for i, src in enumerate(srcs)
    ]
    gif_path = out_dir / "qc.gif"
    mean_path = out_dir / "mean.npy"

    def log(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    # ---- Pass 1: scan all files for one shared global min/max -------------
    log(f"[1/2] Scanning {len(srcs)} TIFFs for shared global min/max")
    gmin = np.iinfo(np.int64).max
    gmax = np.iinfo(np.int64).min
    page_counts: list[int] = []
    for src in srcs:
        with tifffile.TiffFile(str(src)) as tf:
            n_pages = len(tf.pages)
            for i, page in enumerate(tf.pages):
                frame = page.asarray()
                pmin = int(frame.min()); pmax = int(frame.max())
                if pmin < gmin: gmin = pmin
                if pmax > gmax: gmax = pmax
                if (i + 1) % 2000 == 0:
                    log(f"[1/2]   {src.name}  scan {i + 1}/{n_pages}  "
                        f"min={gmin} max={gmax}")
            page_counts.append(n_pages)
            log(f"[1/2]   {src.name}: {n_pages} pages")
    shift = -gmin if gmin < 0 else 0
    shifted_max = gmax + shift
    log(f"[1/2] global min={gmin} max={gmax}  -> shift+={shift}  "
        f"(shifted max={shifted_max})")
    if shifted_max > 65535:
        raise ValueError(
            f"Shifted max {shifted_max} > 65535; uint16 overflow."
        )

    # ---- Pass 2: write each shifted file + accumulate mean / QC samples ---
    # Folding mean accumulation and QC frame sampling into the write
    # loop avoids two extra full reads of the shifted output (passes 3
    # and 4 in the old layout) -- on spinning disks that's the entire
    # cost of the QC/mean step.
    qc_kwargs = dict(qc_params or {})
    downsample_t = max(1, int(qc_kwargs.pop("downsample_t", 4)))
    log(f"[2/2] Writing {len(srcs)} shifted TIFFs into {out_dir} "
        f"(+ mean + QC accumulation)")
    sum_arr: Optional[np.ndarray] = None
    qc_buf: list[np.ndarray] = []
    total_frames = 0
    # ``qc_index`` is the global frame counter across files so the
    # ``i % downsample_t == 0`` sample pattern continues seamlessly
    # over concatenated TIFFs (matches the old pass-4 behaviour where
    # each file was sampled independently with stride ``downsample_t``;
    # the global counter makes the result file-boundary-agnostic).
    qc_index = 0
    for src, dst in zip(srcs, shifted_paths):
        if dst.exists():
            # Already shifted on a prior run -- read it through to
            # update the accumulators so the mean/QC remain correct
            # for the full concatenation.
            log(f"[2/2]   exists, scanning for mean+QC: {dst.name}")
            with tifffile.TiffFile(str(dst)) as tf:
                n_pages = len(tf.pages)
                for i, page in enumerate(tf.pages):
                    frame = page.asarray()
                    if frame.dtype != np.uint16:
                        frame = frame.astype(np.uint16)
                    if sum_arr is None:
                        sum_arr = np.zeros(frame.shape, dtype=np.float64)
                    sum_arr += frame
                    if qc_index % downsample_t == 0:
                        qc_buf.append(frame.copy())
                    qc_index += 1
                    if (i + 1) % 2000 == 0:
                        log(f"[2/2]     scan {i + 1}/{n_pages}")
                total_frames += n_pages
            continue
        log(f"[2/2]   {src.name} -> {dst.name}")
        with tifffile.TiffFile(str(src)) as tf, \
                tifffile.TiffWriter(str(dst), bigtiff=True) as tw:
            n_pages = len(tf.pages)
            for i, page in enumerate(tf.pages):
                frame = page.asarray()
                if shift != 0:
                    frame = (frame.astype(np.int32) + shift).astype(np.uint16)
                elif frame.dtype != np.uint16:
                    frame = frame.astype(np.uint16)
                tw.write(frame, contiguous=True)
                if sum_arr is None:
                    sum_arr = np.zeros(frame.shape, dtype=np.float64)
                sum_arr += frame
                if qc_index % downsample_t == 0:
                    qc_buf.append(frame.copy())
                qc_index += 1
                if (i + 1) % 2000 == 0:
                    log(f"[2/2]     write {i + 1}/{n_pages}")
            total_frames += n_pages

    if sum_arr is None:
        raise ValueError(
            f"preprocess_tiff_group: produced no frames for {out_dir}"
        )
    Y, X = int(sum_arr.shape[0]), int(sum_arr.shape[1])
    mean = (sum_arr / max(1, total_frames)).astype(np.float32)
    np.save(str(mean_path), mean)
    log(f"[2/2]   mean over {total_frames} frames ({Y}x{X})")

    log("[2/2] Writing QC gif")
    movie = (
        np.stack(qc_buf, axis=0) if qc_buf
        else np.empty((0, Y, X), dtype=np.uint16)
    )
    make_qc_gif(movie, gif_path, downsample_t=1, **qc_kwargs)

    log(f"Done. Output -> {out_dir}")

    return PreprocessResult(
        out_dir=out_dir,
        shifted_tiff=shifted_paths[0],
        qc_gif=gif_path,
        mean_image_path=mean_path,
        n_frames=int(total_frames),
        shape_yx=(int(Y), int(X)),
        raw_paths=list(srcs),
        shifted_paths=list(shifted_paths),
    )


def preprocess_offload(src_strs: list[str], data_root: str, *,
                       recording_name: Optional[str] = None,
                       group: bool = False,
                       qc_params: Optional[dict] = None,
                       progress_q=None) -> "PreprocessResult":
    """Picklable :mod:`calliope.core.offload` target for Tab 1.

    Dispatches the single-file vs group preprocessing path in a child
    process so the TIFF shift / mean / QC-GIF work can't freeze the GUI.
    Takes only picklable inputs (path *strings* + a flag + a dict) and
    returns the picklable :class:`PreprocessResult` the tab's ``_on_done``
    already consumes.
    """
    from .offload import make_progress_cb
    cb = make_progress_cb(progress_q)
    srcs = [Path(s) for s in src_strs]
    if group:
        return preprocess_tiff_group(
            src_tiffs=srcs, data_root=data_root,
            recording_name=recording_name, progress_cb=cb,
            qc_params=qc_params)
    return preprocess_tiff(
        src_tiff=srcs[0], data_root=data_root,
        recording_name=recording_name, progress_cb=cb,
        qc_params=qc_params)


# ---------------------------------------------------------------------------
# Discovery helpers used by the GUI
# ---------------------------------------------------------------------------

def list_tiffs(folder: str | Path, max_depth: int = 0) -> list[Path]:
    """Return TIFFs inside ``folder`` sorted by path.

    ``max_depth=0`` (default) keeps the original behaviour: only files
    directly inside ``folder``. ``max_depth=N`` searches up to N levels of
    subdirectories using breadth-first traversal."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    exts = {".tif", ".tiff"}
    max_depth = max(0, int(max_depth))

    from collections import deque
    queue: deque[tuple[Path, int]] = deque([(folder, 0)])
    found: list[Path] = []
    while queue:
        cur_dir, depth = queue.popleft()
        try:
            children = list(cur_dir.iterdir())
        except (OSError, PermissionError):
            continue
        for p in children:
            if p.is_file() and p.suffix.lower() in exts:
                found.append(p)
            elif p.is_dir() and depth < max_depth:
                queue.append((p, depth + 1))
    return sorted(found, key=lambda p: str(p).lower())


def load_existing_preprocess(
    out_dir: str | Path,
    *,
    regenerate_shifted: bool = True,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Optional[PreprocessResult]:
    """If a recording folder already contains preprocessing outputs, load them.

    Recognises both single-TIFF (``shifted_<x>.tif``) and grouped
    (``NNN_shifted_<x>.tif``) layouts.

    Post-detection archive mode: if the shifted TIFFs have been
    deleted by the archive step but compressed raws are present at
    the top level (referenced by ``_calliope_raw_paths.json``), the
    shifted is regenerated on demand by running
    ``shift_tiff_to_uint16`` again so QC / re-detect can continue.

    ``regenerate_shifted=False`` suppresses that regeneration: viewing a
    finished run (QC, lowpass, events, ...) never needs the shifted TIFF
    -- only re-running detection does -- and regenerating it from the
    compressed raw is a multi-gigabyte, multi-minute operation that would
    freeze the UI. With the flag off, an archived run returns a result
    whose ``shifted_tiff`` points at where the file *would* be
    regenerated (so Tab 3 can still name the recording), with
    ``shifted_paths`` empty; the actual regeneration is deferred until
    something genuinely re-detects.
    """
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return None
    shifted_candidates = sorted(
        p for p in out_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in (".tif", ".tiff")
        and "shifted_" in p.name
    )
    gif = out_dir / "qc.gif"
    mean_path = out_dir / "mean.npy"

    if not (gif.exists() and mean_path.exists()):
        return None

    # Archived state: shifted gone but compressed raw sidecar exists.
    # Regenerate the shifted TIFF(s) from the (compressed) raw before
    # returning -- callers expect ``shifted_tiff`` to be a real file.
    raw_sources = read_raw_paths_sidecar(out_dir)
    if not shifted_candidates and not regenerate_shifted:
        # Viewing-only reload: QC / lowpass / events only need qc.gif +
        # mean.npy. The shifted TIFF is required solely for re-detection,
        # so when it has been archived away we still return a usable
        # result instead of None -- otherwise "Load from folder" reports
        # "couldn't parse preprocess outputs" for any finished run whose
        # shifted TIFFs were pruned. shifted_tiff points at where the file
        # *would* be regenerated so Tab 3 can still name the recording;
        # shifted_paths stays empty. The raw sidecar is optional here: it
        # is absent for older runs and manually-placed shifted TIFFs, and
        # viewing doesn't need the raw at all.
        if raw_sources:
            expected = out_dir / f"shifted_{raw_sources[0].name}"
        else:
            expected = out_dir / f"shifted_{out_dir.name}.tif"
        mean = np.load(str(mean_path))
        return PreprocessResult(
            out_dir=out_dir,
            shifted_tiff=expected,
            qc_gif=gif,
            mean_image_path=mean_path,
            n_frames=-1,
            shape_yx=tuple(mean.shape),
            raw_paths=raw_sources,
            shifted_paths=[],
        )
    if not shifted_candidates and raw_sources:
        def _log(msg: str) -> None:
            if progress_cb is not None:
                progress_cb(msg)
        _log(
            f"[preprocess] shifted TIFFs missing; regenerating from "
            f"{len(raw_sources)} archived raw(s) at {out_dir}"
        )
        regenerated: list[Path] = []
        if len(raw_sources) == 1:
            src = raw_sources[0]
            if not src.exists():
                # Maybe the original was deleted; check if the raw was
                # archived into the recording folder under its own name.
                in_folder = out_dir / src.name
                if in_folder.is_file():
                    src = in_folder
                else:
                    _log(
                        f"[preprocess] cannot regenerate: raw missing at "
                        f"{src} (and not in recording folder)"
                    )
                    return None
            dst = out_dir / f"shifted_{src.name}"
            shift_tiff_to_uint16(src, dst, progress_cb=_log)
            regenerated = [dst]
        else:
            # Group case: re-run the grouped shift to keep one shared
            # min across files (matches original Tab 1 behaviour).
            srcs_resolved: list[Path] = []
            for s in raw_sources:
                if s.exists():
                    srcs_resolved.append(s)
                elif (out_dir / s.name).is_file():
                    srcs_resolved.append(out_dir / s.name)
                else:
                    _log(
                        f"[preprocess] cannot regenerate group: raw missing "
                        f"at {s} (and not in recording folder)"
                    )
                    return None
            # Re-run preprocess_tiff_group writing into the same folder.
            # Cheaper would be a dedicated reshift, but this is correct
            # and infrequent enough that the extra mean/gif rewrite cost
            # is acceptable.
            data_root = out_dir.parent
            recording_name = out_dir.name
            result = preprocess_tiff_group(
                srcs_resolved,
                data_root=str(data_root),
                recording_name=recording_name,
                progress_cb=_log,
            )
            shifted_folder = Path(result.shifted_tiff).parent
            setattr(result, "shifted_folder", shifted_folder)
            return result
        shifted_candidates = regenerated

    if not shifted_candidates:
        return None

    mean = np.load(str(mean_path))

    return PreprocessResult(
        out_dir=out_dir,
        shifted_tiff=shifted_candidates[0],
        qc_gif=gif,
        mean_image_path=mean_path,
        n_frames=-1,
        shape_yx=tuple(mean.shape),
        raw_paths=raw_sources,
        shifted_paths=list(shifted_candidates),
    )


# ---------------------------------------------------------------------------
# Headless orchestrator -- the entry point Tab 0's batch runner calls.
# ---------------------------------------------------------------------------
#
# ``preprocess_tiff`` and ``preprocess_tiff_group`` above are already
# pure-function entry points (no Tk dependency); this wrapper picks
# single vs grouped automatically and exposes a uniform
# ``run_preprocess(src_tiffs, data_root, params, ...)`` shape that
# matches the rest of the ``core.*`` headless entry points.


_QC_KEYS = ("downsample_t", "max_size_px", "playback_fps",
            "clip_low", "clip_high")


def _split_preprocess_params(params: dict) -> dict:
    """Pull the QC-gif overrides out of the unified params dict,
    mirroring ``PreprocessTab._start_run``.
    """
    return {k: params[k] for k in _QC_KEYS if k in params}


def run_preprocess(
    src_tiffs: Sequence[str | Path],
    data_root: str | Path,
    params: dict,
    *,
    recording_name: Optional[str] = None,
    figures_dir: Optional[Path] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
):
    """Run Tab 1 preprocessing on one recording.

    Parameters
    ----------
    src_tiffs : sequence of paths
        One or more raw TIFFs. A single path runs ``preprocess_tiff``;
        multiple paths run ``preprocess_tiff_group`` (concatenates
        them in trailing-index order).
    data_root : str | Path
        Where to write the recording folder
        (``<data_root>/<recording_name>/``).
    params : dict
        Same shape as :class:`PreprocessTab.PARAM_SPEC`: any of the
        QC-gif keys (``downsample_t``, ``max_size_px``,
        ``playback_fps``, ``clip_low``, ``clip_high``).
    recording_name : str | None
        Folder name. Defaults to single-TIFF stem or '+'-joined
        stems for groups (matches Tab 1's
        ``_resolved_identifier`` rule).
    figures_dir : Path | None
        If set, copies ``qc.gif`` (the only image artefact) into
        this directory so the batch report's
        ``calliope_figures/preprocess/`` subfolder picks it up.

    Returns
    -------
    PreprocessResult
        Same dataclass Tab 1 publishes, with a ``shifted_folder``
        attribute attached at runtime so callers can pass it to
        ``detection_run.run_detection`` without a second
        ``Path.parent`` walk.
    """
    srcs = [Path(p) for p in src_tiffs]
    if not srcs:
        raise ValueError("run_preprocess: no source TIFFs supplied")

    qc = _split_preprocess_params(params)

    if len(srcs) == 1:
        result = preprocess_tiff(
            src_tiff=srcs[0],
            data_root=str(data_root),
            recording_name=recording_name,
            progress_cb=progress_cb,
            qc_params=qc,
        )
    else:
        result = preprocess_tiff_group(
            src_tiffs=srcs,
            data_root=str(data_root),
            recording_name=recording_name,
            progress_cb=progress_cb,
            qc_params=qc,
        )

    # ``shifted_tiff`` lives inside the recording folder; the
    # detection backend takes a folder, so attach a convenience
    # attribute pointing at it.
    shifted_folder = Path(result.shifted_tiff).parent
    setattr(result, "shifted_folder", shifted_folder)

    # Persist raw source paths next to the recording outputs so the
    # post-detection archive step can locate the originals without
    # needing PreprocessResult threaded through detection_run.
    try:
        write_raw_paths_sidecar(result.out_dir, result.raw_paths)
    except OSError as e:
        if progress_cb is not None:
            progress_cb(f"[preprocess] raw-paths sidecar write failed: {e}")

    if figures_dir is not None:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        try:
            qc_gif = Path(result.qc_gif)
            if qc_gif.exists():
                shutil.copy2(qc_gif, figures_dir / qc_gif.name)
        except Exception as e:
            if progress_cb is not None:
                progress_cb(f"[preprocess] qc gif copy failed: {e}")

    return result
