"""preprocessing.py - raw TIFF -> shifted TIFF + QC GIF + mean image.

What this file does, in one paragraph
-------------------------------------
Two-photon microscopes save raw movies as multi-page TIFF files.
Those frames may have negative pixel values (from dark-current
correction) and inconsistent dynamic ranges across files. This module
streams every frame through a "shift to non-negative + cast to
uint16" pipeline so Suite2p can read them, and at the same time
produces three QC artefacts: a downsampled animated GIF, the per-
pixel mean image, and a list of candidate cell-body blobs detected
on the mean image with a Laplacian-of-Gaussian filter.

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
        blobs.npy                      (soma candidates on mean image)

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
from dataclasses import dataclass, asdict
from pathlib import Path
# ``Callable`` is the type-hint name for "anything you can call like a
# function" -- used below for the optional progress callback.
from typing import Callable, Optional

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
    blobs_path     : path to ``blobs.npy`` (LoG soma candidates).
    n_frames       : number of frames in the movie.
    shape_yx       : (height, width) in pixels.
    n_blobs        : number of soma candidates detected.
    """
    out_dir: Path
    shifted_tiff: Path
    qc_gif: Path
    mean_image_path: Path
    blobs_path: Path
    n_frames: int
    shape_yx: tuple
    n_blobs: int

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
        return d


# ---------------------------------------------------------------------------
# Shift  (make minimum zero, save as uint16)
# ---------------------------------------------------------------------------

def shift_tiff_to_uint16(
    src_tiff: Path,
    dst_tiff: Path,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> np.ndarray:
    """Shift TIFF so min=0 and rewrite as uint16.

    Tries the fast in-RAM path first (load whole stack as int32, shift,
    cast). On ``MemoryError`` falls back to the two-pass streaming
    implementation modelled on ``run_full_pipeline.shift_tiff`` that
    never holds more than a single page in RAM, then returns a
    ``tifffile.memmap`` view of the output so downstream steps (mean
    image, QC gif) keep working unchanged.
    """
    def _log(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

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
        if data.max() >= 65535:
            # uint16's max is 65535. If the dynamic range is wider
            # than that we'd alias high values back to zero -- bail
            # rather than silently corrupt the data.
            raise ValueError(
                f"Shifted range exceeds uint16 ({data.max()}); "
                f"raw dynamic range too wide."
            ) #todo create alternative solution for this case
        data = data.astype(np.uint16)
        _log(f"[shift] writing {dst_tiff.name} ({data.shape[0]} pages)")
        tifffile.imwrite(str(dst_tiff), data)
        _log(f"[shift] done -> {dst_tiff.name}")
        return data
    except MemoryError:
        # Slow path: very large recordings (10+ GB raw) won't fit in
        # RAM. Drop down to a two-pass streaming version that scans
        # for the global min in pass 1 and writes the shifted output
        # one TIFF page at a time in pass 2.
        _log("[shift] MemoryError on in-RAM path; "
             "retrying with streaming shift")
        _shift_tiff_streaming(src_tiff, dst_tiff, progress_cb=_log)
        # Return a memmap of the freshly written output -- callers
        # treat the return value as if it was the in-RAM array.
        return tifffile.memmap(str(dst_tiff), mode="r")


def _shift_tiff_streaming(
    src_tiff: Path,
    dst_tiff: Path,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> None:
    """Page-by-page shift + uint16 rewrite for memory-strapped runs.

    Two passes:
        Pass 1 - scan every TIFF page for global min/max without
                 keeping pages in memory.
        Pass 2 - re-open the source and the output, write each page
                 shifted into the new file.

    Never holds more than one frame in RAM. Uses BigTIFF on the
    output (>4 GB single-file support) since multi-hour recordings
    blow past the standard 4 GB TIFF limit.
    """
    def _log(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

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

    # Pass 2: write page-by-page as uint16 (BigTIFF for >4 GB output).
    _log(f"[shift] streaming write {dst_tiff.name}")
    with tifffile.TiffFile(str(src_tiff)) as tf, \
            tifffile.TiffWriter(str(dst_tiff), bigtiff=True) as tw:
        for i, page in enumerate(tf.pages):
            frame = page.asarray()
            if shift != 0:
                frame = (frame.astype(np.int32) + shift).astype(np.uint16)
            elif frame.dtype != np.uint16:
                frame = frame.astype(np.uint16)
            tw.write(frame, contiguous=True)
            if (i + 1) % 2000 == 0:
                _log(f"[shift]   write {i + 1}/{n_pages}")


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
    # ``np.percentile`` ignores none of the values -- it works on
    # the full array. clip_low=1, clip_high=99.5 keeps the GIF
    # contrast reasonable.
    lo = float(np.percentile(frames, clip_low))
    hi = float(np.percentile(frames, clip_high))
    span = max(hi - lo, 1e-6)
    # Vectorised normalisation: rescale ``[lo, hi]`` -> ``[0, 1]``
    # then clip outliers.
    norm = np.clip((frames.astype(np.float32) - lo) / span, 0.0, 1.0)
    u8 = (norm * 255.0).astype(np.uint8)

    # Convert each frame to a Pillow ``Image`` and resize if too big.
    pil_frames = []
    for f in u8:
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
# Blob detection on the mean image
# ---------------------------------------------------------------------------

def detect_blobs_on_mean(
    mean_img: np.ndarray,
    soma_diameter_px: float = 12.0,
    scale_tol: float = 0.5,
    min_contrast: float = 0.10,
    min_area_px: int = 25,
    max_area_px: int = 400,
) -> list:
    """Lightweight Laplacian-of-Gaussian (LoG) blob detection on the
    mean image.

    What this gets us
    -----------------
    Tab 1 isn't running Suite2p yet -- we just want a rough sanity
    check that "yes, there are cell-shaped objects in this field of
    view". Tab 2 overlays the resulting circles on the mean image so
    the user can confirm before committing to a full Suite2p run.

    Algorithm
    ---------
    1. Subtract a local-median background (high-pass).
    2. Normalise to ~[0, 1] by dividing by the 99.5th percentile.
    3. Run scikit-image's ``blob_log`` searching for blobs of
       sigma in ``[r*(1-tol), r*(1+tol)] / sqrt(2)``, where
       r = soma_diameter / 2.
    4. Reject blobs whose pi*radius^2 area falls outside
       ``[min_area_px, max_area_px]``.

    Returns a list of ``(y, x, radius)`` tuples in pixels.
    """
    # Lazy imports keep ``preprocessing.py`` cheap to import for
    # callers (e.g. tests) that don't need blob detection.
    from skimage.feature import blob_log
    from scipy.ndimage import median_filter

    r = float(soma_diameter_px) / 2.0
    # LoG sigmas relate to blob radius by ``r = sigma * sqrt(2)`` for
    # a 2-D Gaussian. So we invert that to convert the radius range
    # into a sigma range.
    min_sigma = (r * (1.0 - scale_tol)) / np.sqrt(2.0)
    max_sigma = (r * (1.0 + scale_tol)) / np.sqrt(2.0)

    img = mean_img.astype(np.float32)
    # Median filter ~ "what's the typical brightness of a soma-sized
    # patch around me". Subtracting it leaves only the pixels that
    # rise above their local neighbourhood -- candidate cells.
    bg_radius = max(3, int(round(soma_diameter_px)))
    bg_local = median_filter(img, size=bg_radius)
    # ``np.clip(x, 0, None)`` clamps below at 0, leaves above
    # unbounded -- the high-pass result.
    hp = np.clip(img - bg_local, 0.0, None)

    # Normalise the high-pass to [0, 1] by dividing by a robust
    # upper percentile so a single bright pixel doesn't dominate.
    hi = float(np.quantile(hp, 0.995))
    if hi <= 0:
        return []
    norm = np.clip(hp / hi, 0.0, 1.0)

    # ``blob_log`` scans multi-scale; ``num_sigma=6`` samples 6
    # scales between ``min_sigma`` and ``max_sigma``.
    # ``threshold`` is the minimum LoG response a blob must exceed.
    # ``overlap=0.5`` deduplicates blobs whose circles overlap by
    # more than 50%.
    raw = blob_log(
        norm,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=6,
        threshold=float(min_contrast),
        overlap=0.5,
    )
    out = []
    if raw.size == 0:
        return out

    # Each row is (y, x, sigma). Convert to a (y, x, radius) tuple
    # and reject blobs whose area falls outside the user-specified
    # band -- a quick way to drop both noise specks (too small) and
    # vessel artefacts (too big).
    for y, x, sigma in raw:
        radius = float(sigma) * np.sqrt(2.0)
        area = np.pi * radius ** 2
        if area < min_area_px or area > max_area_px:
            continue
        out.append((int(round(y)), int(round(x)), radius))
    return out


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess_tiff(
    src_tiff: str | Path,
    data_root: str | Path,
    recording_name: Optional[str] = None,
    soma_diameter_px: float = 12.0,
    progress_cb: Optional[Callable[[str], None]] = None,
    blob_params: Optional[dict] = None,
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
    soma_diameter_px
        Expected soma diameter used by the preview blob detector.
        (Also passed through to ``detect_blobs_on_mean`` unless overridden
        in ``blob_params``.)
    progress_cb
        Optional callback receiving status strings.
    blob_params
        Optional overrides for ``detect_blobs_on_mean`` (keys:
        ``scale_tol``, ``min_contrast``, ``min_area_px``, ``max_area_px``).
        ``soma_diameter_px`` is taken from the top-level argument.
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
    blobs_path = out_dir / "blobs.npy"

    # Local closure for the progress callback. Captures
    # ``progress_cb`` from the enclosing scope.
    def log(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    # ---- Step 1: shift to non-negative + cast to uint16 ----
    log(f"[1/4] Shifting {src.name} -> {shifted_path.name}")
    shifted = shift_tiff_to_uint16(src, shifted_path, progress_cb=log)
    # Tuple-unpack the shape: ``shifted`` is ``(T, Y, X)`` so we
    # name those three.
    T, Y, X = shifted.shape

    # ---- Step 2: per-pixel mean image ----
    # ``shifted.mean(axis=0)`` averages across the time axis (the
    # first one) -> a (Y, X) image.
    log(f"[2/4] Computing mean image ({T} frames, {Y}x{X})")
    mean = shifted.mean(axis=0).astype(np.float32)
    np.save(str(mean_path), mean)

    # ---- Step 3: candidate-cell detection on the mean ----
    # ``dict(blob_params or {})`` is a defensive copy of the
    # user's overrides; ``setdefault`` only fills the key if the
    # user hasn't already.
    log("[3/4] Detecting preview blobs on mean image")
    blob_kwargs = dict(blob_params or {})
    blob_kwargs.setdefault("soma_diameter_px", soma_diameter_px)
    blobs = detect_blobs_on_mean(mean, **blob_kwargs)
    np.save(str(blobs_path), np.asarray(blobs, dtype=np.float32))

    # ---- Step 4: animated QC GIF ----
    log("[4/4] Writing QC gif")
    make_qc_gif(shifted, gif_path, **(qc_params or {}))

    log(f"Done. Output -> {out_dir}")

    # Wrap the artefacts in the dataclass so callers don't have to
    # remember the file naming conventions.
    return PreprocessResult(
        out_dir=out_dir,
        shifted_tiff=shifted_path,
        qc_gif=gif_path,
        mean_image_path=mean_path,
        blobs_path=blobs_path,
        n_frames=int(T),
        shape_yx=(int(Y), int(X)),
        n_blobs=len(blobs),
    )


def preprocess_tiff_group(
    src_tiffs: list[str | Path],
    data_root: str | Path,
    recording_name: Optional[str] = None,
    soma_diameter_px: float = 12.0,
    progress_cb: Optional[Callable[[str], None]] = None,
    blob_params: Optional[dict] = None,
    qc_params: Optional[dict] = None,
) -> PreprocessResult:
    """End-to-end preprocessing for a group of raw TIFFs treated as one
    continuous recording.

    A single shift constant (derived from the global min across ALL inputs) is
    applied to every file, so the concatenated movie has consistent intensity.
    Each shifted output is written into
    ``<data_root>/<recording_name>/NNN_shifted_<orig>.tif`` with a zero-padded
    order prefix (suite2p uses natsort on data_path, so the prefix preserves
    input order). Mean image, preview blobs, and QC gif are computed across
    the full concatenated stack.

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
    blobs_path = out_dir / "blobs.npy"

    def log(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    # ---- Pass 1: scan all files for one shared global min/max -------------
    log(f"[1/4] Scanning {len(srcs)} TIFFs for shared global min/max")
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
                    log(f"[1/4]   {src.name}  scan {i + 1}/{n_pages}  "
                        f"min={gmin} max={gmax}")
            page_counts.append(n_pages)
            log(f"[1/4]   {src.name}: {n_pages} pages")
    shift = -gmin if gmin < 0 else 0
    shifted_max = gmax + shift
    log(f"[1/4] global min={gmin} max={gmax}  -> shift+={shift}  "
        f"(shifted max={shifted_max})")
    if shifted_max > 65535:
        raise ValueError(
            f"Shifted max {shifted_max} > 65535; uint16 overflow."
        )

    # ---- Pass 2: write each shifted file ----------------------------------
    log(f"[2/4] Writing {len(srcs)} shifted TIFFs into {out_dir}")
    for src, dst in zip(srcs, shifted_paths):
        if dst.exists():
            log(f"[2/4]   exists, skipping: {dst.name}")
            continue
        log(f"[2/4]   {src.name} -> {dst.name}")
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
                if (i + 1) % 2000 == 0:
                    log(f"[2/4]     write {i + 1}/{n_pages}")

    # ---- Pass 3: mean image across the concatenated stack -----------------
    log(f"[3/4] Mean image across {len(shifted_paths)} files")
    sum_arr: Optional[np.ndarray] = None
    total_frames = 0
    Y = X = 0
    for dst in shifted_paths:
        m = tifffile.memmap(str(dst), mode="r")
        if sum_arr is None:
            Y, X = int(m.shape[1]), int(m.shape[2])
            sum_arr = np.zeros((Y, X), dtype=np.float64)
        sum_arr += m.sum(axis=0, dtype=np.float64)
        total_frames += int(m.shape[0])
        del m
    mean = (sum_arr / max(1, total_frames)).astype(np.float32)
    np.save(str(mean_path), mean)
    log(f"[3/4]   mean over {total_frames} frames ({Y}x{X})")

    log("[3/4] Detecting preview blobs on mean image")
    blob_kwargs = dict(blob_params or {})
    blob_kwargs.setdefault("soma_diameter_px", soma_diameter_px)
    blobs = detect_blobs_on_mean(mean, **blob_kwargs)
    np.save(str(blobs_path), np.asarray(blobs, dtype=np.float32))

    # ---- Pass 4: QC gif from down-sampled concatenated frames -------------
    log("[4/4] Writing QC gif")
    qc_kwargs = dict(qc_params or {})
    downsample_t = max(1, int(qc_kwargs.pop("downsample_t", 4)))
    sampled: list[np.ndarray] = []
    for dst in shifted_paths:
        m = tifffile.memmap(str(dst), mode="r")
        sampled.append(np.asarray(m[::downsample_t]))
        del m
    movie = np.concatenate(sampled, axis=0) if len(sampled) > 1 else sampled[0]
    make_qc_gif(movie, gif_path, downsample_t=1, **qc_kwargs)

    log(f"Done. Output -> {out_dir}")

    return PreprocessResult(
        out_dir=out_dir,
        shifted_tiff=shifted_paths[0],
        qc_gif=gif_path,
        mean_image_path=mean_path,
        blobs_path=blobs_path,
        n_frames=int(total_frames),
        shape_yx=(int(Y), int(X)),
        n_blobs=len(blobs),
    )


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


def load_existing_preprocess(out_dir: str | Path) -> Optional[PreprocessResult]:
    """If a recording folder already contains preprocessing outputs, load them.

    Recognises both single-TIFF (``shifted_<x>.tif``) and grouped
    (``NNN_shifted_<x>.tif``) layouts."""
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
    blobs_path = out_dir / "blobs.npy"

    if not (shifted_candidates and gif.exists() and mean_path.exists()):
        return None

    mean = np.load(str(mean_path))
    blobs = (np.load(str(blobs_path)) if blobs_path.exists()
             else np.zeros((0, 3), dtype=np.float32))

    return PreprocessResult(
        out_dir=out_dir,
        shifted_tiff=shifted_candidates[0],
        qc_gif=gif,
        mean_image_path=mean_path,
        blobs_path=blobs_path,
        n_frames=-1,
        shape_yx=tuple(mean.shape),
        n_blobs=int(blobs.shape[0]) if blobs.ndim == 2 else 0,
    )
