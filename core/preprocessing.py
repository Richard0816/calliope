"""
preprocessing.py - raw TIFF -> shifted TIFF + QC gif + mean image.

Wraps the logic in shifting.py into a callable pipeline suitable for the GUI.
Outputs land under ``<data_root>/<recording_name>/``:
    shifted_<original>.tif         (single-TIFF case)
    NNN_shifted_<original>.tif     (multi-TIFF / group case; matches naming in
                                    run_full_pipeline.shift_tiff_group)
    qc.gif
    mean.npy
    blobs.npy                      (detected soma candidates on the mean image)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import tifffile
from PIL import Image


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PreprocessResult:
    out_dir: Path
    shifted_tiff: Path
    qc_gif: Path
    mean_image_path: Path
    blobs_path: Path
    n_frames: int
    shape_yx: tuple
    n_blobs: int

    def to_dict(self) -> dict:
        d = asdict(self)
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
        data = tifffile.imread(str(src_tiff)).astype(np.int32)
        data += int(abs(data.min()))
        if data.max() >= 65535:
            raise ValueError(
                f"Shifted range exceeds uint16 ({data.max()}); "
                f"raw dynamic range too wide."
            ) #todo create alternative solution for this case
        data = data.astype(np.uint16)
        tifffile.imwrite(str(dst_tiff), data)
        return data
    except MemoryError:
        _log("[shift] MemoryError on in-RAM path; "
             "retrying with streaming shift")
        _shift_tiff_streaming(src_tiff, dst_tiff, progress_cb=_log)
        return tifffile.memmap(str(dst_tiff), mode="r")


def _shift_tiff_streaming(
    src_tiff: Path,
    dst_tiff: Path,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> None:
    """Page-by-page shift + uint16 rewrite. Never holds more than one
    frame in RAM; uses BigTIFF on the output to stay safe past 4 GB.
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
    """Write a compact grayscale GIF for visual QC."""
    if movie.ndim != 3:
        raise ValueError(f"Expected (T, Y, X) stack, got shape {movie.shape}")

    frames = movie[::max(1, int(downsample_t))]
    lo = float(np.percentile(frames, clip_low))
    hi = float(np.percentile(frames, clip_high))
    span = max(hi - lo, 1e-6)
    norm = np.clip((frames.astype(np.float32) - lo) / span, 0.0, 1.0)
    u8 = (norm * 255.0).astype(np.uint8)

    pil_frames = []
    for f in u8:
        im = Image.fromarray(f, mode="L")
        if max(im.size) > max_size_px:
            scale = max_size_px / max(im.size)
            new_size = (max(1, int(im.size[0] * scale)),
                        max(1, int(im.size[1] * scale)))
            im = im.resize(new_size, Image.BILINEAR)
        pil_frames.append(im)

    duration_ms = max(1, int(round(1000.0 / playback_fps)))
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
    """
    Lightweight LoG blob pass on the raw mean image (no Suite2p ROIs yet).
    Same geometric tuning as ``adaptive_detection.detect_residual_blobs`` but
    without the residual-masking step. Returns [(y, x, radius), ...].
    """
    from skimage.feature import blob_log
    from scipy.ndimage import median_filter

    r = float(soma_diameter_px) / 2.0
    min_sigma = (r * (1.0 - scale_tol)) / np.sqrt(2.0)
    max_sigma = (r * (1.0 + scale_tol)) / np.sqrt(2.0)

    img = mean_img.astype(np.float32)
    bg_radius = max(3, int(round(soma_diameter_px)))
    bg_local = median_filter(img, size=bg_radius)
    hp = np.clip(img - bg_local, 0.0, None)

    hi = float(np.quantile(hp, 0.995))
    if hi <= 0:
        return []
    norm = np.clip(hp / hi, 0.0, 1.0)

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
    src = Path(src_tiff).resolve()
    if not src.exists():
        raise FileNotFoundError(src)

    name = recording_name or src.stem
    out_dir = Path(data_root).resolve() / name
    out_dir.mkdir(parents=True, exist_ok=True)

    shifted_path = out_dir / f"shifted_{src.name}"
    gif_path = out_dir / "qc.gif"
    mean_path = out_dir / "mean.npy"
    blobs_path = out_dir / "blobs.npy"

    def log(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    log(f"[1/4] Shifting {src.name} -> {shifted_path.name}")
    shifted = shift_tiff_to_uint16(src, shifted_path, progress_cb=log)
    T, Y, X = shifted.shape

    log(f"[2/4] Computing mean image ({T} frames, {Y}x{X})")
    mean = shifted.mean(axis=0).astype(np.float32)
    np.save(str(mean_path), mean)

    log("[3/4] Detecting preview blobs on mean image")
    blob_kwargs = dict(blob_params or {})
    blob_kwargs.setdefault("soma_diameter_px", soma_diameter_px)
    blobs = detect_blobs_on_mean(mean, **blob_kwargs)
    np.save(str(blobs_path), np.asarray(blobs, dtype=np.float32))

    log("[4/4] Writing QC gif")
    make_qc_gif(shifted, gif_path, **(qc_params or {}))

    log(f"Done. Output -> {out_dir}")

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
