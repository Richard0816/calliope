"""Per-pixel temporal correlation image (a.k.a. Vcorr) computation.

Why this module exists
----------------------
Suite2p 1.0 produces a correlation image ``Vcorr`` as part of its detection
pass and stores it in ``plane0/detect_outputs.npy``. The Sparsery detector
fed by that pass already consumes it internally. The Cellpose-SAM second-
pass detector (audit Tier 2 #13) needs the same image, but it runs on the
*registration-only* output (before Sparsery), so Vcorr may not be available
yet -- and on legacy plane folders predating suite2p 1.0 the file may not
exist at all.

``compute_vcorr`` is the fallback: it reads ``data.bin`` in chunks, computes
per-pixel Pearson correlation against the local 4-neighbour mean, and writes
a ``Vcorr.npy`` snapshot to the plane folder. The audit's recommendation
(2026-05-25 §2.4) is that this is the right input for the SAM pass because:

- The mean image is biased toward bright/active cells -- SAM trained on it
  picks the same cells Sparsery does.
- The max projection is contaminated by transient bright pixels.
- The correlation image balances activity (a cell's pixels co-fluctuate
  across time) and anatomy (its footprint is spatially compact).

Reference: Stringer & Pachitariu (Cellpose 3 + Cellpose-SAM, *Nat Methods*
2025 / bioRxiv 2025) recommend feeding anatomically-informative images
through Suite2p's ``anatomical_only`` knob; Vcorr is the version of that
which carries temporal information.

Memory / streaming
------------------
Reading ``data.bin`` as one ``(T, Ly, Lx)`` array would consume
``T * Ly * Lx * 2`` bytes (data.bin is int16). For a 50000-frame, 512x512
recording that's ~26 GB. We instead read in chunks of ``chunk_frames``
frames and accumulate the running pixel statistics
(``sum``, ``sum_of_squares``, ``sum_of_products``) -- O(Ly*Lx) memory
regardless of T.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np


def _streaming_pearson_with_neighbours(
    data_bin_path: Path,
    Ly: int,
    Lx: int,
    nframes: int,
    *,
    chunk_frames: int = 1000,
    dtype_disk: np.dtype = np.dtype(np.int16),
) -> np.ndarray:
    """Walk ``data.bin`` in chunks, accumulate pixel + 4-neighbour-mean
    statistics, and return the per-pixel Pearson correlation as
    ``(Ly, Lx)`` float32.

    Per-frame neighbour mean ``N_t`` is the 4-connected mean (up, down,
    left, right). The Pearson correlation between a pixel's time series
    ``X`` and its neighbours' mean ``N`` reduces to:

        r = (T * Sxy - Sx * Sy) / sqrt((T * Sxx - Sx**2) * (T * Syy - Sy**2))

    where Sx = sum(X), Sy = sum(N), Sxy = sum(X * N), Sxx = sum(X**2),
    Syy = sum(N**2), all summed across the T frames. The four sums can
    be accumulated chunk by chunk, so we never need the full movie in
    memory.

    Edge pixels (those on the FOV border) compute the neighbour mean from
    however many neighbours are available -- 2 for corners, 3 for edges.
    """
    # Per-pixel accumulators. float64 for the dot products since X * N
    # over tens of thousands of frames can overflow float32 precision on
    # bright pixels even though the raw values are int16.
    Sx = np.zeros((Ly, Lx), dtype=np.float64)
    Sy = np.zeros((Ly, Lx), dtype=np.float64)
    Sxx = np.zeros((Ly, Lx), dtype=np.float64)
    Syy = np.zeros((Ly, Lx), dtype=np.float64)
    Sxy = np.zeros((Ly, Lx), dtype=np.float64)

    # Memmap the binary, then slice ``chunk_frames`` at a time and copy
    # those out. Suite2p's data.bin layout is ``(T, Ly, Lx)`` int16.
    bytes_per_frame = Ly * Lx * dtype_disk.itemsize
    expected_size = nframes * bytes_per_frame
    actual_size = Path(data_bin_path).stat().st_size
    if actual_size < expected_size:
        # Some recordings have truncated tails (e.g. acquisition aborted).
        # Reduce nframes to whatever fits cleanly.
        nframes = actual_size // bytes_per_frame
    mm = np.memmap(str(data_bin_path), dtype=dtype_disk, mode="r",
                   shape=(nframes, Ly, Lx))

    # Count of neighbours per pixel (4 interior, 3 on edges, 2 at corners).
    # Used to normalise the chunk-wise neighbour-mean computation.
    neigh_count = np.zeros((Ly, Lx), dtype=np.float32)
    neigh_count[1:, :] += 1  # up neighbour exists
    neigh_count[:-1, :] += 1  # down neighbour exists
    neigh_count[:, 1:] += 1  # left neighbour exists
    neigh_count[:, :-1] += 1  # right neighbour exists
    # ``np.maximum(..., 1.0)`` guards a degenerate 1-px-wide/tall FOV
    # where an edge pixel could have 0 neighbours; any >=2x2 FOV always
    # has >=2 per pixel, so it's a no-op there.
    inv_neigh = 1.0 / np.maximum(neigh_count, 1.0)

    T_total = 0
    for c0 in range(0, nframes, chunk_frames):
        c1 = min(c0 + chunk_frames, nframes)
        # Cast to float32 once per chunk so the per-frame arithmetic
        # stays in cache-friendly precision; promote to float64 only at
        # the per-pixel accumulator stage.
        chunk = np.asarray(mm[c0:c1], dtype=np.float32)
        # Neighbour mean: shift the chunk by one pixel in each of four
        # directions and average. ``np.roll`` would wrap around the FOV
        # edge, so we use bounded slicing + zero-pad implicitly via
        # ``neigh_sum`` accumulation and the precomputed ``neigh_count``.
        neigh_sum = np.zeros_like(chunk)
        neigh_sum[:, 1:, :] += chunk[:, :-1, :]    # up
        neigh_sum[:, :-1, :] += chunk[:, 1:, :]    # down
        neigh_sum[:, :, 1:] += chunk[:, :, :-1]    # left
        neigh_sum[:, :, :-1] += chunk[:, :, 1:]    # right
        neigh = neigh_sum * inv_neigh

        # Accumulate the five running sums. ``.sum(axis=0)`` reduces over
        # the time axis of this chunk, giving per-pixel partial sums in
        # (Ly, Lx).
        Sx += chunk.sum(axis=0, dtype=np.float64)
        Sy += neigh.sum(axis=0, dtype=np.float64)
        Sxx += (chunk.astype(np.float64) ** 2).sum(axis=0)
        Syy += (neigh.astype(np.float64) ** 2).sum(axis=0)
        Sxy += (chunk.astype(np.float64) * neigh.astype(np.float64)).sum(axis=0)
        T_total += int(c1 - c0)

    del mm

    if T_total <= 1:
        return np.zeros((Ly, Lx), dtype=np.float32)

    # Closed-form Pearson r from the accumulated sums. ``np.errstate``
    # silences the divide-by-zero / sqrt-of-negative warnings that fire
    # when a pixel is constant (e.g. dead pixels); we replace those with
    # 0 below.
    with np.errstate(divide="ignore", invalid="ignore"):
        num = T_total * Sxy - Sx * Sy
        den_x = T_total * Sxx - Sx * Sx
        den_y = T_total * Syy - Sy * Sy
        denom = np.sqrt(np.maximum(den_x, 0.0) * np.maximum(den_y, 0.0))
        r = np.where(denom > 0, num / denom, 0.0)

    # Clamp to [-1, 1] to remove tiny float overshoots from accumulator
    # rounding.
    r = np.clip(r.astype(np.float32, copy=False), -1.0, 1.0)
    return r


def compute_vcorr(
    data_bin_path: Union[str, Path],
    Ly: int,
    Lx: int,
    nframes: int,
    *,
    chunk_frames: int = 1000,
    save_to: Union[str, Path, None] = None,
) -> np.ndarray:
    """Return the per-pixel temporal correlation image for a recording.

    Parameters
    ----------
    data_bin_path : Path
        Path to a Suite2p ``data.bin`` (motion-corrected, int16,
        ``(T, Ly, Lx)`` row-major layout).
    Ly, Lx : int
        FOV height / width in pixels.
    nframes : int
        Number of frames in the recording. If ``data.bin`` is shorter
        than ``nframes * Ly * Lx * 2`` bytes (e.g. recording was
        truncated), only the actually-present frames are read.
    chunk_frames : int, optional
        Frames per streaming chunk. Larger = fewer disk reads but more
        RAM; 1000 frames at 512x512 is ~500 MB for the chunk + neighbour
        buffers, which is the sweet spot on the lab box.
    save_to : Path, optional
        If set, persist the Vcorr image to this path as ``.npy``.
        Typical use: ``plane0 / "Vcorr.npy"``.

    Returns
    -------
    Vcorr : ndarray, shape (Ly, Lx), float32
        Per-pixel Pearson r with the 4-neighbour mean. Values in
        ``[-1, 1]``. Bright = anatomically + functionally consistent
        with neighbours (a soma). Dim = isolated noise.
    """
    data_bin_path = Path(data_bin_path)
    if not data_bin_path.exists():
        raise FileNotFoundError(
            f"compute_vcorr: data.bin not found at {data_bin_path}")
    Vcorr = _streaming_pearson_with_neighbours(
        data_bin_path, int(Ly), int(Lx), int(nframes),
        chunk_frames=int(chunk_frames),
    )
    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_to, Vcorr)
    return Vcorr


def load_or_compute_vcorr(
    plane0: Union[str, Path],
    *,
    data_bin_path: Union[str, Path, None] = None,
    Ly: Union[int, None] = None,
    Lx: Union[int, None] = None,
    nframes: Union[int, None] = None,
    chunk_frames: int = 1000,
    save_filename: str = "Vcorr.npy",
) -> np.ndarray:
    """Return Vcorr for a plane folder, computing it if absent.

    Resolution order (cheapest first):

    1. ``plane0 / save_filename`` -- a previously-saved cached copy.
    2. ``utils.load_plane_view(plane0)["Vcorr"]`` -- the one Suite2p
       wrote in ``detect_outputs.npy`` after the Sparsery pass.
    3. Stream ``data.bin`` and compute it ourselves via
       :func:`compute_vcorr`; save under ``plane0 / save_filename`` so
       the next call hits step 1.

    The ``data_bin_path / Ly / Lx / nframes`` arguments are only consulted
    in step 3. Step 1 / step 2 reads ignore them.
    """
    plane0 = Path(plane0)

    # Step 1: cached file.
    cached_path = plane0 / save_filename
    if cached_path.exists():
        try:
            arr = np.load(cached_path)
            return np.asarray(arr, dtype=np.float32)
        except Exception:
            # Corrupted cache; fall through to recompute.
            pass

    # Step 2: detect_outputs.npy via load_plane_view. Lazy import so
    # callers using this module without the rest of calliope don't pay
    # for the matplotlib / suite2p dependency chain.
    try:
        from . import utils as core_utils
        view = core_utils.load_plane_view(plane0)
    except Exception:
        view = {}
    v = view.get("Vcorr")
    if v is not None:
        arr = np.asarray(v, dtype=np.float32)
        # Persist it under the canonical filename so subsequent reads
        # are O(1) regardless of whether the parent reader still has
        # the detect_outputs.npy bundle.
        try:
            np.save(cached_path, arr)
        except Exception:
            pass
        return arr

    # Step 3: compute from scratch.
    if data_bin_path is None:
        data_bin_path = plane0 / "data.bin"
    if Ly is None or Lx is None or nframes is None:
        # Pull dimensions from load_plane_view as a fallback. Skip the
        # compute if we still don't know the shape (caller error).
        try:
            from . import utils as core_utils
            view = core_utils.load_plane_view(plane0)
        except Exception:
            view = {}
        if Ly is None:
            Ly = int(view.get("Ly") or 0)
        if Lx is None:
            Lx = int(view.get("Lx") or 0)
        if nframes is None:
            nframes = int(view.get("nframes") or 0)
        if not (Ly and Lx and nframes):
            raise ValueError(
                "compute_vcorr fallback needs Ly, Lx, nframes -- could not "
                f"resolve from {plane0}; pass them explicitly.")

    return compute_vcorr(
        data_bin_path=data_bin_path,
        Ly=int(Ly), Lx=int(Lx), nframes=int(nframes),
        chunk_frames=int(chunk_frames),
        save_to=cached_path,
    )
