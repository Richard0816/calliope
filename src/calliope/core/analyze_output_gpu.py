"""GPU-accelerated dF/F computation using CuPy.

Mirrors the CPU path in detection_run.compute_dff_memmaps but runs batched
over all ROIs on GPU.

The public surface is:

    _CUPY_OK : bool  — True when CuPy + a CUDA device are available
    process_suite2p_traces_gpu(F, Fneu, fps, *, r, baseline_sec, out_dir,
                               prefix, roi_chunk) -> None
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

# CuPy is an optional GPU dependency, imported via _cuda_import (the one
# place it's handled); absent on CPU-only machines -> fall back to CPU.
try:
    from ._cuda_import import import_cupy
    cp = import_cupy()
    cp.zeros(1)          # trigger device init; raises if no CUDA device
    _CUPY_OK = True
except Exception:
    cp = None
    _CUPY_OK = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gpu_process_all(
    F_cell: np.ndarray,
    F_neuropil: np.ndarray,
    dff_out: np.ndarray,
    *,
    r: float,
    fps: float,
    baseline_sec: float,
    roi_chunk: int | None = None,
):
    """Core GPU pipeline.  F_cell / F_neuropil must be (T, N) float32.

    Results are written directly into the supplied ``(T, N)`` float32
    output array (the open ``w+`` memmap), avoiding a second full host
    result buffer that would otherwise be allocated and copied.
    """
    T, N = F_cell.shape
    if roi_chunk is None or roi_chunk <= 0:
        roi_chunk = N

    # Floor at 1 frame (not 3) to match the CPU reference
    # utils.first_n_min_df_over_f_1d (max(1, ...) then min(., n)); a floor of
    # 3 would diverge from the CPU path for tiny baseline windows.
    n_baseline = max(1, min(T, int(round(baseline_sec * fps))))

    for start in range(0, N, roi_chunk):
        end = min(N, start + roi_chunk)

        Fc = cp.asarray(F_cell[:, start:end], dtype=cp.float32)
        Fn = cp.asarray(F_neuropil[:, start:end], dtype=cp.float32)

        F_corr = Fc - r * Fn
        # Match CPU first_n_min_df_over_f_1d's non-finite handling: linearly
        # interpolate NaN/Inf samples per ROI before computing the baseline,
        # so a single bad sample doesn't turn the whole ROI's dF/F into NaN
        # (which would silently diverge from the CPU path). Only round-trips
        # to the host when non-finite values are actually present.
        finite = cp.isfinite(F_corr)
        if not bool(finite.all()):
            Fc_host = cp.asnumpy(F_corr)             # (T, chunk)
            fin_host = np.isfinite(Fc_host)
            xs = np.arange(T)
            for j in range(Fc_host.shape[1]):
                col_fin = fin_host[:, j]
                if col_fin.all():
                    continue
                if not col_fin.any():
                    Fc_host[:, j] = 0.0              # all-NaN ROI -> finite 0
                else:
                    Fc_host[:, j] = np.interp(
                        xs, np.flatnonzero(col_fin),
                        Fc_host[col_fin, j]).astype(np.float32)
            F_corr = cp.asarray(Fc_host, dtype=cp.float32)
        # match CPU first_n_min_df_over_f_1d: mean of the first-N-min window
        F0 = cp.mean(F_corr[:n_baseline, :], axis=0, keepdims=True)
        F0_safe = cp.maximum(F0, cp.full_like(F0, 1e-9))
        dff = (F_corr - F0) / F0_safe

        dff_out[:, start:end] = cp.asnumpy(dff)

        del Fc, Fn, F_corr, F0, F0_safe, dff
        cp.get_default_memory_pool().free_all_blocks()

    # End-of-function cleanup: sync the stream so the per-chunk pool
    # frees have actually run, then return both pools to the driver.
    # Without this, ``dedicated GPU memory`` in Task Manager stays
    # elevated after the function returns because CuPy's pool keeps the
    # blocks cached for reuse.
    try:
        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass
    try:
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass
    try:
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_suite2p_traces_gpu(
    F: np.ndarray,
    Fneu: np.ndarray,
    fps: float,
    *,
    r: float = 0.7,
    baseline_sec: float = 60.0,
    out_dir: str | None = None,
    prefix: str = "r0p7_",
    roi_chunk: int | None = None,
) -> None:
    """Compute neuropil-corrected dF/F on GPU and write its memmap.

    F and Fneu are the suite2p arrays in (N_ROIs, T) layout.
    Writes:
        <out_dir>/<prefix>dff.memmap.float32
    """
    if not _CUPY_OK:
        raise RuntimeError("CuPy not available")

    # suite2p layout: (N, T) -> transpose to (T, N) for GPU processing
    F = np.asarray(F, dtype=np.float32, order="C")
    Fneu = np.asarray(Fneu, dtype=np.float32, order="C")
    N, T = F.shape
    F_t = np.ascontiguousarray(F.T)    # (T, N)
    Fneu_t = np.ascontiguousarray(Fneu.T)

    out_path = Path(out_dir) if out_dir else Path(".")
    shape = (T, N)
    dff_mm = np.memmap(str(out_path / f"{prefix}dff.memmap.float32"),
                       mode="w+", dtype="float32", shape=shape)

    t0 = time.time()
    # Write results straight into the open memmap instead of allocating a
    # second full (T, N) host array and copying it back.
    _gpu_process_all(
        F_t, Fneu_t, dff_mm,
        r=r, fps=fps, baseline_sec=baseline_sec,
        roi_chunk=roi_chunk,
    )
    print(f"[GPU] {N} ROIs × {T} frames in {time.time() - t0:.2f}s")

    dff_mm.flush()
    del dff_mm
