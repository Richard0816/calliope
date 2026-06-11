"""GPU-accelerated dF/F computation using CuPy.

Mirrors the CPU path in detection_run.compute_dff_memmaps but runs batched
over all ROIs on GPU.  The public surface is:

    _CUPY_OK : bool  — True when CuPy + a CUDA device are available
    process_suite2p_traces_gpu(F, Fneu, fps, *, r, baseline_sec, cutoff_hz,
                               sg_win_ms, sg_poly, out_dir, prefix,
                               roi_chunk) -> None
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

# CuPy is an optional GPU dependency, imported via _cuda_import (the one
# place it's handled); absent on CPU-only machines -> fall back to CPU.
try:
    from ._cuda_import import import_cupy, import_cupyx_scipy_signal
    cp = import_cupy()
    cp_signal = import_cupyx_scipy_signal()
    cp.zeros(1)          # trigger device init; raises if no CUDA device
    _CUPY_OK = True
except Exception:
    cp = None
    cp_signal = None
    _CUPY_OK = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _butter_sos(fps: float, cutoff_hz: float, order: int = 2) -> np.ndarray:
    from scipy.signal import butter
    nyq = fps / 2.0
    cutoff = min(max(1e-4, cutoff_hz), 0.95 * nyq)
    return butter(order, cutoff / nyq, btype="low", output="sos").astype(np.float32)


def _sg_derivative_gpu(x, fps: float, win_ms: float, poly: int):
    """Batched Savitzky-Golay first derivative along axis 0 on GPU."""
    T = x.shape[0]
    win = max(3, int((win_ms / 1000.0) * fps) | 1)
    if win >= T:
        win = max(3, T - (1 - T % 2))
    if win < 3 or T < 3:
        d = cp.empty_like(x)
        d[0, :] = 0.0
        d[1:, :] = (x[1:, :] - x[:-1, :]) * fps
        return d
    if hasattr(cp_signal, "savgol_filter"):
        return cp_signal.savgol_filter(
            x, window_length=win, polyorder=poly,
            deriv=1, delta=1.0 / fps, axis=0,
        ).astype(cp.float32)
    # older CuPy fallback via scipy coefficients + 1-D convolution
    from scipy.signal import savgol_coeffs
    from cupyx.scipy.ndimage import convolve1d
    coeffs = savgol_coeffs(win, poly, deriv=1, delta=1.0 / fps, use="conv").astype(np.float32)
    return convolve1d(x, weights=cp.asarray(coeffs), axis=0, mode="nearest").astype(cp.float32)


def _gpu_process_all(
    F_cell: np.ndarray,
    F_neuropil: np.ndarray,
    dff_out: np.ndarray,
    lp_out: np.ndarray,
    dt_out: np.ndarray,
    *,
    r: float,
    fps: float,
    baseline_sec: float,
    cutoff_hz: float,
    sg_win_ms: float,
    sg_poly: int,
    roi_chunk: int | None = None,
    pad: int = 64,
):
    """Core GPU pipeline.  F_cell / F_neuropil must be (T, N) float32.

    Results are written directly into the supplied ``(T, N)`` float32
    output arrays (the open ``w+`` memmaps), avoiding a second full set of
    host result buffers that would otherwise be allocated and copied.
    """
    T, N = F_cell.shape
    if roi_chunk is None or roi_chunk <= 0:
        roi_chunk = N

    # Floor at 1 frame (not 3) to match the CPU reference
    # utils.first_n_min_df_over_f_1d (max(1, ...) then min(., n)); a floor of
    # 3 would diverge from the CPU path for tiny baseline windows.
    n_baseline = max(1, min(T, int(round(baseline_sec * fps))))
    sos_gpu = cp.asarray(_butter_sos(fps, cutoff_hz))

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

        # causal sosfilt with a leading pad to suppress transient
        pad_block = cp.broadcast_to(dff[:1, :], (pad, dff.shape[1]))
        dff_pad = cp.concatenate([pad_block, dff], axis=0)
        lp = cp_signal.sosfilt(sos_gpu, dff_pad, axis=0)[pad:, :].astype(cp.float32)

        deriv = _sg_derivative_gpu(lp, fps=fps, win_ms=sg_win_ms, poly=sg_poly)

        dff_out[:, start:end] = cp.asnumpy(dff)
        lp_out[:, start:end] = cp.asnumpy(lp)
        dt_out[:, start:end] = cp.asnumpy(deriv)

        del Fc, Fn, F_corr, F0, F0_safe, dff, pad_block, dff_pad, lp, deriv
        cp.get_default_memory_pool().free_all_blocks()

    # End-of-function cleanup: drop the SOS coefficients still on
    # device, sync the stream so the per-chunk pool frees have
    # actually run, then return both pools to the driver. Without
    # this, ``dedicated GPU memory`` in Task Manager stays elevated
    # after the function returns because CuPy's pool keeps the blocks
    # cached for reuse.
    try:
        del sos_gpu
    except Exception:
        pass
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
    cutoff_hz: float = 1.0,
    sg_win_ms: float = 333,
    sg_poly: int = 2,
    out_dir: str | None = None,
    prefix: str = "r0p7_",
    roi_chunk: int | None = None,
) -> None:
    """Compute neuropil-corrected dF/F on GPU and write memmaps.

    F and Fneu are the suite2p arrays in (N_ROIs, T) layout.
    Writes:
        <out_dir>/<prefix>dff.memmap.float32
        <out_dir>/<prefix>dff_lowpass.memmap.float32
        <out_dir>/<prefix>dff_dt.memmap.float32
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
    lp_mm = np.memmap(str(out_path / f"{prefix}dff_lowpass.memmap.float32"),
                      mode="w+", dtype="float32", shape=shape)
    dt_mm = np.memmap(str(out_path / f"{prefix}dff_dt.memmap.float32"),
                      mode="w+", dtype="float32", shape=shape)

    t0 = time.time()
    # Write results straight into the open memmaps instead of allocating a
    # second full (T, N) host array per output and copying it back.
    _gpu_process_all(
        F_t, Fneu_t,
        dff_mm, lp_mm, dt_mm,
        r=r, fps=fps, baseline_sec=baseline_sec,
        cutoff_hz=cutoff_hz, sg_win_ms=sg_win_ms, sg_poly=sg_poly,
        roi_chunk=roi_chunk,
    )
    print(f"[GPU] {N} ROIs × {T} frames in {time.time() - t0:.2f}s")

    dff_mm.flush(); lp_mm.flush(); dt_mm.flush()
    del dff_mm, lp_mm, dt_mm
