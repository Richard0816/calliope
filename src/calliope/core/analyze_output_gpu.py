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

# Runtime-constructed module names avoid Nuitka's static-fold on
# --nofollow-import-to=cupy* (see crosscorrelation.py for the long story).
# "cupy" = 63757079; "cupyx.scipy.signal" = 63757079782e7363697079 + 2e7369676e616c.
try:
    import importlib as _importlib
    cp = _importlib.import_module(bytes.fromhex("63757079").decode())
    cp_signal = _importlib.import_module(
        bytes.fromhex("63757079782e73636970792e7369676e616c").decode()
    )
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
    *,
    r: float,
    fps: float,
    baseline_sec: float,
    cutoff_hz: float,
    sg_win_ms: float,
    sg_poly: int,
    perc: int = 10,
    roi_chunk: int | None = None,
    pad: int = 64,
):
    """Core GPU pipeline.  F_cell / F_neuropil must be (T, N) float32."""
    T, N = F_cell.shape
    if roi_chunk is None or roi_chunk <= 0:
        roi_chunk = N

    n_baseline = max(3, min(T, int(round(baseline_sec * fps))))
    sos_gpu = cp.asarray(_butter_sos(fps, cutoff_hz))

    dff_out = np.empty((T, N), dtype=np.float32)
    lp_out = np.empty((T, N), dtype=np.float32)
    dt_out = np.empty((T, N), dtype=np.float32)

    for start in range(0, N, roi_chunk):
        end = min(N, start + roi_chunk)

        Fc = cp.asarray(F_cell[:, start:end], dtype=cp.float32)
        Fn = cp.asarray(F_neuropil[:, start:end], dtype=cp.float32)

        F_corr = Fc - r * Fn
        # match CPU: percentile of first-N-min window, not mean
        F0 = cp.percentile(F_corr[:n_baseline, :], perc, axis=0, keepdims=True)
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

    return dff_out, lp_out, dt_out


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
    dff, lp, dt = _gpu_process_all(
        F_t, Fneu_t,
        r=r, fps=fps, baseline_sec=baseline_sec,
        cutoff_hz=cutoff_hz, sg_win_ms=sg_win_ms, sg_poly=sg_poly,
        roi_chunk=roi_chunk,
    )
    print(f"[GPU] {N} ROIs × {T} frames in {time.time() - t0:.2f}s")

    dff_mm[:] = dff
    lp_mm[:] = lp
    dt_mm[:] = dt
    dff_mm.flush(); lp_mm.flush(); dt_mm.flush()
    del dff_mm, lp_mm, dt_mm
