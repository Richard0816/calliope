"""Headless low-pass filter + Savitzky-Golay first-derivative
computation for batch execution and Tab 4.

Why this module exists
----------------------
Tab 4's compute loop used to live inline in ``LowpassTab._run_compute``,
which means the batch runner couldn't invoke the same code without
spinning up a Tk window. This module lifts the math out so both the
GUI tab (still does the interactive plotting + slider) and the batch
runner can call the same function.

Public surface
--------------
``compute_lowpass_and_dt(plane0, fps, cutoff_hz, ...)``
    Read ``r0p7_filtered_dff.memmap.float32``, write
    ``r0p7_filtered_dff_lowpass.memmap.float32`` and
    ``r0p7_filtered_dff_dt.memmap.float32``.
``render_lowpass_figures(plane0, fps, cutoff_hz, figures_dir, ...)``
    Render the same three diagnostic plots Tab 4 shows (FFT, raw
    trace, low-pass trace) for the population-mean dF/F and save
    them as PNGs into ``figures_dir``.
``run_lowpass(plane0, params, figures_dir=None, progress_cb=None)``
    Convenience wrapper used by the batch runner: resolves fps + N,
    runs ``compute_lowpass_and_dt``, and (if ``figures_dir`` is set)
    renders the diagnostic figures.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import utils


def _load_kept_count_and_T(plane0: Path) -> tuple[int, int]:
    """Resolve (T, N_kept) from suite2p outputs at ``plane0``.

    Mirrors ``LowpassTab._load_data`` -- prefers
    ``predicted_cell_mask.npy`` (cellfilter), falls back to
    ``iscell.npy`` (suite2p classifier), then to "all ROIs kept".
    """
    F = np.load(plane0 / "F.npy", mmap_mode="r")
    N_total, T = F.shape
    mask_path = plane0 / "predicted_cell_mask.npy"
    if mask_path.exists():
        mask = np.load(mask_path).astype(bool)
    else:
        iscell_path = plane0 / "iscell.npy"
        if iscell_path.exists():
            ic = np.load(iscell_path)
            mask = ((ic[:, 0] > 0) if ic.ndim == 2 else (ic > 0)).astype(bool)
        else:
            mask = np.ones(N_total, dtype=bool)
    n_kept = int(mask.sum())
    if n_kept == 0:
        raise RuntimeError(f"No ROIs survive the cell-filter mask at {plane0}.")
    return int(T), n_kept


def compute_lowpass_and_dt(
    plane0: Path,
    *,
    fps: float,
    cutoff_hz: float,
    filter_order: int = 2,
    sg_win_ms: int = 333,
    sg_poly: int = 2,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """Run causal Butterworth low-pass + SG first-derivative on every
    kept ROI in ``plane0``. Writes two ``(T, N_kept)`` float32 memmaps:

        ``r0p7_filtered_dff_lowpass.memmap.float32``
        ``r0p7_filtered_dff_dt.memmap.float32``

    Returns a summary dict (cutoff, fps, T, N, written paths).

    ``progress_cb`` (if provided) is called with status strings such
    as ``"120/2350 ROIs (3.4s)"`` every ~5% of progress.
    """
    plane0 = Path(plane0)
    T, N = _load_kept_count_and_T(plane0)

    src_path = plane0 / "r0p7_filtered_dff.memmap.float32"
    if not src_path.exists():
        raise FileNotFoundError(
            f"Missing {src_path.name}; run Tab 3 first.")
    src = np.memmap(str(src_path), dtype="float32", mode="r", shape=(T, N))

    lp_path = plane0 / "r0p7_filtered_dff_lowpass.memmap.float32"
    dt_path = plane0 / "r0p7_filtered_dff_dt.memmap.float32"
    lp_mm = np.memmap(str(lp_path), dtype="float32", mode="w+", shape=(T, N))
    dt_mm = np.memmap(str(dt_path), dtype="float32", mode="w+", shape=(T, N))

    sos = None
    report_every = max(1, N // 20)
    t0 = time.time()
    for i in range(N):
        trace = np.asarray(src[:, i], dtype=np.float32)
        lp, _, sos = utils.lowpass_causal_1d(
            trace, fps=fps, cutoff_hz=cutoff_hz, order=filter_order,
            zi=None, sos=sos)
        dt = utils.sg_first_derivative_1d(
            lp, fps=fps, win_ms=sg_win_ms, poly=sg_poly)
        lp_mm[:, i] = lp
        dt_mm[:, i] = dt
        if progress_cb is not None and (i + 1) % report_every == 0:
            progress_cb(f"{i + 1}/{N} ROIs ({time.time() - t0:.1f}s)")

    lp_mm.flush()
    dt_mm.flush()
    del lp_mm, dt_mm, src

    return {
        "cutoff_hz": float(cutoff_hz),
        "fps": float(fps),
        "T": T,
        "N": N,
        "output_files": [str(lp_path), str(dt_path)],
    }


def _population_mean_filtered_dff(plane0: Path, T: int, N: int) -> np.ndarray:
    """Compute the population-mean trace from ``r0p7_filtered_dff`` in
    chunks so very long recordings don't blow memory.
    """
    src_path = plane0 / "r0p7_filtered_dff.memmap.float32"
    src = np.memmap(str(src_path), dtype="float32", mode="r", shape=(T, N))
    out = np.zeros(T, dtype=np.float32)
    chunk = max(1, min(T, 8192))
    for t0 in range(0, T, chunk):
        t1 = min(T, t0 + chunk)
        out[t0:t1] = np.mean(np.asarray(src[t0:t1, :]), axis=1)
    del src
    return out


def render_lowpass_figures(
    plane0: Path,
    *,
    fps: float,
    cutoff_hz: float,
    figures_dir: Path,
    filter_order: int = 2,
) -> list[str]:
    """Render the three diagnostic Tab 4 plots for the population mean
    dF/F and save them as ``fft.png``, ``raw_dff.png``,
    ``lowpass_dff.png`` under ``figures_dir``. Returns the paths.

    Uses the matplotlib ``Agg`` backend so it works in headless batch
    contexts without a display.
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    T, N = _load_kept_count_and_T(plane0)
    mean_dff = _population_mean_filtered_dff(plane0, T, N)
    label = f"mean across {N} kept ROIs"

    yf = np.fft.rfft(mean_dff)
    xf = np.fft.rfftfreq(mean_dff.size, 1.0 / fps)
    power = np.abs(yf) ** 2
    lp, _, _ = utils.lowpass_causal_1d(
        mean_dff, fps=fps, cutoff_hz=cutoff_hz, order=filter_order,
        zi=None, sos=None)
    t = np.arange(mean_dff.size, dtype=np.float32) / fps

    written = []

    fig = plt.Figure(figsize=(8, 2.4), tight_layout=True)
    ax = fig.add_subplot(111)
    ax.semilogy(xf[1:], np.maximum(power[1:], 1e-12), lw=0.8, color="black")
    ax.axvline(cutoff_hz, color="tab:red", lw=1.2, linestyle="--",
               label=f"cutoff {cutoff_hz:.2f} Hz")
    ax.set_xlim(0, min(fps / 2.0, 15.0))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (log)")
    ax.set_title(f"FFT - {label}")
    ax.legend(loc="upper right", fontsize=8)
    p = figures_dir / "fft.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(str(p))

    fig = plt.Figure(figsize=(8, 2.4), tight_layout=True)
    ax = fig.add_subplot(111)
    ax.plot(t, mean_dff, lw=0.6, color="black")
    ax.set_xlim(0, t[-1] if t.size else 1.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("dF/F")
    ax.set_title(f"Raw dF/F - {label}")
    p = figures_dir / "raw_dff.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(str(p))

    fig = plt.Figure(figsize=(8, 2.4), tight_layout=True)
    ax = fig.add_subplot(111)
    ax.plot(t, lp, lw=0.7, color="tab:blue")
    ax.set_xlim(0, t[-1] if t.size else 1.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"dF/F low-pass @ {cutoff_hz:.2f} Hz")
    ax.set_title(f"Low-pass - {label}")
    p = figures_dir / "lowpass_dff.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(str(p))

    return written


def run_lowpass(
    plane0: Path,
    params: dict,
    *,
    figures_dir: Optional[Path] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """Batch-friendly wrapper. ``params`` may include:

      * ``cutoff_hz`` (default 1.0)
      * ``filter_order`` (default 2)
      * ``sg_win_ms`` (default 333)
      * ``sg_poly`` (default 2)
      * ``fps_override`` (0 or absent -> read from notes.csv)

    ``figures_dir``, when provided, gets ``fft.png`` /
    ``raw_dff.png`` / ``lowpass_dff.png`` for the population-mean
    trace.
    """
    plane0 = Path(plane0)
    cutoff = float(params.get("cutoff_hz", 1.0))
    order = int(params.get("filter_order", 2))
    sg_win = int(params.get("sg_win_ms", 333))
    sg_poly = int(params.get("sg_poly", 2))
    fps_override = float(params.get("fps_override", 0.0) or 0.0)
    fps = fps_override if fps_override > 0 \
        else float(utils.get_fps_from_notes(str(plane0)))

    summary = compute_lowpass_and_dt(
        plane0, fps=fps, cutoff_hz=cutoff,
        filter_order=order, sg_win_ms=sg_win, sg_poly=sg_poly,
        progress_cb=progress_cb,
    )
    if figures_dir is not None:
        figs = render_lowpass_figures(
            plane0, fps=fps, cutoff_hz=cutoff,
            figures_dir=Path(figures_dir),
            filter_order=order,
        )
        summary["figures"] = figs
    return summary
