"""Z-drift detection via PCA over per-ROI fluorescence.

Slow axial (z) drift of the imaging plane shows up as a coordinated,
step-like shift across many ROIs at once. Running PCA over the per-ROI
dF/F and watching the first principal component (PC1) -- the dominant
shared mode -- surfaces those steps: a real z-drift event appears as an
abrupt jump in PC1, distinct from the per-cell transients that average
out across the population.

This is a *detection* (QC) step, not a correction. Flagged frames are
reported (a ``zdrift_qc.png`` figure + a flag in ``meta.json``) for the
user to judge. Dropping frames would invalidate the suite2p registration
and ROI detection that produced these traces, so frame exclusion is
deliberately out of scope here.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

PathLike = Union[str, Path]

#: Cap on ROIs used to estimate the PC1 spatial loading. The shared z-drift
#: mode is captured fine by a random subset; this bounds the O(N^2) cost.
_ROI_CAP = 1000
#: Cap on time samples used for the eigen-fit (the loading); the full-length
#: PC1 trajectory is then recovered by projecting all frames onto it.
_TIME_CAP_FIT = 4000


def _pc1_over_time(
    dff_TN: np.ndarray, *, roi_cap: int = _ROI_CAP,
    time_cap_for_fit: int = _TIME_CAP_FIT, seed: int = 0,
) -> Tuple[np.ndarray, float]:
    """Return (PC1 time series (T,), fraction of variance explained).

    Each ROI column is z-scored so bright cells don't dominate. The PC1
    spatial loading is estimated from a (capped) ROI/time subset via the
    ROI covariance eigenvector, then the *full-length* PC1 trajectory is
    recovered by projecting every frame onto that loading.
    """
    X = np.asarray(dff_TN, dtype=np.float32)
    T, N = X.shape
    rng = np.random.default_rng(seed)
    roi_idx = (np.sort(rng.choice(N, roi_cap, replace=False))
               if N > roi_cap else np.arange(N))
    Xr = X[:, roi_idx]
    mu = Xr.mean(axis=0, keepdims=True)
    sd = Xr.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    Z = (Xr - mu) / sd  # (T, n_roi) float32

    # Time-subsample only for the eigen-fit of the spatial loading.
    if T > time_cap_for_fit:
        step = int(np.ceil(T / time_cap_for_fit))
        Zfit = np.asarray(Z[::step], dtype=np.float64)
    else:
        Zfit = np.asarray(Z, dtype=np.float64)
    C = (Zfit.T @ Zfit) / max(1, Zfit.shape[0] - 1)
    evals, evecs = np.linalg.eigh(C)  # ascending
    v = evecs[:, -1]
    # Sign convention: largest-magnitude loading positive (stable plots).
    if v[int(np.argmax(np.abs(v)))] < 0:
        v = -v
    pc1 = (Z @ v).astype(np.float64)  # full-length projection
    total = float(np.sum(evals))
    frac = float(evals[-1] / total) if total > 0 else 0.0
    return pc1, frac


def _find_steps(pc1: np.ndarray, *, step_k: float,
                min_separation_frames: int) -> List[int]:
    """Robustly flag PC1 step changes: frames whose frame-to-frame |ΔPC1|
    exceeds ``step_k`` x (1.4826·MAD) above the median, merged within
    ``min_separation_frames`` into one step. Returns the jump-frame indices.
    """
    d = np.abs(np.diff(pc1))
    if d.size == 0:
        return []
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med))) or 1e-12
    thresh = med + float(step_k) * 1.4826 * mad
    flagged = (np.flatnonzero(d > thresh) + 1).tolist()
    steps: List[int] = []
    for f in flagged:
        if not steps or f - steps[-1] > int(min_separation_frames):
            steps.append(int(f))
    return steps


def _write_zdrift_csv(plane0: Path, pc1: np.ndarray, steps: List[int],
                      fps: float) -> Path:
    """Persist the PC1 trace so the QC figure is fully recreatable from a
    plain CSV. Columns: ``frame, time_s, pc1, is_step`` (``time_s`` blank
    when no frame rate is known). Returns the written CSV path.
    """
    step_set = set(int(s) for s in steps)
    use_seconds = bool(fps and fps > 0)
    out = Path(plane0) / "zdrift_pc1.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "time_s", "pc1", "is_step"])
        for i, val in enumerate(pc1.tolist()):
            t = (i / float(fps)) if use_seconds else ""
            w.writerow([i, t, val, 1 if i in step_set else 0])
    return out


def _render_zdrift_figure(plane0: Path, pc1: np.ndarray, steps: List[int],
                          frac: float, fps: float) -> Path:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    T = int(pc1.size)
    use_seconds = bool(fps and fps > 0)
    x = (np.arange(T) / float(fps)) if use_seconds else np.arange(T)
    fig = Figure(figsize=(10, 3.2), dpi=130)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.plot(x, pc1, lw=0.7, color="0.2")
    for s in steps:
        xs = (s / float(fps)) if use_seconds else s
        ax.axvline(xs, color="crimson", lw=0.8, alpha=0.7)
    ax.set_xlabel("time (s)" if use_seconds else "frame")
    ax.set_ylabel("PC1 (shared mode)")
    n = len(steps)
    ax.set_title(
        f"Z-drift QC: {n} step{'s' if n != 1 else ''} flagged "
        f"(PC1 explains {frac * 100:.1f}% of variance)")
    fig.tight_layout()
    out = Path(plane0) / "zdrift_qc.png"
    fig.savefig(out)
    return out


def detect_zdrift(
    plane0: PathLike,
    *,
    step_k: float = 6.0,
    min_separation_frames: int = 5,
    fps: float = 0.0,
    make_figure: bool = True,
    write_csv: bool = True,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """Detect z-drift steps from the per-ROI dF/F memmap under ``plane0``.

    Computes PC1 over all ROIs' dF/F, flags frames whose frame-to-frame
    PC1 change exceeds ``step_k`` x MAD (robust), merges flags within
    ``min_separation_frames`` into one step, writes a QC figure
    (``zdrift_qc.png``) when ``make_figure``, and stamps the result into
    ``meta.json`` (qc section). Best-effort and read-only with respect to
    the pipeline arrays.

    Returns ``{detected, n_steps, step_frames, pc1_var_explained,
    figure_path}``. Returns a zeroed result (``detected=False``) when the
    dF/F memmap is missing or too short.
    """
    plane0 = Path(plane0)
    result = {
        "detected": False, "n_steps": 0, "step_frames": [],
        "pc1_var_explained": None, "figure_path": None, "csv_path": None,
    }
    f_path = plane0 / "F.npy"
    dff_path = plane0 / "r0p7_dff.memmap.float32"
    if not f_path.exists() or not dff_path.exists():
        return result
    try:
        F = np.load(f_path, mmap_mode="r")
        N, T = int(F.shape[0]), int(F.shape[1])
        del F
    except Exception:
        return result
    if T < 10 or N < 2:
        return result
    try:
        dff = np.memmap(str(dff_path), dtype="float32", mode="r",
                        shape=(T, N))
    except (OSError, ValueError):
        return result

    pc1, frac = _pc1_over_time(np.asarray(dff))
    steps = _find_steps(pc1, step_k=step_k,
                        min_separation_frames=min_separation_frames)
    result["pc1_var_explained"] = round(frac, 6)
    result["n_steps"] = len(steps)
    result["step_frames"] = steps
    result["detected"] = len(steps) > 0

    if write_csv:
        try:
            result["csv_path"] = str(
                _write_zdrift_csv(plane0, pc1, steps, fps))
        except Exception as e:
            if progress_cb is not None:
                progress_cb(f"[zdrift] csv write failed: {e}")

    if make_figure:
        try:
            result["figure_path"] = str(
                _render_zdrift_figure(plane0, pc1, steps, frac, fps))
        except Exception as e:
            if progress_cb is not None:
                progress_cb(f"[zdrift] figure render failed: {e}")

    try:
        from . import recording_meta
        recording_meta.update_meta(
            plane0,
            qc={
                "zdrift_detected": bool(result["detected"]),
                "zdrift_n_steps": int(result["n_steps"]),
                "zdrift_step_frames": list(result["step_frames"]),
                "zdrift_pc1_var_explained": result["pc1_var_explained"],
            },
        )
    except Exception as e:
        if progress_cb is not None:
            progress_cb(f"[zdrift] meta stamp failed: {e}")

    return result
