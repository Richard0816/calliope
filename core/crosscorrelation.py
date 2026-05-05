"""Trimmed copy of ``crosscorrelation.py`` for the calliope GUI.

Contains only the public symbols used by ``crosscorrelation_tab.py`` plus
their transitive private helpers. Unrelated plotting helpers, CLI mains and
legacy GPU/CPU per-pair routines have been dropped. Logic is preserved
verbatim from the original module.
"""

from typing import Optional

import numpy as np
from pathlib import Path

try:
    import cupy as cp
except ImportError:
    cp = None
    print("[warn] CuPy not available; GPU cross-correlation will fall back to CPU.")


def _zscore_cols_xp(X, xp, eps=1e-12):
    """Z-score each column on either numpy or cupy. Column with std==0 -> zeros."""
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd_safe = xp.where(sd == 0, xp.asarray(1.0, dtype=X.dtype), sd)
    Z = (X - mu) / sd_safe
    # Mask zero-std columns to zero
    bad = (sd == 0).reshape(-1)
    if bool(bad.any()):
        Z[:, bad] = 0
    return Z


def batch_xcorr_clusters(X_A, X_B, fps, max_lag_seconds, *,
                         use_gpu=True, ZA_pre=None, ZB_pre=None,
                         dtype=np.float32):
    """Best-lag + zero-lag cross-correlation for ALL pairs (i, j) in
    ``X_A[:, i]`` x ``X_B[:, j]``.

    Returns
    -------
    best_lag_sec : (nA, nB) ndarray
    max_corr     : (nA, nB) ndarray  — biased Pearson r at best lag
    zero_lag_corr: (nA, nB) ndarray  — Pearson r at lag 0

    Algorithm (one matmul per lag, all pairs at once):
      1. Z-score columns of A and B.
      2. Pad ZA on both ends with L zeros.
      3. For each lag k in [-L, L]: shift = a strided view; one matmul
         ``shift.T @ ZB`` returns the (nA, nB) correlation matrix at lag k.
      4. Track running per-pair argmax across lags.
      5. Return best_lag, max_corr, zero_lag_corr (the lag-0 matrix).

    The per-pair Python overhead is gone: cost is dominated by
    ``(2L+1)`` GEMMs, each one giving the full (nA, nB) matrix.
    """
    use_gpu = bool(use_gpu and cp is not None)
    xp = cp if use_gpu else np

    XA = xp.asarray(X_A, dtype=dtype)
    XB = xp.asarray(X_B, dtype=dtype)
    T, nA = XA.shape
    nB = XB.shape[1]

    if ZA_pre is None:
        ZA = _zscore_cols_xp(XA, xp)
    else:
        ZA = xp.asarray(ZA_pre, dtype=dtype)
    if ZB_pre is None:
        ZB = _zscore_cols_xp(XB, xp)
    else:
        ZB = xp.asarray(ZB_pre, dtype=dtype)

    L = int(np.floor(max_lag_seconds * float(fps)))
    L = max(0, min(L, T - 1))
    inv_T = dtype(1.0 / float(T))

    # Pad ZA along time with L zeros on each side; sliding view is cheap.
    pad = xp.zeros((L, nA), dtype=ZA.dtype)
    ZA_pad = xp.concatenate([pad, ZA, pad], axis=0)  # (T+2L, nA)

    neg_inf = xp.asarray(-np.inf, dtype=ZA.dtype)
    best_corr = xp.full((nA, nB), float(neg_inf), dtype=ZA.dtype)
    best_lag_idx = xp.zeros((nA, nB), dtype=xp.int32)
    zero_lag_corr = None

    # Determine zero-lag matrix index (j == L)
    for j in range(2 * L + 1):
        Zs = ZA_pad[j:j + T]                  # (T, nA)
        C = (Zs.T @ ZB) * inv_T               # (nA, nB)
        if (j - L) == 0:
            zero_lag_corr = C.copy() if hasattr(C, "copy") else xp.array(C)
        mask = C > best_corr
        best_corr = xp.where(mask, C, best_corr)
        best_lag_idx = xp.where(mask, xp.int32(j), best_lag_idx)

    if zero_lag_corr is None:
        Zs = ZA_pad[L:L + T]
        zero_lag_corr = (Zs.T @ ZB) * inv_T

    best_lag_sec = ((best_lag_idx.astype(ZA.dtype) - dtype(L))
                    / dtype(float(fps)))

    if use_gpu:
        return (cp.asnumpy(best_lag_sec), cp.asnumpy(best_corr),
                cp.asnumpy(zero_lag_corr))
    return (np.asarray(best_lag_sec), np.asarray(best_corr),
            np.asarray(zero_lag_corr))


def single_pair_xcorr_curve(sigA, sigB, fps, max_lag_seconds):
    """Full cross-correlation curve (Pearson r at every lag) for ONE pair.

    Returns
    -------
    lags_sec : (2L+1,) ndarray
    r        : (2L+1,) ndarray  — biased Pearson r at each lag
        Uses the same convention as ``batch_xcorr_clusters``: z-score each
        signal, then ``r[k] = sum(Z_A_shifted_by_k * Z_B) / T`` with
        ``Z_A`` zero-padded by ``L`` samples on each side.
    """
    a = np.asarray(sigA, dtype=np.float32).reshape(-1)
    b = np.asarray(sigB, dtype=np.float32).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"sigA ({a.shape}) and sigB ({b.shape}) must match.")
    T = a.shape[0]

    mu_a = a.mean(); sd_a = a.std()
    mu_b = b.mean(); sd_b = b.std()
    Za = (a - mu_a) / (sd_a if sd_a > 0 else 1.0)
    Zb = (b - mu_b) / (sd_b if sd_b > 0 else 1.0)
    if sd_a == 0:
        Za[:] = 0
    if sd_b == 0:
        Zb[:] = 0

    L = int(np.floor(max_lag_seconds * float(fps)))
    L = max(1, min(L, T - 1))

    pad = np.zeros(L, dtype=Za.dtype)
    Za_pad = np.concatenate([pad, Za, pad])
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(Za_pad, T)  # (2L+1, T)
    r = (windows @ Zb) / float(T)

    lags = np.arange(-L, L + 1, dtype=np.int64)
    lags_sec = lags.astype(np.float32) / float(fps)
    return lags_sec, np.asarray(r, dtype=np.float32)


def _open_dff_memmap(plane0: Path, prefix: str):
    """Open the {prefix}dff memmap. Honours the cell-filter mask when the
    prefix contains ``filtered`` (matches clustering_tab._load_filtered_dff).

    Returns (dff_memmap, T, N).
    """
    plane0 = Path(plane0)
    F = np.load(plane0 / "F.npy", mmap_mode="r")
    N_total, T = F.shape

    if "filtered" in prefix.split("_"):
        mask_path = plane0 / "predicted_cell_mask.npy"
        if mask_path.exists():
            mask = np.load(mask_path).astype(bool)
        else:
            ic = np.load(plane0 / "iscell.npy")
            mask = ((ic[:, 0] > 0) if ic.ndim == 2 else (ic > 0)).astype(bool)
        if mask.size != N_total:
            raise ValueError(
                f"Cell-filter mask length {mask.size} does not match F.npy "
                f"ROI count {N_total}.")
        N_kept = int(mask.sum())
    else:
        N_kept = N_total

    dff_path = plane0 / f"{prefix}dff.memmap.float32"
    if not dff_path.exists():
        raise FileNotFoundError(f"Missing dF/F memmap: {dff_path}")
    dff = np.memmap(dff_path, dtype="float32", mode="r", shape=(T, N_kept))
    return dff, T, N_kept


def _load_clusters_from_dir(cluster_dir: Path,
                            keep_mask: Optional[np.ndarray] = None):
    """Load all C*_rois.npy from a cluster dir, excluding manual_combined.

    The returned indices are always Suite2p ROI indices (positions in
    ``stat.npy`` / ``F.npy``).

    - New layout (presence of ``_indices_are_suite2p`` marker): the .npy
      files already contain Suite2p indices, returned as-is.
    - Legacy layout (no marker): the files contain filtered-list positions
      that need translating via ``keep_mask`` (the boolean mask used when
      the dF/F was built). If ``keep_mask`` is ``None`` and there's no
      marker, indices are returned unchanged (assumed Suite2p; matches
      the unfiltered-prefix case where the two are identical).
    """
    is_suite2p_layout = (cluster_dir / "_indices_are_suite2p").exists()
    roi_files = [
        f for f in sorted(cluster_dir.glob("*_rois.npy"))
        if "manual_combined" not in f.stem.lower()
    ]
    out = {}
    if (not is_suite2p_layout) and keep_mask is not None:
        mask_pos = np.where(np.asarray(keep_mask, dtype=bool))[0].astype(
            np.int64)
    else:
        mask_pos = None
    for f in roi_files:
        idx = np.load(f).astype(np.int64)
        name = f.stem.replace("_rois", "")
        if mask_pos is not None and idx.size:
            # Legacy filtered-position layout; translate to Suite2p indices.
            if int(idx.max()) >= mask_pos.size:
                # Defensive: out-of-range for a filtered-position layout.
                # Treat as already-Suite2p to avoid an index error.
                out[name] = idx
            else:
                out[name] = mask_pos[idx]
        else:
            out[name] = idx
    return out


def _load_keep_mask(plane0: Path, prefix: str) -> Optional[np.ndarray]:
    """Boolean mask of which Suite2p ROIs are kept for a filtered prefix.

    Returns ``None`` when the prefix is not filtered (in which case the
    'kept' set is every Suite2p ROI). Mirrors clustering_tab._load_filter_mask.
    """
    if "filtered" not in prefix.split("_"):
        return None
    plane0 = Path(plane0)
    pred_path = plane0 / "predicted_cell_mask.npy"
    if pred_path.exists():
        return np.load(pred_path).astype(bool)
    iscell_path = plane0 / "iscell.npy"
    if iscell_path.exists():
        ic = np.load(iscell_path)
        return ((ic[:, 0] > 0) if ic.ndim == 2 else (ic > 0)).astype(bool)
    return None


def _write_pair_summary_csv(path, roisA, roisB, best_lag, max_corr,
                            zero_lag_corr=None, extra_cols=None):
    """Write a CAxCB summary CSV from batched (nA, nB) result matrices.

    extra_cols: optional list of (header_name, scalar_or_array) pairs that
    are appended to every row (e.g. event metadata).
    """
    nA, nB = best_lag.shape
    roisA = np.asarray(roisA, dtype=np.int64)
    roisB = np.asarray(roisB, dtype=np.int64)
    roiA_rep = np.repeat(roisA, nB)
    roiB_tile = np.tile(roisB, nA)
    cols = [roiA_rep, roiB_tile,
            best_lag.reshape(-1), max_corr.reshape(-1)]
    header = ["roiA", "roiB", "best_lag_sec", "max_corr"]
    if zero_lag_corr is not None:
        cols.append(zero_lag_corr.reshape(-1))
        header.append("zero_lag_corr")
    if extra_cols is not None:
        for name, val in extra_cols:
            arr = np.broadcast_to(np.asarray(val), (nA * nB,))
            cols.append(np.asarray(arr))
            header.append(name)

    fmts = ["%d", "%d", "%.6g", "%.6g"]
    if zero_lag_corr is not None:
        fmts.append("%.6g")
    if extra_cols is not None:
        for name, val in extra_cols:
            fmts.append("%.6g" if np.issubdtype(
                np.asarray(val).dtype, np.floating) else "%d")

    out = np.column_stack(cols)
    np.savetxt(path, out, delimiter=",", header=",".join(header),
               comments="", fmt=fmts)


def run_cluster_xcorr_full_fast(
    plane0: Path,
    prefix: str,
    fps: float,
    cluster_folder: str,
    *,
    max_lag_seconds: float = 2.0,
    zero_lag: bool = True,
    use_gpu: bool = True,
    output_subdir: str = "cross_correlation_full",
    progress_cb=None,
):
    """Run cluster x cluster cross-correlation on the FULL recording.

    Uses the batched matmul implementation (one matmul per lag, all ROI
    pairs in the cluster pair at once). Each cluster's z-scored trace
    matrix is built once and reused across cluster-pair combinations.

    Output:
      plane0 / {prefix}cluster_results / {cluster_folder} / {output_subdir} /
        CAxCB / CAxCB_summary.csv
    """
    plane0 = Path(plane0)
    base_dir = plane0 / f"{prefix}cluster_results"
    if cluster_folder:
        base_dir = base_dir / cluster_folder

    # Load the cell-filter mask (if any) so we can (a) interpret legacy
    # cluster .npy files as filtered positions and (b) translate Suite2p
    # ROI indices back into filtered positions for slicing dF/F.
    keep_mask = _load_keep_mask(plane0, prefix)
    clusters = _load_clusters_from_dir(base_dir, keep_mask=keep_mask)
    if len(clusters) < 1:
        raise ValueError(f"No cluster *_rois.npy files in {base_dir}")

    dff, n_frames, n_rois = _open_dff_memmap(plane0, prefix)

    # Suite2p index -> position in the filtered dF/F memmap (or -1 if the
    # ROI is no longer in the keep mask).
    if keep_mask is not None:
        s2p_to_filtered = -np.ones(int(keep_mask.size), dtype=np.int64)
        s2p_to_filtered[np.asarray(keep_mask, dtype=bool)] = np.arange(
            int(keep_mask.sum()), dtype=np.int64)
    else:
        s2p_to_filtered = None

    out_root = base_dir / output_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    gpu_ok = bool(use_gpu and cp is not None)
    xp = cp if gpu_ok else np

    cluster_names = sorted(clusters.keys())

    # Pre-build per-cluster (T, n_in_cluster) matrices and z-score once.
    # Track both the Suite2p ROI list (for the CSV) and the filtered
    # positions (for slicing the dF/F memmap).
    cluster_X = {}
    cluster_Z = {}
    cluster_rois_s2p: dict[str, np.ndarray] = {}
    for name in cluster_names:
        s2p_rois = np.asarray(clusters[name], dtype=np.int64)
        if s2p_to_filtered is not None:
            filt_pos = s2p_to_filtered[s2p_rois]
            valid = filt_pos >= 0
            if not valid.all():
                # Some Suite2p ROIs are not in the current keep mask; skip
                # them so the slicing stays in range.
                dropped = int((~valid).sum())
                print(f"[xcorr] {name}: {dropped} ROI(s) not in current "
                      f"filter, skipped.")
                s2p_rois = s2p_rois[valid]
                filt_pos = filt_pos[valid]
        else:
            filt_pos = s2p_rois
        cluster_rois_s2p[name] = s2p_rois
        # Materialise dF/F for these ROI columns as float32 (avoid memmap
        # views going to GPU repeatedly).
        X = np.asarray(dff[:, filt_pos], dtype=np.float32, order="C")
        cluster_X[name] = X
        # Z-score on the target device.
        if gpu_ok:
            X_dev = cp.asarray(X)
            Z_dev = _zscore_cols_xp(X_dev, cp)
            cluster_Z[name] = Z_dev
        else:
            cluster_Z[name] = _zscore_cols_xp(X, np)

    total_pairs_groups = sum(
        1 for i, _ in enumerate(cluster_names) for _ in cluster_names[i:]
    )
    done_groups = 0

    for iA, cA in enumerate(cluster_names):
        for cB in cluster_names[iA:]:
            # Use Suite2p indices for the CSV columns so they match
            # stat.npy / the ROIs sheet.
            roisA = cluster_rois_s2p[cA]
            roisB = cluster_rois_s2p[cB]
            pair_dir = out_root / f"{cA}x{cB}"
            pair_dir.mkdir(exist_ok=True)
            summary_path = pair_dir / f"{cA}x{cB}_summary.csv"

            best_lag, max_c, z0 = batch_xcorr_clusters(
                cluster_X[cA], cluster_X[cB], fps,
                max_lag_seconds=max_lag_seconds,
                use_gpu=gpu_ok,
                ZA_pre=cluster_Z[cA], ZB_pre=cluster_Z[cB],
            )
            _write_pair_summary_csv(
                summary_path, roisA, roisB, best_lag, max_c,
                zero_lag_corr=z0 if zero_lag else None,
            )

            done_groups += 1
            if progress_cb is not None:
                try:
                    progress_cb(done_groups, total_pairs_groups, f"{cA}x{cB}")
                except Exception:
                    pass

    # Free GPU memory if used
    if gpu_ok:
        cluster_Z.clear()
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    return out_root


def run_cluster_xcorr_per_event_fast(
    plane0: Path,
    prefix: str,
    fps: float,
    cluster_folder: str,
    event_windows,                # iterable of (start_s, end_s)
    *,
    max_lag_seconds: float = 2.0,
    zero_lag: bool = True,
    use_gpu: bool = True,
    output_subdir: str = "cross_correlation_per_event",
    min_event_frames: int = 8,
    progress_cb=None,
):
    """Run cluster x cluster cross-correlation per event window using the
    batched core. Each ROI trace is cropped to ``[start_frame, end_frame]``
    so the best-lag and zero-lag values are local to that event.

    Output:
      plane0 / {prefix}cluster_results / {cluster_folder} / {output_subdir} /
        eventNNNN_<start>s_<end>s / CAxCB / CAxCB_summary.csv
    """
    plane0 = Path(plane0)
    base_dir = plane0 / f"{prefix}cluster_results"
    if cluster_folder:
        base_dir = base_dir / cluster_folder

    keep_mask = _load_keep_mask(plane0, prefix)
    clusters = _load_clusters_from_dir(base_dir, keep_mask=keep_mask)
    if len(clusters) < 1:
        raise ValueError(f"No cluster *_rois.npy files in {base_dir}")

    dff, n_frames, n_rois = _open_dff_memmap(plane0, prefix)

    if keep_mask is not None:
        s2p_to_filtered = -np.ones(int(keep_mask.size), dtype=np.int64)
        s2p_to_filtered[np.asarray(keep_mask, dtype=bool)] = np.arange(
            int(keep_mask.sum()), dtype=np.int64)
    else:
        s2p_to_filtered = None

    event_windows = [(float(s), float(e)) for s, e in event_windows]
    if not event_windows:
        raise ValueError("event_windows is empty - run event detection first.")

    out_root = base_dir / output_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    gpu_ok = bool(use_gpu and cp is not None)
    cluster_names = sorted(clusters.keys())

    # Track Suite2p ROI indices (for the CSV) alongside the filtered
    # positions used to slice into the dF/F memmap.
    cluster_rois_s2p: dict[str, np.ndarray] = {}
    cluster_rois_filt: dict[str, np.ndarray] = {}
    for n in cluster_names:
        s2p_rois = np.asarray(clusters[n], dtype=np.int64)
        if s2p_to_filtered is not None:
            filt_pos = s2p_to_filtered[s2p_rois]
            valid = filt_pos >= 0
            if not valid.all():
                dropped = int((~valid).sum())
                print(f"[xcorr] {n}: {dropped} ROI(s) not in current "
                      f"filter, skipped.")
                s2p_rois = s2p_rois[valid]
                filt_pos = filt_pos[valid]
        else:
            filt_pos = s2p_rois
        cluster_rois_s2p[n] = s2p_rois
        cluster_rois_filt[n] = filt_pos

    total = len(event_windows)
    for ev_idx, (s_sec, e_sec) in enumerate(event_windows):
        f0 = max(0, int(round(s_sec * fps)))
        f1 = min(n_frames, int(round(e_sec * fps)))
        if f1 - f0 < min_event_frames:
            continue
        win_len_sec = (f1 - f0) / float(fps)
        eff_max_lag = min(max_lag_seconds, max(0.0, win_len_sec - 1.0 / fps))
        if eff_max_lag <= 0:
            continue

        ev_label = (
            f"event{ev_idx:04d}_"
            f"{s_sec:.2f}s_{e_sec:.2f}s".replace(".", "p")
        )
        ev_dir = out_root / ev_label
        ev_dir.mkdir(exist_ok=True)

        # Crop dF/F for all ROIs once per event and z-score per cluster.
        clip = np.asarray(dff[f0:f1, :], dtype=np.float32, order="C")

        # Pre-z-score each cluster's traces ONCE for this event.
        cluster_X_ev = {}
        cluster_Z_ev = {}
        for n in cluster_names:
            X = np.asarray(clip[:, cluster_rois_filt[n]],
                           dtype=np.float32, order="C")
            cluster_X_ev[n] = X
            if gpu_ok:
                X_dev = cp.asarray(X)
                cluster_Z_ev[n] = _zscore_cols_xp(X_dev, cp)
            else:
                cluster_Z_ev[n] = _zscore_cols_xp(X, np)

        for iA, cA in enumerate(cluster_names):
            for cB in cluster_names[iA:]:
                roisA = cluster_rois_s2p[cA]
                roisB = cluster_rois_s2p[cB]
                pair_dir = ev_dir / f"{cA}x{cB}"
                pair_dir.mkdir(exist_ok=True)
                summary_path = pair_dir / f"{cA}x{cB}_summary.csv"

                best_lag, max_c, z0 = batch_xcorr_clusters(
                    cluster_X_ev[cA], cluster_X_ev[cB], fps,
                    max_lag_seconds=eff_max_lag,
                    use_gpu=gpu_ok,
                    ZA_pre=cluster_Z_ev[cA], ZB_pre=cluster_Z_ev[cB],
                )
                _write_pair_summary_csv(
                    summary_path, roisA, roisB, best_lag, max_c,
                    zero_lag_corr=z0 if zero_lag else None,
                    extra_cols=[
                        ("event_start_sec", float(s_sec)),
                        ("event_end_sec", float(e_sec)),
                        ("n_frames", int(f1 - f0)),
                    ],
                )

        if gpu_ok:
            cluster_Z_ev.clear()

        if progress_cb is not None:
            try:
                progress_cb(ev_idx + 1, total, ev_label)
            except Exception:
                pass

    if gpu_ok:
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    return out_root
