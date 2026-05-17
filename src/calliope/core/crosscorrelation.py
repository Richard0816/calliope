"""Cluster x cluster cross-correlation (Tab 7 backend + headless
batch entry point).

What "cross-correlation" means here
-----------------------------------
Given two cell traces A(t) and B(t), the cross-correlation at lag k
measures how strongly A leads / trails B by k samples. Conceptually,
shift A by k frames, compute Pearson r against B, repeat for every
``k`` in [-L, +L]. The result is a 1-D function of lag whose:

* peak height = max correlation (how synchronous A and B are);
* peak position = best lag in seconds (positive => A leads B);
* value at lag 0 = "instantaneous" co-firing.

For *cluster x cluster* analysis we compute that for every cell in
cluster A vs every cell in cluster B. A naive implementation is
``nA x nB`` cross-correlation calls -- too slow for hundreds of cells
per cluster. This module's ``batch_xcorr_clusters`` does it all in
**one matmul per lag**: shifting A by k frames simply means slicing
the (T+2L, nA) zero-padded matrix, and one ``shifted_A.T @ B`` covers
every (i, j) pair simultaneously. The whole thing runs on GPU when
CuPy is installed, NumPy otherwise.

Public surface
--------------
``batch_xcorr_clusters(X_A, X_B, fps, max_lag_seconds, ...)``
    The matmul-per-lag routine. Returns three (nA, nB) matrices:
    ``best_lag_sec``, ``max_corr``, ``zero_lag_corr``.
``run_cluster_xcorr_full(plane0, ..., progress_cb, abort_cb)``
    Walk every cluster pair on disk and emit a CSV per pair, one row
    per cell pair.
``run_cluster_xcorr_per_event_fast(plane0, event_windows, ...)``
    Same but cropped to each detected event window from Tab 5.
``zscore_pair_traces`` and the ``xcorr_*`` helpers are used by the
single-pair preview in Tab 7.

What you'll see if you've never read CuPy code
----------------------------------------------
The pattern ``xp = cp if use_gpu else np`` lets the same algorithm
run on either GPU or CPU: ``xp`` is "whichever array module we
chose", and methods like ``xp.zeros(...)`` / ``xp.where(...)`` /
``xp.asarray(...)`` exist on both libraries with the same signatures.
Roughly the equivalent of swapping out ``base`` for ``data.table`` in
R but keeping the same call shapes.
"""

# Type-hint helper. ``Optional[X]`` == ``X or None``.
from typing import Optional

import numpy as np
from pathlib import Path

# CuPy is the GPU-accelerated drop-in for NumPy. We import it
# defensively: not every machine has CUDA, and we'd rather print a
# warning and fall back to CPU than crash on import.
try:
    import cupy as cp
except ImportError:
    cp = None
    print("[warn] CuPy not available; GPU cross-correlation will fall back to CPU.")


def _release_gpu_pools() -> None:
    """Return CuPy's device + pinned-host memory pools to the driver.

    CuPy holds onto freed allocations in its pool by default, which
    means dedicated VRAM stays high even after the run finishes and
    its locals go out of scope. Returning the pool blocks to the
    driver is what makes Task Manager / nvidia-smi reflect the drop.

    Order matters: GC first (drops any Python refs the caller hasn't
    explicitly cleared), then full device sync (catches all streams,
    not just the null one), then pool free, then a final sync so the
    diagnostic print sees the post-release state. ``free_all_blocks``
    is run twice because CuPy occasionally needs a second pass to
    release segments that were "in use" at the moment of the first
    call but became free during the sync between calls.

    Logs ``used_bytes`` and ``total_bytes`` from the default pool so
    the operator can verify the release actually happened. If
    ``used_bytes`` stays high after this returns, something is
    still holding a CuPy reference -- look upstream for an
    unreleased ``cluster_*``, dict entry, or cached scratch tensor.

    No-op if CuPy isn't installed; never raises.
    """
    if cp is None:
        return
    import gc
    try:
        gc.collect()
    except Exception:
        pass
    try:
        # Sync the WHOLE device (every stream), not just the null
        # stream. CuPy kernels can live on non-default streams in
        # newer versions and these would otherwise leave allocations
        # in flight when free_all_blocks runs.
        cp.cuda.Device().synchronize()
    except Exception:
        pass
    mempool = None
    pinned = None
    try:
        mempool = cp.get_default_memory_pool()
    except Exception:
        pass
    try:
        pinned = cp.get_default_pinned_memory_pool()
    except Exception:
        pass
    used_before = total_before = -1
    if mempool is not None:
        try:
            used_before = int(mempool.used_bytes())
            total_before = int(mempool.total_bytes())
        except Exception:
            pass
    if mempool is not None:
        for _ in range(2):
            try:
                mempool.free_all_blocks()
            except Exception:
                pass
            try:
                cp.cuda.Device().synchronize()
            except Exception:
                pass
    if pinned is not None:
        try:
            pinned.free_all_blocks()
        except Exception:
            pass
    used_after = total_after = -1
    if mempool is not None:
        try:
            used_after = int(mempool.used_bytes())
            total_after = int(mempool.total_bytes())
        except Exception:
            pass
    if used_before >= 0 and used_after >= 0:
        print(
            f"[xcorr-cleanup] CuPy pool: "
            f"used {used_before / 1e6:.1f}->{used_after / 1e6:.1f} MB, "
            f"total {total_before / 1e6:.1f}->{total_after / 1e6:.1f} MB"
        )


def _zscore_cols_xp(X, xp, eps=1e-12):
    """Z-score each column independently using either NumPy or CuPy.

    Z-scoring (subtract mean, divide by std) is the standard
    preprocessing step before computing Pearson correlations -- once
    you've z-scored both columns, the correlation reduces to a
    simple inner product / T.

    The ``xp`` argument is either ``numpy`` or ``cupy``; both
    libraries support the same calls (``mean``, ``std``, ``where``).
    Letting the caller pass it in lets the same code run on either
    backend without an ``if use_gpu:`` ladder.

    A column with std=0 (constant trace) would normally produce NaN
    on division -- we guard against that with ``sd_safe`` and then
    zero out the offending columns explicitly.
    """
    # ``axis=0`` -> reduce along rows -> one value per column.
    # ``keepdims=True`` keeps the result shape ``(1, N)`` instead of
    # ``(N,)`` so it broadcasts cleanly against ``X`` of shape
    # ``(T, N)``.
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    # ``xp.where(condition, a, b)`` is a vectorised if/else: pick a
    # where condition is True, b otherwise. Replace zero stds with
    # 1.0 to dodge the division-by-zero; we'll then explicitly zero
    # the bad columns below.
    sd_safe = xp.where(sd == 0, xp.asarray(1.0, dtype=X.dtype), sd)
    Z = (X - mu) / sd_safe
    # Mask zero-std columns to zero so they don't produce spurious
    # correlations.
    bad = (sd == 0).reshape(-1)
    if bool(bad.any()):
        Z[:, bad] = 0
    return Z


def batch_xcorr_clusters(X_A, X_B, fps, max_lag_seconds, *,
                         use_gpu=True, ZA_pre=None, ZB_pre=None,
                         dtype=np.float32):
    """Best-lag + zero-lag cross-correlation for ALL pairs (i, j) in
    ``X_A[:, i]`` x ``X_B[:, j]``.

    Inputs
    ------
    X_A : (T, nA) ndarray  -- ROI traces in cluster A, columns are cells
    X_B : (T, nB) ndarray  -- same for cluster B
    fps : float            -- so we can convert lags to seconds at the end
    max_lag_seconds : float -- search window radius in seconds

    Returns
    -------
    best_lag_sec : (nA, nB) ndarray
        Lag in seconds at which each pair's Pearson r peaks.
        Positive => A leads B.
    max_corr     : (nA, nB) ndarray  -- (biased) Pearson r at best lag
    zero_lag_corr: (nA, nB) ndarray  -- Pearson r at lag 0

    Algorithm: one matmul per lag, all pairs at once
    ------------------------------------------------
    The naive approach -- a separate cross-correlation call per
    (i, j) pair -- is too slow for hundreds of cells per cluster.
    Instead we exploit the fact that at lag k, the Pearson r between
    z-scored ``A_i`` and ``B_j`` equals ``(1/T) * sum(Z_A_i_shifted
    * Z_B_j)``. Stacking all of A as columns of a matrix turns the
    per-pair sum into one big ``shifted_A.T @ Z_B`` matmul that
    yields the entire (nA, nB) correlation matrix at lag k in one
    shot. We do that for every lag in ``[-L, +L]`` and track the
    running per-pair argmax.

    Cost is dominated by ``(2L+1)`` GEMMs (matrix multiplies). Modern
    BLAS implementations -- especially CuPy on a GPU -- chew through
    these very fast, so a 100-lag search over 200 x 300 cells in a
    cluster pair takes a couple of seconds.

    ``ZA_pre`` / ``ZB_pre`` let the caller pre-compute the z-scored
    matrices and reuse them across many cluster pairs (the full-
    recording runner does this).
    """
    # Decide whether to actually use the GPU. We need both the user's
    # opt-in AND a working CuPy install. ``cp is None`` after the
    # try-import at the top of the file marks an unavailable GPU.
    use_gpu = bool(use_gpu and cp is not None)
    # ``xp`` is the array library we'll use throughout. NumPy/CuPy
    # share their public API so the same code runs on either.
    xp = cp if use_gpu else np

    # ``xp.asarray(..., dtype=...)`` is a no-op if the input is
    # already on the right device + dtype, otherwise copies. Saves
    # us from manually shipping data to/from the GPU.
    XA = xp.asarray(X_A, dtype=dtype)
    XB = xp.asarray(X_B, dtype=dtype)
    T, nA = XA.shape
    nB = XB.shape[1]

    # Z-score each cluster's traces unless the caller already did.
    if ZA_pre is None:
        ZA = _zscore_cols_xp(XA, xp)
    else:
        ZA = xp.asarray(ZA_pre, dtype=dtype)
    if ZB_pre is None:
        ZB = _zscore_cols_xp(XB, xp)
    else:
        ZB = xp.asarray(ZB_pre, dtype=dtype)

    # Convert max-lag from seconds to integer frames. ``np.floor``
    # rounds toward zero. Clamp to [0, T-1] so we don't request
    # negative or out-of-array lags.
    L = int(np.floor(max_lag_seconds * float(fps)))
    L = max(0, min(L, T - 1))
    # Precompute 1/T as a same-dtype scalar; multiplying is faster
    # than dividing on every iteration.
    inv_T = dtype(1.0 / float(T))

    # Pad ZA along time with L zeros on each side. ``xp.concatenate``
    # joins arrays along ``axis=0`` (rows); the result has shape
    # ``(T + 2L, nA)``. We then take T-row slices of this padded
    # matrix to "shift" ZA without copying any data.
    pad = xp.zeros((L, nA), dtype=ZA.dtype)
    ZA_pad = xp.concatenate([pad, ZA, pad], axis=0)  # (T+2L, nA)

    # Running per-pair best so far. ``-inf`` means "anything is
    # better than this" so the first iteration always wins.
    neg_inf = xp.asarray(-np.inf, dtype=ZA.dtype)
    best_corr = xp.full((nA, nB), float(neg_inf), dtype=ZA.dtype)
    best_lag_idx = xp.zeros((nA, nB), dtype=xp.int32)
    zero_lag_corr = None

    # Sweep every lag j in [0, 2L]. j corresponds to lag (j - L) in
    # frames, so j = L is lag 0.
    for j in range(2 * L + 1):
        # ``ZA_pad[j:j+T]`` is a view (no copy) of T rows starting
        # at offset j -- effectively ZA shifted left by (j - L)
        # frames.
        Zs = ZA_pad[j:j + T]                  # (T, nA)
        # ``Zs.T @ ZB`` is matrix multiplication ``(nA, T) @ (T,
        # nB) = (nA, nB)``. Each entry is the inner product of a
        # column of Zs against a column of ZB -- the unscaled
        # Pearson r contribution.
        C = (Zs.T @ ZB) * inv_T               # (nA, nB)
        # Stash the lag-0 matrix the moment we hit it.
        if (j - L) == 0:
            zero_lag_corr = C.copy() if hasattr(C, "copy") else xp.array(C)
        # Where this lag improved on the running best, update the
        # running best and the recorded lag index. Vectorised --
        # touches every (i, j) pair at once.
        mask = C > best_corr
        best_corr = xp.where(mask, C, best_corr)
        best_lag_idx = xp.where(mask, xp.int32(j), best_lag_idx)

    # Defensive: if L was clamped to 0 we never iterated, so compute
    # zero_lag_corr explicitly.
    if zero_lag_corr is None:
        Zs = ZA_pad[L:L + T]
        zero_lag_corr = (Zs.T @ ZB) * inv_T

    # Convert the integer lag indices back to seconds.
    best_lag_sec = ((best_lag_idx.astype(ZA.dtype) - dtype(L))
                    / dtype(float(fps)))

    # If we ran on GPU, copy the result matrices back to host RAM
    # (NumPy) before returning -- the caller doesn't want to deal
    # with CuPy arrays.
    if use_gpu:
        return (cp.asnumpy(best_lag_sec), cp.asnumpy(best_corr),
                cp.asnumpy(zero_lag_corr))
    return (np.asarray(best_lag_sec), np.asarray(best_corr),
            np.asarray(zero_lag_corr))


def single_pair_xcorr_curve(sigA, sigB, fps, max_lag_seconds):
    """Full cross-correlation curve (Pearson r at every lag) for ONE pair.

    Used by Tab 7's "single-pair preview" so the user can sanity-
    check what the cross-correlation looks like for a particular
    cell pair before committing to a full cluster x cluster sweep.

    Returns
    -------
    lags_sec : (2L+1,) ndarray
    r        : (2L+1,) ndarray  -- biased Pearson r at each lag

    Algorithm note
    --------------
    Same convention as ``batch_xcorr_clusters``: z-score each signal,
    then ``r[k] = sum(Z_A_shifted_by_k * Z_B) / T`` with ``Z_A``
    zero-padded by ``L`` samples on each side. The implementation
    uses ``numpy.lib.stride_tricks.sliding_window_view`` -- a zero-
    copy way to materialise every length-T shift of ``Za_pad`` as
    rows of a ``(2L+1, T)`` matrix -- so the whole curve falls out
    of one matmul.
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

    # Free GPU memory if used. Drop references first so the pool free
    # actually reclaims those blocks rather than leaving them as live
    # allocations from CuPy's POV.
    if gpu_ok:
        cluster_X.clear()
        cluster_Z.clear()
        _release_gpu_pools()

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
            cluster_X_ev.clear()
            cluster_Z_ev.clear()

        if progress_cb is not None:
            try:
                progress_cb(ev_idx + 1, total, ev_label)
            except Exception:
                pass

    if gpu_ok:
        _release_gpu_pools()

    return out_root


# ---------------------------------------------------------------------------
# Headless orchestrator -- the entry point Tab 0's batch runner calls.
# ---------------------------------------------------------------------------
#
# The matmul-per-lag routines above are already callable headlessly --
# ``run_cluster_xcorr_full_fast`` / ``run_cluster_xcorr_per_event_fast``
# write per-pair summary CSVs. What was missing for batch use is a thin
# wrapper that drives both runs end-to-end (full + per-event when event
# windows are available), loads the resulting CSVs, and renders the
# same violin plots Tab 7 shows in the GUI.


def _save_violin_pair(pair_data, *, save_path: Path, title_suffix: str):
    """Render the same two-violin layout the Tab 7 GUI shows
    (zero-lag correlation + best lag) and save to ``save_path``.
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from ..tabs.crosscorrelation.tab import (
        _get_metric_arrays, _plot_violin, _plot_lag_violin,
    )

    corr_labels, corr_arrays = _get_metric_arrays(pair_data, "zero_lag_corr")
    lag_labels, lag_arrays = _get_metric_arrays(pair_data, "best_lag_sec")

    fig, (ax_corr, ax_lag) = plt.subplots(2, 1, figsize=(10, 8),
                                          tight_layout=True)
    if corr_arrays:
        _plot_violin(
            ax_corr, corr_labels, corr_arrays,
            "Zero-lag correlation",
            f"Per-pair zero-lag correlation  ({title_suffix})",
            show_sig=False,
        )
    else:
        ax_corr.set_axis_off()
        ax_corr.text(0.5, 0.5, "No pair data", ha="center", va="center",
                     transform=ax_corr.transAxes)

    if lag_arrays:
        _plot_lag_violin(
            ax_lag, lag_labels, lag_arrays,
            "Lag at peak correlation (s)",
            f"Per-pair best-lag distribution  ({title_suffix})",
        )
    else:
        ax_lag.set_axis_off()
        ax_lag.text(0.5, 0.5, "No pair data", ha="center", va="center",
                    transform=ax_lag.transAxes)

    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def run_crosscorrelation(
    plane0: Path,
    params: dict,
    *,
    event_windows=None,
    figures_dir: Optional[Path] = None,
    progress_cb=None,
) -> dict:
    """Drive ``run_cluster_xcorr_full_fast`` and (optionally)
    ``run_cluster_xcorr_per_event_fast`` and save violin figures.

    Parameters
    ----------
    plane0 : Path
        Suite2p plane folder.
    params : dict
        ``prefix`` (default ``"r0p7_filtered_"``),
        ``cluster_folder`` (default ``""`` -- ROI files live directly
        under ``<prefix>cluster_results/``; pass a name to use an
        extra subfolder under there),
        ``fps`` (must be supplied; usually carried over from
        upstream stages),
        ``max_lag_seconds`` (default 2.0),
        ``zero_lag`` (default True),
        ``use_gpu`` (default True).
    event_windows : iterable of (start_s, end_s) or None
        If supplied, also run the per-event xcorr.
    figures_dir : Path or None
        If set, save ``full.png`` and ``per_event.png`` violin plots.

    Returns
    -------
    dict
        ``{full_outdir, per_event_outdir}`` (paths or None).
    """
    plane0 = Path(plane0)
    prefix = str(params.get("prefix", "r0p7_filtered_"))
    cluster_folder = str(params.get("cluster_folder", ""))
    fps = float(params.get("fps", 0.0) or 0.0)
    if fps <= 0:
        raise ValueError(
            "run_crosscorrelation requires positive fps in params")
    max_lag = float(params.get("max_lag_seconds", 2.0))
    zero_lag = bool(params.get("zero_lag", True))
    use_gpu = bool(params.get("use_gpu", True))

    full_subdir = "cross_correlation_full"
    run_cluster_xcorr_full_fast(
        plane0, prefix=prefix, fps=fps, cluster_folder=cluster_folder,
        max_lag_seconds=max_lag, zero_lag=zero_lag, use_gpu=use_gpu,
        output_subdir=full_subdir, progress_cb=progress_cb,
    )
    full_outdir = (plane0 / f"{prefix}cluster_results" / cluster_folder
                   / full_subdir)

    per_event_outdir: Optional[Path] = None
    if event_windows is not None and len(event_windows) > 0:
        per_event_subdir = "cross_correlation_per_event"
        run_cluster_xcorr_per_event_fast(
            plane0, prefix=prefix, fps=fps,
            cluster_folder=cluster_folder, event_windows=event_windows,
            max_lag_seconds=max_lag, zero_lag=zero_lag, use_gpu=use_gpu,
            output_subdir=per_event_subdir, progress_cb=progress_cb,
        )
        per_event_outdir = (plane0 / f"{prefix}cluster_results"
                            / cluster_folder / per_event_subdir)

    if figures_dir is not None:
        from ..tabs.crosscorrelation.tab import _load_pair_data
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        if full_outdir.exists():
            pair_data = _load_pair_data(full_outdir)
            _save_violin_pair(
                pair_data, save_path=figures_dir / "full.png",
                title_suffix="full recording",
            )
        if per_event_outdir is not None and per_event_outdir.exists():
            pair_data = _load_pair_data(per_event_outdir)
            _save_violin_pair(
                pair_data, save_path=figures_dir / "per_event.png",
                title_suffix="per event",
            )

    return {
        "full_outdir": str(full_outdir) if full_outdir.exists() else None,
        "per_event_outdir": (str(per_event_outdir)
                             if per_event_outdir is not None
                             and per_event_outdir.exists() else None),
    }

    return out_root
