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
``circular_shift_null_pvalues(X_A, X_B, observed_max_corr, ...)``
    Per-pair p-values from a circular-shift null distribution plus
    Benjamini-Hochberg FDR-corrected p-values. Use to flag which
    pair-correlations from ``batch_xcorr_clusters`` are significantly
    above the autocorrelation-preserving null.
``run_cluster_xcorr_full(plane0, ..., progress_cb, abort_cb)``
    Walk every cluster pair on disk and emit a CSV per pair, one row
    per cell pair. Accepts ``n_shuffles`` to enable the shuffle null.
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
#
# bytes.fromhex(...).decode() is a runtime construction. Nuitka's
# ExpressionImportlibImportModuleCall.computeExpression only static-folds
# compile-time-constant arguments into the nofollow machinery
# (see nuitka/nodes/ImportNodes.py); a function-call result is left as a
# dynamic runtime import. This bypasses --nofollow-import-to=cupy so
# Python's normal sys.path lookup resolves the side-loaded cupy/ in the
# dist root. A literal "cupy" or "cu"+"py" would be folded and blocked.
try:
    import importlib as _importlib
    cp = _importlib.import_module(bytes.fromhex("63757079").decode())
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


def _bh_fdr(pvals):
    """Benjamini-Hochberg FDR-corrected p-values (a.k.a. q-values).

    Pure-NumPy reimplementation of
    ``statsmodels.stats.multitest.multipletests(..., method='fdr_bh')``
    so we don't add statsmodels as a dependency for one function.

    Algorithm:
    1. Sort p-values ascending. Let m = number of tests.
    2. Provisional adjusted p_k = sorted_p_k * m / (k + 1).
    3. Walk from largest to smallest enforcing monotone non-decreasing
       order so the adjusted curve never dips below a later value:
       p_adj_k = min(p_adj_k, p_adj_{k+1}).
    4. Clip to 1.0 and unsort back to the original positions.

    Returns an array shaped like ``pvals`` (small values => significant
    after multiple-comparison correction).
    """
    p = np.asarray(pvals, dtype=np.float64)
    shape = p.shape
    flat = p.ravel()
    m = flat.size
    if m == 0:
        return flat.reshape(shape).copy()
    order = np.argsort(flat, kind="stable")
    sorted_p = flat[order]
    ranks = np.arange(1, m + 1, dtype=np.float64)
    adj_sorted = sorted_p * m / ranks
    # Enforce monotone non-decreasing from the top down: a later
    # p-value should not undercut an earlier one. ``minimum.accumulate``
    # on the reversed array gives a running min from the right.
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    np.clip(adj_sorted, 0.0, 1.0, out=adj_sorted)
    out = np.empty_like(flat)
    out[order] = adj_sorted
    return out.reshape(shape)


def circular_shift_null_pvalues(X_A, X_B, observed_max_corr, fps,
                                 max_lag_seconds, *,
                                 use_gpu=True, ZA_pre=None, ZB_pre=None,
                                 dtype=np.float32, n_shuffles=200, seed=0):
    """Per-pair p-values from a circular-shift null + BH-FDR correction.

    Why circular shifts instead of a parametric test
    ------------------------------------------------
    Calcium traces have heavy temporal autocorrelation (a few hundred
    ms of indicator decay glued to slow brain-state drift), which makes
    a Marcenko-Pastur / white-noise null wrong. Circular shifts preserve
    each ROI's autocorrelation exactly while breaking inter-cluster
    synchrony, so the null distribution captures "what max-corr would
    you expect if cluster A and cluster B were temporally unrelated but
    each kept its own autocorrelation structure". See Cheng et al.
    eLife 2023 (doi:10.7554/eLife.81279) for the convention.

    Null draw
    ---------
    Each shuffle picks ONE random offset in ``[L+1, T-L-1]`` (so the
    shifted window never overlaps the unshifted band that the original
    search already covered) and rolls EVERY column of ``X_A`` by that
    shared offset. Within-A correlations are preserved; the A-vs-B
    alignment is jittered. Then the (nA, nB) shuffled max-correlation
    matrix is recomputed via the same matmul-per-lag routine and
    compared element-wise against ``observed_max_corr``.

    P-value convention
    ------------------
    ``p_value = (1 + #{k : shuffled_max_corr_k >= observed}) /
                (n_shuffles + 1)``

    The ``+1`` numerator follows Phipson & Smyth 2010 ("Permutation
    P-values should never be zero") so a pair that beats every
    shuffle still gets a non-zero p, capped at ``1/(n_shuffles+1)``.

    Returns
    -------
    p_value : (nA, nB) float64 ndarray in (0, 1]
    p_value_fdr : (nA, nB) float64 ndarray in [0, 1]
        Benjamini-Hochberg FDR-corrected version. Compare against your
        target FDR (typically 0.05) to flag significant pairs.

    Edge case: when the trace is too short for the offset band to be
    non-empty, both matrices are filled with 1.0 (no significance).
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

    obs = xp.asarray(observed_max_corr, dtype=ZA.dtype)

    # Avoid offsets near 0 / T so the shifted alignment doesn't
    # accidentally fall inside the observed search window.
    lo = max(1, L + 1)
    hi = max(lo + 1, T - L - 1)
    if hi <= lo or n_shuffles <= 0:
        ones = np.ones((int(nA), int(nB)), dtype=np.float64)
        return ones, ones.copy()

    rng = np.random.default_rng(seed)
    n_hits = xp.zeros((nA, nB), dtype=xp.int32)

    for _k in range(int(n_shuffles)):
        offset = int(rng.integers(lo, hi))
        # ``xp.roll`` along axis=0 rotates rows; column structure of
        # ZA stays intact, so each ROI keeps its autocorrelation.
        ZA_shuf = xp.roll(ZA, offset, axis=0)
        pad = xp.zeros((L, nA), dtype=ZA.dtype)
        ZA_pad = xp.concatenate([pad, ZA_shuf, pad], axis=0)
        shuf_max = xp.full((nA, nB), -np.inf, dtype=ZA.dtype)
        for j in range(2 * L + 1):
            Zs = ZA_pad[j:j + T]
            C = (Zs.T @ ZB) * inv_T
            shuf_max = xp.maximum(shuf_max, C)
        n_hits += (shuf_max >= obs).astype(xp.int32)

    if use_gpu:
        n_hits_np = cp.asnumpy(n_hits)
    else:
        n_hits_np = np.asarray(n_hits)
    p_value = (n_hits_np.astype(np.float64) + 1.0) / (float(n_shuffles) + 1.0)
    p_value_fdr = _bh_fdr(p_value)
    return p_value, p_value_fdr


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
    from . import utils as _cutils

    plane0 = Path(plane0)
    F = np.load(plane0 / "F.npy", mmap_mode="r")
    N_total, T = F.shape

    dff_path = plane0 / f"{prefix}dff.memmap.float32"
    if not dff_path.exists():
        raise FileNotFoundError(f"Missing dF/F memmap: {dff_path}")

    if _cutils.is_filtered_prefix(prefix):
        # Size against the mask the memmap was written with (persisted as
        # r0p7_cell_mask_bool.npy), cross-checked against the file size, so
        # a drifted predicted_cell_mask/iscell can't request a wrong-shaped
        # mapping -> [WinError 8] on Windows. See utils.resolve_filtered_mask.
        _mask, N_kept = _cutils.resolve_filtered_mask(
            plane0, N_total, memmap_path=dff_path, T=T)
    else:
        N_kept = N_total

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
    from . import utils as _cutils
    if not _cutils.is_filtered_prefix(prefix):
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


def compute_partial_correlation(Z, ridge: float = 1e-6):
    """Partial-correlation matrix from a z-scored ``(T, N)`` data matrix.

    Partial correlation ``P[i,j]`` is the correlation between variables
    ``i`` and ``j`` *after removing the linear effect of every other
    variable in ``Z``*. Computed via the precision matrix:

    .. math::

        C       &= Z^\\top Z / T \\\\
        C^{-1}  &= \\text{pinv}(C + \\rho I) \\\\
        P[i,j]  &= -C^{-1}[i,j] / \\sqrt{C^{-1}[i,i] \\, C^{-1}[j,j]}

    The ridge term ``ρ`` (default ``1e-6``) keeps the pseudo-inverse
    well-defined when ``C`` is near-singular (which happens whenever
    columns of ``Z`` are nearly linearly dependent). Well-conditioned
    ``C`` is essentially unchanged.

    **Why this matters for calcium imaging.** Pearson correlation cannot
    distinguish "A and B drive each other" from "A and B are both driven
    by some other ROI C in the dataset" — slow drifts and shared
    brain-state inflate pairwise correlations across the entire
    population. Partial correlation conditions on every other ROI,
    isolating the direct pairwise structure. See Sutera et al. arXiv:
    `1406.7865 <https://arxiv.org/abs/1406.7865>`_ for the connectome-
    inference application; Stevenson arXiv:`1708.01888
    <https://arxiv.org/abs/1708.01888>`_ for the broader connectivity-
    inference review.

    **Conditioning requirement.** Partial correlation needs ``N << T``;
    when ``N`` approaches ``T`` (or exceeds it), ``C`` becomes rank-
    deficient and the precision matrix is essentially noise. Callers
    should refuse to compute ``P`` when ``N >= 0.5 * T`` (see the Tab 7
    wiring for the user-facing guard).

    Parameters
    ----------
    Z : array_like, shape (T, N)
        Already z-scored (per column) data matrix. Caller is responsible
        for the z-scoring -- this function does NOT re-standardise.
    ridge : float
        Ridge regularisation added to ``C`` before inversion. Default
        ``1e-6`` works for typical dF/F z-scored traces; increase if
        ``np.linalg.pinv`` warns about ill-conditioning.

    Returns
    -------
    P : ndarray, shape (N, N), float64
        Partial-correlation matrix. Diagonal is exactly 1. Off-diagonal
        entries are in approximately ``[-1, 1]`` (small numerical
        excursion possible with aggressive ridge).
    """
    Z = np.asarray(Z, dtype=np.float64)  # float64 for pinv stability
    T, N = Z.shape
    if T <= 1 or N <= 1:
        return np.eye(N, dtype=np.float64)
    # Biased covariance: matches the biased-Pearson convention the
    # batch_xcorr_clusters routine already uses (factor of T vs T-1 is
    # constant across pairs, so it doesn't affect ranking / argmax).
    C = (Z.T @ Z) / float(T)
    # Ridge-regularised pseudo-inverse: adds a tiny diagonal so even a
    # rank-deficient C has a well-defined inverse.
    Cinv = np.linalg.pinv(C + ridge * np.eye(N, dtype=C.dtype))
    # Off-diagonal partial corr from the precision matrix; standard
    # graphical-model derivation. ``np.maximum`` clamps tiny negative
    # diagonal entries (which would otherwise produce NaN under sqrt).
    d = np.sqrt(np.maximum(np.diag(Cinv), 1e-12))
    P = -Cinv / np.outer(d, d)
    np.fill_diagonal(P, 1.0)
    return P


def _write_pair_summary_csv(path, roisA, roisB, best_lag, max_corr,
                            zero_lag_corr=None, p_value=None,
                            p_value_fdr=None, partial_corr=None,
                            extra_cols=None):
    """Write a CAxCB summary CSV from batched (nA, nB) result matrices.

    p_value / p_value_fdr (optional): per-pair p-values from
    ``circular_shift_null_pvalues`` and the BH-FDR-corrected version.
    Written as ``p_value`` / ``p_value_fdr`` columns when supplied.

    partial_corr (optional): per-pair partial correlation, sliced from
    the population-wide partial-correlation matrix computed by
    ``compute_partial_correlation`` on the full keep-mask dF/F. Written
    as ``partial_corr_zero_lag`` when supplied.

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
    if p_value is not None:
        cols.append(np.asarray(p_value).reshape(-1))
        header.append("p_value")
    if p_value_fdr is not None:
        cols.append(np.asarray(p_value_fdr).reshape(-1))
        header.append("p_value_fdr")
    if partial_corr is not None:
        cols.append(np.asarray(partial_corr).reshape(-1))
        header.append("partial_corr_zero_lag")
    if extra_cols is not None:
        for name, val in extra_cols:
            arr = np.broadcast_to(np.asarray(val), (nA * nB,))
            cols.append(np.asarray(arr))
            header.append(name)

    fmts = ["%d", "%d", "%.6g", "%.6g"]
    if zero_lag_corr is not None:
        fmts.append("%.6g")
    if p_value is not None:
        fmts.append("%.6g")
    if p_value_fdr is not None:
        fmts.append("%.6g")
    if partial_corr is not None:
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
    n_shuffles: int = 500,
    shuffle_seed: int = 0,
    compute_partial: bool = False,
    progress_cb=None,
):
    """Run cluster x cluster cross-correlation on the FULL recording.

    Uses the batched matmul implementation (one matmul per lag, all ROI
    pairs in the cluster pair at once). Each cluster's z-scored trace
    matrix is built once and reused across cluster-pair combinations.

    When ``n_shuffles > 0`` (default 500) the per-pair circular-shift null is
    also computed and ``p_value`` / ``p_value_fdr`` columns are added to the
    summary CSV. See ``circular_shift_null_pvalues`` for the null semantics.
    Cost scales linearly with ``n_shuffles`` (each draw is one additional
    batched-matmul sweep), so 100-500 is a reasonable range; 500 gives
    sub-1% p-value resolution. Set to 0 to disable. The null distribution
    is the field-standard fix for the autocorrelation-inflated false-positive
    rate that parametric tests (Marcenko-Pastur) suffer on calcium imaging
    data -- see Cheng et al. eLife 2023 (doi:10.7554/eLife.81279).

    When ``compute_partial`` is True, the population-wide partial-correlation
    matrix is computed once across every ROI in the union of all clusters
    (conditioning on every other ROI), then sliced for each cluster pair and
    added to the summary CSV as ``partial_corr_zero_lag``. Partial correlation
    isolates direct pairwise structure from shared-driver inflation -- see
    Sutera et al. arXiv:1406.7865. Requires N_kept < 0.5 * T (otherwise the
    precision matrix is ill-conditioned); the call falls back to "skip and
    warn" when that condition fails.

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

    # Optional one-shot partial-correlation precompute. Partial corr
    # conditions on every other ROI, so it has to be computed over the
    # full union of cluster ROIs (not pair-by-pair) -- otherwise we'd
    # be conditioning only on the two clusters' cells and missing the
    # rest of the population. Cost: O((N_union)^3) pseudo-inverse, done
    # once; the per-cluster-pair extraction is a (nA, nB) numpy slice.
    partial_full: "np.ndarray | None" = None
    partial_index: "dict[int, int]" = {}
    if compute_partial:
        # Union of every cluster's filtered positions (the index space
        # the dF/F memmap actually has columns for). Sort + unique gives
        # us a deterministic ordering for the slice maps below.
        all_filt_pos: list[int] = []
        for name in cluster_names:
            s2p_rois = cluster_rois_s2p[name]
            if s2p_to_filtered is not None:
                fp = s2p_to_filtered[s2p_rois]
                all_filt_pos.extend(int(p) for p in fp if p >= 0)
            else:
                all_filt_pos.extend(int(p) for p in s2p_rois)
        union_pos = np.array(sorted(set(all_filt_pos)), dtype=np.int64)
        n_union = int(union_pos.size)
        # Conditioning requirement: N << T. The audit's cutoff is N <
        # 0.5*T; above that the precision matrix is essentially noise.
        if n_union >= 2 and n_union < 0.5 * int(n_frames):
            X_union = np.asarray(dff[:, union_pos], dtype=np.float32,
                                 order="C")
            Z_union = _zscore_cols_xp(X_union, np)
            partial_full = compute_partial_correlation(Z_union)
            partial_index = {int(p): int(i)
                             for i, p in enumerate(union_pos)}
            print(f"[xcorr] partial-correlation precomputed: "
                  f"N_union={n_union}, T={int(n_frames)} -> "
                  f"P shape {partial_full.shape}")
        else:
            print(f"[xcorr] partial-correlation skipped: "
                  f"N_union={n_union} not < 0.5*T={int(0.5 * n_frames)}; "
                  f"precision matrix would be ill-conditioned.")

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
            p_val = p_val_fdr = None
            if int(n_shuffles) > 0:
                # Distinct seed per cluster pair so the shuffles aren't
                # identical across pairs but still reproducible.
                pair_seed = int(shuffle_seed) + done_groups
                p_val, p_val_fdr = circular_shift_null_pvalues(
                    cluster_X[cA], cluster_X[cB], max_c,
                    fps, max_lag_seconds,
                    use_gpu=gpu_ok,
                    ZA_pre=cluster_Z[cA], ZB_pre=cluster_Z[cB],
                    n_shuffles=int(n_shuffles), seed=pair_seed,
                )

            # Slice the population-wide partial-correlation matrix down
            # to this cluster pair. Map Suite2p ROI ids -> filtered
            # positions -> indices within the union; ROIs not in the
            # keep mask get NaN so the CSV reader knows the value is
            # missing rather than zero.
            partial_block = None
            if partial_full is not None:
                if s2p_to_filtered is not None:
                    fpA = s2p_to_filtered[roisA]
                    fpB = s2p_to_filtered[roisB]
                else:
                    fpA = np.asarray(roisA, dtype=np.int64)
                    fpB = np.asarray(roisB, dtype=np.int64)
                idxA = np.array(
                    [partial_index.get(int(p), -1) for p in fpA],
                    dtype=np.int64)
                idxB = np.array(
                    [partial_index.get(int(p), -1) for p in fpB],
                    dtype=np.int64)
                block = np.full((idxA.size, idxB.size), np.nan,
                                dtype=np.float64)
                validA = idxA >= 0
                validB = idxB >= 0
                if validA.any() and validB.any():
                    block_valid = partial_full[
                        np.ix_(idxA[validA], idxB[validB])]
                    # Scatter back into the (nA, nB) layout.
                    rowmap = np.flatnonzero(validA)
                    colmap = np.flatnonzero(validB)
                    block[np.ix_(rowmap, colmap)] = block_valid
                partial_block = block

            _write_pair_summary_csv(
                summary_path, roisA, roisB, best_lag, max_c,
                zero_lag_corr=z0 if zero_lag else None,
                p_value=p_val, p_value_fdr=p_val_fdr,
                partial_corr=partial_block,
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
    n_shuffles: int = 0,
    shuffle_seed: int = 0,
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


class XcorrAborted(Exception):
    """Raised inside the offload child when the user requests an abort."""


def run_cluster_xcorr_offload(mode: str, plane0_str: str, *,
                              event_windows=None, abort_evt=None,
                              progress_q=None, **params) -> dict:
    """Picklable :mod:`calliope.core.offload` target for Tab 7.

    Runs the full or per-event cross-correlation in a child process so
    the GUI stays responsive (and so a GPU run's CuPy context is torn
    down with the process). ``mode`` is ``"full"`` or ``"per_event"``.

    Cooperative abort: ``abort_evt`` is a spawn ``Event`` (see
    :func:`calliope.core.offload.make_abort_event`). The wrapped
    ``progress_cb`` checks it between batches/events and raises
    :class:`XcorrAborted`, so the run stops cleanly *between* the
    incremental per-pair writes -- the same boundary the old in-thread
    ``RunAborted`` used. Returns a picklable dict the tab dispatches on:
    ``{"mode", "aborted", "out"}``.
    """
    from .offload import make_progress_cb
    base_cb = make_progress_cb(progress_q)

    def cb(done, total, label):
        if abort_evt is not None and abort_evt.is_set():
            raise XcorrAborted()
        if base_cb is not None:
            base_cb(done, total, label)

    plane0 = Path(plane0_str)
    try:
        if mode == "full":
            out = run_cluster_xcorr_full_fast(plane0, progress_cb=cb, **params)
        else:
            out = run_cluster_xcorr_per_event_fast(
                plane0, event_windows=event_windows, progress_cb=cb, **params)
        return {"mode": mode, "aborted": False, "out": str(out)}
    except XcorrAborted:
        return {"mode": mode, "aborted": True, "out": None}


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

    save_path = Path(save_path)
    fig.savefig(save_path, dpi=150)
    fig.savefig(save_path.with_suffix(".svg"))
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
        ``use_gpu`` (default True),
        ``n_shuffles`` (default 500 -- circular-shift null + BH-FDR
        for per-pair p-values; set 0 to disable; per-event path
        does not use this),
        ``compute_partial`` (default False -- population-wide
        partial-correlation matrix sliced into a
        ``partial_corr_zero_lag`` CSV column; skipped if
        N_kept >= 0.5 * T; per-event path does not use this).
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
    n_shuffles = max(0, int(params.get("n_shuffles", 500)))
    compute_partial = bool(params.get("compute_partial", False))

    full_subdir = "cross_correlation_full"
    run_cluster_xcorr_full_fast(
        plane0, prefix=prefix, fps=fps, cluster_folder=cluster_folder,
        max_lag_seconds=max_lag, zero_lag=zero_lag, use_gpu=use_gpu,
        output_subdir=full_subdir, n_shuffles=n_shuffles,
        compute_partial=compute_partial,
        progress_cb=progress_cb,
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
