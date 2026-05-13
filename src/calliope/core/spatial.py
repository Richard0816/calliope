"""calliope.core.spatial - shared helpers for per-event activation-
order ("coactivation") spatial maps.

What this file provides
-----------------------
``CYAN_TO_RED``
    A custom matplotlib colormap that runs cyan -> blue -> red. We
    use it whenever we encode a temporal ordering (early vs. late) on
    a spatial map: ROIs that fired earliest end up cyan, those that
    fired latest end up red, with smooth blue tones in between. The
    ``set_bad("#808080")`` call paints any NaN pixel grey, which is
    how we mark "this pixel doesn't belong to any ROI".
``order_map_for_event(first_time_col, active_mask_col)``
    For one event, rank the participating ROIs by first-onset time.
    Returns a length-``N_rois`` array where rank 1..K marks active
    ROIs in firing order and NaN marks inactive ones.
``paint_order_map(order_rank, stat_filtered, Ly, Lx)``
    Take those ranks, normalise to [0, 1], and paint them onto the
    FOV using ``utils.paint_spatial``. Pixels outside any ROI come
    out as NaN so the cmap renders them grey.
``event_frame_centroids(first_time_col, active_mask_col, stat, fps)``
    Group the ROIs that fired in one event by their first-onset
    *frame* (round ``first_time * fps``) and return one summary dict
    per active frame: the ROIs in that group, their centroid, and
    the 2-D RMS spread.

Used by Tab 8 - Spatial propagation. Ported from the standalone
reference at ``Calcium_imaging_suite2p/spatial_heatmap_updated.py``.
"""

# Type-hint forward-reference shim (see pipeline_gui.py).
from __future__ import annotations

# matplotlib's top module gives us the colormap factory below.
import matplotlib as mpl
import numpy as np

# Path / Optional needed by the headless event-figure renderer
# (``render_spatial_event_figures``) at the bottom of this file.
from pathlib import Path
from typing import Optional

# ``paint_spatial`` does the heavy lifting: it knows how to splat a
# per-ROI scalar onto a (Ly, Lx) image given Suite2p stat entries.
# ``load_plane_view`` is used by the headless renderer below.
from . import scale as scale_mod
from . import utils
from .utils import paint_spatial


# Build a custom three-stop linear colormap. ``N=256`` is the LUT
# size; ``.copy()`` so we can mutate ``set_bad`` without altering the
# class default. ``set_bad`` chooses the colour for NaN pixels; we
# pick a neutral medium grey (``#808080``).
CYAN_TO_RED = mpl.colors.LinearSegmentedColormap.from_list(
    "cyan_to_red", ["#00FFFF", "#0000FF", "#FF0000"], N=256
).copy()
CYAN_TO_RED.set_bad(color="#808080")


def order_map_for_event(first_time_col: np.ndarray,
                        active_mask_col: np.ndarray) -> np.ndarray:
    """Rank 1..K for active ROIs in this event by first-onset time.

    Why we rank instead of using raw times
    --------------------------------------
    Two events of identical duration but different absolute onset
    times should produce the *same* coloured map -- otherwise event
    1 (early in the recording) would always be cyan and event N
    (late) would always be red, which doesn't reflect within-event
    propagation. Ranking decouples the colour scale from absolute
    time.

    Parameters
    ----------
    first_time_col : (N_rois,) float array
        ``first_time[:, ev_idx]`` from ``utils.detect_event_windows``.
        Carries each ROI's first onset time inside this event, or
        NaN if the ROI didn't fire here.
    active_mask_col : (N_rois,) bool array
        ``A[:, ev_idx]`` -- True for ROIs that fired in this event.

    Returns
    -------
    order_rank : (N_rois,) float array
        1 for the earliest-firing ROI, 2 for the next, ..., K for
        the latest. NaN for inactive ROIs (so the caller can colour
        them grey).

    Notes
    -----
    ``np.argsort(..., kind="mergesort")`` is *stable*: ROIs with
    identical onset times retain their original order, which gives
    deterministic colours across re-renders.
    """
    # Start with all-NaN; we'll overwrite the active entries.
    order_rank = np.full_like(first_time_col, np.nan, dtype=float)
    # Active AND has a valid onset time. ``~`` is bitwise-not on
    # the boolean array.
    sel = active_mask_col & ~np.isnan(first_time_col)
    if not np.any(sel):
        # Defensive: empty event -- caller will see all-NaN and paint
        # the FOV grey.
        return order_rank
    times = first_time_col[sel]
    # ``argsort`` returns the indices that would sort ``times``.
    # Stable sort so ties don't change ordering on re-render.
    idx_sorted = np.argsort(times, kind="mergesort")
    # Assign rank 1..K via positional inversion: ``ranks[idx_sorted]
    # = arange(1, K+1)`` says "the ROI at position idx_sorted[0]
    # gets rank 1, the one at idx_sorted[1] gets rank 2, ...".
    ranks = np.empty_like(idx_sorted, dtype=float)
    ranks[idx_sorted] = np.arange(1, idx_sorted.size + 1, dtype=float)
    # Scatter ranks back into the full-size array at the active
    # positions.
    order_rank[np.where(sel)[0]] = ranks
    return order_rank


def paint_order_map(order_rank: np.ndarray, stat_filtered, Ly: int,
                    Lx: int) -> np.ndarray:
    """Paint per-ROI ranks onto an ``(Ly, Lx)`` image normalised to
    ``[0, 1]`` (earliest=0, latest=1).

    With the ``CYAN_TO_RED`` cmap this gives earliest=cyan/blue,
    latest=red. Pixels outside any ROI are returned as NaN so that
    the cmap's ``set_bad`` colour shows them as neutral grey.

    The actual painting is done by ``utils.paint_spatial`` (the
    weighted-pixels splat routine); this function just normalises
    the ranks and handles the "no ROIs in this event" + "fewer than
    two ROIs" edge cases.
    """
    # ``.astype(float).copy()`` defensively detaches from the
    # caller's array so our in-place ops don't surprise them.
    vals = order_rank.astype(float).copy()
    # Coverage = "this pixel belongs to at least one ROI" mask.
    # Computed by painting all-ones -- bin-counts the ROI footprint.
    coverage = paint_spatial(np.ones(len(stat_filtered), dtype=float),
                             stat_filtered, Ly, Lx)
    if np.all(np.isnan(vals)):
        # No active ROIs in this event; paint a NaN canvas with
        # non-ROI pixels also NaN, so downstream imshow renders
        # everything as grey.
        img = paint_spatial(
            np.full_like(order_rank, np.nan, dtype=float),
            stat_filtered, Ly, Lx)
        img[coverage == 0] = np.nan
        return img

    # ``np.nanmax`` ignores NaNs. ``maxr`` is the highest rank K.
    maxr = np.nanmax(vals)
    if maxr > 1:
        # Ranks are in 1..K. Map 1 -> 0, K -> 1 by linear rescale.
        vals = (vals - 1.0) / (maxr - 1.0)
    else:
        # Single active ROI in this event: K=1, so the rescale would
        # divide by zero. Treat the ROI as "earliest" (0).
        vals = np.where(np.isnan(vals), np.nan, 0.0)

    img = paint_spatial(vals, stat_filtered, Ly, Lx)
    # Mark non-ROI pixels NaN so the colormap's ``set_bad`` picks
    # them up as grey.
    img[coverage == 0] = np.nan
    return img


def _roi_centroids_xy(stat_filtered):
    xs = np.array([float(np.median(s["xpix"])) for s in stat_filtered],
                  dtype=float)
    ys = np.array([float(np.median(s["ypix"])) for s in stat_filtered],
                  dtype=float)
    return xs, ys


def event_frame_centroids(first_time_col: np.ndarray,
                          active_mask_col: np.ndarray,
                          stat_filtered, fps: float) -> list[dict]:
    """Group the ROIs that fired during one event by their first-onset
    frame, and compute a centroid + 2-D spread per frame group.

    What this is for
    ----------------
    Tab 8's "frame-to-frame propagation" panel needs, for each frame
    of an event:
        * the centre of mass of the ROIs that fired in that frame;
        * how spread out those ROIs are.
    This routine produces both, in a flat list of dicts ordered by
    frame index.

    Parameters
    ----------
    first_time_col, active_mask_col
        Same shapes / meanings as ``order_map_for_event``.
    stat_filtered : list of Suite2p stat dicts
        Carries the per-ROI ``xpix`` / ``ypix`` arrays we'll
        centroid.
    fps : float
        Frame rate; converts ``first_time`` (s) to frame index.

    Returns
    -------
    list of dict, one per active frame, each with:
        ``frame`` : int
            Absolute frame index in the recording.
        ``rank`` : int
            Position in the event's active-frame sequence (0 = first
            active frame, ...).
        ``rois_local`` : ndarray of int
            Row indices into ``stat_filtered`` for ROIs in this group.
        ``cx``, ``cy`` : float
            Group centroid in pixel coordinates (use the caller's
            ``pix_to_um`` to convert when rendering on µm axes).
        ``sigma_x``, ``sigma_y`` : float
            Per-axis standard deviation. Zero when n=1.
        ``sigma_xy`` : float
            ``sqrt(var_x + var_y)`` -- 2-D RMS spread. Roughly 1.41
            standard deviations in 2-D.
        ``n`` : int
            Number of ROIs in this group.

    Frames with no ROI activations are skipped, so the returned list
    is contiguous in *active* frames, not absolute frame index.
    """
    sel = active_mask_col & ~np.isnan(first_time_col)
    if not np.any(sel):
        return []
    # Active-ROI indices into the full N_kept array.
    roi_idx = np.where(sel)[0]
    # Convert seconds to frame index by ``round(t * fps)``.
    frames = np.round(first_time_col[roi_idx] * float(fps)).astype(int)

    xs_all, ys_all = _roi_centroids_xy(stat_filtered)
    # Subset to only the active ROIs.
    xs = xs_all[roi_idx]
    ys = ys_all[roi_idx]

    out: list[dict] = []
    # ``set(...)`` gives unique frame indices; ``sorted(...)`` orders
    # them ascending. ``enumerate`` then attaches a 0-based rank.
    for rank, fr in enumerate(sorted(set(int(f) for f in frames))):
        # Boolean mask: which active ROIs fall in this exact frame?
        mask = frames == fr
        gx = xs[mask]; gy = ys[mask]
        cx = float(gx.mean()); cy = float(gy.mean())
        # ``np.std`` on a single element returns 0; we still write
        # the if-branch explicitly so the reader knows the n=1 case.
        if gx.size > 1:
            sx = float(gx.std()); sy = float(gy.std())
        else:
            sx = 0.0; sy = 0.0
        out.append({
            "frame": int(fr),
            "rank": int(rank),
            "rois_local": roi_idx[mask],
            "cx": cx, "cy": cy,
            "sigma_x": sx, "sigma_y": sy,
            "sigma_xy": float(np.sqrt(sx * sx + sy * sy)),
            "n": int(gx.size),
        })
    return out


# ============================================================================
# Directional monotonicity (Spearman vs projection angle)
# ============================================================================

def directional_monotonicity_spearman(
    xs: np.ndarray,
    ys: np.ndarray,
    ts: np.ndarray,
    *,
    n_angles: int = 360,
    n_shuffles: int = 10000,
    seed: int = 0,
) -> dict:
    """Find the propagation axis that maximises Spearman ρ between an
    event's activation times and a planar projection of ROI positions.

    Model
    -----
    Treat each active ROI as a point ``(x_i, y_i, t_i)`` where ``t_i``
    is its activation frame within this event. For a unit direction
    ``u(θ) = (cos θ, sin θ)`` define the projection
    ``s_i(θ) = x_i cos θ + y_i sin θ``. If propagation has a coherent
    direction θ*, the ordering of ``s_i(θ*)`` should match the
    ordering of ``t_i``, which is exactly what Spearman's rank
    correlation
    ``ρ(θ) = SpearmanCorr(s_i(θ), t_i)``
    measures (rank-only, so monotone-but-nonlinear relationships still
    score high; sign-only, so flipped direction reads as negative
    ρ instead of being missed).

    Sweep θ across ``[0, 2π)`` at ``n_angles`` evenly-spaced steps,
    pick ``θ*`` as the angle maximising ρ(θ), report:

    * ``rho_obs`` -- ρ(θ*), in [-1, 1]; magnitude is the monotonicity
      strength, 0 ≈ no directional structure, 1 = perfectly monotone
      along θ*.
    * Significance via a permutation test on activation times: at θ*,
      shuffle ``t_i`` across cells ``n_shuffles`` times and compute
      ρ for each shuffle. Spearman is rank-based so we precompute
      the rank of ``s_i(θ*)`` once and permute the rank of ``t``
      (numerically equivalent and ~30x faster than re-ranking each
      shuffle).

    Parameters
    ----------
    xs, ys : (n,) ndarray of float
        ROI centroids, any units (the test is rotation-invariant up
        to θ, so px or µm both work).
    ts : (n,) ndarray of float or int
        Activation times (frame index or seconds; only the *ordering*
        matters for Spearman).
    n_angles : int
        Number of θ samples across ``[0, 2π)``. 360 → 1° resolution.
    n_shuffles : int
        Permutation count for the null distribution. 10000 → p
        resolution ~1e-4; memory cost ~n_shuffles × n_angles × 8 B
        (≈29 MB at the default 10000 × 360 grid, still trivial).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict
        ``{
          'n':         number of points used,
          'thetas':    (n_angles,) θ values (rad),
          'rho_curve': (n_angles,) ρ at each θ,
          'theta_star':float, argmax θ (rad),
          'rho_obs':   float, ρ(θ*),
          'null_rho':  (n_shuffles,) permutation-test ρ values,
          'p_value':   float, P(shuffled ρ >= ρ_obs),
        }``
        Returns ``{}`` (empty dict) when there's insufficient signal
        (``n < 3`` or no spread in ``t``).
    """
    xs = np.asarray(xs, dtype=float).ravel()
    ys = np.asarray(ys, dtype=float).ravel()
    ts = np.asarray(ts, dtype=float).ravel()
    n = int(xs.size)
    if n < 3 or ys.size != n or ts.size != n:
        return {}
    # Spearman is undefined when one variable is constant; bail out
    # cleanly so the renderer can show a "needs more frames" hint
    # instead of a NaN-ridden plot.
    if np.unique(ts).size < 2:
        return {}
    if np.unique(xs).size < 2 and np.unique(ys).size < 2:
        return {}

    # scipy.stats.rankdata is faster than np.argsort+argsort for the
    # sizes we care about (n ~ 5-500) and handles ties cleanly.
    from scipy.stats import rankdata
    rank_t = rankdata(ts, method="average")
    rank_t_c = rank_t - rank_t.mean()
    denom_t = float(np.sqrt((rank_t_c * rank_t_c).sum()))
    if denom_t == 0.0:
        return {}

    thetas = np.linspace(0.0, 2.0 * np.pi, int(n_angles), endpoint=False)
    rho_curve = np.empty(thetas.size, dtype=float)
    # Vectorise the projection over angles: build (n_angles, n)
    # projection matrix in one shot, rank each row independently.
    # For typical n ~ 30 and n_angles = 360 that's ~11k entries —
    # tiny.
    proj = (np.outer(np.cos(thetas), xs)
            + np.outer(np.sin(thetas), ys))                  # (n_angles, n)
    ranks_s = rankdata(proj, method="average", axis=1)       # (n_angles, n)
    ranks_s_c = ranks_s - ranks_s.mean(axis=1, keepdims=True)
    denom_s = np.sqrt((ranks_s_c * ranks_s_c).sum(axis=1))   # (n_angles,)
    safe = denom_s > 0
    rho_curve[~safe] = 0.0
    rho_curve[safe] = (ranks_s_c[safe] @ rank_t_c) / (denom_s[safe] * denom_t)

    best_i = int(np.argmax(rho_curve))
    theta_star = float(thetas[best_i])
    rho_obs = float(rho_curve[best_i])

    # Permutation null. Each shuffle reassigns activation times across
    # cells, then sweeps θ on the same projection grid and records the
    # *max* ρ -- mirroring how ρ_obs itself was picked. Without the
    # max-over-θ step the null would be at a fixed direction and the
    # observed statistic (a max over 360 angles) would compare against
    # an uncorrected baseline, inflating significance.
    rng = np.random.default_rng(int(seed))
    perms = rng.standard_normal((int(n_shuffles), n)).argsort(axis=1)
    perm_ranks = rank_t[perms]                              # (n_shuffles, n)
    perm_ranks_c = perm_ranks - perm_ranks.mean(axis=1, keepdims=True)
    # Reuse the precomputed (n_angles, n) projection ranks; one big
    # matmul gives ρ at every (shuffle, angle) pair. Memory cost at
    # the default 10000 × 360 grid is ~29 MB — still fine; bump
    # n_shuffles further only if you need sub-1e-4 p resolution.
    denom_s_safe = np.where(denom_s > 0, denom_s, 1.0)
    rho_matrix = (perm_ranks_c @ ranks_s_c.T) / (denom_s_safe * denom_t)
    null_rho = rho_matrix.max(axis=1)                       # (n_shuffles,)
    # One-sided p: how often does the null's max ρ match or exceed
    # ρ_obs. ``>=`` so ties count (conservative).
    p_value = float(np.mean(null_rho >= rho_obs))

    return {
        "n": n,
        "thetas": thetas,
        "rho_curve": rho_curve,
        "theta_star": theta_star,
        "rho_obs": rho_obs,
        "null_rho": null_rho,
        "p_value": p_value,
    }


# ============================================================================
# Headless per-event activation-order figure renderer (Tab 0 batch)
# ============================================================================
# Tab 8 is purely visual -- no computation to run headlessly -- but the batch
# runner still wants the per-event activation-order PNGs saved alongside
# everything else under ``calliope_figures/``. ``render_spatial_event_figures``
# renders one PNG per event using the same ``order_map_for_event`` +
# ``paint_order_map`` pipeline the tab uses, without any Tk dependency. Was
# split into ``core/spatial_run.py`` previously; consolidated back into
# ``core/spatial.py`` on 2026-05-11 as part of the core-folder refactor since
# both files belong to the same Tab 8 domain.

def render_spatial_event_figures(
    plane0: Path,
    event_data: dict,
    *,
    figures_dir: Path,
) -> list[str]:
    """Render the order-map-with-arrows panel (Tab 8's most
    informative single view) for every event window. Saved as
    ``event_001.png``, ``event_002.png``, ... under ``figures_dir``.
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    plane0 = Path(plane0)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    ev = np.asarray(event_data.get("event_windows"))
    if ev is None or ev.size == 0:
        return []
    A = np.asarray(event_data["A"])
    first_time = np.asarray(event_data["first_time"])
    kept_idx = np.asarray(event_data["kept_idx"])
    fps: Optional[float] = event_data.get("fps")

    view = utils.load_plane_view(plane0)
    Ly, Lx = int(view["Ly"]), int(view["Lx"])
    pix_to_um = scale_mod.resolve_pix_to_um(view)
    stat_all = np.load(plane0 / "stat.npy", allow_pickle=True)
    stat_filtered = [stat_all[i] for i in kept_idx.tolist()]

    # Suite2p stores ROI pixels with y=0 at the TOP of the FOV (image
    # convention). Render with origin="upper" + a flipped y-extent so the
    # rendered figure's orientation matches suite2p's GUI -- y-axis ticks
    # go 0 at top -> Ly at bottom, centroids land at the suite2p (cx, cy)
    # without needing a Ly - cy flip.
    if pix_to_um is not None:
        scale = float(pix_to_um)
        extent = [0, Lx * scale, Ly * scale, 0]
        xlabel, ylabel = "X (µm)", "Y (µm)"
    else:
        scale = 1.0
        extent = [0, Lx, Ly, 0]
        xlabel, ylabel = "X (px)", "Y (px)"

    written: list[str] = []
    n_events = ev.shape[0]
    for ev_idx in range(n_events):
        first_col = first_time[:, ev_idx]
        active_col = A[:, ev_idx]
        order_rank = order_map_for_event(first_col, active_col)
        img = paint_order_map(order_rank, stat_filtered, Ly, Lx)
        t0 = float(ev[ev_idx, 0]); t1 = float(ev[ev_idx, 1])
        n_active = int(active_col.sum())
        n_total = int(active_col.size)
        frac = n_active / n_total if n_total else 0.0

        frame_groups = []
        if fps:
            frame_groups = event_frame_centroids(
                first_col, active_col, stat_filtered, float(fps))
        cxs = np.array([g["cx"] * scale for g in frame_groups])
        cys = np.array([g["cy"] * scale for g in frame_groups])

        fig = plt.Figure(figsize=(7, 6), tight_layout=True)
        ax = fig.add_subplot(111)
        im = ax.imshow(
            img, origin="upper",
            cmap=CYAN_TO_RED,
            aspect="equal", vmin=0.0, vmax=1.0, extent=extent,
        )
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(
            f"Event {ev_idx + 1} / {n_events}  ({t0:.2f}-{t1:.2f} s)\n"
            f"active = {n_active} / {n_total} ({100 * frac:.1f}%)",
            fontsize=10,
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
                            ticks=[0.0, 0.5, 1.0])
        cbar.ax.set_yticklabels(["earliest", "", "latest"])

        if frame_groups:
            ax.scatter(cxs, cys, s=18, c="white",
                       edgecolors="black", linewidths=0.6, zorder=4)
            for i in range(len(frame_groups) - 1):
                x0, y0 = cxs[i], cys[i]
                x1, y1 = cxs[i + 1], cys[i + 1]
                if abs(x1 - x0) + abs(y1 - y0) < 1e-9:
                    continue
                ax.annotate(
                    "", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle="->", color="white",
                        lw=1.6, shrinkA=2, shrinkB=2,
                        mutation_scale=14),
                    zorder=5,
                )

        out = figures_dir / f"event_{ev_idx + 1:03d}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        written.append(str(out))

    return written
