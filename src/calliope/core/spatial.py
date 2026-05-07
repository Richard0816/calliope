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

# ``paint_spatial`` does the heavy lifting: it knows how to splat a
# per-ROI scalar onto a (Ly, Lx) image given Suite2p stat entries.
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
