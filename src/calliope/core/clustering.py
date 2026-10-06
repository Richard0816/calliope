"""Hierarchical clustering of ROI dF/F traces (Tab 6 backend +
headless batch entry point).

What this file does, in one paragraph
-------------------------------------
Cells in a slice are not independent -- groups of them have tightly
correlated firing patterns ("functional ensembles"). To find those
groups we (1) z-score each ROI's trace, (2) compute the pairwise
Euclidean distance matrix between every pair of ROIs (which on
z-scored data is equivalent to correlation distance up to a
constant: ``||x - y||^2 = 2T * (1 - r)``), (3) run scipy's
hierarchical agglomerative clustering with Ward linkage, and (4)
either auto-pick a flat-cluster count or let the user slide a
threshold along the dendrogram. The output is per-cell cluster
labels -- the same colours then drive both the dendrogram and a
spatial map of the FOV.

This module mixes three layers:

* Palette + plotting + threshold helpers (``resolve_palette``,
  ``count_clusters``, ``auto_choose_threshold``, ``plot_dendrogram``,
  ``plot_spatial``). ``resolve_palette`` / ``count_clusters`` /
  ``auto_choose_threshold`` are shared with the interactive Tab 6;
  the two plotters are used only by the headless ``run_clustering``
  (Tab 6 draws its own interactive figures).
* Tk-free data helpers shared with Tab 6 (``tabs/clustering/logic.py``
  re-exports them): ``ward_linkage``, ``load_filtered_dff``,
  ``stat_for_prefix`` and ``filtered_to_suite2p_indices``. Filtered
  readers size the keep mask via ``utils.resolve_filtered_mask``
  (anchored to the memmap's on-disk column count), never the live
  ``predicted_cell_mask``/``iscell``.
* The headless orchestrator ``run_clustering(plane0, params, ...)``
  that drives the linkage + threshold pick + label remap + cluster
  export + summary writer end-to-end, so callers without a Tk window
  (``core.batch_pipeline.run_recording``, figure-regeneration scripts)
  reproduce Tab 6's output.

Where to look in the source if you've never used SciPy clustering
-----------------------------------------------------------------
* ``scipy.spatial.distance.pdist`` produces a *condensed* distance
  vector (length ``n*(n-1)/2``); each entry is the distance between
  one pair of items.
* ``scipy.cluster.hierarchy.linkage`` consumes that vector and
  returns a ``(n-1, 4)`` array describing the merge tree -- the
  agglomerative history. R users will recognise this as the same
  output ``stats::hclust`` produces.
* ``scipy.cluster.hierarchy.fcluster`` cuts the tree at a chosen
  threshold and returns flat cluster labels.

Public surface
--------------
``run_clustering(plane0, params, *, figures_dir=None, write_summary=True)``
    Headless "compute one clustering" entry point: linkage, cut,
    ``C*_rois.npy`` export, optional figures + summary sheet. Tab 6's
    Run button does NOT call it (it offloads
    ``tabs.clustering.logic.compute_clustering_offload`` and drives
    the cut interactively).
``auto_choose_threshold(Z, target_counts=(4, 5), start, stop, step)``
    Heuristic for "find a cut fraction that splits the dendrogram into
    one of ``target_counts`` clusters". Default cut for Tab 6's Run and
    for ``run_clustering`` when ``params['threshold']`` is None.
``ward_linkage`` / ``load_filtered_dff`` / ``stat_for_prefix`` /
``filtered_to_suite2p_indices``
    Shared data helpers (see above).

Branch re-clustering ("Recluster within branch") lives in the GUI
(``tabs.clustering.logic.ReclusterWindow``), not here.

Designed as a backend for GUI integration: each step is a small
function that accepts parameters and returns plain data, so the GUI
can call them independently (load -> cluster -> plot) without re-
running work. For an R reader this layout mirrors ``hclust`` ->
``cutree`` -> ``plot.dendrogram`` style pipelines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# SciPy's hierarchical-clustering primitives. ``linkage`` builds the
# tree; ``dendrogram`` plots it; ``fcluster`` cuts it at a height;
# ``set_link_color_palette`` lets us colour branches consistently.
from scipy.cluster.hierarchy import (
    linkage,
    dendrogram,
    fcluster,
    set_link_color_palette,
)
# ``pdist`` returns the condensed pairwise-distance vector consumed
# by ``linkage``.
from scipy.spatial.distance import pdist

from . import utils
from .scale import resolve_pix_to_um


# Handy presets a GUI dropdown can offer directly.
CATEGORICAL_PALETTES = [
    "tab10", "tab20", "Set1", "Set2", "Set3",
    "Paired", "Accent", "Pastel1", "Pastel2", "Dark2",
]
CONTINUOUS_PALETTES = [
    "viridis", "plasma", "inferno", "magma", "cividis",
    "coolwarm", "RdBu", "RdYlBu", "Spectral",
    "turbo", "rainbow", "hsv",
]
AVAILABLE_PALETTES = CATEGORICAL_PALETTES + CONTINUOUS_PALETTES


#: Type alias for "anything we'll accept as a palette spec".
#: Used in function signatures throughout this module so the API
#: docs stay readable. ``Union[A, B, C]`` is "any of these types";
#: in Python 3.10+ you can also write ``A | B | C``.
PaletteLike = Union[str, Iterable[str], mpl.colors.Colormap]


def resolve_palette(palette: PaletteLike, n_colors: int = 10) -> list[str]:
    """Normalise a palette spec into a list of hex colour strings.

    Why we need this
    ----------------
    Tab 6 lets the user pick a palette from a dropdown that mixes
    continuous matplotlib colormaps ("viridis"), categorical seaborn
    palettes ("tab10"), and explicit user-supplied colour lists.
    Each of those needs slightly different handling. This helper
    swallows them all and emits a flat list of ``n_colors`` hex
    strings spanning the palette's full range -- so the FIRST and
    LAST clusters in the dendrogram get the most visually different
    colours.

    Accepts
    -------
    - ``"viridis"``, ``"tab10"`` etc. -- matplotlib colormap name.
    - A matplotlib ``Colormap`` instance.
    - An iterable of colour strings: hex (``"#ff0000"``), named
      (``"red"``), or RGB tuples.

    For continuous colormaps, ``n_colors`` samples are drawn evenly
    across [0, 1]. For qualitative ones (``.colors`` attribute), we
    take entries in order, downsampled if the user asked for fewer
    than the cmap provides.
    """
    if not isinstance(palette, (str, mpl.colors.Colormap)):
        colors = [mpl.colors.to_hex(c) for c in palette]
        if n_colors <= 0:
            return colors
        if n_colors <= len(colors):
            # Spread selections evenly across the provided list, so a
            # small cluster count still ends up using endpoints.
            idx = np.linspace(0, len(colors) - 1, n_colors).round().astype(int)
            return [colors[i] for i in idx]
        return colors

    cmap = mpl.colormaps.get_cmap(palette) if isinstance(palette, str) else palette
    n_colors = max(1, int(n_colors))

    # Qualitative colormaps expose a discrete .colors list.
    discrete = getattr(cmap, "colors", None)
    if discrete is not None and len(discrete) <= 20:
        colors = list(discrete)
        if n_colors <= len(colors):
            idx = np.linspace(0, len(colors) - 1, n_colors).round().astype(int)
            return [mpl.colors.to_hex(colors[i]) for i in idx]
        return [mpl.colors.to_hex(c) for c in colors]

    # Continuous: sample n_colors evenly across [0, 1].
    if n_colors == 1:
        return [mpl.colors.to_hex(cmap(0.5))]
    return [
        mpl.colors.to_hex(cmap(i / (n_colors - 1)))
        for i in range(n_colors)
    ]


def count_clusters(Z: np.ndarray, color_threshold: float) -> int:
    """Number of distinct flat clusters below ``color_threshold *
    max(linkage)``.

    Counts ``fcluster`` labels rather than dendrogram branch colours:
    matplotlib's dendrogram renderer can chunk small clusters into a
    single "above_threshold_color" group, so a colour count
    under-reports; ``fcluster`` gives the actual cluster count.
    """
    # A degenerate (0-merge) linkage has no tree to cut -- there is at
    # most one observation, i.e. exactly one trivial cluster.
    if Z is None or len(Z) < 1:
        return 1
    # Convert fractional cut to absolute distance.
    T = color_threshold * np.max(Z[:, 2])
    labels = fcluster(Z, t=T, criterion="distance")
    # ``np.unique`` returns sorted unique values; ``len(...)`` is
    # the count.
    return int(len(np.unique(labels)))


def auto_choose_threshold(
    Z: np.ndarray,
    target_counts: Iterable[int] = (4, 5),
    start: float = 0.90,
    stop: float = 0.05,
    step: float = 0.01,
) -> float:
    """Heuristic: walk the cut fraction down from ``start`` until the
    fcluster count lands in ``target_counts``.

    The default targets (4, 5) match what the lab finds visually
    useful for typical brain-slice recordings. Returns the fractional
    cut height; if no value in the sweep produces a target count, we
    return ``start`` (the user can then drag the slider).

    Tab 6 calls this on first Render so the user sees something
    reasonable without having to touch the slider.

    Counts via ``count_clusters`` (i.e. ``fcluster``), not by counting
    dendrogram leaf colours. Matplotlib's dendrogram renderer
    bundles small clusters under a single ``above_threshold_color``
    group, so the colour-group count under-reports the true
    fcluster count on fragmented recordings -- the auto-pick would
    land at 5 visible colours while the spatial map showed 30+
    actual clusters. Fragmentation bug was caught 2026-05-11 by
    a downstream user; switching to ``count_clusters`` makes the
    auto-pick match what the user sees on the spatial map.
    """
    # ``set(...)`` for fast ``in`` lookup below.
    target_counts = set(target_counts)
    if not target_counts:
        return start
    lo_target = min(target_counts)
    hi_target = max(target_counts)

    # The candidate grid is the same descending ``start -> stop`` sweep as
    # before (``stop - 1e-9`` so np.arange actually visits ``stop``), so we
    # return a value identical to the original linear scan.
    grid = np.arange(start, stop - 1e-9, -step)
    if len(grid) == 0:
        return start

    # ``count_clusters`` is monotonically non-decreasing along this grid:
    # the cut height shrinks as the fraction drops, which can only split
    # clusters (never merge), so the count only grows. That lets us binary
    # search for the first grid index whose count reaches ``lo_target``
    # instead of re-running fcluster at every one of the ~85 steps.
    n = len(grid)
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if count_clusters(Z, float(grid[mid])) >= lo_target:
            hi = mid
        else:
            lo = mid + 1

    # From the first index that reaches the target band, the original scan
    # returned the first grid point whose count is actually in the set.
    # Counts are monotonic, so a short forward scan (bounded by hi_target)
    # reproduces that exactly -- and handles a non-contiguous target set.
    i = lo
    while i < n:
        c = count_clusters(Z, float(grid[i]))
        if c in target_counts:
            return float(grid[i])
        if c > hi_target:
            break
        i += 1
    return start


def plot_dendrogram(
    Z: np.ndarray,
    save_path: Path,
    color_threshold: float,
    palette: PaletteLike = "tab10",
    above_threshold_color: str = "gray",
    n_palette_colors: Optional[int] = None,
    cluster_colors: Optional[Sequence[str]] = None,
) -> list[str]:
    """
    Draw the dendrogram using `palette` for the below-cut branches.

    If `n_palette_colors` is None, it defaults to the actual number of
    below-threshold clusters so the palette spans end-to-end (first
    cluster on one side of the cmap, last cluster on the other).

    `cluster_colors`, when provided, is passed straight to
    `set_link_color_palette` (cluster_colors[0] -> C1, [1] -> C2, ...),
    overriding `palette` entirely.
    """
    if cluster_colors is not None:
        colors = [mpl.colors.to_hex(c) for c in cluster_colors]
    else:
        if n_palette_colors is None:
            n_palette_colors = max(1, count_clusters(Z, color_threshold))
        colors = resolve_palette(palette, n_colors=n_palette_colors)
    set_link_color_palette(colors)

    T = color_threshold * np.max(Z[:, 2])
    plt.figure(figsize=(10, 5))
    dendrogram(Z, color_threshold=T, above_threshold_color=above_threshold_color)
    plt.axhline(T, linestyle="--", linewidth=2)
    plt.text(
        0.99, T, f" cut @ {T:.3g}  ({color_threshold:.2f}×max)",
        transform=plt.gca().get_yaxis_transform(),
        ha="right", va="bottom",
    )
    plt.title("ROI Hierarchical Clustering Dendrogram")
    plt.xlabel("ROIs")
    plt.ylabel("Linkage distance")
    plt.tight_layout()
    save_path = Path(save_path)
    plt.savefig(save_path, dpi=200)
    plt.savefig(save_path.with_suffix(".svg"))
    plt.close()
    return colors


def stat_for_prefix(plane0: Path, prefix: str) -> tuple[list, Optional[np.ndarray]]:
    """Stat list aligned with the columns of the ``<prefix>`` dF/F memmap.

    For filtered prefixes (e.g. ``r0p7_filtered_``), restrict ``stat`` to
    the keep mask so the i-th stat entry matches the i-th dF/F column.
    Returns ``(stat_list, original_indices_or_None)``.

    The mask is resolved the same way :func:`load_filtered_dff` resolves
    it -- anchored to the filtered memmap's on-disk column count via
    ``utils.resolve_filtered_mask`` -- not read blindly from
    ``r0p7_cell_mask_bool.npy`` or the live ``predicted_cell_mask`` /
    ``iscell``. Otherwise, when curation drifts those files after the
    memmap was written, stat is sliced to a different count than the
    dF/F columns and ROI colours land on the wrong cells.
    """
    plane0 = Path(plane0)
    full_stat = list(np.load(plane0 / "stat.npy", allow_pickle=True))
    if utils.is_filtered_prefix(prefix):
        dff_path = plane0 / f"{prefix}dff.memmap.float32"
        F = np.load(plane0 / "F.npy", mmap_mode="r")
        N_total, T = F.shape
        mask, _ = utils.resolve_filtered_mask(
            plane0, N_total, memmap_path=dff_path, T=T)
        return ([s for s, keep in zip(full_stat, mask) if keep],
                np.where(mask)[0])
    return full_stat, None


def plot_spatial(
    root: Path,
    order: Iterable[int],
    link_colors: Iterable[str],
    save_path: Path,
    used_indices: Optional[np.ndarray] = None,
    prefix: str = "r0p7_filtered_",
    title: str = "Spatial map colored by dendrogram ROI colors",
) -> None:
    """
    Paint ROIs on the FOV using the same colors scipy gave the dendrogram
    leaves. Axes are in µm when a pix_to_um conversion is available
    (ops or zoom notes). No colorbar — the image is RGB by construction.

    If `used_indices` is provided, stat is restricted to those rows.
    Otherwise, if `prefix` indicates a filtered ROI set, stat is restricted
    via :func:`stat_for_prefix` (memmap-anchored keep mask) so `order`
    aligns with clustering indices.
    """
    root = Path(root)
    view = utils.load_plane_view(root)
    Ly, Lx = view["Ly"], view["Lx"]

    if used_indices is not None:
        full_stat = list(np.load(root / "stat.npy", allow_pickle=True))
        stat = [full_stat[i] for i in used_indices]
    else:
        stat, _ = stat_for_prefix(root, prefix)

    link_colors = list(link_colors)
    roi_rgb = np.zeros((len(stat), 3))
    for i, roi_idx in enumerate(order):
        roi_rgb[roi_idx, :] = mpl.colors.to_rgb(link_colors[i])

    R = utils.paint_spatial(roi_rgb[:, 0], stat, Ly, Lx)
    G = utils.paint_spatial(roi_rgb[:, 1], stat, Ly, Lx)
    B = utils.paint_spatial(roi_rgb[:, 2], stat, Ly, Lx)
    img = np.dstack([R, G, B])

    coverage = utils.paint_spatial(np.ones(len(stat)), stat, Ly, Lx)
    img[coverage == 0] = np.nan

    pix_to_um = resolve_pix_to_um(view)
    if pix_to_um is not None:
        extent = [0, Lx * pix_to_um, 0, Ly * pix_to_um]
        xlabel, ylabel = "X (µm)", "Y (µm)"
    else:
        extent = None
        xlabel, ylabel = "X (pixels)", "Y (pixels)"

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(img, origin="upper", extent=extent, aspect="equal")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if pix_to_um is not None:
        from .figscale import add_scale_bar
        add_scale_bar(ax, pix_to_um, axes_in_um=True,
                      span_um=Lx * pix_to_um, color="black")
    plt.tight_layout()
    save_path = Path(save_path)
    plt.savefig(save_path, dpi=200)
    plt.savefig(save_path.with_suffix(".svg"))
    plt.close(fig)
    print("Saved", save_path)


# ---------------------------------------------------------------------------
# Tk-free data helpers (shared with tabs/clustering/logic.py)
# ---------------------------------------------------------------------------


def load_filtered_dff(plane0: Path, prefix: str):
    """Open ``<prefix>dff.memmap.float32`` against the cell-filter mask.

    For a "filtered" prefix (e.g. ``r0p7_filtered_``) the memmap was
    written at the size of the keep mask live at Tab 3 time, so we need
    that exact mask to interpret the column count. For an unfiltered
    prefix, the memmap holds every Suite2p ROI.

    Returns ``(dff_array, T, N_kept, keep_mask)``. ``keep_mask`` is
    ``None`` when the prefix is unfiltered.
    """
    plane0 = Path(plane0)
    F = np.load(plane0 / "F.npy", mmap_mode="r")
    N_total, T = F.shape
    is_filtered = utils.is_filtered_prefix(prefix)

    dff_path = plane0 / f"{prefix}dff.memmap.float32"
    if not dff_path.exists():
        raise FileNotFoundError(f"Missing dF/F memmap: {dff_path}")

    if is_filtered:
        # Size against the mask the memmap was written with (persisted as
        # r0p7_cell_mask_bool.npy), cross-checked against the file size, so
        # a drifted predicted_cell_mask/iscell can't request a wrong-shaped
        # mapping -> [WinError 8] on Windows. See utils.resolve_filtered_mask.
        mask, N_kept = utils.resolve_filtered_mask(
            plane0, N_total, memmap_path=dff_path, T=T)
    else:
        mask = None
        N_kept = N_total

    dff = np.memmap(dff_path, dtype="float32", mode="r",
                    shape=(T, N_kept))
    return dff, T, N_kept, mask


def filtered_to_suite2p_indices(plane0: Path, prefix: str) -> Optional[np.ndarray]:
    """Translator from kept-ROI (dF/F column) position to Suite2p ROI index.

    Returns ``np.where(mask)[0]`` for a filtered prefix and ``None`` for an
    unfiltered one (where column position already *is* the Suite2p index).

    Like :func:`stat_for_prefix`, the mask is resolved against the filtered
    memmap's on-disk column count via ``utils.resolve_filtered_mask`` so the
    translator stays the exact length of the linkage's leaves even after
    curation drifts ``predicted_cell_mask``/``iscell``. Export / ROI-id
    callsites previously read the live mask and silently fell back to
    emitting filtered-list positions (not Suite2p ids) whenever
    ``mask.sum()`` no longer matched the leaf count.
    """
    if not utils.is_filtered_prefix(prefix):
        return None
    plane0 = Path(plane0)
    dff_path = plane0 / f"{prefix}dff.memmap.float32"
    F = np.load(plane0 / "F.npy", mmap_mode="r")
    N_total, T = F.shape
    mask, _ = utils.resolve_filtered_mask(
        plane0, N_total, memmap_path=dff_path, T=T)
    return np.where(np.asarray(mask, dtype=bool))[0].astype(int)


def ward_linkage(dff: np.ndarray, method: str = "ward") -> np.ndarray:
    """Ward linkage on z-scored ROI traces with Euclidean distance.

    Z-scoring each ROI to ``(mean=0, std=1)`` flattens amplitude so
    two cells with very different baseline brightness or scale but
    the same activity *shape* land close together. With z-scoring
    applied, squared Euclidean distance and (1 - Pearson r) are
    equivalent up to a constant (``||x - y||^2 = 2T * (1 - r)``)
    so the *pair-ranking* is identical to correlation distance --
    we just use Euclidean because scipy's Ward implementation
    requires a Euclidean metric. The resulting merge heights (and
    therefore every cut threshold) are absolute Ward distances, not
    ``1 - r`` values.

    Why Ward and not average:
    - Ward minimises within-cluster sum-of-squares (Lance-Williams).
      Produces compact, roughly balanced clusters -- the typical
      "4-5 modes" shape lab members find useful.
    - Average linkage merges by mean pairwise distance. On
      unbalanced data it can collapse to "1 huge cluster + a few
      tiny outliers", which is what triggered the earlier
      cluster-fragmentation guard on cross-correlation.

    History: the pre-2026-05-11 GUI used average + correlation
    here. Switched to Ward after noting that z-scoring already kills
    amplitude differences, so the "Ward needs Euclidean" objection
    no longer applied.
    """
    n_roi = int(np.asarray(dff).shape[1]) if np.asarray(dff).ndim == 2 else 0
    if n_roi < 2:
        # pdist on <2 columns returns an empty condensed vector and
        # linkage raises the cryptic "The number of observations cannot
        # be determined on an empty distance matrix". Fail with a clear,
        # actionable message instead.
        raise ValueError(
            f"Need at least 2 ROIs to cluster; got {n_roi}. "
            f"Hierarchical clustering is undefined for a single trace.")
    dff_z = (dff - np.mean(dff, axis=0)) / (np.std(dff, axis=0) + 1e-8)
    dist = pdist(dff_z.T, metric="euclidean")
    return linkage(dist, method=method)


# ---------------------------------------------------------------------------
# Headless orchestrator -- run_clustering
# ---------------------------------------------------------------------------
#
# The interactive Tab 6 drives its linkage / threshold / cluster export
# from the tab class because it also has to update Tk widgets between
# steps. For callers without a Tk window we want a single function with
# no Tk dependency; run_clustering below reproduces the same outputs.
# Cluster .npy files are written directly into
# ``<plane0>/<prefix>cluster_results/`` -- no extra subfolder.


def _visual_cluster_map(Z: np.ndarray, T: float) -> dict[int, int]:
    """Map visual cluster index (1..n, dendrogram left-to-right) to
    the raw fcluster label. Same traversal as Tab 6's memoised
    ``ClusteringTab._compute_visual_cluster_map``.
    """
    info = dendrogram(Z, no_plot=True, color_threshold=T,
                      above_threshold_color="gray")
    leaves = list(info["leaves"])
    labels = fcluster(Z, t=T, criterion="distance")
    seen: list[int] = []
    for leaf_idx in leaves:
        lbl = int(labels[leaf_idx])
        if lbl not in seen:
            seen.append(lbl)
    return {i + 1: lbl for i, lbl in enumerate(seen)}


def run_clustering(
    plane0: Path,
    params: dict,
    *,
    figures_dir: Optional[Path] = None,
    write_summary: bool = True,
) -> dict:
    """Compute a hierarchical clustering of ROI dF/F traces and
    export the cluster ROI lists.

    Parameters
    ----------
    plane0 : Path
        Suite2p plane folder.
    params : dict
        ``prefix`` (default ``"r0p7_filtered_"``),
        ``threshold`` (None = auto-pick via :func:`auto_choose_threshold`),
        ``target_counts`` (default ``(4, 5)``),
        ``palette`` (default ``"tab10"``),
        ``method`` (default ``"ward"``),
        ``metric`` (default ``"euclidean"``).
    figures_dir : Path or None
        If set, saves ``dendrogram.png`` and ``spatial_clusters.png``
        under that directory.
    write_summary : bool
        If True (default), writes/updates the ``Clusters`` sheet in
        ``<plane0>/calliope_summary.xlsx``.

    Returns
    -------
    dict
        ``{Z, threshold, n_clusters, labels, visual_to_label,
        export_dir, prefix, palette, link_colors}``.
    """
    from . import summary_writer as cluster_summary

    plane0 = Path(plane0)
    prefix = str(params.get("prefix", "r0p7_filtered_"))
    palette = params.get("palette", "tab10")
    method = str(params.get("method", "ward"))
    target_counts = tuple(params.get("target_counts", (4, 5)))

    dff, T, N_kept, keep_mask = load_filtered_dff(plane0, prefix)
    dff_arr = np.asarray(dff)
    Z = ward_linkage(dff_arr, method=method)

    threshold_param = params.get("threshold")
    if threshold_param is None:
        frac = auto_choose_threshold(Z, target_counts=target_counts)
        threshold = float(frac) * float(np.max(Z[:, 2]))
    else:
        threshold = float(threshold_param)

    labels = fcluster(Z, t=threshold, criterion="distance")
    visual_to_label = _visual_cluster_map(Z, threshold)
    n_clusters = len(visual_to_label)

    export_dir = plane0 / f"{prefix}cluster_results"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Translate filtered dF/F column positions to Suite2p ROI ids with the
    # SAME memmap-anchored mask load_filtered_dff just used to size the
    # matrix -- never the live predicted_cell_mask/iscell, which drifts
    # after curation (the old live-mask length check then failed and
    # filtered positions were exported under the _indices_are_suite2p
    # marker). For an unfiltered prefix, positions already are Suite2p ids.
    filtered_to_suite2p: Optional[np.ndarray] = None
    if keep_mask is not None:
        filtered_to_suite2p = np.where(
            np.asarray(keep_mask, dtype=bool))[0].astype(int)
        if filtered_to_suite2p.size != len(labels):
            raise ValueError(
                f"{plane0}: keep mask selects {filtered_to_suite2p.size} "
                f"ROIs but the linkage has {len(labels)} leaves; refusing "
                f"to export cluster ROI ids that cannot be translated to "
                f"Suite2p indices.")

    for stale in export_dir.glob("C*_rois.npy"):
        stale.unlink()
    for vid in range(1, n_clusters + 1):
        lbl = int(visual_to_label[vid])
        local_idx = np.where(labels == lbl)[0].astype(int)
        if filtered_to_suite2p is not None:
            roi_idx = filtered_to_suite2p[local_idx]
        else:
            roi_idx = local_idx
        np.save(export_dir / f"C{vid}_rois.npy", roi_idx.astype(int))
    np.save(export_dir / "linkage.npy", Z)
    np.save(export_dir / "threshold_used.npy",
            np.array([threshold], dtype=float))
    # Every C*_rois.npy above holds Suite2p ids (translated for a filtered
    # prefix, identity for an unfiltered one), so the marker telling
    # core.crosscorrelation to skip its legacy translation is now true.
    (export_dir / "_indices_are_suite2p").write_text("1")

    _base_colors = resolve_palette(palette, n_colors=max(1, n_clusters))
    # resolve_palette caps categorical palettes (e.g. tab10 -> 10) at their
    # discrete size, so for n_clusters > len(palette) the list is too short and
    # the link_colors[vid - 1] lookups below would IndexError. Cycle the
    # palette (matching scipy's set_link_color_palette behaviour) so the list
    # always has exactly n_clusters entries. For n_clusters <= len(palette)
    # this is a no-op (i % len == i), so the common case is unchanged.
    link_colors = [
        _base_colors[i % len(_base_colors)]
        for i in range(max(1, n_clusters))
    ]

    if figures_dir is not None:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        dendro_path = figures_dir / "dendrogram.png"
        # plot_dendrogram expects a fractional threshold (0..1).
        zmax = float(np.max(Z[:, 2])) or 1.0
        plot_dendrogram(
            Z, dendro_path, color_threshold=threshold / zmax,
            palette=palette, n_palette_colors=n_clusters,
        )

        spatial_path = figures_dir / "spatial_clusters.png"
        # plot_spatial expects ``order`` = a leaf list and
        # ``link_colors`` = per-leaf colours. Build them from the
        # dendrogram traversal so colours line up with the
        # dendrogram PNG.
        info = dendrogram(Z, no_plot=True, color_threshold=threshold,
                          above_threshold_color="gray")
        order = list(info["leaves"])
        leaf_colors = list(info["leaves_color_list"]) \
            if "leaves_color_list" in info else None
        if leaf_colors is None:
            # Older scipy: derive from labels + palette.
            label_to_color = {
                int(visual_to_label[vid]): link_colors[vid - 1]
                for vid in range(1, n_clusters + 1)
            }
            leaf_colors = [label_to_color[int(labels[i])] for i in order]
        used = filtered_to_suite2p
        try:
            plot_spatial(
                plane0, order=order, link_colors=leaf_colors,
                save_path=spatial_path, used_indices=used,
                prefix=prefix,
                title=f"Cluster spatial map ({n_clusters} clusters)",
            )
        except Exception as e:
            # Don't let figure rendering kill the whole batch step.
            print(f"[clustering] plot_spatial failed: {e}")

    if write_summary:
        # Remap raw fcluster labels to visual ids so the Clusters
        # sheet's cluster_id column matches the dendrogram numbering
        # (1..n in left-to-right leaf order).
        label_to_visual = {int(lbl): vid
                           for vid, lbl in visual_to_label.items()}
        visual_labels = np.array(
            [label_to_visual.get(int(l), 0) for l in labels],
            dtype=int,
        )
        cluster_colors = {vid: link_colors[vid - 1]
                          for vid in range(1, n_clusters + 1)}
        roi_indices = (filtered_to_suite2p.tolist()
                       if filtered_to_suite2p is not None
                       else None)
        # Stamp the Recording sheet too, matching the GUI Tab-6 path
        # (clustering/tab.py). Previously the core runner wrote only the
        # Clusters sheet, so a batch run that skipped event detection
        # ended up with no Recording sheet at all.
        try:
            from .utils import get_fps_from_notes
            fps = float(get_fps_from_notes(str(plane0)))
        except Exception:
            fps = None
        try:
            cluster_summary.update_recording_meta(
                plane0, prefix=prefix, fps=fps,
                T=None, N=int(len(labels)),
                extra={"clustering_n_clusters": int(n_clusters),
                       "clustering_threshold": float(threshold),
                       "clustering_palette": str(palette)})
        except Exception as e:
            print(f"[clustering] update_recording_meta failed: {e}")
        try:
            cluster_summary.write_clusters_sheet(
                plane0,
                cluster_labels=visual_labels,
                cluster_colors=cluster_colors,
                threshold=threshold,
                method=method,
                metric=str(params.get("metric", "euclidean")),
                roi_indices=roi_indices,
            )
        except Exception as e:
            print(f"[clustering] write_clusters_sheet failed: {e}")

    return {
        "Z": Z,
        "threshold": threshold,
        "n_clusters": n_clusters,
        "labels": labels,
        "visual_to_label": visual_to_label,
        "export_dir": str(export_dir),
        "prefix": prefix,
        "palette": palette,
        "link_colors": link_colors,
    }
