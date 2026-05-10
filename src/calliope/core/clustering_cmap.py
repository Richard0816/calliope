"""Hierarchical clustering of ROI dF/F traces (Tab 6 backend).

What this file does, in one paragraph
-------------------------------------
Cells in a slice are not independent -- groups of them have tightly
correlated firing patterns ("functional ensembles"). To find those
groups we (1) compute the pairwise correlation matrix between every
pair of ROIs, (2) treat ``1 - r`` as a distance, (3) run scipy's
hierarchical agglomerative clustering with average linkage, and (4)
either auto-pick a flat-cluster count or let the user slide a
threshold along the dendrogram. The output is per-cell cluster
labels -- the same colours then drive both the dendrogram and a
spatial map of the FOV.

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
``run_clustering(traces, ...)``
    The end-to-end "compute one clustering" function Tab 6's Render
    button calls.
``recluster_branch(linkage_Z, branch_id, ...)``
    Re-cluster a single dendrogram branch in isolation, used when
    the user picks one fat cluster and asks for a finer split.
``auto_choose_threshold(linkage_Z, target_n_clusters)``
    Heuristic for "find a height that splits the dendrogram into ~N
    clusters". Used as the default when the user hits Render.

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
# seaborn provides categorical / continuous colour palettes that look
# nicer than matplotlib's defaults for cluster colouring.
import seaborn as sns
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


def _resolve_pix_to_um(root: Path, ops: dict) -> Optional[float]:
    """Read the µm-per-pixel scale that Tab 3 stored on ``ops``.

    Thin wrapper around :func:`calliope.core.scale.resolve_pix_to_um`
    kept here for backward compatibility -- callers in this module
    pass the recording root by convention but the new resolver only
    needs the ops dict (Tab 3 is responsible for writing
    ``ops['pix_to_um']`` based on the user's zoom or direct
    calibration). ``root`` is unused.
    """
    del root  # historical signature; ops carries the scale now
    return resolve_pix_to_um(ops)


def load_dff(root: Path, prefix: str = "r0p7_filtered_") -> np.ndarray:
    """Open the ``(T, N_kept)`` filtered dF/F memmap for a recording.

    Convenience wrapper around ``utils.s2p_open_memmaps`` that picks
    just the dF/F array (not the lowpass + derivative siblings).
    """
    dff, _, _ = utils.s2p_open_memmaps(root, prefix=prefix)[:3]
    if dff.ndim != 2:
        raise ValueError(f"Expected (T, N) ΔF/F array, got shape {dff.shape}")
    return dff


def run_clustering(
    dff: np.ndarray,
    method: str = "ward",
    metric: str = "euclidean",
) -> np.ndarray:
    """Compute the linkage matrix used to draw a dendrogram.

    Steps:
        1. Z-score each ROI's trace independently so cells with
           higher absolute brightness don't dominate the distance.
        2. Compute the pairwise distance matrix between ROIs in
           condensed form (``scipy.spatial.distance.pdist``).
        3. Run hierarchical agglomerative clustering with the chosen
           ``method`` (Ward by default; the GUI more typically
           overrides this to "average" with the "correlation" metric
           for the 1 - Pearson r distance described in the
           module-level docstring).

    Returns the linkage matrix Z -- shape ``(N-1, 4)`` -- where each
    row records one merge: ``[cluster_a, cluster_b, distance,
    cluster_size]``. R users will recognise this as the same data
    structure ``stats::hclust`` returns under ``$merge`` + ``$height``.
    """
    # ``+ 1e-8`` is a tiny epsilon that avoids dividing by zero on a
    # constant trace (which would otherwise become NaN and poison
    # the distance matrix).
    dff_z = (dff - np.mean(dff, axis=0)) / (np.std(dff, axis=0) + 1e-8)
    # ``.T`` transposes (T, N) -> (N, T); ``pdist`` operates on rows.
    dist_matrix = pdist(dff_z.T, metric=metric)
    return linkage(dist_matrix, method=method)


def count_leaf_color_groups(Z: np.ndarray, color_threshold: float) -> int:
    """Count how many distinct branch colours the dendrogram would
    use at the given cut height fraction.

    ``color_threshold`` is a fraction of ``max(Z[:, 2])`` -- the
    deepest merge -- so that a single "cut depth" parameter scales
    naturally across recordings.

    Used internally by ``auto_choose_threshold`` to find a height
    that produces a target cluster count. ``no_plot=True`` makes
    ``dendrogram`` skip the actual rendering and just return its
    metadata.
    """
    r = dendrogram(Z, no_plot=True, color_threshold=color_threshold * np.max(Z[:, 2]))
    # ``set(...)`` deduplicates the per-leaf colour assignment to
    # give a colour count.
    return len(set(r["leaves_color_list"]))


def count_clusters(Z: np.ndarray, color_threshold: float) -> int:
    """Number of distinct flat clusters below ``color_threshold *
    max(linkage)``.

    Differs from ``count_leaf_color_groups`` because matplotlib's
    dendrogram renderer can chunk small clusters into a single
    "above_threshold_color" group; ``fcluster`` gives the actual
    cluster count.
    """
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
    branch-colour count lands in ``target_counts``.

    The default targets (4, 5) match what the lab finds visually
    useful for typical brain-slice recordings. Returns the fractional
    cut height; if no value in the sweep produces a target count, we
    return ``start`` (the user can then drag the slider).

    Tab 6 calls this on first Render so the user sees something
    reasonable without having to touch the slider.
    """
    # ``set(...)`` for fast ``in`` lookup below.
    target_counts = set(target_counts)
    # Walk from start downward in ``step``-sized decrements.
    # ``stop - 1e-9`` is a hair below the lower bound so np.arange
    # actually visits ``stop``.
    for ct in np.arange(start, stop - 1e-9, -step):
        if count_leaf_color_groups(Z, float(ct)) in target_counts:
            return float(ct)
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
    plt.savefig(save_path, dpi=200)
    plt.close()
    return colors


def plot_heatmap(
    dff: np.ndarray,
    order: np.ndarray,
    save_path: Path,
    heatmap_cmap: str = "magma",
) -> None:
    dff_sorted = dff[:, order]
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        dff_sorted.T,
        cmap=heatmap_cmap,
        cbar_kws={"label": "ΔF/F"},
        xticklabels=False,
        yticklabels=False,
    )
    plt.title("Hierarchical Clustering of ROI ΔF/F Traces")
    plt.xlabel("Time (s)")
    plt.ylabel("ROIs (clustered order)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def _stat_for_prefix(root: Path, prefix: str) -> tuple[list, Optional[np.ndarray]]:
    """
    Match the filtered-ROI convention used by hierarchical_clustering.py:
    when the prefix signals a filtered set (e.g. 'r0p7_filtered_'),
    restrict stat to ROIs kept by the cell mask so indices align with
    the filtered ΔF/F used during clustering.
    """
    root = Path(root)
    stat = list(np.load(root / "stat.npy", allow_pickle=True))
    if "filtered" in prefix.split("_"):
        mask_path = root / "r0p7_cell_mask_bool.npy"
        if mask_path.exists():
            mask = np.load(mask_path)
            used = np.where(mask)[0]
            return [stat[i] for i in used], used
        print(f"⚠️ {mask_path} not found; painting against unfiltered stat.npy.")
    return stat, None


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
    via the cell mask so `order` aligns with clustering indices.
    """
    root = Path(root)
    view = utils.load_plane_view(root)
    Ly, Lx = view["Ly"], view["Lx"]

    if used_indices is not None:
        full_stat = list(np.load(root / "stat.npy", allow_pickle=True))
        stat = [full_stat[i] for i in used_indices]
    else:
        stat, _ = _stat_for_prefix(root, prefix)

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

    pix_to_um = _resolve_pix_to_um(root, view)
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
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print("Saved", save_path)


def cluster_and_plot(
    root: Path,
    prefix: str = "r0p7_filtered_",
    method: str = "ward",
    metric: str = "euclidean",
    palette: PaletteLike = "tab10",
    heatmap_cmap: str = "magma",
    above_threshold_color: str = "gray",
    color_threshold: Optional[float] = None,
    target_counts: Iterable[int] = (4, 5),
    save_dir: Optional[Path] = None,
    save_format: Union[str, Iterable[str]] = "png",
    cluster_colors: Optional[Sequence[str]] = None,
) -> dict:
    """
    End-to-end: load ΔF/F -> cluster -> write dendrogram, heatmap, spatial map.

    `save_format` controls the output extension(s). Pass "png" (default),
    "svg", "pdf", or an iterable like ("png", "svg") to save in multiple
    formats. matplotlib's savefig picks the renderer from the extension.

    `cluster_colors`, when provided, is an explicit list of colors mapping
    onto C1, C2, C3, ... (dendrogram order, left to right). It overrides
    `palette`. If shorter than n_clusters, scipy cycles through it;
    extras beyond n_clusters are ignored.

    Returns a dict with the linkage matrix, leaf order, chosen threshold,
    leaf colors, and the output directory — everything a GUI needs to
    display results or re-render with a different palette.
    """
    save_dir = (
        Path(save_dir)
        if save_dir
        else Path(root) / f"{prefix}cluster_results" / "colourblind_safe"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    formats = (save_format,) if isinstance(save_format, str) else tuple(save_format)
    formats = tuple(f.lstrip(".").lower() for f in formats)

    dff = load_dff(Path(root), prefix=prefix)
    Z = run_clustering(dff, method=method, metric=metric)

    if color_threshold is None:
        color_threshold = auto_choose_threshold(Z, target_counts=target_counts)

    order = np.asarray(dendrogram(Z, no_plot=True)["leaves"], dtype=int)

    # Sample the palette with exactly n_clusters entries so the first
    # cluster lands on one end of the colormap, the last on the other,
    # and everything else is evenly spread in between. Apply BEFORE both
    # the plotted dendrogram and the no_plot extraction so leaves_color_list
    # reflects the user's choice for the spatial map too.
    n_clusters = max(1, count_clusters(Z, color_threshold))
    if cluster_colors is not None:
        palette_colors = [mpl.colors.to_hex(c) for c in cluster_colors]
    else:
        palette_colors = resolve_palette(palette, n_colors=n_clusters)
    set_link_color_palette(palette_colors)

    T = color_threshold * np.max(Z[:, 2])
    link_colors = dendrogram(Z, no_plot=True, color_threshold=T)["leaves_color_list"]

    for ext in formats:
        plot_heatmap(
            dff, order, save_dir / f"cluster_heatmap.{ext}",
            heatmap_cmap=heatmap_cmap,
        )
        plot_dendrogram(
            Z,
            save_dir / f"dendrogram.{ext}",
            color_threshold=color_threshold,
            palette=palette,
            above_threshold_color=above_threshold_color,
            n_palette_colors=n_clusters,
            cluster_colors=cluster_colors,
        )
        plot_spatial(
            Path(root),
            order,
            link_colors,
            save_path=save_dir / f"spatial_dendrogram_colored_rois.{ext}",
            prefix=prefix,
        )

    np.save(save_dir / "cluster_order.npy", order)
    np.save(save_dir / "linkage.npy", Z)

    return {
        "linkage": Z,
        "order": order,
        "color_threshold": color_threshold,
        "n_clusters": n_clusters,
        "link_colors": link_colors,
        "palette_colors": palette_colors,
        "save_dir": save_dir,
    }


