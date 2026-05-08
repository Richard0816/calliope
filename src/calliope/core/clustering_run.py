"""Headless hierarchical clustering for batch execution and Tab 6.

Why this module exists
----------------------
Tab 6's compute pipeline (linkage, threshold pick, label remap,
cluster export, summary writer) was scattered across the tab class.
Tab 0's batch runner needs to invoke the same flow without a Tk
window. This module collects the headless steps in one place; the
tab keeps its interactive widgets but the actual clustering goes
through here.

Public surface
--------------
``run_clustering(plane0, params, *, figures_dir=None,
                 write_summary=True)``
    Compute the linkage, pick a threshold (auto or user-specified),
    cut to flat clusters, write ``<plane0>/<prefix>cluster_results/
    gui_recluster/C{i}_rois.npy`` files (Suite2p ROI ids), the
    accompanying ``linkage.npy`` / ``threshold_used.npy`` /
    ``_indices_are_suite2p`` markers, the Clusters sheet in
    ``calliope_summary.xlsx``, and (optionally) dendrogram +
    spatial cluster map PNGs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster

from . import clustering_cmap as cmap_mod


# Subfolder name kept consistent with Tab 6's ``EXPORT_SUBDIR`` so
# downstream tabs (cross-correlation) can find the exports.
EXPORT_SUBDIR = "gui_recluster"


def _load_filter_mask(plane0: Path) -> Optional[np.ndarray]:
    pred_path = plane0 / "predicted_cell_mask.npy"
    if pred_path.exists():
        return np.load(pred_path).astype(bool)
    iscell_path = plane0 / "iscell.npy"
    if iscell_path.exists():
        ic = np.load(iscell_path)
        return ((ic[:, 0] > 0) if ic.ndim == 2 else (ic > 0)).astype(bool)
    return None


def _load_filtered_dff(plane0: Path, prefix: str):
    """Open ``<prefix>dff.memmap.float32`` against the cell-filter
    mask. Returns (dff_array, T, N_kept, keep_mask).
    """
    plane0 = Path(plane0)
    F = np.load(plane0 / "F.npy", mmap_mode="r")
    N_total, T = F.shape
    is_filtered = "filtered" in prefix.split("_")

    if is_filtered:
        mask = _load_filter_mask(plane0)
        if mask is None:
            raise FileNotFoundError(
                f"{plane0}: prefix {prefix!r} requires a cell-filter mask "
                "(predicted_cell_mask.npy or iscell.npy).")
        if mask.size != N_total:
            raise ValueError(
                f"{plane0}: cell-filter mask length {mask.size} does not "
                f"match F.npy ROI count {N_total}.")
        N_kept = int(mask.sum())
    else:
        mask = None
        N_kept = N_total

    dff_path = plane0 / f"{prefix}dff.memmap.float32"
    if not dff_path.exists():
        raise FileNotFoundError(f"Missing dF/F memmap: {dff_path}")
    dff = np.memmap(dff_path, dtype="float32", mode="r",
                    shape=(T, N_kept))
    return dff, T, N_kept, mask


def _correlation_linkage(dff: np.ndarray,
                         method: str = "average") -> np.ndarray:
    """Linkage on the (1 - Pearson r) distance between every pair of
    ROIs. Mirrors ``tabs.clustering.tab._correlation_linkage`` but
    without the Tk import chain.
    """
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage
    dff_z = (dff - np.mean(dff, axis=0)) / (np.std(dff, axis=0) + 1e-8)
    dist = pdist(dff_z.T, metric="correlation")
    return linkage(dist, method=method)


def _visual_cluster_map(Z: np.ndarray, T: float) -> dict[int, int]:
    """Map visual cluster index (1..n, dendrogram left-to-right) to
    the raw fcluster label. Matches Tab 6's
    ``_compute_visual_cluster_map``.
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
        ``threshold`` (None = auto-pick via
        :func:`clustering_cmap.auto_choose_threshold`),
        ``target_counts`` (default ``(4, 5)``),
        ``palette`` (default ``"tab10"``),
        ``method`` (default ``"average"``),
        ``metric`` (default ``"correlation"``).
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
    method = str(params.get("method", "average"))
    target_counts = tuple(params.get("target_counts", (4, 5)))

    dff, T, N_kept, _mask = _load_filtered_dff(plane0, prefix)
    dff_arr = np.asarray(dff)
    Z = _correlation_linkage(dff_arr, method=method)

    threshold_param = params.get("threshold")
    if threshold_param is None:
        frac = cmap_mod.auto_choose_threshold(Z, target_counts=target_counts)
        threshold = float(frac) * float(np.max(Z[:, 2]))
    else:
        threshold = float(threshold_param)

    labels = fcluster(Z, t=threshold, criterion="distance")
    visual_to_label = _visual_cluster_map(Z, threshold)
    n_clusters = len(visual_to_label)

    export_dir = plane0 / f"{prefix}cluster_results" / EXPORT_SUBDIR
    export_dir.mkdir(parents=True, exist_ok=True)

    filtered_to_suite2p: Optional[np.ndarray] = None
    if "filtered" in prefix.split("_"):
        mask = _load_filter_mask(plane0)
        if mask is not None and int(mask.sum()) == len(labels):
            filtered_to_suite2p = np.where(
                np.asarray(mask, dtype=bool))[0].astype(int)

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
    (export_dir / "_indices_are_suite2p").write_text("1")

    link_colors = cmap_mod.resolve_palette(
        palette, n_colors=max(1, n_clusters))

    if figures_dir is not None:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        dendro_path = figures_dir / "dendrogram.png"
        # plot_dendrogram expects a fractional threshold (0..1).
        zmax = float(np.max(Z[:, 2])) or 1.0
        cmap_mod.plot_dendrogram(
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
            cmap_mod.plot_spatial(
                plane0, order=order, link_colors=leaf_colors,
                save_path=spatial_path, used_indices=used,
                prefix=prefix,
                title=f"Cluster spatial map ({n_clusters} clusters)",
            )
        except Exception as e:
            # Don't let figure rendering kill the whole batch step.
            print(f"[clustering_run] plot_spatial failed: {e}")

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
        try:
            cluster_summary.write_clusters_sheet(
                plane0,
                cluster_labels=visual_labels,
                cluster_colors=cluster_colors,
                threshold=threshold,
                method=method,
                metric=str(params.get("metric", "correlation")),
                roi_indices=roi_indices,
            )
        except Exception as e:
            print(f"[clustering_run] write_clusters_sheet failed: {e}")

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
