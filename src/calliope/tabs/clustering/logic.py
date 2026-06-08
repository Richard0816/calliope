"""Logic + calculations for the Clustering tab.

This module hosts:

1. **Re-export shim** for ``calliope.core`` slices Tab 6 needs:

   - ``clustering`` -- the actual hierarchical clustering algorithm
     (``run_clustering``, ``auto_choose_threshold``, ``count_clusters``)
     plus palette helpers and the dendrogram / spatial-map plotters.
   - ``summary_writer`` -- writes the Clusters sheet to
     ``calliope_summary.xlsx``.
   - ``utils`` -- ``get_fps_from_notes`` and ``paint_spatial`` for the
     spatial map below the dendrogram.

   Re-exporting whole modules (rather than picking individual names)
   keeps the call sites readable: ``clustering.run_clustering(...)``
   is more navigable than a dozen unbound symbol imports.

2. **Pure compute helpers** used by Tab 6's ``ClusteringTab``:
   ``ward_linkage``, ``load_filter_mask``, ``load_filtered_dff``,
   ``stat_for_prefix``, ``build_label_image``, ``spatial_image``.
   Extracted from ``tab.py`` so the GUI module focuses on widget
   layout + lifecycle.

3. **Dialog / popout classes** that aren't strictly "logic" but live
   here for the same reason -- to keep ``tab.py`` under control:
   ``CustomColorDialog`` (per-cluster colour picker), ``ReclusterWindow``
   (sub-tree dendrogram inspector).

Backwards-compat alias: ``clustering_cmap`` is exposed as an alias of
``clustering`` so older import paths (``from .logic import
clustering_cmap as cmap_mod``) keep working through the refactor.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import colorchooser
from typing import Optional

import customtkinter as ctk
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist

from ...core import clustering, summary_writer, utils
from ...gui_common import (
    apply_dark_to_tk_widget, attach_fig_toolbar, install_scroll_router,
)

# Backwards-compat alias for the pre-merge module name. The merged
# module is now ``calliope.core.clustering``; the old name was
# ``clustering_cmap``. Tab 6 still imports it under the ``cmap_mod``
# alias so the rebind keeps the call sites working unchanged.
clustering_cmap = clustering

# Defaults shared between ``tab.py`` and ``ReclusterWindow``.
DEFAULT_PREFIX = "r0p7_filtered_"
DEFAULT_PALETTE = "tab10"
# Above-cut dendrogram links + un-clustered ROIs render in this grey
# so the visual hierarchy stays calm against the dark theme.
ABOVE_CUT_COLOR = "gray"

__all__ = [
    "clustering",
    "clustering_cmap",
    "summary_writer",
    "utils",
    "DEFAULT_PREFIX",
    "DEFAULT_PALETTE",
    "ABOVE_CUT_COLOR",
    "ward_linkage",
    "load_filter_mask",
    "load_filtered_dff",
    "stat_for_prefix",
    "filtered_to_suite2p_indices",
    "build_label_image",
    "spatial_image",
    "render_spatial_map",
    "CustomColorDialog",
    "ReclusterWindow",
]


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------


def ward_linkage(dff: np.ndarray, method: str = "ward") -> np.ndarray:
    """Ward linkage on z-scored ROI traces with Euclidean distance.

    Z-scoring each ROI to ``(mean=0, std=1)`` flattens amplitude so
    two cells with very different baseline brightness or scale but
    the same activity *shape* land close together. With z-scoring
    applied, squared Euclidean distance and (1 - Pearson r) are
    equivalent up to a constant (``||x - y||^2 = 2T * (1 - r)``)
    so the *pair-ranking* is identical to correlation distance --
    we just use Euclidean to keep scipy's Ward implementation
    happy (Ward requires a Euclidean metric).

    Why Ward and not average:
    - Ward minimises within-cluster sum-of-squares (Lance-Williams).
      Produces compact, roughly balanced clusters -- the typical
      "4-5 modes" shape lab members find useful.
    - Average linkage merges by mean pairwise distance. On
      unbalanced data it can collapse to "1 huge cluster + a few
      tiny outliers", which is what triggered the earlier
      cluster-fragmentation guard on cross-correlation.

    History: the pre-2026-05-11 GUI used average + correlation
    here. The docstring claimed Ward was off the table because
    "Ward needs Euclidean" -- technically true, but irrelevant
    once we've z-scored. Switched to Ward after the user pointed
    out that z-scoring already kills amplitude differences.
    """
    n_roi = int(np.asarray(dff).shape[1]) if np.asarray(dff).ndim == 2 else 0
    if n_roi < 2:
        # pdist on <2 columns yields an empty condensed vector and
        # linkage raises the cryptic "empty distance matrix" error.
        # Surface a clear, actionable message instead.
        raise ValueError(
            f"Need at least 2 ROIs to cluster; got {n_roi}. "
            f"Hierarchical clustering is undefined for a single trace.")
    dff_z = (dff - np.mean(dff, axis=0)) / (np.std(dff, axis=0) + 1e-8)
    dist = pdist(dff_z.T, metric="euclidean")
    return linkage(dist, method=method)


def load_filter_mask(plane0: Path) -> Optional[np.ndarray]:
    """Cell-filter keep mask for a Suite2p plane0 directory.

    Tries, in order:
      1. predicted_cell_mask.npy (cell-filter classifier output, preferred)
      2. iscell.npy (suite2p classifier fallback)
    Returns ``None`` if neither file exists.
    """
    pred_path = plane0 / "predicted_cell_mask.npy"
    if pred_path.exists():
        return np.load(pred_path).astype(bool)
    iscell_path = plane0 / "iscell.npy"
    if iscell_path.exists():
        ic = np.load(iscell_path)
        return ((ic[:, 0] > 0) if ic.ndim == 2 else (ic > 0)).astype(bool)
    return None


def load_filtered_dff(plane0: Path, prefix: str):
    """Open ``<prefix>dff.memmap.float32`` against the cell-filter mask.

    For a "filtered" prefix (e.g. ``r0p7_filtered_``) the memmap was written
    at the size of the cell-filter keep mask, so we need that exact mask to
    interpret the column count. For an unfiltered prefix, the memmap holds
    every Suite2p ROI.

    Returns ``(dff_array, T, N_kept, keep_mask)``. ``keep_mask`` is ``None``
    when the prefix is unfiltered.
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


def stat_for_prefix(plane0: Path, prefix: str):
    """Stat list aligned with the columns of the dF/F memmap.

    For filtered prefixes, restrict ``stat`` to the cell-filter keep mask so
    the i-th stat entry matches the i-th column of dF/F. Returns
    ``(stat_list, original_indices_or_None)``.

    The mask MUST be resolved the same way ``load_filtered_dff`` resolves it
    -- anchored to the filtered memmap's on-disk column count via
    ``utils.resolve_filtered_mask`` -- not from the live
    ``predicted_cell_mask``/``iscell``. Otherwise, when curation drifts those
    files after the memmap was written, the stat list is sliced to a different
    count than the dF/F columns and the spatial map fails the
    ``len(stat) != N`` guard with "Spatial unavailable (stat / dF/F mismatch)".
    """
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


def compute_clustering_offload(plane0_str: str, prefix: str, *,
                               progress_q=None) -> dict:
    """Picklable :mod:`calliope.core.offload` target for Tab 6 Run.

    Runs the ``pdist`` + Ward linkage in a child process so the GUI
    stays responsive. Returns only PICKLABLE data: the linkage ``Z`` +
    scalars. The live ``dff`` memmap and ``stat`` list are NOT returned
    (a memmap can't cross the process boundary, and both are cheap to
    re-open by path) -- the tab re-opens them on the main thread in
    ``_on_done``. ``progress_q`` is accepted for offload uniformity even
    though clustering has no progress stream today.
    """
    plane0 = Path(plane0_str)
    dff_mm, T, N, _ = load_filtered_dff(plane0, prefix)
    Z = ward_linkage(dff_mm)
    view = utils.load_plane_view(plane0)
    Lx, Ly = int(view["Lx"]), int(view["Ly"])
    auto_frac = clustering.auto_choose_threshold(Z, target_counts=(4, 5))
    return {
        "Z": Z, "dff_shape": tuple(int(x) for x in dff_mm.shape),
        "Lx": Lx, "Ly": Ly, "auto_frac": float(auto_frac),
        "T": int(T), "N": int(N),
    }


def reload_clustering_offload(plane0_str: str, prefix: str,
                              cluster_dir_str: str, *,
                              progress_q=None) -> dict:
    """Picklable :mod:`calliope.core.offload` target for Tab 6 Reload.

    Loads the saved linkage + threshold and validates the leaf count
    against the current dF/F memmap, all in a child process. Like
    :func:`compute_clustering_offload`, it returns only picklable data;
    the tab re-opens ``dff`` + ``stat`` by path on the main thread.
    """
    plane0 = Path(plane0_str)
    cluster_dir = Path(cluster_dir_str)
    Z = np.load(cluster_dir / "linkage.npy")
    thr = float(np.asarray(np.load(cluster_dir / "threshold_used.npy"),
                           dtype=float).ravel()[0])
    _, T, N, _ = load_filtered_dff(plane0, prefix)
    # The loaded linkage was computed on N_kept ROIs; if the cell-filter
    # mask has changed since the export the shapes won't agree and any
    # downstream recolouring would be silently wrong.
    n_leaves = int(Z.shape[0]) + 1
    if n_leaves != int(N):
        raise ValueError(
            f"Saved linkage has {n_leaves} leaves but the current "
            f"dF/F memmap has {N} columns. Re-run analysis instead "
            f"of reloading.")
    view = utils.load_plane_view(plane0)
    Lx, Ly = int(view["Lx"]), int(view["Ly"])
    zmax = float(np.max(Z[:, 2]))
    return {
        "Z": Z, "Lx": Lx, "Ly": Ly, "saved_threshold": thr,
        "zmax": zmax, "T": int(T), "N": int(N),
        "cluster_dir": str(cluster_dir),
    }


def filtered_to_suite2p_indices(plane0: Path, prefix: str) -> Optional[np.ndarray]:
    """Translator from kept-ROI (dF/F column) position to Suite2p ROI index.

    Returns ``np.where(mask)[0]`` for a filtered prefix and ``None`` for an
    unfiltered one (where column position already *is* the Suite2p index).

    Like :func:`stat_for_prefix`, the mask is resolved against the filtered
    memmap's on-disk column count via ``utils.resolve_filtered_mask`` so the
    translator stays the exact length of the linkage's leaves even after
    curation drifts ``predicted_cell_mask``/``iscell``. Export / ROI-id
    callsites previously read the live mask and silently fell back to emitting
    filtered-list positions (not Suite2p ids) whenever ``mask.sum()`` no
    longer matched the leaf count.
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


def build_label_image(stat, Ly: int, Lx: int) -> np.ndarray:
    """Return a ``(Ly, Lx)`` int32 array where each painted pixel
    holds ``filtered_idx + 1`` (0 = background).

    Used by the spatial-click handler to map a cursor position to
    the position in Tab 6's filtered stat list. Mirrors
    ``Suite2pTab``'s ``build_label_image`` so click semantics stay
    consistent across the two tabs.
    """
    label = np.zeros((int(Ly), int(Lx)), dtype=np.int32)
    for i, s in enumerate(stat, start=1):
        yp = np.asarray(s["ypix"]); xp = np.asarray(s["xpix"])
        ok = (yp >= 0) & (yp < Ly) & (xp >= 0) & (xp < Lx)
        label[yp[ok], xp[ok]] = i
    return label


def spatial_image(stat, Lx, Ly, roi_rgb: np.ndarray) -> np.ndarray:
    """Paint a per-ROI RGB array onto the FOV (NaN background)."""
    R = utils.paint_spatial(roi_rgb[:, 0], stat, Ly, Lx)
    G = utils.paint_spatial(roi_rgb[:, 1], stat, Ly, Lx)
    B = utils.paint_spatial(roi_rgb[:, 2], stat, Ly, Lx)
    img = np.dstack([R, G, B])
    coverage = utils.paint_spatial(np.ones(len(stat)), stat, Ly, Lx)
    img[coverage == 0] = np.nan
    return img


def render_spatial_map(ax, img, Lx, Ly, pix_to_um, title) -> None:
    """Draw a clustering spatial map on ``ax`` in micrometres when a pixel
    calibration is available, else in raw pixels.

    Shared by the Tab 6 main view (``_render_all``) and the recluster
    popout (``ReclusterWindow._render``) so both honour ``pix_to_um``
    identically. Sets an ``imshow`` extent in µm (preserving the
    origin="upper" orientation), labels the axes, and adds a black scale
    bar (contrasts the light NaN background). Pixel fallback keeps the
    prior "x (px)" / "y (px)" labels and no bar.
    """
    ax.clear()
    if pix_to_um:
        extent = [0.0, Lx * pix_to_um, Ly * pix_to_um, 0.0]
        ax.imshow(img, origin="upper", aspect="equal", extent=extent)
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        from ...core.figscale import add_scale_bar
        add_scale_bar(ax, pix_to_um, axes_in_um=True,
                      span_um=Lx * pix_to_um, color="black")
    else:
        ax.imshow(img, origin="upper", aspect="equal")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
    ax.set_title(title)


# ---------------------------------------------------------------------------
# Per-cluster custom-color dialog
# ---------------------------------------------------------------------------


class CustomColorDialog(ctk.CTkToplevel):
    """Modal dialog to override the per-cluster color list.

    ``initial_colors`` is a list of hex strings (length = current
    cluster count). Clicking a swatch opens ``askcolor``; OK writes
    the new list to ``result``.
    """

    def __init__(self, master, initial_colors: list[str]) -> None:
        super().__init__(master)
        self.title("Per-cluster colors")
        self.transient(master)
        self.resizable(False, False)
        self._colors = list(initial_colors)
        self.result: Optional[list[str]] = None
        self._swatches: list[ctk.CTkButton] = []
        self._build_ui()
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        # Consume wheel events so they don't leak through CTk's
        # global bind_all and scroll the main app behind this dialog.
        install_scroll_router(self)

    def _build_ui(self) -> None:
        frm = ctk.CTkFrame(self, fg_color="transparent")
        frm.pack(fill="both", expand=True, padx=10, pady=10)
        ctk.CTkLabel(
            frm, text="Click a swatch to pick that cluster's color.",
            text_color="gray", anchor="w",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))
        for i, hex_color in enumerate(self._colors):
            ctk.CTkLabel(frm, text=f"Cluster {i + 1}").grid(
                row=i + 1, column=0, sticky="w", padx=(0, 8), pady=2)
            btn = ctk.CTkButton(frm, text="", width=48, height=24,
                                fg_color=hex_color, hover_color=hex_color,
                                border_width=1, border_color="gray40",
                                command=lambda idx=i: self._pick(idx))
            btn.grid(row=i + 1, column=1, sticky="w", pady=2)
            self._swatches.append(btn)
            ctk.CTkLabel(frm, text=hex_color).grid(
                row=i + 1, column=2, sticky="w", padx=(8, 0))

        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(fill="x", padx=10, pady=(0, 10))
        ctk.CTkButton(bar, text="Cancel", width=100,
                      command=self._cancel).pack(side="right")
        ctk.CTkButton(bar, text="OK", width=100,
                      command=self._ok).pack(side="right", padx=(0, 6))

    def _pick(self, idx: int) -> None:
        rgb, hex_color = colorchooser.askcolor(
            color=self._colors[idx], parent=self,
            title=f"Pick color for cluster {idx + 1}")
        if hex_color:
            self._colors[idx] = hex_color
            self._swatches[idx].configure(fg_color=hex_color,
                                          hover_color=hex_color)

    def _ok(self) -> None:
        self.result = list(self._colors)
        self.grab_release()
        self.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.grab_release()
        self.destroy()


# ---------------------------------------------------------------------------
# Recluster inspection window
# ---------------------------------------------------------------------------


class ReclusterWindow(ctk.CTkToplevel):
    """Read-only sub-tree view: dendrogram + spatial map for a chosen union
    of branches, with its own threshold slider. Does not modify the parent
    tab's state in any way."""

    def __init__(self, master, *, Z: np.ndarray, stat, Lx: int, Ly: int,
                 init_threshold: float, zmax: float,
                 palette_resolver, branch_label: str, n_rois: int,
                 pix_to_um=None) -> None:
        super().__init__(master)
        self.title(f"Recluster - branch(es) {branch_label}  "
                   f"({n_rois} ROIs)")
        self.geometry("1200x600")
        self.wm_minsize(800, 420)

        self._Z = Z
        self._stat = stat
        self._Lx = int(Lx)
        self._Ly = int(Ly)
        self._pix_to_um = pix_to_um
        self._zmax = float(zmax)
        self._T = float(init_threshold)
        self._palette_resolver = palette_resolver
        self._slider_user_driven = True

        self._build_ui()
        self._render()
        # Consume wheel events so they don't leak through CTk's
        # global bind_all and scroll the main app behind this popout.
        install_scroll_router(self)

    def _build_ui(self) -> None:
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=6, pady=6)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=0)
        body.columnconfigure(2, weight=2)
        body.rowconfigure(0, weight=1)

        # Dendrogram.
        dframe = ctk.CTkFrame(body)
        dframe.grid(row=0, column=0, sticky="nsew")
        ctk.CTkLabel(dframe, text="Sub-dendrogram",
                     font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(6, 0))
        self.d_fig = plt.Figure(figsize=(6.5, 4.5), tight_layout=True)
        self.d_ax = self.d_fig.add_subplot(111)
        self.d_canvas = FigureCanvasTkAgg(self.d_fig, master=dframe)
        attach_fig_toolbar(self.d_canvas, dframe)
        self.d_canvas.get_tk_widget().pack(fill="both", expand=True,
                                           padx=4, pady=(0, 4))

        # Slider.
        sframe = ctk.CTkFrame(body, fg_color="transparent")
        sframe.grid(row=0, column=1, sticky="ns", padx=4)
        ctk.CTkLabel(sframe, text="cut").pack(side="top")
        # tk.Scale (not ttk.Scale) kept here -- ``resolution`` and
        # ``showvalue`` have no clean CTkSlider equivalent.
        self.threshold_scale = tk.Scale(
            sframe, from_=self._zmax, to=0.0,
            resolution=max(self._zmax / 1000.0, 1e-6),
            orient=tk.VERTICAL, length=320, showvalue=False,
            command=self._on_slider)
        self._slider_user_driven = False
        self.threshold_scale.set(self._T)
        self._slider_user_driven = True
        self.threshold_scale.pack(side="top", fill="y", expand=True)
        apply_dark_to_tk_widget(self.threshold_scale)
        self.threshold_readout = tk.StringVar(value=f"{self._T:.3f}")
        ctk.CTkLabel(sframe, textvariable=self.threshold_readout,
                     width=64, anchor="center").pack(side="top")

        # Spatial.
        sp_frame = ctk.CTkFrame(body)
        sp_frame.grid(row=0, column=2, sticky="nsew")
        ctk.CTkLabel(sp_frame, text="Sub-spatial (cluster colors)",
                     font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(6, 0))
        self.s_fig = plt.Figure(figsize=(5.5, 4.5), tight_layout=True)
        self.s_ax = self.s_fig.add_subplot(111)
        self.s_canvas = FigureCanvasTkAgg(self.s_fig, master=sp_frame)
        attach_fig_toolbar(self.s_canvas, sp_frame)
        self.s_canvas.get_tk_widget().pack(fill="both", expand=True,
                                           padx=4, pady=(0, 4))

    def _on_slider(self, raw: str) -> None:
        if not self._slider_user_driven:
            return
        try:
            self._T = float(raw)
        except ValueError:
            return
        self._render()

    def _render(self) -> None:
        T = float(self._T)
        labels = fcluster(self._Z, t=T, criterion="distance")
        n_clusters = int(np.unique(labels).size)
        palette_hex = self._palette_resolver(n_clusters)

        # Build a per-label color table in visual (left-to-right) order so
        # the dendrogram, spatial map, and any future lookups all agree --
        # without depending on scipy's palette cycling.
        info = dendrogram(self._Z, no_plot=True, color_threshold=T,
                          above_threshold_color=ABOVE_CUT_COLOR)
        leaves = list(info["leaves"])
        seen: list[int] = []
        for leaf_idx in leaves:
            lbl = int(labels[leaf_idx])
            if lbl not in seen:
                seen.append(lbl)
        if len(palette_hex) < len(seen):
            palette_hex = list(palette_hex) + [palette_hex[-1]] * (
                len(seen) - len(palette_hex))
        try:
            label_to_color = {lbl: mpl.colors.to_hex(palette_hex[i])
                              for i, lbl in enumerate(seen)}
        except (ValueError, TypeError):
            label_to_color = {lbl: "#cccccc" for lbl in seen}

        # link_color_func gives explicit per-link control instead of trusting
        # scipy's color counter ordering.
        Z = self._Z
        n = Z.shape[0] + 1
        leaf_of_node: dict[int, int] = {}
        for i in range(n - 1):
            a = int(Z[i, 0])
            leaf_of_node[n + i] = a if a < n else leaf_of_node[a]

        def link_color_func(link_id: int) -> str:
            i = link_id - n
            if i < 0 or i >= n - 1 or Z[i, 2] >= T:
                return ABOVE_CUT_COLOR
            return label_to_color.get(int(labels[leaf_of_node[link_id]]),
                                      ABOVE_CUT_COLOR)

        # --- Dendrogram ---
        ax = self.d_ax
        ax.clear()
        dendrogram(self._Z, ax=ax,
                   link_color_func=link_color_func,
                   above_threshold_color=ABOVE_CUT_COLOR,
                   no_labels=True)
        ax.axhline(T, linestyle="--", linewidth=1.5, color="black")
        ax.set_ylabel("1 - r  (linkage distance)")
        ax.set_xlabel(f"{self._Z.shape[0] + 1} ROIs ({n_clusters} clusters)")
        ax.set_title(f"sub-cut @ {T:.3f}  ({T / self._zmax:.2f} x max)")
        self.d_canvas.draw_idle()
        self.threshold_readout.set(f"{T:.3f}")

        # --- Spatial ---
        N = n
        roi_rgb = np.zeros((N, 3))
        for leaf_idx in range(N):
            hex_color = label_to_color.get(int(labels[leaf_idx]),
                                           ABOVE_CUT_COLOR)
            roi_rgb[leaf_idx, :] = mpl.colors.to_rgb(hex_color)

        if self._stat is None or len(self._stat) != N:
            self.s_ax.clear()
            self.s_ax.set_axis_off()
            self.s_ax.text(0.5, 0.5,
                           "Spatial unavailable (stat / dF/F mismatch).",
                           ha="center", va="center",
                           transform=self.s_ax.transAxes)
            self.s_canvas.draw_idle()
            return

        img = spatial_image(self._stat, self._Lx, self._Ly, roi_rgb)
        render_spatial_map(self.s_ax, img, self._Lx, self._Ly,
                           self._pix_to_um, f"{n_clusters} sub-clusters")
        self.s_canvas.draw_idle()
