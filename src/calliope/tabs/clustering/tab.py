"""calliope.tabs.clustering.tab - Tab 6: hierarchical clustering of
ROI traces.

What this tab does
------------------
Cells in a slice are not independent -- groups of them have tightly
correlated firing patterns ("functional ensembles"). Tab 6 finds
those ensembles by:

1. Computing the **pairwise correlation distance** ``1 - Pearson r``
   between every pair of ROI dF/F traces (``scipy.spatial.distance.
   pdist`` with metric="correlation").
2. Running **average-linkage hierarchical agglomerative clustering**
   on that distance vector (``scipy.cluster.hierarchy.linkage``).
3. **Cutting the tree** at a user-selectable height to produce flat
   cluster labels (``fcluster``).
4. Rendering a **dendrogram** + a **spatial map** of the FOV with
   ROIs coloured by cluster. The two stay in sync as the user slides
   the threshold.

The colour palette can be a built-in seaborn / matplotlib palette
(categorical or continuous) or a per-cluster hand-picked set via a
small colour-picker dialog.

What the user sees
------------------
- Path / prefix entry + Browse...
- "Run analysis" button: kicks off the worker that computes
  pdist + linkage (seconds for typical recordings).
- A vertical slider next to the dendrogram for manual threshold
  adjustment; "Manual threshold" toggle to switch between manual and
  auto modes (auto picks a height that lands ~4-5 clusters).
- Palette dropdown + "Per-cluster colors..." dialog.
- Export buttons:
    * "Export *_rois.npy"  -> writes one ROI list per cluster into
      ``<prefix>cluster_results/gui_recluster/Ci_rois.npy``. Tab 7
      reads those files for cluster x cluster cross-correlation.
    * "Reload clusters" -> restore a previously-saved linkage +
      threshold without recomputing.
    * "Recluster within branch" -> open the picked branch in a
      fresh window with its own dendrogram so a fat cluster can be
      sub-divided.

What gets written
-----------------
- ``<prefix>cluster_results/gui_recluster/Ci_rois.npy`` -- per-
  cluster ROI lists in Suite2p ROI ids.
- ``<prefix>cluster_results/gui_recluster/linkage.npy`` and
  ``threshold_used.npy`` -- so "Reload clusters" can restore the
  state.
- ``Clusters`` sheet of ``calliope_summary.xlsx``.

Workflow
--------
1. Pick a Suite2p plane0 folder.
2. Click "Run analysis":
       - Loads dF/F (default prefix r0p7_filtered_).
       - Computes the pairwise correlation-distance linkage
         (1 - Pearson r between every pair of ROIs).
       - Picks a starting cut (auto target 4-5 clusters) and renders
         the dendrogram + cluster-colored spatial map.
3. Adjust the cut:
       - "Manual threshold" toggle ON -> the vertical slider next to
         the dendrogram drives the cut (the dashed line on the
         dendrogram tracks the slider). Spatial map recolors live.
       - Toggle OFF -> auto-threshold (same logic as
         clustering_cmap.auto_choose_threshold).
4. Recolor:
       - Pick a palette from the dropdown (categorical or continuous).
       - Or click "Per-cluster colors..." to define one hex color per
         cluster manually; this overrides the dropdown until reset.
5. "Export *_rois.npy" writes one ROI list per cluster into the
   r0p7_*cluster_results/gui_recluster/ folder so the Cross-correlation
   tab (and other downstream tools) can pick them up.

Cross-correlation lives on its own tab (see ``crosscorrelation_tab.py``).

The whole thing also runs as a standalone window via `python clustering_tab.py`.
"""

from __future__ import annotations

import queue
import sys
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, ttk
from typing import Optional

import matplotlib
matplotlib.use("TkAgg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk,
)
from scipy.cluster.hierarchy import (
    dendrogram,
    fcluster,
    linkage,
)
from scipy.spatial.distance import pdist

from .logic import clustering_cmap as cmap_mod
from .logic import summary_writer
from .logic import utils

from ...gui_common import format_roi_indices


DEFAULT_PREFIX = "r0p7_filtered_"
DEFAULT_PALETTE = "tab10"
ABOVE_CUT_COLOR = "gray"
EXPORT_SUBDIR = "gui_recluster"
POLL_MS = 80


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------


def _attach_fig_toolbar(canvas: FigureCanvasTkAgg,
                        parent_frame,
                        data_basename: str = "plot_data"
                        ) -> NavigationToolbar2Tk:
    """Attach a matplotlib navigation toolbar (pan / zoom / Save Figure)
    above ``canvas`` inside ``parent_frame``. Caller must pack/grid the
    canvas widget AFTER this call so the toolbar lands above it.

    A "Save data..." button is added next to the toolbar; it dumps the
    underlying Line2D / image data to CSV via
    ``plot_data_export.save_figure_data``."""
    from ... import plot_data_export
    tb_frame = ttk.Frame(parent_frame)
    tb_frame.pack(side="top", fill="x")
    tb = NavigationToolbar2Tk(canvas, tb_frame, pack_toolbar=False)
    tb.update()
    tb.pack(side="left", fill="x")
    ttk.Button(
        tb_frame, text="Save data...",
        command=lambda c=canvas, p=tb_frame, b=data_basename:
            plot_data_export.save_figure_data(c.figure, p, b),
    ).pack(side="left", padx=(8, 0))
    return tb


def _correlation_linkage(dff: np.ndarray, method: str = "average") -> np.ndarray:
    """Linkage on the (1 - Pearson r) distance between every pair of ROIs.

    `pdist(metric="correlation")` returns 1 - r for every column pair, so the
    resulting linkage is exactly the cross-correlation hierarchical clustering
    referenced in the GUI prompt. Default to `average` linkage because Ward's
    method assumes a Euclidean distance.
    """
    dff_z = (dff - np.mean(dff, axis=0)) / (np.std(dff, axis=0) + 1e-8)
    dist = pdist(dff_z.T, metric="correlation")
    return linkage(dist, method=method)


def _load_filter_mask(plane0: Path) -> Optional[np.ndarray]:
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


def _load_filtered_dff(plane0: Path, prefix: str):
    """Open ``<prefix>dff.memmap.float32`` against the cell-filter mask.

    For a "filtered" prefix (e.g. ``r0p7_filtered_``) the memmap was written
    at the size of the cell-filter keep mask, so we need that exact mask to
    interpret the column count. For an unfiltered prefix, the memmap holds
    every Suite2p ROI.

    Returns (dff_array, T, N_kept, keep_mask). ``keep_mask`` is ``None`` when
    the prefix is unfiltered.
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


def _stat_for_prefix(plane0: Path, prefix: str):
    """Stat list aligned with the columns of the dF/F memmap.

    For filtered prefixes, restrict ``stat`` to the cell-filter keep mask so
    the i-th stat entry matches the i-th column of dF/F.
    """
    full_stat = list(np.load(plane0 / "stat.npy", allow_pickle=True))
    if "filtered" in prefix.split("_"):
        mask = _load_filter_mask(plane0)
        if mask is not None:
            return ([s for s, keep in zip(full_stat, mask) if keep],
                    np.where(mask)[0])
    return full_stat, None


def _spatial_image(stat, Lx, Ly, roi_rgb: np.ndarray) -> np.ndarray:
    """Paint a per-ROI RGB array onto the FOV (NaN background)."""
    R = utils.paint_spatial(roi_rgb[:, 0], stat, Ly, Lx)
    G = utils.paint_spatial(roi_rgb[:, 1], stat, Ly, Lx)
    B = utils.paint_spatial(roi_rgb[:, 2], stat, Ly, Lx)
    img = np.dstack([R, G, B])
    coverage = utils.paint_spatial(np.ones(len(stat)), stat, Ly, Lx)
    img[coverage == 0] = np.nan
    return img


# ---------------------------------------------------------------------------
# Per-cluster custom-color dialog
# ---------------------------------------------------------------------------


class CustomColorDialog(tk.Toplevel):
    """Modal dialog to override the per-cluster color list.

    `initial_colors` is a list of hex strings (length = current cluster count).
    Clicking a swatch opens askcolor; OK writes the new list to `result`.
    """

    def __init__(self, master, initial_colors: list[str]) -> None:
        super().__init__(master)
        self.title("Per-cluster colors")
        self.transient(master)
        self.resizable(False, False)
        self._colors = list(initial_colors)
        self.result: Optional[list[str]] = None
        self._swatches: list[tk.Button] = []
        self._build_ui()
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._cancel)

    def _build_ui(self) -> None:
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text="Click a swatch to pick that cluster's color.",
                  foreground="gray").grid(row=0, column=0, columnspan=3,
                                          sticky="w", pady=(0, 6))
        for i, hex_color in enumerate(self._colors):
            ttk.Label(frm, text=f"Cluster {i + 1}").grid(
                row=i + 1, column=0, sticky="w", padx=(0, 8), pady=2)
            btn = tk.Button(frm, width=6, bg=hex_color,
                            relief="ridge",
                            command=lambda idx=i: self._pick(idx))
            btn.grid(row=i + 1, column=1, sticky="w", pady=2)
            self._swatches.append(btn)
            ttk.Label(frm, textvariable=tk.StringVar(value=hex_color)).grid(
                row=i + 1, column=2, sticky="w", padx=(8, 0))

        bar = ttk.Frame(self, padding=(10, 0, 10, 10))
        bar.pack(fill="x")
        ttk.Button(bar, text="Cancel", command=self._cancel).pack(side="right")
        ttk.Button(bar, text="OK",
                   command=self._ok).pack(side="right", padx=(0, 6))

    def _pick(self, idx: int) -> None:
        rgb, hex_color = colorchooser.askcolor(
            color=self._colors[idx], parent=self,
            title=f"Pick color for cluster {idx + 1}")
        if hex_color:
            self._colors[idx] = hex_color
            self._swatches[idx].configure(bg=hex_color)

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


class ReclusterWindow(tk.Toplevel):
    """Read-only sub-tree view: dendrogram + spatial map for a chosen union
    of branches, with its own threshold slider. Does not modify the parent
    tab's state in any way."""

    def __init__(self, master, *, Z: np.ndarray, stat, Lx: int, Ly: int,
                 init_threshold: float, zmax: float,
                 palette_resolver, branch_label: str, n_rois: int) -> None:
        super().__init__(master)
        self.title(f"Recluster - branch(es) {branch_label}  "
                   f"({n_rois} ROIs)")
        self.geometry("1200x600")

        self._Z = Z
        self._stat = stat
        self._Lx = int(Lx)
        self._Ly = int(Ly)
        self._zmax = float(zmax)
        self._T = float(init_threshold)
        self._palette_resolver = palette_resolver
        self._slider_user_driven = True

        self._build_ui()
        self._render()

    def _build_ui(self) -> None:
        body = ttk.Frame(self, padding=6)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=0)
        body.columnconfigure(2, weight=2)
        body.rowconfigure(0, weight=1)

        # Dendrogram.
        dframe = ttk.LabelFrame(body, text="Sub-dendrogram", padding=4)
        dframe.grid(row=0, column=0, sticky="nsew")
        self.d_fig = plt.Figure(figsize=(6.5, 4.5), tight_layout=True)
        self.d_ax = self.d_fig.add_subplot(111)
        self.d_canvas = FigureCanvasTkAgg(self.d_fig, master=dframe)
        _attach_fig_toolbar(self.d_canvas, dframe)
        self.d_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Slider.
        sframe = ttk.Frame(body)
        sframe.grid(row=0, column=1, sticky="ns", padx=4)
        ttk.Label(sframe, text="cut").pack(side="top")
        self.threshold_scale = tk.Scale(
            sframe, from_=self._zmax, to=0.0,
            resolution=max(self._zmax / 1000.0, 1e-6),
            orient=tk.VERTICAL, length=320, showvalue=False,
            command=self._on_slider)
        self._slider_user_driven = False
        self.threshold_scale.set(self._T)
        self._slider_user_driven = True
        self.threshold_scale.pack(side="top", fill="y", expand=True)
        self.threshold_readout = tk.StringVar(value=f"{self._T:.3f}")
        ttk.Label(sframe, textvariable=self.threshold_readout,
                  width=8, anchor="center").pack(side="top")

        # Spatial.
        sp_frame = ttk.LabelFrame(body, text="Sub-spatial (cluster colors)",
                                  padding=4)
        sp_frame.grid(row=0, column=2, sticky="nsew")
        self.s_fig = plt.Figure(figsize=(5.5, 4.5), tight_layout=True)
        self.s_ax = self.s_fig.add_subplot(111)
        self.s_canvas = FigureCanvasTkAgg(self.s_fig, master=sp_frame)
        _attach_fig_toolbar(self.s_canvas, sp_frame)
        self.s_canvas.get_tk_widget().pack(fill="both", expand=True)

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
        # the dendrogram, spatial map, and any future lookups all agree —
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

        img = _spatial_image(self._stat, self._Lx, self._Ly, roi_rgb)
        ax = self.s_ax
        ax.clear()
        ax.imshow(img, origin="upper", aspect="equal")
        ax.set_title(f"{n_clusters} sub-clusters")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        self.s_canvas.draw_idle()


# ---------------------------------------------------------------------------
# Main tab
# ---------------------------------------------------------------------------


class ClusteringTab(ttk.Frame):
    """Cross-correlation hierarchical clustering tab."""

    def __init__(self, master, state=None) -> None:
        super().__init__(master, padding=10)
        self.state = state

        self._plane0: Optional[Path] = None
        self._prefix = DEFAULT_PREFIX

        # Analysis cache (filled on Run analysis).
        self._dff: Optional[np.ndarray] = None
        self._Z: Optional[np.ndarray] = None
        self._stat = None
        self._Lx: int = 0
        self._Ly: int = 0
        self._zmax: float = 1.0


        # Color state.
        self._palette_name: str = DEFAULT_PALETTE
        self._custom_colors: Optional[list[str]] = None  # overrides dropdown

        # Threshold state.
        self._auto_threshold: float = 0.7  # fraction of max
        self._manual_T: float = 0.7  # absolute distance

        # Visual cluster index -> raw fcluster label, refreshed every render
        # so the cluster picker matches what the user sees in the dendrogram.
        self._visual_to_label: dict[int, int] = {}
        # Visual cluster index -> hex color shown in the dendrogram, used
        # to paint the dropdown menu items and seed the per-cluster dialog.
        self._visual_to_color: dict[int, str] = {}
        # Per-leaf (= per-ROI) hex color from the same dendrogram pass,
        # used to paint the spatial map so it always agrees with the
        # dendrogram and the menu.
        self._leaf_colors: dict[int, str] = {}

        # Cycles through alternative palettes for recluster inspection
        # windows so their colors differ from the parent (and from each
        # other), since only the parent's color mapping is meaningful.
        self._recluster_palette_idx: int = 0

        # Worker queue.
        self._q: queue.Queue = queue.Queue()
        self._worker: Optional[threading.Thread] = None

        self._slider_user_driven = True  # gate slider events during programmatic sets
        self._summary_after_id: Optional[str] = None  # debounce id for auto-write

        self._build_ui()
        self.after(POLL_MS, self._drain_queue)

        # If used inside pipeline_gui, subscribe to plane0 broadcasts.
        if state is not None:
            try:
                state.subscribe_plane0(self._on_plane0_broadcast)
                state.subscribe_lowpass_ready(self._on_plane0_broadcast)
                if getattr(state, "lowpass_plane0", None) is not None:
                    self._on_plane0_broadcast(state.lowpass_plane0)
                elif getattr(state, "plane0", None) is not None:
                    self._on_plane0_broadcast(state.plane0)
            except Exception:
                pass

    # -- UI ----------------------------------------------------------------

    def _build_ui(self) -> None:
        head = ttk.LabelFrame(
            self, text="Cross-correlation clustering "
                       "(1 - Pearson r, hierarchical, average linkage)",
            padding=8)
        head.pack(fill="x", pady=(0, 6))

        row1 = ttk.Frame(head); row1.pack(fill="x", pady=2)
        ttk.Label(row1, text="plane0:").pack(side="left")
        self.path_var = tk.StringVar(value="")
        ttk.Entry(row1, textvariable=self.path_var, width=70).pack(
            side="left", padx=(4, 4), fill="x", expand=True)
        ttk.Button(row1, text="Browse...",
                   command=self._on_browse).pack(side="left")

        row2 = ttk.Frame(head); row2.pack(fill="x", pady=2)
        self.run_btn = ttk.Button(
            row2, text="Run analysis", command=self._on_run, state="disabled")
        self.run_btn.pack(side="left")
        self.reload_btn = ttk.Button(
            row2, text="Reload clusters",
            command=self._on_reload_clusters, state="disabled")
        self.reload_btn.pack(side="left", padx=(6, 0))
        self.progress = ttk.Progressbar(row2, mode="indeterminate", length=140)
        self.progress.pack(side="left", padx=8)
        self.status_var = tk.StringVar(value="Pick a plane0 folder.")
        ttk.Label(row2, textvariable=self.status_var,
                  font=("", 9, "italic")).pack(side="left", fill="x", expand=True)

        # Recluster controls: drill into one or more clusters (branches) at
        # the current cut and re-cluster only those ROIs at a finer threshold.
        row3 = ttk.Frame(head); row3.pack(fill="x", pady=2)
        ttk.Label(row3, text="Recluster branch(es):").pack(side="left")
        self._cluster_vars: dict[int, tk.BooleanVar] = {}
        self.cluster_menu = tk.Menu(self, tearoff=False)
        self.cluster_menu_btn = ttk.Menubutton(
            row3, text="Pick clusters...", width=24, state="disabled")
        self.cluster_menu_btn.config(menu=self.cluster_menu)
        self.cluster_menu_btn.pack(side="left", padx=(8, 0))

        ttk.Label(row3, text="new threshold (1 - r):").pack(
            side="left", padx=(8, 2))
        self.recluster_thr_var = tk.StringVar(value="")
        ttk.Entry(row3, textvariable=self.recluster_thr_var, width=8).pack(
            side="left")
        self.recluster_btn = ttk.Button(
            row3, text="Recluster (new window)", command=self._on_recluster,
            state="disabled")
        self.recluster_btn.pack(side="left", padx=(8, 0))

        # ROI list of the currently-picked clusters (Suite2p ROI ids).
        # Read-only display + Copy button so the user can paste into Tab 4
        # / Tab 5 manual-ROI entries or external scripts. Auto-refreshes on
        # every menu toggle and every threshold/palette change.
        row3b = ttk.Frame(head); row3b.pack(fill="x", pady=2)
        ttk.Label(row3b, text="ROIs in picked clusters (Suite2p ids):").pack(
            side="left")
        self.roi_list_var = tk.StringVar(value="")
        self.roi_list_entry = ttk.Entry(
            row3b, textvariable=self.roi_list_var, state="readonly")
        self.roi_list_entry.pack(side="left", fill="x", expand=True,
                                 padx=(8, 4))
        self.copy_rois_btn = ttk.Button(
            row3b, text="Copy", command=self._on_copy_roi_list,
            state="disabled")
        self.copy_rois_btn.pack(side="left")

        # Body: dendrogram + slider + spatial.
        body = ttk.Frame(self); body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=0)
        body.columnconfigure(2, weight=2)
        body.rowconfigure(0, weight=1)

        # Dendrogram canvas.
        dframe = ttk.LabelFrame(body, text="Dendrogram", padding=4)
        dframe.grid(row=0, column=0, sticky="nsew")
        self.d_fig = plt.Figure(figsize=(6.5, 4.5), tight_layout=True)
        self.d_ax = self.d_fig.add_subplot(111)
        self._placeholder(self.d_ax, "Run analysis to populate.")
        self.d_canvas = FigureCanvasTkAgg(self.d_fig, master=dframe)
        _attach_fig_toolbar(self.d_canvas, dframe)
        self.d_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Vertical slider for the cut.
        sframe = ttk.Frame(body)
        sframe.grid(row=0, column=1, sticky="ns", padx=4)
        ttk.Label(sframe, text="cut").pack(side="top")
        self.threshold_scale = tk.Scale(
            sframe, from_=1.0, to=0.0, resolution=0.001,
            orient=tk.VERTICAL, length=320, showvalue=False,
            command=self._on_slider, state="disabled")
        self.threshold_scale.pack(side="top", fill="y", expand=True)
        self.threshold_readout = tk.StringVar(value="-")
        ttk.Label(sframe, textvariable=self.threshold_readout,
                  width=8, anchor="center").pack(side="top")

        # Spatial canvas.
        sp_frame = ttk.LabelFrame(body, text="Spatial (cluster colors)",
                                  padding=4)
        sp_frame.grid(row=0, column=2, sticky="nsew")
        self.s_fig = plt.Figure(figsize=(5.5, 4.5), tight_layout=True)
        self.s_ax = self.s_fig.add_subplot(111)
        self._placeholder(self.s_ax, "Run analysis to populate.")
        self.s_canvas = FigureCanvasTkAgg(self.s_fig, master=sp_frame)
        _attach_fig_toolbar(self.s_canvas, sp_frame)
        self.s_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Bottom controls.
        ctl = ttk.Frame(self, padding=(0, 6, 0, 0))
        ctl.pack(fill="x")

        self.manual_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            ctl, text="Manual threshold", variable=self.manual_var,
            command=self._on_manual_toggle).pack(side="left")

        ttk.Label(ctl, text="   palette:").pack(side="left")
        self.palette_var = tk.StringVar(value=DEFAULT_PALETTE)
        palette_box = ttk.Combobox(
            ctl, textvariable=self.palette_var, state="readonly", width=12,
            values=list(cmap_mod.AVAILABLE_PALETTES))
        palette_box.pack(side="left", padx=4)
        palette_box.bind("<<ComboboxSelected>>", self._on_palette_change)

        ttk.Button(ctl, text="Per-cluster colors...",
                   command=self._on_custom_colors).pack(side="left", padx=4)
        ttk.Button(ctl, text="Reset palette",
                   command=self._on_reset_palette).pack(side="left", padx=4)

        ttk.Label(ctl, text="prefix:").pack(side="left", padx=(20, 2))
        self.prefix_var = tk.StringVar(value=DEFAULT_PREFIX)
        ttk.Entry(ctl, textvariable=self.prefix_var, width=18).pack(
            side="left")

        self.export_btn = ttk.Button(
            ctl, text="Export *_rois.npy",
            command=self._on_export, state="disabled")
        self.export_btn.pack(side="right")
        self.summary_btn = ttk.Button(
            ctl, text="Save summary",
            command=self._on_save_summary, state="disabled")
        self.summary_btn.pack(side="right", padx=(0, 6))

    # -- Helpers -----------------------------------------------------------

    def _placeholder(self, ax, text: str) -> None:
        ax.clear()
        ax.set_axis_off()
        ax.text(0.5, 0.5, text, ha="center", va="center",
                transform=ax.transAxes)

    def _on_plane0_broadcast(self, plane0) -> None:
        if plane0 is None:
            return
        self.path_var.set(str(plane0))
        self._set_plane0(Path(plane0))

    def _on_browse(self) -> None:
        path = filedialog.askdirectory(
            title="Select Suite2p plane0 folder",
            initialdir=self.path_var.get() or str(Path.home()))
        if not path:
            return
        self.path_var.set(path)
        self._set_plane0(Path(path))

    def _set_plane0(self, plane0: Path) -> None:
        self._plane0 = plane0
        ok = self._inputs_ready(plane0, self.prefix_var.get())
        self.run_btn.config(state="normal" if ok else "disabled")
        self._refresh_reload_btn()
        if ok:
            self.status_var.set(f"Ready. ({plane0})")
        else:
            self.status_var.set(
                f"Missing dF/F memmaps with prefix {self.prefix_var.get()!r}.")

    def _inputs_ready(self, plane0: Path, prefix: str) -> bool:
        for name in ("F.npy", "stat.npy", "ops.npy",
                     f"{prefix}dff.memmap.float32"):
            if not (plane0 / name).exists():
                return False
        return True

    def _existing_cluster_dir(self, plane0: Path, prefix: str) -> Path:
        return plane0 / f"{prefix}cluster_results" / EXPORT_SUBDIR

    def _has_existing_clusters(self, plane0: Path, prefix: str) -> bool:
        d = self._existing_cluster_dir(plane0, prefix)
        return ((d / "linkage.npy").exists()
                and (d / "threshold_used.npy").exists())

    def _refresh_reload_btn(self) -> None:
        if self._plane0 is None:
            self.reload_btn.config(state="disabled")
            return
        prefix = self.prefix_var.get().strip() or DEFAULT_PREFIX
        ok = (self._inputs_ready(self._plane0, prefix)
              and self._has_existing_clusters(self._plane0, prefix))
        self.reload_btn.config(state="normal" if ok else "disabled")

    # -- Run worker --------------------------------------------------------

    def _on_run(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            messagebox.showinfo("Busy", "Analysis already running.")
            return
        plane0 = Path(self.path_var.get())
        prefix = self.prefix_var.get().strip() or DEFAULT_PREFIX
        if not self._inputs_ready(plane0, prefix):
            messagebox.showerror("Not ready", f"plane0 missing required files: {plane0}")
            return

        self._prefix = prefix
        self.run_btn.config(state="disabled")
        self.progress.start(12)
        self.status_var.set("Computing linkage...")

        def worker():
            try:
                payload = self._compute(plane0, prefix)
                self._q.put(("done", payload))
            except Exception as e:
                self._q.put(("error", f"{e}\n{traceback.format_exc()}"))

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    def _compute(self, plane0: Path, prefix: str) -> dict:
        dff_mm, T, N, _ = _load_filtered_dff(plane0, prefix)
        dff = dff_mm
        Z = _correlation_linkage(dff)
        stat, _ = _stat_for_prefix(plane0, prefix)
        ops = np.load(plane0 / "ops.npy", allow_pickle=True).item()
        Lx, Ly = int(ops["Lx"]), int(ops["Ly"])
        auto_frac = cmap_mod.auto_choose_threshold(Z, target_counts=(4, 5))
        return {
            "Z": Z, "dff": dff, "dff_shape": dff.shape, "stat": stat,
            "Lx": Lx, "Ly": Ly, "auto_frac": auto_frac,
            "T": T, "N": N,
        }

    def _on_reload_clusters(self) -> None:
        """Restore linkage + threshold from a previous Export *_rois.npy run.

        Skips the (expensive) correlation-linkage computation and snaps the
        slider to the saved threshold in manual mode. dF/F + stat + ops are
        still loaded fresh from disk so reclustering and the spatial map
        keep working.
        """
        if self._worker is not None and self._worker.is_alive():
            messagebox.showinfo("Busy", "Analysis already running.")
            return
        plane0 = Path(self.path_var.get())
        prefix = self.prefix_var.get().strip() or DEFAULT_PREFIX
        if not self._inputs_ready(plane0, prefix):
            messagebox.showerror(
                "Not ready", f"plane0 missing required files: {plane0}")
            return
        if not self._has_existing_clusters(plane0, prefix):
            messagebox.showerror(
                "No saved clusters",
                f"Expected {self._existing_cluster_dir(plane0, prefix)} "
                f"with linkage.npy + threshold_used.npy. Run analysis "
                f"first and click 'Export *_rois.npy'.")
            return

        self._prefix = prefix
        self.run_btn.config(state="disabled")
        self.reload_btn.config(state="disabled")
        self.progress.start(12)
        self.status_var.set("Reloading saved clusters...")

        def worker():
            try:
                payload = self._reload_compute(plane0, prefix)
                self._q.put(("reloaded", payload))
            except Exception as e:
                self._q.put(("error", f"{e}\n{traceback.format_exc()}"))

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    def _reload_compute(self, plane0: Path, prefix: str) -> dict:
        cluster_dir = self._existing_cluster_dir(plane0, prefix)
        Z = np.load(cluster_dir / "linkage.npy")
        thr = float(np.asarray(np.load(cluster_dir / "threshold_used.npy"),
                               dtype=float).ravel()[0])
        dff_mm, T, N, _ = _load_filtered_dff(plane0, prefix)
        # The loaded linkage was computed on N_kept ROIs; if the cell-filter
        # mask has changed since the export the shapes won't agree and any
        # downstream recolouring would be silently wrong.
        n_leaves = int(Z.shape[0]) + 1
        if n_leaves != int(N):
            raise ValueError(
                f"Saved linkage has {n_leaves} leaves but the current "
                f"dF/F memmap has {N} columns. Re-run analysis instead "
                f"of reloading.")
        stat, _ = _stat_for_prefix(plane0, prefix)
        ops = np.load(plane0 / "ops.npy", allow_pickle=True).item()
        Lx, Ly = int(ops["Lx"]), int(ops["Ly"])
        zmax = float(np.max(Z[:, 2]))
        return {
            "Z": Z, "dff": dff_mm, "dff_shape": dff_mm.shape, "stat": stat,
            "Lx": Lx, "Ly": Ly, "saved_threshold": thr, "zmax": zmax,
            "T": T, "N": N, "cluster_dir": cluster_dir,
        }

    def _drain_queue(self) -> None:
        try:
            while True:
                kind, payload = self._q.get_nowait()
                if kind == "done":
                    self._on_done(payload)
                elif kind == "reloaded":
                    self._on_reloaded(payload)
                elif kind == "error":
                    self.progress.stop()
                    self.run_btn.config(state="normal")
                    self._refresh_reload_btn()
                    self.status_var.set("Analysis failed.")
                    messagebox.showerror(
                        "Analysis failed", payload.split("\n", 1)[0])
        except queue.Empty:
            pass
        self.after(POLL_MS, self._drain_queue)

    def _on_done(self, data: dict) -> None:
        self.progress.stop()
        self.run_btn.config(state="normal")

        self._Z = data["Z"]
        self._stat = data["stat"]
        self._dff = data["dff"]
        self._Lx = data["Lx"]
        self._Ly = data["Ly"]
        self._zmax = float(np.max(self._Z[:, 2]))

        auto_frac = float(data["auto_frac"])
        self._auto_threshold = auto_frac
        self._manual_T = auto_frac * self._zmax

        # Calibrate slider range to current linkage.
        self._slider_user_driven = False
        self.threshold_scale.config(from_=self._zmax, to=0.0,
                                    resolution=max(self._zmax / 1000.0, 1e-6))
        self.threshold_scale.set(self._manual_T)
        self._slider_user_driven = True

        self._custom_colors = None  # palette resets to dropdown choice
        self.export_btn.config(state="normal")
        self.summary_btn.config(state="normal")
        self.recluster_btn.config(state="normal")
        self.cluster_menu_btn.config(state="normal")
        # Seed the recluster threshold with half the current cut.
        self.recluster_thr_var.set(f"{self._manual_T / 2.0:.3f}")

        self._render_all()
        try:
            self._write_summary()
        except Exception as e:
            print(f"[GUI] cluster summary write failed: {e}")

        n_clusters = int(np.unique(
            fcluster(self._Z, t=self._current_threshold(),
                     criterion="distance")).size)
        self.status_var.set(
            f"OK. N_rois={data['N']}  T={data['T']}  "
            f"auto cut={auto_frac:.2f}xmax  -> {n_clusters} clusters. "
            f"Toggle 'Manual threshold' + slider to tune.")
        self._refresh_reload_btn()

    def _on_reloaded(self, data: dict) -> None:
        self.progress.stop()
        self.run_btn.config(state="normal")

        self._Z = data["Z"]
        self._stat = data["stat"]
        self._dff = data["dff"]
        self._Lx = data["Lx"]
        self._Ly = data["Ly"]
        self._zmax = float(data["zmax"])

        saved_T = float(data["saved_threshold"])
        # Clamp to the slider range so an out-of-band saved threshold (after
        # editing the dF/F memmap, etc.) doesn't wedge the slider.
        saved_T = max(0.0, min(self._zmax, saved_T))
        self._manual_T = saved_T
        self._auto_threshold = (saved_T / self._zmax) if self._zmax > 0 else 0.7

        # Restored runs land in manual mode at the saved cut so the slider
        # tracks the same threshold the user committed to disk.
        self.manual_var.set(True)
        self._slider_user_driven = False
        self.threshold_scale.config(from_=self._zmax, to=0.0,
                                    resolution=max(self._zmax / 1000.0, 1e-6),
                                    state="normal")
        self.threshold_scale.set(self._manual_T)
        self._slider_user_driven = True

        self._custom_colors = None
        self.export_btn.config(state="normal")
        self.summary_btn.config(state="normal")
        self.recluster_btn.config(state="normal")
        self.cluster_menu_btn.config(state="normal")
        self.recluster_thr_var.set(f"{self._manual_T / 2.0:.3f}")

        self._render_all()
        try:
            self._write_summary()
        except Exception as e:
            print(f"[GUI] cluster summary write failed: {e}")

        n_clusters = int(np.unique(
            fcluster(self._Z, t=self._current_threshold(),
                     criterion="distance")).size)
        cluster_dir = data.get("cluster_dir")
        loc = f" from {cluster_dir}" if cluster_dir else ""
        self.status_var.set(
            f"Reloaded{loc}. N_rois={data['N']}  T={data['T']}  "
            f"saved cut={saved_T:.3f}  ({n_clusters} clusters).")
        self._refresh_reload_btn()

    # -- Threshold + palette -----------------------------------------------

    def _current_threshold(self) -> float:
        if self.manual_var.get():
            return float(self._manual_T)
        return float(self._auto_threshold) * float(self._zmax)

    def _on_manual_toggle(self) -> None:
        if self.manual_var.get():
            self.threshold_scale.config(state="normal")
            # Snap slider to the auto value as a starting point.
            self._slider_user_driven = False
            self.threshold_scale.set(self._auto_threshold * self._zmax)
            self._slider_user_driven = True
            self._manual_T = self._auto_threshold * self._zmax
        else:
            self.threshold_scale.config(state="disabled")
        self._render_all()
        self._schedule_summary_write()

    def _on_slider(self, raw: str) -> None:
        if not self._slider_user_driven:
            return
        try:
            self._manual_T = float(raw)
        except ValueError:
            return
        if self._Z is not None:
            self._render_all()
            self._schedule_summary_write()

    def _on_palette_change(self, *_):
        self._palette_name = self.palette_var.get() or DEFAULT_PALETTE
        self._custom_colors = None
        if self._Z is not None:
            self._render_all()
            self._schedule_summary_write()

    def _on_custom_colors(self) -> None:
        if self._Z is None:
            messagebox.showinfo("Run first",
                                "Run the analysis before picking colors.")
            return
        n = max(1, int(np.unique(
            fcluster(self._Z, t=self._current_threshold(),
                     criterion="distance")).size))
        # Source the initial swatches from the dendrogram's actual per-cluster
        # colors so what the user sees in the dialog matches what's painted
        # in the dendrogram and spatial map. Fall back to the palette resolve
        # only if the visual map hasn't been computed yet.
        if self._custom_colors:
            base = list(self._custom_colors)
        elif self._visual_to_color:
            base = [self._visual_to_color.get(i + 1, "#cccccc")
                    for i in range(n)]
        else:
            base = cmap_mod.resolve_palette(self._palette_name, n_colors=n)
        # Normalize length to current cluster count.
        if len(base) < n:
            base = base + [base[-1]] * (n - len(base))
        elif len(base) > n:
            base = base[:n]
        dlg = CustomColorDialog(self.winfo_toplevel(), base)
        self.wait_window(dlg)
        if dlg.result is not None:
            self._custom_colors = dlg.result
            self._render_all()
            self._schedule_summary_write()

    def _on_reset_palette(self) -> None:
        self._custom_colors = None
        if self._Z is not None:
            self._render_all()
            self._schedule_summary_write()

    # -- Recluster ---------------------------------------------------------

    def _compute_visual_cluster_map(self, T: float) -> dict[int, int]:
        """Map visual cluster index (1..n, left-to-right in the dendrogram)
        to the raw fcluster label so the picker matches what the user sees.

        Independent of the active color palette — only depends on tree
        traversal. Colors are assigned separately by ``_build_label_color_map``
        so we never trust scipy's ``set_link_color_palette`` cycling."""
        if self._Z is None:
            return {}
        info = dendrogram(self._Z, no_plot=True, color_threshold=T,
                          above_threshold_color=ABOVE_CUT_COLOR)
        leaves = list(info["leaves"])
        labels = fcluster(self._Z, t=T, criterion="distance")
        seen: list[int] = []
        for leaf_idx in leaves:
            lbl = int(labels[leaf_idx])
            if lbl not in seen:
                seen.append(lbl)
        return {i + 1: lbl for i, lbl in enumerate(seen)}

    def _build_label_color_map(self, n_clusters: int) -> dict[int, str]:
        """Map each fcluster label to the color the user expects for that
        cluster, in visual order. Custom colors win; otherwise the active
        palette is resolved to ``n_clusters`` slots and visual id ``v``
        receives slot ``v - 1``."""
        if self._custom_colors:
            colors = list(self._custom_colors)
        else:
            colors = cmap_mod.resolve_palette(
                self._palette_name, n_colors=max(1, n_clusters))
        if len(colors) < n_clusters:
            colors = colors + [colors[-1]] * (n_clusters - len(colors))
        colors = colors[:n_clusters]
        out: dict[int, str] = {}
        for vid, lbl in self._visual_to_label.items():
            idx = vid - 1
            if 0 <= idx < len(colors):
                try:
                    out[int(lbl)] = mpl.colors.to_hex(colors[idx])
                except (ValueError, TypeError):
                    out[int(lbl)] = "#cccccc"
        return out

    def _make_link_color_func(self, T: float, labels: np.ndarray,
                              label_to_color: dict[int, str]):
        """Return a ``link_color_func`` for ``dendrogram`` that paints each
        link with its descendant cluster's color from ``label_to_color``.
        Bypasses scipy's palette cycling so palette[v - 1] always lands on
        visual cluster Cv, regardless of tree shape."""
        Z = self._Z
        n = Z.shape[0] + 1

        # Pre-compute one descendant leaf per internal node so we can look
        # up the cluster label for any link in O(1).
        leaf_of_node: dict[int, int] = {}
        for i in range(n - 1):
            a = int(Z[i, 0])
            leaf_of_node[n + i] = a if a < n else leaf_of_node[a]

        above = ABOVE_CUT_COLOR

        def func(link_id: int) -> str:
            i = link_id - n
            if i < 0 or i >= n - 1:
                return above
            if Z[i, 2] >= T:
                return above
            leaf = leaf_of_node[link_id]
            return label_to_color.get(int(labels[leaf]), above)

        return func

    @staticmethod
    def _readable_fg(hex_color: str) -> str:
        """Pick black or white text for readability against ``hex_color``."""
        try:
            r, g, b = mpl.colors.to_rgb(hex_color)
        except (ValueError, TypeError):
            return "black"
        # ITU-R BT.601 luma; threshold tuned so saturated mid-tones still
        # pair with white text (matches matplotlib's tab10 expectations).
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        return "black" if luma > 0.6 else "white"

    def _rebuild_cluster_menu(self) -> None:
        """Refresh the dropdown so it lists C1..Cn in dendrogram order with
        each entry painted in the cluster's current color. Ticks for ids
        that still exist after a threshold change are preserved."""
        n = len(self._visual_to_label)
        prev = {k: v.get() for k, v in self._cluster_vars.items()}
        self.cluster_menu.delete(0, "end")
        self._cluster_vars.clear()
        for vid in range(1, n + 1):
            var = tk.BooleanVar(value=prev.get(vid, False))
            self._cluster_vars[vid] = var
            color = self._visual_to_color.get(vid, "#cccccc")
            fg = self._readable_fg(color)
            self.cluster_menu.add_checkbutton(
                label=f"C{vid}", variable=var,
                onvalue=True, offvalue=False,
                background=color, foreground=fg,
                activebackground=color, activeforeground=fg,
                selectcolor=fg,
                command=self._update_cluster_pick_label)
        if n > 0:
            self.cluster_menu.add_separator()
            self.cluster_menu.add_command(
                label="Select all", command=self._cluster_pick_all)
            self.cluster_menu.add_command(
                label="Clear", command=self._cluster_pick_clear)
        self._update_cluster_pick_label()

    def _picked_visual_ids(self) -> list[int]:
        return sorted(v for v, var in self._cluster_vars.items() if var.get())

    def _update_cluster_pick_label(self) -> None:
        picked = self._picked_visual_ids()
        if not picked:
            text = "Pick clusters..."
        elif len(picked) <= 4:
            text = ", ".join(f"C{c}" for c in picked)
        else:
            text = f"{len(picked)} picked  (C{picked[0]}..C{picked[-1]})"
        self.cluster_menu_btn.config(text=text)
        self._refresh_roi_list_var()

    def _picked_cluster_roi_ids(self) -> list[int]:
        """Suite2p ROI indices belonging to the currently-picked clusters,
        sorted ascending. Empty list if nothing is picked or analysis hasn't
        been run yet.
        """
        if self._Z is None:
            return []
        picked = self._picked_visual_ids()
        if not picked:
            return []
        T = self._current_threshold()
        labels = fcluster(self._Z, t=T, criterion="distance")
        target_labels = [self._visual_to_label.get(v) for v in picked]
        target_labels = [l for l in target_labels if l is not None]
        if not target_labels:
            return []
        local_mask = np.isin(
            labels, np.array(target_labels, dtype=labels.dtype))
        local_idx = np.where(local_mask)[0].astype(int)
        # Translate filtered-list positions to Suite2p ROI ids when the
        # active prefix is filtered (mirrors _export_clusters).
        if ("filtered" in self._prefix.split("_")
                and self._plane0 is not None):
            mask = _load_filter_mask(self._plane0)
            if mask is not None and int(mask.sum()) == len(labels):
                translator = np.where(
                    np.asarray(mask, dtype=bool))[0].astype(int)
                return sorted(int(i) for i in translator[local_idx])
        return sorted(int(i) for i in local_idx)

    def _refresh_roi_list_var(self) -> None:
        ids = self._picked_cluster_roi_ids()
        if not ids:
            self.roi_list_var.set("")
            self.copy_rois_btn.config(state="disabled")
            return
        self.roi_list_var.set(format_roi_indices(ids))
        self.copy_rois_btn.config(state="normal")

    def _on_copy_roi_list(self) -> None:
        text = self.roi_list_var.get()
        if not text:
            return
        try:
            self.clipboard_clear()
            self.clipboard_append(text)
            # Force the clipboard to retain its contents after the Tk app
            # exits — without this, a quick run-and-quit loses the copy on
            # some Windows builds.
            self.update()
        except tk.TclError as e:
            print(f"[GUI] clipboard copy failed: {e}")
            return
        self.status_var.set(
            f"Copied {len(self._picked_cluster_roi_ids())} ROI ids to clipboard.")

    def _cluster_pick_all(self) -> None:
        for v in self._cluster_vars.values():
            v.set(True)
        self._update_cluster_pick_label()

    def _cluster_pick_clear(self) -> None:
        for v in self._cluster_vars.values():
            v.set(False)
        self._update_cluster_pick_label()

    def _on_recluster(self) -> None:
        """Open a separate inspection window with a sub-dendrogram and
        sub-spatial map for the union of the selected branches. The main
        clustering state is left untouched."""
        if self._Z is None or self._dff is None:
            messagebox.showinfo("Run first", "Run the analysis before "
                                "reclustering.")
            return
        visual_picked = self._picked_visual_ids()
        if not visual_picked:
            messagebox.showerror(
                "No clusters",
                "Tick at least one cluster in the dropdown.")
            return
        try:
            new_T = float(self.recluster_thr_var.get())
        except ValueError:
            messagebox.showerror("Bad threshold",
                                 "Threshold must be a number.")
            return

        # Translate visual ids (C1..Cn in dendrogram order) to raw fcluster
        # labels using the mapping built during the most recent render.
        try:
            target_labels = [self._visual_to_label[v] for v in visual_picked]
        except KeyError:
            # Stale mapping — rebuild and ask the user to retry.
            self._visual_to_label = self._compute_visual_cluster_map(
                self._current_threshold())
            self._rebuild_cluster_menu()
            messagebox.showerror(
                "Cluster list out of date",
                "The cluster numbering changed — re-pick the clusters.")
            return

        cur_T = self._current_threshold()
        labels = fcluster(self._Z, t=cur_T, criterion="distance")
        mask = np.isin(labels, np.array(target_labels, dtype=labels.dtype))
        global_idx = np.where(mask)[0].astype(int)
        if global_idx.size < 2:
            sizes = ", ".join(
                f"C{v}={int((labels == self._visual_to_label[v]).sum())}"
                for v in visual_picked)
            messagebox.showerror(
                "Selection too small",
                f"Selected branches contain only {global_idx.size} ROI(s); "
                f"need >=2 to recluster.\n\nPer-cluster ROI counts at the "
                f"current cut ({cur_T:.3f}): {sizes}.\n\nTry lowering the "
                f"main threshold to get larger clusters.")
            return

        dff_subset = self._dff[:, global_idx]
        try:
            Z_sub = _correlation_linkage(dff_subset)
        except Exception as e:
            messagebox.showerror("Recluster failed", str(e))
            return

        if isinstance(self._stat, np.ndarray):
            stat_subset = self._stat[global_idx]
        else:
            stat_subset = [self._stat[int(i)] for i in global_idx]

        new_zmax = float(np.max(Z_sub[:, 2]))
        init_T = min(max(new_T, 1e-6), new_zmax)

        # Pick a fresh palette so the recluster window's colors visibly
        # differ from the parent dendrogram (and from earlier recluster
        # windows). Only the parent's color mapping carries meaning; the
        # sub-tree colors are just for visual separation.
        alt_palettes = [p for p in cmap_mod.CATEGORICAL_PALETTES
                        if p != self._palette_name] or ["Set2"]
        alt_palette = alt_palettes[
            self._recluster_palette_idx % len(alt_palettes)]
        self._recluster_palette_idx += 1

        def palette_resolver(n: int) -> list[str]:
            return cmap_mod.resolve_palette(alt_palette,
                                            n_colors=max(1, int(n)))

        branch_str = ",".join(f"C{v}" for v in visual_picked)
        ReclusterWindow(
            self.winfo_toplevel(),
            Z=Z_sub, stat=stat_subset, Lx=self._Lx, Ly=self._Ly,
            init_threshold=init_T, zmax=new_zmax,
            palette_resolver=palette_resolver,
            branch_label=branch_str, n_rois=int(global_idx.size),
        )

    # -- Render ------------------------------------------------------------

    def _resolve_palette_colors(self, n_clusters: int) -> list[str]:
        if self._custom_colors:
            cols = list(self._custom_colors)
            if len(cols) < n_clusters:
                cols = cols + [cols[-1]] * (n_clusters - len(cols))
            return cols[:n_clusters]
        return cmap_mod.resolve_palette(self._palette_name,
                                        n_colors=max(1, n_clusters))

    def _render_all(self) -> None:
        if self._Z is None:
            return
        T = self._current_threshold()
        labels = fcluster(self._Z, t=T, criterion="distance")
        n_clusters = int(np.unique(labels).size)

        # Visual id mapping (depends only on tree traversal, not colors).
        self._visual_to_label = self._compute_visual_cluster_map(T)

        # Build a per-label color table in visual order. This is the single
        # source of truth shared by the dendrogram (link_color_func), the
        # spatial map, the menu, and the per-cluster dialog.
        label_to_color = self._build_label_color_map(n_clusters)

        # Per-cluster colors keyed by visual id (menu + dialog seed).
        self._visual_to_color = {
            vid: label_to_color.get(int(lbl), "#cccccc")
            for vid, lbl in self._visual_to_label.items()
        }

        # Per-leaf (per-ROI) colors (spatial paint). Above-threshold ROIs
        # fall back to the gray above-cut color via dict.get default.
        self._leaf_colors = {
            int(i): label_to_color.get(int(labels[i]), ABOVE_CUT_COLOR)
            for i in range(int(len(labels)))
        }

        self._rebuild_cluster_menu()

        link_color_func = self._make_link_color_func(T, labels, label_to_color)

        # --- Dendrogram ---
        ax = self.d_ax
        ax.clear()
        # link_color_func overrides scipy's color cycling entirely; we
        # don't pass color_threshold (it would be ignored) but the
        # dashed cut line is still drawn at T below.
        dendrogram(self._Z, ax=ax,
                   link_color_func=link_color_func,
                   above_threshold_color=ABOVE_CUT_COLOR,
                   no_labels=True)
        ax.axhline(T, linestyle="--", linewidth=1.5, color="black")
        ax.set_ylabel("1 - r  (linkage distance)")
        ax.set_xlabel(f"{self._Z.shape[0] + 1} ROIs ({n_clusters} clusters)")
        ax.set_title(f"cut @ {T:.3f}"
                     f"  ({T / self._zmax:.2f} x max)"
                     + ("  [manual]" if self.manual_var.get() else "  [auto]"))
        self.d_canvas.draw_idle()

        self.threshold_readout.set(f"{T:.3f}")

        # --- Spatial ---
        N = self._Z.shape[0] + 1
        roi_rgb = np.zeros((N, 3))
        for leaf_idx, hex_color in self._leaf_colors.items():
            if 0 <= leaf_idx < N:
                roi_rgb[leaf_idx, :] = mpl.colors.to_rgb(hex_color)

        if self._stat is None or len(self._stat) != N:
            self._placeholder(self.s_ax,
                              "Spatial unavailable (stat / dF/F mismatch).")
            self.s_canvas.draw_idle()
            return

        img = _spatial_image(self._stat, self._Lx, self._Ly, roi_rgb)
        ax = self.s_ax
        ax.clear()
        ax.imshow(img, origin="upper", aspect="equal")
        ax.set_title(f"{n_clusters} clusters")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        self.s_canvas.draw_idle()

        # Cluster numbering / ROI->cluster assignment may have changed
        # (threshold drag, palette change). Refresh the picked-clusters
        # ROI list so the displayed Suite2p ids stay in sync.
        self._refresh_roi_list_var()

    # -- Export + cross-correlation ----------------------------------------

    def _export_clusters(self) -> Path:
        """Write *_rois.npy files for the current cut into a fresh subfolder.

        Files are numbered in dendrogram visual (left-to-right) order so the
        exported ``C1, C2, ...`` match the cluster numbers shown in the
        dropdown picker and the dendrogram colors — not raw fcluster label
        order.

        Indices written into each ``C*_rois.npy`` are *Suite2p ROI indices*
        (positions in ``stat.npy`` / ``F.npy``), not filtered-list positions.
        A ``_indices_are_suite2p`` marker file is dropped alongside so the
        cross-correlation runner knows it doesn't need to translate the
        legacy filtered-position layout.

        Returns the export directory.
        """
        if self._Z is None or self._plane0 is None:
            raise RuntimeError("Run analysis first.")
        T = self._current_threshold()
        labels = fcluster(self._Z, t=T, criterion="distance")
        # Defensive: ensure the visual mapping is current with this T.
        self._visual_to_label = self._compute_visual_cluster_map(T)

        # If clustering was run on a filter-restricted dF/F view, build a
        # filtered-position -> Suite2p ROI index translator from the same
        # mask the loaders use.
        filtered_to_suite2p: Optional[np.ndarray] = None
        if "filtered" in self._prefix.split("_"):
            mask = _load_filter_mask(self._plane0)
            if mask is not None and int(mask.sum()) == len(labels):
                filtered_to_suite2p = np.where(
                    np.asarray(mask, dtype=bool))[0].astype(int)

        out_dir = (self._plane0 / f"{self._prefix}cluster_results"
                   / EXPORT_SUBDIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Clear stale C*_rois.npy first so old runs don't poison crosscorr.
        for stale in out_dir.glob("C*_rois.npy"):
            stale.unlink()
        for vid in range(1, len(self._visual_to_label) + 1):
            lbl = int(self._visual_to_label[vid])
            local_idx = np.where(labels == lbl)[0].astype(int)
            if filtered_to_suite2p is not None:
                roi_idx = filtered_to_suite2p[local_idx]
            else:
                roi_idx = local_idx
            np.save(out_dir / f"C{vid}_rois.npy", roi_idx.astype(int))
        np.save(out_dir / "linkage.npy", self._Z)
        np.save(out_dir / "threshold_used.npy", np.array([T], dtype=float))
        # Marker so the xcorr runner knows the .npy files contain Suite2p
        # ROI indices (post-2026-04-30 layout) and skips legacy translation.
        (out_dir / "_indices_are_suite2p").write_text("1")
        return out_dir

    def _on_export(self) -> None:
        try:
            out_dir = self._export_clusters()
        except Exception as e:
            messagebox.showerror("Export failed", str(e))
            return
        try:
            self._write_summary()
        except Exception as e:
            print(f"[GUI] cluster summary write failed: {e}")
        self.status_var.set(f"Exported clusters to {out_dir}")

    # -- Summary export ----------------------------------------------------

    def _schedule_summary_write(self, delay_ms: int = 600) -> None:
        """Debounced trigger so slider drags don't keep rewriting the file."""
        if self._Z is None or self._plane0 is None:
            return
        if self._summary_after_id is not None:
            try:
                self.after_cancel(self._summary_after_id)
            except Exception:
                pass
        self._summary_after_id = self.after(
            delay_ms, self._summary_after_fire)

    def _summary_after_fire(self) -> None:
        self._summary_after_id = None
        try:
            self._write_summary()
        except Exception as e:
            print(f"[GUI] cluster summary write failed: {e}")

    def _write_summary(self) -> Optional[Path]:
        if self._Z is None or self._plane0 is None:
            return None
        T = self._current_threshold()
        labels = fcluster(self._Z, t=T, criterion="distance")
        n_clusters = int(np.unique(labels).size)

        # Refresh the visual mapping for this T so the sheet's cluster_id
        # column matches the dropdown / dendrogram numbering instead of the
        # raw fcluster label order.
        self._visual_to_label = self._compute_visual_cluster_map(T)
        label_to_color = self._build_label_color_map(n_clusters)

        # Remap per-ROI fcluster labels to visual ids and re-key the color
        # dict by visual id so the Clusters sheet shows the exact cluster
        # numbers the user sees in the GUI.
        label_to_visual = {int(lbl): vid
                           for vid, lbl in self._visual_to_label.items()}
        visual_labels = np.array(
            [label_to_visual.get(int(l), int(l)) for l in labels],
            dtype=int,
        )
        visual_colors = {
            label_to_visual.get(int(lbl), int(lbl)): color
            for lbl, color in label_to_color.items()
        }

        # The clustering ran on a filter-restricted dF/F matrix, so column i
        # of cluster_labels refers to the i-th *kept* ROI. Translate to
        # Suite2p ROI indices so the Clusters sheet's `roi` column lines
        # up with the ROIs sheet (where `roi` is the Suite2p index).
        if "filtered" in self._prefix.split("_"):
            mask = _load_filter_mask(self._plane0)
            if mask is not None and int(mask.sum()) == len(labels):
                roi_indices = np.where(mask)[0].astype(int).tolist()
            else:
                roi_indices = None
        else:
            roi_indices = None

        try:
            fps = float(utils.get_fps_from_notes(str(self._plane0)))
        except Exception:
            fps = None

        summary_writer.update_recording_meta(
            self._plane0, prefix=self._prefix, fps=fps,
            T=None, N=int(len(labels)),
            extra={"clustering_n_clusters": n_clusters,
                   "clustering_threshold": float(T),
                   "clustering_palette": (
                       "custom" if self._custom_colors else self._palette_name)})
        return summary_writer.write_clusters_sheet(
            self._plane0, visual_labels,
            cluster_colors=visual_colors,
            threshold=float(T),
            method="average", metric="correlation",
            roi_indices=roi_indices)

    def _on_save_summary(self) -> None:
        if self._Z is None or self._plane0 is None:
            messagebox.showinfo("Run first",
                                "Run analysis before saving the summary.")
            return
        try:
            path = self._write_summary()
            if path is not None:
                self.status_var.set(f"Summary -> {path}")
        except Exception as e:
            messagebox.showerror("Summary failed", str(e))

# ---------------------------------------------------------------------------
# Standalone main
# ---------------------------------------------------------------------------


def main() -> None:
    root = tk.Tk()
    root.title("CalLIOPE - Cross-correlation clustering")
    root.geometry("1300x780")
    tab = ClusteringTab(root, state=None)
    tab.pack(fill="both", expand=True)
    root.mainloop()


if __name__ == "__main__":
    main()
