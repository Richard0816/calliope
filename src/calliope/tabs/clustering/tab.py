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
import threading
import tkinter as tk
import traceback

import customtkinter as ctk
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

from .logic import (
    ABOVE_CUT_COLOR, CustomColorDialog, DEFAULT_PALETTE, DEFAULT_PREFIX,
    ReclusterWindow, build_label_image, compute_clustering_offload,
    filtered_to_suite2p_indices, load_filtered_dff, reload_clustering_offload,
    spatial_image, stat_for_prefix, summary_writer, utils, ward_linkage,
)
from .logic import clustering as cmap_mod
from .cluster_popout import ClusterPopout

from ...gui_common import (
    apply_dark_to_tk_widget, attach_fig_toolbar, drain_queue, report_stage_error,
    format_roi_indices,
)


# ``DEFAULT_PREFIX``, ``DEFAULT_PALETTE``, ``ABOVE_CUT_COLOR`` are
# imported from ``.logic`` above so ``ReclusterWindow`` and
# ``ClusteringTab`` share the same constants.
# Cluster ROI .npy files are written directly into
# ``<plane0>/<prefix>cluster_results/`` -- no extra "gui_recluster"
# subfolder. Tab 7 + the headless ``crosscorrelation_run`` already
# accept an empty ``cluster_folder`` and skip the join in that case.
EXPORT_SUBDIR = ""
POLL_MS = 80


# ---------------------------------------------------------------------------
# Backend helpers and dialogs live in ``.logic`` (this file is the GUI).
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Main tab
# ---------------------------------------------------------------------------


class ClusteringTab(ctk.CTkFrame):
    """Cross-correlation hierarchical clustering tab."""

    def __init__(self, master, state=None) -> None:
        super().__init__(master, fg_color="transparent")
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

        # Worker queue. Heavy linkage runs in a child process (see
        # core.offload); ``_job`` is None until the first run.
        self._q: queue.Queue = queue.Queue()
        self._job = None

        self._slider_user_driven = True  # gate slider events during programmatic sets
        self._summary_after_id: Optional[str] = None  # debounce id for auto-write
        # Debounce id coalescing rapid threshold-slider / palette / toggle
        # re-renders so a drag doesn't fire a full dendrogram + spatial
        # repaint per pixel (which froze the GUI mid-drag).
        self._render_after_id: Optional[str] = None

        self._build_ui()
        drain_queue(self, self._q,
                    {"done": self._on_done,
                     "reloaded": self._on_reloaded,
                     "error": self._on_error},
                    poll_ms=POLL_MS)

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
        head = ctk.CTkFrame(self)
        head.pack(fill="x", padx=10, pady=(10, 6))
        ctk.CTkLabel(
            head,
            text="Cross-correlation clustering "
                 "(1 - Pearson r, hierarchical, average linkage)",
            font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(6, 0))

        row1 = ctk.CTkFrame(head, fg_color="transparent")
        row1.pack(fill="x", padx=8, pady=2)
        ctk.CTkLabel(row1, text="plane0:").pack(side="left")
        self.path_var = tk.StringVar(value="")
        ctk.CTkEntry(row1, textvariable=self.path_var, width=560).pack(
            side="left", padx=(4, 4), fill="x", expand=True)
        ctk.CTkButton(row1, text="Browse...", width=90,
                      command=self._on_browse).pack(side="left")

        row2 = ctk.CTkFrame(head, fg_color="transparent")
        row2.pack(fill="x", padx=8, pady=2)
        self.run_btn = ctk.CTkButton(
            row2, text="Run analysis", command=self._on_run, state="disabled")
        self.run_btn.pack(side="left")
        self.reload_btn = ctk.CTkButton(
            row2, text="Reload clusters", width=130,
            command=self._on_reload_clusters, state="disabled")
        self.reload_btn.pack(side="left", padx=(6, 0))
        self.progress = ctk.CTkProgressBar(row2, mode="indeterminate",
                                           width=140)
        self.progress.pack(side="left", padx=8)
        self.status_var = tk.StringVar(value="Pick a plane0 folder.")
        ctk.CTkLabel(row2, textvariable=self.status_var,
                     font=ctk.CTkFont(size=11, slant="italic")).pack(
            side="left", fill="x", expand=True)

        # Recluster controls: drill into one or more clusters (branches) at
        # the current cut and re-cluster only those ROIs at a finer threshold.
        row3 = ctk.CTkFrame(head, fg_color="transparent")
        row3.pack(fill="x", padx=8, pady=2)
        ctk.CTkLabel(row3, text="Recluster branch(es):").pack(side="left")
        self._cluster_vars: dict[int, tk.BooleanVar] = {}
        self.cluster_menu = tk.Menu(self, tearoff=False)
        # ttk.Menubutton kept -- customtkinter has no equivalent cascade menu.
        self.cluster_menu_btn = ttk.Menubutton(
            row3, text="Pick clusters...", width=24, state="disabled")
        self.cluster_menu_btn.config(menu=self.cluster_menu)
        self.cluster_menu_btn.pack(side="left", padx=(8, 0))

        ctk.CTkLabel(row3, text="new threshold (1 - r):").pack(
            side="left", padx=(8, 2))
        self.recluster_thr_var = tk.StringVar(value="")
        ctk.CTkEntry(row3, textvariable=self.recluster_thr_var, width=80).pack(
            side="left")
        self.recluster_btn = ctk.CTkButton(
            row3, text="Recluster (new window)", width=180,
            command=self._on_recluster, state="disabled")
        self.recluster_btn.pack(side="left", padx=(8, 0))

        # ROI list of the currently-picked clusters (Suite2p ROI ids).
        # Read-only display + Copy button so the user can paste into Tab 4
        # / Tab 5 manual-ROI entries or external scripts. Auto-refreshes on
        # every menu toggle and every threshold/palette change.
        row3b = ctk.CTkFrame(head, fg_color="transparent")
        row3b.pack(fill="x", padx=8, pady=(2, 6))
        ctk.CTkLabel(row3b, text="ROIs in picked clusters (Suite2p ids):").pack(
            side="left")
        self.roi_list_var = tk.StringVar(value="")
        self.roi_list_entry = ctk.CTkEntry(
            row3b, textvariable=self.roi_list_var, state="readonly")
        self.roi_list_entry.pack(side="left", fill="x", expand=True,
                                 padx=(8, 4))
        self.copy_rois_btn = ctk.CTkButton(
            row3b, text="Copy", width=80, command=self._on_copy_roi_list,
            state="disabled")
        self.copy_rois_btn.pack(side="left")

        # Body: a single horizontal sash splits (dendrogram + cut
        # slider) from the spatial canvas. Dragging redistributes width
        # between exactly two panes (matching Tab 7's behaviour) so the
        # slider stays tied to the dendrogram it controls.
        # ttk.PanedWindow kept -- customtkinter has no equivalent.
        body = ttk.PanedWindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        left_pane = ctk.CTkFrame(body, fg_color="transparent")
        body.add(left_pane, weight=3)
        left_pane.columnconfigure(0, weight=1)
        left_pane.columnconfigure(1, weight=0)
        left_pane.rowconfigure(0, weight=1)

        # Dendrogram canvas.
        dframe = ctk.CTkFrame(left_pane)
        dframe.grid(row=0, column=0, sticky="nsew")
        ctk.CTkLabel(dframe, text="Dendrogram",
                     font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(6, 0))
        self.d_fig = plt.Figure(figsize=(6.5, 4.5), tight_layout=True)
        self.d_ax = self.d_fig.add_subplot(111)
        self._placeholder(self.d_ax, "Run analysis to populate.")
        self.d_canvas = FigureCanvasTkAgg(self.d_fig, master=dframe)
        attach_fig_toolbar(self.d_canvas, dframe)
        self.d_canvas.get_tk_widget().pack(fill="both", expand=True,
                                           padx=4, pady=(0, 4))

        # Vertical slider for the cut, glued to the right edge of the
        # dendrogram pane (not its own sash pane).
        sframe = ctk.CTkFrame(left_pane, fg_color="transparent")
        sframe.grid(row=0, column=1, sticky="ns", padx=4)
        ctk.CTkLabel(sframe, text="cut").pack(side="top", pady=(4, 0))
        # tk.Scale (not ttk.Scale) kept here -- its ``resolution`` and
        # ``showvalue`` knobs have no clean CTkSlider equivalent, and the
        # apply_dark_to_tk_widget skin already matches the dark surround.
        self.threshold_scale = tk.Scale(
            sframe, from_=1.0, to=0.0, resolution=0.001,
            orient=tk.VERTICAL, length=320, showvalue=False,
            command=self._on_slider, state="disabled")
        self.threshold_scale.pack(side="top", fill="y", expand=True)
        apply_dark_to_tk_widget(self.threshold_scale)
        self.threshold_readout = tk.StringVar(value="-")
        ctk.CTkLabel(sframe, textvariable=self.threshold_readout,
                     width=64, anchor="center").pack(side="top")

        # Spatial canvas.
        sp_frame = ctk.CTkFrame(body)
        body.add(sp_frame, weight=2)
        ctk.CTkLabel(sp_frame, text="Spatial (cluster colors)",
                     font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(6, 0))
        self.s_fig = plt.Figure(figsize=(5.5, 4.5), tight_layout=True)
        self.s_ax = self.s_fig.add_subplot(111)
        self._placeholder(self.s_ax, "Run analysis to populate.")
        self.s_canvas = FigureCanvasTkAgg(self.s_fig, master=sp_frame)
        attach_fig_toolbar(self.s_canvas, sp_frame)
        self.s_canvas.get_tk_widget().pack(fill="both", expand=True,
                                           padx=4, pady=(0, 4))
        # Click an ROI on the spatial map to open a popout showing the
        # heatmap + raster for that ROI's cluster. Resolved via a
        # cached label image (built every render in _render_all) that
        # encodes ``filtered_idx + 1`` per painted pixel.
        self.s_canvas.mpl_connect("button_press_event",
                                  self._on_spatial_click)
        self._spatial_label_img: Optional[np.ndarray] = None
        self._cluster_popout: Optional[ClusterPopout] = None

        # Bottom controls.
        ctl = ctk.CTkFrame(self, fg_color="transparent")
        ctl.pack(fill="x", padx=10, pady=(6, 10))

        self.manual_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            ctl, text="Manual threshold", variable=self.manual_var,
            command=self._on_manual_toggle).pack(side="left")

        ctk.CTkLabel(ctl, text="   palette:").pack(side="left")
        self.palette_var = tk.StringVar(value=DEFAULT_PALETTE)
        palette_box = ctk.CTkComboBox(
            ctl, variable=self.palette_var, state="readonly", width=140,
            values=list(cmap_mod.AVAILABLE_PALETTES),
            command=lambda _v: self._on_palette_change())
        palette_box.pack(side="left", padx=4)

        ctk.CTkButton(ctl, text="Per-cluster colors...", width=160,
                      command=self._on_custom_colors).pack(side="left", padx=4)
        ctk.CTkButton(ctl, text="Reset palette", width=110,
                      command=self._on_reset_palette).pack(side="left", padx=4)

        ctk.CTkLabel(ctl, text="prefix:").pack(side="left", padx=(20, 2))
        self.prefix_var = tk.StringVar(value=DEFAULT_PREFIX)
        ctk.CTkEntry(ctl, textvariable=self.prefix_var, width=140).pack(
            side="left")

        self.export_btn = ctk.CTkButton(
            ctl, text="Export *_rois.npy", width=150,
            command=self._on_export, state="disabled")
        self.export_btn.pack(side="right")
        self.export_figs_btn = ctk.CTkButton(
            ctl, text="Export figures", width=130,
            command=self._on_export_figures, state="disabled")
        self.export_figs_btn.pack(side="right", padx=(0, 6))
        self.summary_btn = ctk.CTkButton(
            ctl, text="Save summary", width=120,
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

    def _repoint_after_copy(self, scratch: Path, final: Path) -> None:
        """BatchTab calls this after a row's scratch->HDD copy. Drop
        any ``np.memmap`` we still hold against a file under
        ``scratch`` so Windows releases the handle before the bulk
        rmtree. The next Run analysis on the new recording reopens
        against the HDD path.

        Covers two holders the path-walking repoint can't reach:
          - ``self._dff`` -- memmap returned by ``load_filtered_dff``
            and stashed in ``_on_done``; never nulled by
            ``_set_plane0`` so it would leak across recordings.
          - The open ``ClusterPopout``'s ``_dff`` -- ``np.asarray``
            on a memmap returns the memmap unchanged, so the popout
            also retains the scratch handle.
        """
        def _under_scratch(arr) -> bool:
            fname = getattr(arr, "filename", None)
            if fname is None:
                return False
            try:
                Path(fname).relative_to(scratch)
            except (TypeError, ValueError, OSError):
                return False
            return True

        if _under_scratch(self._dff):
            self._dff = None

        popout = self._cluster_popout
        if popout is not None:
            try:
                if popout.winfo_exists() and _under_scratch(
                        getattr(popout, "_dff", None)):
                    popout._dff = None
            except Exception:
                pass

    def _set_plane0(self, plane0: Path) -> None:
        self._plane0 = plane0
        ok = self._inputs_ready(plane0, self.prefix_var.get())
        self.run_btn.configure(state="normal" if ok else "disabled")
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
        base = plane0 / f"{prefix}cluster_results"
        return base / EXPORT_SUBDIR if EXPORT_SUBDIR else base

    def _has_existing_clusters(self, plane0: Path, prefix: str) -> bool:
        d = self._existing_cluster_dir(plane0, prefix)
        return ((d / "linkage.npy").exists()
                and (d / "threshold_used.npy").exists())

    def _refresh_reload_btn(self) -> None:
        if self._plane0 is None:
            self.reload_btn.configure(state="disabled")
            return
        prefix = self.prefix_var.get().strip() or DEFAULT_PREFIX
        ok = (self._inputs_ready(self._plane0, prefix)
              and self._has_existing_clusters(self._plane0, prefix))
        self.reload_btn.configure(state="normal" if ok else "disabled")

    # -- Run worker --------------------------------------------------------

    def _on_run(self) -> None:
        if self._job is not None and self._job.running():
            messagebox.showinfo("Busy", "Analysis already running.")
            return
        plane0 = Path(self.path_var.get())
        prefix = self.prefix_var.get().strip() or DEFAULT_PREFIX
        if not self._inputs_ready(plane0, prefix):
            messagebox.showerror("Not ready", f"plane0 missing required files: {plane0}")
            return

        self._prefix = prefix
        self.run_btn.configure(state="disabled")
        self.progress.start()
        self.status_var.set("Computing linkage...")

        # Offload the pdist + Ward linkage to a child process (not a
        # thread). The payload is small + picklable (Z + scalars); the
        # dff memmap and stat list are re-opened by path in _on_done.
        from ...core.offload import run_offloaded
        self._job = run_offloaded(
            self, self._q, compute_clustering_offload,
            args=(str(plane0), prefix),
            poll_ms=POLL_MS,
        )

    def _on_reload_clusters(self) -> None:
        """Restore linkage + threshold from a previous Export *_rois.npy run.

        Skips the (expensive) correlation-linkage computation and snaps the
        slider to the saved threshold in manual mode. dF/F + stat + ops are
        still loaded fresh from disk so reclustering and the spatial map
        keep working.
        """
        if self._job is not None and self._job.running():
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
        self.run_btn.configure(state="disabled")
        self.reload_btn.configure(state="disabled")
        self.progress.start()
        self.status_var.set("Reloading saved clusters...")

        # Offload the reload (np.load linkage + leaf-count validation) to
        # a child process. cluster_dir is resolved here on the main
        # thread and passed as a string; dff + stat are re-opened by path
        # in _on_reloaded.
        cluster_dir = self._existing_cluster_dir(plane0, prefix)
        from ...core.offload import run_offloaded
        self._job = run_offloaded(
            self, self._q, reload_clustering_offload,
            args=(str(plane0), prefix, str(cluster_dir)),
            done_kind="reloaded",
            poll_ms=POLL_MS,
        )

    def _on_error(self, payload: str) -> None:
        self.progress.stop()
        self.run_btn.configure(state="normal")
        self._refresh_reload_btn()
        self.status_var.set("Analysis failed.")
        # Forward to the batch runner (so a worker crash fails-and-
        # continues the queue) and skip the modal when a batch is driving.
        if report_stage_error(self.state, payload):
            return
        messagebox.showerror(
            "Analysis failed", payload.split("\n", 1)[0])

    def _on_done(self, data: dict) -> None:
        self.progress.stop()
        self.run_btn.configure(state="normal")

        # Re-open the dF/F memmap + stat list on the main thread (they
        # can't cross the process boundary; both are cheap by path).
        plane0 = Path(self.path_var.get())
        self._dff, _, _, _ = load_filtered_dff(plane0, self._prefix)
        self._stat, _ = stat_for_prefix(plane0, self._prefix)

        self._Z = data["Z"]
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
        self.export_btn.configure(state="normal")
        self.export_figs_btn.configure(state="normal")
        self.summary_btn.configure(state="normal")
        self.recluster_btn.configure(state="normal")
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

        # Tab 0's batch runner subscribes to this signal to advance to
        # Tab 7. Publishing here (rather than after every recluster) keeps
        # the batch advancing exactly once per row.
        try:
            plane0 = Path(self.path_var.get())
            self.state.set_clusters_ready(plane0)
        except Exception as e:
            print(f"[GUI] clusters_ready publish failed: {e}")

    def _on_reloaded(self, data: dict) -> None:
        self.progress.stop()
        self.run_btn.configure(state="normal")

        # Re-open the dF/F memmap + stat list on the main thread (they
        # can't cross the process boundary; both are cheap by path).
        plane0 = Path(self.path_var.get())
        self._dff, _, _, _ = load_filtered_dff(plane0, self._prefix)
        self._stat, _ = stat_for_prefix(plane0, self._prefix)

        self._Z = data["Z"]
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
        self.export_btn.configure(state="normal")
        self.summary_btn.configure(state="normal")
        self.recluster_btn.configure(state="normal")
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

    def _schedule_render(self, delay_ms: int = 120) -> None:
        """Coalesce rapid re-render requests (slider drag, palette spin,
        manual toggle) into one ``_render_all`` after a short idle gap.

        A full dendrogram + spatial repaint per slider pixel froze the
        GUI mid-drag; debouncing with ``after_cancel`` (mirroring
        ``_schedule_summary_write``) renders only the settled threshold.
        """
        if self._render_after_id is not None:
            try:
                self.after_cancel(self._render_after_id)
            except Exception:
                pass
        self._render_after_id = self.after(delay_ms, self._render_all_now)

    def _render_all_now(self) -> None:
        self._render_after_id = None
        self._render_all()

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
        self._schedule_render()
        self._schedule_summary_write()

    def _on_slider(self, raw: str) -> None:
        if not self._slider_user_driven:
            return
        try:
            self._manual_T = float(raw)
        except ValueError:
            return
        if self._Z is not None:
            self._schedule_render()
            self._schedule_summary_write()

    def _on_palette_change(self, *_):
        self._palette_name = self.palette_var.get() or DEFAULT_PALETTE
        self._custom_colors = None
        if self._Z is not None:
            self._schedule_render()
            self._schedule_summary_write()

    def _on_spatial_click(self, event) -> None:
        """Resolve the clicked ROI on the spatial cluster map and open
        / refocus the cluster popout.

        Skips clicks while a navigation toolbar mode is active so a
        zoom-rectangle drag doesn't open the popout. Background-pixel
        clicks are reported in the status bar instead of opening an
        empty window.
        """
        if event.inaxes is not self.s_ax:
            return
        if event.button != 1:
            return
        toolbar = getattr(self.s_canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return
        if (self._spatial_label_img is None or self._Z is None
                or self._dff is None or self._stat is None):
            return
        if event.xdata is None or event.ydata is None:
            return
        Ly, Lx = self._spatial_label_img.shape
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        if not (0 <= x < Lx and 0 <= y < Ly):
            return
        lbl = int(self._spatial_label_img[y, x])
        if lbl <= 0:
            self.status_var.set(
                f"No ROI at pixel ({x}, {y}); click on a coloured ROI.")
            return
        clicked_idx = lbl - 1

        # Recompute labels at the current threshold so the popout
        # matches whatever the spatial map is showing right now.
        T = self._current_threshold()
        labels = fcluster(self._Z, t=T, criterion="distance")
        if not (0 <= clicked_idx < len(labels)):
            return
        cluster_label = int(labels[clicked_idx])

        cluster_color = self._leaf_colors.get(int(clicked_idx), "#cccccc")
        # Look up Tab 5's event_results when the GUI is wired into the
        # shared AppState; the popout falls back to a derivative-based
        # raster otherwise.
        event_results = None
        if self.state is not None:
            event_results = getattr(self.state, "event_results", None)

        # Resolve fps via lowpass-tab-style notes lookup; harmless to
        # default to 1.0 (popout shows frame indices instead of seconds).
        fps = 1.0
        try:
            from ...core import utils as core_utils
            fps = float(core_utils.get_fps_from_notes(str(self._plane0)))
        except Exception:
            pass

        popout = self._cluster_popout
        if (popout is None or not popout.winfo_exists()
                or getattr(popout, "_plane0", None) != self._plane0):
            try:
                popout = ClusterPopout(self.winfo_toplevel(), self._plane0)
            except Exception as e:
                messagebox.showerror("Cluster popout failed", str(e))
                return
            self._cluster_popout = popout
        popout.show_cluster(
            clicked_idx=clicked_idx,
            cluster_label=cluster_label,
            dff=self._dff, labels=labels, stat=self._stat,
            cluster_color=cluster_color, fps=fps,
            event_results=event_results,
        )

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
            translator = filtered_to_suite2p_indices(
                self._plane0, self._prefix)
            if translator is not None and translator.size == len(labels):
                return sorted(int(i) for i in translator[local_idx])
        return sorted(int(i) for i in local_idx)

    def _refresh_roi_list_var(self) -> None:
        ids = self._picked_cluster_roi_ids()
        if not ids:
            self.roi_list_var.set("")
            self.copy_rois_btn.configure(state="disabled")
            return
        self.roi_list_var.set(format_roi_indices(ids))
        self.copy_rois_btn.configure(state="normal")

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
            Z_sub = ward_linkage(dff_subset)
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

        img = spatial_image(self._stat, self._Lx, self._Ly, roi_rgb)
        # Parallel int label image so the click handler can map a
        # cursor pixel back to the filtered-list ROI index. Encoded
        # as ``filtered_idx + 1`` (0 = background); rebuilt on every
        # render so a re-cluster / palette change keeps it in sync.
        self._spatial_label_img = build_label_image(
            self._stat, self._Ly, self._Lx)
        ax = self.s_ax
        ax.clear()
        ax.imshow(img, origin="upper", aspect="equal")
        ax.set_title(f"{n_clusters} clusters  -- click an ROI for cluster panel")
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
            translator = filtered_to_suite2p_indices(
                self._plane0, self._prefix)
            if translator is not None and translator.size == len(labels):
                filtered_to_suite2p = translator

        out_dir = self._existing_cluster_dir(self._plane0, self._prefix)
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
        # ROI indices directly and skips the filtered-position translation
        # step it would otherwise apply.
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
            translator = filtered_to_suite2p_indices(
                self._plane0, self._prefix)
            if translator is not None and translator.size == len(labels):
                roi_indices = translator.tolist()
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

    def _on_export_figures(self, quiet: bool = False) -> None:
        """Save the live dendrogram + spatial-cluster panels as PNG +
        SVG into ``<save_folder>/calliope_figures/clustering/`` and
        refresh the recording's ``manifest.json``. Dumps whatever the
        user is currently looking at (active palette, manual threshold,
        custom colors) -- the export captures the interactive state,
        not a recomputed default.

        ``quiet=True`` (Tab 0 batch toggle) suppresses messageboxes.
        """
        if self._Z is None or self._plane0 is None:
            if not quiet:
                messagebox.showinfo(
                    "Run first",
                    "Run analysis (or 'Reload clusters') before "
                    "exporting figures.")
            return
        save_folder = self._plane0.parents[3]
        figures_root = save_folder / "calliope_figures"
        cluster_dir = figures_root / "clustering"
        try:
            cluster_dir.mkdir(parents=True, exist_ok=True)
            from ...core.export_manifest import write_export_manifest
            from ...core.utils import safe_recording_id
            rec_id = safe_recording_id(self._plane0)
            d_png = cluster_dir / "dendrogram.png"
            s_png = cluster_dir / "spatial_clusters.png"
            self.d_fig.savefig(d_png, dpi=200)
            self.d_fig.savefig(d_png.with_suffix(".svg"))
            self.s_fig.savefig(s_png, dpi=200)
            self.s_fig.savefig(s_png.with_suffix(".svg"))
            params = {
                "prefix": self.prefix_var.get(),
                "palette": ("custom" if self._custom_colors
                            else self._palette_name),
                "threshold": float(self._current_threshold()),
                "manual_threshold": bool(self.manual_var.get()),
            }
            manifest_path = write_export_manifest(
                figures_root,
                rec_id=rec_id, params=params,
                plane0=self._plane0, ckpt_path=None,
            )
            self.status_var.set(
                f"Exported cluster figures -> {cluster_dir}")
            if not quiet:
                messagebox.showinfo(
                    "Export complete",
                    f"Wrote dendrogram + spatial_clusters (PNG + SVG) "
                    f"to:\n  {cluster_dir}\n\n"
                    f"Manifest:\n  {manifest_path}")
        except Exception as e:
            self.status_var.set(f"Export failed: {e}")
            if not quiet:
                messagebox.showerror("Export failed", str(e))

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
    """Stand-alone launcher for hot-running Tab 6 without the full
    ``pipeline_gui`` app. Mirrors the dark-mode + ttk-restyle setup
    ``pipeline_gui.main`` does so the tab still reads correctly.
    """
    from ...gui_common import apply_ttk_dark_theme
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    apply_ttk_dark_theme(root)
    root.title("CalLIOPE - Cross-correlation clustering")
    root.geometry("1300x780")
    tab = ClusteringTab(root, state=None)
    tab.pack(fill="both", expand=True)
    root.mainloop()


if __name__ == "__main__":
    main()
