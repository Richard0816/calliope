"""Per-cluster heatmap + raster popout for Tab 6.

What it shows
-------------
When the user clicks an ROI on Tab 6's spatial cluster map, this
Toplevel opens with two stacked panels for that ROI's cluster:

    [ Filtered lowpass dF/F heatmap, one row per ROI ]
    [ Event raster, same row order, one dot per onset ]

Plus a status row that names the cluster, lists the cluster's
Suite2p ROI ids, and highlights the row corresponding to the clicked
ROI with a yellow tick on the y-axis.

Data sources
------------
- **Heatmap** is built from ``ClusteringTab._dff`` (the filtered
  lowpass dF/F memmap, ``(T, N_kept)``) -- the same array Tab 6
  feeds into its linkage. Rows are min-max normalised per ROI so
  bursts read regardless of absolute brightness.
- **Raster** prefers ``AppState.event_results`` (Tab 5's per-ROI
  hysteresis onsets) when its ``kept_idx`` aligns with Tab 6's
  filtered stat list and the column count matches the heatmap. When
  not available, a derivative-threshold fallback computes a simple
  raster from the dF/F (``|d/dt|`` above a robust per-ROI threshold)
  so the user still sees something useful.

Single-instance pattern
-----------------------
Tab 6 keeps one popout per plane0; subsequent clicks on different
ROIs (possibly in different clusters) call :meth:`show_cluster` on
the existing popout instead of opening multiple windows.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Optional

import customtkinter as ctk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import to_rgba


def _normalise_rows(arr: np.ndarray) -> np.ndarray:
    """Per-row min-max scale to [0, 1]; zero-variance rows -> 0."""
    out = np.asarray(arr, dtype=np.float32).copy()
    if out.size == 0:
        return out
    lo = out.min(axis=1, keepdims=True)
    hi = out.max(axis=1, keepdims=True)
    rng = hi - lo
    safe = np.where(rng > 1e-12, rng, 1.0)
    out = (out - lo) / safe
    out[(rng <= 1e-12).reshape(-1)] = 0.0
    return np.clip(out, 0.0, 1.0)


def _derivative_raster(dff_col_major: np.ndarray, k: float = 4.0) -> np.ndarray:
    """Fallback raster: per-ROI onsets where ``|d/dt|`` crosses
    ``k * MAD`` above the running median.

    ``dff_col_major`` is ``(N_rois, T)``. Returns a ``uint8`` array of
    the same shape: 1 at frames where a positive-going derivative
    spike is detected, 0 otherwise.
    """
    if dff_col_major.size == 0:
        return np.zeros_like(dff_col_major, dtype=np.uint8)
    diff = np.diff(dff_col_major, axis=1, prepend=dff_col_major[:, :1])
    # Robust per-ROI scale via MAD; fall back to std when degenerate.
    med = np.median(diff, axis=1, keepdims=True)
    mad = np.median(np.abs(diff - med), axis=1, keepdims=True)
    scale = np.where(mad > 1e-9, 1.4826 * mad, diff.std(axis=1, keepdims=True))
    scale = np.where(scale > 1e-9, scale, 1.0)
    z = (diff - med) / scale
    raster = (z >= k).astype(np.uint8)
    return raster


def _slice_state_raster(event_results: Optional[dict],
                        dff_T: int) -> Optional[np.ndarray]:
    """Return the raster from ``event_results`` if its column count
    matches ``dff_T`` (so per-ROI rows can be sliced 1:1 with the
    heatmap), else ``None``.
    """
    if not event_results:
        return None
    rs = event_results.get("raster")
    if rs is None:
        return None
    rs = np.asarray(rs)
    if rs.ndim != 2:
        return None
    # event_results' raster is downsampled to `num_cols` (typically
    # 2000 columns). It only aligns with the dff heatmap when the
    # column count matches -- otherwise different row orderings make
    # the slice meaningless. Phase 1: only use when shapes match.
    if rs.shape[1] != dff_T:
        return None
    return rs


class ClusterPopout(ctk.CTkToplevel):
    """Heatmap + raster Toplevel for the clicked ROI's cluster."""

    def __init__(self, master, plane0: Path) -> None:
        super().__init__(master)
        self.title("Cluster heatmap + raster")
        self.geometry("1300x720")
        self.wm_minsize(900, 500)

        self._plane0 = Path(plane0)

        # Latest snapshot stashed via show_cluster.
        self._dff: Optional[np.ndarray] = None             # (T, N_kept)
        self._labels: Optional[np.ndarray] = None          # (N_kept,)
        self._stat = None                                  # filtered stat
        self._cluster_label: Optional[int] = None
        self._highlight_idx: Optional[int] = None
        self._cluster_color: str = "#cccccc"
        self._fps: float = 1.0
        self._event_results: Optional[dict] = None

        self._fig = plt.Figure(figsize=(13, 6.6), tight_layout=True)
        gs = self._fig.add_gridspec(2, 1, height_ratios=[3, 2])
        self._ax_hm = self._fig.add_subplot(gs[0, 0])
        self._ax_rs = self._fig.add_subplot(gs[1, 0], sharex=self._ax_hm)
        self._canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas.get_tk_widget().pack(side="top", fill="both",
                                          expand=True)

        status_frame = ctk.CTkFrame(self, fg_color="transparent")
        status_frame.pack(side="top", fill="x", padx=6, pady=6)
        self._status_var = tk.StringVar(value="No cluster selected.")
        ttk.Label(status_frame, textvariable=self._status_var,
                  anchor="w").pack(side="left", fill="x", expand=True)
        ctk.CTkButton(status_frame, text="Close (Esc)", width=110,
                      command=self._on_close).pack(side="right")

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Escape>", lambda _e: self._on_close())

    # -- Public ------------------------------------------------------------

    def show_cluster(self, *, clicked_idx: int, cluster_label: int,
                     dff: np.ndarray, labels: np.ndarray, stat,
                     cluster_color: str = "#cccccc",
                     fps: float = 1.0,
                     event_results: Optional[dict] = None) -> None:
        """Repaint the popout for the cluster containing ``clicked_idx``.

        Parameters
        ----------
        clicked_idx : int
            Position in Tab 6's filtered stat list (== column in
            ``dff``) the user clicked. Highlighted in the heatmap
            with a yellow row marker.
        cluster_label : int
            Raw fcluster label of that ROI's cluster.
        dff : np.ndarray
            ``(T, N_kept)`` filtered lowpass dF/F (``ClusteringTab._dff``).
        labels : np.ndarray
            Per-row fcluster labels (``len == N_kept``).
        stat : list
            Filtered stat list (one entry per kept ROI).
        cluster_color : str
            Hex color of the cluster -- used to tint the panel border
            and the cluster summary line so the popout matches the
            Tab 6 spatial map.
        fps : float
            Sampling rate for the time axis. Falls back to frame
            indices when fps <= 0.
        event_results : dict, optional
            Tab 5's render payload; if its raster shape aligns with
            ``dff``, the popout shows real onsets instead of the
            derivative-threshold fallback.
        """
        self._dff = np.asarray(dff)
        self._labels = np.asarray(labels)
        self._stat = stat
        self._cluster_label = int(cluster_label)
        self._highlight_idx = int(clicked_idx)
        self._cluster_color = str(cluster_color or "#cccccc")
        self._fps = float(fps) if fps and fps > 0 else 1.0
        self._event_results = event_results
        self._render()
        self.deiconify()
        self.lift()
        self.focus_set()

    # -- Render ------------------------------------------------------------

    def _render(self) -> None:
        if (self._dff is None or self._labels is None
                or self._cluster_label is None):
            return
        mask = (self._labels == self._cluster_label)
        cluster_idxs = np.where(mask)[0]
        n_in = int(cluster_idxs.size)
        T = int(self._dff.shape[0])

        self._ax_hm.clear()
        self._ax_rs.clear()

        if n_in == 0:
            self._ax_hm.text(0.5, 0.5,
                             f"Cluster {self._cluster_label} is empty",
                             ha="center", va="center",
                             transform=self._ax_hm.transAxes)
            self._canvas.draw_idle()
            self._status_var.set("Empty cluster.")
            return

        cluster_dff = self._dff[:, cluster_idxs].T  # (n_in, T)

        # -- Heatmap ----------------------------------------------------
        hm = _normalise_rows(cluster_dff)
        # Use cluster colour as a tint on the title rectangle so the
        # popout's identity reads at a glance even if multiple windows
        # are open during inspection.
        time_axis = np.arange(T) / self._fps
        self._ax_hm.imshow(
            hm, aspect="auto", interpolation="nearest",
            extent=(float(time_axis[0]) if T else 0.0,
                    float(time_axis[-1]) if T else 1.0,
                    n_in - 0.5, -0.5),
            cmap="magma")
        self._ax_hm.set_ylabel("ROI (within cluster)")
        title_rgba = to_rgba(self._cluster_color, alpha=0.25)
        self._ax_hm.set_title(
            f"Cluster {self._cluster_label}    n = {n_in}    "
            f"clicked ROI position {self._highlight_idx}",
            backgroundcolor=title_rgba)
        # Yellow tick + row outline for the clicked ROI.
        if self._highlight_idx is not None:
            try:
                hl_row = int(np.where(cluster_idxs
                                      == self._highlight_idx)[0][0])
                self._ax_hm.axhline(hl_row, color="yellow",
                                    linewidth=0.8, alpha=0.9)
            except (IndexError, ValueError):
                pass

        # -- Raster -----------------------------------------------------
        state_rs = _slice_state_raster(self._event_results, T)
        if state_rs is not None and state_rs.shape[0] >= max(cluster_idxs) + 1:
            raster = state_rs[cluster_idxs, :]
            raster_src = "Tab 5 onsets"
        else:
            raster = _derivative_raster(cluster_dff)
            raster_src = "|d/dt| fallback"

        # Plot raster as a binary heatmap so 1px-per-onset stays crisp
        # at any window size.
        self._ax_rs.imshow(
            raster, aspect="auto", interpolation="nearest",
            extent=(float(time_axis[0]) if T else 0.0,
                    float(time_axis[-1]) if T else 1.0,
                    n_in - 0.5, -0.5),
            cmap="binary", vmin=0, vmax=1)
        if self._highlight_idx is not None:
            try:
                hl_row = int(np.where(cluster_idxs
                                      == self._highlight_idx)[0][0])
                self._ax_rs.axhline(hl_row, color="yellow",
                                    linewidth=0.8, alpha=0.9)
            except (IndexError, ValueError):
                pass
        self._ax_rs.set_xlabel("Time (s)" if self._fps != 1.0 else "Frame")
        self._ax_rs.set_ylabel("ROI (within cluster)")
        self._ax_rs.set_title(f"Event raster ({raster_src})")

        self._fig.tight_layout()
        self._canvas.draw_idle()

        # Status: cluster summary + the cluster's filtered-position
        # roster, capped so a giant cluster doesn't blow up the label.
        roster = ", ".join(str(int(i)) for i in cluster_idxs[:30])
        if cluster_idxs.size > 30:
            roster += f", ... (+{cluster_idxs.size - 30})"
        self._status_var.set(
            f"Cluster {self._cluster_label}  n={n_in}  "
            f"clicked={self._highlight_idx}  "
            f"raster={raster_src}    members: {roster}")

    # -- Close --------------------------------------------------------------

    def _on_close(self) -> None:
        try:
            plt.close(self._fig)
        except Exception:
            pass
        self.destroy()
