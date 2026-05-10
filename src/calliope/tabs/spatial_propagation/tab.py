"""calliope.tabs.spatial_propagation.tab - Tab 8: Spatial propagation.

What this tab is for
--------------------
Tab 7 tells you "cluster A consistently leads cluster B by 47 ms".
Tab 8 tells you what that *looks like* in space. For each population
event detected by Tab 5, this tab paints the field of view as an
activation-order map: each cell coloured by *when it fired within
that event* on a continuous cyan -> blue -> red scale.

Four panels per event
---------------------
1. **Top-left** - plain activation-order map. ROIs that didn't
   participate in this event stay grey.
2. **Top-right** - same map with white frame-to-frame propagation
   arrows overlaid. Each arrow connects the centroid of the ROIs
   activating in one frame to the centroid for the next active
   frame, so the wavefront's trajectory reads off directly.
3. **Middle (centred)** - vectors-only view: the same arrow chain
   on a blank FOV, with each frame group drawn as a coloured circle
   whose radius is the 2-D standard deviation of contributing ROIs.
4. **Bottom** - one violin per Δframe of pairwise distance,
   aggregated over every active ROI used as a seed (last-frame ROIs
   excluded).

Pure consumer of Tab 5
----------------------
This tab does NOT run any detection of its own. ``__init__``
subscribes to ``AppState.event_results``; the callback
``_on_event_results`` re-paints whenever Tab 5 publishes a new set
of events. Whatever knobs you tuned on Tab 5 (manual ROI subset,
advanced parameters, etc.) are exactly what you see here -- there's
no second copy of the math.

Lifecycle
---------
``__init__``
    Build widgets, subscribe to event_results, immediately ingest
    if a publish already exists.
``_ingest_results(results)``
    Load Suite2p metadata (stat, ops) for the recording, translate
    the kept_idx published by Tab 5 into stat_filtered, enable
    navigation, paint event 1.
``_on_prev`` / ``_on_next`` / ``_on_spin``
    Event navigation. Each clamps the index, calls ``_render_event``,
    updates the Prev/Next button enabled state.
``_render_event(ev_idx)``
    The painting workhorse. Computes the order rank, the
    paint_order_map image, and the frame-grouped centroids; rebuilds
    all four figures from scratch (so colorbars don't stack).
"""

from __future__ import annotations

import tkinter as tk
import traceback
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Optional

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle

from .logic import spatial as spatial_helpers
from ...core import utils as core_utils

from ...gui_common import AppState, attach_fig_toolbar


class SpatialPropagationTab(ttk.Frame):
    """Tab 8: per-event spatial activation-order maps, driven by Tab 5.

    Instance attributes
    -------------------
    ``state`` : the shared AppState bus.
    ``_data`` : the most recent ingested results, shaped like::

        {
            "stat_filtered": list of Suite2p stat dicts (kept ROIs only),
            "kept_idx":      Suite2p ROI ids that Tab 5 iterated,
            "Ly", "Lx":      FOV dimensions,
            "pix_to_um":     scale factor (or None if missing),
            "event_windows": (n_events, 2) seconds,
            "A":             (N, n_events) bool active mask,
            "first_time":    (N, n_events) per-ROI first-onset time,
            "fps":           recording frame rate,
            "plane0":        Path of the recording.
        }

        Set fresh in every ``_ingest_results`` call.
    ``_event_index`` : currently displayed event (0-based).
    ``_plane0_cache`` : cache of (stat, Ly, Lx, pix_to_um) keyed by
        plane0 path so we don't re-read ops.npy on every publish.
    """

    def __init__(self, master, state: AppState) -> None:
        super().__init__(master, padding=10)
        self.state = state
        self._data: Optional[dict] = None  # see _ingest_results
        self._event_index: int = 0
        # Cache of stat/Ly/Lx/pix_to_um keyed by plane0 path so we don't
        # re-load Suite2p metadata on every Tab 5 publish.
        self._plane0_cache: dict[Path, dict] = {}
        # Stash of the *raw* Tab 5 publish (with hysteresis-onset
        # ``A`` / ``first_time``) so we can switch back to it when the
        # user untoggles the spks override after we've replaced
        # ``self._data["A"]`` / ``["first_time"]`` in place.
        self._last_results: Optional[dict] = None
        # Cache of (T, N_kept) spks arrays keyed by (plane0, tuple(kept_idx))
        # -- subset can change between Tab 5 renders (manual ROI subset),
        # so we key on the kept-id list as well.
        self._spks_cache: dict[tuple, np.ndarray] = {}

        self._build_ui()
        # Subscribe to Tab 5's publish channel. Note: ``subscribe_event_results``
        # only registers the callback; it does NOT replay any prior
        # publish. So we manually check for one and re-fire if Tab 5
        # already ran before this tab was instantiated.
        state.subscribe_event_results(self._on_event_results)
        if state.event_results is not None:
            self._on_event_results(state.event_results)

    # -- UI -----------------------------------------------------------------

    def _build_ui(self) -> None:
        header = ttk.LabelFrame(
            self, text="Spatial propagation (events from tab 5)",
            padding=8)
        header.pack(fill="x", pady=(0, 6))

        row = ttk.Frame(header); row.pack(fill="x", pady=2)
        self.status_var = tk.StringVar(
            value="Run Tab 5 first; this tab will populate automatically "
                  "once events are detected.")
        ttk.Label(row, textvariable=self.status_var,
                  font=("", 9, "italic")).pack(side="left")
        ttk.Button(row, text="Refresh from Tab 5",
                   command=self._refresh).pack(side="right")

        # ---- Spks override toggle ----
        # Tab 5's hysteresis-onset detector is the default activation-
        # time source. When this is checked, we instead read each
        # ROI's *first non-zero spks frame* inside each event window
        # and use that as the per-ROI first-onset time. Same painting
        # pipeline downstream; only the ``first_time`` /``A`` matrices
        # change. Mirrors the experimental ``spks`` signal kind in
        # ``Calcium_imaging_suite2p/lead_lag_prototype.py``.
        toggle_row = ttk.Frame(header); toggle_row.pack(fill="x", pady=(2, 0))
        self._use_spks_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            toggle_row,
            text="Use Suite2p spks (deconvolved) instead of "
                 "hysteresis onsets for first-activation time",
            variable=self._use_spks_var,
            command=self._on_use_spks_toggle,
        ).pack(side="left")
        ttk.Label(
            toggle_row,
            text="(activation = first frame with spks > 0 inside the "
                 "event window)",
            foreground="gray", font=("", 8, "italic"),
        ).pack(side="left", padx=(8, 0))

        # Event navigation row.
        nav = ttk.Frame(header); nav.pack(fill="x", pady=(6, 0))
        self.prev_btn = ttk.Button(
            nav, text="< Prev", command=self._on_prev, state="disabled")
        self.prev_btn.pack(side="left")
        self.event_var = tk.IntVar(value=1)
        self.event_spin = ttk.Spinbox(
            nav, from_=1, to=1, width=6, textvariable=self.event_var,
            command=self._on_spin)
        self.event_spin.pack(side="left", padx=(6, 6))
        self.event_spin.state(["disabled"])
        self.event_spin.bind("<Return>", lambda _e: self._on_spin())
        self.event_spin.bind("<FocusOut>", lambda _e: self._on_spin())
        self.next_btn = ttk.Button(
            nav, text="Next >", command=self._on_next, state="disabled")
        self.next_btn.pack(side="left")
        self.event_label_var = tk.StringVar(value="")
        ttk.Label(nav, textvariable=self.event_label_var,
                  font=("", 9, "italic")).pack(side="left", padx=(12, 0))

        body = ttk.Frame(self); body.pack(fill="both", expand=True)
        body.rowconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)

        # Top row, left: plain activation-order map (no arrows).
        f_map_plain = ttk.LabelFrame(
            body, text="Activation order  (cyan = earliest, red = latest)",
            padding=4)
        f_map_plain.grid(row=0, column=0, sticky="nsew",
                         padx=(0, 2), pady=(0, 4))
        self.fig_map_plain = plt.Figure(figsize=(5, 5), tight_layout=True)
        self.ax_map_plain = self.fig_map_plain.add_subplot(111)
        self.ax_map_plain.set_axis_off()
        self.ax_map_plain.text(
            0.5, 0.5, "Run Tab 5 to populate this view.",
            ha="center", va="center",
            transform=self.ax_map_plain.transAxes)
        self.canvas_map_plain = FigureCanvasTkAgg(
            self.fig_map_plain, master=f_map_plain)
        self.toolbar_map_plain = attach_fig_toolbar(
            self.canvas_map_plain, f_map_plain,
            data_basename="coact_order_map")
        self.canvas_map_plain.get_tk_widget().pack(
            fill="both", expand=True)

        # Top row, right: same map with frame-to-frame arrows overlaid.
        f_map_arrows = ttk.LabelFrame(
            body, text="Activation order + frame-to-frame propagation",
            padding=4)
        f_map_arrows.grid(row=0, column=1, sticky="nsew",
                          padx=(2, 0), pady=(0, 4))
        self.fig_map_arrows = plt.Figure(figsize=(5, 5), tight_layout=True)
        self.ax_map_arrows = self.fig_map_arrows.add_subplot(111)
        self.ax_map_arrows.set_axis_off()
        self.ax_map_arrows.text(
            0.5, 0.5, "Run Tab 5 to populate this view.",
            ha="center", va="center",
            transform=self.ax_map_arrows.transAxes)
        self.canvas_map_arrows = FigureCanvasTkAgg(
            self.fig_map_arrows, master=f_map_arrows)
        self.toolbar_map_arrows = attach_fig_toolbar(
            self.canvas_map_arrows, f_map_arrows,
            data_basename="coact_order_map_arrows")
        self.canvas_map_arrows.get_tk_widget().pack(
            fill="both", expand=True)

        # Middle row: vectors-only panel, centred in a 1:2:1 inner-grid
        # so it sits horizontally aligned with the top pair instead of
        # stretching to the full window width.
        body.rowconfigure(2, weight=1)
        middle = ttk.Frame(body)
        middle.grid(row=1, column=0, columnspan=2, sticky="nsew")
        middle.rowconfigure(0, weight=1)
        middle.columnconfigure(0, weight=1)
        middle.columnconfigure(1, weight=2)
        middle.columnconfigure(2, weight=1)

        f_vec = ttk.LabelFrame(
            middle, text="Frame-to-frame propagation vectors  "
                         "(circle radius = 2D std of contributing ROIs)",
            padding=4)
        f_vec.grid(row=0, column=1, sticky="nsew")
        self.fig_vec = plt.Figure(figsize=(5, 5), tight_layout=True)
        self.ax_vec = self.fig_vec.add_subplot(111)
        self.ax_vec.set_axis_off()
        self.canvas_vec = FigureCanvasTkAgg(self.fig_vec, master=f_vec)
        self.toolbar_vec = attach_fig_toolbar(
            self.canvas_vec, f_vec, data_basename="coact_vectors")
        self.canvas_vec.get_tk_widget().pack(fill="both", expand=True)

        # Bottom row: per-frame violin of pairwise distance, every
        # active ROI used as a seed (except those in the last active
        # frame).
        f_dist = ttk.LabelFrame(
            body, text="Pairwise distance vs Δframe  "
                       "(every active ROI as seed; last-frame ROIs "
                       "excluded)",
            padding=4)
        f_dist.grid(row=2, column=0, columnspan=2, sticky="nsew",
                    pady=(4, 0))
        self.fig_dist = plt.Figure(figsize=(8, 4), tight_layout=True)
        self.ax_dist = self.fig_dist.add_subplot(111)
        self.ax_dist.set_axis_off()
        self.canvas_dist = FigureCanvasTkAgg(self.fig_dist, master=f_dist)
        self.toolbar_dist = attach_fig_toolbar(
            self.canvas_dist, f_dist, data_basename="dist_vs_delta_t")
        self.canvas_dist.get_tk_widget().pack(fill="both", expand=True)

    # -- Tab 5 ingestion ----------------------------------------------------

    def _refresh(self) -> None:
        if self.state.event_results is None:
            messagebox.showinfo(
                "No events yet",
                "Click Render on Tab 5 first.")
            return
        self._on_event_results(self.state.event_results)

    def _on_event_results(self, results: dict) -> None:
        try:
            self._ingest_results(results)
        except Exception as e:
            self.status_var.set("Tab 5 publish failed; see console.")
            print(f"[Tab8] ingest failed: {e}\n{traceback.format_exc()}")

    def _ingest_results(self, results: dict) -> None:
        plane0 = results.get("plane0")
        ev_windows = results.get("event_windows")
        A = results.get("A")
        first_time = results.get("first_time")
        kept_idx = results.get("kept_idx")
        fps = results.get("fps")

        if (plane0 is None or ev_windows is None
                or A is None or first_time is None
                or kept_idx is None):
            self.status_var.set(
                "Tab 5 publish missing fields; run Tab 5 again.")
            return

        plane0 = Path(plane0)
        meta = self._load_plane0_meta(plane0)

        kept_idx_arr = np.asarray(kept_idx, dtype=int)
        # Translate the kept Suite2p ids that Tab 5 iterated into stat
        # entries from this plane0. ``kept_idx`` already encodes any
        # manual-subset filtering Tab 5 applied.
        try:
            stat_filtered = [meta["stat_all"][i] for i in kept_idx_arr]
        except IndexError as e:
            raise RuntimeError(
                f"kept_idx out of range for stat.npy: {e}") from None

        # Stash the raw Tab 5 publish so the spks-toggle off path can
        # restore the hysteresis-onset matrices later without
        # touching Tab 5.
        self._last_results = {
            "plane0": plane0,
            "event_windows": np.asarray(ev_windows),
            "A": np.asarray(A),
            "first_time": np.asarray(first_time),
            "kept_idx": kept_idx_arr,
            "fps": float(fps) if fps is not None else None,
        }

        self._data = {
            "stat_filtered": stat_filtered,
            "kept_idx": kept_idx_arr,
            "Ly": meta["Ly"], "Lx": meta["Lx"],
            "pix_to_um": meta["pix_to_um"],
            "event_windows": np.asarray(ev_windows),
            "A": np.asarray(A),
            "first_time": np.asarray(first_time),
            "fps": float(fps) if fps is not None else None,
            "plane0": plane0,
        }

        # If the user has the spks override toggled on, swap in the
        # spks-derived ``A`` / ``first_time`` before painting.
        if self._use_spks_var.get():
            self._apply_spks_override(quiet=True)

        n_events = int(self._data["event_windows"].shape[0])
        if n_events == 0:
            self.status_var.set(
                f"Tab 5 detected 0 events for {plane0}.")
            self.event_spin.config(from_=1, to=1)
            self.event_spin.state(["disabled"])
            self.prev_btn.config(state="disabled")
            self.next_btn.config(state="disabled")
            self.event_label_var.set("")
            for fig, canvas in (
                    (self.fig_map_plain, self.canvas_map_plain),
                    (self.fig_map_arrows, self.canvas_map_arrows),
                    (self.fig_vec, self.canvas_vec),
                    (self.fig_dist, self.canvas_dist)):
                fig.clear()
                ax = fig.add_subplot(111)
                ax.set_axis_off()
                ax.text(0.5, 0.5, "No events detected",
                        ha="center", va="center",
                        transform=ax.transAxes)
                canvas.draw_idle()
            return

        self.status_var.set(
            f"Synced from Tab 5: {n_events} event"
            f"{'' if n_events == 1 else 's'}, "
            f"{kept_idx_arr.size} ROIs.  "
            f"Use Prev / Next or the spinner to click through.")
        self.event_spin.config(from_=1, to=n_events)
        self.event_spin.state(["!disabled"])
        # Stay on the current event index if it's still valid; otherwise
        # snap back to event 1. This makes "re-render Tab 5" feel
        # incremental rather than resetting.
        if self._event_index >= n_events:
            self._event_index = 0
        self.event_var.set(self._event_index + 1)
        self._update_nav_buttons()
        self._render_event(self._event_index)

    def _load_plane0_meta(self, plane0: Path) -> dict:
        cached = self._plane0_cache.get(plane0)
        if cached is not None:
            return cached
        view = core_utils.load_plane_view(plane0)
        stat_all = np.load(plane0 / "stat.npy", allow_pickle=True)
        meta = {
            "stat_all": stat_all,
            "Ly": int(view["Ly"]), "Lx": int(view["Lx"]),
            "pix_to_um": view.get("pix_to_um", None),
        }
        self._plane0_cache[plane0] = meta
        return meta

    # -- Spks override ------------------------------------------------------

    def _on_use_spks_toggle(self) -> None:
        """User flipped the spks toggle. Either swap in the spks-
        derived first-activation matrices and re-render, or restore
        the hysteresis-onset matrices we stashed on the most recent
        Tab 5 publish.
        """
        if self._data is None or self._last_results is None:
            return
        if self._use_spks_var.get():
            self._apply_spks_override(quiet=False)
        else:
            # Restore the hysteresis-onset matrices Tab 5 published.
            self._data["A"] = self._last_results["A"]
            self._data["first_time"] = self._last_results["first_time"]
            self.status_var.set(
                "Restored Tab 5 hysteresis-onset activation times.")
        # Recompute every panel so the new matrices show up.
        self._render_event(self._event_index)

    def _apply_spks_override(self, *, quiet: bool) -> None:
        """Replace ``self._data["A"]`` / ``["first_time"]`` with
        spks-derived versions for the current event windows.

        ``quiet=True`` suppresses the status bar update -- used when
        the override is applied automatically as part of an ingest
        rather than a user toggle.
        """
        if self._data is None:
            return
        plane0 = self._data["plane0"]
        kept_idx = np.asarray(self._data["kept_idx"], dtype=int)
        ev_windows = np.asarray(self._data["event_windows"])
        fps = self._data.get("fps")
        if fps is None or fps <= 0:
            self.status_var.set(
                "Cannot use spks: fps unknown for this recording.")
            self._use_spks_var.set(False)
            return
        try:
            spks = self._load_spks(plane0, kept_idx)
        except FileNotFoundError as e:
            self.status_var.set(f"Spks not found: {e}")
            self._use_spks_var.set(False)
            return
        except Exception as e:
            self.status_var.set(f"Spks load failed: {e}")
            self._use_spks_var.set(False)
            return
        A_spks, first_time_spks = self._first_time_from_spks(
            spks, ev_windows, float(fps))
        self._data["A"] = A_spks
        self._data["first_time"] = first_time_spks
        if not quiet:
            n_active_per_event = A_spks.sum(axis=0)
            self.status_var.set(
                f"Using spks: median {int(np.median(n_active_per_event))} "
                f"ROIs / event "
                f"(min {int(n_active_per_event.min())}, "
                f"max {int(n_active_per_event.max())}).")

    def _load_spks(self, plane0: Path, kept_idx: np.ndarray) -> np.ndarray:
        """Read ``spks.npy`` from a plane0 folder, subset to the kept
        ROIs, and transpose to ``(T, N_kept)``.

        Cached by ``(plane0, tuple(kept_idx))`` so repeated event-
        navigation clicks don't re-read the file. The actual
        orientation-detect + slicing logic lives in
        ``core.utils.s2p_load_spks``; this method just adds caching
        on top.
        """
        cache_key = (str(plane0), tuple(int(i) for i in kept_idx))
        cached = self._spks_cache.get(cache_key)
        if cached is not None:
            return cached
        from ...core.utils import s2p_load_spks
        spks_kept = s2p_load_spks(plane0, kept_idx)
        self._spks_cache[cache_key] = spks_kept
        return spks_kept

    @staticmethod
    def _first_time_from_spks(spks: np.ndarray,
                              event_windows: np.ndarray,
                              fps: float) -> tuple[np.ndarray, np.ndarray]:
        """Per-event first-activation time, derived from spks.

        For each event window ``[t0, t1]`` (seconds) and each ROI:

        * Slice the column to the corresponding frame range
          ``[f0, f1)`` where ``f0 = floor(t0 * fps)`` and
          ``f1 = ceil(t1 * fps)``.
        * Find the first frame where ``spks > 0`` -- that's the
          first deconvolved-spike estimate inside the window.
        * If no such frame exists, mark inactive (``A=False``,
          ``first_time=NaN``).

        Returns the same ``(A, first_time)`` shape Tab 5 publishes,
        ready to drop into ``self._data``.
        """
        T, N_kept = spks.shape
        n_events = int(event_windows.shape[0])
        A = np.zeros((N_kept, n_events), dtype=bool)
        first_time = np.full((N_kept, n_events), np.nan, dtype=np.float64)
        for ev in range(n_events):
            t0 = float(event_windows[ev, 0])
            t1 = float(event_windows[ev, 1])
            f0 = max(0, int(np.floor(t0 * fps)))
            f1 = min(T, int(np.ceil(t1 * fps)))
            if f1 <= f0:
                continue
            chunk = spks[f0:f1, :]              # (n_frames, N_kept)
            active_mask = chunk > 0.0           # any deconvolved spike
            any_active = active_mask.any(axis=0)
            if not any_active.any():
                continue
            # ``argmax`` on a boolean returns the first ``True`` index
            # per column. We only trust that index when ``any_active``
            # is True; otherwise the column is all-False and argmax
            # returns 0 spuriously.
            first_local = np.argmax(active_mask, axis=0)
            cols = np.where(any_active)[0]
            A[cols, ev] = True
            first_time[cols, ev] = (f0 + first_local[cols]) / fps
        return A, first_time

    # -- Event navigation ---------------------------------------------------

    def _update_nav_buttons(self) -> None:
        if self._data is None:
            self.prev_btn.config(state="disabled")
            self.next_btn.config(state="disabled")
            return
        n = int(self._data["event_windows"].shape[0])
        i = self._event_index
        self.prev_btn.config(
            state=("normal" if (n > 1 and i > 0) else "disabled"))
        self.next_btn.config(
            state=("normal" if (n > 1 and i < n - 1) else "disabled"))

    def _on_prev(self) -> None:
        if self._data is None:
            return
        if self._event_index > 0:
            self._event_index -= 1
            self.event_var.set(self._event_index + 1)
            self._render_event(self._event_index)
            self._update_nav_buttons()

    def _on_next(self) -> None:
        if self._data is None:
            return
        n = int(self._data["event_windows"].shape[0])
        if self._event_index < n - 1:
            self._event_index += 1
            self.event_var.set(self._event_index + 1)
            self._render_event(self._event_index)
            self._update_nav_buttons()

    def _on_spin(self) -> None:
        if self._data is None:
            return
        try:
            v = int(self.event_var.get())
        except (ValueError, tk.TclError):
            return
        n = int(self._data["event_windows"].shape[0])
        v = max(1, min(n, v))
        if v != self.event_var.get():
            self.event_var.set(v)
        new_idx = v - 1
        if new_idx == self._event_index:
            return
        self._event_index = new_idx
        self._render_event(self._event_index)
        self._update_nav_buttons()

    # -- Plotting -----------------------------------------------------------

    def _render_event(self, ev_idx: int) -> None:
        data = self._data
        if data is None:
            return
        ev = data["event_windows"]
        if ev_idx < 0 or ev_idx >= ev.shape[0]:
            return
        first_col = data["first_time"][:, ev_idx]
        active_col = data["A"][:, ev_idx]
        order_rank = spatial_helpers.order_map_for_event(
            first_col, active_col)
        img = spatial_helpers.paint_order_map(
            order_rank, data["stat_filtered"], data["Ly"], data["Lx"])

        t0 = float(ev[ev_idx, 0]); t1 = float(ev[ev_idx, 1])
        n_active = int(active_col.sum())
        n_total = int(active_col.size)
        frac = n_active / n_total if n_total else 0.0

        Ly, Lx = data["Ly"], data["Lx"]
        pix_to_um = data.get("pix_to_um", None)
        fps = data.get("fps", None)
        scale = float(pix_to_um) if pix_to_um is not None else 1.0
        # Suite2p stores ROI pixels with y=0 at the TOP of the FOV (image
        # convention). Render with origin="upper" + a flipped y-extent so
        # the on-screen orientation matches suite2p's GUI: y axis ticks go
        # 0 at the top -> Ly at the bottom, and centroid (cx, cy) lands at
        # the suite2p coordinate without needing to subtract from Ly.
        if pix_to_um is not None:
            extent = [0, Lx * scale, Ly * scale, 0]
            xlabel, ylabel = "X (µm)", "Y (µm)"
        else:
            extent = [0, Lx, Ly, 0]
            xlabel, ylabel = "X (px)", "Y (px)"
        fov_w = Lx * scale; fov_h = Ly * scale

        # Per-frame centroid + 2D-spread groups.
        frame_groups = []
        if fps:
            frame_groups = spatial_helpers.event_frame_centroids(
                first_col, active_col, data["stat_filtered"], float(fps))

        title_top_left = (
            f"Event {ev_idx + 1} / {ev.shape[0]}  "
            f"({t0:.2f}-{t1:.2f} s)\n"
            f"active = {n_active} / {n_total} ({100 * frac:.1f}%)"
        )
        title_top_right = (
            f"Event {ev_idx + 1} / {ev.shape[0]}  "
            f"({t0:.2f}-{t1:.2f} s)\n"
            f"{len(frame_groups)} active frame"
            f"{'' if len(frame_groups) == 1 else 's'}"
        )

        cxs = np.array([g["cx"] * scale for g in frame_groups])
        cys = np.array([g["cy"] * scale for g in frame_groups])

        def _paint_order(fig, ax_attr, title, with_arrows: bool):
            fig.clear()
            ax = fig.add_subplot(111)
            im = ax.imshow(
                img, origin="upper",
                cmap=spatial_helpers.CYAN_TO_RED,
                aspect="equal", vmin=0.0, vmax=1.0, extent=extent)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            cbar = fig.colorbar(
                im, ax=ax, fraction=0.04, pad=0.02,
                ticks=[0.0, 0.5, 1.0])
            cbar.ax.set_yticklabels(["earliest", "", "latest"])
            if with_arrows and frame_groups:
                ax.scatter(
                    cxs, cys, s=18, c="white",
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
                        zorder=5)
            setattr(self, ax_attr, ax)

        # Top-left: plain order map. Top-right: same map + arrows.
        _paint_order(self.fig_map_plain, "ax_map_plain",
                     title_top_left, with_arrows=False)
        _paint_order(self.fig_map_arrows, "ax_map_arrows",
                     title_top_right, with_arrows=True)

        # ---- Bottom panel: vectors-only with std circles ----
        self.fig_vec.clear()
        self.ax_vec = self.fig_vec.add_subplot(111)
        self.ax_vec.set_aspect("equal", adjustable="box")
        self.ax_vec.set_xlim(0, fov_w)
        self.ax_vec.set_ylim(0, fov_h)
        self.ax_vec.set_xlabel(xlabel)
        self.ax_vec.set_ylabel(ylabel)
        self.ax_vec.set_title(
            f"Frame-to-frame propagation  -  "
            f"{len(frame_groups)} active frame"
            f"{'' if len(frame_groups) == 1 else 's'}",
            fontsize=10)
        self.ax_vec.grid(True, color="0.92", linewidth=0.5)
        self.ax_vec.set_axisbelow(True)

        if frame_groups:
            n_groups = len(frame_groups)
            cmap = spatial_helpers.CYAN_TO_RED
            # The data ``sigma_xy = sqrt(var_x + var_y)`` is roughly
            # 1.41 standard deviations; we render only a fraction of
            # that as the circle radius and cap it so a single diffuse
            # frame can't blot out the FOV.
            sigma_radius_frac = 0.20
            min_r = 0.004 * max(fov_w, fov_h)
            max_r = 0.040 * max(fov_w, fov_h)
            # Colour-map by absolute frame index so the colorbar ticks
            # are meaningful even when active frames are non-contiguous.
            f_min = frame_groups[0]["frame"]
            f_max = frame_groups[-1]["frame"]
            f_span = max(f_max - f_min, 1)
            for g in frame_groups:
                cx = g["cx"] * scale
                cy = g["cy"] * scale
                r = g["sigma_xy"] * scale * sigma_radius_frac
                r = max(min_r, min(r, max_r))
                t = (g["frame"] - f_min) / f_span
                color = cmap(t)
                self.ax_vec.add_patch(Circle(
                    (cx, cy), r, facecolor=color, edgecolor="black",
                    alpha=0.65, linewidth=0.7, zorder=3))
            for i in range(n_groups - 1):
                x0, y0 = cxs[i], cys[i]
                x1, y1 = cxs[i + 1], cys[i + 1]
                if abs(x1 - x0) + abs(y1 - y0) < 1e-9:
                    continue
                self.ax_vec.annotate(
                    "", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle="->", color="black",
                        lw=1.4, shrinkA=4, shrinkB=4,
                        mutation_scale=12),
                    zorder=4)
            # Scale bar: same colormap as the circles, ticked with the
            # actual frame indices spanned by this event.
            from matplotlib import cm as _cm
            from matplotlib.colors import Normalize as _Normalize
            sm = _cm.ScalarMappable(
                norm=_Normalize(vmin=f_min, vmax=max(f_max, f_min + 1)),
                cmap=cmap)
            sm.set_array([])
            cbar = self.fig_vec.colorbar(
                sm, ax=self.ax_vec, fraction=0.04, pad=0.02)
            if f_max > f_min:
                ticks = sorted({f_min,
                                int(round((f_min + f_max) / 2)),
                                f_max})
            else:
                ticks = [f_min]
            cbar.set_ticks(ticks)
            cbar.ax.set_yticklabels([str(t) for t in ticks])
            cbar.set_label("frame")
        else:
            self.ax_vec.text(
                0.5, 0.5, "no per-frame activations",
                ha="center", va="center",
                transform=self.ax_vec.transAxes)

        # ---- Bottom-most panel: every-ROI-as-seed violin per Δframe ----
        self._render_dist_panel(
            first_col, active_col, data, ev_idx, fps, scale,
            xlabel_unit=("µm" if pix_to_um is not None else "px"))

        self.event_label_var.set(
            f"event {ev_idx + 1} / {ev.shape[0]}  "
            f"({t0:.2f}-{t1:.2f} s, {n_active}/{n_total} ROIs)")
        self.canvas_map_plain.draw_idle()
        self.canvas_map_arrows.draw_idle()
        self.canvas_vec.draw_idle()
        self.canvas_dist.draw_idle()

    def _render_dist_panel(self, first_col, active_col, data, ev_idx,
                           fps, scale, xlabel_unit) -> None:
        """One violin per Δframe, aggregated over every active ROI used
        as a seed.

        Seed sweep covers every active frame *up to the second-last*
        (ROIs in the last active frame are excluded as seeds, since
        they would only contribute same-frame, Δframe=0 distances).
        For each seed s with frame ``f_s`` and every *other* active ROI
        ``o`` whose frame satisfies ``f_o >= f_s``, we bin
        ``dist(s, o)`` at ``Δframe = f_o - f_s``. So a 5-ROI first
        frame contributes its 4 within-frame distances per seed
        (Δframe=0) plus all later frames at Δframe>0, a 10-ROI second
        frame likewise contributes its 9 within-frame distances per
        seed (with frame 1 cropped out) plus everything after, etc. --
        the violin at each Δframe is the union of those distance
        distributions across every valid seed.

        Inspired by ``plot_event_distance_vs_delta_t.py`` from the
        legacy Calcium_imaging_suite2p worktree, but generalised to use
        every ROI as a seed instead of just the earliest-firing one.
        """
        self.fig_dist.clear()
        self.ax_dist = self.fig_dist.add_subplot(111)

        sel = active_col & ~np.isnan(first_col)
        if not np.any(sel) or not fps:
            self.ax_dist.set_axis_off()
            self.ax_dist.text(
                0.5, 0.5,
                "no participating ROIs" if not np.any(sel)
                else "fps unknown; cannot bin onsets into frames",
                ha="center", va="center",
                transform=self.ax_dist.transAxes)
            return

        roi_local = np.where(sel)[0]
        n = int(roi_local.size)
        if n < 2:
            self.ax_dist.set_axis_off()
            self.ax_dist.text(
                0.5, 0.5,
                "need at least 2 active ROIs to form pairs",
                ha="center", va="center",
                transform=self.ax_dist.transAxes)
            return

        fps_f = float(fps)
        times = first_col[roi_local].astype(float)
        frames = np.rint(times * fps_f).astype(int)

        stat_filtered = data["stat_filtered"]
        xs = np.array([float(np.median(stat_filtered[i]["xpix"]))
                       for i in roi_local], dtype=float)
        ys = np.array([float(np.median(stat_filtered[i]["ypix"]))
                       for i in roi_local], dtype=float)

        # Pairwise Δframe and distance matrices. Rows are seeds, cols
        # are "other" ROIs; keep entries where Δframe >= 0, seed !=
        # other, and the seed's frame is strictly before the last
        # active frame (so the last-frame ROIs never get seeded).
        df_mat = frames[None, :] - frames[:, None]
        dx = xs[None, :] - xs[:, None]
        dy = ys[None, :] - ys[:, None]
        dist_mat = np.hypot(dx, dy) * float(scale)
        f_max_event = int(frames.max())
        seed_valid = frames < f_max_event           # (n,) bool
        keep = ((df_mat >= 0)
                & ~np.eye(n, dtype=bool)
                & seed_valid[:, None])
        df_flat = df_mat[keep]
        dist_flat = dist_mat[keep]

        if df_flat.size == 0:
            self.ax_dist.set_axis_off()
            self.ax_dist.text(
                0.5, 0.5,
                "all active ROIs share a single frame; "
                "no Δframe pairs to plot",
                ha="center", va="center",
                transform=self.ax_dist.transAxes)
            return

        unique_frames = sorted(set(int(f) for f in df_flat.tolist()))
        distributions = [dist_flat[df_flat == f] for f in unique_frames]
        f_min = unique_frames[0]
        f_max = unique_frames[-1]
        n_pairs = int(df_flat.size)

        parts = self.ax_dist.violinplot(
            distributions, positions=unique_frames, widths=0.8,
            showmeans=False, showmedians=True, showextrema=False,
        )
        for body in parts.get("bodies", []):
            body.set_facecolor("#4C72B0")
            body.set_edgecolor("#1f3b66")
            body.set_alpha(0.55)
        if "cmedians" in parts:
            parts["cmedians"].set_color("k")
            parts["cmedians"].set_linewidth(1.0)

        rng = np.random.default_rng(0)
        # Skip the per-pair dot scatter on huge events to keep the
        # rendering responsive; the violins are still informative.
        if n_pairs <= 4000:
            for f, dist_arr in zip(unique_frames, distributions):
                if dist_arr.size > 1:
                    jitter = rng.uniform(-0.18, 0.18, size=dist_arr.size)
                else:
                    jitter = np.zeros(1)
                self.ax_dist.scatter(
                    np.full_like(dist_arr, f, dtype=float) + jitter,
                    dist_arr,
                    s=6, color="black", alpha=0.25, edgecolor="none",
                    zorder=3)

        self.ax_dist.set_xlabel(
            f"Δ frame between seed and other ROI (fps={fps_f:.3f})")
        self.ax_dist.set_ylabel(f"pairwise distance ({xlabel_unit})")
        n_seeds = int(seed_valid.sum())
        self.ax_dist.set_title(
            f"Event {ev_idx + 1}  -  pairwise distance vs Δframe  "
            f"(every ROI in the first {n_seeds} of {n} as seed, "
            f"last-frame ROIs excluded; "
            f"{n_pairs} ordered pairs, "
            f"{len(unique_frames)} Δframe bin"
            f"{'' if len(unique_frames) == 1 else 's'})",
            fontsize=10)
        try:
            secax = self.ax_dist.secondary_xaxis(
                "top",
                functions=(lambda f, _fps=fps_f: f / _fps,
                           lambda s, _fps=fps_f: s * _fps),
            )
            secax.set_xlabel("Δt (s)")
        except Exception:
            pass
        self.ax_dist.grid(True, axis="y", alpha=0.25)
        if (f_max - f_min) <= 25:
            self.ax_dist.set_xticks(np.arange(f_min, f_max + 1))

