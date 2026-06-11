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

import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Optional

import customtkinter as ctk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .logic import spatial as spatial_helpers
from ...core import utils as core_utils

from ...gui_common import AppState, attach_fig_toolbar


class SpatialPropagationTab(ctk.CTkFrame):
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
        super().__init__(master, fg_color="transparent")
        self.state = state
        self._data: Optional[dict] = None  # see _ingest_results
        self._event_index: int = 0
        # The directional-monotonicity test (10k-shuffle permutation null)
        # is vectorised but heavy; it used to run synchronously on the
        # main thread and froze the GUI on every event-nav click. It now
        # runs on a background thread (the big matmul releases the GIL).
        # ``_mono_after_id`` debounces rapid prev/next so only the settled
        # event computes; ``_mono_token`` tags each request so a stale
        # thread's result for a previous event is dropped.
        self._mono_after_id: Optional[str] = None
        self._mono_token: int = 0
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
        header = ctk.CTkFrame(self)
        header.pack(fill="x", padx=10, pady=(10, 6))
        ctk.CTkLabel(
            header, text="Spatial propagation (events from tab 5)",
            font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(6, 0))

        # Persistent recording header so the user always knows which
        # plane0 the activation-order maps + monotonicity panels
        # come from. Updated by ``_ingest_results`` when Tab 5
        # publishes a new event_results dict.
        rec_row = ctk.CTkFrame(header, fg_color="transparent")
        rec_row.pack(fill="x", padx=8, pady=(0, 4))
        self.recording_var = tk.StringVar(value="No recording loaded.")
        ctk.CTkLabel(rec_row, textvariable=self.recording_var,
                     font=ctk.CTkFont(size=12, slant="italic")).pack(
            side="left")

        row = ctk.CTkFrame(header, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=2)
        self.status_var = tk.StringVar(
            value="Run Tab 5 first; this tab will populate automatically "
                  "once events are detected.")
        ctk.CTkLabel(row, textvariable=self.status_var,
                     font=ctk.CTkFont(size=11, slant="italic")).pack(
            side="left")
        ctk.CTkButton(row, text="Refresh from Tab 5",
                      command=self._refresh).pack(side="right")
        self.export_btn = ctk.CTkButton(
            row, text="Export figures", width=130,
            command=self._on_export_figures, state="disabled")
        self.export_btn.pack(side="right", padx=(0, 6))

        # ---- Spks override toggle ----
        # Tab 5's hysteresis-onset detector is the default activation-
        # time source. When this is checked, we instead read each
        # ROI's *first non-zero spks frame* inside each event window
        # and use that as the per-ROI first-onset time. Same painting
        # pipeline downstream; only the ``first_time`` /``A`` matrices
        # change. Mirrors the experimental ``spks`` signal kind in
        # ``Calcium_imaging_suite2p/lead_lag_prototype.py``.
        toggle_row = ctk.CTkFrame(header, fg_color="transparent")
        toggle_row.pack(fill="x", padx=8, pady=(2, 0))
        self._use_spks_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            toggle_row,
            text="Use Suite2p spks (deconvolved) instead of "
                 "hysteresis onsets for first-activation time",
            variable=self._use_spks_var,
            command=self._on_use_spks_toggle,
        ).pack(side="left")
        ctk.CTkLabel(
            toggle_row,
            text="(activation = first frame with spks > 0 inside the "
                 "event window)",
            text_color="gray", font=ctk.CTkFont(size=11, slant="italic"),
        ).pack(side="left", padx=(8, 0))

        # Event navigation row.
        nav = ctk.CTkFrame(header, fg_color="transparent")
        nav.pack(fill="x", padx=8, pady=(6, 6))
        self.prev_btn = ctk.CTkButton(
            nav, text="< Prev", command=self._on_prev, state="disabled",
            width=80)
        self.prev_btn.pack(side="left")
        self.event_var = tk.IntVar(value=1)
        # ttk.Spinbox: CTk doesn't ship its own spinbox primitive, so
        # we use the ttk version here for event navigation.
        self.event_spin = ttk.Spinbox(
            nav, from_=1, to=1, width=6, textvariable=self.event_var,
            command=self._on_spin)
        self.event_spin.pack(side="left", padx=(6, 6))
        self.event_spin.state(["disabled"])
        self.event_spin.bind("<Return>", lambda _e: self._on_spin())
        self.event_spin.bind("<FocusOut>", lambda _e: self._on_spin())
        self.next_btn = ctk.CTkButton(
            nav, text="Next >", command=self._on_next, state="disabled",
            width=80)
        self.next_btn.pack(side="left")
        self.event_label_var = tk.StringVar(value="")
        ctk.CTkLabel(nav, textvariable=self.event_label_var,
                     font=ctk.CTkFont(size=11, slant="italic")).pack(
            side="left", padx=(12, 0))

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        body.rowconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)

        # Top row, left: plain activation-order map (no arrows).
        f_map_plain = ctk.CTkFrame(body)
        f_map_plain.grid(row=0, column=0, sticky="nsew",
                         padx=(0, 2), pady=(0, 4))
        ctk.CTkLabel(
            f_map_plain,
            text="Activation order  (cyan = earliest, red = latest)",
            font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(6, 0))
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
            fill="both", expand=True, padx=4, pady=(0, 4))

        # Top row, right: same map with frame-to-frame arrows overlaid.
        f_map_arrows = ctk.CTkFrame(body)
        f_map_arrows.grid(row=0, column=1, sticky="nsew",
                          padx=(2, 0), pady=(0, 4))
        ctk.CTkLabel(
            f_map_arrows,
            text="Activation order + frame-to-frame propagation",
            font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(6, 0))
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
            fill="both", expand=True, padx=4, pady=(0, 4))

        # Bottom row: directional monotonicity test. Asks "is there a
        # monotone propagation axis at all, and how strong is it?" via
        # Spearman ρ on a planar projection of ROI positions vs
        # activation times, with a shuffle null for significance.
        #
        # Layout history (2026-05-10): a middle "vectors-only" panel
        # (frame-centroid circles with 2D-std radii + arrows on a blank
        # FOV) was dropped because its information was already conveyed
        # by the white-arrow overlay on the top-right activation-order
        # map, and a per-Δframe pairwise-distance violin was replaced
        # with this monotonicity test -- spread alone didn't tell us
        # about direction.
        body.rowconfigure(1, weight=1)
        f_dist = ctk.CTkFrame(body)
        f_dist.grid(row=1, column=0, columnspan=2, sticky="nsew",
                    pady=(4, 0))
        ctk.CTkLabel(
            f_dist,
            text="Directional monotonicity  "
                 "(Spearman ρ on projected position vs activation time, "
                 "shuffle null for p-value)",
            font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(6, 0))
        self.fig_dist = plt.Figure(figsize=(8, 4.2), tight_layout=True)
        self.ax_dist = self.fig_dist.add_subplot(111)
        self.ax_dist.set_axis_off()
        self.canvas_dist = FigureCanvasTkAgg(self.fig_dist, master=f_dist)
        self.toolbar_dist = attach_fig_toolbar(
            self.canvas_dist, f_dist, data_basename="monotonicity")
        self.canvas_dist.get_tk_widget().pack(fill="both", expand=True,
                                              padx=4, pady=(0, 4))

    # -- Tab 5 ingestion ----------------------------------------------------

    def _refresh(self) -> None:
        if self.state.event_results is None:
            messagebox.showinfo(
                "No events yet",
                "Click Render on Tab 5 first.")
            return
        self._on_event_results(self.state.event_results)

    def _on_export_figures(self, quiet: bool = False) -> None:
        """Render one PNG + SVG per event window into
        ``<save_folder>/calliope_figures/spatial_propagation/`` and
        refresh the manifest. Uses the in-memory ``self._data`` from
        Tab 5's most recent publish -- including any spks override
        currently active -- so the export matches what the user is
        scrolling through on the canvases.

        ``quiet=True`` (Tab 0 batch toggle) suppresses messageboxes.
        """
        if self._data is None:
            if not quiet:
                messagebox.showinfo(
                    "No events yet",
                    "Run Tab 5 first; this tab populates automatically "
                    "when events are detected.")
            return
        plane0 = self._data.get("plane0")
        if plane0 is None:
            if not quiet:
                messagebox.showerror(
                    "No plane0",
                    "Tab 5 publish did not include a plane0.")
            return
        plane0 = Path(plane0)
        save_folder = core_utils.save_folder_for_plane0(plane0)
        figures_root = save_folder / "calliope_figures"
        sp_dir = figures_root / "spatial_propagation"
        self.status_var.set("Exporting spatial-propagation figures ...")
        self.update_idletasks()
        try:
            from ...core.spatial import render_spatial_event_figures
            from ...core.export_manifest import write_export_manifest
            from ...core.utils import safe_recording_id
            rec_id = safe_recording_id(plane0)
            written = render_spatial_event_figures(
                plane0, self._data, figures_dir=sp_dir)
            params = {
                "use_spks_override": bool(self._use_spks_var.get()),
                "n_events": int(self._data["event_windows"].shape[0]),
            }
            manifest_path = write_export_manifest(
                figures_root,
                rec_id=rec_id, params=params,
                plane0=plane0, ckpt_path=None,
            )
            n_files = len(written) * 2  # PNG + SVG per event
            self.status_var.set(
                f"Exported {n_files} files -> {sp_dir}")
            if not quiet:
                messagebox.showinfo(
                    "Export complete",
                    f"Wrote {n_files} per-event spatial figures "
                    f"(PNG + SVG) to:\n  {sp_dir}\n\n"
                    f"Manifest:\n  {manifest_path}")
        except Exception as e:
            self.status_var.set(f"Export failed: {e}")
            if not quiet:
                messagebox.showerror("Export failed", str(e))

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

        if plane0 is not None:
            from ...core.utils import infer_recording_id
            try:
                rec_id = infer_recording_id(plane0)
            except Exception:
                rec_id = Path(plane0).parent.name
            self.recording_var.set(
                f"Recording: {rec_id}    plane0: {plane0}")

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

        # Precompute per-ROI centroids once: _render_event is called per
        # slider move and previously rebuilt these via an N_kept-length
        # np.median loop on every render.
        xs_all, ys_all = spatial_helpers._roi_centroids_xy(stat_filtered)
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
            "xs_all": xs_all,
            "ys_all": ys_all,
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
            self.prev_btn.configure(state="disabled")
            self.next_btn.configure(state="disabled")
            self.event_label_var.set("")
            for fig, canvas in (
                    (self.fig_map_plain, self.canvas_map_plain),
                    (self.fig_map_arrows, self.canvas_map_arrows),
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
        self.export_btn.configure(state="normal")
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
            if n_active_per_event.size == 0:
                # 0-event recording: A_spks is (N_kept, 0), so median/min/max
                # would raise on the empty array.
                self.status_var.set("Using spks: no events to summarize.")
            else:
                self.status_var.set(
                    f"Using spks: median "
                    f"{int(np.median(n_active_per_event))} ROIs / event "
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
            self.prev_btn.configure(state="disabled")
            self.next_btn.configure(state="disabled")
            return
        n = int(self._data["event_windows"].shape[0])
        i = self._event_index
        self.prev_btn.configure(
            state=("normal" if (n > 1 and i > 0) else "disabled"))
        self.next_btn.configure(
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
                first_col, active_col, data["stat_filtered"], float(fps),
                xs_all=data.get("xs_all"), ys_all=data.get("ys_all"))

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
            if pix_to_um is not None:
                from ...core.figscale import add_scale_bar
                add_scale_bar(ax, pix_to_um, axes_in_um=True,
                              span_um=extent[1], color="white")
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

        # ---- Bottom panel: directional monotonicity test ----
        # See the layout-history note in ``_build_ui`` for why this
        # bottom panel is the monotonicity test rather than a
        # vectors-only spatial layout.
        self._render_monotonicity_panel(
            first_col, active_col, data, ev_idx, fps, scale,
            xlabel_unit=("µm" if pix_to_um is not None else "px"))

        self.event_label_var.set(
            f"event {ev_idx + 1} / {ev.shape[0]}  "
            f"({t0:.2f}-{t1:.2f} s, {n_active}/{n_total} ROIs)")
        self.canvas_map_plain.draw_idle()
        self.canvas_map_arrows.draw_idle()
        self.canvas_dist.draw_idle()

    def _render_monotonicity_panel(self, first_col, active_col, data,
                                   ev_idx, fps, scale, xlabel_unit) -> None:
        """Directional-monotonicity panel.

        Tests whether the active ROIs' activation times propagate in a
        coherent direction:

        1. Sweep θ ∈ [0, 2π). Project each ROI's (x, y) onto
           ``u(θ) = (cos θ, sin θ)`` to get a 1-D coordinate
           ``s_i(θ)``.
        2. Compute Spearman's rank correlation
           ``ρ(θ) = SpearmanCorr(s_i(θ), t_i)`` per θ. Spearman is
           rank-only, so any monotone propagation along ``u(θ)`` reads
           as |ρ| → 1; the sign tells us which way along that axis.
        3. ``ρ_obs = ρ(θ*)`` where θ* maximises ρ. Magnitude is the
           monotonicity strength; near 0 means no direction, near 1
           means perfectly monotone.
        4. Permutation test for significance: at θ*, shuffle the
           activation times across cells ``N`` times and recompute
           ρ_k. The fraction of shuffles with ``ρ_k >= ρ_obs`` is the
           empirical p-value.

        The panel is three subplots:

        * **(a)** Scatter of ROI positions coloured by activation
          time; arrow along ``u(θ*)`` scaled by ``ρ_obs`` shows the
          dominant axis (no arrow when ρ_obs ≤ 0).
        * **(b)** ``ρ(θ)`` curve with the maximum marked.
        * **(c)** Permutation null histogram with ``ρ_obs`` overlaid
          as a vertical line and the p-value in the title.

        Heavy lifting (Spearman sweep + shuffle null) lives in
        :func:`spatial_helpers.directional_monotonicity_spearman` so
        a headless caller could reuse it without dragging in the GUI.
        """
        self.fig_dist.clear()

        # Invalidate any in-flight worker for a previous event and cancel
        # a pending debounced launch. Done up front so the early-return
        # guards below (no/too-few ROIs) also drop a stale result instead
        # of letting it paint over the guard message.
        self._mono_token += 1
        if self._mono_after_id is not None:
            try:
                self.after_cancel(self._mono_after_id)
            except Exception:
                pass
            self._mono_after_id = None

        sel = active_col & ~np.isnan(first_col)
        if not np.any(sel):
            self.ax_dist = self.fig_dist.add_subplot(111)
            self.ax_dist.set_axis_off()
            self.ax_dist.text(
                0.5, 0.5, "no participating ROIs",
                ha="center", va="center",
                transform=self.ax_dist.transAxes)
            return

        roi_local = np.where(sel)[0]
        n = int(roi_local.size)
        if n < 3:
            self.ax_dist = self.fig_dist.add_subplot(111)
            self.ax_dist.set_axis_off()
            self.ax_dist.text(
                0.5, 0.5,
                "need >= 3 active ROIs for monotonicity test",
                ha="center", va="center",
                transform=self.ax_dist.transAxes)
            return

        # Pixel positions; the test is rotation-invariant so the px↔µm
        # conversion only matters for the scatter axis labels. Reuse the
        # per-ROI centroids _ingest_results precomputed once into
        # xs_all/ys_all (same np.median over stat_filtered, positionally
        # aligned), instead of recomputing 2*N_active medians on every
        # event-nav click. Fall back to the per-call loop for older _data.
        stat_filtered = data["stat_filtered"]
        xs_all = data.get("xs_all")
        ys_all = data.get("ys_all")
        if xs_all is not None and ys_all is not None:
            xs = np.asarray(xs_all, dtype=float)[roi_local]
            ys = np.asarray(ys_all, dtype=float)[roi_local]
        else:
            xs = np.array([float(np.median(stat_filtered[i]["xpix"]))
                           for i in roi_local], dtype=float)
            ys = np.array([float(np.median(stat_filtered[i]["ypix"]))
                           for i in roi_local], dtype=float)
        ts_seconds = first_col[roi_local].astype(float)
        # Spearman only cares about ordering, so either frame index or
        # seconds works as t. Use frames when fps is known (matches the
        # frame-binning used elsewhere in this tab) so colorbar ticks
        # read as integers; fall back to seconds otherwise.
        if fps:
            ts = np.rint(ts_seconds * float(fps)).astype(int)
            t_label = "activation frame"
        else:
            ts = ts_seconds
            t_label = "activation time (s)"

        # The Spearman sweep + 10k-shuffle null is vectorised (one big
        # matmul) so numpy releases the GIL -- run it on a background
        # thread instead of the main thread so the GUI doesn't freeze on
        # every event-nav click. Debounce rapid prev/next (only the
        # settled event computes) and tag the request so a stale result
        # for a previous event is dropped. A process would be wrong here:
        # the ~3.4s spawn per click would make event browsing unusable.
        ctx = {"xs": xs, "ys": ys, "ts": ts, "scale": scale,
               "xlabel_unit": xlabel_unit, "n": n, "t_label": t_label,
               "ev_idx": ev_idx}

        # Placeholder while the worker runs (cleared above).
        self.ax_dist = self.fig_dist.add_subplot(111)
        self.ax_dist.set_axis_off()
        self.ax_dist.text(
            0.5, 0.5, "computing directional monotonicity ...",
            ha="center", va="center", transform=self.ax_dist.transAxes)

        # Token already bumped at the top of this method; capture it so
        # the worker's result is dropped if the user navigates onward.
        token = self._mono_token
        self._mono_after_id = self.after(
            150, lambda: self._start_mono_worker(token, ctx))

    def _start_mono_worker(self, token: int, ctx: dict) -> None:
        """Run the monotonicity sweep off the main thread, then marshal
        the result back via ``after(0, ...)``. Stale results (the user
        navigated to another event) are dropped in ``_on_mono_done``."""
        self._mono_after_id = None

        def worker():
            try:
                result = spatial_helpers.directional_monotonicity_spearman(
                    ctx["xs"], ctx["ys"], ctx["ts"],
                    n_angles=360, n_shuffles=10000, seed=0)
            except Exception:
                result = None
            try:
                self.after(0, lambda: self._on_mono_done(token, result, ctx))
            except Exception:
                pass

        threading.Thread(target=worker, daemon=True).start()

    def _on_mono_done(self, token: int, result, ctx: dict) -> None:
        """Draw the monotonicity panels on the main thread (or drop a
        stale result if the user has since navigated to another event)."""
        if token != self._mono_token:
            return
        self.fig_dist.clear()
        if not result:
            self.ax_dist = self.fig_dist.add_subplot(111)
            self.ax_dist.set_axis_off()
            self.ax_dist.text(
                0.5, 0.5,
                "monotonicity test undefined "
                "(no spread in t or in x/y)",
                ha="center", va="center",
                transform=self.ax_dist.transAxes)
            self.canvas_dist.draw_idle()
            return
        self._draw_monotonicity_panels(result, ctx)
        self.canvas_dist.draw_idle()

    def _draw_monotonicity_panels(self, result: dict, ctx: dict) -> None:
        """Render the three monotonicity subplots from a computed result.

        Split out of :meth:`_render_monotonicity_panel` so the drawing
        runs on the main thread once the background sweep returns.
        """
        xs = ctx["xs"]; ys = ctx["ys"]; ts = ctx["ts"]
        scale = ctx["scale"]; xlabel_unit = ctx["xlabel_unit"]
        n = ctx["n"]; t_label = ctx["t_label"]; ev_idx = ctx["ev_idx"]

        thetas = result["thetas"]
        rho_curve = result["rho_curve"]
        theta_star = float(result["theta_star"])
        rho_obs = float(result["rho_obs"])
        null_rho = result["null_rho"]
        p_value = float(result["p_value"])

        gs = self.fig_dist.add_gridspec(1, 3, width_ratios=[1.0, 1.1, 1.0])
        ax_xy = self.fig_dist.add_subplot(gs[0, 0])
        ax_curve = self.fig_dist.add_subplot(gs[0, 1])
        ax_null = self.fig_dist.add_subplot(gs[0, 2])
        # Keep self.ax_dist pointing at *something* — downstream code
        # (status updates, fig save) references it; pick the curve
        # since that's the panel's signature image.
        self.ax_dist = ax_curve

        # (a) Scatter coloured by activation time, with a θ* arrow.
        # Units on this subplot follow the rest of Tab 8 (µm when
        # available); the arrow's length encodes ρ_obs so a weak
        # direction looks short and a strong one stretches across.
        xs_disp = xs * float(scale)
        ys_disp = ys * float(scale)
        sc = ax_xy.scatter(
            xs_disp, ys_disp, c=ts, s=30,
            cmap=spatial_helpers.CYAN_TO_RED,
            edgecolors="black", linewidths=0.4)
        if rho_obs > 0:
            cx = float(np.mean(xs_disp))
            cy = float(np.mean(ys_disp))
            span = max(
                float(np.ptp(xs_disp)), float(np.ptp(ys_disp)), 1.0)
            half = 0.45 * span * rho_obs
            ax_xy.annotate(
                "",
                xy=(cx + half * np.cos(theta_star),
                    cy + half * np.sin(theta_star)),
                xytext=(cx - half * np.cos(theta_star),
                        cy - half * np.sin(theta_star)),
                arrowprops=dict(arrowstyle="->", color="black",
                                lw=1.6, mutation_scale=16))
        ax_xy.set_aspect("equal")
        ax_xy.set_xlabel(f"X ({xlabel_unit})")
        ax_xy.set_ylabel(f"Y ({xlabel_unit})")
        ax_xy.set_title(
            f"ROI positions  ({n} active)\ncolor = {t_label}",
            fontsize=9)
        ax_xy.invert_yaxis()                              # match Tab 8 orientation
        cb = self.fig_dist.colorbar(sc, ax=ax_xy, fraction=0.045,
                                    pad=0.02)
        cb.ax.tick_params(labelsize=8)

        # (b) ρ(θ) curve.
        theta_deg = np.degrees(thetas)
        ax_curve.plot(theta_deg, rho_curve, color="#1f3b66", lw=1.2)
        ax_curve.axhline(0.0, color="gray", lw=0.6, alpha=0.6)
        ax_curve.axvline(np.degrees(theta_star), color="red",
                         lw=1.0, ls="--", alpha=0.8)
        ax_curve.scatter([np.degrees(theta_star)], [rho_obs],
                         color="red", s=30, zorder=4)
        ax_curve.set_xlim(0, 360)
        ax_curve.set_ylim(min(-1.0, float(rho_curve.min()) - 0.05),
                          max(1.0, float(rho_curve.max()) + 0.05))
        ax_curve.set_xticks([0, 90, 180, 270, 360])
        ax_curve.set_xlabel("Projection direction θ (deg)")
        ax_curve.set_ylabel("Spearman ρ(θ)")
        ax_curve.set_title(
            f"Event {ev_idx + 1}: ρ* = {rho_obs:+.2f}  "
            f"@ θ* = {np.degrees(theta_star):.0f}°",
            fontsize=10)
        ax_curve.grid(True, alpha=0.25)

        # (c) Permutation null histogram.
        ax_null.hist(null_rho, bins=40, color="#999999",
                     edgecolor="#555555", alpha=0.85)
        ax_null.axvline(rho_obs, color="red", lw=1.6,
                        label=f"ρ_obs = {rho_obs:+.2f}")
        ax_null.set_xlim(-1.0, 1.0)
        ax_null.set_xlabel("Spearman ρ (shuffled t)")
        ax_null.set_ylabel("count")
        ax_null.set_title(
            f"Permutation null (N = {null_rho.size})\n"
            f"p = {p_value:.3f}", fontsize=9)
        ax_null.legend(loc="upper left", fontsize=8, frameon=False)
        ax_null.grid(True, alpha=0.25)

