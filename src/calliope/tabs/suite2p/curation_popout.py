"""Per-ROI curation popout for Tab 3.

What it shows
-------------
When the user clicks an ROI on Tab 3's "All detected ROIs" panel,
this Toplevel opens with one panel per available background image
(mean / max / correlation / reference / channel 2, whichever the
recording carries), plus a ROI-footprint panel along the top row
and a ΔF/F trace spanning the bottom row.

A status row beneath the figure shows the current iscell flag and
the model's p(cell) when available. A **Cell** / **Not cell** flip
pair persists the curator's verdict, and a **Retrain cell filter**
button (disabled until the user has flipped at least one
classification this session) re-trains the model on the updated
labels CSV in a worker thread, warm-starting from Tab 3's selected
checkpoint when the parent supplies one.

Side effects of a flip
----------------------
1. Mutates ``<plane0>/iscell.npy[roi, 0]`` to the new value.
2. Upserts one row in the curation CSV
   (``cellfilter.config.LABELS_CSV``, by default the in-project
   ``calliope/data/cellfilter_labels.csv``, created lazily) keyed on
   ``(plane0_path, ROI_number)``. Columns follow
   ``cellfilter.config.LABELS_CSV_COLUMNS``: ``plane0_path,
   recording_ID, ROI_number, user_defined_cell, timestamp_iso``;
   ``plane0_path`` is what the trainer resolves, ``recording_ID`` is
   for human auditing only. This is the same CSV
   ``cellfilter.train.main()`` reads, so retraining picks up the new
   labels immediately.
3. Tracks the flip in ``self._flipped`` so the parent Tab 3 can
   refresh its overlay panels when the popout closes (or when the
   user moves to another ROI).

Single-instance pattern
-----------------------
Tab 3 keeps one ``CurationPopout`` per plane0. Subsequent clicks on
new ROIs call :meth:`show_roi` on the existing popout instead of
opening a second window.
"""

from __future__ import annotations

import csv
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from typing import Callable, Optional

import customtkinter as ctk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk,
)
from matplotlib.patches import Rectangle

from ...core.utils import (DISPLAY_CLIP_HIGH_PCT, DISPLAY_CLIP_LOW_PCT,
                           lowpass_zero_phase_1d)
from ...gui_common import install_scroll_router


def _open_dff(plane0: Path, n_total: int) -> tuple[Optional[np.memmap], int]:
    """Open the unfiltered all-ROI dF/F memmap for trace display.

    Returns ``(dff, T)`` or ``(None, 0)`` when
    ``r0p7_dff.memmap.float32`` (one column per Suite2p ROI, the only
    prefix any writer uses) is missing or its size is not a whole
    number of ``n_total``-ROI frames. The trace panel then shows a
    "no dF/F" placeholder; there is no F.npy fallback.
    """
    path = plane0 / "r0p7_dff.memmap.float32"
    if not path.exists() or n_total <= 0:
        return None, 0
    floats = path.stat().st_size // 4
    if floats % n_total != 0:
        return None, 0
    T = floats // n_total
    try:
        return (np.memmap(path, dtype="float32", mode="r",
                          shape=(T, n_total)),
                T)
    except OSError:
        return None, 0


def _fps_for(plane0: Path, fallback: Optional[float] = None) -> float:
    from ...core import utils
    if fallback is None:
        fallback = utils.DEFAULT_FPS
    try:
        return float(utils.get_fps_from_notes(str(plane0),
                                              default_fps=fallback))
    except Exception:
        return fallback


def _curation_csv_path() -> Path:
    """Resolve the curation CSV path the cell-filter trainer reads.

    Imports the path lazily from ``cellfilter.config`` so we always
    pick up the active configuration. As of 2026-05-12 this resolves
    to ``src/calliope/data/cellfilter_labels.csv`` (in-project) by
    default.
    """
    from ...core.cellfilter import config as _cf_cfg
    return Path(_cf_cfg.LABELS_CSV)


def _csv_columns() -> tuple[str, ...]:
    """Pull the canonical column list from ``cellfilter.config`` so
    the popout writer never drifts from what ``dataset.load_labels``
    expects."""
    from ...core.cellfilter import config as _cf_cfg
    return tuple(_cf_cfg.LABELS_CSV_COLUMNS)


def _ensure_csv_header(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(list(_csv_columns()))


def _upsert_label(csv_path: Path, plane0_path: Path,
                  rec_id: str, roi: int, label: int) -> None:
    """Upsert one curation row in the labels CSV.

    Canonical schema: ``plane0_path, recording_ID, ROI_number,
    user_defined_cell, timestamp_iso``. The ``plane0_path`` column
    is the source of truth for which recording the ROI belongs to;
    ``recording_ID`` is kept for human-readable auditing.

    Behavior:
      - If a row already exists for ``(plane0_path, roi)``, replace
        it in place (label + timestamp updated). The CSV stays
        compact -- one row per labelled ROI instead of an
        append-only history.
      - Otherwise append a new row.

    Why upsert rather than append: ``dataset.load_labels`` already
    de-dupes via ``drop_duplicates(keep="last")``, so the trainer
    wouldn't be confused either way -- but a curator who flips an
    ROI from "cell" to "not cell" then back to "cell" would
    accumulate three rows on append, and the CSV becomes hard to
    audit. Rewriting the whole file on each flip is fine: the CSV
    is at most a few thousand rows of plain text (~tens of KB), so
    a full rewrite is microseconds on any modern SSD.
    """
    import datetime as _dt
    _ensure_csv_header(csv_path)
    ts = _dt.datetime.now().isoformat(timespec="seconds")
    target_key = (str(plane0_path), int(roi))
    new_row = [str(plane0_path), str(rec_id), int(roi),
               int(label), ts]

    # Read every existing row, swap in the new one where it
    # matches, else fall through and append at the end.
    header: list[str] | None = None
    rows: list[list[str]] = []
    found = False
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            header = list(_csv_columns())
        for row in reader:
            if not row:
                continue
            # Index 0 = plane0_path, index 2 = ROI_number. Tolerate
            # malformed rows (hand edits, partial writes) by passing
            # them through unchanged.
            try:
                key = (row[0], int(row[2]))
            except (IndexError, ValueError):
                rows.append(row)
                continue
            if key == target_key:
                rows.append(new_row)
                found = True
            else:
                rows.append(row)
    if not found:
        rows.append(new_row)

    if header is None:
        header = list(_csv_columns())
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


class CurationPopout(ctk.CTkToplevel):
    """Toplevel curation window for one Suite2p ROI.

    Reuse by calling :meth:`show_roi` to switch focus to a different
    ROI (without re-creating Tk widgets / matplotlib axes).
    """

    # Backgrounds we know how to display, in preferred-listing order.
    # The popout shows one panel per AVAILABLE background (plus a
    # dedicated ROI-footprint panel) so the user can compare the
    # same ROI against every projection the recording carries.
    # Mirrors ``Suite2pTab.KNOWN_BG_IMAGES`` -- keep the two lists in
    # sync if a new background type is added upstream.
    _KNOWN_BACKGROUNDS: tuple[tuple[str, str], ...] = (
        ("meanImg",       "Mean"),
        ("meanImgE",      "Mean (enhanced)"),
        ("max_proj",      "Max projection"),
        ("Vcorr",         "Correlation"),
        ("refImg",        "Reg. ref"),
        ("meanImg_chan2", "Mean ch2"),
    )

    def __init__(self, master, plane0: Path, stat, ops: dict,
                 *, on_iscell_changed=None,
                 get_size_outlier_z: Optional[Callable[[], float]] = None,
                 get_ckpt_path: Optional[Callable[[], str]] = None,
                 on_checkpoint_trained: Optional[
                     Callable[[Path], None]] = None) -> None:
        """Build the popout.

        Optional parent hooks (all called on the Tk main thread):

        ``get_size_outlier_z``
            Returns the parent's active ``size_outlier_z`` so the
            post-Retrain re-predict applies the same size gate as the
            parent's detection run (``0`` = off). ``None`` -> the
            ``cellfilter.config.SIZE_OUTLIER_Z`` default.
        ``get_ckpt_path``
            Returns the parent's selected cell-filter checkpoint.
            Retrain warm-starts from it (``config.WARM_START_LR``)
            when it exists; ``None`` / missing file -> train from
            random init with ``config.LR``.
        ``on_checkpoint_trained``
            Called with the new ``CHECKPOINT_DIR/best.pt`` after a
            Retrain that wrote one, so the parent can select it for
            later detection runs.
        """
        super().__init__(master)
        self.title("ROI curation")
        self.geometry("1500x720")
        self.wm_minsize(960, 520)

        self._plane0 = Path(plane0)
        self._stat = stat
        self._ops = ops
        self._mean_img = np.asarray(ops["meanImg"])
        # Pre-collect every background image the recording supplies
        # (as ``(key, label, padded_ndarray)`` tuples). All are
        # already pad-aligned to the mean-image's full FOV so the
        # ROI locator rectangle + ROI scatter overlay land at the
        # same pixel coordinates on every panel. Computed once
        # because the underlying ``ops`` dict doesn't change for a
        # single popout instance.
        self._backgrounds: list[tuple[str, str, np.ndarray]] = (
            self._collect_backgrounds(ops, self._mean_img))
        self._n_total = len(stat)
        self._dff, self._T = _open_dff(self._plane0, self._n_total)
        self._fps = _fps_for(self._plane0)
        self._rec_id = self._derive_recording_id(self._plane0)
        self._get_size_outlier_z = get_size_outlier_z
        self._get_ckpt_path = get_ckpt_path
        self._on_checkpoint_trained = on_checkpoint_trained
        self._csv_path = _curation_csv_path()
        # Listener fired after every successful flip so the parent
        # tab can repaint its overlays without us reaching back.
        self._on_iscell_changed = on_iscell_changed

        self._iscell = self._load_iscell()
        self._probs = self._load_probs()

        # Set of ROI indices flipped this session — gates the Retrain
        # button. We don't undo flips on close; the iscell.npy + CSV
        # writes are persistent.
        self._flipped: set[int] = set()
        self._train_thread: Optional[threading.Thread] = None
        self._train_state_after_id: Optional[str] = None

        self._roi_idx: Optional[int] = None
        # Persists across show_roi() calls so a curator sweeping a
        # queue can enable the 1 Hz overlay once and see it on every
        # subsequent ROI without re-clicking.
        self._show_lowpass: bool = False
        self._build_ui()
        # Forward Esc to a graceful close.
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Escape>", lambda _e: self._on_close())
        # Keyboard shortcuts: 1 = cell, 0 = not a cell. Bound at
        # the Toplevel level for the case where focus is on the Tk
        # side of the window (status bar, button strip).
        self.bind("1", lambda _e: self._set_label(1))
        self.bind("0", lambda _e: self._set_label(0))
        # And again via matplotlib's own key-event channel for the
        # case where focus is on the figure canvas or the toolbar.
        # The NavigationToolbar consumes Tk keyboard events on its
        # child buttons, so Tk-level digit bindings get dropped
        # intermittently when the cursor hovers over the toolbar.
        # ``FigureCanvasBase.mpl_connect`` is independent of Tk
        # focus -- it fires whenever the mouse is over the figure
        # -- so as long as the user is looking at the panels, the
        # digit keys work reliably.
        self._canvas.mpl_connect(
            "key_press_event", self._on_mpl_key)
        # Make the canvas focus-takeable so it actually receives key
        # events when clicked.
        self._canvas.get_tk_widget().configure(takefocus=True)
        # Consume wheel events so they don't leak through CTk's
        # global bind_all and scroll the main app behind this popout.
        install_scroll_router(self)

    @classmethod
    def _collect_backgrounds(cls, ops: dict, mean_img: np.ndarray
                             ) -> list[tuple[str, str, np.ndarray]]:
        """Return ``[(key, label, img)]`` for every background image
        the recording carries.

        - Iterates ``_KNOWN_BACKGROUNDS`` in display order.
        - Drops entries whose ``ops`` value is missing or not a 2D
          array.
        - Pads any cropped projection (suite2p crops ``max_proj`` to
          the registration ``y/xrange``) back to the full mean-image
          FOV so locator rectangles + ROI overlays line up across
          every panel.
        """
        out: list[tuple[str, str, np.ndarray]] = []
        H, W = mean_img.shape
        y0 = int(np.asarray(ops.get("yrange", [0, H]))[0])
        x0 = int(np.asarray(ops.get("xrange", [0, W]))[0])
        for key, label in cls._KNOWN_BACKGROUNDS:
            img = ops.get(key)
            if not isinstance(img, np.ndarray) or img.ndim != 2:
                continue
            img = np.asarray(img, dtype=np.float32)
            if img.shape != (H, W):
                padded = np.zeros((H, W), dtype=np.float32)
                mh, mw = img.shape
                padded[y0:y0 + mh, x0:x0 + mw] = img
                img = padded
            out.append((key, label, img))
        return out

    # -- Layout -------------------------------------------------------------

    def _build_ui(self) -> None:
        # Figure: top row = one panel per available background +
        # one panel for the ROI footprint; bottom row = ΔF/F trace
        # spanning every column. The panel count is dynamic so a
        # recording carrying ``meanImgE`` / ``Vcorr`` / ``refImg``
        # gets a dedicated panel for each background it actually has.
        n_bg = len(self._backgrounds)
        # Always reserve one column for the footprint panel; if no
        # backgrounds are available (unusual) we still want a 1-col
        # grid so add_gridspec doesn't get ncols=0.
        ncols = max(1, n_bg + 1)
        # Width scales with panel count so each panel stays roughly
        # square in the default window. ~3 in per panel + ~2 in for
        # the toolbar padding.
        fig_w = max(12.0, 3.0 * ncols)
        self._fig = plt.Figure(figsize=(fig_w, 6.6), tight_layout=True)
        gs = self._fig.add_gridspec(2, ncols, height_ratios=[3, 1.5])
        self._img_axes = [self._fig.add_subplot(gs[0, c])
                          for c in range(ncols)]
        self._trace_ax = self._fig.add_subplot(gs[1, :])
        self._canvas = FigureCanvasTkAgg(self._fig, master=self)
        # Matplotlib's built-in NavigationToolbar gives the curator
        # pan / box-zoom / home / save controls. Zooming on the
        # trace subplot is the primary use: when a recording has
        # one big event among thousands of frames, the user wants
        # to drag-zoom into it to inspect baseline. Toolbar applies
        # to the whole figure but works per-axes -- selecting the
        # trace and dragging zooms only the trace.
        self._toolbar = NavigationToolbar2Tk(
            self._canvas, self, pack_toolbar=False)
        self._toolbar.update()
        # Dark-skin the toolbar to match the rest of the popout chrome.
        from ...gui_common import restyle_matplotlib_toolbar
        restyle_matplotlib_toolbar(self._toolbar)
        # Drop focus-traversal from every toolbar child so clicking
        # a pan/zoom tool doesn't move keyboard focus onto the
        # toolbar. The tool's mouse handlers still fire normally
        # (they're bound on mouse-button events, not focus); we
        # just keep the keyboard-focus path glued to the popout so
        # ``1`` / ``0`` / arrow-key bindings keep working after
        # the user adjusts a zoom box.
        for _child in self._toolbar.winfo_children():
            try:
                _child.configure(takefocus=False)
            except tk.TclError:
                pass
        self._toolbar.pack(side="top", fill="x")
        self._canvas.get_tk_widget().pack(side="top", fill="both",
                                          expand=True)

        # Status + control row.
        status_frame = ctk.CTkFrame(self, fg_color="transparent")
        status_frame.pack(side="top", fill="x", padx=6, pady=6)
        self._status_var = tk.StringVar(value="No ROI selected.")
        ctk.CTkLabel(status_frame, textvariable=self._status_var,
                     anchor="w").pack(side="left", fill="x", expand=True)

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(side="top", fill="x", padx=6, pady=(0, 6))
        self._cell_btn = ctk.CTkButton(
            btn_frame, text="Cell (1)", width=130,
            command=lambda: self._set_label(1))
        self._cell_btn.pack(side="left", padx=(0, 6))
        self._not_cell_btn = ctk.CTkButton(
            btn_frame, text="Not a cell (0)", width=130,
            command=lambda: self._set_label(0))
        self._not_cell_btn.pack(side="left", padx=(0, 6))
        # Retrain stays disabled until the user has flipped at least
        # one ROI's classification this session.
        self._retrain_btn = ctk.CTkButton(
            btn_frame, text="Retrain cell filter", width=180,
            state="disabled", command=self._on_retrain)
        self._retrain_btn.pack(side="left", padx=(20, 0))
        # Promote stays disabled until the user has flipped at least
        # one ROI; it overlays those flips onto predicted_cell_mask.npy
        # so downstream tabs honour them without a full retrain.
        self._promote_btn = ctk.CTkButton(
            btn_frame, text="Promote to filter mask", width=200,
            state="disabled", command=self._on_promote)
        self._promote_btn.pack(side="left", padx=(6, 0))

        self._train_status_var = tk.StringVar(value="")
        ctk.CTkLabel(btn_frame, textvariable=self._train_status_var
                     ).pack(side="left", padx=(10, 0))

        ctk.CTkButton(btn_frame, text="Close (Esc)", width=110,
                      command=self._on_close).pack(side="right")
        # Packed AFTER Close with side="right" so it lands left-of-
        # Close (right-side packs build right-to-left).
        self._lowpass_btn = ctk.CTkButton(
            btn_frame, text="Show 1 Hz lowpass", width=170,
            command=self._on_toggle_lowpass)
        self._lowpass_btn.pack(side="right", padx=(0, 6))

    # -- Public ------------------------------------------------------------

    def show_roi(self, roi_idx: int) -> None:
        """Switch focus to ``roi_idx`` and repaint the panels."""
        if roi_idx is None or not (0 <= int(roi_idx) < self._n_total):
            return
        self._roi_idx = int(roi_idx)
        self._render_panels()
        self._refresh_status()
        # Bring the popout to the front so a click on Tab 3 always
        # surfaces the inspection panels.
        self.deiconify()
        self.lift()
        self.focus_set()

    # -- Render ------------------------------------------------------------

    def _render_panels(self) -> None:
        if self._roi_idx is None:
            return
        s = self._stat[self._roi_idx]
        xpix = np.asarray(s["xpix"])
        ypix = np.asarray(s["ypix"])
        if xpix.size == 0 or ypix.size == 0:
            return

        for ax in self._img_axes:
            ax.clear()
        self._trace_ax.clear()

        x0, x1 = int(xpix.min()), int(xpix.max())
        y0, y1 = int(ypix.min()), int(ypix.max())
        pad = 8

        def _locator(ax) -> None:
            ax.add_patch(Rectangle(
                (x0 - pad, y0 - pad),
                (x1 - x0) + 2 * pad, (y1 - y0) + 2 * pad,
                fill=False, edgecolor="yellow", linewidth=1.2))

        # One panel per available background: background image +
        # yellow locator rectangle + faint red ROI scatter so the
        # user can confirm the ROI's pixels at the location all
        # projections agree on. The first panel's title carries the
        # recording id so the user knows which dataset they're
        # curating without checking the parent tab's header.
        for ax, (_key, label, img) in zip(self._img_axes,
                                          self._backgrounds):
            vmin, vmax = np.percentile(
                img, [DISPLAY_CLIP_LOW_PCT, DISPLAY_CLIP_HIGH_PCT])
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
            ax.scatter(xpix, ypix, s=1.4, c="red", alpha=0.55,
                       edgecolors="none")
            _locator(ax)
            title = label
            if _key == self._backgrounds[0][0]:
                title = f"{label}  ({self._rec_id})"
            ax.set_title(title)
            ax.set_xticks([]); ax.set_yticks([])

        # Trailing panel: ROI footprint. Always the last subplot so
        # the curator can see the pure mask shape regardless of
        # which projection they're comparing it against.
        ax_fp = self._img_axes[len(self._backgrounds)]
        ax_fp.scatter(xpix, ypix, s=4, c="black")
        ax_fp.invert_yaxis()
        ax_fp.set_aspect("equal")
        ax_fp.set_title(f"ROI footprint  (#{self._roi_idx})")
        ax_fp.set_xticks([]); ax_fp.set_yticks([])

        # Trace.
        if self._dff is not None and self._T > 0:
            trace = np.asarray(self._dff[:, self._roi_idx])
            time = np.arange(self._T) / max(self._fps, 1e-6)
            self._trace_ax.plot(time, trace, lw=0.8, label="ΔF/F")
            # ``sosfiltfilt`` is IIR — a single NaN poisons the rest
            # of the output, so skip the overlay (raw still renders)
            # when the trace isn't fully finite. The overlay is a
            # zero-phase preview so the lowpass line sits on top of
            # the same peaks visible in the raw trace; the on-disk
            # lowpass memmap that feeds event detection still uses the
            # causal filter to preserve real-time interpretation.
            if self._show_lowpass and np.isfinite(trace).all():
                y_lp, _ = lowpass_zero_phase_1d(
                    trace, self._fps, cutoff_hz=1.0, order=2)
                self._trace_ax.plot(time, y_lp, lw=1.2, color="C1",
                                    label="1 Hz lowpass")
                self._trace_ax.legend(loc="upper right", fontsize=8,
                                      frameon=False)
            self._trace_ax.set_xlabel("Time (s)")
        else:
            self._trace_ax.text(0.5, 0.5,
                                "No dF/F memmap on disk",
                                ha="center", va="center",
                                transform=self._trace_ax.transAxes)
            self._trace_ax.set_xticks([]); self._trace_ax.set_yticks([])
        self._trace_ax.set_title("ΔF/F")
        self._trace_ax.margins(x=0)

        self._fig.tight_layout()
        self._canvas.draw_idle()
        # The matplotlib toolbar (NavigationToolbar2Tk) sometimes
        # eats keyboard focus when its buttons get hover events
        # during redraws. ``focus_set`` on the Toplevel re-arms the
        # ``1`` / ``0`` / arrow-key bindings on the popout itself.
        # Scheduled via ``after(0, ...)`` so it lands AFTER the
        # redraw queue settles.
        try:
            self.after(0, self.focus_set)
        except Exception:
            pass

    def _refresh_status(self) -> None:
        if self._roi_idx is None:
            self._status_var.set("No ROI selected.")
            return
        is_cell = (int(self._iscell[self._roi_idx, 0]) == 1
                   if self._iscell is not None else None)
        cell_str = ("cell" if is_cell
                    else "not a cell" if is_cell is False
                    else "?")
        prob_str = (f"  p(cell)={self._probs[self._roi_idx]:.3f}"
                    if self._probs is not None else "")
        flipped = " [flipped this session]" if (
            self._roi_idx in self._flipped) else ""
        self._status_var.set(
            f"ROI #{self._roi_idx}    iscell={cell_str}{prob_str}"
            f"{flipped}    [1]=cell  [0]=not a cell  [Esc]=close")

    def _on_mpl_key(self, event) -> None:
        """matplotlib-side key router. Fires from
        ``FigureCanvasBase.mpl_connect("key_press_event", ...)``
        whenever the mouse is over the figure, regardless of which
        Tk child has keyboard focus.

        Mirrors the Toplevel-level Tk bindings (``1``, ``0``,
        ``Escape``) so the curator gets reliable shortcuts even
        when the matplotlib toolbar has consumed Tk focus.
        ``event.key`` is matplotlib's canonical key name -- digits
        come through as bare ``"1"`` / ``"0"``, escape as
        ``"escape"``.
        """
        key = getattr(event, "key", None)
        if key == "1":
            self._set_label(1)
        elif key == "0":
            self._set_label(0)
        elif key == "escape":
            self._on_close()

    # -- Actions -----------------------------------------------------------

    def _set_label(self, value: int) -> None:
        """Persist the curator's label for the current ROI.

        Always writes the label CSV + fires ``on_iscell_changed``,
        even when the value already matches ``iscell.npy``. The
        confirmation case ("model said cell, curator presses 1 to
        agree") is the most common in practice; an early-return on
        "no change" would silently swallow those presses, which
        would look like a broken button AND skip the auto-advance
        the parent tab wires through ``on_iscell_changed``.
        Explicit human labels also carry more weight for
        retraining than the model's seed prediction, so logging
        confirmations is what we want.
        """
        if self._roi_idx is None:
            return
        if self._iscell is None:
            # Retry: a reused popout may predate a fixed/re-written file.
            self._iscell = self._load_iscell()
        if self._iscell is None:
            messagebox.showwarning(
                "iscell.npy unavailable",
                f"{self._iscell_err} in {self._plane0}; cannot persist flip.")
            return
        prev = int(self._iscell[self._roi_idx, 0])
        # Update iscell.npy on disk regardless. When ``prev ==
        # value`` the write is a no-op semantically (same byte
        # pattern), but the np.save call is microseconds and
        # keeps the code path uniform.
        try:
            self._iscell[self._roi_idx, 0] = float(value)
            np.save(self._plane0 / "iscell.npy", self._iscell)
        except Exception as e:
            messagebox.showerror("iscell write failed", str(e))
            self._iscell[self._roi_idx, 0] = prev
            return
        # Always upsert the labels CSV. ``_upsert_label`` finds the
        # existing ``(plane0_path, roi)`` row and replaces it (or
        # appends if new), so confirmation presses just refresh the
        # timestamp -- exactly what we want for "human re-confirmed
        # this label at time T".
        try:
            _upsert_label(self._csv_path, self._plane0,
                          self._rec_id, self._roi_idx, value)
        except Exception as e:
            messagebox.showerror(
                "Curation CSV write failed",
                f"iscell.npy was updated but the curation CSV at\n"
                f"{self._csv_path}\ncould not be written:\n{e}")

        self._flipped.add(self._roi_idx)
        self._retrain_btn.configure(state="normal")
        self._promote_btn.configure(state="normal")
        if self._on_iscell_changed is not None:
            try:
                self._on_iscell_changed(self._roi_idx, value)
            except Exception:
                pass
        self._refresh_status()

    def _on_promote(self) -> None:
        """Overlay this session's manual flips onto
        ``predicted_cell_mask.npy`` so downstream stages (Tab 5/6/7,
        cross-correlation) honour the curator's labels without a
        full cell-filter retrain.

        Reads the current ``predicted_cell_mask.npy`` if it exists
        (CNN output), otherwise seeds from ``iscell.npy[:, 0]`` so
        the file is created even when the cell-filter has never run.
        For each ROI in ``self._flipped`` the mask cell is replaced
        with ``self._iscell[idx, 0]`` -- which already reflects every
        flip and flip-back the curator made this session. CNN
        predictions for un-touched ROIs are preserved.
        """
        if not self._flipped:
            return
        if self._iscell is None:
            # Retry: a reused popout may predate a fixed/re-written file.
            self._iscell = self._load_iscell()
        if self._iscell is None:
            messagebox.showwarning(
                "iscell.npy unavailable",
                f"{self._iscell_err} in {self._plane0}; cannot promote.")
            return
        n_flipped = len(self._flipped)
        from ...core.cellfilter import config as _cf_cfg
        mask_path = self._plane0 / _cf_cfg.PREDICTED_MASK_NAME
        plural = "s" if n_flipped != 1 else ""
        if not messagebox.askyesno(
                "Promote to filter mask",
                f"Promote {n_flipped} flipped ROI{plural} to "
                f"{_cf_cfg.PREDICTED_MASK_NAME}?\n\n"
                f"The filtered dF/F memmap will be rewritten to the new "
                f"cell set, so tabs that read it (e.g. lowpass n_kept) "
                f"refresh immediately. Already-computed results "
                f"(Tab 5/6/7, cross-correlation) must be re-run to "
                f"reflect the change. CNN predictions for the other ROIs "
                f"are preserved."):
            return
        try:
            if mask_path.exists():
                mask = np.load(mask_path).astype(bool)
                if mask.shape[0] != self._n_total:
                    messagebox.showerror(
                        "Shape mismatch",
                        f"{mask_path.name} has length "
                        f"{mask.shape[0]} but this recording has "
                        f"{self._n_total} ROIs. The popout is stale "
                        f"-- close it and reopen Tab 3.")
                    return
            else:
                # No CNN run yet -- seed from iscell, which already
                # carries the session's flips. The result is
                # "Suite2p classifier + manual flips".
                mask = (np.asarray(self._iscell[:, 0]) > 0)
            for idx in self._flipped:
                mask[int(idx)] = bool(
                    int(self._iscell[int(idx), 0]) == 1)
            np.save(mask_path, mask.astype(bool))
        except Exception as e:
            messagebox.showerror(
                "Promote failed",
                f"Could not write {mask_path}:\n{e}")
            return
        # Mask on disk now reflects every flip in this session; clear
        # the pending set so external guards (Tab 3 export button) can
        # tell a clean state from a dirty one.
        self._flipped.clear()
        self._train_status_var.set(
            f"Promoted {n_flipped} label{plural} to "
            f"{_cf_cfg.PREDICTED_MASK_NAME}")
        # Sentinel (-1, -1) tells the parent tab to reload the keep
        # mask broadly rather than refresh a single ROI's panels.
        if self._on_iscell_changed is not None:
            try:
                self._on_iscell_changed(-1, -1)
            except Exception:
                pass

    def has_pending_promotion(self) -> bool:
        """True when ROI labels have been flipped in this session but
        not yet promoted to ``predicted_cell_mask.npy``. Used by the
        Tab 3 figure-export button to refuse exporting figures that
        wouldn't match what the curator is looking at.
        """
        return bool(self._flipped)

    def _size_outlier_z(self) -> float:
        """Active size gate for the post-Retrain re-predict."""
        from ...core.cellfilter import config as _cfg
        if self._get_size_outlier_z is not None:
            try:
                return float(self._get_size_outlier_z())
            except Exception:
                pass
        return float(_cfg.SIZE_OUTLIER_Z)

    def _warm_start_ckpt(self) -> Optional[Path]:
        """Parent-selected checkpoint to warm-start Retrain from, or
        ``None`` (train from random init) when the parent gives none
        or the file is missing."""
        if self._get_ckpt_path is None:
            return None
        try:
            raw = str(self._get_ckpt_path() or "").strip()
        except Exception:
            return None
        if not raw:
            return None
        path = Path(raw)
        return path if path.is_file() else None

    def _on_retrain(self) -> None:
        if self._train_thread is not None and self._train_thread.is_alive():
            return
        from ...core.cellfilter import config as _cfg
        # Snapshot Tk-side state on the main thread; the worker must
        # not touch Tk variables.
        warm_start = self._warm_start_ckpt()
        size_z = self._size_outlier_z()
        if warm_start is not None:
            init_line = (f"Fine-tunes the selected checkpoint\n"
                         f"{warm_start}\n(lr {_cfg.WARM_START_LR:g}).")
        else:
            init_line = (f"Trains from random init (lr {_cfg.LR:g}); no "
                         f"selected checkpoint to warm-start from.")
        # Tab 3 (the only parent that supplies the checkpoint hook)
        # also rewrites the filtered dF/F memmap on the (-1, -1)
        # sentinel and selects the new best.pt.
        parent_line = (
            " Tab 3 then rewrites the filtered dF/F memmap to the new "
            "cell set and selects the new best.pt as its checkpoint."
            if self._on_checkpoint_trained is not None else "")
        if not messagebox.askyesno(
                "Retrain cell filter",
                f"Retrain the cell filter now?\n\n"
                f"Reads {self._csv_path}\n"
                f"and writes best.pt / last.pt to\n"
                f"{_cfg.CHECKPOINT_DIR}.\n{init_line}\n\n"
                f"Training runs in the background and may take a while. "
                f"Flip buttons stay enabled, but the trainer reads the "
                f"labels CSV when it starts, so flips made during "
                f"training go into the next round.\n\n"
                f"When it finishes, this recording is re-predicted "
                f"(size_outlier_z={size_z:g}) and "
                f"{_cfg.PREDICTED_MASK_NAME} is overwritten."
                f"{parent_line}"):
            return
        self._retrain_btn.configure(state="disabled")
        self._train_status_var.set("Training... see console for epoch log")
        self._train_thread = threading.Thread(
            target=self._train_worker, args=(warm_start, size_z),
            daemon=True)
        self._train_thread.start()
        # Poll the worker so the status label flips back when done.
        self._poll_train_state()

    def _train_worker(self, warm_start: Optional[Path] = None,
                      size_outlier_z: Optional[float] = None) -> None:
        """Run training, then immediately re-score the current
        recording's ROIs with the fresh checkpoint so Tab 3's panels
        + the popout's status bar reflect the new predictions
        without a manual re-detect.

        Sequence (all in this worker thread; main thread only
        consumes ``self._train_result`` / ``self._trained_ckpt``):
          1. ``cellfilter.train.main()`` -- reads the curation CSV,
             writes new ``best.pt`` + ``last.pt``. Warm-starts from
             ``warm_start`` at ``config.WARM_START_LR`` when given,
             else random init at ``config.LR``.
          2. Load the fresh ``best.pt`` and run
             ``predict_recording`` against ``self._plane0`` with
             ``size_outlier_z`` (the parent's live setting, snapshot
             when Retrain was clicked) so
             ``predicted_cell_prob.npy`` / ``predicted_cell_mask.npy``
             reflect the new model.
          3. Stash ``("ok", ...)`` for the poller. The poller flips
             the status label, reloads ``self._probs``, and fires
             ``on_iscell_changed(-1, -1)``.

        Errors in step 1 surface as ``("err", msg)``; errors in step
        2 surface as ``("warn", msg)`` -- the new checkpoint is
        still on disk and usable, we just couldn't auto-refresh. A
        run whose validation AUROC was never defined writes no
        ``best.pt``; that is a ``("warn", msg)`` with no re-predict.
        """
        from ...core.cellfilter import config as _cfg
        self._trained_ckpt = None
        if size_outlier_z is None:
            size_outlier_z = float(_cfg.SIZE_OUTLIER_Z)
        try:
            from ...core.cellfilter import train as _train
            if warm_start is not None:
                best_auc = _train.main(warm_start=warm_start,
                                       lr=_cfg.WARM_START_LR)
            else:
                best_auc = _train.main()
        except Exception as e:
            self._train_result = ("err", f"Training failed: {e}")
            return

        ckpt_path = _cfg.CHECKPOINT_DIR / "best.pt"
        # ``train.main`` returns -1.0 when no epoch had a defined
        # validation AUROC, i.e. it never wrote best.pt this run. Any
        # best.pt on disk is then from an earlier run -- don't
        # re-predict with it or hand it to the parent as "new".
        if best_auc is None or not (best_auc >= 0):
            self._train_result = (
                "warn",
                "Training finished but validation AUROC was never "
                "defined (the validation split lacks one class), so no "
                "new best.pt was written. Label more ROIs of both "
                "classes and retrain.")
            return
        self._trained_ckpt = ckpt_path

        # Step 2: auto re-predict against this plane0.
        try:
            from ...core.cellfilter.predict import (
                load_model_from_checkpoint,
                predict_recording,
            )
            import torch
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            model = load_model_from_checkpoint(ckpt_path, device)
            predict_recording(self._rec_id, model, device,
                              plane0=self._plane0,
                              size_outlier_z=size_outlier_z)
            self._train_result = (
                "ok",
                "Training complete. predicted_cell_mask.npy refreshed.")
        except Exception as e:
            self._train_result = (
                "warn",
                f"Training complete but re-predict failed: {e}")

    def _poll_train_state(self) -> None:
        if self._train_thread is None:
            return
        if self._train_thread.is_alive():
            self._train_state_after_id = self.after(
                500, self._poll_train_state)
            return
        result = getattr(self, "_train_result", ("err", "no result"))
        kind, msg = result
        # A new best.pt was written (even if the re-predict then
        # failed): let the parent select it so later detection runs
        # use the retrained model instead of the bundled one.
        trained = getattr(self, "_trained_ckpt", None)
        if trained is not None and self._on_checkpoint_trained is not None:
            try:
                self._on_checkpoint_trained(Path(trained))
                msg = f"{msg} Selected checkpoint: {trained}"
            except Exception:
                pass
        self._train_status_var.set(msg)
        # Leave the button enabled so the user can iterate (more flips
        # -> more training rounds).
        self._retrain_btn.configure(state="normal")
        if kind == "err":
            messagebox.showerror("Cell filter training", msg)
        elif kind == "warn":
            messagebox.showwarning("Cell filter training", msg)
        else:
            # ``ok``: the re-predict pass also succeeded, so the new
            # ``predicted_cell_prob.npy`` / ``predicted_cell_mask.npy``
            # are on disk. Reload the popout's own probability cache
            # so the status bar shows fresh p(cell) values, then fire
            # the same ``(-1, -1)`` sentinel Promote uses. Tab 3
            # treats it as "predicted_cell_mask.npy changed": it
            # reloads the keep mask, repaints, AND rewrites the
            # filtered dF/F memmap (releasing downstream handles
            # first) via ``_regenerate_after_promote``. The Retrain
            # confirmation dialog says so up front.
            self._probs = self._load_probs()
            self._refresh_status()
            if self._on_iscell_changed is not None:
                try:
                    self._on_iscell_changed(-1, -1)
                except Exception:
                    pass
        self._train_thread = None

    # -- Lowpass overlay ----------------------------------------------------

    def _on_toggle_lowpass(self) -> None:
        self._show_lowpass = not self._show_lowpass
        self._lowpass_btn.configure(
            text="Hide lowpass" if self._show_lowpass
            else "Show 1 Hz lowpass")
        self._render_panels()

    # -- Close --------------------------------------------------------------

    def _on_close(self) -> None:
        if self._train_state_after_id is not None:
            try:
                self.after_cancel(self._train_state_after_id)
            except Exception:
                pass
            self._train_state_after_id = None
        try:
            plt.close(self._fig)
        except Exception:
            pass
        self.destroy()

    # -- Helpers -----------------------------------------------------------

    @staticmethod
    def _derive_recording_id(plane0: Path) -> str:
        """Human-readable recording id for the labels CSV's
        ``recording_ID`` column and the re-predict log line.

        Audit-only: the trainer keys rows on ``plane0_path``, so this
        value never affects which recording a label resolves to. Uses
        the same ``utils.infer_recording_id`` as every other tab;
        falls back to the plane0's parent folder name if that raises.
        """
        from ...core import utils as _u
        try:
            return str(_u.infer_recording_id(plane0))
        except Exception:
            return Path(plane0).parent.name

    def _load_iscell(self) -> Optional[np.ndarray]:
        path = self._plane0 / "iscell.npy"
        if not path.exists():
            self._iscell_err = "No iscell.npy"
            return None
        try:
            arr = np.load(path)
            if arr.ndim == 1:
                # Some pipelines write a 1-D array; promote to (N,2)
                # so the column-0 / column-1 access pattern stays the
                # same.
                arr = np.stack(
                    [arr.astype(float), np.zeros_like(arr, dtype=float)],
                    axis=1)
            if arr.shape[0] != self._n_total:
                # Stale / partially-written file (e.g. re-detection changed
                # the ROI count). Treat as absent rather than indexing past
                # the end in _refresh_status (self._iscell[self._roi_idx]).
                self._iscell_err = (
                    f"iscell.npy has {arr.shape[0]} ROIs but stat.npy has "
                    f"{self._n_total} (stale file from another detection run?)")
                return None
            return arr.astype(float)
        except Exception as e:
            self._iscell_err = f"iscell.npy could not be read ({e})"
            return None

    def _load_probs(self) -> Optional[np.ndarray]:
        path = self._plane0 / "predicted_cell_prob.npy"
        if not path.exists():
            return None
        try:
            arr = np.load(path).astype(float)
            if arr.shape[0] != self._n_total:
                # Stale / partially-written prob file -> treat as absent so
                # _refresh_status's self._probs[self._roi_idx] can't go OOB.
                return None
            return arr
        except Exception:
            return None
