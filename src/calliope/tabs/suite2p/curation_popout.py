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
labels CSV.

Side effects of a flip
----------------------
1. Mutates ``<plane0>/iscell.npy[roi, 0]`` to the new value.
2. Appends a row to the curation CSV
   (``cellfilter.config.LABELS_CSV`` by default,
   ``F:\\roi_curation.csv``) with columns
   ``recording_ID, ROI_number, user_defined_cell``. This is the same
   CSV ``cellfilter.train.main()`` reads, so retraining picks up the
   new labels immediately.
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
from typing import Optional

import customtkinter as ctk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk,
)
from matplotlib.patches import Rectangle

from ...gui_common import install_scroll_router


def _padded_max_image(ops: dict, mean_img: np.ndarray) -> np.ndarray:
    """Return ops['max_proj'] padded back into a full-frame image.

    Suite2p's ``ops['max_proj']`` is cropped to ``ops['yrange']`` x
    ``ops['xrange']``; pad it back so ROI coords (full-frame) align.
    Falls back to mean image when neither max_proj nor maxImg is set.
    """
    max_img = ops.get("max_proj", None)
    if max_img is None:
        max_img = ops.get("maxImg", None)
    if max_img is None:
        return mean_img
    max_img = np.asarray(max_img)
    if max_img.shape == mean_img.shape:
        return max_img
    full = np.zeros_like(mean_img, dtype=max_img.dtype)
    yr = ops.get("yrange", (0, max_img.shape[0]))
    xr = ops.get("xrange", (0, max_img.shape[1]))
    y0, y1 = int(yr[0]), int(yr[1])
    x0, x1 = int(xr[0]), int(xr[1])
    h = min(y1 - y0, max_img.shape[0])
    w = min(x1 - x0, max_img.shape[1])
    full[y0:y0 + h, x0:x0 + w] = max_img[:h, :w]
    return full


def _open_dff(plane0: Path, n_total: int) -> tuple[Optional[np.memmap], int]:
    """Open the unfiltered all-ROI dF/F memmap for trace display.

    Returns ``(dff, T)`` or ``(None, 0)`` when no compatible memmap
    is on disk. We try the standard ``r0p7_dff.memmap.float32`` first
    (one column per Suite2p ROI), then fall back to the F.npy-derived
    trace if that's all we have.
    """
    candidates = [
        plane0 / "r0p7_dff.memmap.float32",
        plane0 / "dff.memmap.float32",
    ]
    for path in candidates:
        if not path.exists():
            continue
        n_bytes = path.stat().st_size
        floats = n_bytes // 4
        if floats % n_total != 0:
            continue
        T = floats // n_total
        try:
            return (np.memmap(path, dtype="float32", mode="r",
                              shape=(T, n_total)),
                    T)
        except OSError:
            continue
    return None, 0


def _fps_for(plane0: Path, fallback: float = 15.07) -> float:
    try:
        from ...core import utils
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
                 *, on_iscell_changed=None) -> None:
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
        # Keep ``_max_img`` as the legacy fall-back for the
        # ROI-overlay panel; it's the recording's max projection
        # padded to full FOV (or the mean if max isn't on disk).
        self._max_img = _padded_max_image(ops, self._mean_img)
        self._n_total = len(stat)
        self._dff, self._T = _open_dff(self._plane0, self._n_total)
        self._fps = _fps_for(self._plane0)
        self._rec_id = self._derive_recording_id(self._plane0)
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

        self._train_status_var = tk.StringVar(value="")
        ctk.CTkLabel(btn_frame, textvariable=self._train_status_var
                     ).pack(side="left", padx=(10, 0))

        ctk.CTkButton(btn_frame, text="Close (Esc)", width=110,
                      command=self._on_close).pack(side="right")

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
            vmin, vmax = np.percentile(img, [1, 99])
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
            self._trace_ax.plot(time, trace, lw=0.8)
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
            messagebox.showwarning(
                "iscell.npy missing",
                f"No iscell.npy in {self._plane0}; cannot persist flip.")
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
        if self._on_iscell_changed is not None:
            try:
                self._on_iscell_changed(self._roi_idx, value)
            except Exception:
                pass
        self._refresh_status()

    def _on_retrain(self) -> None:
        if self._train_thread is not None and self._train_thread.is_alive():
            return
        if not messagebox.askyesno(
                "Retrain cell filter",
                f"Run cellfilter.train.main() now?\n\n"
                f"This reads {self._csv_path}\n"
                f"and writes new checkpoints to "
                f"cellfilter.config.CHECKPOINT_DIR.\n\n"
                f"Training is synchronous and may take a while; the "
                f"GUI stays responsive but the popout will block "
                f"further flips until training finishes."):
            return
        self._retrain_btn.configure(state="disabled")
        self._train_status_var.set("Training... see console for epoch log")
        self._train_thread = threading.Thread(
            target=self._train_worker, daemon=True)
        self._train_thread.start()
        # Poll the worker so the status label flips back when done.
        self._poll_train_state()

    def _train_worker(self) -> None:
        """Run training, then immediately re-score the current
        recording's ROIs with the fresh checkpoint so Tab 3's panels
        + the popout's status bar reflect the new predictions
        without a manual re-detect.

        Sequence (all in this worker thread; main thread only
        consumes ``self._train_result``):
          1. ``cellfilter.train.main()`` -- reads the curation CSV,
             writes new ``best.pt`` + ``last.pt``.
          2. Load the fresh ``best.pt`` and run
             ``predict_recording`` against ``self._plane0`` so
             ``predicted_cell_prob.npy`` / ``predicted_cell_mask.npy``
             reflect the new model.
          3. Stash ``("ok", ...)`` for the poller. The poller flips
             the status label, reloads ``self._probs``, and fires
             ``on_iscell_changed`` so the parent Tab 3 repaints
             panels 2 + 3 against the new mask.

        Errors in step 1 surface as ``("err", msg)``; errors in step
        2 surface as ``("warn", msg)`` -- the new checkpoint is
        still on disk and usable, we just couldn't auto-refresh.
        """
        try:
            from ...core.cellfilter import train as _train
            _train.main()
        except Exception as e:
            self._train_result = ("err", f"Training failed: {e}")
            return

        # Step 2: auto re-predict against this plane0.
        try:
            from ...core.cellfilter import config as _cfg
            from ...core.cellfilter.predict import (
                load_model_from_checkpoint,
                predict_recording,
            )
            from ...core import utils as _u
            import torch
            ckpt_path = _cfg.CHECKPOINT_DIR / "best.pt"
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            model = load_model_from_checkpoint(ckpt_path, device)
            try:
                rec_id = _u.infer_recording_id(self._plane0)
            except Exception:
                rec_id = self._rec_id
            predict_recording(rec_id, model, device, plane0=self._plane0)
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
            # so the status bar shows fresh p(cell) values, then ping
            # the parent Tab 3 so it reloads the keep mask and
            # repaints panels 2 + 3. Using the existing
            # ``on_iscell_changed`` channel with a sentinel ROI index
            # of -1 keeps the parent-side handler trivial: it already
            # ignores the index and just reloads the keep mask.
            self._probs = self._load_probs()
            self._refresh_status()
            if self._on_iscell_changed is not None:
                try:
                    self._on_iscell_changed(-1, -1)
                except Exception:
                    pass
        self._train_thread = None

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
        """Return the recording id matching ``cellfilter`` conventions.

        ``cellfilter.config`` resolves a recording by name relative
        to ``DATA_ROOT`` / ``EXTRA_DATA_ROOTS``. The name is the
        directory two levels above plane0
        (``<recording>/suite2p/plane0`` for the standalone runs, or
        ``<recording>/detection/final/suite2p/plane0`` for the
        calliope pipeline). We pick the deepest non-suite2p ancestor
        as the recording id so the CSV row matches the trainer's
        lookup.
        """
        ancestors = [a for a in plane0.parents
                     if a.name not in ("plane0", "suite2p", "final",
                                       "detection")]
        return ancestors[0].name if ancestors else plane0.parent.name

    def _load_iscell(self) -> Optional[np.ndarray]:
        path = self._plane0 / "iscell.npy"
        if not path.exists():
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
            return arr.astype(float)
        except Exception:
            return None

    def _load_probs(self) -> Optional[np.ndarray]:
        path = self._plane0 / "predicted_cell_prob.npy"
        if not path.exists():
            return None
        try:
            return np.load(path).astype(float)
        except Exception:
            return None
