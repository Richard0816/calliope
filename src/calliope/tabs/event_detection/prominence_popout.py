"""Prominence-distribution popout for Tab 5 (Event Detection).

What it shows
-------------
A histogram of the prominences of every candidate peak in the
smoothed onset-density curve, with a vertical red line at the
user's current ``min_prominence`` threshold. Drag the slider beneath
the plot to slide the threshold along the X axis and see live how
many candidate peaks would pass.

Why this exists
---------------
The smoothed-density peak distribution is typically bimodal:
small noise ripples cluster near zero, and real population events
sit at a clearly higher prominence. Picking ``min_prominence`` by
hand from the Advanced dialog is guesswork — this popout shows the
valley between the two modes so the user can place the cut there.

Null-floor overlay
------------------
The popout also draws the circular-shift NULL prominence floor (its
p95 and p99) as vertical reference lines, when available. The null is
built by ``utils.circular_shift_null_prominences`` — independently
time-shifting each ROI's onset train so any cross-cell coincidence is
pure chance — and answers "how tall does a density bump get from
random coincidence alone?". A real population event must clear that
floor. By default (``auto_min_prominence``) the null p99 is applied
automatically as the per-recording prominence floor, and this popout's
live threshold line snaps onto that floor once it is computed — so what
you see matches what detection applies. The user can still drag or type
a manual ``min_prominence`` instead; that is an explicit override, and
applying it turns auto off. The null is computed off-thread by Tab 5 and
pushed in via :meth:`ProminencePopout.set_null_percentiles`.

Wiring
------
* Tab 5 holds a single-instance reference (``self._prominence_popout``);
  re-clicking the "Prominence distribution..." button focuses the
  existing window rather than spawning a duplicate.
* On Apply the popout calls back into Tab 5 with the chosen value;
  Tab 5 updates ``self._params["min_prominence"]`` and re-renders.
* The slider only changes the *visual* threshold line + count in the
  popout — no detection runs until Apply.
"""

from __future__ import annotations

import tkinter as tk
from typing import Callable, Optional

import customtkinter as ctk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ...gui_common import install_scroll_router


def _slider_range(prominences: np.ndarray,
                  current_value: float) -> tuple[float, float]:
    """Pick a sensible slider [lo, hi] enclosing the distribution and
    the current threshold."""
    if prominences.size == 0:
        return (0.0, max(current_value * 2.0, 1.0))
    lo = float(min(prominences.min(), current_value, 0.0))
    hi = float(max(prominences.max(), current_value)) * 1.05
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


class ProminencePopout(ctk.CTkToplevel):
    """Histogram + slider popout for tuning ``min_prominence``."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        prominences: np.ndarray,
        current_value: float,
        on_apply: Callable[[float], None],
        null_percentiles: Optional[dict] = None,
        null_pending: bool = False,
        auto_on: bool = False,
        auto_percentile: float = 99.0,
        title: str = "Prominence distribution",
    ) -> None:
        super().__init__(parent)
        self.title(title)
        self.geometry("760x520")
        # ``wm_minsize`` is the raw Tk method; ``CTkToplevel.minsize``
        # short-circuits on a not-yet-realised window.
        self.wm_minsize(560, 380)

        self._prominences = np.asarray(prominences, dtype=np.float64).ravel()
        self._initial_value = float(current_value)
        self._on_apply = on_apply
        # Circular-shift null floor: {percentile: value}, or None until the
        # off-thread computation in Tab 5 finishes. ``null_pending`` toggles
        # a "computing..." annotation so the user knows it's on the way.
        self._null_percentiles = dict(null_percentiles or {})
        self._null_pending = bool(null_pending)
        self._null_lines: list = []
        self._null_note = None
        # When ``auto_min_prominence`` is on, detection applies the null
        # p{auto_pct} floor (not the static ``min_prominence``). The popout
        # snaps its live threshold onto that floor once it arrives so what's
        # shown matches what detection uses. Dragging the slider / typing a
        # value is an explicit manual override (Apply then turns auto off).
        self._auto_on = bool(auto_on)
        self._auto_pct = float(auto_percentile)
        self._user_overriding = False

        slider_lo, slider_hi = _slider_range(self._prominences, current_value)
        self._slider_lo = slider_lo
        self._slider_hi = slider_hi

        self.value_var = tk.DoubleVar(value=float(current_value))
        self.count_var = tk.StringVar(value="")

        self._build_ui()
        self._build_figure()
        self._update_threshold_line(current_value)
        self._refresh_count_label()

        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.bind("<Escape>", lambda _e: self._on_cancel())
        # Consume wheel events so they don't leak through CTk's
        # global bind_all and scroll the main app behind this popout.
        install_scroll_router(self)

    # ---- UI scaffolding ----------------------------------------------------

    def _build_ui(self) -> None:
        outer = ctk.CTkFrame(self, fg_color="transparent")
        outer.pack(fill="both", expand=True, padx=8, pady=8)

        # Top: histogram canvas.
        self._canvas_frame = ctk.CTkFrame(outer)
        self._canvas_frame.pack(fill="both", expand=True)
        ctk.CTkLabel(self._canvas_frame, text="Candidate peak prominences",
                     font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(6, 0))

        # Middle: slider + numeric readout.
        slider_row = ctk.CTkFrame(outer, fg_color="transparent")
        slider_row.pack(fill="x", pady=(8, 0))
        ctk.CTkLabel(slider_row, text="min_prominence:").pack(side="left")

        self._scale = ctk.CTkSlider(
            slider_row,
            from_=self._slider_lo,
            to=self._slider_hi,
            orientation="horizontal",
            variable=self.value_var,
            command=self._on_slider_move,
        )
        self._scale.pack(side="left", fill="x", expand=True, padx=(8, 8))

        # Numeric entry so the user can type an exact value.
        self._entry = ctk.CTkEntry(slider_row, width=100)
        self._entry.insert(0, f"{float(self.value_var.get()):.4f}")
        self._entry.bind("<Return>", self._on_entry_commit)
        self._entry.bind("<FocusOut>", self._on_entry_commit)
        self._entry.pack(side="left")

        # Bottom: live "keeping N / M" + buttons.
        bot = ctk.CTkFrame(outer, fg_color="transparent")
        bot.pack(fill="x", pady=(6, 0))
        ctk.CTkLabel(bot, textvariable=self.count_var,
                     font=ctk.CTkFont(size=11, slant="italic")).pack(
            side="left")

        btns = ctk.CTkFrame(bot, fg_color="transparent")
        btns.pack(side="right")
        ctk.CTkButton(btns, text="Cancel", width=90,
                      command=self._on_cancel).pack(side="right", padx=(6, 0))
        ctk.CTkButton(btns, text="Reset", width=90,
                      command=self._on_reset).pack(side="right", padx=(6, 0))
        ctk.CTkButton(btns, text="Apply", width=90,
                      command=self._on_apply_click).pack(side="right")

    def _build_figure(self) -> None:
        self.fig = plt.Figure(figsize=(7.2, 3.4), tight_layout=True)
        self.ax = self.fig.add_subplot(111)

        if self._prominences.size == 0:
            self.ax.text(
                0.5, 0.5,
                "No candidate peaks in the smoothed density.\n"
                "Render Tab 5 first.",
                ha="center", va="center", transform=self.ax.transAxes,
            )
            self._threshold_line = None
        else:
            # Adaptive bin count: scales with sqrt(N) but capped so very
            # large N doesn't shred the histogram into single-count bars.
            n_bins = int(np.clip(
                np.ceil(np.sqrt(self._prominences.size)) * 1.5, 12, 60))
            self.ax.hist(self._prominences, bins=n_bins,
                         color="#4682B4", edgecolor="white", linewidth=0.4)
            self.ax.set_xlabel("Peak prominence")
            self.ax.set_ylabel("# candidate peaks")
            self._threshold_line = self.ax.axvline(
                float(self.value_var.get()),
                color="crimson", linewidth=1.6, linestyle="--",
                label="min_prominence",
            )

        self.canvas = FigureCanvasTkAgg(self.fig, master=self._canvas_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True,
                                         padx=4, pady=(0, 4))
        # Null-floor reference lines (+ legend) on top of the histogram.
        self._draw_null_lines()

    # ---- Event handlers ----------------------------------------------------

    def _on_slider_move(self, _val: str) -> None:
        # CTkSlider's command passes the new value as a string; we
        # already have the live float via the linked ``value_var``.
        self._mark_user_override()
        v = float(self.value_var.get())
        # Sync the entry box without retriggering its commit handler.
        self._entry.delete(0, tk.END)
        self._entry.insert(0, f"{v:.4f}")
        self._update_threshold_line(v)
        self._refresh_count_label()

    def _on_entry_commit(self, _evt=None) -> None:
        raw = self._entry.get().strip()
        try:
            v = float(raw)
        except ValueError:
            # Restore prior value silently if the user typed garbage.
            self._entry.delete(0, tk.END)
            self._entry.insert(0, f"{float(self.value_var.get()):.4f}")
            return
        self._mark_user_override()
        v = max(self._slider_lo, min(self._slider_hi, v))
        self.value_var.set(v)
        self._update_threshold_line(v)
        self._refresh_count_label()

    def _on_reset(self) -> None:
        # Reset clears any manual override and returns to the baseline (the
        # auto floor when auto is on, else the value the popout opened with).
        if self._auto_on:
            self._user_overriding = False
            if self._threshold_line is not None:
                self._threshold_line.set_label(
                    f"applied (auto · null p{int(self._auto_pct)})")
        self.value_var.set(self._initial_value)
        self._entry.delete(0, tk.END)
        self._entry.insert(0, f"{self._initial_value:.4f}")
        self._update_threshold_line(self._initial_value)
        self._draw_null_lines()
        self._refresh_count_label()

    def _on_apply_click(self) -> None:
        v = float(self.value_var.get())
        try:
            self._on_apply(v)
        finally:
            self.destroy()

    def _on_cancel(self) -> None:
        self.destroy()

    # ---- Drawing helpers ---------------------------------------------------

    def _update_threshold_line(self, v: float) -> None:
        if self._threshold_line is None:
            return
        self._threshold_line.set_xdata([v, v])
        self.canvas.draw_idle()

    def _apply_auto_floor(self, value: float) -> None:
        """Snap the live threshold onto the auto (null-percentile) floor so
        the popout reflects the value detection actually applies when
        ``auto_min_prominence`` is on. Also makes this the Reset target."""
        value = float(value)
        # Grow the slider range if the floor sits outside the initial span.
        if value > self._slider_hi:
            self._slider_hi = value * 1.05
            self._scale.configure(to=self._slider_hi)
        if value < self._slider_lo:
            self._slider_lo = min(0.0, value)
            self._scale.configure(from_=self._slider_lo)
        self._initial_value = value  # Reset returns to the auto floor
        self.value_var.set(value)
        self._entry.delete(0, tk.END)
        self._entry.insert(0, f"{value:.4f}")
        self._update_threshold_line(value)
        if self._threshold_line is not None:
            self._threshold_line.set_label(
                f"applied (auto · null p{int(self._auto_pct)})")

    def _mark_user_override(self) -> None:
        """First manual slider/entry edit while auto is on: this value will
        replace the auto floor on Apply, so relabel the threshold line and
        restore the null p{auto_pct} reference line."""
        if self._auto_on and not self._user_overriding:
            self._user_overriding = True
            if self._threshold_line is not None:
                self._threshold_line.set_label("min_prominence (manual)")
            self._draw_null_lines()

    # ---- Null-floor overlay ------------------------------------------------

    # p95 lighter, p99 darker; both clearly distinct from the crimson
    # threshold line.
    _NULL_STYLE = {
        95.0: dict(color="#7f8c8d", linestyle=":", linewidth=1.4),
        99.0: dict(color="#34495e", linestyle="-.", linewidth=1.4),
    }

    def set_null_percentiles(self, null_percentiles: Optional[dict]) -> None:
        """Push the circular-shift null floor in after off-thread compute.

        ``null_percentiles`` maps percentile -> prominence (e.g.
        ``{95.0: 0.0022, 99.0: 0.0028}``). Passing ``None`` / empty marks
        the null as unavailable (e.g. the null produced no peaks). Safe to
        call from the Tk main thread after the window is built.
        """
        self._null_pending = False
        self._null_percentiles = dict(null_percentiles or {})
        if self.winfo_exists():
            # If auto is driving the floor (and the user hasn't overridden),
            # snap the live threshold onto the per-recording null p{auto_pct}
            # -- the exact value detect_event_windows applies.
            if self._auto_on and not self._user_overriding:
                _v = self._null_percentiles.get(self._auto_pct)
                if _v is None:  # configured pct absent -> fall back to p99
                    _v = self._null_percentiles.get(99.0)
                if _v is not None and np.isfinite(_v) and _v > 0:
                    self._apply_auto_floor(float(_v))
            self._draw_null_lines()
            self._refresh_count_label()

    def _draw_null_lines(self) -> None:
        """(Re)draw the null p95/p99 vertical lines, note, and legend."""
        if getattr(self, "_threshold_line", None) is None:
            # Empty-distribution placeholder figure -- nothing to overlay.
            return
        # Clear any prior null artists so repeated calls don't stack lines.
        for ln in self._null_lines:
            try:
                ln.remove()
            except (ValueError, AttributeError):
                pass
        self._null_lines = []
        if self._null_note is not None:
            try:
                self._null_note.remove()
            except (ValueError, AttributeError):
                pass
            self._null_note = None

        finite = {p: v for p, v in self._null_percentiles.items()
                  if v is not None and np.isfinite(v)}
        if finite:
            for p in sorted(finite):
                # When auto is driving the floor, the crimson "applied" line
                # already marks p{auto_pct}; don't double-draw it.
                if (self._auto_on and not self._user_overriding
                        and float(p) == self._auto_pct):
                    continue
                style = self._NULL_STYLE.get(
                    float(p),
                    dict(color="#7f8c8d", linestyle=":", linewidth=1.4))
                self._null_lines.append(self.ax.axvline(
                    float(finite[p]),
                    label=f"null p{int(p)} (floor)", **style))
        elif self._null_pending:
            self._null_note = self.ax.text(
                0.99, 0.80, "null floor: computing…",
                ha="right", va="top", transform=self.ax.transAxes,
                fontsize=8, color="gray", style="italic")
        elif self._null_percentiles == {} and not self._null_pending:
            # Computation finished but yielded nothing usable.
            pass

        # Rebuild the legend so it picks up whatever lines now exist.
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(handles, labels, loc="upper right", fontsize=9)
        if getattr(self, "canvas", None) is not None:
            self.canvas.draw_idle()

    def _refresh_count_label(self) -> None:
        N = int(self._prominences.size)
        v = float(self.value_var.get())
        if N == 0:
            self.count_var.set("No candidate peaks available.")
            return
        kept = int(np.sum(self._prominences >= v))
        auto_floor = None
        if self._auto_on and not self._user_overriding:
            _f = self._null_percentiles.get(
                self._auto_pct, self._null_percentiles.get(99.0))
            if _f is not None and np.isfinite(_f) and _f > 0:
                auto_floor = float(_f)
        # Auto is on but the per-recording floor is still being computed:
        # say so rather than implying the static fallback value is the floor.
        if (self._auto_on and not self._user_overriding
                and auto_floor is None and self._null_pending):
            self.count_var.set(
                f"AUTO floor (null p{int(self._auto_pct)}): computing… "
                f"({N} candidate peaks)")
            return
        label = (f"AUTO floor (null p{int(self._auto_pct)})"
                 if auto_floor is not None else "min_prominence")
        msg = f"keeping {kept} / {N} candidate peaks at {label} = {v:.4f}"
        # Ratio-to-p99 is only informative for a manual value (when auto is
        # applied the live value *is* the p99 floor, i.e. 1.00x).
        p99 = self._null_percentiles.get(99.0)
        if (auto_floor is None and p99 is not None
                and np.isfinite(p99) and p99 > 0):
            msg += f"  ({v / p99:.2f}× null p99 floor)"
        self.count_var.set(msg)
