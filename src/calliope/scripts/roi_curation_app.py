"""Standalone ROI curation app for the cell-filter retraining loop.

Usage
-----
    python -m calliope.scripts.roi_curation_app [plane0 ...] [options]

Where each positional ``plane0`` is a suite2p plane0 directory
(``<recording>/detection/final/suite2p/plane0``). Zero or more
plane0s can be pre-loaded on the command line; the app's top-bar
**Add plane0** / **Remove** buttons let the curator extend or trim
the list at runtime so a single curation session can span many
recordings.

Options
-------
    --ckpt PATH         Override the cell-filter checkpoint.
                        Defaults to ``CHECKPOINT_DIR/best.pt``.
    --sort MODE         ROI navigation order:
                          ``unsure`` (default; ``|p(cell) - 0.5|``
                                      ascending so the most-ambiguous
                                      ROIs come first -- the most
                                      informative for retraining)
                          ``prob_asc``  (lowest p(cell) first --
                                          good for spotting false
                                          positives the model
                                          missed)
                          ``prob_desc`` (highest p(cell) first --
                                          good for spotting false
                                          negatives in the
                                          ``not a cell`` set)
                          ``index``     (raw Suite2p ROI order)
    --repredict         Force a re-predict pass even if
                        ``predicted_cell_prob.npy`` already exists
                        in plane0.

What it does
------------
1. Top window: a Listbox showing every plane0 in the current
   session, **Add plane0** (file picker), **Remove**, sort-order
   dropdown, **Re-predict selected**.
2. Selecting (or double-clicking) a plane0 entry:
     a. If ``predicted_cell_prob.npy`` is missing in that plane0
        (or **Re-predict** was clicked), loads the checkpoint and
        runs ``predict_recording`` so the curator has p(cell)
        scores to sort by.
     b. (Re)creates the standard ``CurationPopout`` widget bound to
        that plane0. The popout opens with the ROI sorted-order
        position 0 (most ambiguous, lowest p(cell), etc., depending
        on --sort).
3. Inside the popout:
     Left / Right    Prev / Next ROI (according to --sort)
     1 / 0           Mark current ROI as cell / not a cell
     Esc / X         Close the popout (the main list window stays)
4. Every cell / not-cell flip writes to:
     - ``<plane0>/iscell.npy``    (in-place float update)
     - ``<labels_csv>``           (one append row per flip with the
                                    new schema: plane0_path,
                                    recording_ID, ROI_number,
                                    user_defined_cell, timestamp_iso)
5. The curator can switch plane0s at any time; the popout closes
   and reopens against the new recording, preserving the labels
   CSV across switches (so a single retrain afterwards picks up
   everything labelled in the session).

Why this exists separately from Tab 3's popout
----------------------------------------------
Tab 3's popout is a one-off "click an ROI, flip it" workflow tied
to whichever recording the pipeline GUI is currently inspecting.
This standalone app is for *training-data collection*: a curator
walks dozens or hundreds of ROIs across many recordings in one
sitting, then retrains. Decoupled from the pipeline GUI so a
curator can label without spinning up suite2p / GPU stuff they
don't need.
"""
from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np


# ---------------- plane0 metadata cache ----------------

class _Plane0State:
    """Per-plane0 data the app needs to drive the popout.

    Lazily populated: ``stat`` / ``ops`` / ``probs`` / ``sorted_idx``
    are only loaded the first time the curator selects this plane0
    from the list (so adding 20 recordings doesn't pay 20 predicts
    up front; only the one the curator actually picks).
    """

    def __init__(self, plane0: Path):
        self.plane0 = Path(plane0)
        self.stat = None
        self.ops = None
        self.probs: np.ndarray | None = None
        self.sorted_idx: list[int] = []
        self.loaded = False

    def load(self, ckpt_arg: str | None, force_predict: bool,
             sort_mode: str) -> None:
        """Resolve probs (running predict if needed), load stat +
        ops, build the ``sorted_idx`` order. Idempotent: ``load(...)``
        on an already-loaded state re-sorts but doesn't reload arrays
        unless ``force_predict`` is True.
        """
        if not (self.plane0 / "stat.npy").exists():
            raise FileNotFoundError(
                f"no stat.npy in {self.plane0}; not a suite2p "
                f"plane0 folder")
        # Predict (or re-predict) if needed.
        from ..core.cellfilter import config as C
        prob_path = self.plane0 / C.PREDICTED_PROB_NAME
        if force_predict or not prob_path.exists():
            self._run_predict(ckpt_arg)
        # Load arrays + recompute sort.
        if not self.loaded:
            self.stat = np.load(self.plane0 / "stat.npy",
                                allow_pickle=True)
            ops_path = self.plane0 / "ops.npy"
            if not ops_path.exists():
                raise FileNotFoundError(
                    f"no ops.npy in {self.plane0}")
            self.ops = np.load(ops_path, allow_pickle=True).item()
            self.loaded = True
        self.probs = (np.load(prob_path).astype(float)
                      if prob_path.exists() else None)
        self.sorted_idx = _sort_order(
            int(len(self.stat)), self.probs, sort_mode)

    def _run_predict(self, ckpt_arg: str | None) -> None:
        from ..core.cellfilter import config as C
        from ..core.cellfilter.predict import (
            load_model_from_checkpoint,
            predict_recording,
        )
        from ..core import utils as _u
        import torch

        ckpt_path = Path(ckpt_arg) if ckpt_arg else (C.CHECKPOINT_DIR / "best.pt")
        if not ckpt_path.exists():
            print(
                f"[curation] no checkpoint at {ckpt_path}; ROIs will "
                f"be shown in Suite2p order without p(cell) scores",
                file=sys.stderr)
            return
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"[curation] scoring {self.plane0.name} with "
              f"{ckpt_path} on {device}...")
        model = load_model_from_checkpoint(ckpt_path, device)
        try:
            rec_id = _u.infer_recording_id(self.plane0)
        except Exception:
            rec_id = self.plane0.parent.name
        predict_recording(rec_id, model, device, plane0=self.plane0)


def _sort_order(n_total: int, probs: np.ndarray | None,
                mode: str) -> list[int]:
    """Return the ROI indices in the user-chosen browsing order."""
    if probs is None or len(probs) != n_total:
        return list(range(n_total))
    idx = np.arange(n_total)
    if mode == "unsure":
        return list(idx[np.argsort(np.abs(probs - 0.5))])
    if mode == "prob_asc":
        return list(idx[np.argsort(probs)])
    if mode == "prob_desc":
        return list(idx[np.argsort(-probs)])
    if mode == "index":
        return list(idx)
    raise SystemExit(
        f"unknown --sort mode {mode!r}; expected one of "
        f"unsure / prob_asc / prob_desc / index")


# ---------------- main app ----------------

class CurationApp:
    """Tk root window: plane0 list + the curation popout sub-window.

    The plane0 list is the persistent control panel (always visible);
    the curation popout is a child Toplevel that the app destroys
    and recreates each time the curator switches to a different
    plane0. Keeping the list in its own window means the curator
    can keep adding plane0s to the queue while a curation popout
    is open.
    """

    def __init__(self, plane0_paths: list[Path],
                 *, ckpt_arg: str | None,
                 sort_mode: str,
                 force_predict_initial: bool) -> None:
        self._ckpt_arg = ckpt_arg
        self._sort_mode = sort_mode
        self._force_predict_initial = force_predict_initial

        self._root = tk.Tk()
        self._root.title("CalLIOPE — ROI Curation")
        self._root.geometry("760x320")

        # Per-plane0 state, indexed by ``str(plane0)`` so the
        # Listbox's selected-row index lines up cleanly.
        self._states: dict[str, _Plane0State] = {}
        for p in plane0_paths:
            self._states[str(p)] = _Plane0State(p)

        # Active popout (a CurationPopout instance) and the plane0
        # it's currently showing. ``None`` between switches.
        self._popout = None
        self._popout_plane0: Path | None = None
        # Sorted-order position inside the currently-active plane0.
        self._cur_pos = 0

        self._build_ui()
        self._refresh_listbox()

    # ---- UI ----

    def _build_ui(self) -> None:
        top = ttk.Frame(self._root, padding=8)
        top.pack(fill="both", expand=True)

        ttk.Label(top, text="Plane0 folders to curate:",
                  font=("", 10, "bold")).pack(anchor="w")

        list_row = ttk.Frame(top)
        list_row.pack(fill="both", expand=True, pady=(2, 6))
        self._listbox = tk.Listbox(
            list_row, height=10, activestyle="dotbox",
            selectmode="browse")
        self._listbox.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(list_row, orient="vertical",
                           command=self._listbox.yview)
        sb.pack(side="right", fill="y")
        self._listbox.configure(yscrollcommand=sb.set)
        # Single-click selects; double-click opens curation.
        self._listbox.bind("<Double-Button-1>",
                           lambda _e: self._on_open_selected())
        self._listbox.bind("<Return>",
                           lambda _e: self._on_open_selected())

        btn_row = ttk.Frame(top); btn_row.pack(fill="x", pady=(0, 6))
        ttk.Button(btn_row, text="Add plane0...",
                   command=self._on_add).pack(side="left")
        ttk.Button(btn_row, text="Remove",
                   command=self._on_remove).pack(side="left",
                                                  padx=(6, 0))
        ttk.Button(btn_row, text="Open curation",
                   command=self._on_open_selected).pack(
                       side="left", padx=(6, 0))
        ttk.Button(btn_row, text="Re-predict selected",
                   command=self._on_repredict).pack(
                       side="left", padx=(6, 0))

        sort_row = ttk.Frame(top); sort_row.pack(fill="x")
        ttk.Label(sort_row, text="Sort ROIs by:").pack(side="left")
        self._sort_var = tk.StringVar(value=self._sort_mode)
        sort_box = ttk.Combobox(
            sort_row, textvariable=self._sort_var,
            values=("unsure", "prob_asc", "prob_desc", "index"),
            state="readonly", width=10)
        sort_box.pack(side="left", padx=(6, 0))
        sort_box.bind(
            "<<ComboboxSelected>>",
            lambda _e: self._on_sort_changed())

        self._status_var = tk.StringVar(
            value="Add one or more plane0 folders to begin.")
        ttk.Label(top, textvariable=self._status_var,
                  font=("", 9, "italic")).pack(anchor="w",
                                               pady=(6, 0))

    def _refresh_listbox(self) -> None:
        self._listbox.delete(0, "end")
        for key in self._states:
            self._listbox.insert("end", key)
        if self._states:
            self._listbox.selection_clear(0, "end")
            self._listbox.selection_set(0)

    # ---- Actions ----

    def _on_add(self) -> None:
        path = filedialog.askdirectory(
            title="Select a suite2p plane0 folder")
        if not path:
            return
        p = Path(path).resolve()
        if not (p / "stat.npy").exists():
            messagebox.showerror(
                "Not a plane0 folder",
                f"{p} has no stat.npy. Pick the recording's "
                f"<...>/detection/final/suite2p/plane0/ directory.")
            return
        key = str(p)
        if key in self._states:
            self._status_var.set(f"Already in list: {key}")
            return
        self._states[key] = _Plane0State(p)
        self._refresh_listbox()
        # Select the just-added entry so a quick Enter / double-
        # click opens it without scrolling.
        self._listbox.selection_clear(0, "end")
        self._listbox.selection_set("end")
        self._listbox.see("end")
        self._status_var.set(f"Added: {key}")

    def _on_remove(self) -> None:
        sel = self._listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        key = list(self._states.keys())[idx]
        # If the popout is showing this plane0 we close it so the
        # user doesn't keep flipping labels for a removed entry.
        if (self._popout is not None
                and self._popout_plane0 is not None
                and str(self._popout_plane0) == key):
            self._close_popout()
        self._states.pop(key, None)
        self._refresh_listbox()
        self._status_var.set(f"Removed: {key}")

    def _on_sort_changed(self) -> None:
        self._sort_mode = self._sort_var.get()
        # Re-sort every loaded state; un-loaded states pick up the
        # new mode lazily on first open.
        for st in self._states.values():
            if st.loaded:
                st.sorted_idx = _sort_order(
                    int(len(st.stat)), st.probs, self._sort_mode)
        # If a popout is open, jump to position 0 in the new sort.
        if self._popout is not None and self._popout_plane0 is not None:
            st = self._states[str(self._popout_plane0)]
            self._cur_pos = 0
            if st.sorted_idx:
                self._show_current_roi()

    def _on_repredict(self) -> None:
        sel = self._listbox.curselection()
        if not sel:
            messagebox.showinfo(
                "Re-predict",
                "Select a plane0 entry first.")
            return
        idx = sel[0]
        key = list(self._states.keys())[idx]
        st = self._states[key]
        try:
            st.load(self._ckpt_arg, force_predict=True,
                    sort_mode=self._sort_mode)
        except Exception as e:
            messagebox.showerror("Predict failed", str(e))
            return
        self._status_var.set(
            f"Re-predicted {st.plane0.name} "
            f"({len(st.sorted_idx)} ROIs)")
        # If the popout is currently showing this plane0 we refresh.
        if (self._popout is not None
                and self._popout_plane0 == st.plane0):
            self._open_for(st)

    def _on_open_selected(self) -> None:
        sel = self._listbox.curselection()
        if not sel:
            messagebox.showinfo(
                "Open curation",
                "Select a plane0 entry from the list first.")
            return
        idx = sel[0]
        key = list(self._states.keys())[idx]
        st = self._states[key]
        try:
            st.load(self._ckpt_arg,
                    force_predict=self._force_predict_initial,
                    sort_mode=self._sort_mode)
        except Exception as e:
            messagebox.showerror("Load failed", str(e))
            return
        self._open_for(st)

    # ---- Popout management ----

    def _open_for(self, st: _Plane0State) -> None:
        """Open (or re-open) the curation popout against ``st``.

        Destroys any existing popout first so the curator only ever
        has one curation window up at a time.

        ``on_iscell_changed`` is wired to ``_on_flipped`` so every
        successful cell / not-cell flip in the popout auto-advances
        to the next sorted ROI. Without this the curator would have
        to press the label key AND an arrow key for each ROI --
        twice the keystrokes per labelled sample.
        """
        self._close_popout()
        from ..tabs.suite2p.curation_popout import CurationPopout
        self._popout = CurationPopout(
            self._root, st.plane0, st.stat, st.ops,
            on_iscell_changed=self._on_flipped)
        self._popout_plane0 = st.plane0
        self._cur_pos = 0
        # Bind Prev / Next ROI navigation. Bound on the popout
        # Toplevel AND on the embedded matplotlib canvas because the
        # NavigationToolbar likes to take focus on hover -- without
        # the second binding the arrow keys feel dropped when the
        # mouse is over the figure.
        self._popout.bind("<Right>", lambda _e: self._step(+1))
        self._popout.bind("<Left>", lambda _e: self._step(-1))
        try:
            _cw = self._popout._canvas.get_tk_widget()
            _cw.bind("<Right>", lambda _e: self._step(+1))
            _cw.bind("<Left>", lambda _e: self._step(-1))
        except Exception:
            pass
        # The user opened the popout via a listbox double-click, so
        # keyboard focus is still on the listbox in the root window.
        # ``focus_force`` yanks it onto the popout so the curator's
        # next keystroke (``1`` / ``0`` / arrow) lands here without
        # an extra click to refocus.
        try:
            self._popout.focus_force()
        except Exception:
            pass
        # When the user closes the popout (Esc or X) we clear our
        # references but leave the main list window open so they
        # can pick another plane0.
        self._popout.protocol(
            "WM_DELETE_WINDOW", self._close_popout)
        self._popout.bind("<Escape>", lambda _e: self._close_popout())
        self._show_current_roi()
        self._status_var.set(
            f"Curating {st.plane0.name}  ({len(st.sorted_idx)} ROIs)")

    def _close_popout(self) -> None:
        if self._popout is not None:
            try:
                self._popout.destroy()
            except Exception:
                pass
        self._popout = None
        self._popout_plane0 = None

    def _on_flipped(self, _roi_idx: int, _new_value: int) -> None:
        """Listener fired by ``CurationPopout`` after a successful
        cell / not-cell flip. Auto-advances to the next ROI in the
        current sort order so the curator can label many ROIs in a
        row without touching the arrow keys. Wraps around at the
        end of the sorted list -- matches what Right-arrow would do.
        """
        self._step(+1)

    def _step(self, delta: int) -> None:
        if self._popout is None or self._popout_plane0 is None:
            return
        st = self._states[str(self._popout_plane0)]
        if not st.sorted_idx:
            return
        self._cur_pos = (self._cur_pos + delta) % len(st.sorted_idx)
        self._show_current_roi()

    def _show_current_roi(self) -> None:
        if self._popout is None or self._popout_plane0 is None:
            return
        st = self._states[str(self._popout_plane0)]
        if not st.sorted_idx:
            return
        roi_idx = st.sorted_idx[self._cur_pos]
        self._popout.show_roi(roi_idx)
        prob_str = (f"  p(cell)={st.probs[roi_idx]:.3f}"
                    if st.probs is not None else "")
        self._popout.title(
            f"ROI curation — {st.plane0.name}  "
            f"({self._cur_pos + 1} of {len(st.sorted_idx)})  "
            f"sort={self._sort_mode}{prob_str}")

    # ---- Main loop ----

    def mainloop(self) -> None:
        self._root.mainloop()


# ---------------- entry point ----------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "plane0", nargs="*",
        help="Zero or more plane0 directories to pre-load. The "
             "app's Add plane0 button can extend the list at "
             "runtime; passing nothing here is fine.")
    ap.add_argument(
        "--ckpt", default=None,
        help="Cell-filter checkpoint path. Defaults to "
             "cellfilter.config.CHECKPOINT_DIR/best.pt.")
    ap.add_argument(
        "--sort", default="unsure",
        choices=("unsure", "prob_asc", "prob_desc", "index"),
        help="ROI browsing order (default: unsure).")
    ap.add_argument(
        "--repredict", action="store_true",
        help="Force re-predict even if predicted_cell_prob.npy "
             "exists for plane0s opened from this session.")
    args = ap.parse_args(argv)

    plane0_paths: list[Path] = []
    for arg in args.plane0:
        p = Path(arg).resolve()
        if not p.is_dir():
            print(f"[curation] skipping {p}: not a directory",
                  file=sys.stderr)
            continue
        if not (p / "stat.npy").exists():
            print(f"[curation] skipping {p}: no stat.npy",
                  file=sys.stderr)
            continue
        plane0_paths.append(p)

    app = CurationApp(
        plane0_paths,
        ckpt_arg=args.ckpt,
        sort_mode=args.sort,
        force_predict_initial=args.repredict,
    )
    app.mainloop()


if __name__ == "__main__":
    main()
