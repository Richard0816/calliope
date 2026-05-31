"""calliope.tabs.batch.tab - Tab 0: Batch runner.

What this tab does
------------------
Drives the full Tabs 1 + 3-8 pipeline (preprocess -> detection ->
lowpass -> events -> clustering -> xcorr -> spatial figures) over a
list of recordings by **driving the GUI tabs themselves** rather than
calling headless backends. For each row it:

  1. Sets the relevant tab's input fields the same way a user would.
  2. Switches the notebook to that tab so live progress is visible.
  3. Calls the tab's ``_on_run`` (or equivalent).
  4. Subscribes one-shot to the AppState publish channel that signals
     the stage's completion (``set_result``, ``set_plane0``,
     ``set_lowpass_ready``, ``set_event_results``,
     ``set_clusters_ready``, ``set_xcorr_ready``).
  5. On receive, advances to the next stage. The user can switch into
     any tab mid-run to see the live state (panels, progress, log).

What the user sees
------------------
- Top: working directory + recursion depth + output folder + Scan;
  "Apply defaults to all rows" + "Run all" / "Abort".
- Middle: scrollable list of recording rows. Each row has an
  identifier entry, a list of TIFFs (with Browse), an "Edit
  parameters..." button, and a per-row status / progress label.
- Bottom: orchestration log -- per-stage start/done/timing for the
  active row. The detailed suite2p / preprocess output is shown in
  each individual tab's own console (which the user can view by
  clicking that tab during the run).

State persistence
-----------------
The list of rows + per-row params is serialised to
``<output>/calliope_batch.json`` on every Run All so closing &
re-opening the GUI restores the queue.

After every Run All, ``<output>/batch_report.csv`` is written with
``recording_id, status, plane0, total_s, <stage>_status,
<stage>_duration_s`` columns for downstream analysis.
"""

from __future__ import annotations

import csv
import gc
import json
import os
import queue
import shutil
import threading
import time
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk
from typing import Optional

from ...gui_common import (
    AppState, apply_dark_to_tk_widget, attach_resize_handle,
    drain_queue, open_advanced, pick_tiffs_dialog, spec_defaults,
)
from ...core.detection_run import archive_recording_post_detection
from .logic import (
    BATCH_JSON_NAME, BATCH_REPORT_NAME, SCRATCH_DETECTION_KEEP_DIRS,
    SCRATCH_PRUNE_FILE_PATTERNS, SCRATCH_RECORDING_INTERMEDIATE_DIRS,
    SINGLE_DRIVE_FINALIZE_RATE_BYTES_PER_SEC, TIFF_GLOBS,
    build_batch_param_spec, copy_one_dedup, is_scratch_mirror_skippable,
    measure_tree, mirror_sync_pass, prune_scratch_tree,
    synchronous_final_sweep, throttled_copy2,
)


class BatchRow:
    """One recording in the queue. Owns its widgets, identifier,
    tiff list, parameter overrides, and status label.
    """

    def __init__(self, parent, tab: "BatchTab",
                 *, identifier: str = "",
                 tiffs: Optional[list[str]] = None,
                 params: Optional[dict] = None) -> None:
        self.tab = tab
        self.frame = ctk.CTkFrame(parent, fg_color="transparent")
        self.frame.pack(fill="x", padx=4, pady=2)

        self.id_var = tk.StringVar(value=identifier)
        self.tiff_var = tk.StringVar(
            value="; ".join(tiffs or []))
        self.params: dict = dict(params) if params is not None else {}
        self.status_var = tk.StringVar(value="ready")
        # Per-row selection checkbox -- read by BatchTab._on_merge_selected
        # to combine multiple recordings into a single grouped row that
        # mirrors Tab 1's "select multiple TIFFs as one recording" path.
        self.selected_var = tk.BooleanVar(value=False)

        ctk.CTkCheckBox(self.frame, text="", width=24,
                        variable=self.selected_var) \
            .grid(row=0, column=0, padx=(0, 4))
        ctk.CTkEntry(self.frame, textvariable=self.id_var, width=200) \
            .grid(row=0, column=1, padx=(0, 6))
        ctk.CTkEntry(self.frame, textvariable=self.tiff_var, width=360) \
            .grid(row=0, column=2, padx=(0, 4), sticky="ew")
        ctk.CTkButton(self.frame, text="Browse...", width=90,
                      command=self._on_browse) \
            .grid(row=0, column=3, padx=(0, 4))
        ctk.CTkButton(self.frame, text="Edit params", width=110,
                      command=self._on_edit_params) \
            .grid(row=0, column=4, padx=(0, 4))
        ctk.CTkLabel(self.frame, textvariable=self.status_var,
                     width=140, anchor="w") \
            .grid(row=0, column=5, padx=(0, 4))
        ctk.CTkButton(self.frame, text="x", width=28,
                      command=self._on_remove) \
            .grid(row=0, column=6)
        self.frame.columnconfigure(2, weight=1)

    @property
    def identifier(self) -> str:
        return self.id_var.get().strip() or "unnamed"

    def tiff_list(self) -> list[str]:
        raw = self.tiff_var.get().strip()
        if not raw:
            return []
        return [s.strip() for s in raw.split(";") if s.strip()]

    def to_dict(self) -> dict:
        return {
            "identifier": self.id_var.get(),
            "tiffs": self.tiff_list(),
            "params": dict(self.params),
        }

    def set_status(self, msg: str) -> None:
        self.status_var.set(msg)

    def merge_defaults(self, defaults: dict) -> None:
        """Overwrite per-row params with defaults; called by the
        top-level "Apply to all rows" button."""
        self.params = dict(defaults)

    def destroy(self) -> None:
        self.frame.destroy()

    def _on_browse(self) -> None:
        paths = pick_tiffs_dialog(
            self.tab,
            title=f"TIFFs for '{self.identifier}'",
            initial_dir=self.tab.workdir_var.get().strip()
                        or self.tab.outdir_var.get().strip(),
            current_paths=self.tiff_list(),
            initial_depth=int(self.tab.depth_var.get() or 0),
        )
        if paths is not None:
            self.tiff_var.set("; ".join(paths))

    def _on_edit_params(self) -> None:
        # Start from row params -> tab defaults -> spec defaults.
        merged = dict(self.tab.default_params)
        merged.update(self.params)
        if open_advanced(
                self.tab,
                f"Edit parameters - '{self.identifier}'",
                self.tab.PARAM_SPEC, merged):
            self.params = dict(merged)

    def _on_remove(self) -> None:
        self.tab.remove_row(self)


class BatchTab(ctk.CTkFrame):
    """Tab 0: queue of recordings + run-all worker."""

    POLL_MS = 100

    PARAM_SPEC = build_batch_param_spec()

    def __init__(self, master, state: AppState) -> None:
        super().__init__(master, fg_color="transparent")
        self.state = state
        self._rows: list[BatchRow] = []
        self.default_params: dict = spec_defaults(self.PARAM_SPEC)

        self._log_queue: queue.Queue = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._abort_flag = threading.Event()

        # Deferred-cleanup janitor: a long-lived daemon thread that
        # retries ``shutil.rmtree`` on scratch_rec dirs the row-end
        # finalize couldn't remove on the first pass. Windows holds
        # file handles open on ``np.memmap`` files even after the
        # corresponding Python object is GC'd; downstream tabs
        # (Tab 4 / 5 / 6 / ...) viewing the just-finished recording
        # keep memmap handles on its scratch files until they're
        # reopened against a different recording or the app closes.
        # The janitor keeps retrying every ~30 s; eventually those
        # handles release (next recording's repoint reopens the
        # tab's memmaps elsewhere, or the user navigates away) and
        # the rmtree succeeds. No manual cleanup is ever required.
        #
        # Row status is kept in sync via ``_cleanup_row_map``: when
        # ``_finalize_worker`` enqueues a scratch_rec it also maps
        # the path -> (row, base_status) so the janitor can flip
        # the row label from ``"<status> (cleanup pending)"`` to
        # ``"<status> (done)"`` once rmtree succeeds. The map's
        # mutation lives under the same lock as the queue itself.
        self._cleanup_queue_lock = threading.Lock()
        self._cleanup_queue: list[Path] = []
        self._cleanup_row_map: dict[Path, tuple] = {}
        self._cleanup_thread = threading.Thread(
            target=self._scratch_janitor_loop,
            name="calliope-scratch-janitor",
            daemon=True)
        self._cleanup_thread.start()

        self._build_ui()
        drain_queue(self, self._log_queue,
                    {"log": self._on_log_payload,
                     "status": self._on_status_payload,
                     "done": self._on_batch_done},
                    poll_ms=self.POLL_MS)

    # ---- Scratch capacity pre-flight ----------------------------------

    # Multiplier on the row's source TIFF size that approximates the
    # peak scratch footprint while the row's pipeline is in flight:
    #   - shifted TIFF written by preprocess  ≈ 1.0x source
    #   - registered binary (_shared_reg/data.bin) ≈ 1.0x source
    #   - sparsery pass binary (sparsery_pass/data.bin) ≈ 1.0x source
    #   - final outputs (F/Fneu/stat/dff memmaps, npy arrays) ≈ 0.2x
    # The detection-stage prune drops the two ~1.0x binaries after
    # extraction folds into final/, but the peak (mid-detection)
    # holds all of them simultaneously. Sized to that peak so the
    # guard prevents the failure mode we actually hit.
    _SCRATCH_BUDGET_MULTIPLIER: float = 3.2

    # Extra safety margin on top of the per-row budget so we don't
    # arm-wrestle the disk-full threshold with millisecond-level
    # precision. Filesystem overhead + suite2p's transient working
    # files + a buffer for the OS page cache.
    _SCRATCH_SAFETY_MARGIN_BYTES: int = 2 * 1024 * 1024 * 1024  # 2 GB

    # How often the pre-flight re-checks free space when we're
    # waiting on background cleanups to free things up. Scheduled
    # via ``self.after`` so the GUI stays responsive.
    _SCRATCH_PREFLIGHT_RETRY_MS: int = 15_000

    # Throttle for the "scratch low, waiting for cleanup" log message
    # so the run log doesn't fill up if a finalize is slow.
    _SCRATCH_WAIT_LOG_INTERVAL_S: float = 60.0

    def _row_scratch_budget(self, row: "BatchRow") -> int:
        """Estimate the peak scratch footprint (bytes) for ``row``.

        Sums the row's source TIFFs and multiplies by
        ``_SCRATCH_BUDGET_MULTIPLIER``. TIFFs we can't ``stat``
        contribute 0 -- worst case the guard under-estimates and
        the pipeline catches the disk-full further down. The pre-
        flight is a heuristic, not a hard guarantee.
        """
        total = 0
        for t in row.tiff_list():
            try:
                total += Path(t).stat().st_size
            except OSError:
                continue
        return int(total * self._SCRATCH_BUDGET_MULTIPLIER)

    def _check_scratch_capacity(self, row: "BatchRow") -> str:
        """Return one of ``"ok"`` / ``"wait"`` / ``"fail"`` for whether
        we can safely start ``row``'s pipeline given current scratch
        free space and what's in flight.

        Behavior:
          - free >= needed + margin -> ``ok``
          - low free, anything in flight (finalize thread alive OR
            cleanup queue non-empty) -> ``wait`` (caller re-arms)
          - low free, nothing in flight -> ``fail`` (nothing will
            ever free up; honest error beats hanging forever)

        Logs a "scratch low, waiting" line at most once per
        ``_SCRATCH_WAIT_LOG_INTERVAL_S`` so the run log doesn't
        spam when a finalize is genuinely slow.
        """
        if self._batch_scratch_root is None:
            return "ok"
        needed = self._row_scratch_budget(row)
        try:
            free = shutil.disk_usage(
                str(self._batch_scratch_root)).free
        except OSError:
            # Can't measure -- defer to runtime disk-full handling.
            return "ok"
        if free >= needed + self._SCRATCH_SAFETY_MARGIN_BYTES:
            return "ok"
        # Low space. Anything pending that could free things up?
        with self._cleanup_queue_lock:
            pending_cleanup = len(self._cleanup_queue)
        in_flight_finalize = sum(
            1 for t in self._batch_finalize_threads if t.is_alive())
        if pending_cleanup == 0 and in_flight_finalize == 0:
            return "fail"
        now = time.time()
        last = getattr(self, "_last_scratch_wait_log", 0.0)
        if now - last >= self._SCRATCH_WAIT_LOG_INTERVAL_S:
            self._last_scratch_wait_log = now
            self._append_log(
                f"  [batch] scratch low: {free / 1e9:.1f} GB free, "
                f"need ~{needed / 1e9:.1f} GB for '{row.identifier}' "
                f"(+ {self._SCRATCH_SAFETY_MARGIN_BYTES / 1e9:.0f} GB "
                f"margin); waiting for {in_flight_finalize} "
                f"finalize + {pending_cleanup} janitor cleanup to "
                f"drain before starting next row")
        return "wait"

    def _fail_row_no_space(self, row: "BatchRow") -> None:
        """Mark ``row`` failed with an insufficient-scratch error,
        record it in ``_batch_results`` and the row label, and emit
        a single explanatory log line. Called when the pre-flight
        determines no cleanup will free enough space.
        """
        ident = row.identifier
        try:
            free = shutil.disk_usage(
                str(self._batch_scratch_root)).free
        except OSError:
            free = 0
        needed = self._row_scratch_budget(row)
        err = (
            f"insufficient scratch space: {free / 1e9:.1f} GB free "
            f"on {self._batch_scratch_root}, need "
            f"~{needed / 1e9:.1f} GB + "
            f"{self._SCRATCH_SAFETY_MARGIN_BYTES / 1e9:.0f} GB "
            f"margin, no background cleanup in flight")
        self._append_log(f"[{ident}] FAILED: {err}")
        row.set_status("failed (no scratch space)")
        self._batch_results.append({
            "recording_id": ident,
            "status": "failed",
            "plane0": "",
            "stages": {},
            "total_s": 0.0,
            "error": err,
        })

    # ---- Deferred scratch cleanup --------------------------------------

    _CLEANUP_RETRY_INTERVAL_S: float = 30.0

    def _sweep_scratch_orphans(self, scratch_root: Path) -> None:
        """Queue every subdirectory of ``scratch_root`` that isn't a
        recording identifier in the about-to-start batch.

        Called once at the top of ``_on_run_all`` (scratch-routing
        mode only). Catches the case where a previous batch crashed,
        was force-quit, or hit a hard lock that survived the
        janitor's retry window when the app exited -- on fresh
        startup all file handles are released, so these orphans
        clear on the first janitor pass.

        Does NOT touch directories that match the current queue's
        identifiers: those are about to be rewritten by
        ``_batch_next_row``'s force-rerun rmtree.
        """
        if not scratch_root.exists():
            return
        active_idents = {r.identifier for r in self._rows}
        try:
            children = [c for c in scratch_root.iterdir() if c.is_dir()]
        except OSError:
            return
        orphans = [c for c in children if c.name not in active_idents]
        for p in orphans:
            self._enqueue_scratch_cleanup(p)
        if orphans:
            self._log_queue.put((
                "log",
                f"  [batch] scratch janitor: queued {len(orphans)} "
                f"orphan dir{'' if len(orphans) == 1 else 's'} from "
                f"a previous session for cleanup"))

    def _after_safe(self, fn, *args) -> None:
        """``self.after(0, fn, *args)`` from a background thread, with
        the ``TclError`` you get when the widget is being destroyed
        suppressed so a late janitor callback can't crash on app
        shutdown.
        """
        try:
            self.after(0, lambda: fn(*args))
        except Exception:
            pass

    def _mark_row_cleanup_done(self, row, base_status: str) -> None:
        """Update a row's status label to reflect that the janitor
        successfully cleared its scratch_rec. Runs on the Tk main
        thread (scheduled via ``_after_safe``).
        """
        try:
            row.set_status(
                f"{base_status} (done)" if base_status else "done")
        except Exception:
            pass

    def _enqueue_scratch_cleanup(self, scratch_rec: Path,
                                 *, row=None,
                                 base_status: str = "") -> None:
        """Queue ``scratch_rec`` for periodic rmtree retries by the
        janitor thread. Idempotent: re-enqueuing an already-queued
        path is a no-op. The janitor removes entries that no longer
        exist on disk so this stays bounded over a long session.

        ``row`` + ``base_status`` are optional UI-binding fields:
        when supplied, the janitor flips the row's status label
        from ``"<base_status> (cleanup pending)"`` to
        ``"<base_status> (done)"`` once rmtree succeeds, so the
        Tab 0 status column reflects real cleanup progress.
        """
        with self._cleanup_queue_lock:
            if scratch_rec not in self._cleanup_queue:
                self._cleanup_queue.append(scratch_rec)
            if row is not None:
                self._cleanup_row_map[scratch_rec] = (row, base_status)

    def _scratch_janitor_loop(self) -> None:
        """Background loop that retries rmtree on queued scratch_recs.

        Runs forever (daemon). Each pass:
          - Snapshots the queue under the lock
          - For each entry: drop if already gone, else try rmtree
          - Successful entries are removed from the queue
          - Sleeps ``_CLEANUP_RETRY_INTERVAL_S`` between passes

        ``shutil.rmtree(ignore_errors=False)`` raises on first lock
        failure -- we catch and leave the entry queued for the next
        pass. ``gc.collect()`` runs before each retry so any
        recently-released Python memmap references give up their
        Windows handles before we try again.
        """
        while True:
            try:
                with self._cleanup_queue_lock:
                    snapshot = list(self._cleanup_queue)

                for path in snapshot:
                    try:
                        cleared_now = False
                        if not path.exists():
                            cleared_now = True
                        else:
                            gc.collect()
                            shutil.rmtree(str(path), ignore_errors=False)
                            cleared_now = True
                            self._log_queue.put((
                                "log",
                                f"  [batch] scratch janitor: cleared "
                                f"{path}"))
                        if cleared_now:
                            with self._cleanup_queue_lock:
                                if path in self._cleanup_queue:
                                    self._cleanup_queue.remove(path)
                                binding = self._cleanup_row_map.pop(
                                    path, None)
                            if binding is not None:
                                row, base = binding
                                self._after_safe(
                                    self._mark_row_cleanup_done,
                                    row, base)
                    except OSError:
                        # Still locked; try again next pass.
                        continue
            except Exception:
                # The janitor must never die -- any unexpected error
                # would leak scratch indefinitely.
                pass
            time.sleep(self._CLEANUP_RETRY_INTERVAL_S)

    # -- UI ----------------------------------------------------------------

    def _build_ui(self) -> None:
        top = ctk.CTkFrame(self)
        top.pack(fill="x", padx=10, pady=(10, 6))
        ctk.CTkLabel(top, text="Batch run setup",
                     font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(6, 0))

        row = ctk.CTkFrame(top, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=2)
        ctk.CTkLabel(row, text="Working dir:", width=110,
                     anchor="w").pack(side="left")
        self.workdir_var = tk.StringVar()
        ctk.CTkEntry(row, textvariable=self.workdir_var) \
            .pack(side="left", fill="x", expand=True, padx=(0, 4))
        ctk.CTkButton(row, text="Browse...", width=90,
                      command=self._on_pick_workdir).pack(side="left")
        ctk.CTkLabel(row, text="Depth:").pack(side="left", padx=(8, 2))
        self.depth_var = tk.IntVar(value=2)
        ttk.Spinbox(row, from_=0, to=6, width=4,
                    textvariable=self.depth_var).pack(side="left")
        ctk.CTkButton(row, text="Scan", width=80,
                      command=self._on_scan).pack(side="left", padx=(8, 0))

        row = ctk.CTkFrame(top, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=2)
        ctk.CTkLabel(row, text="Output folder:", width=110,
                     anchor="w").pack(side="left")
        self.outdir_var = tk.StringVar()
        ctk.CTkEntry(row, textvariable=self.outdir_var) \
            .pack(side="left", fill="x", expand=True, padx=(0, 4))
        ctk.CTkButton(row, text="Browse...", width=90,
                      command=self._on_pick_outdir).pack(side="left")
        ctk.CTkButton(row, text="Reload queue", width=120,
                      command=self._on_reload_queue) \
            .pack(side="left", padx=(8, 0))
        # CSV export / import. JSON-via-output-folder is the canonical
        # roundtrip; CSV is the spreadsheet-friendly alternative.
        ctk.CTkButton(row, text="Export CSV...", width=120,
                      command=self._on_export_queue_csv) \
            .pack(side="left", padx=(4, 0))
        ctk.CTkButton(row, text="Load CSV...", width=110,
                      command=self._on_load_queue_csv) \
            .pack(side="left", padx=(4, 0))

        # Optional scratch dir on a fast drive (SSD/NVMe). When set, every
        # recording's intermediate I/O lives on this scratch path during
        # the run, then bulk-copies to the slow output folder above.
        row = ctk.CTkFrame(top, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=2)
        ctk.CTkLabel(row, text="Scratch dir (SSD):", width=110,
                     anchor="w").pack(side="left")
        self.scratch_dir_var = tk.StringVar()
        ctk.CTkEntry(row, textvariable=self.scratch_dir_var) \
            .pack(side="left", fill="x", expand=True, padx=(0, 4))
        ctk.CTkButton(row, text="Browse...", width=90,
                      command=self._on_pick_scratch).pack(side="left")
        ctk.CTkLabel(row,
                     text="(optional; outputs still land in the folder above)",
                     text_color="gray").pack(side="left", padx=(8, 0))

        # Toggle for whether the headless pipeline emits PNG + SVG per
        # stage under ``<output>/<id>/calliope_figures/``. Default ON so
        # batch runs are self-documenting; turn off only when you're
        # short on disk and only want the npy / xlsx outputs.
        row = ctk.CTkFrame(top, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=(2, 0))
        ctk.CTkLabel(row, text="", width=110).pack(side="left")
        self.save_figures_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            row,
            text="Save figures during batch (PNG + SVG per stage + manifest)",
            variable=self.save_figures_var,
        ).pack(side="left")

        # --- Output: what to save ------------------------------------
        # Lets the user shrink each recording's output folder by dropping
        # large artefacts they don't need to re-run detection or archive
        # externally. Defaults preserve today's behavior exactly:
        # registration + shifted TIFF kept, raw compression off. The
        # preset bulk-sets the three checkboxes; users can still tweak
        # each one afterward. Read at finalize (see _finalize_recording_row
        # / _start_continuous_mirror), never persisted -- matches
        # save_figures_var, which is also session-only.
        self.keep_registration_var = tk.BooleanVar(value=True)
        self.keep_shifted_var = tk.BooleanVar(value=True)
        self.compress_raw_var = tk.BooleanVar(value=False)

        def _apply_save_preset(choice: str) -> None:
            if choice == "Bare minimum (Calliope-only)":
                # Keep only what every CalLIOPE tab needs to render/reload
                # its results. data.bin and the shifted TIFF are used only
                # for re-running detection / Tab 1 reload, not for
                # rendering, so they go. No external raw archive.
                self.keep_registration_var.set(False)
                self.keep_shifted_var.set(False)
                self.compress_raw_var.set(False)
            else:  # "Keep everything" -- today's behavior
                self.keep_registration_var.set(True)
                self.keep_shifted_var.set(True)
                self.compress_raw_var.set(False)

        row = ctk.CTkFrame(top, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=(6, 0))
        ctk.CTkLabel(row, text="What to save", width=110,
                     anchor="w").pack(side="left")
        self.save_preset_var = tk.StringVar(value="Keep everything")
        ctk.CTkSegmentedButton(
            row,
            values=["Keep everything", "Bare minimum (Calliope-only)"],
            variable=self.save_preset_var,
            command=_apply_save_preset,
        ).pack(side="left")

        row = ctk.CTkFrame(top, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=(2, 0))
        ctk.CTkLabel(row, text="", width=110).pack(side="left")
        ctk.CTkCheckBox(
            row,
            text="Keep suite2p registration (data.bin)",
            variable=self.keep_registration_var,
        ).pack(side="left")
        ctk.CTkCheckBox(
            row,
            text="Keep shifted / motion-corrected TIFF",
            variable=self.keep_shifted_var,
        ).pack(side="left", padx=(12, 0))
        ctk.CTkCheckBox(
            row,
            text="Archive raw TIFF (compressed)",
            variable=self.compress_raw_var,
        ).pack(side="left", padx=(12, 0))

        row = ctk.CTkFrame(top, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=(0, 0))
        ctk.CTkLabel(
            row,
            text=("Dropping registration means detection can't be re-run "
                  "without re-registering; dropping the shifted TIFF means "
                  "Tab 1 reload must regenerate it. ROIs + dF/F always kept."),
            width=110, anchor="w", justify="left",
            text_color="gray",
        ).pack(side="left", fill="x", expand=True)

        row = ctk.CTkFrame(top, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=(6, 6))
        ctk.CTkButton(row, text="Apply defaults to all rows", width=200,
                      command=self._on_edit_defaults) \
            .pack(side="left")
        ctk.CTkButton(row, text="+ Add row", width=110,
                      command=self.add_row) \
            .pack(side="left", padx=(8, 0))
        # Merge selected (checked) rows into one grouped recording.
        ctk.CTkButton(row, text="Merge selected", width=130,
                      command=self._on_merge_selected) \
            .pack(side="left", padx=(8, 0))
        self.run_btn = ctk.CTkButton(
            row, text="Run all", width=110,
            command=self._on_run_all)
        self.run_btn.pack(side="right")
        self.abort_btn = ctk.CTkButton(
            row, text="Abort", width=90,
            command=self._on_abort, state="disabled")
        self.abort_btn.pack(side="right", padx=(0, 6))

        # Headers + scrollable rows -----------------------------------------
        # First column = the per-row selection checkbox (no header text);
        # the rest match BatchRow's grid order.
        # Each header label width is in CTkLabel pixels (~7 per char) so
        # it lines up with each column underneath.
        headers = ctk.CTkFrame(self, fg_color="transparent")
        headers.pack(fill="x", padx=14)
        for txt, w in (("", 24),
                       ("Identifier", 200),
                       ("TIFF file(s) (semicolon-sep)", 360),
                       ("", 90), ("", 110), ("Status", 140), ("", 28)):
            ctk.CTkLabel(headers, text=txt, width=w, anchor="w") \
                .pack(side="left", padx=(0, 4))

        # Recordings list and Run log each live in their own CTkFrame
        # wrap with a draggable grip below. Resizing one extends just
        # that panel and the surrounding scrollable wrapper grows the
        # tab body to absorb it -- no top-level PanedWindow so dragging
        # one panel never shrinks the other.
        body_wrap = ctk.CTkFrame(self, fg_color="transparent")
        body_wrap.pack(fill="x", padx=14, pady=(0, 0))
        body = ctk.CTkFrame(body_wrap)
        body.pack(fill="both", expand=True)
        ctk.CTkLabel(body, text="Recordings",
                     font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(6, 0))
        # tk.Canvas because CTk has no scrollable-canvas widget --
        # we need a canvas-window so the inner rows frame can grow
        # independently of the viewport. Dark-skin the canvas *before*
        # we add the rows-frame window so CTkFrame's "transparent"
        # mode resolves to the dark canvas bg (not the default white
        # at construction time).
        canvas_holder = ctk.CTkFrame(body, fg_color="transparent")
        canvas_holder.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        canvas = tk.Canvas(canvas_holder, highlightthickness=0)
        apply_dark_to_tk_widget(canvas)
        canvas.pack(side="left", fill="both", expand=True)
        sb = ctk.CTkScrollbar(canvas_holder, orientation="vertical",
                              command=canvas.yview)
        sb.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=sb.set)
        self._rows_frame = ctk.CTkFrame(canvas, fg_color="transparent")
        self._rows_canvas_id = canvas.create_window(
            (0, 0), window=self._rows_frame, anchor="nw")

        def _on_frame_resize(_e=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_resize(e):
            canvas.itemconfig(self._rows_canvas_id, width=e.width)

        self._rows_frame.bind("<Configure>", _on_frame_resize)
        canvas.bind("<Configure>", _on_canvas_resize)
        attach_resize_handle(self, body_wrap, min_height=180,
                             initial_height=260)

        # Console -----------------------------------------------------------
        console_wrap = ctk.CTkFrame(self, fg_color="transparent")
        console_wrap.pack(fill="x", padx=14, pady=(0, 14))
        console_frame = ctk.CTkFrame(console_wrap)
        console_frame.pack(fill="both", expand=True)
        ctk.CTkLabel(console_frame, text="Run log",
                     font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(6, 0))
        self.log = ctk.CTkTextbox(console_frame, height=180, wrap="word",
                                  state="disabled")
        self.log.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        # Wheel routing is centralised in the app root's
        # ``install_scroll_router``: when the cursor sits over the
        # rows canvas the walk-up finds this scrollable tk.Canvas
        # first and scrolls it; everywhere else in the tab falls
        # through to scrolling the tab wrapper.
        attach_resize_handle(self, console_wrap, min_height=160,
                             initial_height=220)

    # -- Row management ---------------------------------------------------

    def add_row(self, *, identifier: str = "",
                tiffs: Optional[list[str]] = None,
                params: Optional[dict] = None) -> BatchRow:
        row = BatchRow(self._rows_frame, self,
                       identifier=identifier, tiffs=tiffs,
                       params=params if params is not None
                                       else dict(self.default_params))
        self._rows.append(row)
        return row

    def remove_row(self, row: BatchRow) -> None:
        if row in self._rows:
            self._rows.remove(row)
        row.destroy()

    def _clear_rows(self) -> None:
        for row in list(self._rows):
            row.destroy()
        self._rows.clear()

    @staticmethod
    def _trailing_index(name: str) -> int:
        """Last run of digits in a path's stem; -1 if there isn't one.

        Used to order TIFFs that came in via different rows but together
        form a continuous recording (e.g. ``rec_000.tif``,
        ``rec_001.tif``, ...). Mirrors ``PreprocessTab._trailing_index``
        so merged rows feed the preprocess tab in the same order the
        user would have got selecting them in Tab 1's listbox.
        """
        import re
        stem = Path(name).stem
        m = re.search(r"(\d+)(?!.*\d)", stem)
        return int(m.group(1)) if m else -1

    def _on_merge_selected(self) -> None:
        """Combine all checked rows into one grouped recording row.

        Mirrors what Tab 1 does when the user selects multiple TIFFs in
        its listbox:
          - Concatenate every checked row's TIFFs in trailing-index order.
          - Auto-derive the new identifier as ``+``-joined TIFF stems
            (matches PreprocessTab._resolved_identifier).
          - Inherit per-row params from the first checked row.
          - Replace the checked rows in place at the position of the
            first checked row, then remove the rest.
        """
        if getattr(self, "_batch_active", False):
            messagebox.showinfo("Busy", "Stop the batch run before merging.")
            return
        checked = [r for r in self._rows if r.selected_var.get()]
        if len(checked) < 2:
            messagebox.showinfo(
                "Pick at least 2 rows",
                "Tick the box on each row you want to merge into one "
                "grouped recording, then click Merge selected.")
            return

        # Pool the TIFFs and order by trailing index so the merged
        # recording reads in numeric sequence (rec_000 -> rec_001 -> ...).
        # Dedupe by absolute path so the same TIFF can't appear twice.
        seen: set[str] = set()
        merged_tiffs: list[str] = []
        for r in checked:
            for t in r.tiff_list():
                key = str(Path(t).resolve())
                if key in seen:
                    continue
                seen.add(key)
                merged_tiffs.append(t)
        merged_tiffs.sort(key=self._trailing_index)

        merged_id = "+".join(Path(t).stem for t in merged_tiffs)
        merged_params = dict(checked[0].params or {})

        # Splice: insert at the first checked row's position, drop the
        # checked set, then bring the new row visually to that slot.
        first = checked[0]
        insert_at = self._rows.index(first)
        for r in checked:
            self.remove_row(r)
        new_row = self.add_row(
            identifier=merged_id, tiffs=merged_tiffs, params=merged_params)
        # Move the new row to where the first checked row used to be.
        if new_row is not self._rows[insert_at]:
            self._rows.remove(new_row)
            self._rows.insert(insert_at, new_row)
            # Re-pack so the on-screen order matches the list order.
            for r in self._rows:
                r.frame.pack_forget()
            for r in self._rows:
                r.frame.pack(fill="x", padx=4)

        self._append_log(
            f"Merged {len(checked)} rows into '{merged_id}' "
            f"({len(merged_tiffs)} TIFFs).")

    # -- Top-bar actions ---------------------------------------------------

    def _on_pick_workdir(self) -> None:
        path = filedialog.askdirectory(title="Working directory")
        if path:
            self.workdir_var.set(path)

    def _on_pick_outdir(self) -> None:
        path = filedialog.askdirectory(title="Output folder")
        if path:
            self.outdir_var.set(path)

    def _on_pick_scratch(self) -> None:
        path = filedialog.askdirectory(title="Scratch directory (SSD)")
        if path:
            self.scratch_dir_var.set(path)

    def _on_scan(self) -> None:
        """Walk the working directory up to ``depth`` levels and add
        one row per TIFF file.

        Each ``.tif`` / ``.tiff`` becomes its own queue row keyed by
        the file's stem (filename without extension). When a folder
        holds multiple TIFFs, each one gets its own row -- the user
        can still combine them into a grouped recording afterwards
        via "Merge selected" if they're actually a multi-file series.

        Identifier collisions (two TIFFs with the same stem under
        different folders) are disambiguated by appending the
        parent folder name: ``<stem>__<parent_folder>``. Without
        the suffix the queue would write both rows into the same
        ``<output>/<ident>/`` and the second run would clobber the
        first.
        """
        root = self.workdir_var.get().strip()
        if not root:
            messagebox.showinfo("Pick working directory",
                                "Choose a folder to scan first.")
            return
        depth = max(0, int(self.depth_var.get()))
        root_path = Path(root)
        if not root_path.is_dir():
            messagebox.showerror("Bad path", f"{root} is not a folder.")
            return

        # Collect every TIFF we find at any depth, deduped by
        # resolved path so a folder appearing twice in the walk
        # doesn't double-add.
        tiff_paths: list[Path] = []
        seen: set[str] = set()
        for path in self._walk_to_depth(root_path, depth):
            for t in self._tiffs_in(path):
                key = str(Path(t).resolve())
                if key in seen:
                    continue
                seen.add(key)
                tiff_paths.append(Path(t))
        if not tiff_paths:
            messagebox.showinfo(
                "No TIFFs",
                f"Found no .tif/.tiff files under {root_path} "
                f"at depth {depth}.")
            return

        # Build identifiers from each file's stem, disambiguating
        # collisions with a parent-folder suffix. Sort by resolved
        # path so the queue order is reproducible.
        tiff_paths.sort()
        stem_counts: dict[str, int] = {}
        for t in tiff_paths:
            stem_counts[t.stem] = stem_counts.get(t.stem, 0) + 1
        for t in tiff_paths:
            stem = t.stem
            if stem_counts[stem] > 1:
                ident = f"{stem}__{t.parent.name}"
            else:
                ident = stem
            self.add_row(identifier=ident, tiffs=[str(t)])
        self._log(
            f"Scan: added {len(tiff_paths)} rows "
            f"(one per TIFF) from {root_path}")

    @staticmethod
    def _walk_to_depth(root: Path, depth: int):
        """Yield ``root`` and every subfolder up to ``depth`` levels
        deep (depth=0 means just ``root``).
        """
        yield root
        if depth <= 0:
            return
        stack = [(root, 0)]
        while stack:
            cur, d = stack.pop()
            if d >= depth:
                continue
            try:
                children = [c for c in cur.iterdir() if c.is_dir()]
            except OSError:
                continue
            for child in children:
                yield child
                stack.append((child, d + 1))

    @staticmethod
    def _tiffs_in(folder: Path) -> list[str]:
        out: list[str] = []
        for pat in TIFF_GLOBS:
            out.extend(str(p) for p in folder.glob(pat))
        return sorted(set(out))

    def _on_edit_defaults(self) -> None:
        snapshot = dict(self.default_params)
        if not open_advanced(
                self, "Default parameters (apply to all rows)",
                self.PARAM_SPEC, snapshot):
            return
        self.default_params = snapshot
        for row in self._rows:
            row.merge_defaults(self.default_params)
        self._log(f"Defaults updated and applied to {len(self._rows)} rows.")

    def _on_reload_queue(self) -> None:
        out = self.outdir_var.get().strip()
        if not out:
            messagebox.showinfo("Pick output folder",
                                "Set the output folder first.")
            return
        path = Path(out) / BATCH_JSON_NAME
        if not path.exists():
            messagebox.showinfo("No saved queue",
                                f"{path} not found.")
            return
        try:
            data = json.loads(path.read_text())
        except Exception as e:
            messagebox.showerror("Reload failed", str(e))
            return
        self._clear_rows()
        defaults = data.get("default_params")
        if isinstance(defaults, dict):
            self.default_params = defaults
        scratch = data.get("scratch_dir")
        if isinstance(scratch, str):
            self.scratch_dir_var.set(scratch)
        for entry in data.get("rows", []):
            self.add_row(
                identifier=entry.get("identifier", ""),
                tiffs=entry.get("tiffs", []),
                params=entry.get("params", {}),
            )
        self._log(f"Loaded {len(self._rows)} rows from {path}")

    # -- CSV import / export ----------------------------------------------
    #
    # Schema: a flat three-column CSV that Excel / pandas can edit
    # without ceremony. Columns:
    #   identifier  -- the row's name (drives the output folder)
    #   tiffs       -- semicolon-separated absolute paths to TIFFs
    #   params      -- JSON-encoded dict of per-row overrides
    #                  (e.g. {"gcamp_variant": "GCaMP6f  (tau = 0.7 s)",
    #                         "cutoff_hz": 0.5}); empty string == no
    #                  overrides (the tab's current defaults apply at
    #                  run time via the existing merge order).
    # Header row required. The ``params`` column is OPTIONAL on read:
    # a CSV exported by an older Calliope (only identifier / tiffs)
    # still loads cleanly -- missing column == empty dict per row.
    # Extra columns are ignored on read.

    _CSV_HEADERS = ("identifier", "tiffs", "params")

    def _on_export_queue_csv(self) -> None:
        """Write the current queue to a CSV the user picks. One row
        per recording. Skips empty rows (no identifier + no TIFFs).
        """
        if not self._rows:
            messagebox.showinfo("Empty queue",
                                "Add at least one row before exporting.")
            return
        # Default to the output folder if set, otherwise the working
        # dir, otherwise wherever the user last chose.
        initial_dir = (self.outdir_var.get().strip()
                       or self.workdir_var.get().strip())
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Export queue to CSV",
            defaultextension=".csv",
            initialdir=initial_dir or None,
            initialfile="calliope_queue.csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            n_written = 0
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(self._CSV_HEADERS)
                for r in self._rows:
                    tiffs = r.tiff_list()
                    ident = r.id_var.get().strip()
                    if not ident and not tiffs:
                        continue
                    # Re-join with "; " (space after semicolon) to
                    # match the GUI's display format. CSV's own
                    # quoting handles any commas inside paths.
                    # Params serialize as a compact JSON blob so the
                    # entire override dict round-trips in one column;
                    # empty dict writes as "" to keep the CSV readable
                    # for the common no-overrides case.
                    params_blob = (json.dumps(r.params, sort_keys=True)
                                   if r.params else "")
                    w.writerow([ident, "; ".join(tiffs), params_blob])
                    n_written += 1
            self._log(f"Exported {n_written} rows to {path}")
        except OSError as e:
            messagebox.showerror("Export failed", str(e))

    def _on_load_queue_csv(self) -> None:
        """Pick a CSV (with the same schema ``_on_export_queue_csv``
        writes) and replace the queue with its contents.

        The ``params`` column is optional: CSVs exported by older
        Calliope versions (identifier / tiffs only) still load, with
        each row's params defaulting to ``{}`` so the tab's current
        defaults apply at run time via the existing merge order.
        """
        initial_dir = (self.outdir_var.get().strip()
                       or self.workdir_var.get().strip())
        path = filedialog.askopenfilename(
            parent=self,
            title="Load queue from CSV",
            initialdir=initial_dir or None,
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        bad_params_rows = 0
        try:
            with open(path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None or \
                        "tiffs" not in reader.fieldnames:
                    messagebox.showerror(
                        "Bad CSV",
                        f"{path} doesn't have a 'tiffs' column. "
                        f"Expected headers: "
                        f"{', '.join(self._CSV_HEADERS)} "
                        f"(params is optional)")
                    return
                has_params_col = "params" in reader.fieldnames
                rows: list[tuple[str, list[str], dict]] = []
                for entry in reader:
                    ident = (entry.get("identifier") or "").strip()
                    raw_tiffs = (entry.get("tiffs") or "").strip()
                    if not raw_tiffs and not ident:
                        continue
                    # Accept either "; " or ";" as the separator so
                    # hand-edited CSVs work whether or not the user
                    # kept the GUI's trailing space.
                    tiffs = [s.strip() for s in raw_tiffs.split(";")
                             if s.strip()]
                    params: dict = {}
                    if has_params_col:
                        raw_params = (entry.get("params") or "").strip()
                        if raw_params:
                            try:
                                parsed = json.loads(raw_params)
                                if isinstance(parsed, dict):
                                    params = parsed
                                else:
                                    bad_params_rows += 1
                            except (ValueError, TypeError):
                                bad_params_rows += 1
                    rows.append((ident, tiffs, params))
        except OSError as e:
            messagebox.showerror("Load failed", str(e))
            return
        except csv.Error as e:
            messagebox.showerror("Bad CSV", f"Parse error: {e}")
            return
        if not rows:
            messagebox.showinfo("Empty CSV",
                                f"{path} had no rows to load.")
            return
        self._clear_rows()
        for ident, tiffs, params in rows:
            # Empty params dict falls through to the tab's current
            # defaults at run time via the existing merge order.
            self.add_row(identifier=ident, tiffs=tiffs, params=params)
        self._log(f"Loaded {len(rows)} rows from {path}")
        if bad_params_rows:
            self._log(
                f"  WARNING: {bad_params_rows} row(s) had unparseable "
                f"'params' JSON; those rows loaded with empty overrides.")

    # -- Persistence ------------------------------------------------------

    def _save_queue(self) -> None:
        out = self.outdir_var.get().strip()
        if not out:
            return
        out_path = Path(out)
        out_path.mkdir(parents=True, exist_ok=True)
        payload = {
            "default_params": self.default_params,
            "scratch_dir": self.scratch_dir_var.get().strip(),
            "rows": [r.to_dict() for r in self._rows],
        }
        (out_path / BATCH_JSON_NAME).write_text(
            json.dumps(payload, indent=2, default=str))

    # -- Run all ----------------------------------------------------------

    # ---------------------------------------------------------------------
    # Batch orchestration
    # ---------------------------------------------------------------------
    # The batch runner drives the GUI tabs in sequence. For each row it
    # walks STAGES, sets the relevant tab's input fields the same way a
    # user would, then calls that tab's ``_on_run`` (or equivalent) and
    # subscribes one-shot to the AppState publish channel that signals
    # the stage's completion (``set_result``, ``set_plane0``,
    # ``set_lowpass_ready``, ``set_event_results``,
    # ``set_clusters_ready``, ``set_xcorr_ready``). When the publish
    # fires, it advances to the next stage. Spatial propagation (Tab 8)
    # has no Run button -- it renders reactively from
    # ``set_event_results`` so the orchestrator just waits a beat for
    # it to draw and advances.
    #
    # Why drive the GUI rather than call into headless backends: lets
    # the user click into any tab mid-run and see the live state
    # (panels, progress, log) as the pipeline progresses through that
    # recording.

    STAGES = (
        "preprocess",
        "detection",
        "lowpass",
        "event_detection",
        "clustering",
        "crosscorrelation",
        "spatial_propagation",
    )

    # Notebook indices matching pipeline_gui's add() order. Tab 0 is
    # this batch tab; the orchestrator never selects it during a run.
    _STAGE_TAB_INDEX = {
        "preprocess":         1,
        "detection":          3,
        "lowpass":            4,
        "event_detection":    5,
        "clustering":         6,
        "crosscorrelation":   7,
        "spatial_propagation": 8,
    }

    def _check_drive_collisions(self, out: str) -> list[str]:
        """Return a list of warning strings describing R/W contention
        risks for the planned run. Empty list = layout is clean.

        Compares ``os.stat(path).st_dev`` (the OS device id of a
        path's underlying filesystem) across:

        - the output folder
        - every queued row's source TIFFs
        - the scratch dir (if set)

        Same ``st_dev`` = same partition = same physical drive (in
        practice -- this can miss exotic LVM / RAID setups, but
        covers the standard single-disk-per-mountpoint case on
        Windows + Linux). Failures to stat are silently dropped so
        the warning is best-effort and never blocks the run.

        Two concrete contention modes called out:
        - Source TIFFs and Output on the same drive: HDD seeks
          between the next row's preprocess read and the previous
          row's finalize write. 2-5x throughput drop.
        - Scratch and Output on the same drive: defeats the
          purpose of scratch -- every byte still passes through
          the slow drive once.
        """
        out_path = Path(out)
        try:
            out_dev = out_path.resolve().stat().st_dev
        except OSError:
            return []

        # Map device id -> example path for nicer warning messages.
        source_devs: dict[int, str] = {}
        for row in self._rows:
            for t in row.tiff_list():
                try:
                    parent = Path(t).resolve().parent
                    dev = parent.stat().st_dev
                    source_devs.setdefault(dev, str(parent))
                except OSError:
                    continue

        scratch = self.scratch_dir_var.get().strip()
        scratch_dev: Optional[int] = None
        scratch_path: Optional[Path] = None
        if scratch:
            try:
                scratch_path = Path(scratch)
                scratch_dev = scratch_path.resolve().stat().st_dev
            except OSError:
                pass

        issues: list[str] = []
        if out_dev in source_devs:
            issues.append(
                f"Source TIFFs and Output folder are on the same "
                f"physical drive (example source: {source_devs[out_dev]} "
                f"vs output: {out_path}). HDDs collapse to ~30-60 MB/s "
                f"during mixed reads+writes -- expect 2-5x slower "
                f"preprocess + finalize overlap.")
        if (scratch_dev is not None and scratch_path is not None
                and scratch_dev == out_dev):
            issues.append(
                f"Scratch dir and Output folder are on the same "
                f"physical drive ({scratch_path} <-> {out_path}). "
                f"The SSD->HDD transfer becomes same-drive copying; "
                f"the fast-disk savings disappear.")
        return issues

    def _on_run_all(self) -> None:
        if getattr(self, "_batch_active", False):
            messagebox.showinfo("Busy", "A batch run is already in progress.")
            return
        if not self._rows:
            messagebox.showinfo("Empty queue",
                                "Add at least one row before running.")
            return
        out = self.outdir_var.get().strip()
        if not out:
            messagebox.showinfo("Pick output folder",
                                "Set the output folder first.")
            return
        bad = [r for r in self._rows if not r.tiff_list()]
        if bad:
            messagebox.showerror(
                "Empty TIFFs",
                f"Row '{bad[0].identifier}' has no TIFF files.")
            return
        # Same-drive contention warning. When source TIFFs and the
        # output folder live on the same physical drive, the HDD has
        # to context-switch heads between reading source data for
        # the next row's preprocess and writing the previous row's
        # finalize -- net throughput collapses to 30-60 MB/s
        # (2-5x slower than separate-drive). When scratch dir is on
        # the same drive as the output folder, the SSD->HDD path
        # becomes SSD->SSD->SSD and the slow-drive savings vanish.
        # Best-effort: if anything looks wrong, warn the user and
        # let them proceed anyway.
        issues = self._check_drive_collisions(out)
        if issues:
            joined = "\n\n".join(f"- {msg}" for msg in issues)
            if not messagebox.askyesno(
                    "Same-drive contention warning",
                    "The drive layout for this batch has potential "
                    "I/O contention:\n\n"
                    f"{joined}\n\n"
                    "Mitigations:\n"
                    "- Move source TIFFs to a separate physical "
                    "drive (best).\n"
                    "- Set a Scratch dir on a fast drive (SSD/NVMe) "
                    "different from the output folder.\n"
                    "- Or just accept the slowdown.\n\n"
                    "Run anyway?"):
                return
        # Single overwrite confirmation up front. Tab 1's _on_force_rerun
        # would otherwise pop a "this will overwrite existing outputs"
        # dialog per row, which is annoying when the user has 30 of them.
        if not messagebox.askyesno(
                "Run all",
                f"Force-rerun preprocessing on {len(self._rows)} "
                f"recordings, overwriting any existing outputs in:\n"
                f"  {out}\n\nContinue?"):
            return
        self._save_queue()

        # Resolve the PipelineApp instance so the orchestrator can
        # reach sibling tabs by attribute. The toplevel widget is
        # PipelineApp itself (it inherits from tk.Tk).
        self._app = self.winfo_toplevel()

        self._batch_active = True
        # Reset the suite2p batch_size adaptive sizer so the queue
        # starts fresh: first recording uses the frame-size-aware
        # initial estimate, recordings 2..N learn from observed peak
        # RAM. Without this reset, batch_size would carry over from a
        # previous queue (potentially mis-tuned for a different
        # cohort's frame sizes).
        from ...core import utils as _core_utils
        _core_utils.reset_adaptive_sizer()
        self._batch_queue = list(self._rows)
        self._batch_results: list[dict] = []
        # In-flight scratch->HDD copies. Each daemon thread does its
        # own copytree/rmtree without blocking the next row's start.
        # _batch_finish polls this list and writes the report only
        # once every thread has joined, so user-facing paths in the
        # report are guaranteed to exist on disk.
        self._batch_finalize_threads: list[threading.Thread] = []
        self._finalize_waiting = False
        # Continuous-mirror state when scratch is on a different
        # physical drive from the output folder: one daemon thread
        # per row that polls scratch and copies new files to HDD
        # while the pipeline is still computing. See
        # ``_start_mirror_for_row`` / ``_drain_mirror_for_row``.
        self._mirror_workers: dict[str, dict] = {}
        # True iff scratch and output are on different physical
        # drives -- only set when scratch routing is enabled.
        # Toggles between continuous-mirror mode (two-drive) and
        # the row-end semantic-priority finalize (single-drive).
        self._two_drive_layout: bool = False
        self._batch_final_out_root = Path(out)
        # Resolve the optional scratch dir. When set, every recording
        # writes its intermediates to scratch (fast disk) and the full
        # tree is copied to ``_batch_final_out_root`` at row end. When
        # blank, ``_batch_out_root`` and ``_batch_final_out_root`` are
        # the same path and the row state machine behaves identically
        # to the pre-scratch implementation.
        scratch = self.scratch_dir_var.get().strip()
        self._batch_scratch_root: Optional[Path] = None
        if scratch:
            scratch_root = Path(scratch)
            try:
                same = (scratch_root.resolve()
                        == self._batch_final_out_root.resolve())
            except OSError:
                same = False
            if same:
                messagebox.showerror(
                    "Bad scratch dir",
                    "Scratch dir is the same as the output folder. "
                    "Pick a separate location on the fast drive.")
                return
            try:
                scratch_root.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                messagebox.showerror(
                    "Bad scratch dir",
                    f"Could not create scratch directory:\n{e}")
                return
            self._batch_scratch_root = scratch_root
            self._batch_out_root = scratch_root
            # Determine whether scratch and output sit on different
            # physical drives. ``os.stat(...).st_dev`` is the OS
            # device id of a path's underlying filesystem; same
            # value = same partition = same drive (covers the
            # common single-disk-per-mountpoint case on Windows +
            # Linux). When the answer is yes, we use continuous
            # mirror mode (background copy throughout the row) so
            # the transfer time overlaps with pipeline computation.
            # Otherwise the row-end semantic-priority finalize
            # runs as before -- there's no advantage to copying
            # mid-row when it just steals bandwidth from the
            # active stage.
            try:
                scratch_dev = scratch_root.resolve().stat().st_dev
                out_dev = self._batch_final_out_root.resolve().stat().st_dev
                self._two_drive_layout = (scratch_dev != out_dev)
            except OSError:
                # If we can't stat, fall back to the safer
                # single-drive behavior (row-end finalize).
                self._two_drive_layout = False
            # Sweep orphans from a previous session (app crash,
            # power loss, anti-virus killing the process mid-rmtree).
            # Anything in scratch root that isn't an identifier in
            # the current queue gets queued for janitor cleanup --
            # since this is the start of a fresh batch, prior file
            # handles from the previous process are gone and the
            # first rmtree pass almost always succeeds.
            self._sweep_scratch_orphans(scratch_root)
        else:
            self._batch_out_root = self._batch_final_out_root
            self._two_drive_layout = False
        self._abort_flag.clear()

        # Pause Tab 2's QC GIF animation for the duration of the batch.
        # The user can flip back to Tab 2 mid-batch but we don't want
        # to spin the animation loop on every recording when nobody's
        # asking for it. Snapshot the previous setting so we can restore.
        self._qc_animate_was = None
        try:
            qc_tab = self._app.qc_tab
            self._qc_animate_was = bool(qc_tab._gif_animate_var.get())
            if self._qc_animate_was:
                qc_tab._gif_animate_var.set(False)
                qc_tab._on_animate_toggle()
        except Exception:
            pass

        self.run_btn.configure(state="disabled")
        self.abort_btn.configure(state="normal")
        for row in self._rows:
            row.set_status("queued")

        self._append_log("=== Batch run starting ===")
        if self._batch_scratch_root is not None:
            mode = ("continuous mirror (two-drive)"
                    if self._two_drive_layout
                    else "row-end finalize (single-drive)")
            self._append_log(
                f"  scratch dir: {self._batch_scratch_root} "
                f"-> {self._batch_final_out_root}  [{mode}]")
        self.after(0, self._batch_next_row)

    def _on_abort(self) -> None:
        """Request a graceful stop after the current pipeline stage
        finishes.

        Why "after current stage" and not "right now"
        ---------------------------------------------
        Pipeline stages (suite2p register-only, suite2p run_plane,
        dF/F GPU compute, cluster linkage, ...) write incrementally
        to disk. Killing them mid-flight would leave half-written
        binaries that ``suite2p._find_existing_binaries`` would
        latch onto on the next run -- corrupting state and
        producing silent failures. So abort is graceful by design:

        1. Set the flag immediately.
        2. The currently-running stage finishes naturally (could be
           seconds for lowpass, minutes for detection).
        3. When that stage publishes its completion signal,
           ``_begin_stage(next_stage)`` checks the flag, marks the
           current row "aborted", and calls ``_on_row_done``.
        4. ``_on_row_done`` triggers
           ``_finalize_via_mirror_async`` which signals
           ``stop_event`` on the current row's mirror -- the mirror
           does one final drain pass (catches files written between
           stage-end and stop), then exits.
        5. ``_batch_next_row`` sees the abort flag and goes straight
           to ``_batch_finish``, which polls in-flight finalize
           threads (any previous rows' mirrors still draining) and
           joins them before writing the report.

        Background agents are NOT abandoned -- they drain whatever's
        left on scratch to HDD before exiting, so the HDD copy is
        consistent. The trade-off is that abort isn't instant: it
        can take anywhere from seconds (between-stages click) to
        the full remaining stage runtime to complete. The button
        flips to "Aborting..." on click so the user knows the
        request landed.
        """
        if self._abort_flag.is_set():
            # Second click while abort is already in progress: just
            # remind the user we're still waiting.
            self._append_log(
                "[batch] abort already requested; current stage "
                "still running -- please wait.")
            return
        self._abort_flag.set()
        # Disable + relabel so the user can see the request was
        # registered. Restored in _batch_finish.
        self.abort_btn.configure(state="disabled", text="Aborting...")
        cur_stage = getattr(self, "_current_stage", None) or "?"
        ident = (self._current_row.identifier
                 if getattr(self, "_current_row", None) is not None
                 else "?")
        self._append_log(
            f"[batch] ABORT requested.\n"
            f"  - Current stage '{cur_stage}' on row '{ident}' will "
            f"finish first (cannot kill suite2p / GPU compute "
            f"mid-flight without corrupting state).\n"
            f"  - Row '{ident}' will then be marked 'aborted' and "
            f"the rest of its stages skipped.\n"
            f"  - The mirror worker drains any unsynced files from "
            f"scratch to HDD before exiting (no data loss).\n"
            f"  - Remaining queued rows will NOT start.\n"
            f"  - Batch finish waits for every in-flight finalize "
            f"thread to join before writing the report.")
        # Mark the current row visually so the user can scan the
        # status column and see "aborting".
        try:
            self._current_row.set_status(
                f"aborting (waiting for {cur_stage} to finish)")
        except Exception:
            pass

    # ---- Row + stage state machine -------------------------------------

    def _batch_next_row(self) -> None:
        """Pop the next row off the queue and start it at preprocess.

        Runs on the Tk main thread (scheduled via ``after``).
        """
        if self._abort_flag.is_set() or not self._batch_queue:
            self._batch_finish()
            return

        # Scratch space pre-flight. Peek (don't pop yet) at the next
        # row, estimate its peak scratch footprint, and bail vs wait
        # vs proceed based on free space + what's in flight:
        #   - enough free space: proceed (normal path)
        #   - low space, finalize threads still draining or cleanup
        #     queue non-empty: wait 15 s and re-check (re-arm via
        #     ``after`` so the GUI stays responsive)
        #   - low space, nothing in flight: nothing will free up;
        #     fail the row with a clear error and advance to the next
        if (self._batch_scratch_root is not None
                and self._batch_queue):
            next_row = self._batch_queue[0]
            verdict = self._check_scratch_capacity(next_row)
            if verdict == "wait":
                self.after(self._SCRATCH_PREFLIGHT_RETRY_MS,
                           self._batch_next_row)
                return
            if verdict == "fail":
                failed = self._batch_queue.pop(0)
                self._fail_row_no_space(failed)
                self.after(0, self._batch_next_row)
                return

        self._current_row = self._batch_queue.pop(0)
        ident = self._current_row.identifier
        rec_save = self._batch_out_root / ident
        try:
            # In scratch mode, wipe any stale scratch leftovers from a
            # previous interrupted run for this identifier — the user
            # already answered the force-rerun confirmation in
            # _on_run_all and a half-written scratch dir would confuse
            # downstream caching (suite2p's _find_existing_binaries
            # would match on the partial folder).
            if self._batch_scratch_root is not None and rec_save.exists():
                shutil.rmtree(rec_save, ignore_errors=True)
            rec_save.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self._append_log(
                f"[{ident}] FAILED to create output folder: {e}")
            self._current_row.set_status("failed")
            self._batch_results.append({
                "recording_id": ident, "status": "failed",
                "plane0": "", "total_s": 0.0,
                "error": f"mkdir failed: {e}",
            })
            self.after(0, self._batch_next_row)
            return

        self._current_row_save = rec_save
        self._row_t0 = time.time()
        self._row_results = {
            "recording_id": ident,
            "status": "running",
            "plane0": "",
            "stages": {},
        }

        self._append_log("")
        tiffs = self._current_row.tiff_list()
        if len(tiffs) == 1:
            self._append_log(
                f"--- Recording '{ident}' ({Path(tiffs[0]).name}) ---")
        else:
            self._append_log(
                f"--- Recording '{ident}' "
                f"({len(tiffs)} TIFFs, trailing-index order):")
            for t in tiffs:
                self._append_log(f"      {Path(t).name}")
            self._append_log("---")
        if self._batch_scratch_root is not None:
            final_rec = self._batch_final_out_root / ident
            self._append_log(f"  scratch -> {rec_save}")
            self._append_log(f"  final   -> {final_rec}")
            # Always start the continuous mirror in scratch mode.
            # Two-drive: ``throttle_provider=None`` -> always
            # unthrottled (no HDD contention to manage).
            # Single-drive: pass a closure that returns the
            # throttle only when the active stage is preprocess
            # (the one stage that reads source TIFFs from the
            # same drive). All other stages operate on scratch
            # files + GPU, so the mirror can saturate disk
            # bandwidth without hurting them.
            if self._two_drive_layout:
                throttle_provider = None
            else:
                def throttle_provider() -> Optional[int]:
                    # ``self._current_stage`` is set on the Tk
                    # main thread and read here from the mirror
                    # daemon. Atomic single-attribute read under
                    # the GIL is safe; up to ~one pass of
                    # staleness across a stage transition is
                    # tolerable.
                    if getattr(self, "_current_stage", None) == "preprocess":
                        return SINGLE_DRIVE_FINALIZE_RATE_BYTES_PER_SEC
                    return None
            self._start_mirror_for_row(
                ident, rec_save, final_rec,
                throttle_provider=throttle_provider)
        else:
            self._append_log(f"  output -> {rec_save}")

        self._begin_stage("preprocess")

    def _apply_row_params_to_tabs(self, row: "BatchRow") -> None:
        """Push ``row.params`` into each pipeline tab BEFORE its run
        method fires.

        The orchestrator drives each tab via its native ``_on_run`` /
        ``_on_compute`` / ``_on_render`` / ``_start_run``, every one of
        which reads from the tab's own ``_params`` dict and/or Tk vars.
        Without this push, the per-row "Edit params..." dialog has no
        runtime effect -- it just round-trips through
        ``calliope_batch.json``.

        For each key the row defines, we copy it into the matching tab's
        ``_params`` dict (for PARAM_SPEC-style fields) and/or set the
        matching Tk variable (for UI-only knobs like Tab 3's GCaMP
        variant, Tab 4's cutoff slider, Tab 5's manual subset toggles,
        Tab 6's prefix/palette/threshold, and Tab 7's xcorr knobs).
        Keys the row didn't override leave the tab state untouched.
        """
        p = row.params or {}
        if not p:
            return
        app = self._app

        # Tab 1: preprocess -- 5 QC-gif knobs read from self._params
        prep = getattr(app, "preprocess_tab", None)
        if prep is not None:
            for k in ("downsample_t", "max_size_px", "playback_fps",
                      "clip_low", "clip_high"):
                if k in p:
                    prep._params[k] = p[k]

        # Tab 3: detection -- PARAM_SPEC fields go through _params; the
        # baseline + GCaMP knobs live as Tk vars and are read directly
        # in _on_run.
        det = getattr(app, "detection_tab", None)
        if det is not None:
            for entry in det.PARAM_SPEC:
                name = entry["name"]
                if name in p:
                    det._params[name] = p[name]
            # Tab 3 dF/F baseline lives under dff_baseline_mode /
            # dff_baseline_min in the batch spec (renamed to avoid
            # colliding with Tab 5's PARAM_SPEC ``baseline_mode``).
            if "dff_baseline_mode" in p:
                det.baseline_var.set(str(p["dff_baseline_mode"]))
            if "dff_baseline_min" in p:
                det.baseline_min_var.set(str(p["dff_baseline_min"]))
            if "gcamp_variant" in p:
                # Round-trip through ``_resolve_gcamp_tau`` so labels saved
                # under the pre-2026-05-26 schema (e.g. ``"GCaMP8m  (tau =
                # 0.137 s)"`` -- no leading "j", aggressive cultured-neuron
                # tau) get snapped to the current canonical label
                # (``"jGCaMP8m (tau = 0.25 s)"``) at load time. Without
                # this, the StringVar holds a string that's not in the
                # combobox values list and the user can't tell from
                # looking at the dropdown that their saved selection has
                # been silently migrated. The run-time path in
                # ``_on_run`` also calls the resolver, so this is a
                # belt-and-suspenders fix that also makes the migration
                # visible in the UI before the run starts.
                raw_label = str(p["gcamp_variant"])
                _, snapped_label, kind = det._resolve_gcamp_tau(raw_label)
                if kind != "exact":
                    print(f"[batch] migrated stale gcamp_variant "
                          f"{raw_label!r} -> {snapped_label!r}")
                det.gcamp_var.set(snapped_label)

        # Tab 4: lowpass -- PARAM_SPEC fields + the live cutoff slider.
        lp = getattr(app, "lowpass_tab", None)
        if lp is not None:
            for entry in lp.PARAM_SPEC:
                name = entry["name"]
                if name in p:
                    lp._params[name] = p[name]
            if "cutoff_hz" in p:
                try:
                    lp.cutoff_var.set(float(p["cutoff_hz"]))
                except (TypeError, ValueError):
                    pass

        # Tab 5: event detection -- PARAM_SPEC fields + the three Tk
        # vars _on_render snapshots into _params on the main thread.
        ev = getattr(app, "event_tab", None)
        if ev is not None:
            for entry in ev.PARAM_SPEC:
                name = entry["name"]
                if name in p:
                    ev._params[name] = p[name]
            if "manual_subset_enabled" in p:
                ev.manual_subset_var.set(bool(p["manual_subset_enabled"]))
            if "manual_roi_spec" in p:
                ev.manual_roi_var.set(str(p["manual_roi_spec"]))
            if "onset_source" in p:
                ev.onset_source_var.set(str(p["onset_source"]))

        # Tab 6: clustering -- no PARAM_SPEC; Tk vars + internal state.
        cl = getattr(app, "clustering_tab", None)
        if cl is not None:
            if "prefix" in p:
                cl.prefix_var.set(str(p["prefix"]))
            if "palette" in p:
                cl.palette_var.set(str(p["palette"]))
                cl._palette_name = str(p["palette"])
            if "threshold" in p:
                try:
                    t = float(p["threshold"])
                    # threshold == 0 means "auto"; leave the tab alone
                    # so its auto_choose_threshold path runs.
                    if t > 0.0:
                        cl._manual_T = t
                        try:
                            cl.threshold_scale.set(t)
                        except Exception:
                            pass
                except (TypeError, ValueError):
                    pass

        # Tab 7: xcorr -- no PARAM_SPEC; everything flows through
        # _gather_params reading Tk vars.
        xc = getattr(app, "xcorr_tab", None)
        if xc is not None:
            if "max_lag_seconds" in p:
                xc.maxlag_var.set(str(p["max_lag_seconds"]))
            if "zero_lag" in p:
                xc.zero_lag_var.set(bool(p["zero_lag"]))
            if "use_gpu" in p:
                xc.gpu_var.set(bool(p["use_gpu"]))

    def _apply_save_policy_to_detection_tab(self) -> None:
        """Translate the "What to save" toggles into Tab 3's
        post-detection archive params.

        Separate from ``_apply_row_params_to_tabs`` because that method
        early-returns when a row has no per-row overrides, whereas the
        save policy must apply to EVERY batch run. Tab 3's
        ``_archive_recording`` runs at the tail of detection in both
        scratch and non-scratch mode, defaulting to compress + delete
        shifted + delete data.bin. Left untouched it would (a) ignore
        the keep toggles, deleting artifacts the user asked to keep, and
        (b) in scratch mode double-compress against the finalize worker.

        Single owner per artifact:

        * scratch mode -- the finalize worker owns compression (gated on
          ``compress_raw_var``) and ``prune_scratch_tree`` owns the
          shifted / data.bin keep-or-drop (gated on the keep toggles).
          So Tab 3's archive is disabled entirely; it must never touch
          the scratch tree.
        * non-scratch mode -- there is no finalize worker or
          ``prune_scratch_tree`` (outputs land directly at the final
          dir), so Tab 3's archive IS the single owner: compress per
          ``compress_raw_var`` and delete per the keep toggles.

        Idempotent; safe to call on every stage transition.
        """
        det = getattr(self._app, "detection_tab", None)
        if det is None or not hasattr(det, "_params"):
            return
        if self._batch_scratch_root is not None:
            det._params["archive_post_detection"] = False
        else:
            det._params["archive_post_detection"] = True
            det._params["compress_raw_post_detection"] = \
                bool(self.compress_raw_var.get())
            det._params["delete_shifted_post_detection"] = \
                not bool(self.keep_shifted_var.get())
            det._params["delete_final_data_bin_post_detection"] = \
                not bool(self.keep_registration_var.get())

    def _begin_stage(self, stage: str) -> None:
        """Trigger ``stage`` for the current row + arm the one-shot
        completion listener.
        """
        if self._abort_flag.is_set():
            self._row_results["status"] = "aborted"
            self._on_row_done()
            return

        # Push row-level overrides into each tab so the Edit params
        # dialog actually controls the batch run. Idempotent per stage
        # (cheap dict / Tk var writes); re-applied on every transition
        # to defend against intermediate mutations.
        try:
            self._apply_row_params_to_tabs(self._current_row)
        except Exception as e:
            ident = (self._current_row.identifier
                     if getattr(self, "_current_row", None) is not None
                     else "?")
            self._append_log(
                f"[{ident}] WARN: failed to apply per-row params: {e}")

        # "What to save" archive policy -- applies to every run, not just
        # rows with per-row overrides, so it lives outside
        # _apply_row_params_to_tabs (which early-returns on empty params).
        try:
            self._apply_save_policy_to_detection_tab()
        except Exception as e:
            self._append_log(
                f"  WARN: failed to apply save policy: {e}")

        self._current_stage = stage
        self._stage_t0 = time.time()
        self._current_row.set_status(f"running {stage}")
        self._append_log(f"  [{stage}] starting")

        # Switch the notebook to the active tab so the user sees panels
        # update live. Failure to switch is non-fatal.
        try:
            tab_idx = self._STAGE_TAB_INDEX.get(stage)
            if tab_idx is not None:
                self._app._show_tab(tab_idx)
        except Exception:
            pass

        method = getattr(self, f"_stage_{stage}", None)
        if method is None:
            self._fail_stage(stage, RuntimeError(f"no handler for {stage}"))
            return
        try:
            method()
        except Exception as e:
            self._fail_stage(stage, e)

    def _arm_completion(self, subscribe_fn, expected_stage: str,
                        path_extractor=lambda payload: None) -> None:
        """Attach a one-shot AppState listener that fires ``_on_stage_done``
        the first time it's called for the current stage.

        AppState's subscribe API has no unsubscribe; we use an internal
        flag to short-circuit subsequent calls. The listener also bails
        if the orchestrator has already moved on to a different stage.
        """
        fired = [False]
        def cb(payload):
            if fired[0] or self._current_stage != expected_stage:
                return
            fired[0] = True
            try:
                hint = path_extractor(payload)
            except Exception:
                hint = None
            self.after(0, lambda: self._on_stage_done(expected_stage, hint))
        subscribe_fn(cb)

    # ---- Per-stage triggers --------------------------------------------

    def _stage_preprocess(self) -> None:
        prep = self._app.preprocess_tab
        row = self._current_row
        tiffs = [Path(t).resolve() for t in row.tiff_list()]

        # Pick a working dir + depth that's guaranteed to surface all
        # of the row's TIFFs in Tab 1's listbox:
        #   - dir = deepest common parent of every row TIFF (so all of
        #     them are reachable from one tree),
        #   - depth = max number of directory hops between dir and any
        #     TIFF (so _refresh_tiffs's recursive walk reaches them).
        # Falls back gracefully when ``commonpath`` raises (e.g. mixed
        # drive letters on Windows): in that case we use the parent of
        # the first TIFF and depth 0, which still matches single-folder
        # rows correctly.
        try:
            common = Path(os.path.commonpath([str(t) for t in tiffs]))
        except ValueError:
            common = tiffs[0].parent
        # ``commonpath`` on a single path returns that path verbatim
        # (e.g. a .tif file), and on real directories ``is_file`` may
        # not be a reliable check (file may not exist yet). Cover both
        # by treating any path that looks like a TIFF as a file.
        if common.suffix.lower() in (".tif", ".tiff") or common.is_file():
            common = common.parent
        max_depth = 0
        for t in tiffs:
            try:
                rel_parts = t.relative_to(common).parts
            except ValueError:
                rel_parts = (t.name,)
            # rel_parts includes the filename; depth counts the
            # subdirectory hops between ``common`` and the file's parent.
            max_depth = max(max_depth, len(rel_parts) - 1)

        # Mimic the user filling in Tab 1.
        prep.dir_var.set(str(common))
        prep.depth_var.set(max_depth)
        prep.data_root_var.set(str(self._batch_out_root))
        prep.identifier_var.set(row.identifier)

        # Populate the listbox via Tab 1's own refresher, then select
        # the row's TIFFs by RESOLVED PATH (listbox items are relative
        # to dir_var, so we reconstruct absolute paths and compare).
        prep._refresh_tiffs()
        listbox = prep.tiff_listbox
        listbox.selection_clear(0, "end")
        targets = {str(t.resolve()) for t in tiffs}
        for i in range(listbox.size()):
            abs_path = (common / listbox.get(i)).resolve()
            if str(abs_path) in targets:
                listbox.selection_set(i)

        self._arm_completion(
            self.state.subscribe, "preprocess",
            path_extractor=lambda r: getattr(r, "out_dir", None),
        )
        # Force-rerun bypasses the "load existing outputs" cache so the
        # batch always re-runs the recording from scratch. Skip Tab 1's
        # _on_force_rerun wrapper -- it pops a confirmation dialog per
        # row that the user already answered once in _on_run_all.
        self.after(50, lambda: prep._start_run(force=True))

    def _stage_detection(self) -> None:
        det = self._app.detection_tab
        # Tab 3 already auto-populates inputs from the AppState.result
        # publish that just fired; nothing to set on our side.
        self._arm_completion(
            self.state.subscribe_plane0, "detection",
            path_extractor=lambda p: p,
        )
        self.after(50, det._on_run)

    def _stage_lowpass(self) -> None:
        lp = self._app.lowpass_tab
        self._arm_completion(
            self.state.subscribe_lowpass_ready, "lowpass",
            path_extractor=lambda p: p,
        )
        self.after(50, lp._on_compute)

    def _stage_event_detection(self) -> None:
        ev = self._app.event_tab
        self._arm_completion(
            self.state.subscribe_event_results, "event_detection",
            path_extractor=lambda r: r.get("plane0") if isinstance(r, dict) else None,
        )
        # Tab 5's Render button -- it builds the per-ROI onset matrix,
        # detects population events, and publishes set_event_results.
        self.after(50, ev._on_render)

    def _stage_clustering(self) -> None:
        """Cluster, then export *_rois.npy.

        Tab 7 needs the per-cluster ROI lists Tab 6 writes via the
        Export button -- without them, _on_run_full bails because the
        cluster folder doesn't exist. Tab 6's _on_export is synchronous
        (no worker), so we just run it inline after the analysis
        publishes set_clusters_ready and before advancing to xcorr.
        """
        cl = self._app.clustering_tab
        fired = [False]

        def cb(plane0):
            if fired[0] or self._current_stage != "clustering":
                return
            fired[0] = True
            try:
                cl._on_export()
            except Exception as e:
                self._append_log(
                    f"  [clustering] export FAILED: {e}\n"
                    f"{traceback.format_exc()}")
                self.after(0, lambda: self._fail_stage(
                    "clustering", e))
                return
            self._append_log(
                "  [clustering] cluster ROI lists exported")
            self.after(0, lambda: self._on_stage_done(
                "clustering", plane0))

        self.state.subscribe_clusters_ready(cb)
        self.after(50, cl._on_run)

    def _stage_crosscorrelation(self) -> None:
        """Run the full-recording xcorr first, then the per-event xcorr.

        Both Tab 7 entry points push to the same set_xcorr_ready
        channel, so we use a two-step internal counter: first publish
        triggers _on_run_per_event; second publish advances the batch.

        (Previously this had a cluster-fragmentation guard that
        skipped per-event xcorr when ``n_clusters > 20`` AND >70% of
        clusters had <5 ROIs. The guard was a workaround for a bug
        in ``auto_choose_threshold`` which counted dendrogram colour
        groups instead of real ``fcluster`` labels -- the auto-pick
        could land at "5 clusters visible" while fcluster returned
        30+, producing the over-fragmented input the guard was
        catching. Bug fixed 2026-05-11 in
        ``clustering_cmap.auto_choose_threshold``; guard removed.)
        """
        xc = self._app.xcorr_tab
        self._xcorr_step = "full"
        fired = [False, False]   # one slot per step, shared via closure

        def cb(payload):
            # First publish: full done -> kick off per-event.
            if not fired[0] and self._current_stage == "crosscorrelation":
                fired[0] = True
                self._append_log("  [crosscorrelation] full done; "
                                 "starting per-event")
                self._xcorr_step = "per_event"
                self.after(200, xc._on_run_per_event)
                return
            # Second publish: per-event done -> advance to spatial.
            if not fired[1] and self._current_stage == "crosscorrelation":
                fired[1] = True
                hint = payload
                self.after(0, lambda: self._on_stage_done(
                    "crosscorrelation", hint))

        self.state.subscribe_xcorr_ready(cb)
        self.after(50, xc._on_run_full)

    def _stage_spatial_propagation(self) -> None:
        """Tab 8 has no Run button -- it renders reactively from the
        ``set_event_results`` publish that already fired during the
        event-detection stage. Switching the notebook gives the user
        the live render; we wait a beat then advance.
        """
        # Advance after a short delay to let the renderer paint.
        self.after(750,
                   lambda: self._on_stage_done("spatial_propagation", None))

    # ---- Stage / row completion ---------------------------------------

    def _on_stage_done(self, stage: str, path_hint) -> None:
        if self._current_stage != stage:
            return
        elapsed = time.time() - self._stage_t0
        self._row_results["stages"][stage] = {
            "status": "ok",
            "duration_s": elapsed,
            "error": None,
        }
        if stage == "detection" and path_hint is not None:
            self._row_results["plane0"] = str(path_hint)
        self._append_log(f"  [{stage}] done ({elapsed:.1f}s)")

        # If the "Save figures during batch" toggle is on, ask the
        # tab that just finished to dump its figures + refresh the
        # per-recording manifest. Called with ``quiet=True`` so a
        # 7-stage batch doesn't bury the user under 7 popups per row.
        if self.save_figures_var.get():
            try:
                self._export_stage_figures(stage)
            except Exception as e:
                self._append_log(
                    f"  [{stage}] figure export failed: {e}")

        idx = self.STAGES.index(stage) + 1
        if idx >= len(self.STAGES):
            self._on_row_done()
        else:
            self._begin_stage(self.STAGES[idx])

    def _export_stage_figures(self, stage: str) -> None:
        """Dispatch the per-tab figure export for the stage that just
        finished. Keeps the batch tab decoupled from each tab's
        internals -- we just call the handler the per-tab button uses,
        with ``quiet=True`` to suppress messageboxes.
        """
        stage_to_tab = {
            "detection": "detection_tab",
            "lowpass": "lowpass_tab",
            "event_detection": "event_tab",
            "clustering": "clustering_tab",
            "crosscorrelation": "xcorr_tab",
            "spatial_propagation": "spatial_tab",
        }
        attr = stage_to_tab.get(stage)
        if attr is None:
            return  # preprocess writes its qc.gif via a separate path
        tab = getattr(self._app, attr, None)
        if tab is None:
            return
        export = getattr(tab, "_on_export_figures", None)
        if export is None:
            return
        export(quiet=True)
        self._append_log(f"  [{stage}] figures exported")

    def _fail_stage(self, stage: str, exc: BaseException) -> None:
        elapsed = time.time() - self._stage_t0
        self._row_results["stages"][stage] = {
            "status": "failed",
            "duration_s": elapsed,
            "error": str(exc),
        }
        self._row_results["status"] = "failed"
        self._append_log(
            f"  [{stage}] FAILED: {exc}\n{traceback.format_exc()}")
        self._on_row_done()

    def _on_row_done(self) -> None:
        elapsed = time.time() - self._row_t0
        self._row_results["total_s"] = elapsed
        if self._row_results["status"] == "running":
            self._row_results["status"] = "ok"

        # Finalize: copy this recording's tree from scratch to the real
        # output folder, then delete the scratch copy. We do this on
        # success AND failure paths so the user gets whatever partial
        # outputs the pipeline produced before a crash, and scratch
        # never accumulates abandoned runs across the queue.
        #
        # The copy runs in a daemon thread so the next recording can
        # start its preprocess (and SSD writes) immediately while the
        # previous row's bulk SSD->HDD transfer drains in the
        # background. The plane0 path rewrite happens synchronously
        # against the *projected* final location -- the file may not
        # exist on HDD yet when we record the path here, but
        # _batch_finish joins every in-flight thread before writing
        # the report, so by the time anything reads _batch_results
        # the path is real on disk.
        status = self._row_results["status"]
        # Track whether this row spawned a background finalize so the
        # status label can show "(transferring)" -> "(done)" instead of
        # the bare status. In non-scratch mode the row's status stays
        # exactly the same as before.
        scratch_in_flight = False
        if self._batch_scratch_root is not None:
            ident = self._current_row.identifier
            scratch_rec = self._batch_scratch_root / ident
            final_rec = self._batch_final_out_root / ident
            plane0 = self._row_results.get("plane0")
            if plane0:
                try:
                    rel = Path(plane0).relative_to(scratch_rec)
                    self._row_results["plane0"] = str(final_rec / rel)
                except ValueError:
                    pass
            # Pass the row reference so the async worker can update the
            # status label when the copy finishes -- by that time
            # ``self._current_row`` will already be pointing at a
            # different row.
            #
            # Unified finalize: both two-drive and single-drive layouts
            # use the continuous mirror started in ``_batch_next_row``.
            # The mirror's throttle setting differs per layout (see
            # ``_start_mirror_for_row``); the finalize logic is
            # identical: signal stop, drain, repoint, rmtree.
            self._finalize_via_mirror_async(
                ident, scratch_rec, final_rec,
                row=self._current_row, base_status=status)
            scratch_in_flight = True

        self._batch_results.append(self._row_results)
        if scratch_in_flight:
            self._current_row.set_status(f"{status} (transferring)")
        else:
            self._current_row.set_status(status)
        self._append_log(
            f"--- '{self._current_row.identifier}' "
            f"{status} in {elapsed:.1f}s ---")
        self.after(0, self._batch_next_row)

    def _start_mirror_for_row(self, ident: str,
                              scratch_rec: Path,
                              final_rec: Path,
                              *,
                              throttle_provider=None) -> None:
        """Spawn a daemon thread that continuously mirrors
        ``scratch_rec`` -> ``final_rec`` while the row's pipeline
        runs.

        Fires whenever scratch routing is enabled, in both
        single-drive and two-drive layouts. The throttle is now
        stage-aware via ``throttle_provider``: a zero-arg callable
        the worker calls at the start of each sync pass to get
        the current ``int|None`` throttle rate.

        Why a callable instead of a fixed rate
        --------------------------------------
        On a single-drive setup, the mirror only contends with
        the active pipeline stage when *that stage* is reading
        from the same physical drive. The main offender is the
        preprocess stage reading source TIFFs from HDD; the other
        stages (detection / lowpass / events / clustering / xcorr
        / spatial) operate on scratch files + GPU and don't pull
        big bandwidth off the output drive. So during preprocess
        we throttle the mirror; during other stages we let it
        saturate the disk (user wanted "100% disk usage when
        there's no real reader to share with").

        Two-drive layout passes ``throttle_provider=None``
        (unthrottled always). Single-drive layout passes a
        closure that returns ``SINGLE_DRIVE_FINALIZE_RATE_BYTES_PER_SEC``
        only when ``self._current_stage == "preprocess"``, ``None``
        otherwise. See ``_batch_next_row`` for the wiring.

        Thread safety: ``self._current_stage`` is set on the Tk
        main thread (in ``_begin_stage``) and read by the mirror
        daemon. Python's GIL makes single-attribute reads atomic
        and string references are immutable, so the worst case is
        one cycle of staleness across a stage transition -- the
        mirror might throttle for one extra pass after preprocess
        ends, or run unthrottled for one extra pass before
        preprocess starts. Both windows are <2 s; not worth a
        lock.

        The worker polls every 2 s when nothing's changed, every
        0.2 s after a copy lands (to drain bursts quickly). Skips
        files whose mtime is < 1 s old (in-flight writes) and
        anything ``_is_scratch_mirror_skippable`` says is an
        intermediate the row-end prune would have dropped.

        Tracked in ``self._mirror_workers[ident]`` as a dict of
        ``{thread, stop, synced, seen, inode_to_dst,
        throttle_provider}`` so ``_finalize_via_mirror_async``
        can signal stop + wait for the final pass to complete.
        """
        if self._batch_scratch_root is None:
            return
        stop_evt = threading.Event()
        synced_evt = threading.Event()
        seen: dict[Path, tuple[int, float]] = {}
        inode_to_dst: dict[tuple[int, int], str] = {}
        # Persist final_rec.mkdir up front so the mirror doesn't
        # race the first file's parent creation.
        final_rec.mkdir(parents=True, exist_ok=True)

        def _emit(line: str) -> None:
            self.after(0, lambda l=line: self._append_log(l))

        def _resolve_throttle() -> Optional[int]:
            """Best-effort throttle lookup; the provider may raise
            during teardown, in which case fall back to no
            throttle."""
            if throttle_provider is None:
                return None
            try:
                return throttle_provider()
            except Exception:
                return None

        def _worker() -> None:
            # Mode message describes the throttle policy at start.
            # The actual rate per pass is resolved at runtime via
            # ``_resolve_throttle``.
            if throttle_provider is None:
                mode_msg = "unthrottled (two-drive)"
            else:
                mode_msg = (f"throttled to "
                            f"{SINGLE_DRIVE_FINALIZE_RATE_BYTES_PER_SEC / 1e6:.0f} "
                            f"MB/s during preprocess only "
                            f"(single-drive)")
            _emit(f"  [batch] [{ident}] mirror: started, {mode_msg}")
            n_passes = 0
            while True:
                current_throttle = _resolve_throttle()
                try:
                    changed = mirror_sync_pass(
                        scratch_rec, final_rec, seen, inode_to_dst,
                        throttle_bytes_per_sec=current_throttle,
                        keep_registration=self.keep_registration_var.get(),
                        keep_shifted=self.keep_shifted_var.get())
                except Exception as e:
                    _emit(f"  [batch] [{ident}] mirror: pass failed: "
                          f"{e} -- continuing")
                    changed = False
                n_passes += 1
                if stop_evt.is_set():
                    break
                # Brief sleep when caught up so the worker doesn't
                # spin; shorter sleep right after a busy pass to
                # drain bursts of new files quickly.
                time.sleep(0.2 if changed else 2.0)
            # Final sync pass after stop signal, with the stability
            # window dropped to zero so files written milliseconds
            # before stop_evt still get caught. Use the current
            # throttle (which is likely None by now because the
            # row's pipeline is done -- preprocess only fires
            # mid-row).
            try:
                mirror_sync_pass(
                    scratch_rec, final_rec, seen, inode_to_dst,
                    stability_window_s=0.0,
                    throttle_bytes_per_sec=_resolve_throttle(),
                    keep_registration=self.keep_registration_var.get(),
                    keep_shifted=self.keep_shifted_var.get())
            except Exception as e:
                _emit(f"  [batch] [{ident}] mirror: final pass "
                      f"failed: {e}")
            _emit(f"  [batch] [{ident}] mirror: drained after "
                  f"{n_passes} passes, {len(seen)} files at HDD")
            synced_evt.set()

        t = threading.Thread(
            target=_worker,
            name=f"calliope-mirror-{ident}",
            daemon=True)
        self._mirror_workers[ident] = {
            "thread": t,
            "stop": stop_evt,
            "synced": synced_evt,
            "seen": seen,
            "inode_to_dst": inode_to_dst,
            "throttle_provider": throttle_provider,
        }
        t.start()
        self._batch_finalize_threads.append(t)

    def _finalize_via_mirror_async(self, ident: str,
                                   scratch_rec: Path,
                                   final_rec: Path,
                                   *,
                                   row=None,
                                   base_status: str = "") -> None:
        """Two-drive finalize: stop the row's continuous mirror,
        wait for it to drain, then repoint AppState + rmtree
        scratch. Runs in a daemon thread so the orchestrator can
        advance to the next row immediately.

        Most of the SSD->HDD copy already happened during pipeline
        computation -- the drain is typically <5 s of catching the
        last few stage-output writes.
        """
        worker = self._mirror_workers.pop(ident, None)
        if worker is None:
            # Defensive: mirror should always be running in scratch
            # mode (started in ``_batch_next_row``). If somehow it's
            # not, log + bulk-rmtree without drain so we don't leak
            # scratch -- the user loses any unwritten outputs but
            # gets a clean state.
            def _emit_orphan(line: str) -> None:
                self.after(0, lambda l=line: self._append_log(l))
            _emit_orphan(
                f"  [batch] [{ident}] finalize: mirror state "
                f"missing; rmtreeing scratch without drain")
            shutil.rmtree(str(scratch_rec), ignore_errors=True)
            if row is not None:
                self.after(
                    0, lambda: row.set_status(
                        f"{base_status} (transfer failed)"
                        if base_status else "transfer failed"))
            return

        def _emit(line: str) -> None:
            self.after(0, lambda l=line: self._append_log(l))

        def _set_row_status(label: str) -> None:
            if row is None:
                return
            self.after(0, lambda: row.set_status(label))

        def _finalize_worker() -> None:
            # Phase 1: signal the mirror to stop + drain.
            _emit(f"  [batch] [{ident}] finalize: signaling mirror "
                  f"to drain")
            t0 = time.time()
            worker["stop"].set()
            # Wait up to 30 min for the mirror's final pass to
            # finish. On a contended HDD (next row's preprocess
            # already writing to the same disk) the drain of ~20-30
            # GB of registered binaries can take 5-10 min; the
            # original 5-min timeout fired mid-copy and races with
            # the rmtree below. The synchronous sweep that follows
            # this wait is the actual safety net -- this timeout is
            # just an upper bound to keep stuck mirrors from
            # blocking the queue forever.
            if not worker["synced"].wait(timeout=1800.0):
                _emit(f"  [batch] [{ident}] finalize: mirror drain "
                      f"timed out after 30 min; sync sweep will "
                      f"catch remaining files")
            # Phase 2: prune (drops sparsery_pass etc. -- mirror
            # already skipped them, this catches anything that
            # might have slipped through e.g. data_raw.bin written
            # late). Just to be safe.
            n_pruned, bytes_pruned = prune_scratch_tree(
                scratch_rec,
                keep_registration=self.keep_registration_var.get(),
                keep_shifted=self.keep_shifted_var.get(),
                final_rec=final_rec)
            prune_msg = ""
            if n_pruned > 0:
                prune_msg = (f" (pruned {n_pruned} skipped path"
                             f"{'' if n_pruned == 1 else 's'}, "
                             f"{bytes_pruned / 1e9:.2f} GB)")
            # Phase 3: SYNCHRONOUS final sweep. Walks scratch one
            # more time on our own thread (no daemon, no time
            # bound) and copies any non-skip file that isn't already
            # at HDD. If the mirror's drain finished cleanly this is
            # a no-op walk; if it timed out or raced with rmtree,
            # this is the safety net that copies whatever it left
            # behind. Either way: by the time this returns, every
            # output file is at HDD and rmtree below is safe.
            n_swept, bytes_swept = synchronous_final_sweep(
                scratch_rec, final_rec, worker["inode_to_dst"],
                keep_registration=self.keep_registration_var.get(),
                keep_shifted=self.keep_shifted_var.get())
            sweep_msg = ""
            if n_swept > 0:
                sweep_msg = (f" (sync sweep recovered {n_swept} file"
                             f"{'' if n_swept == 1 else 's'}, "
                             f"{bytes_swept / 1e9:.2f} GB the mirror "
                             f"missed)")

            # Phase 3b: optional raw-TIFF archive. Off by default
            # (matches current behavior -- nothing is compressed). When
            # the user ticks "Archive raw TIFF (compressed)" we write
            # the Zstd-compressed raw into the final output dir. We pass
            # delete_shifted=False / delete_final_bin=False: the shifted
            # TIFF + data.bin keep/drop decision is owned entirely by
            # prune_scratch_tree above (driven by the keep toggles), so
            # the archive step never deletes anything itself.
            archive_msg = ""
            if self.compress_raw_var.get():
                try:
                    # raw_paths=None -> the archive reads the
                    # ``_calliope_raw_paths.json`` sidecar that
                    # ``run_preprocess`` wrote into the recording folder
                    # (``final_rec``), falling back to scanning for raw
                    # TIFFs there. Compression lands the Zstd archive in
                    # ``final_rec`` next to the outputs.
                    summary = archive_recording_post_detection(
                        rec_root=final_rec,
                        compress_raw=True,
                        delete_shifted=False,
                        delete_final_bin=False,
                        progress_cb=_emit,
                    )
                    n_comp = summary.get("compressed", 0)
                    if n_comp > 0:
                        archive_msg = (
                            f" (archived {n_comp} raw TIFF"
                            f"{'' if n_comp == 1 else 's'} compressed)")
                except Exception as e:  # never let archive abort finalize
                    _emit(f"  [batch] [{ident}] raw archive failed: {e}")

            # Phase 4: marshal repoint to Tk main thread + block.
            repoint_done = threading.Event()

            def _repoint_and_signal() -> None:
                try:
                    self._repoint_pipeline_after_copy(
                        scratch_rec, final_rec)
                finally:
                    repoint_done.set()

            self.after(0, _repoint_and_signal)
            if not repoint_done.wait(timeout=30.0):
                _emit(f"  [batch] [{ident}] finalize: repoint "
                      f"timed out; proceeding with rmtree")
            # Phase 5: bulk rmtree scratch. Sync sweep above
            # guarantees every output file is already at HDD, so
            # this is safe regardless of whether the mirror drained
            # cleanly or timed out.
            #
            # First attempt: standard rmtree. The repoint hooks
            # above already nulled known long-lived memmap caches
            # (Tab 3 popout, Tab 6, Tab 7) so the common case is
            # that ``gc.collect`` releases the file handles and
            # rmtree succeeds on the first pass. Residual failures
            # come from transient locks (antivirus scanning, an
            # Explorer window in scratch, a popout reopened between
            # the drop and the rmtree). One retry after a 2-s pause
            # covers those. Anything still locked after that gets
            # handed off to the long-lived scratch janitor (see
            # ``_scratch_janitor_loop``) which retries every ~30 s
            # until the handles release. No manual cleanup is ever
            # needed.
            gc.collect()
            shutil.rmtree(str(scratch_rec), ignore_errors=True)
            if scratch_rec.exists():
                time.sleep(2.0)
                gc.collect()
                shutil.rmtree(str(scratch_rec), ignore_errors=True)
            deferred_msg = ""
            handed_to_janitor = False
            if scratch_rec.exists():
                n_left, bytes_left = measure_tree(scratch_rec)
                if n_left > 0:
                    self._enqueue_scratch_cleanup(
                        scratch_rec, row=row, base_status=base_status)
                    handed_to_janitor = True
                    deferred_msg = (
                        f" [deferred: {n_left} file"
                        f"{'' if n_left == 1 else 's'} "
                        f"({bytes_left / 1e9:.2f} GB) still locked; "
                        f"janitor will retry every "
                        f"{int(self._CLEANUP_RETRY_INTERVAL_S)}s]")
            n_synced = len(worker["seen"]) + n_swept
            _emit(f"  [batch] [{ident}] finalize: mirror drained "
                  f"+ repointed + scratch cleared "
                  f"({n_synced} files on HDD, "
                  f"{time.time() - t0:.1f}s){prune_msg}{sweep_msg}"
                  f"{archive_msg}{deferred_msg}")
            if handed_to_janitor:
                _set_row_status(
                    f"{base_status} (cleanup pending)"
                    if base_status else "cleanup pending")
            else:
                _set_row_status(
                    f"{base_status} (done)" if base_status else "done")

        t = threading.Thread(
            target=_finalize_worker,
            name=f"calliope-mirror-finalize-{ident}",
            daemon=True)
        t.start()
        self._batch_finalize_threads.append(t)

    def _repoint_pipeline_after_copy(self, scratch: Path,
                                     final: Path) -> None:
        """Swap any in-memory plane0 references from ``scratch`` to
        the corresponding path under ``final``.

        Tk main thread only. Called between the SSD->HDD copy and
        the bulk scratch rmtree so the GUI's reactive reload paths
        (Tab 4 reading dF/F memmaps from ``plane0``, Tab 6 reloading
        cluster files, etc.) never end up pointed at a path we're
        about to delete.

        We bypass ``AppState.set_plane0`` / ``set_lowpass_ready``
        and write the attributes directly -- those setters fire
        every subscribed listener, which on the active row would
        re-trigger Tab 4's lowpass recompute / Tab 5's render. A
        path rewrite shouldn't have side effects, so we just swap
        the values in place.

        Tab-local ``_plane0`` / ``_final_plane0`` / ``_plane0_path``
        caches get the same direct rewrite. We touch every tab the
        pipeline_gui exposes -- a stale cache on any one of them
        would survive the AppState swap.
        """
        def _rewrite(cur):
            """Return the HDD path under ``final`` that mirrors
            ``cur`` under ``scratch`` -- or ``None`` if ``cur``
            isn't under scratch (so the caller can leave it alone).
            """
            if cur is None:
                return None
            try:
                cur_p = Path(cur)
                if cur_p == scratch:
                    return final
                rel = cur_p.relative_to(scratch)
                return final / rel
            except (TypeError, ValueError, OSError):
                return None

        # AppState channels. Direct attribute write avoids re-firing
        # listeners (which would, e.g., recompute Tab 4's lowpass
        # from a path that's about to disappear).
        state = self.state
        for attr in ("plane0", "lowpass_plane0"):
            new_p = _rewrite(getattr(state, attr, None))
            if new_p is not None:
                setattr(state, attr, new_p)

        # ``AppState.event_results`` is a dict that carries plane0
        # inside it. If Tab 5 published with the scratch path,
        # rewrite the value in place so Tab 8 / cluster popout /
        # anyone else reading ``state.event_results["plane0"]`` gets
        # the HDD twin.
        evr = getattr(state, "event_results", None)
        if isinstance(evr, dict):
            new_p = _rewrite(evr.get("plane0"))
            if new_p is not None:
                evr["plane0"] = new_p

        # ``AppState.result`` is the PreprocessResult dataclass.
        # Phase 1 (``_repoint_preprocess_result``) may have already
        # swapped the four file paths during the row; this row-end
        # pass also rewrites the ``out_dir`` field (left at scratch
        # during the row because Tabs 4-8 were still writing there).
        # Idempotent: if Phase 1 already ran, the four file paths
        # are already at HDD and ``_rewrite`` returns ``None`` for
        # them (no-op).
        pre_result = getattr(state, "result", None)
        if pre_result is not None:
            for fld in ("out_dir", "shifted_tiff", "qc_gif",
                        "mean_image_path"):
                new_p = _rewrite(getattr(pre_result, fld, None))
                if new_p is not None:
                    try:
                        setattr(pre_result, fld, new_p)
                    except (AttributeError, TypeError):
                        continue

        # Tab-local caches. The pipeline_gui exposes each tab as a
        # named attribute on the toplevel; iterate the standard list
        # and try every field name we know stashes a plane0.
        app = getattr(self, "_app", None) or self.winfo_toplevel()
        tab_attrs = ("preprocess_tab", "qc_tab", "detection_tab",
                     "lowpass_tab", "event_tab", "clustering_tab",
                     "xcorr_tab", "spatial_tab")
        plane0_fields = ("_plane0", "_final_plane0", "_plane0_path",
                         "_current_plane0", "plane0")
        for tab_attr in tab_attrs:
            tab = getattr(app, tab_attr, None)
            if tab is None:
                continue
            for fld in plane0_fields:
                new_p = _rewrite(getattr(tab, fld, None))
                if new_p is not None:
                    try:
                        setattr(tab, fld, new_p)
                    except (AttributeError, TypeError):
                        # Property without a setter, or read-only;
                        # leave it and move on.
                        continue

        # Dict-stored plane0 caches. Some tabs stash the recording's
        # plane0 path INSIDE a dict-valued attribute (e.g. Tab 3's
        # ``_panel_cache["plane0"]`` carries the path the curation
        # popout reads ``iscell.npy`` / ``stat.npy`` from). The
        # attribute-walking loop above can't reach inside those
        # dicts, so we ask each tab to repoint its own structured
        # caches via an opt-in ``_repoint_after_copy`` method. Tabs
        # that don't implement the hook are unaffected.
        for tab_attr in tab_attrs:
            tab = getattr(app, tab_attr, None)
            if tab is None:
                continue
            hook = getattr(tab, "_repoint_after_copy", None)
            if callable(hook):
                try:
                    hook(scratch, final)
                except Exception:
                    continue

        # Memmap-handle release. Path rewrites above don't close
        # any open ``np.memmap`` -- so tabs that cache a memmap as
        # instance state (Tab 6 ``_dff``, Tab 7 ``_dff_cache``, the
        # Tab 3 curation popout's ``_dff``, the Tab 6 cluster
        # popout's ``_dff``) implement their own ``_repoint_after_copy``
        # hook that nulls the cache when its backing file lives
        # under ``scratch``. The hook ran in the loop above; the
        # ``gc.collect`` here forces the released ``np.memmap``
        # objects to drop their Windows file handles before the
        # caller's bulk ``rmtree`` fires.
        #
        # Why this matters: without the explicit drop, the LAST row
        # of a batch + any open popout would keep the scratch dF/F
        # memmap locked indefinitely (no next-row publish to fire
        # ``_set_plane0``, no orphan sweep until the next batch
        # starts). The janitor would spin every 30 s and never
        # succeed until app exit.
        gc.collect()

    def _batch_finish(self) -> None:
        # Wait phase: if background scratch->HDD finalizes are still
        # running, poll every 500 ms until they all join. Without this
        # wait the report would reference paths that the worker hasn't
        # written yet (the report itself is fine — it goes to a
        # non-scratch HDD location — but the plane0 paths inside it
        # would point at not-yet-existing files).
        pending = [t for t in self._batch_finalize_threads
                   if t.is_alive()]
        if pending:
            if not self._finalize_waiting:
                verb = ("aborted" if self._abort_flag.is_set()
                        else "done")
                self._append_log(
                    f"=== Batch processing {verb}; waiting for "
                    f"{len(pending)} background scratch -> HDD "
                    f"copies to finish... ===")
                self._finalize_waiting = True
            self.after(500, self._batch_finish)
            return
        self._finalize_waiting = False

        self._batch_active = False
        self.run_btn.configure(state="normal")
        # Restore the abort button's label + (disabled) state ready
        # for the next batch. _on_abort flipped it to "Aborting..."
        # while the request was in flight.
        self.abort_btn.configure(state="disabled", text="Abort")
        if self._abort_flag.is_set():
            self._append_log("=== Batch run aborted ===")
        else:
            self._append_log("=== Batch run done ===")
        # Report goes to the user-facing output folder, not scratch —
        # scratch may already be partially deleted by per-row finalize.
        try:
            self._write_report(
                self._batch_final_out_root, self._batch_results)
        except Exception as e:
            self._append_log(f"[batch] report write failed: {e}")
        # End-of-batch scratch verification. Each row's finalize
        # already rmtree'd its own scratch_rec; we walk the whole
        # scratch root here as a belt-and-braces check that nothing
        # leaked through (Windows file-lock survivors, finalize crash
        # mid-rmtree, downstream tabs still holding memmaps on a
        # just-finished recording, etc.). Cases:
        #   - scratch root empty -> rmdir the root, log "clean"
        #   - leftovers exist    -> hand them to the long-lived
        #                           scratch janitor (started in
        #                           __init__, retries every 30 s
        #                           until handles release). The user
        #                           sees a one-line status report;
        #                           no manual intervention required.
        if self._batch_scratch_root is not None:
            scratch_root = self._batch_scratch_root
            leftover_subdirs: list[Path] = []
            try:
                for entry in scratch_root.iterdir():
                    leftover_subdirs.append(entry)
            except OSError as e:
                self._append_log(
                    f"[batch] scratch root unreadable at end of "
                    f"batch ({scratch_root}): {e}")
                leftover_subdirs = []

            if not leftover_subdirs:
                # Empty scratch root -- rmdir the per-batch root we
                # created so we don't leave a dangling empty folder.
                try:
                    scratch_root.rmdir()
                    self._append_log(
                        f"[batch] scratch dir clean -- removed "
                        f"{scratch_root}")
                except OSError as e:
                    # Empty but rmdir failed (probably the user's
                    # configured scratch root, not a per-batch
                    # subfolder we'd be allowed to remove). Fine.
                    self._append_log(
                        f"[batch] scratch dir empty; rmdir skipped: "
                        f"{e}")
            else:
                total_n = 0
                total_b = 0
                for sub in leftover_subdirs:
                    n, b = measure_tree(sub)
                    total_n += n
                    total_b += b
                    self._enqueue_scratch_cleanup(sub)
                names = ", ".join(p.name for p in leftover_subdirs[:5])
                if len(leftover_subdirs) > 5:
                    names += f", +{len(leftover_subdirs) - 5} more"
                self._append_log(
                    f"[batch] {len(leftover_subdirs)} "
                    f"subdir{'' if len(leftover_subdirs) == 1 else 's'} "
                    f"still locked at batch end "
                    f"({total_n} files, {total_b / 1e9:.2f} GB); "
                    f"janitor will retry every "
                    f"{int(self._CLEANUP_RETRY_INTERVAL_S)}s until "
                    f"handles release. No action needed.\n"
                    f"  pending: {names}")
        # Restore the user's Tab 2 animation preference.
        if self._qc_animate_was:
            try:
                qc_tab = self._app.qc_tab
                qc_tab._gif_animate_var.set(True)
                qc_tab._on_animate_toggle()
            except Exception:
                pass
        # Return the user to the batch tab so they see the summary.
        try:
            self._app._show_tab(0)
        except Exception:
            pass

    @staticmethod
    def _write_report(out_root: Path, results: list[dict]) -> None:
        """Flatten the per-row stage status dicts into a CSV."""
        if not results:
            return
        report_path = out_root / BATCH_REPORT_NAME
        all_stages: list[str] = []
        for r in results:
            for st in (r.get("stages") or {}):
                if st not in all_stages:
                    all_stages.append(st)
        header = ["recording_id", "status", "plane0", "total_s", "error"]
        for st in all_stages:
            header.append(f"{st}_status")
            header.append(f"{st}_duration_s")
        with report_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in results:
                row = [
                    r.get("recording_id", ""),
                    r.get("status", ""),
                    r.get("plane0", ""),
                    f"{r.get('total_s', 0.0):.2f}",
                    r.get("error", ""),
                ]
                stages = r.get("stages") or {}
                for st in all_stages:
                    info = stages.get(st) or {}
                    row.append(info.get("status", ""))
                    row.append(f"{info.get('duration_s', 0.0):.2f}")
                w.writerow(row)

    # -- Console handlers ------------------------------------------------

    def _on_log_payload(self, payload) -> None:
        self._append_log(str(payload))

    def _on_status_payload(self, payload) -> None:
        row, status = payload
        if row in self._rows:
            row.set_status(str(status))

    def _on_batch_done(self, _payload) -> None:
        self.run_btn.configure(state="normal")
        self.abort_btn.configure(state="disabled")
        self._append_log("[batch] all rows processed.")

    def _append_log(self, text: str) -> None:
        if not text:
            return
        self.log.configure(state="normal")
        self.log.insert("end", text.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _log(self, msg: str) -> None:
        self._log_queue.put(("log", msg))
