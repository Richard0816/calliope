# Tab 0 — Batch runner

Drives the full **Tabs 1 + 3-8** pipeline over a list of recordings without
the user having to click each tab one recording at a time.

---

## What the user sees

- **Resizable layout.** The recordings list and the run log each have their own draggable grip below them — drag down to grow either panel; the scrollable tab body absorbs the extra height. Default heights: recordings ~260 px, run log ~220 px.
- **Top bar.** Working dir + recursion **Depth** + **Scan**, output folder +
  **Reload queue**, optional **Scratch dir (SSD)** for fast intermediate I/O,
  **Apply settings to selected**, **Select all**, **+ Add row**,
  **Run all** / **Abort**.
  **Apply settings to selected** opens the Advanced params dialog and
  overwrites the params of the **checked** rows only (same selection the
  **Merge selected** button uses); **Select all** ticks/unticks every
  row's checkbox in one click, so the old "apply to every row" behaviour
  is just Select all → Apply settings to selected.
  **Scan** walks the working dir up to **Depth** levels deep and adds **one
  row per TIFF file** (NOT one row per folder — multiple TIFFs in the same
  folder each get their own row). Identifier defaults to the file stem; if
  two TIFFs share a stem under different folders, the parent folder name
  gets suffixed (`<stem>__<parent>`) so the rows don't collide on
  `<output>/<ident>/`. If the TIFFs are actually a multi-file recording,
  select the relevant rows and click **Merge selected** afterwards.
- **Rows** — a scrollable list. Each row is one recording: identifier, the
  TIFF list, a Tab 1-style **Browse...** picker, an **Edit params...** button
  (per-row override), a status label, and an **x** button to remove.
- **Run log** — live console below the rows, fed by a `QueueWriter` (same
  pattern Tabs 1 / 3 use). Captures both `stdout` and `stderr` from the
  worker thread so suite2p's logger output, GUI prints, and stage progress
  callbacks all stream into the same panel.

The TIFF picker dialog mirrors Tab 1 exactly: working-dir entry + depth
spinner + Refresh button + multi-select Listbox. Files already on the row
are pre-selected. Confirm replaces the row's TIFF list; Cancel leaves it
untouched.

---

## How a row gets processed

The Run-all worker iterates rows in queue order. For each row:

1. Snapshot effective params: start from `BatchTab.default_params`, layer the
   row's `params` dict on top.
2. Build the per-recording save folder at `<output>/<identifier>/` and
   resolve the TIFF list (raw paths from the row, no staging — Tab 1's
   `preprocess_tiff_group` accepts files from arbitrary parents and
   concatenates them in trailing-index order).
3. Call `core.batch_pipeline.run_recording(src_tiffs, save_folder, params,
   ...)`. The orchestrator chains the per-stage headless entry points
   (`run_preprocess` lives in `core/preprocessing.py`, `run_clustering`
   in `core/clustering.py`, `run_crosscorrelation` in
   `core/crosscorrelation.py`, `render_spatial_event_figures` in
   `core/spatial.py`; the three stages with no math counterpart live in
   their own `*_run.py` modules):

   ```
   preprocessing.run_preprocess -> detection_run -> lowpass_run
       -> event_detection_run -> clustering.run_clustering
       -> crosscorrelation.run_crosscorrelation
       -> spatial.render_spatial_event_figures
   ```

   Each stage is wrapped in its own try/except. If a non-detection stage
   fails, the recording is marked **partial** and the worker moves on.
   Detection failure is fatal for that row only; the queue continues.

   The detection stage produces the same per-recording artifacts as the
   GUI: the `<plane0>/meta.json` provenance sidecar (with the
   `gcamp_variant` label stamped for parity) and the z-drift QC outputs
   (`zdrift_qc.png` / `zdrift_pc1.csv` plus the `qc` fields in
   `meta.json`) are written headlessly on every batch row too.
4. Update the row's status label (`queued` → `running` → `ok` / `partial` /
   `failed` / `aborted` / `no_rois`).

#### Zero-ROI ("genuinely noise") recordings

A recording that is just noise can finish detection with **no ROIs
surviving the cell filter**. `detection_run.compute_filtered_dff` then
writes no `r0p7_filtered_dff.memmap.float32`, so the lowpass stage (and
everything downstream) has nothing to read — the lowpass tab would error
in `_on_compute` without publishing `set_lowpass_ready`, hanging the GUI
batch forever. Immediately after detection, `_on_stage_done` calls
`_detection_has_no_rois(plane0)` (mirrors the lowpass reader: filtered
memmap present → ROIs exist; else count `predicted_cell_mask` / `iscell`).
If zero, `_skip_row_no_rois` marks `lowpass … spatial_propagation` as
**skipped**, sets the row status to **`no_rois`**, and finalizes the row
normally — so the detection folder + figures that *were* produced are
kept. The headless orchestrator `core.batch_pipeline.run_recording`
applies the same guard (`_no_rois_survive`) and returns `status="no_rois"`.

### Abort behavior

The Abort button sets `_abort_flag` (a `threading.Event`) and
relabels itself to "Aborting..." until the batch finishes. Abort
is **graceful by design** — it doesn't kill a stage mid-flight
(suite2p / GPU compute / cluster linkage can't be safely
interrupted without corrupting state). Instead the flag is
checked at every stage transition:

1. Current stage runs to completion (could be seconds or minutes
   depending on which stage was active).
2. `_begin_stage(next_stage)` sees `_abort_flag` set → marks the
   current row's status `"aborted"` → calls `_on_row_done` (skips
   the remaining stages for this row).
3. `_on_row_done` triggers `_finalize_via_mirror_async`, which
   sets the mirror's `stop_event`. The mirror does ONE final
   drain pass (with `stability_window_s=0.0` so files written
   right before stop still get caught), then exits.
4. `_batch_next_row` sees `_abort_flag` set → calls
   `_batch_finish` (skips the remaining queued rows).
5. `_batch_finish` polls `_batch_finalize_threads` until every
   in-flight mirror drain + repoint + rmtree completes, then
   writes the report and restores the buttons.

**Background agents are NOT abandoned** — they drain whatever's
left on scratch to HDD before exiting, so the HDD copy is
consistent. The trade-off is that abort isn't instant: total
abort time = `(remaining current-stage time) + (mirror drain) +
(any previous rows' finalize threads finishing)`. The current
row's status label flips to `"aborting (waiting for {stage} to
finish)"` so the user can see what's blocking.

Second click on the Abort button (already disabled, but if it
somehow fires) is idempotent — logs "abort already requested;
please wait" and short-circuits.

### Scratch keep-set (what survives the SSD→HDD copy)

`_prune_scratch_tree(scratch)` runs immediately before
`_copytree_dedup_hardlinks` and deletes known-redundant files so the
HDD copy doesn't waste bandwidth on artefacts the GUI can't reach.

**Dropped:**

- `**/data_raw.bin` / `**/data_raw_chan2.bin` — suite2p's
  pre-registration movie cache (written when `keep_movie_raw=True`).
  Same size as the registered `data.bin` (~8 GB on a 10-minute
  recording) but fully superseded once registration completes.
- `<rec>/sparsery_pass/` — sparsery's detection-pass folder.
  `sparse_plus_cellpose.run` writes per-pass `F.npy`, `Fneu.npy`,
  `spks.npy`, `stat.npy` here, then `merge_and_extract` folds the
  stat list into `final/` and re-extracts traces from the registered
  binary. Nothing reads the per-pass arrays after merge — ~300-500 MB
  of pure clutter per recording.
- `<rec>/cellpose_pass/` — cellpose's detection-pass folder (`stat.npy`
  + `cellpose_masks.npy`). Same story: consumed by `merge_and_extract`
  into `final/`, not read after. Small (~5-10 MB) but still pointless
  to copy.
- `<rec>/detection/<X>/` where `X` is not `_shared_reg` or `final` —
  defensive policy applied if a future code path ever uses this
  nesting (no-op on the current layout).
- `**/*.tmp`, `**/*.lock` — suite2p partial-write artefacts.

**Kept:**

- `<raw.name>.tif` — the Zstd-compressed raw written by Tab 3's
  post-detection archive step. Top-level. Source of truth for any
  future re-shift / re-register; readable transparently by
  `tifffile.imread`.
- `_calliope_raw_paths.json` — sidecar listing the original raw
  source paths. Tab 1's reload uses it to find the raw when
  regenerating a missing shifted; the archive step writes it
  during preprocess.
- `detection/_shared_reg/suite2p/plane0/` — registration metadata
  only (`ops.npy`, `db.npy`, `settings.npy`, `reg_outputs/`). The
  `data.bin` binary is pruned by `prune_detection_intermediates`
  and the hardlink twin under `final/` is dropped by the archive
  step.
- `detection/final/suite2p/plane0/` — all suite2p detection outputs
  (`F.npy`, `Fneu.npy`, `stat.npy`, `iscell.npy`, `spks.npy`,
  `ops.npy`, `db.npy`, `settings.npy`, `detect_outputs.npy`,
  `predicted_cell_prob.npy`, `predicted_cell_mask.npy`,
  `r0p7_dff*.memmap.float32`, `r0p7_filtered_*.memmap.float32`,
  `r0p7_cell_mask_bool.npy`, `calliope_calibration.npy`,
  `calliope_summary.xlsx`). Note: `data.bin` is **no longer kept**
  — the archive step drops it because nothing downstream reads it.
- `r0p7_filtered_cluster_results/` — `C*_rois.npy`, `linkage.npy`,
  `threshold_used.npy`, `_indices_are_suite2p` marker.
- `calliope_figures/<stage>/` — every batch-report PNG.
- `qc.gif`, `mean.npy` — Tab 1 outputs at the recording root.

The shifted TIFF (`shifted_*.tif`) is **no longer kept** after the
archive step. If a future workflow needs it (`load_existing_preprocess`
for QC reload, or a fresh `do_registration=True` rerun), it is
regenerated on demand from the compressed raw in ~1 min per 9 GB.

The patterns + keep set live in `SCRATCH_PRUNE_FILE_PATTERNS` and
`SCRATCH_DETECTION_KEEP_DIRS` at the top of `tab.py`. Add to them if
a future stage starts writing more redundant intermediates; nothing
in the keep set should ever need explicit listing because the
default is "preserve".

### What to save (user-selectable output footprint)

The **What to save** group in Tab 0 (preset + three checkboxes, just
under **Save figures during batch**) lets the user drop large artifacts
they don't need, on top of the always-on prune above. **Always kept**
(every tab needs them to render): the `plane0` ROI arrays
`F`/`Fneu`/`spks`/`stat`/`iscell`/`ops`, the `r0p7_*` dF/F + low-pass
memmaps, `calliope_summary.xlsx`, and figures. **Defaults preserve
today's behavior exactly** — an untouched run drops nothing extra.

| Control | Default | Effect when changed | Consequence |
|---------|---------|---------------------|-------------|
| Keep suite2p registration (`data.bin`) | on | off → drop the registered movie `data.bin` (both the `_shared_reg/` and `plane0/` hardlink twins) | detection can't be re-run without re-registering from the TIFF |
| Keep shifted / motion-corrected TIFF | on | off → drop `shifted_*.tif` / `NNN_shifted_*.tif` | a Tab 1 reload must regenerate it (from the compressed raw, if archived, or by re-preprocessing) |
| Archive raw TIFF (compressed) | off | on → write the Zstd-compressed raw into the output folder | adds a compact archival copy of the raw |
| Preset **Keep everything** | selected | sets all three to defaults | today's behavior |
| Preset **Bare minimum (Calliope-only)** | — | registration + shifted both off, archive off | smallest folder that still renders every tab |

Mechanics & wiring:

- The shifted TIFF and `data.bin` are still **produced** during the run
  (detection consumes them); they are pruned at row finalize, not skipped
  at write time.
- The keep-flags thread from the three `BooleanVar`s through
  `prune_scratch_tree(..., keep_registration=, keep_shifted=, final_rec=)`,
  `is_scratch_mirror_skippable(...)`, `synchronous_final_sweep(...)`, and
  `mirror_sync_pass(...)`. The mirror skips dropped artifacts so they're
  never copied to HDD; the finalize prune also removes any twin the mirror
  already copied to `final_rec` (hence the `final_rec` argument).
- `data.bin` is hardlinked between `_shared_reg/data.bin` and
  `plane0/data.bin` (same inode — obs 663). **Both** twins are unlinked,
  or the inode survives via the remaining link and no space is freed.
  Dropping `data.bin` never touches the ROI arrays.
- **Single owner per artifact across scratch / non-scratch mode.** Tab 3's
  own end-of-detection archive (`Suite2pTab._archive_recording`, reading the
  `*_post_detection` param keys) runs at the tail of *every* detection,
  including batch runs (the batch tab drives detection via `det._on_run`).
  Left at its all-True defaults it would ignore the keep toggles and, in
  scratch mode, double-compress against the finalize worker.
  `BatchTab._apply_save_policy_to_detection_tab` (called from `_begin_stage`
  on every run, not just rows with per-row overrides) reconciles this:
  - **scratch mode** → Tab 3 archive disabled
    (`archive_post_detection=False`); the finalize worker owns compression
    (gated on `compress_raw_var`) and `prune_scratch_tree` owns the
    shifted / `data.bin` keep-or-drop (gated on the keep toggles).
  - **non-scratch mode** → there is no finalize worker or
    `prune_scratch_tree` (outputs land directly at the final dir), so Tab 3's
    archive *is* the single owner: `compress_raw_post_detection` ←
    `compress_raw_var`, `delete_shifted_post_detection` ← `not keep_shifted`,
    `delete_final_data_bin_post_detection` ← `not keep_registration`.
- The finalize worker's optional raw archive runs via
  `core.detection_run.archive_recording_post_detection(compress_raw=True,
  delete_shifted=False, delete_final_bin=False)` so that step owns *only*
  compression — the keep/drop of shifted + `data.bin` belongs solely to
  `prune_scratch_tree`, avoiding double-handling.
- Toggles are session-only (not persisted), matching **Save figures**.

### Export NWB file (optional)

An **Export NWB file** parameter (Output group, default off) writes a DANDI-style `<recording_id>.nwb` for each recording. When set, the export runs at the end of the row's pipeline — after event detection, so population event windows are included — and is mirrored to the output drive with the rest of the outputs. The file carries the ImagingPlane, per-ROI PlaneSegmentation (with `iscell` probability and ROI source), Fluorescence / Neuropil / DfOverF series, the OASIS deconvolved spike estimate, and event windows; imaging metadata is read from each recording's `meta.json`. NWB support is an optional dependency (`pip install 'calliope[nwb]'`). The step is best-effort: a missing pynwb or a write error is logged and never stops the batch.

### Continuous mirror (both single-drive and two-drive scratch)

The transfer architecture is now uniform across drive layouts:
a daemon thread spawned at row start (`_start_mirror_for_row`)
polls scratch every 2 s and copies new/changed files to HDD
throughout the row's pipeline computation. The only difference
between single-drive and two-drive setups is whether the
mirror's copy operations are bandwidth-throttled:

- **Two-drive** (`_two_drive_layout=True`, set when
  `os.stat(scratch).st_dev != os.stat(output).st_dev` at run
  start): mirror runs **unthrottled** always. SSD reads and
  HDD writes are on different drives, so the mirror's HDD
  writes don't compete with the active stage's SSD reads.
- **Single-drive** (scratch and output on the same physical
  drive): **stage-aware throttle**. The mirror runs at full
  speed during every stage EXCEPT preprocess; during preprocess
  it caps at `SINGLE_DRIVE_FINALIZE_RATE_BYTES_PER_SEC`
  (10 MB/s by default) via `_throttled_copy2`. Rationale:
  preprocess is the one stage that reads source TIFFs from the
  shared drive — everything else (detection / lowpass / events
  / clustering / xcorr / spatial) operates on scratch files +
  GPU and doesn't pull big bandwidth off the output drive. So
  during those stages the disk is fair game for the mirror to
  saturate. Hardlinks (`os.link`) bypass the throttle since
  they're metadata only.

**Sequence:**

1. `_batch_next_row` resolves `rec_save` and calls
   `_start_mirror_for_row(ident, scratch_rec, final_rec,
   throttle_provider=...)`. The provider is `None` in
   two-drive mode (always unthrottled), or a closure in
   single-drive mode that returns
   `SINGLE_DRIVE_FINALIZE_RATE_BYTES_PER_SEC` when
   `self._current_stage == "preprocess"` and `None` otherwise.
   The mirror worker calls the provider at the start of every
   sync pass so the throttle setting adapts as the pipeline
   advances through stages. Thread-safe via the GIL: single-
   attribute reads are atomic, and up to one pass (~2 s) of
   staleness across a stage transition is tolerable.
2. The pipeline runs: Tab 1 writes shifted TIFF to scratch,
   Tab 3 writes plane0 outputs to scratch, Tabs 4-8 write
   memmaps and figures. After each write the mirror's next
   poll picks up the new file and copies it to HDD.
3. Each mirror pass via `_mirror_sync_pass(scratch, final,
   seen, inode_to_dst, throttle_bytes_per_sec=...)`:
   - Walks `scratch`, dedups already-synced files via the
     `(size, mtime)` cache in `seen`.
   - Skips intermediates the row-end prune would have dropped:
     `data_raw*.bin`, `sparsery_pass/`, `cellpose_pass/`,
     `detection/<not-keep>/`, `*.tmp`, `*.lock` (see
     `_is_scratch_mirror_skippable`).
   - Defers files whose mtime is < 1 s old — likely mid-write
     (suite2p flushing `data.bin` incrementally). The next pass
     catches them once writes stabilise.
   - Uses `_copy_one_dedup` so `_shared_reg/data.bin` ↔
     `final/data.bin` arrive at HDD as a hardlinked pair.
     Throttle (when set) paces each non-hardlink copy via
     `_throttled_copy2`.
   - Returns True iff anything was copied; the worker sleeps
     0.2 s after a busy pass (drain bursts) and 2 s when idle.
4. `_on_row_done` calls `_finalize_via_mirror_async(ident,
   scratch_rec, final_rec)`. The finalize worker:
   a. Sets the mirror's `stop_event`. The mirror does ONE more
      pass with `stability_window_s=0.0` so files written
      milliseconds before stop still get caught.
   b. Waits on the mirror's `synced_event` (5 min timeout).
      Usually completes in seconds in two-drive mode; may
      take a few minutes in throttled single-drive mode if
      a lot of data landed late in the pipeline.
   c. Runs `_prune_scratch_tree` defensively (catches any
      `data_raw.bin` that landed late, etc.).
   d. Marshals `_repoint_pipeline_after_copy` to the Tk main
      thread; blocks on a 30 s timeout.
   e. `shutil.rmtree(scratch_rec)` — scratch fully freed.
      First attempt runs after a `gc.collect()` to release any
      memmap handles in the just-finished pipeline worker.

      The `_repoint_pipeline_after_copy` step (d above) does two
      things that matter for the rmtree: (1) rewrite path-string
      attributes (`_plane0`, `_lowpass_plane0`, etc.) on every tab
      so future reloads land on HDD, and (2) invoke each tab's
      opt-in `_repoint_after_copy(scratch, final)` hook. The hook
      handles two responsibilities the attribute walker can't:
      rewriting path data stored inside dict-valued attributes,
      and **explicitly nulling any `np.memmap` whose backing file
      lives under `scratch`** so Windows releases the file handle
      before the bulk rmtree fires.

      Tabs that hold long-lived memmap state implement the hook:
      - **Tab 3 (suite2p)** — rewrites `_panel_cache["plane0"]`
        (the path its curation popout reads `iscell.npy` /
        `stat.npy` from), rewrites any open `CurationPopout`'s
        `_plane0`, and drops `CurationPopout._dff` (the trace-strip
        memmap opened from `r0p7_dff.memmap.float32`).
      - **Tab 6 (clustering)** — drops `self._dff` (the filtered
        dF/F memmap stashed in `_on_done`) and the open
        `ClusterPopout._dff`. `_set_plane0` doesn't reset
        `self._dff` so without this hook the scratch handle would
        leak across recordings.
      - **Tab 7 (cross-correlation)** — drops `_dff_cache` if its
        memmap points under scratch. Next preview reopens lazily
        against the HDD twin.

      Without these explicit drops, the LAST row of a batch + any
      open popout would keep the scratch dF/F memmap locked
      indefinitely (no next-row publish to fire `_set_plane0`, no
      orphan sweep until the next batch starts), and the janitor
      would spin every 30 s with no progress until app exit. New
      tab-level memmap caches need a corresponding `_repoint_after_copy`
      entry on their tab — the batch tab stays decoupled from
      tab internals.

      If rmtree still fails on the first try (e.g., antivirus is
      scanning, or a popout reopened the memmap between the drop
      and the rmtree), a second attempt fires after a 2 s wait +
      another `gc.collect`. Anything still locked is enqueued to
      the long-lived **scratch janitor** (a daemon thread started
      in `__init__`) which retries every 30 s until the handles
      release. The user is never asked to clean up manually.
5. Row status flips from `"<status> (transferring)"` to
   `"<status> (done)"` once the finalize worker finishes.

**Throttle trade-off** - at 10 MB/s the throttle only applies
during the preprocess stage; everything else runs at full
disk speed. Preprocess typically lasts 1-5 minutes on a
single-drive setup, during which the mirror copies at most
~30-150 MB before unthrottling. The rest of the row's
SSD->HDD work happens at saturated bandwidth. Tune
`SINGLE_DRIVE_FINALIZE_RATE_BYTES_PER_SEC` at the top of
`tab.py` if you want a different preprocess-time rate; the
non-preprocess passes ignore it.

**Trade-offs / failure modes:**

- Mirror copies are eventually consistent: a file written
  mid-row might be copied multiple times (once per poll
  interval where its mtime advanced) until writes stabilise.
  Worst case is wasted HDD bandwidth, not data loss.
- Mirror drain timeout (5 min): logs and proceeds with rmtree
  anyway; any unfinalized scratch files are lost.
- Repoint timeout (30 s): logs and proceeds with rmtree;
  user retries via Browse if they hit a stale AppState path.
- `_mirror_workers[ident]` carries the state until
  `_finalize_via_mirror_async` pops it; the worker thread is
  added to `_batch_finalize_threads` so `_batch_finish`'s wait
  phase joins it before writing the report.

### Scratch capacity pre-flight

Before popping each row, `_batch_next_row` calls
`_check_scratch_capacity(next_row)` (scratch-routing branch only)
to decide whether scratch has room for the row's peak footprint:

- **Budget:** `_row_scratch_budget(row)` sums the row's source
  TIFF sizes and multiplies by `_SCRATCH_BUDGET_MULTIPLIER = 3.2`.
  That covers the peak mid-detection footprint:
  shifted TIFF (~1.0×) + registered `_shared_reg/data.bin` (~1.0×)
  + `sparsery_pass/data.bin` (~1.0×) + final outputs (~0.2×). The
  per-row prune drops the two ~1.0× binaries after extraction, and
  the post-detection archive (Tab 3 step 6) then compresses the
  raw into the recording folder, deletes the shifted, and drops
  the orphaned `final/data.bin` — the residual stabilises around
  ~0.7× source-size per recording (~0.5× compressed raw + ~0.2×
  npy outputs). Budget is still sized to the peak, not the
  residual.
- **Margin:** `_SCRATCH_SAFETY_MARGIN_BYTES = 2 GB` so filesystem
  overhead and suite2p's transient working files don't push us
  over a tight budget.

Three verdicts:

- `"ok"` — free ≥ needed + margin. Pop the row and start
  preprocess.
- `"wait"` — free is below the threshold, but at least one
  finalize thread is alive or the janitor queue is non-empty.
  Re-arm via `self.after(_SCRATCH_PREFLIGHT_RETRY_MS,
  _batch_next_row)` (15 s) and check again. The GUI stays
  responsive; the row stays at the head of the queue. Log line
  throttled to once per minute so a slow finalize doesn't fill
  the console.
- `"fail"` — free is below the threshold AND nothing is in
  flight. No cleanup will ever free more space, so pop the row,
  call `_fail_row_no_space(row)` (records the failure in
  `_batch_results`, sets status to `"failed (no scratch space)"`,
  emits one error line), and advance to the next row. The
  failure shows up in the final `batch_report.csv` so the user
  can re-queue once they free space.

This is a heuristic, not a guarantee: rows we can't `stat`
contribute 0 to the budget, and `shutil.disk_usage` failures
fall through to `"ok"` (defer to runtime disk-full handling).
The pre-flight is a circuit-breaker for the common case where
a finalize hasn't drained yet; it doesn't replace the
`OSError(Errno 28)` propagation if something genuinely runs out.

### Row status and the janitor

Each row's status label runs through the following transitions during scratch-routed batches:

| State | Set by |
|---|---|
| `queued` | `_on_run_all` at queue start. |
| `running <stage>` | `_begin_stage(stage)` at each stage start. |
| `<status> (transferring)` | `_on_row_done`; row's HDD copy is still draining in the daemon mirror. |
| `<status> (cleanup pending)` | `_finalize_worker` when the row-end rmtree hit a lock and handed `scratch_rec` to the janitor. |
| `<status> (done)` | `_finalize_worker` on a clean rmtree, OR the janitor's `_mark_row_cleanup_done` callback after a successful retry. |
| `failed (no scratch space)` | `_fail_row_no_space` from the capacity pre-flight. |

The janitor binding lives in `_cleanup_row_map: dict[Path, (row, base_status)]`. When `_finalize_worker` enqueues a still-locked scratch_rec it also writes the binding; when the janitor's pass succeeds it pops the binding and schedules `_mark_row_cleanup_done(row, base_status)` on the Tk main thread via `_after_safe`. End-of-batch leftovers swept into the queue by `_batch_finish` have no row binding (they're not in `_rows` anymore) — the janitor still clears them, just without a row label to update.

### Deferred scratch cleanup (scratch janitor)

A daemon thread (`_scratch_janitor_loop`) is started in
`BatchTab.__init__` and lives for the lifetime of the tab. It
polls `_cleanup_queue` (a `list[Path]` guarded by
`_cleanup_queue_lock`) every `_CLEANUP_RETRY_INTERVAL_S` (30 s):

- `gc.collect()` before each `shutil.rmtree` attempt so any
  recently-released Python `np.memmap` references give up their
  Windows file handles before we retry.
- Entries that no longer exist (cleared since last pass) are
  removed from the queue.
- Successful rmtrees log `"scratch janitor: cleared <path>"`.
- Failures stay in the queue for the next pass.

Two enqueue paths:

1. **Row-end finalize fallback.** When `_finalize_via_mirror_async`'s
   two rmtree attempts both fail, the leftover `scratch_rec` is
   handed to the janitor. The row's finalize log line shows
   `[deferred: N files (X GB) still locked; janitor will retry
   every 30s]` instead of the old `WARN ... manual cleanup needed`
   message.
2. **Orphan sweep at batch start.** `_sweep_scratch_orphans` runs
   inside `_on_run_all` (scratch-routing branch only): it scans
   the user's scratch root and queues every subdirectory whose
   name isn't an active row identifier. Catches the case where a
   previous batch crashed, was force-quit, or hit a hard lock
   that never released before app exit. On fresh app startup all
   prior process handles are gone, so these orphans clear on the
   first janitor pass.

The janitor's existence means the user never sees a "manual
cleanup needed" message. The batch-finish "scratch not empty"
case now reports a one-liner stating the janitor will handle
the residuals and proceeds without any required action.

The pre-2026-05-11 architecture had two separate transfer
strategies (row-end semantic-priority finalize for single-drive
vs continuous mirror for two-drive). Unified to the mirror-only
design after the user pointed out that single-drive could use
the same mirror with throttle as the only difference. Code
deleted in the unification: `_finalize_to_disk`,
`_finalize_to_disk_async`, `_phase_copy_files`,
`_phase_delete_source_files`, `_is_background_path`,
`_copytree_dedup_hardlinks`, `_move_tree_dedup_hardlinks`
(all module-level helpers that supported the four-phase
finalize path).

### Same-drive contention warning (pre-run check)

Before the force-rerun confirmation in `_on_run_all`, the tab
runs `_check_drive_collisions(out)` to compare
`os.stat(path).st_dev` across the output folder, every queued
source TIFF, and the optional scratch dir. Same `st_dev` =
same partition = same physical drive (works on Windows + Linux;
exotic LVM / RAID setups may be missed, but the common
single-disk-per-mountpoint case is covered).

Two contention modes get called out:

- **Source + Output on same drive** — HDDs collapse to ~30–60
  MB/s during mixed read+write because the head seeks between
  the next row's preprocess read and the previous row's
  finalize write. 2–5× throughput drop.
- **Scratch + Output on same drive** — the SSD→HDD transfer
  becomes same-drive copying; the fast-disk savings disappear.

When either issue is detected, a modal dialog lists the
specific paths involved and offers proceed/cancel. Best-effort:
`stat` failures (unreachable mountpoint, etc.) are silently
ignored so the warning never blocks a legitimate run.

---

## Scratch dir (optional fast-disk routing)

Calcium-imaging pipelines are I/O-bound: the registered movie
(`detection/_shared_reg/suite2p/plane0/data.bin`, often 8+ GB) gets read
many times during detection, dF/F, and event extraction. When the
**Scratch dir (SSD)** field is set, the Batch tab redirects every
recording's working directory to that path during the run, then
bulk-copies the finished tree to the slow output folder and deletes the
scratch copy. The model is:

- Read each source TIFF once from the slow drive (during preprocess).
- Write the shifted uint16 TIFF, the suite2p binary, all per-pass
  detection outputs, dF/F memmaps, and figures to scratch.
- After the row finishes (success, partial, or failure),
  `_finalize_to_disk_async(ident, scratch_rec, final_rec)` spawns a
  daemon thread that runs four phases interleaved with one
  Tk-marshalled callback. Files are split by *semantic role* via
  `_is_background_path(rel)`: anything the live GUI reloads (Tab 1
  shifted TIFF + QC outputs, all of `plane0/`, dF/F memmaps,
  cluster export, summary workbook, etc.) is **pipeline-loading**;
  anything only consumed by manual suite2p re-runs or the
  post-batch report viewer is **background** (currently `data.bin`,
  `data_chan2.bin`, and anything under `calliope_figures/`).
  1. **Copy pipeline-loading.** `_phase_copy_files(include=
     not _is_background_path)` writes every pipeline-loading file
     to HDD with sources intact. Preserves intra-tree hardlinks
     via the persistent `(st_dev, st_ino)` map (the map carries
     over to the background phase so straddling hardlinks still
     work).
  2. **Repoint.** `self.after(0, _repoint_pipeline_after_copy)`
     marshals to the Tk main thread and rewrites any in-memory
     plane0 reference under `scratch` to its HDD twin —
     `AppState.plane0`, `AppState.lowpass_plane0`,
     `AppState.event_results["plane0"]`, and every tab's local
     `_plane0` / `_final_plane0` / `_plane0_path` cache. Direct
     attribute writes (no `set_plane0` call) so we don't re-fire
     subscribers and accidentally trigger Tab 4's lowpass
     recompute on a path rewrite. Worker blocks on a
     `threading.Event` until the repoint completes (30 s timeout
     as a safety valve). **Every file a reactive tab might read
     is at HDD by this point, so the user can start interacting
     with the just-finished recording immediately** — the
     background copy continues invisibly afterwards.
  3. **Delete pipeline-loading sources.**
     `_phase_delete_source_files(include=not _is_background_path)`
     walks scratch bottom-up and unlinks every pipeline-loading
     source. Scratch shrinks by ~plane0 + shifted-TIFF total
     (~10-20 GB in a typical recording).
  4. **Copy + delete background.**
     `_phase_copy_files(include=_is_background_path)` then
     `_phase_delete_source_files(include=_is_background_path)`.
     The GUI doesn't read these (`data.bin` is only for manual
     suite2p re-runs; figures only for batch reports), so any
     in-flight reload can't race the copy/delete pair.
  - **Safety-net rmtree.** `shutil.rmtree(scratch, ignore_errors=True)`
    catches any empty dirs the per-file unlinks couldn't clear
    (antivirus locks, race with a Windows handle, etc.).

  Why semantic priority and not size: the user can be reading the
  recording the moment the repoint fires (Step 2), regardless of
  whether the slow ~8 GB `data.bin` has finished transferring.
  Scratch-freeing is a side benefit — Step 3 still frees the bulk
  of scratch (shifted TIFF is the biggest pipeline-loading file)
  before Step 4 even starts.
- An earlier move-as-you-go variant (`_move_tree_dedup_hardlinks`)
  copied + immediately unlinked each source file to free scratch
  progressively, with no copy/delete split. It got reverted
  because every file in the tree spent some window between
  source-deleted and destination-still-being-copied where a GUI
  reload could fail. The phased version above bounds that risk
  to the small-file phase, where the window is <1 s. The
  unconditional move helper is kept in the module for non-scratch
  contexts.
- The orchestrator advances to the next row immediately after
  `_on_row_done` spawns the worker, so the SSD-bound preprocess
  of recording N+1 overlaps with the HDD-bound copy of recording
  N. `_batch_finalize_threads` tracks every in-flight worker;
  `_batch_finish` polls the list and waits for every thread to
  join before writing `batch_report.csv` so the report's `plane0`
  paths are guaranteed to exist on disk.
- The plane0 path in `_batch_results` is rewritten to its projected
  HDD location synchronously in `_on_row_done`, so the report sees
  the right path even though the file hasn't physically landed yet
  when the rewrite happens.

The user's slow output folder still ends up with the full set of
outputs, including `_shared_reg/` so a manual GUI re-run picks up the
cached registration.

If finalize fails (disk full, permission denied) the scratch tree is
left intact and a recovery hint is logged. Leave the field blank to
skip scratch entirely — the row state machine then writes directly to
the slow output folder, matching pre-scratch behavior.

---

## What lands on disk per row

```
<output>/<identifier>/
├── <raw.name>.tif                     ← Zstd-compressed raw (archive step)
├── _calliope_raw_paths.json           ← sidecar locating the originals
├── mean.npy, qc.gif                   ← Tab 1
├── detection/final/suite2p/plane0/    ← Tab 3 (F/Fneu/stat/ops/iscell + dF/F memmaps + cellfilter + meta.json + zdrift_qc.png/zdrift_pc1.csv; data.bin dropped)
├── detection/_shared_reg/suite2p/plane0/   ← registration metadata only (no data.bin)
├── <recording_id>.nwb                  ← optional (Export NWB file on)
├── calliope_summary.xlsx              ← Recording + ROIs (detection stage) + EventWindows/EventOnsets/RoiEventTimes/EventMonotonicity (event-detection stage) + Clusters (clustering stage) sheets. The `Recording` + `ROIs` sheets are written by `detection_run.run_detection(write_summary=True)`; before 2026-05-28 the batch detection stage wrote no sheets, so the workbook had no ROIs sheet and only got a Recording sheet if event detection ran.
├── calliope_figures/
│   ├── manifest.json                  ← rec_id, calliope_git_sha, gcamp_variant, tau, fs, pix_to_um, n_cells_{kept,total}, cellfilter_ckpt_sha256, full params dict
│   ├── preprocess/qc.gif              ← copy of Tab 1's QC GIF
│   ├── detection/{all_rois,kept_rois}.{png,svg}
│   ├── lowpass/{fft,raw_dff,lowpass_dff}.{png,svg}
│   ├── event_detection/{heatmap,raster,event_detection}.{png,svg}
│   ├── clustering/{dendrogram,spatial_clusters}.{png,svg}
│   ├── crosscorrelation/{full,per_event}.{png,svg}
│   └── spatial_propagation/event_001.{png,svg}, event_002.{png,svg}, ...
└── (cluster_results / cross_correlation_full subfolders as Tabs 6/7 produce)
```

Per-recording footprint: ~6.5 GB on a 10-min 512×512 @ 15 fps recording (~4.5 GB compressed raw + ~1.9 GB npy outputs). Down from ~20 GB in the pre-archive layout. The `shifted_*.tif` (~9 GB) and `data.bin` (~9 GB, hardlinked) are both dropped post-detection.

Across the whole batch:

```
<output>/
├── calliope_batch.json                ← queue snapshot (rows + per-row params + defaults)
└── batch_report.csv                   ← one row per recording, one column per stage
```

`batch_report.csv` columns: `recording_id, status, plane0, total_s, error,
<stage>_status, <stage>_duration_s` for each `stage` in
`(preprocess, detection, lowpass, event_detection, clustering,
crosscorrelation, spatial_propagation)`.

---

## Persistence

`calliope_batch.json` is written automatically on every Run All:

```json
{
  "default_params": { ... 74 keys ... },
  "scratch_dir": "C:/calliope_scratch",
  "rows": [
    {
      "identifier": "2024-07-01_00018",
      "tiffs": ["D:/data/.../00018.tif"],
      "params": { ... overrides only — usually empty ... }
    },
    ...
  ]
}
```

The **Reload queue** button restores the queue from the JSON in the current
output folder (so you can quit the GUI mid-run, restart, and pick up the
queue).

---

## Per-row params dialog

`BatchTab.PARAM_SPEC` is a unified list assembled by `_build_batch_param_spec`
from:

- `PreprocessTab.PARAM_SPEC` — 5 knobs (QC GIF).
- `Suite2pTab.PARAM_SPEC` — 24 knobs (Sparsery / Cellpose / Merge / dF/F /
  Default low-pass / Pixel scale [`scope_zoom`, `fov_um_reference`,
  `um_per_pixel`] / GPU).
- Hand-built detection keys `gcamp_variant` (choice; sets the suite2p
  deconvolution τ, drawn from `Suite2pTab.GCAMP_OPTIONS`) and
  `gcamp_tau_custom` (float; a positive value overrides the variant τ,
  `0` = use the dropdown). These live as Tk vars on Tab 3, not in its
  PARAM_SPEC, so they're surfaced here and applied onto `gcamp_var` /
  `custom_tau_var` by `Suite2pTab.apply_batch_row` (the public per-tab
  seam the batch runner calls — it no longer pokes tab internals
  directly; `_apply_row_params_to_tabs` just dispatches to each tab's
  `apply_batch_row`).
- `LowpassTab.PARAM_SPEC` — 5 knobs + an injected `cutoff_hz` (which Tab 4
  binds to its slider rather than a PARAM_SPEC entry).
- `EventDetectionTab.PARAM_SPEC` — 26 knobs (per-ROI hysteresis + display +
  population events density / peaks / baseline / boundaries / Gaussian fit).
- Hand-built clustering keys (`prefix`, `threshold`, `palette`).
- Hand-built xcorr keys (`max_lag_seconds`, `zero_lag`, `use_gpu`).
- Pipeline-wide (`baseline_mode`, `baseline_min`).

Each source group's label is rewritten with a numbered stage prefix
(`1. Preprocess - QC GIF`, `3. Low-pass - Butterworth`, …) so the
Advanced dialog reads top-to-bottom in operation order.

---

## Files

```
tabs/batch/
├── __init__.py    ← re-exports BatchTab
├── tab.py         ← BatchTab (queue + worker), BatchRow, _pick_tiffs_dialog
└── README.md      ← (this file)
```

The orchestrator and per-stage compute modules live in `core/`:

```
core/
├── batch_pipeline.py            ← run_recording (driver)
├── preprocessing.py             ← Tab 1 (math + run_preprocess)
├── detection_run.py             ← Tab 3 pipeline
├── lowpass_run.py               ← Tab 4
├── event_detection_run.py       ← Tab 5
├── clustering.py                ← Tab 6 (math + run_clustering)
├── crosscorrelation.py          ← Tab 7 (math + run_crosscorrelation)
└── spatial.py                   ← Tab 8 (math + render_spatial_event_figures)
```

Note: stages whose math and headless wrapper were previously split
across paired `*_run.py` files have been consolidated. The remaining
`*_run.py` modules (`detection_run`, `lowpass_run`,
`event_detection_run`) keep that name because they have no separate
math counterpart — the algorithm lives in `core/utils.py` /
`core/sparse_plus_cellpose.py` / `core/suite2p_pipeline.py` and the
`*_run` file is just the headless wrapper.

---

## Reproducibility tips

- **Save the unified Advanced dialog before Run All.** The dialog returns
  immediately on OK and writes back into `BatchTab.default_params`. The
  next `_save_queue` (which fires inside `_on_run_all`) snapshots those
  defaults into `calliope_batch.json` alongside per-row overrides, so the
  JSON is the canonical record of what was actually run.
- **`batch_report.csv` is the audit log.** Re-run only the rows whose
  status is `failed` or `partial`; the others have all their figures and
  per-tab outputs on disk already. Rows with status `no_rois` were
  genuinely noise (zero ROIs survived the cell filter) — their downstream
  stages are intentionally `skipped`, so there is nothing to re-run.
- **A stage worker that crashes fails-and-continues — it never stalls.**
  The driver triggers each stage's tab worker, then waits on a one-shot
  AppState "ready" signal. If the worker raises *before* publishing that
  signal, there is nothing to wait for, so the queue would hang forever.
  The fix: every analysis tab's `_on_error` calls
  `gui_common.report_stage_error(self.state, msg)`, which publishes the
  failure on the `AppState` stage-error channel (and, while a batch is
  active, suppresses the tab's blocking modal so it can't freeze the
  queue). `_arm_completion` (and the custom clustering / xcorr stages)
  subscribe to that channel and call `_fail_stage` the instant a worker
  errors — sharing the one-shot guard with the ready path so whichever
  arrives first wins. `state.batch_active` is set in `_on_run_all` and
  cleared in `_batch_finish`.
- **RAM out-of-memory => the recording is auto-retried conservatively.**
  If a stage worker dies with a memory-allocation error (`MemoryError`,
  numpy *"Unable to allocate…"*, CUDA/CuPy *"out of memory"*), the whole
  recording is **re-run from preprocess with progressively lower resource
  settings** before being marked failed — up to `_MAX_OOM_RETRIES` (2)
  attempts. `_fail_stage` classifies the error via `_is_oom_error` and, if
  retries remain, calls `_retry_row_conservative`: it tears down the
  failed attempt race-free (stops the scratch mirror, rmtrees the partial
  scratch + HDD output) in a daemon thread, then re-queues the same row at
  the front. `_apply_resource_profile` (called each stage in
  `_begin_stage`) then applies the conservative bundle for
  `row._oom_attempt > 0`: suite2p `batch_size` ×0.5 then ×0.25 via the
  adaptive sizer's `conservative_scale` (`utils.AdaptiveBatchSizer`), GPU
  off (dF/F + xcorr), `nbins=500`, and a coarser preprocess QC downsample
  (the one preprocess allocation not covered by its streaming fallback).
  The overrides are snapshotted/restored so they never leak into later
  normal rows. Disk-full errors are **not** retried (smaller batches can't
  free disk) — they fall through to a normal failure. RAM OOM is most
  likely in detection/suite2p; preprocess itself is largely OOM-safe
  (single-TIFF shift auto-falls-back to streaming; the grouped path is
  already streaming), so re-running from preprocess is safe.
- **Failures in one stage don't poison the next.** Detection failure
  short-circuits the rest of that row; lowpass / events / clustering /
  xcorr / spatial each fail independently. Spatial-propagation figures
  silently skip when event detection produced no event windows.
- **<2 ROIs => clustering + xcorr are skipped, not crashed.** Clustering
  needs at least two ROIs (hierarchical linkage is undefined for a single
  trace — `pdist` on one column raises "empty distance matrix"). A
  recording with exactly one kept ROI clears detection / lowpass / events
  but can't cluster. After event detection, `_on_stage_done` checks
  `_kept_roi_count(plane0)` and, when `<2`, calls
  `_skip_clustering_xcorr_few_rois`: clustering + crosscorrelation are
  marked `skipped` and the row jumps to spatial propagation (which only
  needs events), so it still finishes with its event/spatial output.
  `core.clustering._ward_linkage` / `tabs.clustering.logic.ward_linkage`
  also raise a clear `ValueError` for this case, and the headless
  `batch_pipeline.run_recording` applies the same `_kept_roi_count` skip.
- **No events => per-event xcorr is skipped, not stalled.** When event
  detection finds zero population events, the xcorr tab's
  `_on_run_per_event` bails early without publishing `set_xcorr_ready`,
  which would otherwise hang the batch's two-step crosscorrelation
  completion counter. `_stage_crosscorrelation` checks
  `xc._event_windows` after the full-recording xcorr completes and, when
  empty, advances straight to stage-done. The full-recording xcorr (no
  events needed) still runs and its figures are kept. The headless
  `crosscorrelation.run_crosscorrelation` already guards the same case
  (`if event_windows is not None and len(event_windows) > 0`).
