# Tab 0 — Batch runner

Drives the full **Tabs 1 + 3-8** pipeline over a list of recordings without
the user having to click each tab one recording at a time.

---

## What the user sees

- **Top bar.** Working dir + recursion **Depth** + **Scan**, output folder +
  **Reload queue**, **Apply defaults to all rows**, **+ Add row**, **Run all**
  / **Abort**.
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
   ...)`. The orchestrator chains the per-stage `core/*_run.py` modules:

   ```
   preprocess_run -> detection_run -> lowpass_run -> event_detection_run
                  -> clustering_run -> crosscorrelation_run -> spatial_run
   ```

   Each stage is wrapped in its own try/except. If a non-detection stage
   fails, the recording is marked **partial** and the worker moves on.
   Detection failure is fatal for that row only; the queue continues.
4. Update the row's status label (`queued` → `running` → `ok` / `partial` /
   `failed` / `aborted`).

The Abort button sets a `threading.Event` the worker checks between rows.
It does **not** kill a running stage mid-flight — that would risk leaving
half-written memmaps on disk.

---

## What lands on disk per row

```
<output>/<identifier>/
├── shifted_<…>.tif                    ← Tab 1 (single) or NNN_shifted_<…>.tif (group)
├── mean.npy, blobs.npy, qc.gif        ← Tab 1
├── detection/final/suite2p/plane0/    ← Tab 3 (F/Fneu/stat/ops/iscell + dF/F memmaps + cellfilter)
├── calliope_summary.xlsx              ← ROIs + EventWindows + Clusters sheets
├── calliope_figures/
│   ├── preprocess/qc.gif              ← copy of Tab 1's QC GIF
│   ├── detection/{all_rois,kept_rois}.png
│   ├── lowpass/{fft,raw_dff,lowpass_dff}.png
│   ├── event_detection/{heatmap,raster,event_detection}.png
│   ├── clustering/{dendrogram,spatial_clusters}.png
│   ├── crosscorrelation/{full,per_event}.png
│   └── spatial_propagation/event_001.png, event_002.png, ...
└── (cluster_results / cross_correlation_full subfolders as Tabs 6/7 produce)
```

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

- `PreprocessTab.PARAM_SPEC` — 10 knobs (blob detection + QC GIF).
- `Suite2pTab.PARAM_SPEC` — 23 knobs (Sparsery / Cellpose / Merge / dF/F /
  Default low-pass / Pixel scale / GPU).
- `LowpassTab.PARAM_SPEC` — 5 knobs + an injected `cutoff_hz` (which Tab 4
  binds to its slider rather than a PARAM_SPEC entry).
- `EventDetectionTab.PARAM_SPEC` — 26 knobs (per-ROI hysteresis + display +
  population events density / peaks / baseline / boundaries / Gaussian fit).
- Hand-built clustering keys (`prefix`, `threshold`, `palette`).
- Hand-built xcorr keys (`max_lag_seconds`, `zero_lag`, `use_gpu`).
- Pipeline-wide (`baseline_mode`, `baseline_min`).

Each source group's label is rewritten with a numbered stage prefix
(`1. Preprocess - Blob detection`, `3. Low-pass - Butterworth`, …) so the
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
├── preprocess_run.py            ← Tab 1
├── detection_run.py             ← Tab 3 pipeline
├── lowpass_run.py               ← Tab 4
├── event_detection_run.py       ← Tab 5
├── clustering_run.py            ← Tab 6
├── crosscorrelation_run.py      ← Tab 7
└── spatial_run.py               ← Tab 8 figures
```

---

## Reproducibility tips

- **Save the unified Advanced dialog before Run All.** The dialog returns
  immediately on OK and writes back into `BatchTab.default_params`. The
  next `_save_queue` (which fires inside `_on_run_all`) snapshots those
  defaults into `calliope_batch.json` alongside per-row overrides, so the
  JSON is the canonical record of what was actually run.
- **`batch_report.csv` is the audit log.** Re-run only the rows whose
  status is `failed` or `partial`; the others have all their figures and
  per-tab outputs on disk already.
- **Failures in one stage don't poison the next.** Detection failure
  short-circuits the rest of that row; lowpass / events / clustering /
  xcorr / spatial each fail independently. Spatial-propagation figures
  silently skip when event detection produced no event windows.
