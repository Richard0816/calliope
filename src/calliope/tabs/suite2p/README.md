# Tab 3 — Suite2p Detection

**Goal.** Take the shifted TIFF from Tab 1 and produce, per cell:
- a spatial footprint (which pixels belong to it, with weights),
- a fluorescence trace `F[t]` over time,
- a neuropil trace `Fneu[t]` (out-of-focus contamination),
- a relative-change-from-baseline trace `dF/F[t]`,
- a default low-pass dF/F and its first time-derivative,
- a learned "is this really a cell?" classifier verdict.

This is the heaviest tab — it kicks off Suite2p, Cellpose, the dF/F computation (CPU or CuPy GPU), the cell-filter CNN inference, and finally rasterises everything onto the mean image for visual review.

---

## 1. Inputs

- `result.shifted_tiff` (and its sibling shifted TIFFs in the same folder) from Tab 1.
- The in-source **base settings** dict from `calliope.core.calliope_settings` (defines per-pixel size, registration knobs, etc.). The user's Advanced parameters and the "Edit suite2p settings..." popout override individual fields at runtime.
- A **cell-filter checkpoint** (`.pt`) for the trained `CellFilter` model. Ships bundled at `<calliope/data>/cellfilter_best.pt` and is loaded by default. Users can point the Browse... field at their own retrained checkpoint; if the file is missing or unreadable, the third panel falls back to Suite2p's `iscell.npy`.
- The **per-recording GCaMP variant**, picked from the dropdown in the Tab 3 header. Resolves to the Suite2p deconvolution `tau` (seconds). Default is GCaMP6m (`tau = 1.0 s`).

---

## 2. Pipeline steps

The tab worker (`Suite2pTab._run_pipeline`) is roughly:

```python
final_plane0 = sparse_plus_cellpose.run(...)   # ROI detection
self._stamp_pix_to_um(final_plane0, params)    # ops['pix_to_um'] from zoom or direct
_run_dff(final_plane0, baseline_mode, ...)     # dF/F + lowpass + derivative
if ckpt_path:
    _run_cellfilter(final_plane0, ckpt_path)   # PyTorch keep/drop
_run_filtered_dff(final_plane0)                # slice dF/F to kept ROIs
```

The same chain is also exposed as a single headless callable:
`core.detection_run.run_detection(tiff_folder, save_folder, params, ...)`.
That's what Tab 0's batch runner invokes — both the tab and the batch
worker share one code path, so anything tested via the GUI behaves
identically when batched.

### suite2p 1.0 adapter

Suite2p 1.0.0.1 (PyPI 2026-02-11) made two breaking changes the rest of
this tab is shielded from:

1. `run_s2p` no longer accepts the flat `ops=` kwarg; it takes a nested
   `db=` + `settings=` schema, with renames such as
   `ops['nbinned'] -> settings['detection']['nbins']`,
   `ops['high_pass'] -> settings['detection']['highpass_time']`, and
   `ops['roidetect'] -> settings['run']['do_detection']`.
2. Registration / detection progress moved from `print()` to
   `logging.getLogger('suite2p')`.

`core/adaptive_detection.py` carries the translation: the
`_OPS_TO_SETTINGS` + `_OPS_RENAMES` tables map flat keys to their new
nested locations, `_coerce_to_default_type` converts legacy
float-stored bools (`two_step_registration`, `nonrigid`,
`do_bidiphase`, `look_one_level_down`, …) to real `bool`/`int` so
`range(1 + settings['two_step_registration'])` doesn't crash, and an
empty-list normalisation rewrites `subfolders=[]` / `file_list=[]`
to `None` so the new `get_file_list` doesn't shortcut into "no files
found". `_ensure_s2p_logger` rebinds the suite2p logger's stream to
the current `sys.stderr` on each `_run_s2p` call so registration
progress reaches the GUI's `redirect_stderr` capture.

CalLIOPE keeps a flat ops dict as its internal lingua franca because
the on-disk base ops `.npy` is shaped that way and downstream tabs
read `ops['Ly']` / `ops['spatial_scale']` / `ops['pix_to_um']`
straight from suite2p's own output. Translation only happens at the
`run_s2p` call boundary.

### Step 1 — `sparse_plus_cellpose.run` (ROI detection)

A union of two complementary detectors. Sparsery is bright-and-bursty; Cellpose is a generalist segmenter that catches morphologically obvious cells which never fired enough during the recording for Sparsery to spot.

**Sparsery (Suite2p built-in detector).** Driven by the ops dict:

| ops key | Default | What it does |
|---|---|---|
| `threshold_scaling` | 0.85 | Lower = more ROIs (less stringent). |
| `high_pass` | 100 frames | Frame-domain high-pass to suppress slow drift before detection. |
| `smooth_sigma` | 1.0 | Spatial Gaussian smoothing applied during detection. |
| `max_iterations` | 1500 | Detection loop iteration cap. |
| `spatial_scale` | 0 (auto) | Suite2p's coarse-to-fine scale parameter. |
| `preclassify` | 0.0 | Pre-detection classifier threshold (off by default). |
| `sparse_mode` | True | Use Sparsery (vs the older Cellpose/anatomical pipeline). |

The user can also set `hard_cap` — a safety abort that bails out if Sparsery returns more ROIs than this (default 60,000), since runaway detection usually means the parameters are wrong.

**Cellpose pass.** Run on the **mean image** (default `cellpose_channel_input='meanImg'`). Parameters:

| Param | Default | Notes |
|---|---|---|
| `cellpose_model_type` | `cyto2` | Generalist cytoplasm model. |
| `cellpose_diameter` | 0 | 0 = let Cellpose auto-estimate. |
| `cellpose_flow_threshold` | 0.8 | Standard Cellpose flow threshold. |
| `cellpose_cellprob_threshold` | -1.0 | More inclusive than the default 0. |

**Merge.** Each Cellpose ROI is dropped if its overlap with any Sparsery ROI exceeds `max_overlap` (default `0.3`, i.e. 30% of the Cellpose ROI's pixels covered by a Sparsery ROI). The remaining Cellpose ROIs are appended to Sparsery's output to form the final `stat.npy`.

**dF/F per Suite2p convention** is computed downstream (Step 2). Suite2p also produces:
- `F.npy` — `(N_total, T)` per-ROI raw fluorescence (sum over `xpix/ypix` weighted by `lam`).
- `Fneu.npy` — `(N_total, T)` neuropil traces.
- `stat.npy` — list of dicts; each carries `xpix`, `ypix`, `lam`, `med`, `radius`, etc.
- `ops.npy` — the running ops with mean/max-projection images appended.
- `iscell.npy` — Suite2p's built-in classifier output (binary + confidence).

The result lives at `<out_dir>/detection/final/suite2p/plane0/`. From here on we call this `plane0`.

### Step 2 — dF/F, low-pass, derivative

The neuropil-corrected, baselined trace per ROI is:

```
F_corr[t]  = F[i, t] − r · Fneu[i, t]                      # neuropil correction
F0[t]      = baseline of F_corr (rolling pct OR first-N-min mean)
dF/F[t]    = (F_corr[t] − F0[t]) / F0[t]                   # relative change
```

Two **baseline modes** controlled by the radio buttons:

- **Rolling**: `F0[t]` is a rolling 10th-percentile over a `win_sec=45 s` window (`utils.robust_df_over_f_1d`). Sliding-window percentile (`scipy.ndimage.percentile_filter`) tracks slow brightness changes (photobleaching, slice swelling) without being skewed by transients.
- **First N minutes**: `F0` is a single scalar = the 10th percentile of the first `baseline_min · 60 · fps` frames (`utils.first_n_min_df_over_f_1d`). Best for short recordings where the baseline doesn't drift, and **required** for the GPU path.

`r = 0.7` is the **neuropil correction coefficient** (Chen et al. 2013). Subtracting `0.7×Fneu` removes ~70% of out-of-focus contamination on average; the residual is what's actually from the soma.

After dF/F, the tab pre-computes:

- **Low-pass dF/F** at `default_lowpass_hz = 1.0 Hz`, order-2 causal Butterworth via `utils.lowpass_causal_1d` (see `tabs/lowpass/README.md` for the SOS state-space details).
- **Savitzky-Golay first derivative** of the low-pass trace, window `default_sg_win_ms = 333 ms`, polynomial order `2`, via `utils.sg_first_derivative_1d`. The output dimension is `dF/F per second`.

These three arrays are written as `(T, N_total)` `float32` memmaps in `plane0/`:

- `r0p7_dff.memmap.float32`
- `r0p7_dff_lowpass.memmap.float32`
- `r0p7_dff_dt.memmap.float32`

The `r0p7_` prefix encodes "neuropil coefficient r=0.7" and is hardcoded by older downstream tools (clustering, crosscorr, paper figures); changing it would require updating those.

#### CPU loop

```python
for i in range(N):
    trace = F[i] − r·Fneu[i]
    dff   = first_n_min_df_over_f_1d(trace, ...)  or  robust_df_over_f_1d(trace, ...)
    lp, _, sos = lowpass_causal_1d(dff, fps, cutoff_hz, order=2, sos=cached)
    dt  = sg_first_derivative_1d(lp, fps, win_ms=333, poly=2)
    dff_mm[:, i] = dff
    lp_mm[:, i]  = lp
    dt_mm[:, i]  = dt
```

The Butterworth `sos` is computed once and reused across ROIs.

#### GPU path (`_maybe_run_dff_gpu`)

If `use_gpu_dff` is true **and** `baseline_mode == 'first_n'` **and** `analyze_output_gpu` imports successfully **and** CuPy is available:

```python
gpu.process_suite2p_traces_gpu(
    F, Fneu, fps=fps, r=r, baseline_sec=baseline_min*60,
    cutoff_hz=cutoff_hz, sg_win_ms=..., sg_poly=...,
    out_dir=plane0, prefix='r0p7_', roi_chunk=None_or_int,
)
```

Vectorises the per-ROI loop on GPU and writes the same three memmaps. The rolling baseline is **not** supported on GPU because it requires a windowed percentile filter that doesn't have a clean CuPy implementation.

`gpu_roi_chunk` lets the user split into smaller VRAM chunks if a recording overflows.

### Step 3 — Cell-filter CNN (overview)

Suite2p's built-in `iscell.npy` classifier is noisy across recordings — it tends to keep blood vessels, dendritic clutter, and bright-pixel artefacts. The cell filter is an in-house, hand-trained replacement.

**What it is.** A small **two-branch convolutional neural network** (PyTorch, ~250k parameters) that scores each detected ROI from 0 (artefact) to 1 (real cell). It looks at two things per ROI:

- **Spatial branch** — a 32×32 pixel patch around the ROI's centroid, three channels: the **mean image**, the **max-projection image**, and a **binary mask** showing which pixels Suite2p assigned to this ROI. A 2D CNN compresses this to a 64-d embedding. Real cells look round, bright, and locally distinct; vessels are elongated, dendrites are stringy.
- **Temporal branch** — the ROI's z-scored dF/F trace. A 1D CNN compresses it to a 64-d embedding. Real cells have transient calcium events; artefacts look like flat noise or slow drift.

The two embeddings are concatenated (128-d) and passed through a small MLP head that outputs a single logit; `sigmoid` gives the cell probability.

**How it's trained.** A `roi_curation.csv` of hand-labelled ROIs (from this lab's recordings) provides positive/negative examples. The training loop uses BCE loss with a positive-class weight to handle class imbalance, splits validation by ROI, augments by random temporal cropping, and selects the best checkpoint by validation AUROC.

**How it's applied here.** The Tab 3 worker:

```python
from calliope.core.cellfilter.model import CellFilter
from calliope.core.cellfilter.predict import predict_recording

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model = CellFilter().to(device); model.load_state_dict(ckpt['model']); model.eval()
predict_recording(rec_id, model, device, plane0=plane0)
```

`predict_recording` walks every ROI in the recording and writes:
- `predicted_cell_prob.npy` — float, length `N_total`, the model's score in `[0, 1]`.
- `predicted_cell_mask.npy` — boolean, length `N_total`, `True` iff `prob >= 0.5`.

If the checkpoint is missing, the tab logs a warning and the third panel falls back to Suite2p's `iscell.npy[:, 0] > 0`.

→ See [`core/cellfilter/README.md`](../../core/cellfilter/README.md) for the full architecture, dataset construction, training procedure, and inference details.

### Step 4 — Filtered dF/F

`_run_filtered_dff(plane0)`: load `r0p7_dff.memmap.float32` of shape `(T, N_total)`, slice columns by the keep mask, write `r0p7_filtered_dff.memmap.float32` of shape `(T, N_kept)`. Done in 4096-frame chunks to bound memory.

Also writes `r0p7_cell_mask_bool.npy` (a copy of the keep mask under the legacy filename) so older downstream tools (clustering scripts, paper figures) that hardcode that name keep working.

If the **Also write filtered dF/F as CSV** checkbox is on, `_write_filtered_dff_csv` mirrors the memmap into `r0p7_filtered_dff.csv` with one column per kept ROI, headed `roi_<i>` (the Suite2p index), in 4096-frame chunks. The frame index is preserved as the CSV index.

### Step 5 — Intermediate-binary cleanup

The shared helper `core.detection_run.prune_detection_intermediates(save_folder, *, progress_cb=None)` runs immediately after `_run_filtered_dff` and reclaims the ~26 GB/recording of intermediate suite2p binaries that the detection passes leave behind:

- drops `<save>/detection/sparsery_pass/` (detection-only pass binary + per-pass `F/Fneu/spks/stat` already folded into `final/`),
- drops `<save>/detection/cellpose_pass/` (defensive — same policy as the sparsery pass),
- drops `<save>/detection/_shared_reg/suite2p/plane0/data.bin` (registered-movie binary; regeneratable from the shifted TIFF in ~5 min for a manual re-detect),
- drops `data_raw*.bin` / `*.tmp` / `*.lock` anywhere under `detection/`,
- **keeps** `_shared_reg/ops.npy` + the rest of the registration metadata for audit, and **keeps** everything under `final/`.

The same helper is called from `core.detection_run.run_detection` so headless drivers (`batch_pipeline.run_recording`, any external agent that calls `run_detection` in a loop) get the cleanup unconditionally — not just GUI runs. This mirrors `BatchTab._prune_scratch_tree`'s policy but runs at the tail of every detection. Without it, an agent that runs detection over many recordings will fill the save drive (12 recordings ≈ 312 GB; observed disk-full at recording 13 in the 2026-05-12 batch).

`gc.collect()` runs before the prune so any lingering suite2p `np.memmap` references are released and Windows lets `unlink`/`rmtree` succeed on the first pass. Errors on individual paths are caught and reported via `progress_cb` so a single file lock doesn't abort the whole cleanup.

### Step 6 — Post-detection archive

Immediately after the prune, `core.detection_run.archive_recording_post_detection(save_folder, ...)` collapses the recording folder to the smallest stable footprint:

1. **Compress raws into the recording folder.** Reads `_calliope_raw_paths.json` (written by Tab 1's `run_preprocess`) to locate the originals, then re-encodes each as Zstd-19 + horizontal predictor and writes it to `<rec>/<raw.name>.tif` at top level. Byte-equality is verified by decompressing the temp file and comparing to the original array before an atomic rename swaps it into place — the source is never overwritten unless the round-trip matches exactly. ~50–60% size reduction is typical on int16/int32 fluorescence TIFFs.
2. **Delete the shifted TIFFs.** Top-level `*shifted_*.tif` files go away once compression has succeeded for every raw. They're regenerable from the compressed raw via `load_existing_preprocess` (Tab 2's reload path) — see Tab 1's README §4.
3. **Delete `<rec>/detection/final/suite2p/plane0/data.bin`.** This is the hardlink twin of the now-pruned `_shared_reg/data.bin`; both sides have to be unlinked to actually free the ~9 GB on disk. Nothing downstream (Tabs 4–8) reads it — only `merge_and_extract` opens it during extraction, after which F.npy / Fneu.npy / spks.npy carry forward.
4. **Optionally delete the user's external raw originals** (off by default; opt-in via `delete_external_raw_after_archive`).

If compression fails on any raw, the archive step aborts **before** any deletions so the recording stays in a re-runnable state.

Per-recording disk goes from ~20 GB (shifted + orphan `final/data.bin`) to ~6.5 GB (compressed raw + npy outputs).

**Param keys** (read from the same `params` dict that drives the rest of detection):

| Key | Default | Effect |
|---|---|---|
| `archive_post_detection` | `True` | Master switch for the archive step. |
| `compress_raw_post_detection` | `True` | Compress raws into the recording folder. |
| `delete_shifted_post_detection` | `True` | Remove top-level `*shifted_*.tif`. |
| `delete_final_data_bin_post_detection` | `True` | Remove `detection/final/.../data.bin`. |
| `delete_external_raw_after_archive` | `False` | Destructive opt-in: delete the original raw at its source path after a verified compressed copy lands. |
| `raw_compression_level` | `19` | Zstd compression level (1–22). |

**Re-detection cost.** Once archived, re-running detection costs an extra ~1 min for `load_existing_preprocess` to regenerate the shifted from the compressed raw, plus the usual ~5 min of registration + detection. The acceptable trade for ~13 GB/recording saved.

---

## 3. Outputs (in `plane0/`)

| File | Shape | Notes |
|---|---|---|
| `stat.npy` | `(N_total,)` object array of dicts | per-ROI footprint + statistics |
| `ops.npy` | dict | Suite2p ops + projection images |
| `F.npy` | `(N_total, T)` | raw per-ROI fluorescence |
| `Fneu.npy` | `(N_total, T)` | neuropil trace |
| `iscell.npy` | `(N_total, 2)` | Suite2p classifier (binary + confidence) |
| `r0p7_dff.memmap.float32` | `(T, N_total)` | dF/F (note: time-major) |
| `r0p7_dff_lowpass.memmap.float32` | `(T, N_total)` | low-pass at default 1 Hz |
| `r0p7_dff_dt.memmap.float32` | `(T, N_total)` | SG first derivative |
| `predicted_cell_mask.npy` | `(N_total,)` bool | cell-filter keep mask |
| `predicted_cell_prob.npy` | `(N_total,)` float | cell-filter confidence |
| `r0p7_filtered_dff.memmap.float32` | `(T, N_kept)` | dF/F restricted to kept ROIs |
| `r0p7_cell_mask_bool.npy` | `(N_total,)` bool | legacy duplicate of the keep mask |
| `r0p7_filtered_dff.csv` | optional | per-frame CSV view of filtered dF/F |

A summary workbook (`<rec>_summary.xlsx`) with an ROIs sheet is also written via `summary_writer.write_rois_sheet`.

---

## 4. UI panels

1. **Suite2p console** — captures stdout/stderr from the worker (`QueueWriter` + `contextlib.redirect_stdout`).
2. **Detected ROIs (raw Suite2p output)** — overlays a `nipy_spectral` label image on a chosen background (`meanImgE` by default; user can switch to `meanImg`, `max_proj`, `Vcorr`, `refImg`, `meanImg_chan2`). **Click any ROI** to open the curation popout (see §6).
3. **After cell-filter** — same background; if `predicted_cell_prob.npy` is present, overlays a `viridis` heatmap of `prob ∈ [0.5, 1.0]` so you can see *how confident* the classifier is per ROI. Otherwise overlays the keep set in `nipy_spectral`.

---

## 6. Curation popout (click an ROI on panel 2)

Module: `tabs/suite2p/curation_popout.py`. Mirrors the standalone reference UI in `Calcium_imaging_suite2p/roi_curation_app.py`. Single-instance per plane0 — clicking a different ROI swaps focus on the existing window.

Panels:
1. Mean image with a yellow locator box around the clicked ROI's bbox.
2. Max projection with the same locator box (max projection is reconstructed from `ops['max_proj']` padded back to full frame using `ops['yrange']` / `ops['xrange']`).
3. Max projection + ROI footprint scatter overlay (red).
4. ROI footprint alone (axis-equal scatter for shape inspection).
5. ΔF/F trace from `r0p7_dff.memmap.float32` (full all-ROI memmap, not the filtered one — figure 2 panel cell rejects are still inspectable).

Controls:
- **Cell (1)** / **Not a cell (0)** flip iscell.npy in place and append to `cellfilter.config.LABELS_CSV` (default `F:\roi_curation.csv`) with columns `recording_ID, ROI_number, user_defined_cell` so the next training run picks up the new labels. Tab 3's panels 2 + 3 repaint immediately to reflect the new keep set.
- **Retrain cell filter** stays disabled until the user has flipped at least one ROI's classification this session. On click it spawns a worker thread that runs `cellfilter.train.main()` (synchronous, ~40 epochs); the GUI stays responsive but the popout's flip buttons stay enabled (the trainer reads the CSV at start, so further mid-training flips land in the *next* round).
- **Promote to filter mask** stays disabled until the user has flipped at least one ROI this session. On click it overlays the flipped indices onto `predicted_cell_mask.npy` (the file every downstream tab reads), preserving CNN predictions for un-touched ROIs. Use this when you want manual flips to take effect downstream (Tab 5/6/7, cross-correlation) *without* the cost of a full retrain. If `predicted_cell_mask.npy` does not yet exist (cell-filter never ran on this recording), the file is created from `iscell.npy[:, 0]` with the flips applied.
- **Show 1 Hz lowpass** overlays a 1 Hz order-2 causal Butterworth (`utils.lowpass_causal_1d`, the same filter the pipeline applies in Step 2) on top of the raw ΔF/F trace so the curator can preview what the trace will look like after the downstream lowpass. Toggle state persists across ROI switches — flip it on once and it stays on for the rest of the queue. Silently skipped on traces with non-finite samples (sosfilt is IIR and would propagate NaN).

Keybinds: `1` = cell, `0` = not a cell, `Esc` = close.

---

## 5. Parameters (full list)

```
Sparsery:
  threshold_scaling=0.85, high_pass=100, smooth_sigma=1.0,
  max_iterations=1500, spatial_scale=0, preclassify=0.0,
  hard_cap=60000

Cellpose:
  model=cyto2, diameter=0, flow_threshold=0.8,
  cellprob_threshold=-1.0
Merge:
  max_overlap=0.3

dF/F:
  fps_override=0 (0=auto from notes via get_fps_from_notes; falls back to 15.07),
  neuropil_coef r = 0.7, baseline_pct = 10, win_sec = 45 (rolling)

Default lowpass / derivative:
  cutoff = 1.0 Hz, sg_win_ms = 333, sg_poly = 2

GPU:
  use_gpu_dff = True, gpu_roi_chunk = 0 (0=all)

Pixel scale (µm calibration written to ops['pix_to_um']):
  scope_zoom = 0.0    (0 = skip; uses lab reference FOV 3080.9 µm at 1×)
  um_per_pixel = 0.0  (0 = skip; explicit calibration; wins over zoom)
```

Baseline mode (radio): `rolling` (45 s rolling 10th pct) or `first_n` (first N minutes mean of 10th pct).

The pixel-scale knobs are resolved through `core.scale.resolve_pix_to_um`
after detection and stamped onto `plane0/ops.npy['pix_to_um']` so Tabs 6
(spatial cluster map) and 8 (per-event order maps) render axes in µm
without any further user input.

---

## 6. Re-implementation checklist

1. Wire up Suite2p (https://github.com/MouseLand/suite2p) and Cellpose (https://github.com/MouseLand/cellpose). The merge needs a per-ROI footprint set (each cell knows its `xpix`, `ypix`, `lam`).
2. Compute the union with the overlap drop: for each Cellpose ROI, if any Sparsery ROI shares >`max_overlap` of its pixels, discard the Cellpose ROI; otherwise append it to the final ROI list.
3. Implement both dF/F baselines:
   - **Rolling**: `scipy.ndimage.percentile_filter(F_corr, size=(45·fps)|1, percentile=10, mode='nearest')`, eps = `max(percentile(F0, 1), 1e-9)`.
   - **First-N-minute**: scalar `F0 = percentile(F_corr[:N_baseline], 10)` where `N_baseline = round(baseline_min · 60 · fps)`; eps = `max(F0, 1e-9)`.
4. Implement the causal Butterworth low-pass via `scipy.signal.butter(order=2, btype='low', output='sos')` + `sosfilt`, initialising `zi` to the first sample to avoid a startup transient.
5. Implement the SG derivative via `scipy.signal.savgol_filter(x, window_length=odd, polyorder=2, deriv=1, delta=1/fps)`.
6. Train (or load) a small CNN that takes the dF/F + footprint of an ROI and outputs `P(real cell)`. Save mask and prob to `predicted_cell_mask.npy`, `predicted_cell_prob.npy`. Fall back to `iscell.npy[:,0] > 0` if no model.
7. Slice the dF/F memmap by the keep mask in chunks (≤ 4096 frames at a time) to write `r0p7_filtered_dff.memmap.float32`.
8. Honour the on-disk filename conventions exactly — every later tab and figure script reads `r0p7_filtered_dff.memmap.float32` and `predicted_cell_mask.npy` by name.


## UI affordances

Tab 3 inherits the global customtkinter dark theme from `pipeline_gui`.

- **Resizable console.** The "Suite2p console" (Panel 1, top) carries a draggable handle below it — drag down to grow the log without squeezing the detection panels beneath; the scrollable tab body absorbs the extra height.
- **Detection panels** (2: raw ROIs, 3: after cell-filter mask). Matplotlib figures keep their white facecolor; toolbars are dark-skinned.
- **Click an ROI** on the detected-ROIs panel to open the **CurationPopout** (`CurationPopout`) — resizable Toplevel that displays five per-ROI panels (locator + trace + four backgrounds) and lets the user re-label cells / retrain the cell-filter.
- **Edit suite2p settings...** opens a second `AdvancedDialog`-style popout for arbitrary `ops` overrides; "Advanced..." opens the standard PARAM_SPEC form.
