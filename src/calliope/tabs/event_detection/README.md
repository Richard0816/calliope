# Tab 5 — Event Detection

**Goal.** Two products:
1. **Per-ROI onset times** — for each cell, the frames at which it transitioned from "quiet" to "firing." This is the calcium-imaging equivalent of a spike train.
2. **Population event windows** — `[start_s, end_s]` intervals during which many cells fired together. The calcium-imaging equivalent of an EEG spike or ictal burst.

Three panels: a sorted heatmap of low-pass dF/F, a sorted onset raster, and the population-density trace with detected event windows shaded.

A **Prominence distribution...** button (header, next to *Advanced...*) opens a popout with the histogram of every candidate peak's prominence in the smoothed onset density. A draggable slider over the X axis lets you preview the `min_prominence` threshold against the distribution — drop it into the valley between noise and real events, watch the "keeping N / M" count update live, click **Apply** and Tab 5 re-renders against the new threshold automatically. Reset restores the previously-applied value; Cancel/Esc closes without changes.

---

## 1. Inputs

From `plane0/` (after Tabs 3 + 4):
- `F.npy` — `(N_total, T)` (used only for the shape).
- `r0p7_filtered_dff_lowpass.memmap.float32` — `(T, N_kept)` low-pass dF/F.
- `r0p7_filtered_dff_dt.memmap.float32` — `(T, N_kept)` SG-smoothed first derivative.
- `predicted_cell_mask.npy` (or `iscell.npy` fallback) for the keep mask.

---

## 2. Pipeline steps

The render worker (`EventDetectionTab._compute_render_data`) delegates the
entire compute body to
`core.event_detection_run.run_event_detection(plane0, params, *,
figures_dir=None, write_summary=True, progress_cb=None)`. The same call
is what Tab 0's batch worker invokes — there's a single source of truth
for the per-ROI hysteresis + population-event detection.

When `figures_dir` is set, `run_event_detection` saves
`heatmap.png` / `raster.png` / `event_detection.png` (the same three
plots the GUI shows live). When `write_summary` is True (default for
batch; False inside the interactive tab, since the tab triggers its own
`_write_summary` after rendering), it also appends the `EventWindows`,
`EventOnsets`, `RoiEventTimes`, and `EventMonotonicity` sheets to
`<plane0>/calliope_summary.xlsx`. `EventMonotonicity` carries one row
per event (`event_id, n_active, theta_star_deg, rho_obs, p_value,
p_value_fdr, n_shuffles, u_x, u_y`) computed by
`core.spatial.directional_monotonicity_spearman` — the same Spearman-
rank propagation-direction test Tab 8's bottom panel renders.
`p_value_fdr` is the Benjamini-Hochberg-adjusted q-value across the
per-event tests in this recording (compare against your target FDR,
typically 0.05). The correction is applied automatically by
`_compute_event_monotonicity` over events with ≥ 3 active ROIs;
under-populated events get a blank cell.

### Step 0 — (Optional) manual ROI subset

A checkbox + entry box lets the user restrict the analysis to a manual list of Suite2p ROI ids (same syntax as Tab 4: `0,3,5-9`, parsed via `gui_common.parse_manual_roi_spec`).

When **enabled**:
- The list is **intersected with the cell-filter keep mask** — ids outside the keep mask are skipped (logged via `format_roi_indices` for readability).
- The heatmap, raster, and population event detection only see those ROIs.
- The CSV export uses the same subset.

When **disabled** (default), all ROIs that survived the cell-filter keep mask are used, exactly as before.

Bookkeeping arrays:
- `kept_full_idx = np.flatnonzero(mask)` — Suite2p ids of the columns in the `(T, N_kept_full)` filtered memmap.
- `pos_in_memmap` — which of those columns to actually read.
- `kept_idx` — the Suite2p ids that ended up being processed (= `kept_full_idx[pos_in_memmap]`).

The data dict that survives in `self._last_data` carries `kept_idx`, `pos_in_memmap`, and `N_kept_full` so the heatmap CSV export can re-open the full memmap and slice down to the same subset.

### Step 1 — Per-ROI hysteresis onset detection

For each kept ROI `i`:

```python
lp_i = lowpass[:, i]                # T-length float32
dt_i = derivative[:, i]             # T-length float32
z, _, _ = mad_z(dt_i)               # robust z-score of the derivative
onsets = hysteresis_onsets(z, z_enter, z_exit, fps, min_sep_s)
```

#### `mad_z(x)` — robust z-score

```python
med  = median(x)
mad  = median(abs(x − med)) + 1e-12
z    = (x − med) / (1.4826 · mad)
```

The factor `1.4826` makes `1.4826·MAD` an unbiased estimator of `σ` for normal data. This is robust to outliers — a few high-derivative samples (the very things you're trying to detect) don't inflate the threshold the way `(x − mean) / std` would.

#### `hysteresis_onsets(z, z_hi, z_lo, fps, min_sep_s)` — Schmitt trigger

```python
above_hi = z >= z_hi
active = False
onsets = []
for t in range(z.size):
    if not active and above_hi[t]:
        active = True
        onsets.append(t)
    elif active and z[t] <= z_lo:
        active = False
```

Then merge onsets closer than `min_sep_s · fps` frames into the earliest of the run.

**Why hysteresis.** A single threshold flickers — when `z` is hovering around the threshold, every frame produces a new onset. The dual `z_hi=3.5` / `z_lo=1.5` enforces "only count a new onset after the cell has fully de-activated."

**Why on the derivative.** GCaMP fluorescence rises fast (~50–200 ms) and decays slowly (hundreds of ms to seconds). The derivative peaks at the moment of fastest rise, which is the closest you can get to spike onset without doing model-based deconvolution.

`onsets_by_roi[i]` ends up as `frame_idx / fps`, i.e. onset times in **seconds**.

### Step 2 — Display arrays

The heatmap and raster are downsampled to `time_cols_target = 1200` columns for display:

```python
downsample = max(1, T // 1200)
num_cols   = T // downsample
```

For each ROI:
- `lp_ds = lp_i[:num_cols*downsample].reshape(num_cols, downsample).mean(axis=1)`  ← bin-mean of low-pass.
- `(lo, hi) = percentile(lp_ds, [1, 99])`
- `heatmap[i] = clip((lp_ds − lo)/(hi − lo), 0, 1) · 255`  ← uint8 row.
- For the raster: `raster[i, onsets // downsample] = 1` (binary marks).

ROIs are sorted by event count (`-event_counts.argsort()`) so the most active cells go to the top of both panels.

The `event_counts[i]` is just `onsets.size`.

### Step 3 — Population event detection (`utils.detect_event_windows`)

This converts per-ROI onsets into population-level events. The math is in `utils.py` and broken into helpers (`_build_density`, `_detect_density_peaks`, `_boundaries_from_peaks`, `_activation_matrix_from_windows`).

#### 3a — Onset density

```python
duration_s = T / fps
n_bins     = max(1, int(round(duration_s / bin_sec)))
edges      = arange(n_bins + 1) * bin_sec
counts     = histogram( concat(onsets_by_roi), bins=edges ).counts
if normalize_by_num_rois:
    counts /= len(onsets_by_roi)
smooth     = gaussian_filter1d(counts, sigma=smooth_sigma_bins)
```

`bin_sec = 0.025` (25 ms) and `smooth_sigma_bins = 1.5` are tuned for events <0.5 s. Larger bins blur out short events; smaller bins make `find_peaks` choke on every single onset.

`normalize_by_num_rois` makes density an "average onsets per ROI per bin," comparable across recordings with different ROI counts.

#### 3b — Peak detection (`scipy.signal.find_peaks`)

```python
wlen_bins = max(3, round(prominence_wlen_s / bin_sec)) | 1   # NEW: local window
peaks, _ = find_peaks(
    smooth,
    prominence  = min_prominence,        # 0.002 (NEW; OLD was 0.007)
    width       = min_width_bins,        # 1.0 bin (NEW; OLD was 2.0)
    distance    = min_distance_bins,     # 4.0 bins ≈ 100 ms at bin_sec=0.025
    wlen        = wlen_bins,             # 1 s window for prominence calc
)
```

`wlen` is the new addition — *local* prominence so a small peak near a big one isn't unfairly compared to that big one's full descent.

#### 3c — Boundary walking

For each peak `p`:

1. **Baseline trace** — either:
   - `global`: a flat scalar = `percentile(smooth, baseline_percentile=5)`, broadcast.
   - `rolling` (default): per-bin rolling 5th percentile over `baseline_window_s = 5 s`. The window is short on purpose — within-burst baselines drift; a long window over-smooths through the burst.
2. **Noise** — quiet-region MAD: take `smooth` values where `smooth < percentile(smooth, noise_quiet_percentile=40)`, compute their MAD, multiply by `noise_mad_factor = 1.4826`. This estimates "what does dispersion look like when nothing is happening?"
3. **End threshold** = `baseline + end_threshold_k · noise` where `end_threshold_k = 1.5`.
4. **Walk** outward from peak `p`:
   - Left: decrement `t` until `smooth[t] < end_threshold[t]`, or until `(p − t)·bin_sec >= max_walk_duration_s = 2 s`, or until `t == 0`.
   - Right: increment `t` similarly.

#### 3d — Watershed split + duration cap (NEW for short-event mode)

After all `(start_s, end_s)` pairs are walked:

- **Watershed split** (`enable_watershed_split=True`): if `end_s[k] > start_s[k+1]`, find the bin of minimum smoothed density between `peak[k]` and `peak[k+1]` and set both windows' boundary there. Splits two real events that were "swallowed" into one wide window.
- **Hard duration cap** (`max_event_duration_s = 0.5 s`): if `end_s − start_s > 0.5`, clamp it to centred-on-peak ±0.25 s.
- **Symmetric clamp** (`enforce_symmetric_clamp=False` by default): if true, *every* window becomes peak ± 0.25 s, not just the ones that exceed the cap. Useful for stereotyping but loses the natural boundaries.

The `merge_gap_s` parameter is kept for compatibility but is **disabled** when `enable_watershed_split=True` (the new logic supersedes it).

#### 3e — Activation matrix

`A[i, e] = True` iff ROI `i` has at least one onset inside event `e`'s `[start_s, end_s]`.
`first_time[i, e] = min(onsets_i ∩ [start_s, end_s])` or `NaN`.

These are returned alongside `event_windows`.

### Step 4 — Diagnostics + plotting

`detect_event_windows(..., return_diagnostics=True)` returns a dict with `time_centers_s`, `binned_density`, `smoothed_density`, `baseline_trace`, `end_threshold_trace`, `baseline_noise`, `peak_s`, `peak_height`, `mu_s`, `sigma_s` (Gaussian-fit centres/widths if enabled), and `boundary_source_left/right` (which heuristic produced each boundary).

`utils.plot_event_detection(diagnostics, ax)` plots the smoothed density + baseline + end-threshold traces and marks each peak. `utils.shade_event_windows(ax, windows, color='C1', alpha=0.20)` adds the orange shaded rectangles per detected window.

### Step 5 — Disk export

The "Save Data…" buttons next to the heatmap and raster panels write **full-resolution** per-frame CSVs (rows = frames, cols = ROIs):

- Heatmap CSV: pulls actual `float32` values from `r0p7_filtered_dff_lowpass.memmap.float32` (not the 0–255 display version).
- Raster CSV: rebuilds 0/1 onset markers at frame resolution from `onsets_by_roi` (round `onsets_s · fps` to int).

Both prepend a `time_s` column = `frame_index / fps` and a header comment with `fps`, `frame_seconds`, `n_frames`, `n_rois_kept`.

Per-pair summary export goes to `summary_writer.write_events_sheets`, which writes `EventWindows` and `Onsets` sheets to `<rec>_summary.xlsx`.

---

## 3. Outputs

In-memory only (no new on-disk arrays beyond the summary sheets and optional CSVs). Tabs 6 and 7 read the same memmaps and re-derive what they need.

The `summary_writer.write_events_sheets` writes:
- **EventWindows** sheet: columns `event_id, start_s, end_s, duration_s, peak_s` etc.
- **Onsets** sheet: long format — one row per `(roi, onset_s)` pair.

---

## 4. Parameter reference (`EventDetectionParams`)

Defaults are tuned for **<0.5 s epileptiform events**. Every entry below is editable from the *Advanced…* dialog.

| Field | Default | Purpose |
|---|---|---|
| `bin_sec` | 0.025 | Density histogram bin width. |
| `smooth_sigma_bins` | 1.5 | Gaussian smoothing of the density. |
| `normalize_by_num_rois` | True | Normalise to onsets/ROI/bin. |
| `min_prominence` | 0.002 | `find_peaks` prominence floor. Also editable visually via the *Prominence distribution...* popout. |
| `min_width_bins` | 1.0 | Peak min width. |
| `min_distance_bins` | 4.0 | Min separation between peaks (~100 ms). |
| `prominence_wlen_s` | 1.0 | Local window for prominence. |
| `baseline_mode` | `rolling` | `rolling` (per-bin) or `global` (scalar). |
| `baseline_percentile` | 5.0 | Percentile for baseline estimate. |
| `baseline_window_s` | 5.0 | Rolling window length. |
| `noise_quiet_percentile` | 40.0 | Threshold below which density values feed the MAD noise estimate. |
| `noise_mad_factor` | 1.4826 | MAD → σ factor. |
| `end_threshold_k` | 1.5 | `end = baseline + k·noise`. |
| `max_walk_duration_s` | 2.0 | Cap on one-sided walk. |
| `max_event_duration_s` | 0.5 | Hard cap on **final** event duration. |
| `enable_watershed_split` | True | Split overlapping windows at the local minimum between peaks. |
| `enforce_symmetric_clamp` | False | Force every window to peak ± max/2. |
| `merge_gap_s` | 0.0 | Legacy overlap-merge gap; ignored when watershed-split is on. |
| `use_gaussian_boundary` | False | Fit a Gaussian to each peak and use its quantile cut as the boundary. |
| `gaussian_quantile` | 0.99 | Quantile of the Gaussian fit used to set the boundary. |
| `gaussian_fit_pad_s` | 0.5 | Padding around each peak included in the Gaussian fit. |
| `gaussian_min_sigma_s` | 0.05 | Lower bound on the Gaussian fit's sigma. |

The last five rows (the legacy merge gap and the Gaussian-fit refinement) are inactive in the default short-event configuration but remain editable so a user can revert to the older boundary heuristics by flipping `use_gaussian_boundary` (and disabling `enable_watershed_split`) without leaving the GUI.

Per-ROI hysteresis params (`PARAM_SPEC`):
- `z_enter = 3.5`, `z_exit = 1.5`, `min_sep_s = 0.1`.

Display: `time_cols_target = 1200` columns.

Manual subset (UI-only, not in `PARAM_SPEC`):
- `manual_subset_var` — checkbox; when on, the entry box is parsed as a Suite2p ROI id spec and intersected with the keep mask.
- `manual_roi_var` — text entry; format `0,3,5-9` (same as Tab 4).

---

## 5. Re-implementation checklist

0. **(Optional) manual ROI subset:** if the user supplies a Suite2p id spec, parse it (`parse_manual_roi_spec`), intersect with the keep mask (`np.flatnonzero(mask)`), and build a `pos_in_memmap` array of column indices into the `(T, N_kept_full)` filtered memmap. All loops below iterate over `pos_in_memmap` instead of `range(N_kept_full)`.
1. **Per-ROI onsets:**
   - `mad_z(x) = (x − median(x)) / (1.4826 · median(abs(x − median(x))))`.
   - Schmitt trigger on the derivative's z-score with thresholds `(z_hi, z_lo)`; merge consecutive onsets closer than `min_sep_s · fps` frames.
2. **Population density:**
   - Histogram all onsets across ROIs into bins of `bin_sec`.
   - Optionally normalise by ROI count.
   - `gaussian_filter1d(sigma=smooth_sigma_bins)`.
3. **Peak detection:** `scipy.signal.find_peaks` with `prominence=min_prominence`, `width=min_width_bins`, `distance=min_distance_bins`, `wlen=round(prominence_wlen_s/bin_sec)|1`.
4. **Baseline:** rolling 5th percentile over `baseline_window_s` (or scalar global). **Noise:** MAD of the smooth-density values below the 40th percentile, scaled by `1.4826`. **End threshold:** `baseline + end_threshold_k · noise`.
5. **Walk** outward from each peak until the trace falls below `end_threshold` *or* you hit `max_walk_duration_s`.
6. **Watershed split** overlapping windows at the bin of minimum smoothed density between consecutive peaks.
7. **Hard cap** any final window > `max_event_duration_s` to peak ± `max_event_duration_s/2`.
8. **Activation matrix:** for each event, mark every ROI that has at least one onset inside `[start_s, end_s]`; record the earliest such onset.
9. The display heatmap is per-ROI 1st/99th-percentile-normalised `lp_ds` with rows sorted by event count (descending). The raster is binary onsets at downsampled bin resolution.

The full annotated reference is in `core/utils.py` lines ~250–820.


## UI affordances

Tab 5 inherits the global customtkinter dark theme from `pipeline_gui`.

- **Per-panel resize grips.** All three stacked matplotlib panels (1. heatmap, 2. event raster, 3. population event detection) carry a draggable handle below them. Drag any grip to grow that panel — the scrollable tab body absorbs the extra height. The other panels stay at their current size (no PanedWindow sash redistribution).
- **Popouts.**
  - **Prominence distribution** (`ProminencePopout`) — opens from the "Prominence distribution..." button after a render completes; resizable Toplevel with a histogram of candidate-peak prominences and a slider for picking `min_prominence` interactively.
- **Onset source** + **Manual ROI subset** rows let you pick between derivative / Suite2p `spks` and optionally restrict the heatmap + raster to a user-supplied ROI list.
