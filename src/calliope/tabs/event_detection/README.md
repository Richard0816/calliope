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

#### `refine_onsets(onsets, lp, fps, min_rise_dff, backtrack_s)` — foot backtrack + optional rise gate

The Schmitt trigger fires partway *up* the rise (where the derivative robust-z
crosses `z_enter`), not at its foot. `refine_onsets` (`utils.py:458`, called at
`event_detection_run.py:366`) walks each crossing back along the low-pass dF/F
to the local minimum that starts the rise, so the reported frame lands at event
**onset**. Two crossings on one rise collapse to a single foot.

If `min_rise_dff > 0` it additionally drops onsets whose foot→peak rise on the
low-pass dF/F is below `min_rise_dff` **absolute dF/F**. The floor is absolute
on purpose: a per-ROI robust-z floor cannot separate a flat noisy ROI from a
real one (the trigger already pre-selects the steepest excursions, which sit
several robust-σ above their feet on noise and signal alike), whereas an
absolute dF/F floor removes noise-*amplitude* excursions everywhere — which is
what zeroes out flat, low-SNR ROIs. It is **off by default** (a sensible value
is recording/indicator dependent); the principal defense against chance onsets
remains the population-level null-prominence floor (§ population peaks below).

`onsets_by_roi[i]` ends up as `frame_idx / fps`, i.e. onset times in **seconds**.

### Step 2 — Display arrays

The heatmap and raster are downsampled to `time_cols_target = 1200` columns for display:

```python
downsample = max(1, T // 1200)
num_cols   = T // downsample
```

For each ROI:
- `lp_ds = lp_i[:num_cols*downsample].reshape(num_cols, downsample).mean(axis=1)`  ← bin-mean of low-pass.
- `(lo, hi) = percentile(lp_ds, [1, 99.5])`  ← shared `utils.DISPLAY_CLIP_{LOW,HIGH}_PCT`
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

`bin_sec = 0.05` (50 ms) and `smooth_sigma_bins = 1.5` are tuned for short (sub-second) events. Larger bins blur out short events; smaller bins make `find_peaks` choke on every single onset.

`normalize_by_num_rois` makes density an "average onsets per ROI per bin," comparable across recordings with different ROI counts.

#### 3b — Peak detection (`scipy.signal.find_peaks`)

```python
wlen_bins = max(3, round(prominence_wlen_s / bin_sec)) | 1   # NEW: local window
peaks, _ = find_peaks(
    smooth,
    prominence  = min_prominence,        # 0.002 (NEW; OLD was 0.007)
    width       = min_width_bins,        # 1.0 bin (NEW; OLD was 2.0)
    distance    = min_distance_bins,     # 4.0 bins ≈ 200 ms at bin_sec=0.05
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
- **Hard duration cap** (`max_event_duration_s = 1.3 s`): if `end_s − start_s > 1.3`, clamp it to centred-on-peak ±0.65 s.
- **Symmetric clamp** (`enforce_symmetric_clamp=False` by default): if true, *every* window becomes peak ± `max_event_duration_s/2` (0.65 s at the default cap), not just the ones that exceed the cap. Useful for stereotyping but loses the natural boundaries.

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

## 4. Advanced settings (`Advanced…` dialog)

Every knob below lives in the 32-entry `PARAM_SPEC` at `tab.py:136`. The
*Advanced…* button opens a generated dialog (`gui_common.open_advanced`);
defaults are primed via `spec_defaults(PARAM_SPEC)` into `self._params`.
Twenty-six of these map 1:1 to `core.utils.EventDetectionParams` (the
`_EVENT_DETECTION_FIELDS` tuple, `event_detection_run.py:44`); the other six
(`z_enter`, `z_exit`, `min_sep_s`, `min_rise_dff`, `onset_backtrack_s`,
`time_cols_target`) are consumed directly by the run worker. **Defaults below are exactly the GUI `PARAM_SPEC` values**,
which `_build_event_params` (`event_detection_run.py:192`) writes over the
dataclass defaults before detection — so they are the source of truth. In a
couple of cases the GUI value differs from the *live* `EventDetectionParams`
default: GUI `bin_sec=0.05` vs dataclass `0.025`, and GUI
`max_event_duration_s=1.3` vs dataclass `0.5`. The dataclass keeps the
pre-short-event values commented next to each field, but the numbers below
(the live GUI defaults) are what actually runs.

Defaults are tuned for **short epileptiform events** (sub-second). Headings
match the `PARAM_SPEC` `group` field.

### Per-ROI hysteresis

The Schmitt-trigger onset detector. Per kept ROI the worker computes
`z = mad_z(derivative)` (`utils.py:380`) then
`hysteresis_onsets(z, z_enter, z_exit, fps, min_sep_s)`
(`utils.py:405`, called at `event_detection_run.py:359`). Operates in robust-z
units on the SG derivative, so values are comparable across cells regardless of
raw dF/F scale.

- **Hysteresis enter (`z_enter`) — default `3.5`.**
  - *What it does:* the up-crossing threshold. An onset is emitted the frame `z`
    first rises ≥ `z_enter` while the cell is "inactive". Units = robust σ
    (`1.4826·MAD` of the derivative). Sensible range ~2.5–5.
  - *What it means to you:* the main sensitivity knob for per-cell firing.
    Lower it to catch weaker transients (more onsets, more false positives from
    noise); raise it to keep only sharp, confident rises (fewer onsets). Must
    stay above `z_exit` or the trigger never re-arms.
- **Hysteresis exit (`z_exit`) — default `1.5`.**
  - *What it does:* the down-crossing threshold. After an onset the cell stays
    "active" (suppressing new onsets) until `z` falls ≤ `z_exit`. The gap
    between enter and exit is what prevents flicker.
  - *What it means to you:* controls how fully a cell must de-activate before it
    can fire again. Raise it (toward `z_enter`) to re-arm sooner — more onsets on
    sustained activity, risk of double-counting one event. Lower it to demand a
    full return to baseline — fewer, cleaner onsets, but a cell riding an
    elevated baseline may never re-arm.
- **Min separation (`min_sep_s`) — default `0.1`.**
  - *What it does:* post-merge dead-time, seconds. Onsets within
    `min_sep_s · fps` frames of the previously kept onset are dropped
    (`utils.py:446`). `0` disables merging.
  - *What it means to you:* the refractory floor on a single cell. Raise it to
    collapse onset bursts into one event (fewer onsets); lower it toward 0 to
    keep every up-crossing. Interacts with `z_exit`: both gate how fast one cell
    can re-fire.
- **Min onset rise (`min_rise_dff`) — default `0.0` (off).**
  - *What it does:* after the foot backtrack, drops any onset whose foot→peak
    rise on the low-pass dF/F is below this many **absolute dF/F** units
    (`refine_onsets`, `utils.py:458`). `0` disables the gate (backtrack only).
  - *What it means to you:* an opt-in floor that removes noise-amplitude onsets
    on flat / low-SNR cells. It is absolute (not robust-z) by design — a per-ROI
    normalised floor can't tell a phantom cell from a real one. A typical
    starting value is ~`0.2`, but the right number depends on your dF/F scale
    and indicator, so it ships off. Phantom suppression is primarily the job of
    the population null-prominence floor; this is a per-cell sanity gate.
- **Onset backtrack (`onset_backtrack_s`) — default `0.0` (auto).**
  - *What it does:* cap (seconds) on how far `refine_onsets` walks an onset back
    to the foot of its rise (and forward to the peak for the rise gate). The
    foot is the `argmin` over the look-back, but the walk never crosses a sample
    already as high as the onset itself, so it can't reach back into a separate
    earlier bump. `0` = **auto**: `max(tau, 0.8/cutoff_hz)` per recording.
  - *What it means to you:* the foot lives on the **low-pass** dF/F, so the rise
    timescale is the *slower* of the indicator decay `tau` and the low-pass
    period `1/cutoff` — not the (sub-second) spike. Fast indicator → the
    low-pass floor wins (jGCaMP8m tau 0.25 s at 1 Hz → 0.8 s); slow indicator →
    `tau` wins (GCaMP6s → 1.5 s). Set a positive value to override.

### Display

- **Heatmap time columns (`time_cols_target`) — default `1200`.**
  - *What it does:* target column count for the downsampled heatmap and raster.
    The worker sets `downsample = max(1, T // time_cols_target)` and bin-means
    the low-pass into `T // downsample` columns (`event_detection_run.py:317`).
    Display-only — does **not** touch detection or the full-resolution CSV
    export.
  - *What it means to you:* purely cosmetic/performance. Raise it for finer
    time resolution on long recordings (bigger figure, slower redraw); lower it
    if the panels feel sluggish. Detected events and onset times are unchanged.

### Population events — density

The smoothed onset-density curve (`_build_density`, `utils.py:1185`) that
peak-picking runs on.

- **Density bin (`bin_sec`) — default `0.05`.**
  - *What it does:* histogram bin width in seconds for pooled onsets across all
    ROIs. Sets the density's time resolution; also the unit for every
    `*_bins`/`wlen` conversion downstream.
  - *What it means to you:* the master time scale. Smaller bins resolve closely
    spaced events but make `find_peaks` choke on single-onset spikes; larger
    bins blur short events together. Changing it implicitly rescales
    `min_width_bins`, `min_distance_bins`, and `prominence_wlen_s` (all defined
    relative to seconds via `/ bin_sec`).
- **Smoothing sigma (`smooth_sigma_bins`) — default `1.5`.**
  - *What it does:* Gaussian-filter σ (in bins) applied to the binned density
    (`gaussian_filter1d`, `mode="nearest"`). Turns the spiky histogram into a
    continuous curve.
  - *What it means to you:* the smoothness/noise trade-off. Raise it for a
    cleaner curve and fewer spurious peaks (but real short events smear and can
    merge); lower it for sharper peaks at the cost of jitter that `find_peaks`
    may mistake for events.
- **Normalize by ROI count (`normalize_by_num_rois`) — default `True`.**
  - *What it does:* divides counts by the number of ROIs, turning the density
    into "fraction of cells firing per bin" (`utils.py:1234`).
  - *What it means to you:* keep on so a fixed `min_prominence` means the same
    thing across recordings with different cell counts. Turn off only if you
    deliberately want raw counts — then prominence/baseline thresholds become
    recording-size-dependent.

### Population events — peaks

Peak detection on the smoothed density (`_detect_density_peaks` →
`scipy.signal.find_peaks`, `utils.py:1411`). The prominence floor is either
auto-derived from the per-recording circular-shift null or fixed — see below.

- **Auto min prominence (`auto_min_prominence`) — default `True`.**
  - *What it does:* when on, the effective prominence floor is the
    `auto_min_prominence_percentile` of that recording's circular-shift null
    (`null_prominence_percentiles` → `circular_shift_null_prominences`,
    `utils.py:1319`; applied at `utils.py:1008`). Each ROI's onset train is
    independently circular-shifted to destroy cross-cell coincidence, the
    density is rebuilt with the *same* pipeline, and every candidate peak's
    prominence is pooled. A degenerate/empty null falls back to the fixed
    `min_prominence`. The value actually used is reported in
    `diagnostics["min_prominence_used"]`.
  - *What it means to you:* the recommended default — the floor tracks each
    recording's own coincidence-noise level instead of one global guess. It
    errs slightly permissive (borderline events survive to curation rather than
    being silently dropped). Untick it (or apply a value in the *Prominence
    distribution…* popout, which auto-unchecks it) to pin the fixed
    `min_prominence` instead. Cost: it runs `auto_min_prominence_n_shuffles`
    shuffles per render.
- **Auto prominence percentile (`auto_min_prominence_percentile`) — default `99.0`.**
  - *What it does:* which percentile of the null prominence distribution becomes
    the floor when auto is on.
  - *What it means to you:* fitted to ~99 against nine hand-tuned recordings
    (null p99 matched the manual median within a few percent; p95 ran ~50 % too
    low). Lower it (toward 95) to admit more borderline events; raise it for a
    stricter floor. Useful range ~95–99.
- **Auto prominence shuffles (`auto_min_prominence_n_shuffles`) — default `200`.**
  - *What it does:* number of circular-shift iterations building the null.
  - *What it means to you:* more shuffles = steadier floor, slower render (a few
    seconds at 200). Drop it (e.g. 50–100) for faster interactive iteration; the
    floor gets noisier between renders.
- **Auto prominence seed (`auto_min_prominence_seed`) — default `0`.**
  - *What it does:* RNG seed for the shuffles (`np.random.default_rng(seed)`).
  - *What it means to you:* fix it for reproducible event sets across re-renders
    and batch reruns; change it only to check the floor's run-to-run stability.
- **Min peak prominence (`min_prominence`) — default `0.002`.**
  - *What it does:* the fixed `find_peaks` prominence floor on the smoothed
    density. **Used only when `auto_min_prominence` is off.** Also the value the
    *Prominence distribution…* slider edits (applying there turns auto off).
  - *What it means to you:* with auto off, this is the single most important
    sensitivity knob — drop it into the valley between the noise hump and the
    real-event lobe in the prominence histogram. Raising it = fewer, more
    confident events; lowering it = more events including noise. With auto on it
    is inert (and only the fallback if the null degenerates).
- **Min peak width (`min_width_bins`) — default `1.0`.**
  - *What it does:* `find_peaks` minimum width at half-prominence, in density
    bins.
  - *What it means to you:* rejects single-bin spikes. At default `bin_sec=0.05`
    one bin = 50 ms. Raise it to demand broader population bumps (fewer events);
    leave low for fine bins where real peaks are narrow.
- **Min peak separation (`min_distance_bins`) — default `4.0`.**
  - *What it does:* minimum bin distance between accepted peaks (`distance=` in
    `find_peaks`); the lower of two too-close peaks is dropped. ≈ 200 ms at
    `bin_sec=0.05`.
  - *What it means to you:* the population-level refractory. Raise it to stop one
    event being split into two; lower it to resolve rapid back-to-back bursts.
    Scales with `bin_sec`.
- **Prominence window (`prominence_wlen_s`) — default `1.0`.**
  - *What it does:* local window (seconds → odd bin count via
    `prominence_wlen_s/bin_sec | 1`, `utils.py:1028`) for the `wlen=` prominence
    calculation, so a small peak next to a big one is measured against the
    nearby valley, not the big peak's full descent.
  - *What it means to you:* raise it toward whole-trace behaviour if you want
    global prominence; lower it so closely spaced events of differing height are
    each judged on local contrast. Mostly leave at 1 s.

### Population events — baseline

Baseline + noise estimation that sets the boundary-walk threshold
(`_estimate_rolling_baseline` / `_estimate_noise_from_quiet`, `utils.py:1473`).

- **Baseline mode (`baseline_mode`) — default `rolling`** (choices: `rolling`, `global`).
  - *What it does:* `rolling` = per-bin `baseline_percentile` over a
    `baseline_window_s` window (`percentile_filter`); `global` = one scalar
    percentile of the whole density, broadcast.
  - *What it means to you:* keep `rolling` when baselines drift within a burst
    (events sitting on an elevated floor still get walked correctly). Switch to
    `global` for short, stable recordings where a flat baseline is cleaner and
    cheaper.
- **Baseline percentile (`baseline_percentile`) — default `5.0`.**
  - *What it does:* the percentile taken as "baseline" (rolling per-window or
    global).
  - *What it means to you:* lower = baseline hugs the troughs (wider event
    windows, since the trace stays above threshold longer); higher = baseline
    rides up into activity (tighter windows). 5 keeps it near the quiet floor.
- **Rolling window (`baseline_window_s`) — default `5.0`.**
  - *What it does:* window length (seconds) for the rolling baseline; converted
    to bins via `1/bin_sec` and forced odd. Ignored in `global` mode.
  - *What it means to you:* short windows track within-burst drift (the intent
    here); too short and the baseline climbs into events (truncating them); too
    long and it over-smooths through bursts (windows balloon).
- **Quiet percentile (noise) (`noise_quiet_percentile`) — default `40.0`.**
  - *What it does:* only density residuals below this percentile feed the MAD
    noise estimate (`utils.py:1512`) — the "what does dispersion look like when
    nothing is happening" set.
  - *What it means to you:* lower it if active bins are leaking into the noise
    estimate (inflating it, shrinking windows); raise it if too few quiet bins
    make the estimate unstable. 40 is a lenient cutoff that still excludes
    obvious peaks.
- **MAD → sigma factor (`noise_mad_factor`) — default `1.4826`.**
  - *What it does:* scales the quiet-region MAD toward a Gaussian σ.
  - *What it means to you:* the textbook constant — leave it. It only sets the
    absolute scale of `noise`, which `end_threshold_k` already tunes, so adjust
    `end_threshold_k` instead.

### Population events — boundaries

Walking each peak outward to a window, then splitting/clamping
(`_boundaries_from_peaks`, `utils.py:1647`).

- **End threshold k (`end_threshold_k`) — default `1.5`.**
  - *What it does:* boundary level = `baseline + k · noise`
    (`utils.py:1695`); the walk stops when the density drops below it.
  - *What it means to you:* the width knob for individual events. Lower it →
    walk continues further down toward baseline → wider windows; raise it →
    windows clip closer to the peak. Pairs with `baseline_percentile`/noise.
- **Max walk duration (`max_walk_duration_s`) — default `2.0`.**
  - *What it does:* hard cap (seconds) on how far each one-sided walk may travel
    from the peak before stopping (`walk_steps`, `utils.py:1715`). A search
    radius, not the final duration.
  - *What it means to you:* a safety rail so a never-crossing threshold can't
    run a window to the recording edge. Keep it comfortably above
    `max_event_duration_s`; lower it only if windows occasionally over-extend
    before the hard cap kicks in.
- **Max event duration (`max_event_duration_s`) — default `1.3`.**
  - *What it does:* the physiological hard cap on the **final** window. Any
    window longer than this is clamped to peak ± `max_event_duration_s/2`
    (`utils.py:1810`; 0.65 s each side at the default).
  - *What it means to you:* the dominant duration constraint — set it to the
    longest event you consider one event. Too small and genuine long events get
    truncated; too large and merged/runaway windows survive. Interacts with
    `enforce_symmetric_clamp`.
- **Watershed-split overlaps (`enable_watershed_split`) — default `True`.**
  - *What it does:* when two walked windows overlap, cut both at the
    minimum-density valley between their peaks (`utils.py:1748`), so close events
    stay separate. When on, the legacy `merge_gap_s` path is disabled.
  - *What it means to you:* keep on for distinct back-to-back epileptiform
    events. Turn off only to fall back to the old merge-overlapping behaviour
    (then `merge_gap_s` applies).
- **Force symmetric clamp (`enforce_symmetric_clamp`) — default `False`.**
  - *What it does:* when on, **every** window is forced to peak ±
    `max_event_duration_s/2`, not just the ones exceeding the cap
    (`utils.py:1816`).
  - *What it means to you:* turn on to stereotype all events to a fixed width
    (handy for averaging/alignment) at the cost of each event's natural
    boundaries. Leave off to keep walked widths up to the cap.
- **Merge gap (`merge_gap_s`) — default `0.0`.**
  - *What it does:* legacy overlap-merge gap (seconds). **Ignored while
    `enable_watershed_split` is on** (the watershed logic supersedes it,
    `utils.py:1728`).
  - *What it means to you:* only relevant if you disable watershed split — then
    windows closer than this gap merge. Left at 0 / inert by default.

### Population events — gaussian fit

Optional boundary refinement (disabled by default; the watershed + hard cap
replace it). Kept editable for reverting to the older heuristic.

- **Use Gaussian-fit boundary (`use_gaussian_boundary`) — default `False`.**
  - *What it does:* when on, fits a moments-based Gaussian to each peak and uses
    its `gaussian_quantile` cut as the boundary, intersected with the walked
    window (`utils.py:1775`).
  - *What it means to you:* an alternative to baseline-walk boundaries for
    smooth, well-isolated peaks. Off in short-event mode; flip on (and consider
    disabling watershed split) only to reproduce the legacy boundary behaviour.
    The next three knobs do nothing while this is off.
- **Gaussian quantile (`gaussian_quantile`) — default `0.99`.**
  - *What it does:* the Gaussian quantile (→ z-score) defining the cut; 0.99 ≈
    ±2.33 σ.
  - *What it means to you:* higher = wider windows (more of the tail);
    lower = tighter. Only active with `use_gaussian_boundary`.
- **Gaussian fit pad (`gaussian_fit_pad_s`) — default `0.5`.**
  - *What it does:* seconds of padding around each peak included in the fit
    window.
  - *What it means to you:* more pad gives the fit more shoulder to estimate σ
    from (steadier on broad peaks, but risks pulling in neighbours). Only active
    with `use_gaussian_boundary`.
- **Gaussian min sigma (`gaussian_min_sigma_s`) — default `0.05`.**
  - *What it does:* lower bound (seconds) on the fitted σ, so a near-degenerate
    fit can't collapse to a zero-width window.
  - *What it means to you:* raise it if Gaussian-mode windows come out
    implausibly narrow. Only active with `use_gaussian_boundary`.

**Auto prominence floor (summary).** By default the prominence floor is not a
single fixed number: with `auto_min_prominence` on, detection derives it for
*each recording* from that recording's circular-shift null (p99 by default). It
is a deliberately weak, per-recording estimate that tracks the recording's own
noise level and errs slightly permissive. Fully overridable — untick
`auto_min_prominence` (or apply a value in the *Prominence distribution…*
popout) to fall back to fixed `min_prominence`, or nudge
`auto_min_prominence_percentile` within ~95–99. The `min_prominence_used` value
is recorded in the diagnostics dict.

The Gaussian-fit group and `merge_gap_s` are inactive in the default
short-event configuration but stay editable so you can revert to the older
boundary heuristics (set `use_gaussian_boundary` on and/or `enable_watershed_split`
off) without leaving the GUI.

**Manual subset (UI-only, not in `PARAM_SPEC`).** A header checkbox
(`manual_subset_var`) + text entry (`manual_roi_var`, format `0,3,5-9`) restrict
the heatmap / raster / detection to a Suite2p ROI id list intersected with the
keep mask. The **Onset source** radios (`onset_source_var`) pick `derivative`
(default) vs Suite2p `spks`. These are snapshotted by `_on_render`, not edited
in the *Advanced…* dialog.

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
3. **Peak detection:** `scipy.signal.find_peaks` with `prominence=`*effective floor*, `width=min_width_bins`, `distance=min_distance_bins`, `wlen=round(prominence_wlen_s/bin_sec)|1`. The effective floor is the circular-shift null percentile when `auto_min_prominence` is on (default), else the fixed `min_prominence`; the value actually used is reported in `diagnostics["min_prominence_used"]`.
4. **Baseline:** rolling 5th percentile over `baseline_window_s` (or scalar global). **Noise:** MAD of the smooth-density values below the 40th percentile, scaled by `1.4826`. **End threshold:** `baseline + end_threshold_k · noise`.
5. **Walk** outward from each peak until the trace falls below `end_threshold` *or* you hit `max_walk_duration_s`.
6. **Watershed split** overlapping windows at the bin of minimum smoothed density between consecutive peaks.
7. **Hard cap** any final window > `max_event_duration_s` to peak ± `max_event_duration_s/2`.
8. **Activation matrix:** for each event, mark every ROI that has at least one onset inside `[start_s, end_s]`; record the earliest such onset.
9. The display heatmap is per-ROI 1st/99.5th-percentile-normalised `lp_ds` with rows sorted by event count (descending). The raster is binary onsets at downsampled bin resolution.

The full annotated reference is in `core/utils.py` lines ~250–820.


## UI affordances

Tab 5 inherits the global customtkinter dark theme from `pipeline_gui`.

- **Per-panel resize grips.** All three stacked matplotlib panels (1. heatmap, 2. event raster, 3. population event detection) carry a draggable handle below them. Drag any grip to grow that panel — the scrollable tab body absorbs the extra height. The other panels stay at their current size (no PanedWindow sash redistribution).
- **Popouts.**
  - **Prominence distribution** (`ProminencePopout`) — opens from the "Prominence distribution..." button after a render completes; resizable Toplevel with a histogram of candidate-peak prominences and a slider for picking `min_prominence` interactively. It also overlays the **circular-shift null floor** (p95 dotted, p99 dash-dot vertical reference lines): the null independently time-shifts each ROI's onset train so any cross-cell coincidence is pure chance, and `min_prominence` should sit at/above that floor. The null is computed off-thread (`utils.circular_shift_null_prominences`, 200 shuffles) so the popout opens instantly and the lines fill in a few seconds later (a "null floor: computing…" note shows meanwhile); the count label then reports the threshold as a `×` multiple of null p99. By default the same null now also *drives* the floor automatically (`auto_min_prominence`, p99) rather than only being shown — see the Auto prominence floor note in §4. The popout is the manual escape hatch: **applying a value here turns `auto_min_prominence` off** so your pick takes effect. Same null engine as `scripts/null_prominence_audit.py`.
- **Onset source** + **Manual ROI subset** rows let you pick between derivative / Suite2p `spks` and optionally restrict the heatmap + raster to a user-supplied ROI list.
