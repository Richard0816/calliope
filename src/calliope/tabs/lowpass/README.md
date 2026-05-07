# Tab 4 — Low-pass Filter

**Goal.** Pick a low-pass cutoff for the dF/F traces, see its effect live, and write per-cell low-pass and derivative arrays at the chosen cutoff.

This tab is interactive: a slider drives the cutoff, three panels (FFT, raw dF/F, low-pass dF/F) update in real time, and a single button commits the current cutoff to disk.

---

## 1. Inputs

From `plane0/`:
- `r0p7_filtered_dff.memmap.float32` of shape `(T, N_kept)` — preferred.
- Falls back to `r0p7_dff.memmap.float32` of shape `(T, N_total)` masked by `predicted_cell_mask.npy` / `iscell.npy` if the filtered memmap is missing.
- `F.npy` for `(N_total, T)` shape lookup.
- `predicted_cell_mask.npy` (preferred) or `iscell.npy` for the keep mask.
- `predicted_cell_prob.npy` (only used by the "best-scoring ROI" trace source).

The tab subscribes to the AppState `plane0` broadcast so it auto-loads when Tab 3 finishes; "Reload from folder…" lets the user point at any `plane0`.

`fps` is resolved via `utils.get_fps_from_notes(plane0)` (default fallback 15.07 Hz). See `utils.py` for the YYYY-MM-DD_NNN folder-name convention it expects.

---

## 2. Pipeline steps

### Step 1 — Choose a representative trace

The "Trace source" radio + entry chooses what to compute the FFT and live previews on:

- **Mean across kept ROIs**: `mean(dff, axis=1)`
- **Median across kept ROIs**: `median(dff, axis=1)`
- **Best-scoring ROI**: ROI with the largest `predicted_cell_prob`.
- **Manual ROI(s)**: user-supplied ROI ids/ranges (e.g. `0,3,5-9`) using `gui_common.parse_manual_roi_spec`. If multiple are given, aggregated by `mean` or `median` (combobox).

Manual ids are **always** Suite2p indices into the full `(T, N_total)` `r0p7_dff.memmap.float32`, irrespective of whether they survive the cell-filter mask. Mean/median/best aggregations operate on the kept-only `(T, N_kept)` memmap.

### Step 2 — FFT power spectrum

Computed once per trace via `fft_all_rois.compute_fft(trace, fps)`:

```python
N = trace.size
yf = np.fft.rfft(trace - mean(trace))
xf = np.fft.rfftfreq(N, d=1.0/fps)
power = abs(yf)**2 / N
```

(Look up `fft_all_rois.py` in the legacy `Calcium_imaging_suite2p/` repo for the exact code.) The plot is `semilogy(xf[1:], maximum(power[1:], 1e-12))` with `xlim = (0, min(fps/2, 15))`. A red dashed `axvline` marks the current cutoff and updates live as the slider moves.

**Why log y.** Calcium signal lives mostly below ~3 Hz at typical GCaMP kinetics; shot noise is roughly white above that. On a log axis the elbow between signal and noise is the cutoff you usually want.

### Step 3 — Causal low-pass filter (`utils.lowpass_causal_1d`)

Order-2 (configurable) Butterworth realised as second-order sections (SOS):

```python
nyq    = fps / 2
cutoff = clip(cutoff_hz, 1e-4, 0.95*nyq)            # avoid Nyquist edge
sos    = butter(order, cutoff/nyq, btype='low', output='sos')
zi     = zeros((sos.shape[0], 2))
zi[:, 0] = x[0]; zi[:, 1] = x[0]                    # init state to first sample
y, zf  = sosfilt(sos, x, zi=zi)                     # one-direction (causal) filter
```

Two important choices:

1. **Causal**, not zero-phase. We do **not** use `filtfilt` because we want the filtered output to depend only on past samples (no leakage from future activity into earlier timepoints). The downstream derivative + onset detection assumes the smoothed signal is causal.
2. **State initialisation to the first sample.** A zero-initial state would inject a startup transient as the filter "ramped up" from 0 to the dF/F mean. Setting `zi` to repeated copies of `x[0]` makes the filter start at steady state.

The same filter, with the same `sos`, is reused per ROI in the worker — `sos` is computed once and cached.

### Step 4 — Savitzky-Golay first derivative (`utils.sg_first_derivative_1d`)

```python
win = max(3, int((win_ms/1000) * fps) | 1)         # nearest odd
y_dot = savgol_filter(x, window_length=win, polyorder=poly,
                      deriv=1, delta=1.0/fps)
```

SG fits a low-order polynomial (default order 2) in a sliding window and returns the polynomial's first derivative at the centre. Compared to a finite difference `(x[t+1] − x[t]) · fps`, this is far more noise-tolerant — the polynomial fit averages the local samples instead of differencing two adjacent (noisy) ones.

`delta=1/fps` makes the output's units `dF/F per second`.

If the window is wider than the trace, falls back to a forward-difference gradient.

### Step 5 — Compute (write to disk)

`_run_compute(plane0, T, N, fps, cutoff)`:

```python
src = memmap('r0p7_filtered_dff.memmap.float32', shape=(T, N))  # input
lp_mm = memmap('r0p7_filtered_dff_lowpass.memmap.float32', shape=(T, N), mode='w+')
dt_mm = memmap('r0p7_filtered_dff_dt.memmap.float32',     shape=(T, N), mode='w+')

sos = None
for i in range(N):
    trace = src[:, i].astype(float32)
    lp, _, sos = lowpass_causal_1d(trace, fps, cutoff_hz=cutoff,
                                   order=order, sos=sos)
    dt = sg_first_derivative_1d(lp, fps, win_ms=sg_win_ms, poly=sg_poly)
    lp_mm[:, i] = lp
    dt_mm[:, i] = dt
```

Both memmaps are flushed at the end. The tab broadcasts `state.set_lowpass_ready(plane0)` which Tabs 5–7 listen for.

---

## 3. Outputs (in `plane0/`)

| File | Shape | dtype |
|---|---|---|
| `r0p7_filtered_dff_lowpass.memmap.float32` | `(T, N_kept)` | float32 |
| `r0p7_filtered_dff_dt.memmap.float32` | `(T, N_kept)` | float32 |

These overwrite the defaults that Tab 3 wrote at 1 Hz cutoff, so re-running Tab 4 at, say, 0.5 Hz changes Tab 5's onsets accordingly.

---

## 4. UI behaviour

- **Slider** spans `[CUTOFF_MIN, CUTOFF_MAX]` (default `0.01 – 10 Hz`, configurable via Advanced). The default starting cutoff is 1.0 Hz.
- **Slider debounce.** `_on_slider` uses `after(80, self._apply_cutoff)` so dragging fires at most every 80 ms, otherwise re-rendering would lag behind the slider.
- **Cutoff entry box.** `_on_entry` clamps to bounds and snaps the slider to the typed value; Enter applies it.
- **Source change.** Reloads the chosen trace, recomputes FFT, redraws all three panels.
- **Compute** button writes the memmaps in a worker thread and pushes status updates through a queue (`_compute_queue`).

---

## 5. Parameters (`PARAM_SPEC`)

| Param | Default |
|---|---|
| `filter_order` | 2 (Butterworth order) |
| `sg_win_ms` | 333 ms |
| `sg_poly` | 2 |
| `cutoff_min` | 0.01 Hz (slider lower bound) |
| `cutoff_max` | 10.0 Hz (slider upper bound) |

---

## 6. Re-implementation checklist

1. `scipy.signal.butter(order, cutoff/nyq, btype='low', output='sos')` + `scipy.signal.sosfilt` for the filter.
2. Initialise `zi = ones((sos.shape[0], 2)) * x[0]` to suppress the startup transient.
3. `scipy.signal.savgol_filter(x, window_length=odd_window, polyorder=2, deriv=1, delta=1/fps)` for the derivative.
4. An FFT power-spectrum routine: subtract the mean, `rfft`, square, normalise by `T`, plot on `semilogy`.
5. A live slider that calls (1) per drag and updates a Matplotlib axvline + the low-pass plot. Debounce via `after(80, …)` to avoid melting the GUI thread.
6. A worker that re-runs the SOS filter + SG derivative once per ROI (re-using the cached `sos`) and writes two `(T, N)` `float32` memmaps with the standardised filenames.
7. The trace-source rules: aggregate over the kept-only memmap for mean/median/best, but reach into the full `(T, N_total)` memmap when the user passes Suite2p ROI ids manually.
