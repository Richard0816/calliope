# Tab 4 — Low-pass Filter

**Goal.** Pick a low-pass cutoff for the dF/F traces, see its effect live, and write per-cell low-pass and derivative arrays at the chosen cutoff.

This tab is interactive: a slider drives the cutoff, three panels (FFT, raw dF/F, low-pass dF/F) update in real time, and a single button commits the current cutoff to disk.

---

## 1. Inputs

From `plane0/`:
- `r0p7_filtered_dff.memmap.float32` of shape `(T, N_kept)` — preferred. The keep mask / `N_kept` is resolved via `utils.resolve_filtered_mask(plane0, N_total, memmap_path=<filtered memmap>, T=T)` — **anchored to the memmap's on-disk column count**, not the live `predicted_cell_mask`/`iscell`. This prevents `[WinError 8] Not enough memory resources` when curation has drifted the cell mask since Tab 3 wrote the memmap (a wrong-shaped mapping request, not real memory pressure).
- Falls back to `r0p7_dff.memmap.float32` of shape `(T, N_total)` masked by the live `predicted_cell_mask.npy` / `iscell.npy` **only when the filtered memmap is missing** (no fixed file size to anchor against).
- `F.npy` for `(N_total, T)` shape lookup.
- `r0p7_cell_mask_bool.npy` — the authoritative keep mask persisted by Tab 3 alongside the memmap; `resolve_filtered_mask` prefers it, then `predicted_cell_mask.npy`, then `iscell.npy`.
- `predicted_cell_prob.npy` (only used by the "best-scoring ROI" trace source).

The tab subscribes to the AppState `plane0` broadcast so it auto-loads when Tab 3 finishes; "Reload from folder…" lets the user point at any `plane0`.

`fps` comes from Tab 3's **FPS override** (which drives both the dF/F frame rate and suite2p's `fs`); when it is left at 0 the pipeline falls back to `utils.DEFAULT_FPS` (15.07 Hz) and logs a one-time warning to verify it matches your acquisition. The legacy notes-XLSX lookup in `utils.get_fps_from_notes` is now opt-in (pass an explicit `notes_root`); it expects the YYYY-MM-DD_NNN folder-name convention documented in `utils.py`.

---

## 2. Pipeline steps

> **Headless entry point.** `core/lowpass_run.py` exposes
> `compute_lowpass_and_dt(plane0, fps, cutoff_hz, ...)` (the per-ROI compute
> loop), `render_lowpass_figures(plane0, ..., figures_dir)` (saves
> `fft.png` / `raw_dff.png` / `lowpass_dff.png` for the population-mean
> trace), and `run_lowpass(plane0, params, figures_dir=None, progress_cb=None)`
> (compute + figures). Tab 4's `_run_compute` delegates to
> `compute_lowpass_and_dt`, so the GUI and batch worker share one code path.

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

1. **Causal**, not zero-phase. We do **not** use `filtfilt` here because the downstream derivative + onset detection treats the smoothed signal as causal. The deeper reason this is *safe for cross-correlation*: applying the *same* LTI filter to every ROI trace preserves the location of the inter-ROI cross-correlation peak. The filtered-CCG equals the raw-CCG convolved with the autocorrelation of the filter's impulse response, and that autocorrelation is **even (symmetric)** regardless of whether the filter itself is causal. So relative timing between any two ROIs survives identical filtering — only the **absolute** onset time of each ROI is shifted, by roughly half the impulse-response duration (≈ 0.16 s for the default order-2 Butterworth at 1 Hz cutoff and 15 fps). Subtract this constant offset if aligning to external events (stimulus, behavior, etc.).
   The visualisation overlay uses `utils.lowpass_zero_phase_1d` (`sosfiltfilt`) instead — appropriate only because that path is comparing a single trace against its raw version, not computing CCGs.
   **Future upgrade (planned, not yet wired):** for sharp GCaMP8 transients the IIR's frequency-dependent group delay starts to matter (the constant-group-delay intuition only holds asymptotically for a Butterworth). A linear-phase FIR replacement would make the constant-group-delay assumption exact across all frequencies.
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

- **Slider** spans `[CUTOFF_MIN, CUTOFF_MAX]` (default `0.01 – 7 Hz`, configurable via Advanced — see §5). The default starting cutoff is 1.0 Hz.
- **Slider debounce.** `_on_slider` uses `after(80, self._apply_cutoff)` so dragging fires at most every 80 ms, otherwise re-rendering would lag behind the slider.
- **Cutoff entry box.** `_on_entry` clamps to bounds and snaps the slider to the typed value; Enter applies it.
- **Source change.** Reloads the chosen trace, recomputes FFT, redraws all three panels.
- **View toggle.** Two radio buttons under the trace-source row switch the y-axis between `dF/F` (canonical, what the on-disk memmap holds) and `Robust z (median ± 1.4826·MAD)` — a view-only transform that re-expresses each trace via `core.utils.mad_z`. No memmap is written; the toggle just rebuilds the raw + low-pass panels. The robust z view fixes the "positive artifacts following negative deviations" artifact that ΔF/F0 with a rolling baseline creates for inhibited cells (Vanwalleghem & Constantin, *Frontiers Neural Circuits* 2021, "The Curse of Negativity") and makes magnitudes comparable across recordings (the absolute-fluorescence interpretation is lost in exchange).
- **Compute** button writes the memmaps in a child process (`core/offload.py` — not a thread; the per-ROI scipy loop would otherwise hold the GIL and freeze the GUI) and pushes status updates back through a queue (`_compute_queue`).

---

## 5. Advanced settings (`PARAM_SPEC`)

Five tunables open in the standard `AdvancedDialog` (the gear / "Advanced…"
button → `_on_advanced`, `tab.py:699`). The cutoff itself is *not* here — it
lives on the live slider/entry; these only set the slider's **bounds** plus
the fixed filter/derivative knobs that the next **Compute** click bakes into
the on-disk memmaps. The same five flow through `apply_batch_row`
(`tab.py:157`) so Tab 0 batch rows can override them per recording, then on to
`core/lowpass_run.compute_lowpass_and_dt` (`filter_order`, `sg_win_ms`,
`sg_poly` at `lowpass_run.py:64-66`).

### Low-pass filter

| Setting (`key`) | Default | What it does | What it means to you |
|---|---|---|---|
| Butterworth order (`filter_order`) | `2` | Passed as `order` to `butter(order, cutoff/nyq, btype='low', output='sos')` in `utils.lowpass_causal_1d` (`utils.py:295`). Sets how many biquad second-order sections the SOS filter has: a Butterworth of order *n* rolls off at ≈ `6·n` dB/octave above the cutoff. The same `sos` is designed once and reused for every ROI in the Compute loop. | Raise it (3–4) for a sharper signal/noise split — frequencies just above the cutoff are attenuated harder, so high-frequency shot noise leaks through less. The cost: a steeper IIR has **longer group delay** and more ringing, so causal onset times shift later (the ≈ half-impulse-response lag in §3 grows) and sharp GCaMP8 transients can develop a small overshoot. Lower it to `1` for a very gentle, near-lag-free roll-off at the expense of letting more noise through. Order `2` is the validated default; only change it if the FFT elbow is unusually steep/shallow. Interacts with the cutoff: a higher order makes the cutoff *location* matter more. |

### Derivative

These govern the Savitzky–Golay first derivative written to
`r0p7_filtered_dff_dt.memmap.float32`, which **is** the signal Tab 5
thresholds for onset detection (`utils.sg_first_derivative_1d`,
`utils.py:347`). Loosening or tightening them directly changes how many
events Tab 5 finds.

| Setting (`key`) | Default | What it does | What it means to you |
|---|---|---|---|
| SG derivative window (ms) (`sg_win_ms`) | `333` | Converted to an **odd** sample count `win = max(3, int((sg_win_ms/1000)*fps) | 1)` (`utils.py:363`), then used as `savgol_filter(..., window_length=win, deriv=1, delta=1/fps)`. SG fits a `sg_poly`-degree polynomial across `win` samples and reports its analytic slope at the centre. At the default 15.07 fps, 333 ms ≈ a 5-sample window. If `win` ≥ trace length it shrinks to the largest odd window that fits; if still < 3 it falls back to a one-sample finite difference. | The single most impactful derivative knob. **Widen** it (e.g. 500–800 ms) to average over more samples → a smoother, less noisy `dt` trace and fewer spurious Tab 5 onsets, but slow real onsets get blurred and time-smeared (more lag, fused close events). **Narrow** it (e.g. 150–200 ms) to track fast transients crisply at the price of a noisier derivative and more false onsets. Always specified in **milliseconds**, so it self-adjusts when `fps` changes — you don't have to recompute sample counts. Re-Compute after changing it, then re-check Tab 5. |
| SG polynomial order (`sg_poly`) | `2` | The `polyorder` of the fitted polynomial (`utils.py:377`). Degree 2 (quadratic) lets the local fit follow curvature — a rising/peaking transient — rather than assuming a straight line. SciPy requires `sg_poly < window_length`, so it must stay below the effective `sg_win_ms` sample count. | Leave at `2` for almost all data. Raising to `3`–`4` makes the fit hug sharp inflections more faithfully (useful only for very fast GCaMP8 kinetics with a wide window) but also lets the polynomial chase noise, partly undoing the smoothing `sg_win_ms` buys you. Lowering to `1` makes the derivative a plain moving-slope estimate — very smooth, but it underestimates the steepness of fast onsets. Gotcha: a high `sg_poly` with a **narrow** `sg_win_ms` can violate `poly < window` and error out; widen the window if you raise the order. |

### Slider bounds

These only reshape the **live cutoff slider** in the GUI; they are not written
to disk and have no effect in headless/batch runs (which take an explicit
`cutoff_hz`). Applied in `_on_advanced` (`tab.py:704-711`).

| Setting (`key`) | Default | What it does | What it means to you |
|---|---|---|---|
| Slider min (Hz) (`cutoff_min`) | `0.01` | Becomes the slider's `from_` and `self.CUTOFF_MIN`. After the dialog closes the current cutoff is re-clamped into `[cutoff_min, cutoff_max]`. | Raise it if you never want to drag below some floor; lower it to explore very aggressive smoothing (a sub-0.1 Hz cutoff flattens almost everything but the slowest envelope). Cosmetic/convenience only — it constrains the slider, not the math. |
| Slider max (Hz) (`cutoff_max`) | `7.0` | Becomes the slider's `to` and `self.CUTOFF_MAX`. | Raise it (toward but below Nyquist, `fps/2`) to audition near-passthrough cutoffs; the filter core independently clamps any cutoff to `0.95·Nyquist` (`utils.py:291`), so a `cutoff_max` above Nyquist just can't take effect. **Gotcha:** the whole bounds update is silently ignored unless `cutoff_min < cutoff_max` (`tab.py:706`) — set them in the right order. |

---

## 6. Re-implementation checklist

1. `scipy.signal.butter(order, cutoff/nyq, btype='low', output='sos')` + `scipy.signal.sosfilt` for the filter.
2. Initialise `zi = ones((sos.shape[0], 2)) * x[0]` to suppress the startup transient.
3. `scipy.signal.savgol_filter(x, window_length=odd_window, polyorder=2, deriv=1, delta=1/fps)` for the derivative.
4. An FFT power-spectrum routine: subtract the mean, `rfft`, square, normalise by `T`, plot on `semilogy`.
5. A live slider that calls (1) per drag and updates a Matplotlib axvline + the low-pass plot. Debounce via `after(80, …)` to avoid melting the GUI thread.
6. A worker that re-runs the SOS filter + SG derivative once per ROI (re-using the cached `sos`) and writes two `(T, N)` `float32` memmaps with the standardised filenames.
7. The trace-source rules: aggregate over the kept-only memmap for mean/median/best, but reach into the full `(T, N_total)` memmap when the user passes Suite2p ROI ids manually.


## UI affordances

Tab 4 inherits the global customtkinter dark theme from `pipeline_gui`.

- **Cutoff slider.** `CTkSlider` driving the cutoff value; the numeric Entry next to it lets the user type an exact value.
- **Three stacked matplotlib panels** (FFT power spectrum, raw dF/F, low-pass + derivative preview) live in `ttk.LabelFrame`s — matplotlib figures keep their white facecolor; the surrounding chrome is dark. Each canvas has a dark-skinned navigation toolbar (pan/zoom/save + Save data CSV).
- **No popouts.** Advanced parameters open in the standard `AdvancedDialog`.
