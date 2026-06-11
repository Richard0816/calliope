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
   **Future upgrade (planned, not yet wired):** for sharp GCaMP8 transients the IIR's frequency-dependent group delay starts to matter (the constant-group-delay intuition only holds asymptotically for a Butterworth). A linear-phase FIR replacement would make the constant-group-delay assumption exact across all frequencies. See `docs/pipeline_audit_2026-05-25.md` §2.8 for the implementation plan.
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
- **View toggle.** Two radio buttons under the trace-source row switch the y-axis between `dF/F` (canonical, what the on-disk memmap holds) and `Robust z (median ± 1.4826·MAD)` — a view-only transform that re-expresses each trace via `core.utils.mad_z`. No memmap is written; the toggle just rebuilds the raw + low-pass panels. The robust z view fixes the "positive artifacts following negative deviations" artifact that ΔF/F0 with a rolling baseline creates for inhibited cells (Vanwalleghem & Constantin, *Frontiers Neural Circuits* 2021, "The Curse of Negativity") and makes magnitudes comparable across recordings (the absolute-fluorescence interpretation is lost in exchange).
- **Compute** button writes the memmaps in a child process (`core/offload.py` — not a thread; the per-ROI scipy loop would otherwise hold the GIL and freeze the GUI) and pushes status updates back through a queue (`_compute_queue`).

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


## UI affordances

Tab 4 inherits the global customtkinter dark theme from `pipeline_gui`.

- **Cutoff slider.** `CTkSlider` driving the cutoff value; the numeric Entry next to it lets the user type an exact value.
- **Three stacked matplotlib panels** (FFT power spectrum, raw dF/F, low-pass + derivative preview) live in `ttk.LabelFrame`s — matplotlib figures keep their white facecolor; the surrounding chrome is dark. Each canvas has a dark-skinned navigation toolbar (pan/zoom/save + Save data CSV).
- **No popouts.** Advanced parameters open in the standard `AdvancedDialog`.
