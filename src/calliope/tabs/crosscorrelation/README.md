# Tab 7 — Cross-Correlation

**Goal.** For each pair of clusters (Cᵢ × Cⱼ) and each pair of cells inside them, compute the time-lagged Pearson correlation. Report:
- `best_lag_sec` — lag at which correlation peaks (positive = ROI A leads).
- `max_corr` — Pearson r at that lag.
- `zero_lag_corr` — Pearson r at lag 0.

Two modes: **full recording** and **per event** (cropped to each window from Tab 5). Plus a **violin plot** summarising every pair's distribution and a **single-pair preview** for inspecting individual cell pairs.

---

## 1. Inputs

From `plane0/`:
- `r0p7_filtered_dff.memmap.float32` — `(T, N_kept)` filtered dF/F.
- `predicted_cell_mask.npy` (or `iscell.npy`) — for prefix translation.
- `<plane0>/r0p7_filtered_cluster_results/gui_recluster/C*_rois.npy` — cluster files from Tab 6 (Suite2p ROI indices, with the `_indices_are_suite2p` marker).
- Event windows for per-event mode: live from Tab 5 via `AppState.event_results` (subscribed at `__init__`), with the `EventWindows` sheet of `<rec>_summary.xlsx` as a cold-start fallback when no Tab 5 publish exists yet.

---

## 2. Pipeline steps

### Step 1 — Open dF/F and the cluster files

`_open_dff_memmap(plane0, prefix)`:

1. `F.npy` shape gives `N_total, T`.
2. If `prefix` contains `filtered`, load the keep mask, set `N_kept = mask.sum()`, return `(memmap, T, N_kept)`.
3. Otherwise the memmap is `(T, N_total)`.

`_load_clusters_from_dir(cluster_dir, keep_mask)`:

- If `_indices_are_suite2p` marker exists: read each `C*_rois.npy` as-is (Suite2p ROI indices).
- Otherwise (legacy layout): translate filtered-list positions to Suite2p indices via `np.where(keep_mask)[0]`.
- `manual_combined*` files are ignored.

### Step 2 — Batched cross-correlation (`batch_xcorr_clusters`)

The core algorithm. Given two cluster matrices `X_A: (T, nA)` and `X_B: (T, nB)`, compute the (nA, nB) `best_lag_sec`, `max_corr`, `zero_lag_corr` matrices in **one matmul per lag**.

```python
ZA = zscore_cols(X_A)          # (T, nA)
ZB = zscore_cols(X_B)          # (T, nB)
L  = floor(max_lag_seconds * fps)

pad    = zeros((L, nA))
ZA_pad = concat([pad, ZA, pad])         # (T+2L, nA)

best_corr     = full((nA, nB), -inf)
best_lag_idx  = zeros((nA, nB), int32)
zero_lag_corr = None

for j in range(2L + 1):
    Zs = ZA_pad[j : j + T]              # sliding view, no copy
    C  = (Zs.T @ ZB) / T                # (nA, nB) Pearson r at lag (j - L)
    if (j - L) == 0:
        zero_lag_corr = C.copy()
    mask         = C > best_corr
    best_corr    = where(mask, C, best_corr)
    best_lag_idx = where(mask, j, best_lag_idx)

best_lag_sec = (best_lag_idx - L) / fps
```

Why this is fast: a naïve "for each pair, slide one signal past the other" is O(nA · nB · 2L · T) Python-loop work. The batched form replaces it with `(2L+1)` `(nA × T) · (T × nB)` matmuls, which the BLAS / cuBLAS GEMM kernel pulverises. For two 200-cell clusters at 9000 frames and `L = 30` (2 s at 15 fps) that's `61` GEMMs of size `200 × 9000 · 9000 × 200` — milliseconds.

#### `_zscore_cols_xp(X, xp, eps=1e-12)`

```python
mu      = X.mean(axis=0, keepdims=True)
sd      = X.std(axis=0, keepdims=True)
sd_safe = where(sd == 0, 1, sd)
Z       = (X − mu) / sd_safe
Z[:, sd==0] = 0                          # zero-out constant columns
```

This produces the **biased Pearson r**: `(Z_a · Z_b) / T`, not the unbiased `cov / (sd_a · sd_b)`. The two differ by a `T/(T-1)` factor that is constant across pairs, so it makes no difference for argmax.

#### GPU path

When `use_gpu=True` and `cupy` is importable:

```python
xp = cupy
XA = xp.asarray(X_A); XB = xp.asarray(X_B)
# ... same algorithm with xp instead of np ...
return cp.asnumpy(...)
```

A single `cp.asarray` move at the top, all matmuls on GPU, one `cp.asnumpy` at the end. No CPU↔GPU pingpong inside the loop.

### Step 3 — Full-recording mode (`run_cluster_xcorr_full_fast`)

```python
for (i, A_rois) in enumerate(clusters):
    for (j, B_rois) in enumerate(clusters):
        X_A = dff[:, A_rois]
        X_B = dff[:, B_rois]
        best_lag, best_corr, zero_corr = batch_xcorr_clusters(
            X_A, X_B, fps, max_lag_seconds, use_gpu=use_gpu)
        write_pair_summary_csv(out_root / f"C{i+1}xC{j+1}", ...)
        progress_cb(...)
```

The output layout is:

```
<plane0>/r0p7_filtered_cluster_results/gui_recluster/cross_correlation_full/
├── C1xC1/
│   ├── C1xC1_summary.csv          ← per-pair best_lag_sec, max_corr, zero_lag_corr
│   ├── C1xC1_best_lag.npy         ← (nA, nB) array
│   ├── C1xC1_max_corr.npy
│   └── C1xC1_zero_lag.npy
├── C1xC2/
├── ...
```

The summary CSV has columns:

```
roi_A, roi_B, best_lag_sec, max_corr, zero_lag_corr
```

`roi_A` and `roi_B` are **Suite2p ROI indices**.

### Step 4 — Per-event mode (`run_cluster_xcorr_per_event_fast`)

For each `(start_s, end_s)` window (from Tab 5's in-memory publish if available, else from the EventWindows xlsx sheet):

```python
f0, f1 = round(start_s * fps), round(end_s * fps)
X_A    = dff[f0:f1, A_rois]
X_B    = dff[f0:f1, B_rois]
# clamp max_lag to (f1 - f0 - 1) / fps so it's never wider than the window
batch_xcorr_clusters(X_A, X_B, ...)
```

Output layout adds an event subdirectory:

```
.../cross_correlation_full/
├── eventwise/
│   ├── event_0000/
│   │   ├── C1xC2/...
│   ├── event_0001/
│   ├── ...
```

`max_lag` is automatically capped to the available window length so a 0.4 s event with default `max_lag = 2 s` won't crash — it gets clamped down.

The per-event run returns one CSV per (event, cluster pair) and is what gets aggregated for "during seizures, who led whom?" analyses.

### Step 5 — Single-pair preview

`single_pair_xcorr_curve(sigA, sigB, fps, max_lag_seconds)` computes the **full curve** (not just the peak) for one pair:

```python
T = sigA.size
Za = zscore(sigA); Zb = zscore(sigB)
L  = floor(max_lag_seconds * fps)
pad      = zeros(L)
Za_pad   = concat([pad, Za, pad])
windows  = sliding_window_view(Za_pad, T)        # (2L+1, T)
r        = (windows @ Zb) / T
lags_sec = arange(-L, L+1) / fps
return lags_sec, r
```

Plotted with the peak marked in red and the zero-lag value marked in green. The user can crop to a specific event from a dropdown (rebuilt from the EventWindows on tab refresh).

### Step 6 — Violin plots

`ViolinWindow` reads every `*_summary.csv` in `cross_correlation_full/` (or per-event) and renders two violins per cluster pair:

1. **Top panel — zero-lag correlation** distribution (`zero_lag_corr` column): one violin per pair, ordered by natural pair key (`C1xC2`, `C1xC10` correctly ordered).
2. **Bottom panel — best-lag distribution** (`best_lag_sec` column), coloured by sign and significance:
   - **Blue** (`#6FA8FF`) — mean lag > 0 and significant: ROI A consistently *leads* ROI B.
   - **Red** (`#E87B73`) — mean lag < 0 and significant: ROI A consistently *lags* ROI B.
   - **Gray** (`#CFCFCF`) — not significant.

#### Sign-flip permutation test (`_sign_flip_pvalue`)

For each pair's lag distribution `values` of length `n`:

```python
observed = abs(values.mean())
hits = 0
for _ in range(n_perm):                              # default 10000
    signs    = random_choice([-1, +1], size=n)
    permuted = (signs * values).mean()
    if abs(permuted) >= observed:
        hits += 1
p = hits / n_perm
```

Implemented in **chunks of 256 permutations** to keep memory at `O(chunk · n)` instead of `O(n_perm · n)` — important for large clusters (200 × 200 ROI pairs would OOM with the naïve allocation).

This tests `H0: lag distribution is symmetric around 0` (no consistent lead/lag direction). Rejecting `H0` means the cluster pair has a directional relationship.

Significance markers: `***` (`p < 0.001`), `**` (`p < 0.01`), `*` (`p < 0.05`).

### Step 7 — Aborting

A long full-recording run on a large cluster set can take minutes. The Abort button sets `_abort_event`; the next call to `progress_cb` from inside the worker raises `RunAborted`, which unwinds the worker cleanly and posts an `("aborted", …)` queue message instead of an error dialog.

---

## 3. Outputs (on disk)

```
<plane0>/r0p7_filtered_cluster_results/gui_recluster/
├── cross_correlation_full/
│   ├── C{i}xC{j}/
│   │   ├── C{i}xC{j}_summary.csv
│   │   ├── C{i}xC{j}_best_lag.npy
│   │   ├── C{i}xC{j}_max_corr.npy
│   │   └── C{i}xC{j}_zero_lag.npy
│   └── eventwise/
│       └── event_NNNN/
│           └── C{i}xC{j}/...
```

(See `core/crosscorrelation.py: _write_pair_summary_csv` and the two `run_*_fast` functions for the exact layout.)

---

## 4. Parameters

| Param | Default | Notes |
|---|---|---|
| `prefix` | `r0p7_filtered_` | dF/F memmap prefix (must match Tab 6 export). |
| `cluster_folder` | `gui_recluster` | Subfolder under `r0p7_filtered_cluster_results/`. |
| `fps` | from notes (fallback 15.07) | Editable at runtime. |
| `max_lag_seconds` | 2.0 | Search range ±L = ±round(max_lag · fps). |
| `also output zero-lag corr` | True | Track the lag-0 matrix during the per-lag loop. |
| `use GPU if available` | True | Falls back to NumPy if CuPy is missing. |

For the violin permutation test: `n_perm=10000`, `chunk=256`, `seed=0`.

---

## 5. UI flow

1. Set plane0 (auto-broadcast from Tab 5 or browse).
2. Set prefix / cluster_folder / fps / max_lag.
3. **Run full-recording cross-correlation** — runs until done (or abort).
4. **Reload event windows** if you've edited the xlsx by hand (Tab 5 reruns auto-push via `event_results`).
5. **Run per-event cross-correlation** — only enabled when events exist.
6. **Violin plot** — opens a separate window over the `cross_correlation_full/` outputs.
7. **Single-pair preview** — pick two ROI indices (Suite2p numbers), choose full or specific event, click Plot.

---

## 6. Re-implementation checklist

1. **Z-score columns + biased Pearson r:** `(Z_A · Z_B) / T` after zero-mean-unit-std per column (with constant-column protection).
2. **Batched lag search:** pad ZA on both ends with `L` zeros, slide a `(T, nA)` view across the padded array, do one `(nA, nB)` GEMM per lag, track running per-pair argmax.
3. **GPU path** identical to (2) but with `cupy` arrays; a single `asarray` in and `asnumpy` out.
4. **Cluster file format:** `C{i}_rois.npy` containing Suite2p ROI indices, plus a `_indices_are_suite2p` marker file. Honour both layouts on load.
5. **Per-event windowing:** prefer Tab 5's in-memory `event_results` for the current plane0; fall back to the EventWindows sheet of `<rec>_summary.xlsx`. Convert seconds to `(f0, f1)` frame indices and clamp `max_lag` to the window length so zero-pad doesn't dominate.
6. **Sign-flip permutation:** random ±1 signs over the values array, recompute mean, count `|permuted| ≥ |observed|`. Chunk the random-sign matrix to bound memory.
7. **Single-pair full curve:** same z-scoring, but use `numpy.lib.stride_tricks.sliding_window_view` and one `(2L+1, T) · (T,)` matmul to get all lags in one shot.
8. **Output schema:** per-pair `{summary.csv, best_lag.npy, max_corr.npy, zero_lag.npy}` so downstream analyses can read the CSV summaries or the dense matrices as needed.
