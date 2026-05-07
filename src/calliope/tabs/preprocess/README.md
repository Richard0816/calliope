# Tab 1 — Preprocess

**Goal.** Convert one or more raw multi-page TIFFs into:
- a uint16, non-negative TIFF (Suite2p's expected format),
- a `mean.npy` time-average,
- a `qc.gif` downsampled animated preview,
- a `blobs.npy` list of candidate cell-body locations on the mean image.

All outputs land under `<data_root>/<recording_name>/`.

---

## 1. Inputs

A list of raw TIFF stacks of shape `(T, Y, X)`:
- `T` = number of frames (~9000 for a 10-min recording at 15 fps).
- `Y, X` = the image dimensions (typically 512×512).
- `dtype` = signed or unsigned 16-bit integer; on some scopes the values can be **negative** because of dark-current correction.

If multiple TIFFs are selected, they are treated as **one continuous recording**: a single shift constant is computed across all of them so brightness is consistent across the join, and they are renamed with a numeric prefix (`000_`, `001_`, …) so Suite2p's `natsort` reads them in order.

---

## 2. Pipeline steps

The driver function for a single TIFF is `preprocess_tiff` in `core/preprocessing.py`. For groups it's `preprocess_tiff_group`. Both run the same four logical steps; the group version replaces step 1 with a two-pass scan-then-write, and step 3 sums frame-counts across files for the mean.

### Step 1 — Shift to non-negative uint16

For Suite2p, pixel values must fit in `uint16` (range `[0, 65535]`). Raw frames may have negative values, so we shift the entire stack by `−min(stack)` (which is `0` if the min is already non-negative).

**Single-file fast path** (`shift_tiff_to_uint16`):

```python
data = tifffile.imread(src).astype(np.int32)  # whole stack into RAM
data += abs(min(data))                        # make global min = 0
if data.max() >= 65535:
    raise ValueError(...)                     # dynamic range too wide
data = data.astype(np.uint16)
tifffile.imwrite(dst, data)
```

**Streaming fallback** (`_shift_tiff_streaming`) — triggered when the in-RAM read raises `MemoryError`. Two passes:

1. **Scan**: open `tifffile.TiffFile(src)`, iterate over `tf.pages`, track per-page `min` and `max`, accumulate global `gmin`, `gmax`. No frame held in RAM beyond the current one.
2. **Write**: open the source again and a `TiffWriter(dst, bigtiff=True)` (BigTIFF avoids the 4 GB single-file cap), shift each page by `−gmin`, cast to `uint16`, write.

**Group path** (`preprocess_tiff_group`): does Pass 1 across **every** input file before computing the shared shift constant, so the relative brightness across files is preserved. Pass 2 writes each shifted file independently.

**Why this ordering matters biologically.** Per-file shifts would introduce phantom intensity steps at file boundaries that look exactly like calcium events. The group path's single shared shift constant is the only way to keep the dF/F baseline coherent across stitched files.

---

### Step 2 — Mean image

Single file: `mean = movie.mean(axis=0)`, cast to `float32`, save as `mean.npy`.

Group: stream-sum each file's `m.sum(axis=0, dtype=np.float64)` into a running 2D accumulator, divide by total frame count at the end. This avoids materialising the concatenated movie.

---

### Step 3 — Preview blob detection

`detect_blobs_on_mean(mean_img, soma_diameter_px=12, scale_tol=0.5, min_contrast=0.10, min_area_px=25, max_area_px=400)`:

1. **Background subtract.** Apply a 2D **median filter** with radius `bg_radius = max(3, round(soma_diameter_px))` to estimate the slowly-varying background, then subtract: `hp = clip(img − bg_local, 0, ∞)`. This removes large-scale brightness gradients (uneven illumination, vignetting) so the LoG is sensitive only to local bumps.
2. **Robust normalize.** `norm = clip(hp / quantile(hp, 0.995), 0, 1)`. The 99.5th percentile is more robust than `max` to a few hot pixels.
3. **Laplacian-of-Gaussian blob detection** (`skimage.feature.blob_log`) with sigma range:
   - `r = soma_diameter_px / 2`
   - `min_sigma = r·(1 − scale_tol) / √2`
   - `max_sigma = r·(1 + scale_tol) / √2`
   - `num_sigma = 6` evenly-spaced sigmas
   - `threshold = min_contrast` after normalisation
   - `overlap = 0.5` (drop blobs that overlap >50% with another)
4. **Area filter.** For each `(y, x, σ)`, `radius = σ·√2`, `area = π·radius²`. Keep only blobs with `min_area_px ≤ area ≤ max_area_px` (default 25–400 px² ≈ soma-sized).

Save as `blobs.npy` with shape `(N, 3)` columns `[y, x, radius]`.

The `√2` factor in the sigma↔radius conversion comes from the LoG's mathematical definition: a blob of radius `r` is detected most strongly at a Gaussian `σ = r/√2`.

**This is not the final ROI list.** It's a sanity check shown in Tab 2.

---

### Step 4 — QC GIF

`make_qc_gif(movie, gif_path, downsample_t=4, max_size_px=512, playback_fps=15, clip_low=1, clip_high=99.5)`:

1. **Time downsample.** Keep every `downsample_t`-th frame.
2. **Intensity rescale.** Clip to `[percentile(1), percentile(99.5)]`, rescale to `[0, 255]`, cast to `uint8`. Robust percentiles, not min/max.
3. **Spatial downscale** (PIL bilinear) until the long edge ≤ `max_size_px`.
4. **GIF write** via `PIL.Image.save(append_images=...)` with `duration=1000/playback_fps` ms per frame, `loop=0`, `optimize=True`.

For groups, the per-file downsampled frames are concatenated before encoding, with `downsample_t=1` passed through (since downsampling already happened during sampling).

---

## 3. Outputs (on disk)

```
<data_root>/<recording_name>/
├── shifted_<orig>.tif        ← single-file path
│   or 000_shifted_<orig>.tif, 001_shifted_<orig>.tif, ...   ← group path
├── mean.npy                  ← float32, shape (Y, X)
├── blobs.npy                 ← float32, shape (N_blobs, 3); columns y, x, radius
└── qc.gif
```

The `PreprocessResult` dataclass that gets passed to other tabs records:
- `out_dir`, `shifted_tiff`, `qc_gif`, `mean_image_path`, `blobs_path`
- `n_frames` (group total; `-1` when loaded from disk)
- `shape_yx`
- `n_blobs`

---

## 4. Discovery helpers

- `list_tiffs(folder, max_depth=0)` — BFS over the working directory, returns sorted `Path`s of `.tif/.tiff`. Default `max_depth=0` only scans the top level; the GUI's spinbox lets the user dig N levels deep.
- `load_existing_preprocess(out_dir)` — reads `shifted_*.tif`, `mean.npy`, `blobs.npy`, `qc.gif` from disk; returns a `PreprocessResult` with `n_frames=-1` (unknown without reopening). Used by the "load existing" code paths in Tabs 1 and 2.

---

## 5. Parameters (from `PreprocessTab.PARAM_SPEC`)

| Param | Default | Effect |
|---|---|---|
| `soma_diameter_px` | 12.0 | Expected soma diameter in pixels; sets LoG sigma range. |
| `scale_tol` | 0.5 | Sigma range = `r·(1 ± scale_tol)/√2`. |
| `min_contrast` | 0.10 | LoG threshold after percentile normalisation. |
| `min_area_px` | 25 | Reject blobs smaller than this. |
| `max_area_px` | 400 | Reject blobs larger than this. |
| `downsample_t` | 4 | Keep every Nth frame for the QC GIF. |
| `max_size_px` | 512 | Long-edge cap for GIF spatial size. |
| `playback_fps` | 15 | GIF playback speed. |
| `clip_low` / `clip_high` | 1 / 99.5 | Percentile clip range for GIF normalisation. |

---

## 6. Re-implementation checklist

To reproduce Tab 1 from scratch you need:

1. `tifffile` to read multi-page TIFFs page-by-page (`TiffFile.pages`, `TiffWriter`, `bigtiff=True`).
2. `numpy` for the shift arithmetic and mean accumulation.
3. `Pillow` (`PIL.Image`) for GIF encoding.
4. `scipy.ndimage.median_filter` and `skimage.feature.blob_log` for the blob preview.
5. The decision tree: try in-RAM shift, on `MemoryError` fall back to two-pass streaming.
6. The group invariant: one shared shift constant across all files, computed before any writes begin.
7. The output filename conventions (`shifted_<orig>.tif` or `NNN_shifted_<orig>.tif`) — Suite2p's `natsort` depends on the leading numeric prefix to read multi-file groups in order.
