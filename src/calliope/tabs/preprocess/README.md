# Tab 1 — Preprocess

**Goal.** Convert one or more raw multi-page TIFFs into:
- a uint16, non-negative TIFF (Suite2p's expected format),
- a `mean.npy` time-average,
- a `qc.gif` downsampled animated preview.

All outputs land under `<data_root>/<recording_name>/`.

History: an earlier `detect_blobs_on_mean` step wrote `blobs.npy` (LoG soma candidates) for Tab 2 to overlay on the mean image. Removed 2026-05-12 — nothing downstream consumed `blobs.npy`, and skipping the LoG pass saves a `skimage`/`scipy.ndimage` import cost on every preprocess.

---

## 1. Inputs

A list of raw TIFF stacks of shape `(T, Y, X)`:
- `T` = number of frames (~9000 for a 10-min recording at 15 fps).
- `Y, X` = the image dimensions (typically 512×512).
- `dtype` = signed or unsigned 16-bit integer; on some scopes the values can be **negative** because of dark-current correction.

If multiple TIFFs are selected, they are treated as **one continuous recording**: a single shift constant is computed across all of them so brightness is consistent across the join, and they are renamed with a numeric prefix (`000_`, `001_`, …) so Suite2p's `natsort` reads them in order.

---

## 2. Pipeline steps

The driver function for a single TIFF is `preprocess_tiff` in `core/preprocessing.py`. For groups it's `preprocess_tiff_group`. Both run the same three logical steps; the group version replaces step 1 with a two-pass scan-then-write, and step 2 sums frame-counts across files for the mean.

> **Headless entry point.** `core/preprocessing.py:run_preprocess(src_tiffs, data_root, params, *, recording_name=None, figures_dir=None, progress_cb=None)` picks single vs grouped automatically and copies the QC GIF into `figures_dir` if provided. Tab 0's batch pipeline calls this; the interactive Tab 1 still calls `preprocess_tiff{,_group}` directly.

### Step 1 — Shift to non-negative uint16

For Suite2p, pixel values must fit in `uint16` (range `[0, 65535]`). Raw frames may have negative values, so we shift the entire stack by `−min(stack)` (which is `0` if the min is already non-negative).

**Single-file fast path** (`shift_tiff_to_uint16`):

```python
data = tifffile.imread(src).astype(np.int32)  # whole stack into RAM
data += abs(min(data))                        # make global min = 0
if data.max() > 65535:
    raise ValueError(...)                     # exceeds uint16 max (65535 is valid)
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

### Step 3 — QC GIF

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
├── qc.gif
└── _calliope_raw_paths.json  ← raw source paths for the archive step
```

The `PreprocessResult` dataclass that gets passed to other tabs records:
- `out_dir`, `shifted_tiff`, `qc_gif`, `mean_image_path`
- `n_frames` (group total; `-1` when loaded from disk)
- `shape_yx`
- `raw_paths` — absolute paths to the original raw TIFF(s) used for this recording
- `shifted_paths` — every shifted TIFF written for this recording

`_calliope_raw_paths.json` is a small JSON sidecar holding the raw source paths. Tab 3's post-detection archive step reads it back to compress the originals into the recording folder once detection has succeeded — see Tab 3's README, section *Post-detection archive*.

---

## 4. Discovery helpers

- `list_tiffs(folder, max_depth=0)` — BFS over the working directory, returns sorted `Path`s of `.tif/.tiff`. Default `max_depth=0` only scans the top level; the GUI's spinbox lets the user dig N levels deep.
- `load_existing_preprocess(out_dir, *, progress_cb=None)` — reads `shifted_*.tif`, `mean.npy`, `qc.gif` from disk; returns a `PreprocessResult` with `n_frames=-1` (unknown without reopening). Used by the "load existing" code paths in Tabs 1 and 2. **Archived-state recovery:** if the shifted TIFFs have been deleted by Tab 3's post-detection archive step but `_calliope_raw_paths.json` is present, the shifted is regenerated on demand from the (now-compressed) raw — single-file recordings re-run `shift_tiff_to_uint16`, multi-file groups re-run `preprocess_tiff_group` so the shared global shift constant is recovered.

---

## 5. Advanced settings (Advanced... dialog)

The **Advanced...** button opens a modal `AdvancedDialog` auto-built from `PreprocessTab.PARAM_SPEC` (`tab.py:100`). Edits mutate `self._params` in place; they take effect on the **next** Run (the snapshot in `_start_run` is taken when you click Run, so re-opening Advanced mid-run is ignored — `tab.py:493`).

**Scope: these five knobs only touch the `qc.gif`.** None of them affect the shifted TIFF, `mean.npy`, the dF/F baseline, or anything Suite2p / detection consumes downstream. The GIF is a *visual QC artifact* (played back on Tab 2) — tuning these changes only what you see, never the science. So getting them "wrong" costs you nothing but a less-readable preview.

### QC gif

All five are consumed by `make_qc_gif` (`core/preprocessing.py:623`), except that `downsample_t` is actually applied one step earlier — during the shift read loop (`shift_tiff_to_uint16(..., qc_downsample_t=...)`, `core/preprocessing.py:178`), so the heavy stack is never re-read just to sample frames. `make_qc_gif` is then called with `downsample_t=1` on the already-sampled frames (`core/preprocessing.py:771`, `:936`).

| Setting (`key`) | Default | What it does | What it means to you |
|---|---|---|---|
| `downsample_t` | `4` | Keep every Nth frame: `movie[::max(1, int(downsample_t))]` (`:650`); in the real pipeline the stride is applied during the shift pass (`:178`). Clamped to ≥1, so `0`/negative ⇒ keep every frame. A 9000-frame 10-min @ 15 fps movie ÷ 4 ≈ 2250 GIF frames. | Higher = fewer frames = smaller, faster-to-write GIF that **skips more of the recording** (fast transients can fall between kept frames). Lower (toward 1) = every frame, smooth but huge file and slow encode. Raise it for long recordings; lower it if you're hunting brief events in the preview. |
| `max_size_px` | `512` | Long-edge cap. If `max(im.size) > max_size_px`, each frame is bilinear-resized so its longest side equals the cap (`:673`). At/below the cap (512×512 raw) it's a no-op. | Higher = sharper preview, bigger file. Lower = coarser, smaller file — useful if you're scanning many recordings and only need to spot gross motion/drift/focus. Set it ≥ your raw frame size to keep full resolution. |
| `playback_fps` | `15` | Per-frame display duration: `duration_ms = max(1, round(1000 / playback_fps))` written into the GIF header (`:683`). 15 fps ⇒ ~67 ms/frame. Does **not** resample — it only sets playback speed. | Purely cosmetic playback speed. Raise to skim long recordings faster; lower to slow-motion through a region. Note it pairs with `downsample_t`: at `downsample_t=4` the GIF plays 4× wall-clock speed regardless of this value, since 3 of every 4 frames are gone. |
| `clip_low` | `1.0` | Lower percentile for the display window. `lo = np.percentile(sample, clip_low)` over a strided ≤200-frame subsample (`:654`–`:656`); pixels ≤ `lo` map to black. Keep in `0`–`100` and below `clip_high`. | Raise to crush more dark background to black (higher contrast, but you may clip dim cells); lower (toward 0) to preserve faint signal at the cost of a washed-out, noisy-looking floor. |
| `clip_high` | `99.5` | Upper percentile. `hi = np.percentile(sample, clip_high)`; pixels ≥ `hi` saturate to white (`:657`). Window `[lo, hi]` is rescaled to `[0, 255]` per frame. Keep in `0`–`100` and above `clip_low`. | Lower it (toward, say, 98) to brighten and reveal dim structure when bright outliers are washing out the frame; raise toward 100 to keep bright peaks from saturating. This is the main knob if the GIF looks all-black or all-white. Percentiles are robust to a few hot pixels — that's why they're used instead of min/max. |

> The dialog only type-casts your entries (`gui_common.AdvancedDialog._coerce`) — it does **not** validate ranges or ordering. A percentile outside `0`–`100` makes `np.percentile` raise and the run aborts; `clip_low ≥ clip_high` doesn't crash but yields a degenerate (uniform/garbage) GIF.

---

## 6. Re-implementation checklist

To reproduce Tab 1 from scratch you need:

1. `tifffile` to read multi-page TIFFs page-by-page (`TiffFile.pages`, `TiffWriter`, `bigtiff=True`).
2. `numpy` for the shift arithmetic and mean accumulation.
3. `Pillow` (`PIL.Image`) for GIF encoding.
4. The decision tree: try in-RAM shift, on `MemoryError` fall back to two-pass streaming.
5. The group invariant: one shared shift constant across all files, computed before any writes begin.
6. The output filename conventions (`shifted_<orig>.tif` or `NNN_shifted_<orig>.tif`) — Suite2p's `natsort` depends on the leading numeric prefix to read multi-file groups in order.


## UI affordances

Tab 1 inherits the global customtkinter dark theme from `pipeline_gui`.

- **Resizable panels (drag grips).** Panel 2 (TIFF files listbox) and Panel 5 (Log) each carry a draggable handle below them. Drag down to extend that panel — the scrollable tab body absorbs the extra height. Panels 1 (Working directory), 3 (Output root), and 4 (Run) stay at natural height.
- **Scroll on hover.** Spinning the wheel anywhere over the TIFF list scrolls the listbox; elsewhere on the tab it scrolls the tab body when content overflows.
- **No popouts.** TIFF picking is inline in the listbox; the Advanced parameters open in a modal `AdvancedDialog` (resizable; live PARAM_SPEC form).
