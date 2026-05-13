# Tab 2 — QC Preview

**Goal.** Display the QC GIF and mean image produced by Tab 1 so the user can decide visually whether to proceed.

This is the smallest tab: no new computation, just two paired views and some memory-management plumbing.

---

## 1. Inputs

A `PreprocessResult` published by Tab 1 (or loaded from a folder). Specifically:
- `result.qc_gif` — path to `qc.gif`
- `result.mean_image_path` — path to `mean.npy`

A "Reload from folder…" button calls `load_existing_preprocess(path)` and re-publishes the result, letting you point Tab 2 at a previously processed recording without re-running Tab 1.

History: the right-hand panel used to overlay LoG soma-candidate circles on the mean image, fed by a `detect_blobs_on_mean` pass that wrote `blobs.npy`. The overlay was a holdover from before Tab 3's Sparsery + Cellpose detection came online; nothing downstream ever consumed `blobs.npy`. Removed 2026-05-12.

---

## 2. Pipeline steps

### Step 1 — GIF playback

`_load_gif(gif_path)`:

1. Cancel any running `after()` callback.
2. `PIL.Image.open(gif_path)`; iterate `im.seek(im.tell() + 1)` until `EOFError`. Each frame becomes a `tkinter.PhotoImage` and is appended to `self._gif_frames`.
3. If the **Animate** checkbox is OFF, decode only the **first frame** (saves ~1 MB per 512×512 frame in PhotoImage memory) and stop.
4. If ON, schedule `_advance_gif()` every `FRAME_MS = 66` ms (≈15 fps display rate).

`_advance_gif()` cycles `self._gif_index` modulo the frame count and updates the `Label` widget's image. The `gif_label.image = frame` assignment is required to prevent Python from garbage-collecting the `PhotoImage` while it's on screen.

### Step 2 — Memory-management hooks

`PhotoImage` frames are eager — each one is a fully-decoded RGBA buffer in Tk's heap. For a 512×512×N-frame movie this can be tens of MB per recording. The tab implements three levers:

- **Animate toggle** (`_on_animate_toggle`): OFF → freeze to a single still and free the rest of the buffer; ON → reload from disk and resume.
- **`on_tab_hidden()`**: called by `PipelineApp` when the user switches away. Always freezes the buffer regardless of the Animate setting.
- **`on_tab_shown()`**: called when the user switches back. If Animate is ON, re-decodes the GIF; otherwise leaves the still in place.

`_freeze_to_still()` keeps exactly one frame alive (the one currently visible), drops the rest, and calls `gc.collect()`.

### Step 3 — Mean image

`_draw_mean_image(result)`:

1. Load `mean = np.load(result.mean_image_path)`.
2. Display: `ax.imshow(mean, cmap='gray', vmax=quantile(mean, 0.995))`. The `vmax` percentile clamp prevents a few hot pixels from dominating the colour scale.

Title: `"Mean image"`.

---

## 3. Outputs

None — Tab 2 is read-only.

---

## 4. UI layout

Two side-by-side panels in a `ttk.Frame`:
- **Left**: a `ttk.Label` whose `image` attribute is the current animated frame, plus a status string ("`<N> frames`" or "Animate off (single frame)").
- **Right**: a `matplotlib.Figure` with the mean image, wrapped in a `FigureCanvasTkAgg` and the standard `attach_fig_toolbar` (pan/zoom + Save Figure).

The header shows `"Recording: <name>   (<Y> x <X>)"` once a result is available.

---

## 5. Re-implementation checklist

1. A loop that decodes a multi-frame GIF into a list of `PhotoImage` objects and cycles them via `Tk.after`.
2. A `BooleanVar`-driven freeze/thaw lifecycle so the frame buffer doesn't stick around when the tab is hidden.
3. `matplotlib` + `FigureCanvasTkAgg` to draw the mean image.
4. The percentile-based `vmax` (default `0.995`) for the mean image display so that the few hottest pixels don't wash out the rest.
5. Reload-from-folder support via `load_existing_preprocess` (re-publishes the same `PreprocessResult` shape).

That's it. The biological work is all in Tab 1; Tab 2 is the eyeball.
