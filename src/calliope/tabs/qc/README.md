# Tab 2 — QC Preview

**Goal.** Display the QC GIF and mean image produced by Tab 1 so the user can decide visually whether to proceed.

This is the smallest tab: no new computation, just two paired views and a frame-streaming playback loop.

---

## 1. Inputs

A `PreprocessResult` published by Tab 1 (or loaded from a folder). Specifically:
- `result.qc_gif` — path to `qc.gif`
- `result.mean_image_path` — path to `mean.npy`

A "Reload from folder…" button calls `load_existing_preprocess(path)` and re-publishes the result, letting you point Tab 2 at a previously processed recording without re-running Tab 1.

If the folder has been through Tab 3's post-detection archive — shifted TIFF gone, compressed raw at top level — the reload transparently regenerates the shifted from the raw before returning (driven by the `_calliope_raw_paths.json` sidecar). Expect a brief stall (~1 min for a 9 GB recording) while the reshift runs.

History: the right-hand panel used to overlay LoG soma-candidate circles on the mean image, fed by a `detect_blobs_on_mean` pass that wrote `blobs.npy`. The overlay was a holdover from before Tab 3's Sparsery + Cellpose detection came online; nothing downstream ever consumed `blobs.npy`. Removed 2026-05-12.

---

## 2. Pipeline steps

### Step 1 — GIF playback (one frame at a time)

The movie is **streamed** straight from an open `PIL.Image` handle: only a single decoded frame is ever held in memory, so playback costs the same whether the clip is 5 seconds or 5 minutes. There is no multi-megabyte frame buffer to build, free, or rebuild — the decision that made the old buffered design's "Animate" toggle silently fail (see History below).

`_load_gif(gif_path)`:

1. `_stop_playback()` (cancel any pending tick) and `_close_gif()` (release the previous handle).
2. If the path is missing, set status `"GIF missing."` and stop.
3. `self._gif_im = PIL.Image.open(gif_path)`, then `_show_frame(0)` to paint the first frame immediately (so something is visible even while paused).
4. If the **Animate** checkbox is ON, `_start_playback()`; otherwise show `"Paused (frame 1)"`.

`_show_frame(index)` seeks the open handle to `index`, builds one `ImageTk.PhotoImage`, and assigns it to the label (and to `self.gif_label.image`, required so Python doesn't garbage-collect the live image). Seeks only ever advance by **one** or wrap to **0**, which keeps Pillow's palette/disposal compositing correct. The total frame count (`self._gif_n_frames`) is learned lazily the first time playback runs off the end (`EOFError`) — no eager full-clip scan, so a long GIF never blocks the UI.

`_advance_gif()` shows the next frame (wrapping at the end) and re-arms via `self.after(FRAME_MS=66, ...)` (≈15 fps).

### Step 2 — Playback scheduling + visibility hooks

All scheduling flows through two guarded helpers so two clocks can never run at once (the old double-scheduling race):

- **`_start_playback()`** — idempotent: cancels any pending tick before arming a new one.
- **`_stop_playback()`** — cancels the pending tick; safe to call repeatedly.

The levers:

- **Animate toggle** (`_on_animate_toggle`): OFF → `_stop_playback()`, the current frame stays on screen; ON → resume from the open handle (or reopen from `self._gif_path` if the handle was released). It is **never a silent no-op** — with nothing loaded it shows `"Can't play - no QC movie loaded…"`.
- **`on_tab_hidden()`**: `_stop_playback()` + `_close_gif()` (drops the OS file lock). The last frame stays visible.
- **`on_tab_shown()`**: if Animate is ON, reopen from `self._gif_path` and resume; otherwise leave the still in place.

History: the original design eagerly decoded **every** frame into a `list[ImageTk.PhotoImage]` and the OFF toggle sliced that list to one frame + `gc.collect()`. Re-checking Animate then had to re-read the file from disk — which silently did nothing if the file had moved/been deleted or no recording was loaded, and froze the UI while re-decoding a long clip (and `ImageTk.PhotoImage` must be built on the Tk main thread, so off-thread decoding can't fix that). Replaced with frame-at-a-time streaming 2026-05-28; regression tests live in `tests/test_qc_gif_playback.py`.

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

Two side-by-side panels inside a `CTkFrame` content host (dark theme via customtkinter):
- **Left**: a `ttk.Label` whose `image` attribute is the current streamed frame, plus a status string (`"Playing - frame N/M"`, `"Paused (frame N)"`, or a clear error such as `"Can't play - no QC movie loaded…"`). The label stays `ttk.Label` because it hosts a Pillow `ImageTk.PhotoImage`, which `CTkLabel` cannot wrap.
- **Right**: a `matplotlib.Figure` with the mean image, wrapped in a `FigureCanvasTkAgg` and the standard `attach_fig_toolbar` (pan/zoom + Save Figure). Matplotlib figures keep their white facecolor — "dark frame, light plots" — while the toolbar itself is dark-skinned by `restyle_matplotlib_toolbar`.

The header shows `"Recording: <name>   (<Y> x <X>)"` once a result is available, and a **Reload from folder...** `CTkButton` opens a directory picker so the user can re-load an already-preprocessed recording from disk.

History: the tab was migrated to customtkinter dark mode in May 2026 along with the rest of the GUI; previously this section described a plain `ttk.Frame` + system-theme look.

---

## Advanced settings

**This tab has none.** Tab 2 is a read-only QC *viewer* — it renders artefacts Tab 1 already wrote (`qc.gif`, `mean.npy`) and publishes nothing downstream (see [§3 Outputs](#3-outputs)). There is no `PARAM_SPEC`, no "Advanced..." dialog, and no inline numeric control that changes a result. The only two widgets are presentation-only:

| Control | Default | What it does | What it means to you |
| --- | --- | --- | --- |
| **Animate** checkbox (`_gif_animate_var`, `tab.py:109`) | `True` (playing) | A `tk.BooleanVar` play/pause switch. ON calls `_start_playback()` (`tab.py:298`), which arms the `self.after(self.FRAME_MS, ...)` frame clock (`tab.py:306`; `FRAME_MS = 66` ≈ 15 fps); OFF calls `_stop_playback()` and freezes the current frame. It only starts/stops the Tk timer — it never re-reads the file or alters any pixel. | Pure viewing convenience. Pause to inspect a single frame (motion artefact, a bright transient) or to stop the clock while you work in another panel. No effect on any output, so toggle freely. Resuming reopens the handle lazily if it was released on tab-hide. |
| **Reload from folder...** button (`_reload_from_folder`, `tab.py:180`) | — | Opens a directory picker, calls `load_existing_preprocess(path)`, and re-publishes the returned `PreprocessResult` on `AppState`. Takes no parameter. | Use it to view an already-processed recording without re-running Tab 1. If the folder was archived by Tab 3 (shifted TIFF removed), the reload transparently regenerates the shifted movie first — expect a brief stall (~1 min for a 9 GB recording). |

**One hardcoded display constant, not user-tunable.** The mean-image `vmax` clamp uses `DISPLAY_CLIP_HIGH_PCT = 99.5` (`core/utils.py:639`, applied at `tab.py:369` as `np.quantile(mean, 99.5/100)`). This caps the grayscale at the 99.5th percentile so a few hot pixels can't wash out the mean image. It is a shared module constant — the same one used by `detection_run.py:785` and `event_detection_run.py:379` for their background/mean displays — and is *not* surfaced as a setting on any tab. Editing it would require a code change in `core/utils.py`. If you need a different contrast for inspection, use the matplotlib toolbar (pan/zoom) on the right panel instead.

**Where the real knobs live.** Everything that actually shapes the artefacts you are eyeballing here is set upstream in **Tab 1 (Preprocess)** — registration, the QC-GIF frame sampling, and the mean-image computation. If the movie looks wrong or the mean is off, change parameters there and re-run, then re-view here.

---

## 5. Re-implementation checklist

1. A loop that streams a multi-frame GIF one frame at a time from an open `PIL.Image` handle (seek → one `PhotoImage` → `Tk.after`), holding only the current frame in memory.
2. A `BooleanVar`-driven play/pause that stops the clock (and a tab-hide hook that also closes the file handle) — never a silent no-op when resuming.
3. `matplotlib` + `FigureCanvasTkAgg` to draw the mean image.
4. The percentile-based `vmax` (default `0.995`) for the mean image display so that the few hottest pixels don't wash out the rest.
5. Reload-from-folder support via `load_existing_preprocess` (re-publishes the same `PreprocessResult` shape).

That's it. The biological work is all in Tab 1; Tab 2 is the eyeball.
