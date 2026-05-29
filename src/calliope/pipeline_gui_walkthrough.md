# CalLIOPE — Pipeline Walkthrough 

**CalLIOPE** = **Cal**cium **L**ive-imaging **O**utput **P**ipeline for **E**piletiform-recordings.

This document is a high-level tour of the GUI for someone who has taken intro neuroscience but is new to calcium imaging analysis. It explains, for each of the eight tabs, **what biological signal we're chasing, what the tab does to the data, and why a working neuroscientist would care about the result**. The deeper "how the math works" lives in the `README.md` inside each tab's folder.

---

## What's in the input file?

Every analysis starts from a multi-page TIFF: a stack of 2D images captured at ~15 frames/second by a two-photon microscope. Each pixel reports the brightness of a calcium-sensitive fluorescent protein (a **GCaMP** or similar GECI) in a small chunk of brain tissue. When neurons fire, intracellular Ca²⁺ rises, GCaMP brightens, and that pixel gets brighter. So **brightness over time at a neuron's location = a proxy for neural firing**.

The recordings here are from human/animal brain slices in conditions that drive **epileptiform** activity (synchronized seizure-like bursts). The whole pipeline is designed to answer questions like:

- Which cells fired, and when?
- Were they synchronised? In what spatial pattern?
- Did one population consistently lead another?
- Did seizure-like population events have a stereotyped propagation pattern?

---

## The pipeline at a glance

```
TIFF stack
   │
   ▼
[1] Preprocess ─── shift intensities to uint16, build mean image, find candidate cell bodies
   │
   ▼
[2] QC Preview ── animated GIF + mean image with circles around the candidate cells
   │
   ▼
[3] Suite2p Detection ─── find ROIs (cell footprints), extract raw fluorescence,
                          compute dF/F (relative change in brightness),
                          apply a learned cell/non-cell classifier
   │
   ▼
[4] Low-pass filter ─── smooth each ROI's dF/F to suppress shot noise and
                        compute its first time-derivative (rate of brightening)
   │
   ▼
[5] Event detection ─── find when each cell fires (per-ROI onsets) and
                        find population-wide events (when many cells fire together)
   │
   ▼
[6] Clustering ─── group cells whose dF/F traces co-vary (similar activity = similar ensemble)
   │
   ▼
[7] Cross-correlation ─── for each pair of clusters, ask "does cluster A lead cluster B,
                          and by how much?" — both across the whole recording and within
                          each detected event
   │
   ▼
[8] Spatial propagation ── per event, paint a map of who fired earliest (cyan) → latest (red)
                           and click through events one at a time
```

Each stage writes its outputs to disk so later tabs can re-load without recomputing.

**Reopening a finished run.** The sidebar's **📂 Open run folder…** button reloads an entire recording in one click: pick the recording folder and every tab repopulates in dependency order (QC movie + mean → detection ROIs/traces → lowpass → events → clusters → cross-correlation), no recompute. Each tab still has its own *Reload from folder…* button for loading a single stage in isolation; the sidebar button is the "open the whole run" shortcut. One stage is special — event detection's results are never written to disk, so they're recomputed from the lowpass memmaps using the **saved params from the run's `manifest.json`** (not the current GUI settings) so the reloaded events match the original run. If the manifest lacks event-detection params (e.g. a hand-run recording whose manifest was last overwritten by a later stage) or was produced by a different CalLIOPE version, the load summary says so. Implemented in `core/run_loader.py` (folder inventory) + `PipelineApp._on_open_run`.

A separate **Tab 0 (Batch runner)** sits in front of all of this and chains the same per-stage code paths headlessly across many recordings — see the next section.

---

## Tab 0 — Batch runner

**What it does (operationally).** Tab 0 is a queue runner. You point it at a working directory, click *Scan*, and it auto-creates one row per TIFF file at the configured search depth (multiple TIFFs in the same folder each get their own row; use **Merge selected** afterwards if they're actually one multi-file recording). Each row holds an identifier (defaults to the TIFF's stem, with the parent folder appended on collision), a list of TIFFs, and a per-row parameter override dialog. Hit **Run all** and the worker drives the same code paths the interactive Tabs 1, 3, 4, 5, 6, 7, 8 do — concretely the `core.batch_pipeline.run_recording` orchestrator, which calls `core.preprocess_run` → `core.detection_run` → `core.lowpass_run` → `core.event_detection_run` → `core.clustering_run` → `core.crosscorrelation_run` → `core.spatial_run` in sequence.

The per-row **Edit parameters** button opens a unified Advanced dialog with every PARAM_SPEC entry from Tabs 1 + 3 + 4 + 5 plus the clustering / xcorr knobs (74 in total), grouped in pipeline order: `1. Preprocess - …` → `2. Detection - …` → `3. Low-pass - …` → `4. Events - …` → `5. Clustering` → `6. Cross-correlation` → `7. Pipeline-wide`. The top-level **Apply defaults to all rows** button opens the same dialog and writes its output into every row's params, so per-row dialogs are only needed when one recording wants different settings.

**What lands on disk.** For each row, every figure each tab would normally show in the GUI is rendered headlessly to PNG under `<output>/<recording_id>/calliope_figures/<stage>/` (subfolders: `preprocess`, `detection`, `lowpass`, `event_detection`, `clustering`, `crosscorrelation`, `spatial_propagation`). The queue + per-row params snapshot to `<output>/calliope_batch.json` on every Run All so closing the GUI doesn't lose the work; a **Reload queue** button restores it. After every Run All, `<output>/batch_report.csv` summarises status / duration per stage per recording, with continue-on-error semantics so one bad recording doesn't stop the queue.

**Optional fast-disk routing.** A **Scratch dir (SSD)** field next to the output folder accepts a path on a fast drive (NVMe / SSD). When set, every recording does its intermediate I/O on scratch. A daemon thread per row polls scratch every 2 s and copies new/changed files to HDD throughout the row's pipeline computation; at row end the mirror is signaled to stop, drains the last few writes, repoints `AppState.plane0` / `AppState.result` / tab-local caches to HDD, and `rmtree`s scratch.

Whether scratch and output share a physical drive (detected via `os.stat(...).st_dev` at Run All start) determines a single setting: **two-drive layout** runs the mirror unthrottled (SSD reads and HDD writes don't compete when they're on separate drives); **single-drive layout** throttles the mirror to ~10 MB/s (`SINGLE_DRIVE_FINALIZE_RATE_BYTES_PER_SEC`) so it can't starve the active stage's reads on the same physical drive. The mirror skips known intermediates (`data_raw.bin`, `sparsery_pass/`, `cellpose_pass/`, `*.tmp`) and defers files whose mtime is < 1 s old (in-flight writes). The finalize runs in a daemon thread so the next recording's preprocess can start on scratch (SSD writes) immediately while the previous row's bulk SSD→HDD copy drains in the background; `_batch_finish` waits for every in-flight copy to join before writing `batch_report.csv` so the report's plane0 paths always point at files that exist on HDD. Hardlinks between `_shared_reg/data.bin` and per-pass `data.bin` are preserved across the copy so the destination drive doesn't double-bill the registered movie. The slow output folder still receives the same outputs as before (including `_shared_reg/`, so a manual GUI re-run still skips re-registration); leave the field blank to write directly to the slow drive.

**Why it matters.** Sitting in front of the GUI to drive 78 recordings through seven tabs each is unscalable. Tab 0 lets you set sensible defaults once, queue everything overnight, and come back to a folder of figures and a CSV report you can sanity-check the next morning.

→ See `tabs/batch/README.md` for the row state machine, the JSON-persistence schema, and the `core/batch_pipeline.py` orchestrator.

---

## Tab 1 — Input & Preprocess

**What it does (biology-first).** Raw TIFFs from a microscope can have negative pixel values (because of dark-current correction) and inconsistent dynamic ranges across files. Tab 1 *shifts* the intensities so the minimum value is 0 (without distorting the relative differences that carry the biology), packs them into 16-bit unsigned integers (Suite2p's expected format), then makes:

- A **mean image** — averaging every frame in time gives you the steady spatial picture of the slice. Cell bodies become visible because they're slightly brighter and rounder than neuropil.
- A **QC GIF** — a downsampled animated preview so you can eyeball whether the recording moved, drifted, or has obvious motion artefacts before investing in detection.

**Why it matters.** Two-photon recordings can be noisy. If your mean image looks blurry or your QC GIF shows the field of view sliding, no amount of downstream cleverness will save you — better to know now and either re-register or reshoot.

**Biological output of this stage.** None on its own. Tab 1 is a hygiene step.

→ See `tabs/preprocess/README.md` for the streaming intensity shift and QC GIF code paths.

---

## Tab 2 — QC Preview

**What it does.** Pure visualization. Plays the GIF from Tab 1 next to the mean image. An **Animate** checkbox pauses/resumes playback; the movie is streamed one frame at a time, so pausing and resuming are instant and never re-read the file.

**Why it matters.** This is where you catch:

- **Z-drift** — if the whole field appears to brighten/dim or focus changes mid-recording, your "neuron" might be the same cell sliding through focus, not actual activity.
- **XY motion** — if cells visibly translate, ROIs assigned in frame 1 won't match the cells in frame N. Tab 1 doesn't motion-correct (Suite2p does that in Tab 3), but blatant motion is worth knowing.
- **Photobleaching** — overall intensity decay over the recording.
- **Bubbles, debris, dead patches** — biological/optical artefacts that should mask out regions.

**Biological output.** Confidence (or lack of it) in the recording. If this tab looks bad, fix the experiment, not the analysis.

→ See `tabs/qc/README.md` for the (very small) GIF-playback + mean-image rendering logic.

---

## Tab 3 — Suite2p Detection

**What it does.** This is the workhorse. It runs **Suite2p** (a popular calcium-imaging detection package; Pachitariu et al., Janelia) plus an in-house **Cellpose** pass and produces:

1. **ROIs** — for each detected cell body, a list of pixels (a *footprint*) plus a per-pixel weight (a *lambda* mask) saying "this pixel is 0.7-of-a-cell, this one is 1.0-of-a-cell." Two algorithms run:
   - **Sparsery** (the default Suite2p detector) — finds bright, sparse, time-varying spots.
   - **Cellpose** (cyto2) — a generalist deep-learning cell-segmentation model, run on the mean image as a "second opinion" to catch cells that didn't fire enough during the recording for Sparsery to find them. Cellpose ROIs that overlap >30% with Sparsery ROIs are dropped to avoid double-counting.
2. **Raw traces F and Fneu** — per-ROI fluorescence over time, plus the surrounding "neuropil" trace which contains contamination from out-of-focus cells.
3. **dF/F** — `(F − r·Fneu − F₀) / F₀` per ROI. This is the **biologically meaningful** signal: relative change from baseline brightness. We subtract `0.7 × neuropil` to remove contamination (`r=0.7` is a community standard from Chen et al. 2013), then divide by a baseline `F₀` (either rolling 10th percentile or the mean of the first few minutes) so a 50% increase in fluorescence reads as `0.5` regardless of how bright the original cell was.
4. **Low-pass dF/F + first derivative** — pre-computed at default settings so Tabs 4–7 have a starting point.
5. **Cell filter** — a small PyTorch CNN (`cellfilter/`) trained on hand-labelled ROIs that scores each ROI 0–1 as "this is a real cell" vs "this is an artefact / bright pixel / blood vessel." The user sees both panels: all detected ROIs, and only those that pass the classifier.

**Interactive curation.** Click any ROI on the "All detected ROIs" panel to open a popout with five inspection panels (mean image, max projection, max + ROI, ROI footprint, ΔF/F trace) modeled on `roi_curation_app.py`. **Cell (1) / Not a cell (0)** flip `iscell.npy` in place and append to the curation CSV (`cellfilter.config.LABELS_CSV`, default `F:\roi_curation.csv`). A **Retrain cell filter** button stays disabled until the user has flipped at least one classification this session; clicking it spawns `cellfilter.train.main()` in a worker thread. A **Promote to filter mask** button (also gated on at least one flip) overlays this session's flips onto `predicted_cell_mask.npy` so downstream tabs (5/6/7, cross-correlation) honour the curator's labels *without* paying for a full retrain — CNN predictions for un-touched ROIs are preserved. Tab 3's panels 2 + 3 repaint immediately after every flip so the keep-set state stays in sync.

**Why it matters biologically.** ROIs are how the entire pipeline anthropomorphises pixels into cells. Get this wrong and:

- **Too many ROIs**: noise traces dominate clustering and event detection.
- **Too few ROIs**: real cells get excluded, lowering statistical power.
- **dF/F mis-baselined**: a slow drift gets read as a sustained event.

The cell-filter step gives you a reproducible "is this really a cell?" decision instead of relying on Suite2p's built-in `iscell.npy` (which is noisy across recordings).

**Biological output.** A clean per-cell trace of *relative firing-related fluorescence* over the whole recording, for every cell that the classifier thinks is real.

**Intermediate cleanup.** After the final outputs (`F`, `Fneu`, dF/F memmaps, cell-filter outputs) are written, the shared helper `core.detection_run.prune_detection_intermediates` prunes the ~26 GB/recording of intermediate Suite2p binaries — `detection/sparsery_pass/`, `detection/cellpose_pass/`, and `detection/_shared_reg/suite2p/plane0/data.bin`. The `_shared_reg/ops.npy` registration metadata stays for audit. The cleanup fires from both Tab 3's GUI worker and the headless `run_detection` (which `core.batch_pipeline.run_recording` invokes), so direct-run agents and external drivers get it automatically. Without this prune, an agent that runs detection over many recordings will fill the save drive (~12 recordings ≈ 312 GB).

**Post-detection archive.** Immediately after the prune, `core.detection_run.archive_recording_post_detection` collapses the recording to its smallest stable footprint: it locates the raw TIFF source(s) via the `_calliope_raw_paths.json` sidecar Tab 1 wrote, re-encodes each as a Zstd-19 + horizontal-predictor lossless TIFF at `<rec>/<raw.name>.tif` (top level), verifies byte-equality after decompression before atomic-renaming into place, then deletes the now-redundant `shifted_*.tif` and the orphaned hardlink twin `detection/final/.../data.bin`. Per-recording disk drops from ~20 GB to ~6.5 GB (~50–60% size reduction on the raw). If you later need to re-detect, `load_existing_preprocess` regenerates the shifted from the compressed raw in ~1 min — total re-detect cost goes from ~5 min to ~6 min, the acceptable trade for the disk savings. Every step has an opt-out via `params` (`archive_post_detection`, `compress_raw_post_detection`, `delete_shifted_post_detection`, `delete_final_data_bin_post_detection`, `delete_external_raw_after_archive`, `raw_compression_level`); destructive deletion of the user's external raw at its source path is the only opt-in (off by default).

→ See `tabs/suite2p/README.md` for: Sparsery + Cellpose merge logic, neuropil-correction math, the two dF/F baseline modes (rolling vs first-N-minutes), the GPU dF/F path (CuPy), and how the cell-filter CNN is loaded and applied.

**Exporting figures.** Once detection finishes (or after "Load existing panels"), the **Export figures** button writes the detection ROI overlay panels (`all_rois.{png,svg}`, `kept_rois.{png,svg}`) to `<save_folder>/calliope_figures/detection/` along with a `manifest.json` at `<save_folder>/calliope_figures/manifest.json` capturing the calliope git SHA, the GCaMP variant, tau, fs, cell counts, and the cellfilter checkpoint SHA-256. The export is blocked when the curation popout has un-promoted flips — promote first so the exported figures match what you're looking at. For the full per-stage figure set (detection + lowpass + events + clustering + crosscorrelation + spatial), run the recording through **Tab 0 — Batch runner**; `core/batch_pipeline.py` writes the same `calliope_figures/<stage>/` tree plus manifest as a normal part of the batch pipeline.

---

## Tab 4 — Low-pass filter

**What it does.** Lets you interactively pick a **low-pass cutoff** (in Hz) and watch three panels update:

1. The **FFT power spectrum** of the chosen trace (mean across cells, median, best cell, or a manual subset). The cutoff line is a vertical dashed marker.
2. The **raw** dF/F trace.
3. The **low-pass-filtered** dF/F trace at the chosen cutoff.

When you click *Compute*, it writes per-cell low-pass and derivative arrays to disk for Tab 5.

**Why it matters biologically.** GCaMP has slow kinetics (rise ~50–200 ms, decay ~hundreds of ms to seconds), so calcium events look like smoothed bumps regardless of how briefly the underlying spikes fired. We want to:

- Suppress **shot noise** (frame-to-frame photon-counting jitter), which is high-frequency.
- Preserve the actual transient (which lives at frequencies up to maybe 5 Hz for fast GCaMPs).

If the cutoff is **too low**, you smear real events together and miss short bursts. **Too high**, and noise drives spurious "events" in Tab 5. The FFT panel is your tool for picking a cutoff at the elbow between signal and noise.

The first **derivative** of the low-pass trace (Savitzky-Golay smoothed) is what Tab 5 uses for onset detection — *we don't threshold on dF/F itself; we threshold on its rate of rise*, because that detects the moment a cell starts firing rather than the moment its calcium has decayed back to baseline.

**Biological output.** A cleaned per-cell trace and its derivative, ready for event detection.

→ See `tabs/lowpass/README.md` for the causal Butterworth SOS filter, the SG-derivative formula, and the on-disk memmap layout.

---

## Tab 5 — Event detection

**What it does.** Two levels of "what's happening?":

1. **Per-ROI onsets.** For each cell, it computes a robust z-score of the derivative (using median + MAD instead of mean + std, which is robust to outliers — see Stern et al. 2024) and runs **hysteresis thresholding**: a cell starts firing when its derivative-z crosses the *enter* threshold (default `3.5σ`), and stops when it falls below the *exit* threshold (default `1.5σ`). The dual threshold prevents flickering at the boundary. Onsets within `0.1 s` are merged.
2. **Population events.** Once you have onsets for every cell, you build a histogram across time (`bin_sec=0.025` → 25 ms bins), Gaussian-smooth it (`sigma=1.5 bins`), then `find_peaks` on the smoothed density to find moments where many cells fired in a narrow window. Around each peak, the algorithm walks outward until the density falls back to baseline + `k·noise`, then enforces a hard physiological cap (default `0.5 s` — anything longer than this in epileptiform recordings is probably two events fused). If two walked windows overlap, they get *watershed-split* at the local minimum between their peaks.

The three panels show: (1) sorted-by-activity heatmap of the low-pass dF/F, (2) sorted event raster (one dot per onset), (3) the smoothed density with detected event windows shaded.

A **Prominence distribution...** button opens a popout with the histogram of every candidate peak's prominence (re-runs `find_peaks` against the cached smoothed density with the prominence floor lifted to ~0). Drag the slider along the X axis to preview where `min_prominence` cuts the distribution — the bimodality between noise ripples (near zero) and real events (clearly higher) is usually obvious, and the slider lets the user drop the threshold into the valley between the two modes. Apply re-renders Tab 5 automatically.

**Why it matters biologically.** This is the bridge from "calcium signals" to "events that a neurophysiologist would talk about." A **per-ROI onset** is the closest you get to a spike train without doing deconvolution. A **population event** is the calcium-imaging equivalent of an EEG ictal/interictal spike — the kind of thing you'd point to in a paper and say "the slice seized for 0.4 seconds, here are the cells that participated."

The defaults are tuned for **short epileptiform events** (<0.5 s). If you're studying a slower phenomenon (e.g. spreading depolarization, tens of seconds), you'd loosen the duration cap and lengthen the rolling baseline window.

**Biological output.** (a) Per-cell spike-train-like onset times, (b) population event windows in seconds, (c) for every event × cell combination, did that cell participate?

→ See `tabs/event_detection/README.md` for the full mathematical specification of MAD-z, hysteresis, the density-based detector, and every parameter in `EventDetectionParams`.

---

## Tab 6 — Clustering

**What it does.** Z-scores each cell's dF/F trace, computes pairwise Euclidean distance (which on z-scored data is equivalent to `1 − Pearson r` up to a constant: `||x − y||² = 2T · (1 − r)` so pair-ranking is identical to correlation distance), and runs **hierarchical clustering with Ward linkage** — minimum within-cluster variance, produces compact balanced clusters. Output: a dendrogram + a spatial map where every cell is colored by its cluster.

The user can:
- Slide a horizontal "cut" line up/down the dendrogram to control how fine or coarse the clustering is. Auto mode picks a target of 4–5 clusters.
- Pick a categorical or continuous palette, or set per-cluster colors manually.
- **Click any ROI on the spatial map** to open a popout showing the heatmap + event raster restricted to that ROI's cluster (per-row min-max ΔF/F + Tab 5 onsets when available, else a derivative-threshold fallback). The clicked ROI is highlighted with a yellow row marker.
- "Recluster" a selected branch (or branches) in a separate window — useful when one cluster is huge and looks heterogeneous.
- Export each cluster's ROI ids to `.npy` files for Tab 7.

**Why it matters biologically.** Cells in a slice are not independent. They form **functional ensembles**: groups whose firing patterns are tightly correlated, often because they share inputs or are synaptically coupled. Identifying these ensembles is half of what circuit neuroscience cares about.

In epileptiform tissue specifically, you often see:
- A **core cluster** of cells that participate in nearly every seizure-like event.
- **Satellite clusters** that only join during particular events.
- **Quiet clusters** that look correlated only because they share the same low-amplitude noise (these are the ones a manual recluster usually splits up).

The spatial map then asks the obvious follow-up: *do these functional clusters correspond to anatomical structure?* If cluster colors form contiguous patches, you're seeing local micro-circuits. If they're salt-and-pepper, you're seeing distributed networks.

**Biological output.** Per-cell cluster assignments + a per-cluster ROI list, ready to ask "does cluster A lead cluster B?"

→ See `tabs/clustering/README.md` for the z-scored Euclidean / Ward-linkage algorithm, the `auto_choose_threshold` heuristic, and the export format consumed by Tab 7.

---

## Tab 7 — Cross-correlation

**What it does.** For each pair of clusters (Cᵢ, Cⱼ), and for each pair of cells (one from Cᵢ, one from Cⱼ), it computes the **time-lagged Pearson correlation**: shift cell-A's trace by `k` frames, ask "how correlated is shifted-A with B?", repeat for `k ∈ [−L, +L]`, and report:

- `best_lag_sec`: the lag at which correlation is highest (in seconds; positive ⇒ A leads B by that many seconds).
- `max_corr`: the correlation value at that lag.
- `zero_lag_corr`: correlation at `lag = 0` (instantaneous co-firing).

Two modes:
- **Full recording** — uses the entire dF/F trace.
- **Per event** — re-runs cross-correlation cropped to each event window from Tab 5, so you can ask "during *this* seizure, who led whom?"

A **violin plot** window summarises the distribution of best lags and zero-lag correlations across all cell pairs in each cluster pair. Pairs whose mean lag is significantly non-zero (sign-flip permutation test, `p < 0.05`) are coloured **blue (lead)** or **red (lag)**; ns pairs are **gray**.

Per-pair p-values from a **circular-shift null distribution** (default 500 shuffles; "shuffles" GUI field) are written into the summary CSV alongside `max_corr`, with a Benjamini-Hochberg-adjusted `p_value_fdr` column for thresholding across the N×N matrix. The null preserves each ROI's autocorrelation while jittering the inter-cluster alignment, which is the field-standard fix for the autocorrelation-inflated false-positive rate that parametric tests suffer on calcium imaging traces (Cheng et al. eLife 2023, [doi:10.7554/eLife.81279](https://doi.org/10.7554/eLife.81279)). Set the field to 0 to skip the null computation.

Pearson r is computed on filtered dF/F. **GCaMP rise + decay convolve every spike with a ~tens-to-hundreds-of-ms exponential**, so the visible CCG peak is blurred by the indicator's autocorrelation and correlation magnitudes are inflated relative to spike-time correlations (Yatsenko/Mishne, eLife 2021, [doi:10.7554/eLife.68046](https://doi.org/10.7554/eLife.68046)). The apparent lag resolution is indicator- not frame-rate-limited.

**Why it matters biologically.** Synchronisation is necessary but not sufficient. *Direction* matters:

- Two cells firing together at lag = 0 may share an input.
- Cell A consistently leading cell B by 30 ms suggests A → B.
- A cluster that consistently leads other clusters during seizures is a **candidate initiation zone** — the place where ictal activity is born and from which it propagates. This is the kind of finding that maps directly onto clinical questions about which brain region to target with surgery or stimulation.

The pipeline's purpose, in the end, is to take a dish full of neurons firing chaotically on a microscope slide and turn it into a directed graph: *this region drives that region, in these events, with this latency.*

**Biological output.** Per cluster pair, distributions of lead/lag latency and synchrony strength — both globally and event-by-event.

→ See `tabs/crosscorrelation/README.md` for the batched matmul algorithm (one matrix multiply per lag covers every cell pair in a cluster pair), the GPU/CPU paths (CuPy), the sign-flip permutation test for significance, and the per-event windowing logic.

---

## Tab 8 — Spatial propagation

**What it does.** For each population event detected in Tab 5, shows three figures: a top pair of side-by-side activation-order maps and a full-width **directional monotonicity analysis** underneath. The **top-left** is the plain activation-order map: each cell is coloured by *when it fired within that event*, on a continuous cyan → blue → red scale (cyan = earliest, red = latest), with non-participating ROIs left grey — useful as a clean reference image. The **top-right** is the same map with white arrows overlaid that connect the centroid of the cells firing in each frame to the centroid for the next active frame, so you can read the temporal trajectory directly off the spatial layout. The **bottom panel** asks the harder question: "is there a *direction* of propagation, and how strong is it?" — sweep θ ∈ [0°, 360°), project each ROI's `(x_i, y_i)` onto `u(θ) = (cos θ, sin θ)` to get `s_i(θ)`, then compute Spearman's rank correlation `ρ(θ) = SpearmanCorr(s_i(θ), t_i)`. The angle that maximises ρ is the dominant propagation axis; its ρ value (0 = no direction, 1 = perfectly monotone along that axis) is the strength of the relationship. Significance comes from a permutation test that shuffles activation times across cells and re-runs the full θ sweep per shuffle (so the p-value corrects for the multiple-direction search baked into the max-over-θ statistic). Three subplots: ROI positions coloured by `t_i` with a θ* arrow scaled by `ρ_obs`, the `ρ(θ)` curve with the peak marked, and the permutation null histogram with `ρ_obs` overlaid + the empirical p-value. A spinner and Prev / Next buttons let you flip through events one at a time.

This tab is a **pure consumer of Tab 5**: it doesn't re-detect anything. Tab 5 publishes its event windows, active-mask, and per-ROI first-onset times via the shared `AppState`, and Tab 8 subscribes. Whatever knobs you tuned on Tab 5 (manual ROI subset, advanced parameters, etc.) are exactly what gets visualised — re-render Tab 5 and Tab 8 updates automatically.

**Why it matters biologically.** Tab 7 tells you "cluster A consistently leads cluster B by 47 ms". Tab 8 tells you what that *looks like* in space — the very thing that's missing from the violin plots. If event after event lights up cyan in the upper-left and progresses to red in the lower-right, you're seeing a stereotyped propagation wave: the kind of result that lets you point at a specific anatomical region as the seizure-initiation zone. Per-event maps also expose **heterogeneity**: maybe most events propagate left-to-right, but a few flip direction — that's a meaningful biological observation that gets averaged away in a single summary plot. The directional monotonicity test on the bottom panel makes that "direction or no direction?" question quantitative per event: a `ρ_obs ≈ 1` event has a sharp, reproducible propagation axis; a `ρ_obs ≈ 0` event is spatially synchronous or chaotic, and the high p-value tells you not to read tea-leaves into the centroid arrows above.

V1 ships only the activation-order map. Future view modes (planned, ported from the legacy `spatial_heatmap_updated.py` reference): per-event propagation arrows + speed/angle CSV, per-ROI relative-lag violin plots, and scalar feature maps (event rate, peak ΔF/F, peak derivative-z) painted on the same canvas.

**Biological output.** A click-through atlas of "who fired first, who fired last" for every detected population event in the recording.

→ See `tabs/spatial_propagation/README.md` for the inputs, the order-rank → painted-image pipeline (two helpers in `core/spatial.py`), the PARAM_SPEC defaults, and the planned roadmap of additional view modes.

---

## Putting it all together — a worked example

You record a 10-minute, 15 fps calcium movie of a brain slice in low-magnesium aCSF (a classic seizure-induction protocol). What does the pipeline tell you?

| Tab | What you learn |
|---|---|
| 1–2 | The recording is stable, no obvious drift, ~200 candidate cell-shaped objects in the field of view. |
| 3 | Suite2p + Cellpose find 312 ROIs; the cell filter accepts 187 as real cells. dF/F is computed; baseline is a rolling 10th-percentile over 45 s. |
| 4 | The FFT shows clear signal up to ~3 Hz and white noise above. You pick a 1 Hz cutoff. |
| 5 | 187 cells produce 4,200 onsets across the recording. The population-event detector finds 23 events, mean duration 0.32 s, well under the 0.5 s cap — consistent with brief interictal spikes. |
| 6 | Hierarchical clustering at the auto cut groups the cells into 4 clusters. The spatial map shows cluster C1 is a tight clump in the upper-left quadrant; C2 is a more diffuse ring; C3 and C4 are scattered. |
| 7 | Across all 23 events, C1 leads C2 by a mean of 47 ms (`p < 0.001`), and C2 leads C3 by 22 ms (`p < 0.01`). C4's lags are not significantly different from zero — it follows the population without driving it. |
| 8 | Clicking through the 23 per-event activation-order maps, the upper-left quadrant lights up cyan in 19/23 events with red trailing toward the lower-right — a stereotyped propagation pattern that visualises the C1 → C2 → C3 lead/lag found in Tab 7. |

**Biological story.** The slice has a focal seizure-initiation zone (C1, upper-left), which recruits a peri-focal ring (C2), which then drives a more distributed downstream cluster (C3), while a fourth cluster (C4) participates passively. **This is the kind of result that justifies a paper.**

---

## Where to look in the code

```
src/calliope/
├── pipeline_gui.py            ← top-level Tk app, builds the 9 tabs (0-8)
├── pipeline_gui_walkthrough.md  ← (this file)
├── core/                       ← pure-Python algorithms shared across tabs
│   ├── preprocessing.py        ← Tab 1: shift / mean / QC GIF + run_preprocess
│   ├── sparse_plus_cellpose.py ← Suite2p + Cellpose merge
│   ├── adaptive_detection.py   ← register-only + sparsery loops; suite2p 1.0 db/settings adapter
│   ├── utils.py                ← dF/F, lowpass, mad_z, event detection, fps lookup
│   ├── crosscorrelation.py     ← Tab 7: batched + single-pair xcorr + run_crosscorrelation
│   ├── clustering.py           ← Tab 6: palettes + auto-threshold + dendrogram + spatial paint + run_clustering
│   ├── spatial.py              ← Tab 8: cyan→red colormap + order-rank painting + render_spatial_event_figures
│   ├── scale.py                ← pix↔µm helpers (zoom-based + direct override)
│   ├── detection_run.py        ← headless Tab 3 pipeline (spc + dF/F + cellfilter + filtered_dff)
│   ├── lowpass_run.py          ← headless Tab 4 (compute + figures)
│   ├── event_detection_run.py  ← headless Tab 5 (compute + figures + summary write)
│   ├── batch_pipeline.py       ← run_recording orchestrator (Tabs 1+3-8 chained)
│   └── cellfilter/             ← PyTorch CNN for the keep/drop classifier
└── tabs/
    ├── batch/                  ← Tab 0 (BatchTab + BatchRow + Tab 1-style TIFF picker)
    ├── preprocess/             ← Tab 1
    ├── qc/                     ← Tab 2
    ├── suite2p/                ← Tab 3
    ├── lowpass/                ← Tab 4
    ├── event_detection/        ← Tab 5
    ├── clustering/             ← Tab 6
    ├── crosscorrelation/       ← Tab 7
    └── spatial_propagation/    ← Tab 8
```

Each tab folder contains:
- `tab.py` — the Tk widgets, threading, plotting, and disk I/O.
- `logic.py` — a re-export shim that pulls computational functions from `core/`.
- `README.md` — the math + reproducibility notes for that tab (read these in order to recreate the pipeline from scratch).

Each interactive Tab in 1, 3-8 delegates its actual computation to the matching `core/<tab>_run.py` so Tab 0's batch worker uses identical code. If you want to run a single stage from a Python script without the GUI, the `core/*_run.py` modules are the entry points.

---

## Suggested reading order

1. **This file**, end-to-end. Get the biology in your head.
2. `tabs/preprocess/README.md` and `tabs/suite2p/README.md` — the data flow from TIFF to dF/F.
3. `tabs/lowpass/README.md` and `tabs/event_detection/README.md` — how raw fluorescence becomes spike-like events.
4. `tabs/clustering/README.md` and `tabs/crosscorrelation/README.md` — how events become circuit-level statements.
5. `tabs/spatial_propagation/README.md` — how those statements translate back into pictures of the slice.
6. `tabs/batch/README.md` — once you understand any single stage, this shows you how to run all of them across many recordings without touching the GUI.
7. `tabs/qc/README.md` — last (it's the smallest tab).

By the end, you should be able to re-implement any single stage of the pipeline from scratch given the math and parameter values in its README.
