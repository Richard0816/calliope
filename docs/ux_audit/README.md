# CalLIOPE user walkthrough: annoyances and fixes (2026-10-14)

CalLIOPE was driven end to end through its GUI, the way a new user would
use it. It ran under a virtual display, one click or text entry at a time,
with a screenshot after each step. Every popup was logged. Three inputs
were used:

| Input | What it is | Why |
|---|---|---|
| `rec_wave.tif` (synthetic) | 256x256, 2700 frames @ 15.08 Hz, 70 GCaMP-like somata, 8 seizure-like bursts sweeping top-left to bottom-right over 0.4 s, 12% photobleaching, +-1.5 px jitter, signed int16 with negative offset | Ground truth for every stage (cells, events, propagation direction) |
| `rec_split_part001/002.tif` (synthetic) | One 30 Hz recording split over two TIFFs, ImageJ `finterval` = 1/30 s | Multi-file grouping and frame-rate handling |
| `rec_noise.tif` (synthetic) | Noise only, no cells | The batch "no cells" path |
| `s75_PTZexp1` (real) | Suite2p `Fall.mat` from [Stern-MA/2PCISz_RecDetect](https://github.com/Stern-MA/2PCISz_RecDetect) ExampleData: mouse PTZ seizure, 512x512, 30 Hz, 36,030 frames (20 min), 138 ROIs | Real seizure dynamics in Tabs 4 to 8. The zip has no raw movie, so Tabs 1 to 3 could not use it. |

`tools/make_fake_recordings.py` regenerates the synthetic data and its
`*_truth.npz` ground truth (`python tools/make_fake_recordings.py <out_dir>`). `tools/gui_driver.py` + `tools/send.py` drive the GUI one command at a time; `tools/import_fall_mat.py` bridges a Suite2p `Fall.mat` into a CalLIOPE run folder.

Environment: Linux, Python 3.12, 4 CPU cores, 15.7 GB RAM, no GPU.

The cloud sandbox blocked huggingface.co, so the Cellpose model could not
download. To get past Tab 3, the Cellpose pass was stubbed to return no
ROIs, so detection ran on Sparsery alone. The batch-size constant was
also raised to stop the OOM crash (finding 1). Every other step used the
GUI as shipped.

## Scorecard: did the science come out right?

| Stage | Result on synthetic ground truth |
|---|---|
| Preprocess | OK, 8 s. Negative offset shifted correctly. |
| Detection (Sparsery only) | 73 ROIs for 70 true cells; cell filter kept 71. Good. |
| Low-pass | OK, 16 s. |
| Events | All 8 bursts found, plus 2 chance coincidences (the auto threshold is permissive by design). Good. |
| Clustering | Clusters follow the wave spatially. Good, but includes a 1-ROI cluster (finding 7). |
| Cross-correlation | Cluster lags in the right direction (+66 ms, +199 ms). Good, but the stats include self-pairs (finding 6). |
| Spatial propagation | **Failed at defaults.** 0/8 bursts show significant propagation; per-cell timing error IQR 0.36 s against a 0.40 s wave (finding 4). |
| 30 Hz recording in batch | **Wrong.** Ran at an assumed 15.08 Hz: every time and duration doubled, 2 spurious events (finding 3). |

## Re-test on a workstation (2026-10-14/15)

The cloud pass stubbed Cellpose and had no GPU. This pass re-ran what it
couldn't, on Windows 11, Python 3.11, 32 GB RAM, RTX 3050 6 GB, real
Cellpose 4.1.1, GUI driven by `tools/gui_driver.py`. Memory was shared
with other jobs, so heavy runs were capped.

| Check | Result |
|---|---|
| Real Cellpose on `rec_wave` (GPU, stock settings) | Cellpose 70 ROIs (70/70 true cells, 0 false); Sparsery 73 (70/70 + 3); merged 73; cell filter kept 71. Preprocess 16 s, detection 137 s incl. archive, peak RSS 3.8 GiB. |
| Finding 1 OOM with a GPU | Not reproduced: the GPU clamp splits registration into 4 batches. The OOM is a CPU-path problem. The `[batch_size]` lines went to process stdout, not the Tab 3 console. |
| Finding 12 on `rec_noise`, real Cellpose | Different from the cloud: Sparsery found 0 ROIs, but **Cellpose found 18 blobs** (median 3,454 px, max 8,218), so the run went on to extraction and the process tree grew past 15 GiB until it was stopped. A size cap on Cellpose ROIs now drops these (see below). |
| Finding 3, `rec_split` in batch | After the fix: 30 Hz read from `finterval` with no input; 6/6 bursts; Batch Scan grouped the two parts. |
| Finding 4, timing vs truth (Cellpose ROIs) | Detection onsets: bias -0.20 s, IQR 0.30 s, rho 0.14, wave significant 3/8. Half-max crossing on 5 Hz zero-phase dF/F: bias +0.00 s, IQR 0.05 s, rho 0.82, **8/8** (`rec_split`: rho 0.93, 6/6). |
| Finding 5 cause (real PTZ, Tab 5 GUI defaults) | **Correction:** the fragmentation is not the 1.3 s cap. The 357-381 s seizure became 26 back-to-back windows, **none** at the cap; window gaps inside seizures are 0.05-0.2 s. Merging windows with gaps <= 1 s gives exactly the three seizures (357-381, 400-406, 414-429 s). |
| Finding 8 (real PTZ) | Cell filter kept 46/138; Suite2p `iscell` 113. prob >= 0.3 keeps 113, >= 0.2 keeps 125. The `Fall.mat` `iscell[:,0]` is a human curation (classifier prob >= 0.5 keeps 90): against it the CNN's AUROC is 0.91, Suite2p's 0.88, and their mean 0.98. The CNN ranks well but is under-confident on foreign data. |
| dF/F baseline (real PTZ) | With the first-2-min baseline the late median dF/F is -1.20. The drift warning now fires. |
| Tab 5 freeze (real lab recording, 679 kept ROIs) | Detection already ran off-thread (30 s); the GUI froze ~23 s afterwards in three openpyxl load/save cycles of `calliope_summary.xlsx` on the Tk thread. |
| Blob ROIs on a real lab recording | All 5 kept blobs (1.2-8.8 k px, cell-filter prob 0.87-1.0) came from the Cellpose pass, whose real cells were ~80 px; Sparsery ROIs are ~20 px. |
| Manual subset from a Tab 6 cluster (52 ROIs) | Tab 4 titles ran off the figure. |

## Status on branch `ux-audit-fixes`

| Finding | Status |
|---|---|
| 1 OOM | Detection runs in a child process; RAM budget uses available RAM. CPU multiplier not yet recalibrated (needs a measured CPU run). |
| 2 batch_size override | Fixed. |
| 3 frame rate | Read from ImageJ / ScanImage / OME metadata, shown and editable in Tab 3, confirmation when unknown. Batch-table column not done. |
| 4 timing | Half-max within-event timing (Advanced: "Within-event timing"). Tab 8 jitter warning not done. |
| 5 seizures | Back-to-back windows merged (`merge_gap_s`, default 1 s); classes local burst / population burst / seizure-like. Real PTZ: 86 windows -> 35 events. |
| 6, 7, 9-14, 16, 17 | Fixed (Cellpose prefetch progress bar and an "Open run" mismatch banner not done; the latter is unneeded since tabs are rebuilt). |
| 8 cell filter | "Keep ROIs by" row in Tab 3 (prob cutoff or Suite2p iscell), kept/iscell readout with a low-keep hint, Apply without re-running the CNN. |
| 15 archive | Interactive runs no longer archive; batch still does. |
| 18 start tab | Opens on the last tab picked; Tab 1 on first launch. |
| New | Tab 5 summary write off the GUI thread; Cellpose ROI size cap (`cellpose_max_area_factor`, default 4x a typical Cellpose cell); subset titles clipped with the full list in the status line. |
| dF/F baseline | Kept as first-N minutes (decision), with the drift warning. |

## Findings, most severe first

### A. Wrong results or lost work

**1. Tab 3 crashes the whole app (OOM kill) on a 354 MB movie.**
The window disappears with no message, taking all unsaved state with it.
`utils.pick_batch_size` budgets `0.8 x total RAM - 4 GiB` and models
16 bytes per pixel per frame. For 256x256 frames that predicts 2.7 GiB
and picks `batch_size=8000`, so all frames go in one batch. Measured
peak was 13.8 GB. With the multiplier raised 10x it still peaked at
9.2 GB during batch detection. The budget also uses *total* RAM, not
*available* RAM.
- Fix: run detection in a child process (like Tabs 1 and 4 to 7), so an
  OOM kills the worker and the GUI can report it.
- Fix: base the budget on `psutil.virtual_memory().available`.
- Fix: recalibrate the CPU multiplier against a measured peak, and cap
  CPU batches (e.g. 500 frames).
- Fix: on a RAM OOM in single-recording mode, retry with a smaller batch,
  as batch mode already does.

**2. The user's `batch_size` override is silently discarded.**
Values from "Edit suite2p settings..." are merged in at
`core/suite2p_pipeline.py:893`, then overwritten at line 933-934 by
`get_adaptive_sizer().pick(...)`. The obvious workaround for finding 1
does nothing.
- Fix: only auto-pick when the user has not set a value, and log which
  value won.

**3. Frame rate is silently assumed to be 15.08 Hz.**
A 30 Hz recording whose TIFF carried `finterval=0.0333` was processed at
15.08 Hz. Consequences:
- Every event time doubled (true 48.7 s reported as 97.0 s).
- Every duration doubled, so the "1.3 s" cap is effectively 0.65 s.
- The deconvolution tau is mis-scaled.
- Two spurious events appeared.

The only signal is one `[fps] WARNING` line, buried among PyTorch
warnings in the console. The setting is `fps_override` in Tab 3's
Advanced dialog (and one of 84 per-row keys in batch), where `0 = auto`
really means "15.08".
- Fix: read the frame rate in Tab 1 from ImageJ `finterval`, ScanImage
  `SI.hRoiManager.scanFrameRate`, or OME metadata, and show it next to
  the TIFF list as an editable field.
- Fix: show it in Tab 3's header and as a column in the batch table.
- Fix: if it can't be determined, require the user to confirm it before
  Run, rather than defaulting silently.
- The same applies to pixel size (µm/px), which drives the scale bars.

**4. Tab 8's propagation maps can't recover a known wave at default settings.**
Per-cell first-activation times compared with ground truth:

| Cutoff (Tab 4) | Timing bias | Timing IQR | rho(estimated, true) | Tab 8 significant |
|---|---|---|---|---|
| 1 Hz (default) | -0.22 s | 0.36 s | ~0.1 | 0 / 8 |
| 3 Hz | -0.21 s | 0.19 s | 0.26 | 4 / 8 |

The default smoothing and onset backtracking are tuned for *detecting*
events, but they add timing jitter about as large as a typical
propagation delay. The map then shows confident-looking colour noise.
- Fix: estimate within-event timing on a separate path, e.g. half-max
  crossing on a lightly filtered trace, or each cell's lag to the
  event's population template, with sub-frame interpolation.
- Fix: report a per-event timing-uncertainty estimate.
- Fix: show a warning on Tab 8 when the jitter is comparable to the
  event's activation spread.
- Fix: look into the constant -0.2 s bias (backtrack window
  `max(tau, 0.8/cutoff)` = 0.8 s at 1 Hz).

**5. Real seizures are cut into dozens of "events".**
On the real PTZ recording, the seizure (~355 to 380 s) and the following
sustained wave (~395 to 445 s) came out as about 60 back-to-back windows
of 0.5 to 1.3 s (86 events total). That is because of the hard
`max_event_duration_s = 1.3` cap. For a tool aimed at epileptiform data,
the seizure itself is the most important event.
- Fix: merge runs of abutting capped windows into one "sustained event"
  and classify events (interictal spike, seizure, spreading wave), with
  each class getting its own cap.
- At minimum, warn when many windows hit the cap back to back.

**6. Within-cluster cross-correlation includes self-pairs and duplicates.**
`C2xC2_summary.csv` has 676 rows for 26 ROIs: all ordered pairs,
including 26 self-pairs with r = 1, lag 0, p = 0.002. These rows go into
the FDR correction and the violin plots. The singleton cluster `C1xC1`
is reported as r = 1.0, p = 0.002. The violin plot also marks C1 as
"significantly lagging" (***), but that is one cell's lag counted 26
times (pseudo-replication).
- Fix: use only i < j pairs within a cluster.
- Fix: skip clusters with fewer than 2 ROIs (or a configurable minimum)
  in pair statistics.
- Fix: run the significance test on ROI-level summaries, not on pairs
  that reuse the same cell.

**7. The auto cut makes tiny clusters.**
Synthetic run: sizes 29/26/15/**1**, with the singleton drawn in grey and
nearly invisible. Real run: 7 clusters, three with only **2** ROIs. The
status line says only "4 clusters".
- Fix: set a minimum cluster size in `auto_choose_threshold`.
- Fix: print the sizes in the status line ("4 clusters: 29/26/15/1").
- Fix: draw every cluster in a visible colour.

**8. The bundled cell filter rejects most cells on another lab's data.**
Real recording: CalLIOPE kept **46 of 138** ROIs, where Suite2p's
classifier accepted 113.
- Fix: show both counts and the keep rate in Tab 3.
- Fix: warn when the keep rate is unusually low.
- Fix: offer a probability-threshold slider and a "use Suite2p iscell"
  fallback, so a user isn't forced into curation and retraining just to
  get started.

### B. Workflow and state problems

**9. The Cellpose model downloads mid-run, and failure is cryptic.**
The ~1 GB download starts only after registration and Sparsery have
finished. When huggingface.co is unreachable (proxy, firewall, offline
rig), the popup reads `<urlopen error Tunnel connection failed: 403
Forbidden>`, and there is no way to skip Cellpose.
- Fix: check for or prefetch the model at startup or before Run, with a
  progress bar.
- Fix: catch the failure and explain it ("Couldn't download the Cellpose
  model from huggingface.co; place `cpsam` in ~/.cellpose/models or
  untick Cellpose").
- Fix: add a "Skip Cellpose pass" option and a local model path field.

**10. Tab 7 doesn't see clusters exported from Tab 6.**
After "Export *_rois.npy", Tab 7 still says "Cluster files or dF/F
memmap missing." and both Run buttons stay disabled until the user finds
"Reload dF/F & clusters". This happened on both recordings.
- Fix: have Export publish `set_clusters_ready`, or write the cluster
  files automatically when the analysis finishes.

**11. "Open run folder" leaves other tabs on the previous recording.**
After opening the real run:
- Tab 2 still showed `rec_wave`'s QC movie.
- Tab 8 still said "Synced from Tab 5: 10 events, 71 ROIs. Recording:
  rec_wave".
- Tab 3's console still held the old recording's archive log.

Exporting from Tab 8 at that point would mix recordings. It also lands
on Tab 2 even when the run has no QC outputs.
- Fix: clear every tab on open (or on any `plane0` change).
- Fix: show a mismatch banner whenever a tab's recording differs from the
  sidebar's "now viewing" label.
- Fix: land on the first stage that actually loaded.

**12. Batch: an empty recording "fails" instead of finishing as `no_rois`.**
When Sparsery and Cellpose both find nothing,
`sparse_plus_cellpose` raises `RuntimeError: merge produced 0 ROIs`.
The documented `no_rois` guard only covers ROIs that are detected and
then all rejected by the cell filter. The batch run log shows only
"'rec_noise' failed in 20.6s" (the reason appears only in
batch_report.csv and Tab 3's console).
- Fix: treat zero merged ROIs as `no_rois`.
- Fix: print the error message in the batch run log line.

**13. Batch Scan doesn't group multi-file recordings.**
Each part file becomes its own row. The real example recording was
acquired as 41 files (`..._00001_00001.tif` to `_00041.tif`), so one
recording means 41 rows to tick and merge by hand.
- Fix: group files automatically when they share a stem and differ only
  in a trailing index (`_\d+$`, `_partNNN`), with an option to undo.

**14. There's no way to start from an existing Suite2p result.**
A user with `Fall.mat` or a `suite2p/plane0` folder (as in the example
data) can't load it from the GUI. Converting it and calling
`compute_dff_memmaps` / `predict_cell_filter` / `compute_filtered_dff`
took a script.
- Fix: add an "Import Suite2p output..." button (plane0 folder or
  `Fall.mat`) that writes the CalLIOPE layout and runs dF/F plus the
  cell filter.

**15. Interactive detection archives the raw data automatically.**
After a single interactive Tab 3 run, CalLIOPE immediately compresses
the raw TIFF and deletes the shifted TIFF and `data.bin`. "Run
detection" stays disabled until that finishes. That makes sense for
overnight batches, but it's surprising after a one-off exploratory run.
- Fix: default archiving off for interactive runs (keep it on in batch),
  or ask once.

### C. Install and launch

**16. On Linux, a missing tkinter is only discovered at launch.**
`pip install` succeeds, then `calliope` dies with
`ModuleNotFoundError: No module named 'tkinter'`.
- Fix: mention `sudo apt install python3.12-tk` (or the distro
  equivalent) in the README.
- Fix: check for tkinter in `main()` and print that hint instead of a
  traceback.

**17. The install is 5.5 GB on Linux.**
`requirements.txt` says torch is "the CPU build from PyPI", but on Linux
PyPI serves `torch 2.11.0+cu130` plus 2.7 GB of NVIDIA libraries.
- Fix: correct the comment.
- Fix: document a CPU-only install (`--index-url
  https://download.pytorch.org/whl/cpu`) for laptops.

**18. The app opens on Tab 0 (Batch runner).**
The README says "load a TIFF on the first tab"; a new user lands on the
densest power-user screen.
- Fix: open on Tab 1, or remember the last tab.

### D. Papercuts

**Tab 1**
- Typing or pasting a working directory doesn't fill the TIFF list or
  the output root; only Browse does. Run then fails with "Missing
  output". Bind `<Return>`/`<FocusOut>` (or a trace on `dir_var`) to
  refresh and set the default output root.
- The default output root is `<workdir parent>/data`, which gives
  `data/data/...` when the TIFFs already live under `data/`.
- Reloading existing outputs reports "Done - n_frames=?".

**Tab 2**
- The QC GIF plays at native size (tiny next to the mean image).
- The frame counter reads "frame 98/?", with no scrubber and no time in
  seconds.
- There is no mean-intensity-over-time trace, which is the quickest way
  to spot the bleaching and z-drift the docs say to check here.
- Matplotlib figures are white on a dark UI (in every tab); use a dark
  mpl style.

**Tab 3**
- The status text is clipped at the right edge.
- The indeterminate progress bar animates while idle (also Tabs 4 to 7).
- It says "No recording loaded." while the status says "Ready".
- The console is cluttered with PyTorch sparse-tensor `UserWarning`s;
  filter them so warnings that matter are visible.

**dF/F baseline**
- The default "First 2 minutes" baseline doesn't correct photobleaching:
  on the synthetic data, dF/F drifts to about -0.1 by the end.
- Consider rolling-percentile as the default, or detrend.
- At least warn when the late-recording median dF/F is far from 0.

**Tab 4**
- The FFT is a raw periodogram, so finding "the elbow" is guesswork.
  Use a Welch or smoothed PSD and suggest a cutoff.
- The FFT fills the viewport, so the trace panels are below the fold.
- The "Low-pass dF/F" preview uses zero-phase filtering, while Compute
  writes causal output. Label the preview, or overlay the causal trace.

**Tab 5**
- The heatmap and raster x-axes are in "downsampled bins", whose width
  changes per recording (133 ms vs 1000 ms). Use seconds.
- The status text runs under the "Reload from folder..." button.
- The legend covers the density trace.

**Tab 7**
- There's no progress or ETA; full xcorr on the real data took 6 min.
- The violin popout is a fixed size that clips its x labels, and its
  legend overlaps the significance stars.
- It doesn't say whether it shows full-recording or per-event results.

## Timings (4-core CPU, no GPU)

| Step | rec_wave (2700 x 256^2) | Real PTZ (36,030 frames, 138 ROIs) |
|---|---|---|
| Install (pip, README steps) | 2 min, 5.5 GB | |
| Preprocess | 8 s | n/a (no movie) |
| Detection (Sparsery only, after OOM workaround) | ~15 s + archive | n/a (imported Fall.mat; dF/F 0.7 s, cell filter 2.8 s) |
| Low-pass compute | 16 s | ~1 s |
| Event render | 4 s | 13 s |
| Clustering | 1 s | 1 s |
| Full xcorr (500 shuffles) | 14 s | 348 s |
| Per-event xcorr | 5 s | not run |
| Batch, 2 recordings | 80 s total; peak RAM 9.2 GB | |

## Screenshots

`screenshots/` has the key frames referenced above: launch, QC, Tab 3
before and after, Tab 4 FFT and traces, Tab 5, Tab 6, the Tab 7 violin,
Tab 8, and the real-data Tab 3 and Tab 5.

## Raw seizure movies for a future pass

The real example data contains only extracted traces. A search for
public raw two-photon seizure movies found one usable candidate:
Zenodo 5931656 (Niemeyer et al., *Brain* 2022, larval zebrafish PTZ,
`ImagingDataFiles_BRAIN.zip`, ~7.2 GB of motion-corrected TIFFs). It
could not be downloaded from this sandbox. DANDI's seizure datasets are
electrophysiology only.
