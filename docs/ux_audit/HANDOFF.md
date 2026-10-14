# Handoff prompt for a local agent

Paste everything below the line into a local Claude Code session opened
at the root of a `calliope` checkout.

---

You are picking up a usability and correctness audit of CalLIOPE, a
customtkinter desktop GUI that turns two-photon calcium-imaging TIFFs
into cells, dF/F, events, clusters, cross-correlation and propagation
maps (tabs 0 to 8). A cloud session already drove the GUI end to end and
wrote up the findings. Your job is to finish the parts the cloud
sandbox couldn't do, then fix the findings in priority order.

## Start here

1. `git fetch origin claude/relaxed-carson-zxn6ip` and check that branch
   out (or branch from it). Everything below lives in `docs/ux_audit/`:
   - `README.md`: the findings, numbered 1 to 18 plus papercuts, each
     with evidence and a suggested fix. Read it fully first.
   - `screenshots/`: the GUI state behind each finding.
   - `tools/make_fake_recordings.py <out_dir>`: synthetic recordings
     with ground truth in `*_truth.npz` (cell centres, burst frames,
     fs, per-cell dF/F). `rec_wave` has 70 cells and 8 bursts that sweep
     top-left to bottom-right over 0.4 s at 15.08 Hz. `rec_split` is
     30 Hz, split over two TIFFs with ImageJ `finterval` set.
     `rec_noise` has no cells.
   - `tools/gui_driver.py <workdir>` + `tools/send.py <workdir> "<code>"`:
     launch the real GUI and run Python snippets on its Tk thread
     (helpers: `app`, `tab(i)`, `click(widget, text)`, `shot(name)`,
     `widgets(...)`, `NEXT_PATHS` for scripted file dialogs). Every
     messagebox is logged to `<workdir>/popups.log` and auto-answered.
     On Linux without a display, wrap the driver in `xvfb-run -a`.
   - `tools/import_fall_mat.py <Fall.mat> <run_folder>`: converts a
     Suite2p `Fall.mat` into CalLIOPE's
     `<run>/detection/final/suite2p/plane0` layout, then runs dF/F, the
     cell filter and filtered dF/F.
2. Install exactly as `README.md` (repo root) says: Python 3.11 or 3.12
   venv, `pip install -r requirements.txt`, `pip install -e . --no-deps`.
   Run `pytest tests/` once to get a baseline before you change
   anything.

## Phase 1: finish the testing the cloud couldn't

The cloud sandbox had no GPU, 15.7 GB of RAM, and a network policy that
blocked huggingface.co and zenodo.org. Two of its numbers come from
workarounds, so re-check them on this machine:

- The **Cellpose pass was stubbed out** (the model download from
  huggingface.co failed). Run Tab 3 for real on `rec_wave` and record
  ROI counts and timings. Also re-check finding 12 (noise recording
  "failed" instead of `no_rois`): real Cellpose may find some blobs on
  `rec_noise`.
- **The OOM crash (finding 1)** was worked around by setting
  `utils.RAM_OVERHEAD_MULTIPLIER = 80` in the running process. Reproduce
  the crash with the stock setting and record this machine's total RAM,
  the `[batch_size]` line Tab 3 logs, and peak RSS. Watch the GUI
  process, not just the system total.
- **Real raw movie.** The example data used so far
  (`github.com/Stern-MA/2PCISz_RecDetect`, `ExampleData.zip`) has only a
  `Fall.mat`, no movie. Get raw seizure TIFFs from Zenodo record 5931656
  (Niemeyer et al., Brain 2022, larval zebrafish PTZ; one ~7.2 GB zip,
  `ImagingDataFiles_BRAIN.zip`). If disk or bandwidth is tight, use
  `remotezip` to pull one TIFF out of the zip. Extract it into its own
  empty folder. Check its frame rate and pixel size from the paper or
  the TIFF metadata, then run it through Tabs 1 to 8. Note anything that
  isn't already in the report.
- If a CUDA GPU is present, also run with the GPU paths on (dF/F, xcorr
  via CuPy) and note any differences.

Add what you learn to `docs/ux_audit/README.md` in its existing format
(evidence, then a fix). Correct any finding that turns out to be an
artifact of the cloud workarounds.

## Phase 2: fix, in this order

Work through these as separate, reviewable commits (or small PRs), each
with a regression test under `tests/` in the style of the existing
ones. Re-run the driver on the synthetic data after each group to
confirm the fix shows up in the GUI, not just in a unit test.

1. **Crash and silent overrides.** Finding 1: compute the budget from
   available RAM, recalibrate the CPU multiplier against a measured
   peak, and run detection in a child process so an OOM can't kill the
   GUI. Finding 2: the user's `batch_size` from "Edit suite2p
   settings..." is overwritten at
   `src/calliope/core/suite2p_pipeline.py:933-934`; only auto-pick when
   unset.
2. **Frame rate (finding 3).** Read fs from ImageJ `finterval`,
   ScanImage and OME metadata in Tab 1. Show it as an editable field in
   Tab 1, Tab 3's header and a batch-table column. Require explicit
   confirmation when it's unknown instead of silently using 15.08 Hz.
   Do the same for µm/px. The test: `rec_split` must run at 30 Hz with
   no user input, and its event times must match `*_truth.npz`.
3. **Statistics bugs.** Finding 6: within-cluster xcorr must use i < j
   pairs only and skip clusters with fewer than 2 ROIs. Fix the pair
   statistics before touching the violin plot. Finding 7: give the auto
   cut a minimum cluster size, and show cluster sizes in the status line.
4. **Workflow and state.** Finding 9: Cellpose model preflight, a clear
   error message, a skip option and a local model path. Finding 10:
   Export in Tab 6 must refresh Tab 7. Finding 11: "Open run folder"
   must clear every tab. Finding 12: zero merged ROIs becomes `no_rois`,
   and the batch log shows the error. Finding 13: auto-group
   multi-file recordings in batch Scan. Finding 14: an "Import Suite2p
   output..." button (`tools/import_fall_mat.py` is a working
   reference). Finding 16: a friendly tkinter check plus a README note.
   Finding 17: correct the torch comment in `requirements.txt` and
   document a CPU-only install.
5. **Papercuts** (section D of the report), cheapest first.

## Ask Richard before changing these

These change scientific defaults or what results mean. Propose them, with
evidence from the synthetic ground truth and the real data, and wait for
a decision:

- Finding 4: the within-event timing estimator, and whether Tab 8 should
  warn when timing jitter is comparable to the activation spread.
- Finding 5: seizure handling (merging capped windows, event classes,
  per-class duration caps).
- Finding 8: cell-filter keep-threshold UI and the Suite2p `iscell`
  fallback.
- The dF/F baseline default (first-N-minutes vs rolling percentile).
- Finding 15: auto-archiving after interactive detection.
- Finding 18: which tab the app opens on.

## Gotchas from the first session

- Any script that launches the GUI must keep its top-level code under
  `if __name__ == "__main__":`. CalLIOPE offloads work with
  multiprocessing `spawn`, which re-imports the launching script in
  every child process. Without the guard, each child starts a second GUI
  (and a second driver that steals your commands).
- Don't kill processes with `pkill -f <pattern>` when the pattern also
  appears in your own shell's command line; it kills your shell.
- `requirements.txt` pins `suite2p==1.0.0.1` exactly (the code
  monkey-patches its internals). Don't upgrade it.
- The cell filter, Tab 4 and later tabs read the *filtered* memmaps
  anchored to `r0p7_cell_mask_bool.npy`, not the live mask. Keep that in
  mind for any fix touching curation or the cell filter.

## Done means

- Phase 1 results are written into the report.
- Every Phase 2 item you took on has a test, the existing suite passes,
  and a fresh driver run on `rec_wave` and `rec_split` shows the fix.
- A short summary for Richard lists what was fixed, what was deferred
  and why, and the open questions from "Ask Richard first", with your
  recommendation for each.
