# CalLIOPE

**Cal**cium **L**ive-imaging **O**utput **P**ipeline for **E**pileptiform-recordings.

A desktop GUI that takes raw two-photon calcium-imaging TIFFs and turns
them into cells, dF/F traces, events, clusters, and propagation maps.
Every step is a tab; you click through them left to right.

## Quick start

You need **Python 3.11 or 3.12** (not 3.13+) and a few GB of disk.
Install on a local drive, not a network drive.

**1. Get the code.** Either click the green **Code** button on GitHub →
**Download ZIP** and unzip it, or:

```bash
git clone https://github.com/Richard0816/calliope.git
```

**2. Install.** Open a terminal in the unzipped/cloned folder and run:

```bash
python -m venv venv
venv\Scripts\activate              # Windows
# source venv/bin/activate         # macOS / Linux
pip install -r requirements.txt
pip install -e . --no-deps
```

The first line must use Python 3.11 or 3.12 — on Windows,
`py -3.12 -m venv venv` picks it explicitly. Check with
`python --version` after activating.

**3. Run.**

```bash
calliope
```

**4. Analyse a recording.** Load a TIFF on the first tab and work
through the tabs in order. The first Detection run downloads the
cellpose model (~1 GB, one time).

## Next steps

- [Pipeline walkthrough](src/calliope/pipeline_gui_walkthrough.md) — what
  each tab does and why, for someone new to calcium imaging.
- [Install options](docs/install.md) — uv, conda, GPU acceleration,
  development install.
- [Troubleshooting](docs/troubleshooting.md) — install errors, Detection
  failures, slow first runs.
- Each tab folder under `src/calliope/tabs/` has a `README.md` covering
  the maths and parameters.

## Pipeline stages

1. **Preprocess** — intensity shift to a non-negative uint16 TIFF, mean image, QC GIF.
2. **QC preview** — inspect the QC movie and mean image before heavy compute.
3. **Detection** — Suite2p registration + Suite2p/Cellpose ROI extraction, dF/F, cell-filter model.
4. **Low-pass filter** — Butterworth low-pass + Savitzky-Golay derivative.
5. **Event detection** — per-ROI onsets and population event windows.
6. **Clustering** — hierarchical clustering of filtered traces.
7. **Cross-correlation** — ROI×ROI xcorr, optional GPU via CuPy.
8. **Spatial propagation** — per-event spatial figures on the mean image.

A **Batch runner** tab queues recordings and chains all eight stages.

## Changes in this release

Changes that affect existing outputs, scripts or batch files:

- **NWB export.** `iscell_prob` now holds the Suite2p classifier
  probability (`iscell.npy` column 1). The 0/1 accept label moved to a
  new `iscell` column. New columns: `cellfilter_prob` (CalLIOPE
  cell-filter probability) and `kept` (1 = ROI kept for downstream
  analysis).
- **GPU fallback.** On a machine with CuPy installed but no reachable
  CUDA device, GPU stages now fall back to the CPU path instead of
  failing at the first GPU operation.
- **Batch report.** `batch_report.csv` has a row-level `error` column
  with the failing stage's message.
- **Batch parameters.** New per-row keys: `zdrift_qc`,
  `keep_registration_cache`, `n_shuffles`, `compute_partial`. Removed:
  `cutoff_min` / `cutoff_max` (they only set Tab 4's slider range and
  had no effect in a batch run). A blank or missing batch cell now means
  the batch default, not the value the previous row left in the tab.
- **Clustering export.** `cluster_results/` also saves
  `linkage_roi_ids.npy` (the Suite2p ids the linkage was built on);
  Reload and Export refuse when the kept ROIs have changed since.
- **`meta.json`.** `dff.baseline_min_sec` is now in seconds (it held
  minutes under a seconds name).
- **Detection outputs.** Detection no longer writes the unfiltered
  `r0p7_dff_lowpass.memmap.float32` / `r0p7_dff_dt.memmap.float32`
  (nothing read them; Tab 4's `r0p7_filtered_dff_lowpass` / `_dt` are
  unchanged). The three Tab 3 settings that only shaped them,
  `default_lowpass_hz`, `default_sg_win_ms` and `default_sg_poly`, are
  removed; old settings or batch files that still carry them load and
  ignore them. Copies of the two files already on disk are unused and
  can be deleted. `meta.json`'s `lowpass` section is no longer stamped.
- **Dependencies.** `seaborn` and `scikit-image` are no longer
  required.

## License

MIT.
