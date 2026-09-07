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

1. **Preprocess** — rigid registration + intensity normalization.
2. **QC preview** — inspect the registered recording before heavy compute.
3. **Detection** — Suite2p + Cellpose ROI extraction, dF/F, cell-filter model.
4. **Low-pass filter** — Butterworth low-pass + Savitzky-Golay derivative.
5. **Event detection** — per-ROI onsets and population event windows.
6. **Clustering** — hierarchical clustering of filtered traces.
7. **Cross-correlation** — ROI×ROI xcorr, optional GPU via CuPy.
8. **Spatial propagation** — per-event spatial figures on the mean image.

A **Batch runner** tab queues recordings and chains all eight stages.

## License

MIT.
