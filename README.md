# CalLIOPE

**Cal**cium **L**ive-imaging **O**utput **P**ipeline for **E**piletiform-recordings.

A self-contained **customtkinter** dark-mode GUI for the Suite2p-based
2-photon calcium imaging analysis pipeline. Each pipeline stage lives
on its own tab; shared backend modules (preprocessing, signal
processing, hierarchical clustering, cross-correlation, cellpose-based
detection, the cell-filter PyTorch model) ship inside the package.

## UI / Look & feel

- **Dark mode** via [customtkinter](https://customtkinter.tomschimansky.com/).
  Matplotlib figures keep their white facecolor (dark frame, light
  plots) so plot contrast stays maximised.
- **Sidebar navigation** — left-hand vertical bar with one button per
  tab. Click to swap content; the active tab is highlighted with the
  CTk accent colour. Replaces the old top tab-bar.
- **Drag-resizable panels.** Every tab that hosts a log console
  (Tabs 0 / 1 / 3 / 7) and every multi-panel layout exposes draggable
  resize grips so the user can give one panel more room without
  squeezing its neighbours. The tab body grows; the scrollable wrapper
  absorbs the extra height.
- **Hover-scroll** — spin the mouse wheel anywhere on a tab to scroll
  it (no need to aim at the scrollbar).
- **Resizable popouts** — the prominence, curation, cluster, recluster,
  and violin popout windows all open at a sensible initial size with a
  `minsize` floor; drag any corner to resize.

## Screenshots

Drop screenshots into `docs/screenshots/` matching the filenames
referenced below; the README will render them inline once they exist.

| | |
|---|---|
| Sidebar navigation + dark theme | ![sidebar](docs/screenshots/sidebar.png) |
| Tab 0 — Batch runner with resize grips | ![tab0-batch](docs/screenshots/tab0-batch.png) |
| Tab 3 — Suite2p detection with curation popout | ![tab3-curation](docs/screenshots/tab3-curation.png) |
| Tab 6 — Clustering dendrogram + spatial map | ![tab6-clustering](docs/screenshots/tab6-clustering.png) |

## Layout

Standard src layout — repo root holds `pyproject.toml`, `README.md`,
and the package source under `src/calliope/`:

```
calliope/                           # repo root (where you run pip install)
├── pyproject.toml
├── README.md
├── LICENSE                         # MIT
├── .gitignore
├── .gitattributes                  # LF line endings cross-platform
├── docs/
│   └── screenshots/                # README screenshots land here
├── tests/
│   └── test_imports.py             # pytest smoke test (imports + headless GUI walk)
└── src/
    └── calliope/                   # the actual Python package
        ├── pipeline_gui.py         # PipelineApp coordinator (sidebar + content host)
        ├── gui_common.py           # AppState, AdvancedDialog, palette + Tk helpers
        ├── plot_data_export.py     # "Save data..." button writer
        ├── core/                   # backend modules used by the tabs
        │   ├── preprocessing.py        # raw TIFF -> shifted TIFF + QC gif + mean
        │   ├── summary_writer.py       # cross-recording XLSX writer
        │   ├── clustering.py           # hierarchical clustering + palette helpers
        │   ├── crosscorrelation.py     # batched ROI x ROI cross-correlation
        │   ├── utils.py                # signal processing + suite2p I/O helpers
        │   ├── sparse_plus_cellpose.py # full detection pipeline (cellpose + Suite2p)
        │   ├── suite2p_pipeline.py     # native Suite2p (db, settings) wrapper
        │   ├── detection_run.py        # post-detection prune + archive
        │   ├── lowpass_run.py          # low-pass + derivative memmap writer
        │   └── cellfilter/             # PyTorch cell-filter model + dataset
        ├── tabs/
        │   ├── batch/              # Tab 0: Batch runner (queue + worker)
        │   ├── preprocess/         # Tab 1: Input & Preprocess
        │   ├── qc/                 # Tab 2: QC Preview
        │   ├── suite2p/            # Tab 3: Suite2p Detection
        │   ├── lowpass/            # Tab 4: Low-pass filter
        │   ├── event_detection/    # Tab 5: Event detection
        │   ├── clustering/         # Tab 6: Clustering
        │   ├── crosscorrelation/   # Tab 7: Cross-correlation
        │   └── spatial_propagation/# Tab 8: Spatial propagation
        └── data/                   # bundled resource files (.npy / .csv)
```

Each tab subfolder has a `tab.py` (the customtkinter widget tree) and a
`logic.py` for pure compute / I/O helpers + a re-export shim into
`calliope.core`. Tabs 3, 6, and 7 also have one or more `*_popout.py`
files for resizable detail windows (curation, cluster heatmap+raster,
recluster sub-tree, violin plots).

## Installing

### Using pip

Editable install (development), from the repo root:

```bash
pip install -e .
```

For development (runs the smoke-test suite):

```bash
pip install -e ".[dev]"
pytest tests/
```

For GPU cross-correlation, install the CuPy extra that matches your
CUDA toolkit (`nvcc --version` or check `CUDA_PATH`):

```bash
# CUDA 11.2 - 11.8
pip install -e ".[gpu-cuda11]"

# CUDA 12.x
pip install -e ".[gpu-cuda12]"
```

GPU support is optional — the cross-correlation module falls back to
NumPy automatically if CuPy isn't installed.

> **Note on CUDA 13.** The CuPy team currently ships pre-built wheels
> only for CUDA 11 / 12. CUDA 13 toolkits work fine for the rest of the
> pipeline (PyTorch, Suite2p), but the GPU cross-correlation extras
> above don't have a CuPy wheel to match. Workaround: install the
> CUDA 12 toolkit alongside, then `pip install -e ".[gpu-cuda12]"`.

### Using conda

If you prefer conda (or `mamba`), create a fresh environment and then
install the package with `pip` inside it. Suite2p, cellpose, and PyTorch
all have well-tested conda-forge builds, so pulling them through conda
first avoids the heaviest source builds:

```bash
# create + activate env (Python 3.10 is a known-good pin)
conda create -n calliope python=3.10
conda activate calliope

# pull the heavy scientific stack from conda-forge
conda install -c conda-forge numpy pandas scipy matplotlib seaborn \
    scikit-image openpyxl pillow psutil tifffile pytorch

# install calliope itself (editable install from the repo root)
pip install -e .
```

For GPU cross-correlation inside a conda env, install the matching CuPy
build from conda-forge instead of the pip extra:

```bash
# CUDA 11.x
conda install -c conda-forge cupy cudatoolkit=11.8

# CUDA 12.x
conda install -c conda-forge cupy cuda-version=12
```

Tip: `mamba` is a drop-in replacement for `conda` that resolves the
environment much faster — swap `conda` for `mamba` in any of the
commands above if you have it installed.

## Running the GUI

After `pip install`:

```bash
calliope
```

Or as a module:

```bash
python -m calliope
```

## Dependencies

- `customtkinter` (dark-mode GUI framework)
- `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`
- `scikit-image`, `openpyxl`, `Pillow`, `psutil`, `tifffile`, `imagecodecs`
- `torch` (cell-filter model)
- `suite2p`, `cellpose` (detection)
- `cupy` (optional, GPU cross-correlation — see CUDA note above)
- `pytest` (optional, `[dev]` extra)

## Publishing

This repo is already in the standard src layout, so PyPI publishing
is straightforward:

```bash
pip install build twine
python -m build               # builds sdist + wheel into dist/
twine upload dist/*           # upload to PyPI (or ``--repository testpypi``)
```
