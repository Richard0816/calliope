# CalLIOPE

**Cal**cium **L**ive-imaging **O**utput **P**ipeline for **E**piletiform-recordings.

A self-contained Tk GUI for the Suite2p-based 2-photon calcium imaging
analysis pipeline. Each pipeline stage lives on its own tab; shared
backend modules (preprocessing, signal processing, hierarchical
clustering, cross-correlation, cellpose-based detection, the cell-filter
PyTorch model) ship inside the package.

## Layout

Standard src layout — repo root holds `pyproject.toml`, `README.md`,
and the package source under `src/calliope/`:

```
calliope/                           # repo root (where you run pip install)
├── pyproject.toml
├── README.md
├── .gitignore
└── src/
    └── calliope/                   # the actual Python package
        ├── pipeline_gui.py         # PipelineApp coordinator (tabs assembled here)
        ├── gui_common.py           # AppState, AdvancedDialog, Tk helpers
        ├── plot_data_export.py     # "Save data..." button writer
        ├── core/                   # backend modules used by the tabs
        │   ├── preprocessing.py        # raw TIFF -> shifted TIFF + QC gif + mean
        │   ├── summary_writer.py       # cross-recording XLSX writer
        │   ├── clustering_cmap.py      # palette + dendrogram helpers
        │   ├── crosscorrelation.py     # batched ROI x ROI cross-correlation
        │   ├── utils.py                # signal processing + suite2p I/O helpers
        │   ├── adaptive_detection.py   # ROI detection (cellpose-based)
        │   ├── brute_force_ops.py      # cellpose pass orchestration
        │   ├── sparse_plus_cellpose.py # full detection pipeline
        │   └── cellfilter/             # PyTorch cell-filter model + dataset
        ├── tabs/
        │   ├── preprocess/         # Tab 1: Input & Preprocess
        │   ├── qc/                 # Tab 2: QC Preview
        │   ├── suite2p/            # Tab 3: Suite2p Detection
        │   ├── lowpass/            # Tab 4: Low-pass filter
        │   ├── event_detection/    # Tab 5: Event detection
        │   ├── clustering/         # Tab 6: Clustering
        │   └── crosscorrelation/   # Tab 7: Cross-correlation
        └── data/                   # bundled resource files (.npy / .csv)
```

Each tab subfolder has a `tab.py` (the Tk class) and a `logic.py`
re-exporting just the slice of `calliope.core` the tab calls.

## Installing

### Using pip

Editable install (development), from the repo root:

```bash
pip install -e .
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

- `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`
- `scikit-image`, `openpyxl`, `Pillow`, `psutil`, `tifffile`
- `torch` (cell-filter model)
- `suite2p`, `cellpose` (detection)
- `cupy` (optional, GPU cross-correlation)

## Publishing

This repo is already in the standard src layout, so PyPI publishing
is straightforward:

```bash
pip install build twine
python -m build               # builds sdist + wheel into dist/
twine upload dist/*           # upload to PyPI (or ``--repository testpypi``)
```
