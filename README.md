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

Editable install (development), from the repo root:

```bash
pip install -e .
```

For GPU cross-correlation, install the optional `gpu` extras:

```bash
pip install -e ".[gpu]"
```

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
