# CalLIOPE

**Cal**cium **L**ive-imaging **O**utput **P**ipeline for **E**piletiform-recordings.

A self-contained Tk GUI for the Suite2p-based 2-photon calcium imaging
analysis pipeline. Each pipeline stage lives on its own tab; shared
backend modules (preprocessing, signal processing, hierarchical
clustering, cross-correlation, cellpose-based detection, the cell-filter
PyTorch model) ship inside the package.

## Layout

```
calliope/
├── pipeline_gui.py             # PipelineApp coordinator (tabs assembled here)
├── gui_common.py               # AppState, AdvancedDialog, Tk helpers
├── plot_data_export.py         # "Save data..." button writer
├── core/                       # backend modules used by the tabs
│   ├── preprocessing.py        # raw TIFF -> shifted TIFF + QC gif + mean
│   ├── summary_writer.py       # cross-recording XLSX writer
│   ├── clustering_cmap.py      # palette + dendrogram helpers
│   ├── crosscorrelation.py     # batched ROI x ROI cross-correlation
│   ├── utils.py                # signal processing + suite2p I/O helpers
│   ├── adaptive_detection.py   # ROI detection (cellpose-based)
│   ├── brute_force_ops.py      # cellpose pass orchestration
│   ├── sparse_plus_cellpose.py # full detection pipeline
│   └── cellfilter/             # PyTorch cell-filter model + dataset
└── tabs/
    ├── preprocess/             # Tab 1: Input & Preprocess
    ├── qc/                     # Tab 2: QC Preview
    ├── suite2p/                # Tab 3: Suite2p Detection
    ├── lowpass/                # Tab 4: Low-pass filter
    ├── event_detection/        # Tab 5: Event detection
    ├── clustering/             # Tab 6: Clustering
    └── crosscorrelation/       # Tab 7: Cross-correlation
```

Each tab subfolder has a `tab.py` (the Tk class) and a `logic.py`
re-exporting just the slice of `calliope.core` the tab calls.

Bundled resources (Suite2p ops `.npy`, AAV metadata `.csv`) live under
`calliope/data/`.

## Running the GUI

From the directory containing the `calliope/` package:

```bash
python -m calliope
```

Or after `pip install`:

```bash
calliope
```

## Installing

Editable install (development):

```bash
pip install -e .
```

(Run from inside the `calliope/` directory, where `pyproject.toml` lives.)

For GPU cross-correlation, install the optional `gpu` extras:

```bash
pip install -e ".[gpu]"
```

## Dependencies

- `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`
- `scikit-image`, `openpyxl`, `Pillow`, `psutil`, `tifffile`
- `torch` (cell-filter model)
- `suite2p`, `cellpose` (detection)
- `cupy` (optional, GPU cross-correlation)

## Publishing notes

The current layout has `pyproject.toml` *inside* the `calliope/` source
directory and uses Hatchling's `[tool.hatch.build.targets.wheel.sources]`
table to remap the build root onto `calliope/` in the wheel. This works
for `pip install -e .` and `python -m build`, but the conventional PyPI
layout is to wrap the package in a parent project folder:

```
calliope-project/
├── pyproject.toml              # move it up one level
├── README.md                   # move it up one level
├── LICENSE
└── calliope/                   # this directory, untouched
    ├── __init__.py
    └── ...
```

When migrating to that layout, drop the `[tool.hatch.build.targets.*.sources]`
remapping from `pyproject.toml` (it's no longer needed).
