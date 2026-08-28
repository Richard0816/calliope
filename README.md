# CalLIOPE

**Cal**cium **L**ive-imaging **O**utput **P**ipeline for **E**pileptiform-recordings.

A self-contained GUI for the Suite2p-based 2-photon calcium imaging
analysis pipeline. Each pipeline stage lives on its own tab; shared
backend modules (preprocessing, signal processing, hierarchical
clustering, cross-correlation, cellpose-based detection, the
cell-filter PyTorch model) ship inside the package.

## Pipeline stages

Raw two-photon TIFFs flow through:

1. **Preprocess** — rigid registration + intensity normalization;
   writes a shifted TIFF, mean image, and QC GIF in one pass.
2. **QC preview** — inspect the shifted recording and mean image
   before committing compute.
3. **Detection** — Suite2p sparsery + Cellpose ROI extraction, dF/F
   computation, and a PyTorch cell-filter that prunes non-neuronal ROIs.
4. **Low-pass filter** — Butterworth low-pass + Savitzky-Golay
   derivative; writes filtered + derivative memmaps per recording.
5. **Event detection** — per-ROI hysteresis onsets and population-level
   event windows from the dF/F density.
6. **Clustering** — hierarchical clustering of the filtered traces with
   an auto-threshold or user-pinned cut height.
7. **Cross-correlation** — full-recording and per-event ROI×ROI xcorr,
   with optional GPU acceleration via CuPy.
8. **Spatial propagation** — per-event spatial figures over the
   recording's mean image.

The **Batch runner** (Tab 0) queues recordings and chains all eight
stages per row, with per-recording parameter overrides persisted to a
JSON sidecar and round-tripped through CSV.

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
├── tests/
│   └── test_imports.py             # pytest smoke test (imports + headless GUI walk)
└── src/
    └── calliope/                   # the actual Python package
        ├── pipeline_gui.py         # GUI entry point
        ├── gui_common.py           # shared GUI helpers (AppState, AdvancedDialog)
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

Each tab subfolder has a `tab.py` (the widget tree) and a `logic.py`
for pure compute / I/O helpers + a re-export shim into
`calliope.core`. Tabs 3, 6, and 7 also have one or more `*_popout.py`
files for detail windows (curation, cluster heatmap+raster, recluster
sub-tree, violin plots).

## Installing

> **Python 3.11 or 3.12 is required.** The tested scientific stack
> (NumPy 2.4, pandas 3.0, Suite2p 1.0.0.1) needs Python ≥3.11 — on 3.10
> or earlier `pip` silently back-solves to an older, untested set of
> packages (a common cause of "works in the GUI but the Detection tab
> errors"). On 3.13 or newer there are no matching wheels yet, so `pip`
> falls back to compiling packages from source and fails without a C++
> toolchain (e.g. `Could not find vswhere.exe`). The default button on
> python.org installs the newest release — pick a **3.12.x** installer
> explicitly, then build the venv with `py -3.12 -m venv .venv`. Check
> with `python --version`.

> **Install onto a local disk, not a network/UNC drive.** Creating the
> venv on a mapped network home (e.g. `H:\` → `\\files.example.ca\...`)
> makes `pip` fail during wheel install with a bare `AssertionError`,
> and even if it installed, Suite2p's memory-mapped binaries are slow
> and lock-prone over SMB. Clone and build under `C:\` (or
> `%LOCALAPPDATA%`), and keep your imaging data on a local disk too.

CalLIOPE pins `suite2p==1.0.0.1` exactly (the detection code patches that
specific Suite2p release) and constrains the rest of the stack to tested
ranges. For a guaranteed-reproducible environment use one of the two
**locked** paths below; the loose path is fine for development but can
drift as upstream releases move.

### Reproducible install (recommended)

Two equivalent paths reproduce the exact tested environment.

**With [uv](https://docs.astral.sh/uv/)** (uses the committed `uv.lock`):

```bash
uv sync                 # builds .venv at the exact locked versions
uv run calliope         # launch
```

**With pip + the pinned lockfile:**

```bash
python -m venv venv                       # from a Python 3.11 / 3.12 interpreter
venv\Scripts\activate                     # Windows
# source venv/bin/activate                # macOS / Linux
pip install -r requirements.txt           # exact tested versions (incl. PyTorch)
pip install -e . --no-deps                # CalLIOPE itself; deps already satisfied
```

`requirements.txt` pins the CUDA 12.6 PyTorch wheel, which is
self-contained and also runs on CPU-only machines. For a smaller
CPU-only download, change the index URL at the top of `requirements.txt`
to `https://download.pytorch.org/whl/cpu` and drop the `+cu126` suffix
from the torch line before installing.

### Latest-compatible install (development)

Pulls the newest versions allowed by the ranges in `pyproject.toml` —
handy for development, not guaranteed identical to the tested set:

```bash
pip install -e .            # runtime
pip install -e ".[dev]"     # + pytest, then: pytest tests/
```

### conda / mamba

Suite2p, cellpose, and PyTorch have well-tested conda-forge builds, so
pulling them through conda first avoids the heaviest source builds:

```bash
conda create -n calliope python=3.11        # 3.11 or 3.12
conda activate calliope
conda install -c conda-forge numpy pandas scipy matplotlib seaborn \
    scikit-image openpyxl pillow psutil tifffile pytorch
pip install -e .                            # CalLIOPE on top of the conda stack
```

`mamba` is a faster drop-in for `conda`.

### GPU acceleration (optional)

The GPU cross-correlation path needs a CuPy wheel matching your CUDA
toolkit (`nvcc --version` / `CUDA_PATH`); without it CalLIOPE falls back
to NumPy automatically.

```bash
pip install -e ".[gpu-cuda11]"   # CUDA 11.2 – 11.8
pip install -e ".[gpu-cuda12]"   # CUDA 12.x
```

> **CUDA 13.** CuPy ships wheels only for CUDA 11 / 12. The rest of the
> pipeline (PyTorch, Suite2p) is fine on CUDA 13; for the GPU extra,
> install the CUDA 12 toolkit alongside and use `".[gpu-cuda12]"`.

### First-run note

The first time you run **Detection**, cellpose downloads its segmentation
model (one-time, needs internet). CalLIOPE's own cell-filter checkpoint is
bundled in the package, so it needs no download.

## Running the GUI

After `pip install`:

```bash
calliope
```

Or as a module:

```bash
python -m calliope
```

## Troubleshooting

Symptoms below are grouped by the stage where they show up. Each lists the
usual root cause and the fix.

### Install fails

**`pip` downloads `*.tar.gz` and tries to compile / `meson` / `Could not
find vswhere.exe` / `metadata-generation-failed`.**
No prebuilt wheel matched your interpreter, so `pip` fell back to a source
build that needs a C++ toolchain. Almost always the Python version is
newer than 3.12. Confirm with:

```powershell
python -c "import platform; print(platform.python_version(), platform.architecture()[0])"
```

If it isn't `3.12.x` (or `3.11.x`) `64bit`, install a 3.12 release and
rebuild the venv: `deactivate; Remove-Item -Recurse -Force .venv;
py -3.12 -m venv .venv`.

**`AssertionError` deep in `pip`'s `_install_wheel` (after everything
downloads), or `Cache entry deserialization failed` warnings.**
The venv is on a network/UNC drive (a mapped home like `H:\` →
`\\server\user\...`). Recreate the project and venv on a **local** disk
(`C:\...` or `%LOCALAPPDATA%\...`). Venvs are not supported on network
shares.

**`PowerShell ... Activate.ps1 cannot be loaded because running scripts
is disabled`.**
Run `Set-ExecutionPolicy -Scope Process RemoteSigned` once in that window,
then re-run the activate line. Or use `.\.venv\Scripts\activate.bat` from
`cmd`.

### Detection (Tab 3) fails

**`AssertionError: Torch not compiled with CUDA enabled`** (in
`dcnv.preprocess` / `baseline_maximin`).
You're on a CPU-only `torch` build (`torch ... +cpu`) on a machine with no
working CUDA GPU. This is fixed in current CalLIOPE (the deconvolution
baseline now follows the resolved device) — `git pull` to update. If you
must stay on an older checkout, the pipeline otherwise runs fine on CPU,
just slowly.

**`Unable to allocate N GiB for an array ...` even though Task Manager
shows plenty of free RAM.**
On Windows the binding limit is the *commit charge* (RAM + pagefile), not
"in use" RAM — a single large contiguous allocation can fail when commit
is near the limit. Check it:

```powershell
$os=Get-CimInstance Win32_OperatingSystem
[PSCustomObject]@{
  CommitLimit_GB=[math]::Round($os.TotalVirtualMemorySize/1MB,1)
  CommitUsed_GB =[math]::Round(($os.TotalVirtualMemorySize-$os.FreeVirtualMemory)/1MB,1)
}
```

If `CommitUsed` is close to `CommitLimit`, close other memory-hungry
processes (or reboot) and make sure the pagefile is system-managed
(System → About → Advanced system settings → Performance → Virtual
memory). CalLIOPE already disables Suite2p's PC registration-quality
metrics (`run.do_regmetrics`) by default to avoid one large transient
allocation; you can re-enable it in Tab 3's "Edit suite2p settings…"
popout if you want the QC numbers.

**Suite2p registration is extremely slow, or `WinError 8` / stale
`stat`/dF/F mismatch errors appear.**
The recording (or its `detection/` output) is on a network drive. Move the
raw TIFFs and output folder to a local disk and re-run.

### First run is slow / downloads ~1 GB

Expected. The first Detection run downloads the cellpose segmentation
model (one-time, needs internet) to `~/.cellpose`, and on a CPU-only
machine each cellpose pass takes several minutes. CalLIOPE's own
cell-filter checkpoint is bundled and needs no download.

### GPU extra installed but nothing is faster

`pip install -e ".[gpu-cuda11]"` / `".[gpu-cuda12]"` only helps on a
machine with a working NVIDIA GPU and a matching CUDA runtime; it adds
CuPy for the cross-correlation step. Without one, CalLIOPE falls back to
NumPy automatically — the extra is harmless but does nothing. Note the
default `requirements.txt` `torch` is the **CPU** build; for GPU compute
also reinstall the CUDA wheel:
`pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126`.

## Dependencies

- `customtkinter` (GUI framework)
- `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`
- `scikit-image`, `openpyxl`, `Pillow`, `psutil`, `tifffile`, `imagecodecs`
- `torch` (cell-filter model)
- `suite2p`, `cellpose` (detection)
- `cupy` (optional, GPU cross-correlation — see CUDA note above)
- `pytest` (optional, `[dev]` extra)
