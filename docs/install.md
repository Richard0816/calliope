# Installing CalLIOPE

> **Python 3.11 or 3.12 is required.** The tested scientific stack
> (NumPy 2.4, pandas 3.0, Suite2p 1.0.0.1) needs Python ≥3.11 — on 3.10
> or earlier `pip` silently back-solves to an older, untested set of
> packages (a common cause of "works in the GUI but the Detection tab
> errors"). On 3.13 or newer there are no matching wheels yet, so `pip`
> falls back to compiling packages from source and fails without a C++
> toolchain (e.g. `Could not find vswhere.exe`). The default button on
> python.org installs the newest release — pick a **3.12.x** installer
> explicitly, then build the venv with `py -3.12 -m venv venv`. Check
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

## Reproducible install (recommended)

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

`requirements.txt` pins the plain PyPI `torch` wheel, which is the CPU
build on Windows and macOS (it runs on any machine). For NVIDIA GPU
acceleration, reinstall the CUDA build afterwards:

```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126
```

> **venv folder name.** The pip path above (and the README) uses `venv`;
> `uv sync` creates `.venv`. Either works. Just use the same name when
> activating, deleting or rebuilding it.

## Latest-compatible install (development)

Pulls the newest versions allowed by the ranges in `pyproject.toml` —
handy for development, not guaranteed identical to the tested set:

```bash
pip install -e .            # runtime
pip install -e ".[dev]"     # + pytest, then: pytest tests/
```

## conda / mamba

conda supplies the compiled scientific stack and PyTorch from
conda-forge, which avoids the heaviest source builds. `pip install -e .`
then adds the rest from PyPI: Suite2p (pinned exactly to `1.0.0.1`),
cellpose and customtkinter.

```bash
conda create -n calliope python=3.11        # 3.11 or 3.12
conda activate calliope
conda install -c conda-forge numpy pandas scipy matplotlib openpyxl \
    pillow psutil tifffile imagecodecs numba pytorch
pip install -e .                            # Suite2p, cellpose, customtkinter + CalLIOPE
```

`mamba` is a faster drop-in for `conda`.

## GPU acceleration (optional)

The GPU paths — Tab 3's GPU dF/F (`use_gpu_dff`, first-N baseline mode)
and Tab 7's GPU cross-correlation — need a CuPy wheel matching your CUDA
toolkit (`nvcc --version` / `CUDA_PATH`); without it CalLIOPE falls back
to the CPU / NumPy path automatically.

```bash
pip install -e ".[gpu-cuda11]"   # CUDA 11.2 – 11.8
pip install -e ".[gpu-cuda12]"   # CUDA 12.x
```

> **CUDA 13.** CuPy ships wheels only for CUDA 11 / 12. The rest of the
> pipeline (PyTorch, Suite2p) is fine on CUDA 13; for the GPU extra,
> install the CUDA 12 toolkit alongside and use `".[gpu-cuda12]"`.

## NWB export (optional)

Writing NWB files for DANDI (Tab 3's *Export NWB* button, or the batch
`export_nwb` option) needs the `nwb` extra (`pynwb` + `nwbinspector`).
The base pipeline never imports them; without them the export reports
that pynwb is missing and everything else runs normally.

```bash
pip install -e ".[nwb]"
```

## First-run note

The first time you run **Detection**, cellpose downloads its segmentation
model (one-time, needs internet). CalLIOPE's own cell-filter checkpoint is
bundled in the package, so it needs no download.
