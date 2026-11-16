# Troubleshooting

Symptoms below are grouped by the stage where they show up. Each lists the
usual root cause and the fix.

## Install fails

**`pip` downloads `*.tar.gz` and tries to compile / `meson` / `Could not
find vswhere.exe` / `metadata-generation-failed`.**
No prebuilt wheel matched your interpreter, so `pip` fell back to a source
build that needs a C++ toolchain. Almost always the Python version is
newer than 3.12. Confirm with:

```powershell
python -c "import platform; print(platform.python_version(), platform.architecture()[0])"
```

If it isn't `3.12.x` (or `3.11.x`) `64bit`, install a 3.12 release and
rebuild the venv: `deactivate; Remove-Item -Recurse -Force venv;
py -3.12 -m venv venv`. (The pip install in the README uses `venv`; if
you installed with `uv sync`, the folder is `.venv` instead.)

**`AssertionError` deep in `pip`'s `_install_wheel` (after everything
downloads), or `Cache entry deserialization failed` warnings.**
The venv is on a network/UNC drive (a mapped home like `H:\` →
`\\server\user\...`). Recreate the project and venv on a **local** disk
(`C:\...` or `%LOCALAPPDATA%\...`). Venvs are not supported on network
shares.

**`PowerShell ... Activate.ps1 cannot be loaded because running scripts
is disabled`.**
Run `Set-ExecutionPolicy -Scope Process RemoteSigned` once in that window,
then re-run the activate line. Or use `venv\Scripts\activate.bat` from
`cmd` (`.venv\Scripts\activate.bat` for a `uv sync` install).

## Detection (Tab 3) fails

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

## First run is slow / downloads ~1 GB

Expected. The first Detection run downloads the Cellpose-SAM model
(`cpsam`, ~1.2 GB, one time, needs internet) and on a CPU-only machine
each Cellpose pass takes several minutes. CalLIOPE's own cell-filter
checkpoint is bundled and never downloads.

The model is saved to `~/.cellpose/models/cpsam`
(`C:\Users\<you>\.cellpose\models\cpsam` on Windows,
`/Users/<you>/.cellpose/models/cpsam` on macOS), or to the folder in the
`CELLPOSE_LOCAL_MODELS_PATH` environment variable if that is set. Tab 3
checks for that file before every run and only asks "Download it during
this run?" when it is missing.

**Asked to download the Cellpose model on every run.** The download is
not ending up in that file. Check whether it exists and is ~1.2 GB after
a run. If not, the download is being blocked (network proxy or
firewall to huggingface.co, a full disk, or a sync/antivirus tool
removing the folder). Any one of these fixes it:

- Download `https://huggingface.co/mouseland/cellpose-sam/resolve/main/cpsam`
  in a browser and save it as `~/.cellpose/models/cpsam` (no extension).
- Save the file anywhere and set Tab 3 → Advanced… → "Cellpose model
  file" to it.
- Untick "Run Cellpose pass" in Tab 3 → Advanced… to detect with
  Sparsery only (no model needed).

## GPU extra installed but nothing is faster

`pip install -e ".[gpu-cuda11]"` / `".[gpu-cuda12]"` only helps on a
machine with a working NVIDIA GPU and a matching CUDA runtime; it adds
CuPy for Tab 3's GPU dF/F path and Tab 7's cross-correlation. Without one, CalLIOPE falls back to
NumPy automatically — the extra is harmless but does nothing. Note the
default `requirements.txt` `torch` is the **CPU** build; for GPU compute
also reinstall the CUDA wheel:
`pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126`.
