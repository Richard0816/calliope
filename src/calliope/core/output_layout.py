"""User-facing output layout: one ``<recording>/output/`` folder.

Everything a user is meant to open lives here; the rest of the recording
folder (``detection/…/plane0``, memmaps, caches) is CalLIOPE's own::

    <recording>/output/
        calliope_summary.xlsx
        README.txt            (lists every file and the tab that wrote it)
        <recording>.nwb       (if exported)
        figures/manifest.json (parameters + versions of the run)
        figures/<stage>/...   (PNG + SVG)
        data/<stage>/...      (CSV, or .npy for big traces in compact mode)

``EXPORTS`` is the catalogue of optional items; which are on is a
per-user setting in ``~/.calliope/exports.json`` (see docs/
output_redesign_plan.md for the decisions behind the defaults).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

OUTPUT_DIRNAME = "output"
SETTINGS_FILE = Path.home() / ".calliope" / "exports.json"

# Same stage names the figure renderers have always used.
TAB_DIRS = {1: "preprocess", 3: "detection", 4: "lowpass",
            5: "event_detection", 6: "clustering", 7: "crosscorrelation",
            8: "spatial_propagation", 0: "nwb"}

# key -> (tab, label, kind, default). kind: "img" | "data" | "big" (T x N).
EXPORTS: dict[str, tuple[int, str, str, bool]] = {
    "qc_movie": (1, "QC movie (qc.gif)", "img", True),
    "rois_table": (3, "ROI table (centroid, area, cell prob, kept)", "data", True),
    "roi_maps": (3, "All / kept ROI maps", "img", True),
    "trace_f": (3, "Raw F, kept ROIs", "big", False),
    "trace_fneu": (3, "Neuropil Fneu, kept ROIs", "big", False),
    "trace_dff": (3, "Filtered dF/F, kept ROIs", "big", False),
    "trace_spks": (3, "Deconvolved spks, kept ROIs", "big", False),
    "zdrift": (3, "z-drift trace + QC plot", "data", True),
    "lowpass_plots": (4, "FFT / raw / low-pass plots", "img", True),
    "lowpass_plot_data": (4, "Data behind the Tab 4 plots", "data", False),
    "trace_lowpass": (4, "Low-pass dF/F, kept ROIs", "big", False),
    "trace_dt": (4, "Derivative of low-pass dF/F", "big", False),
    "event_windows": (5, "Event windows (incl. class)", "data", True),
    "event_onsets": (5, "Per-ROI onsets", "data", True),
    "event_monotonicity": (5, "Per-event propagation stats", "data", True),
    "event_plots": (5, "Heatmap / raster / density plots", "img", True),
    "event_density_data": (5, "Density / baseline / threshold curves", "data", False),
    "event_heatmap_data": (5, "Heatmap matrix (time x ROI)", "big", False),
    "event_raster_data": (5, "Onset raster (time x ROI)", "big", False),
    "cluster_membership": (6, "ROI -> cluster", "data", True),
    "cluster_plots": (6, "Dendrogram / spatial cluster map", "img", True),
    "cluster_mean_traces": (6, "Mean dF/F per cluster", "data", False),
    "xcorr_pairs": (7, "Pair tables (lag, r, p, q)", "data", True),
    "xcorr_plots": (7, "Violin plots", "img", True),
    "xcorr_violin_data": (7, "Data behind the violins", "data", False),
    "propagation_maps": (8, "Per-event order maps", "img", True),
    "activation_times": (8, "Per-ROI activation time per event", "data", False),
    "nwb": (0, "NWB file", "data", False),
}
COMPACT_KEY = "_compact"  # True -> big traces as .npy instead of CSV


def default_settings() -> dict:
    return {k: v[3] for k, v in EXPORTS.items()} | {COMPACT_KEY: False}


def load_settings(path: Path = SETTINGS_FILE) -> dict:
    """Saved toggles over the defaults; unknown keys are dropped."""
    s = default_settings()
    try:
        saved = json.loads(Path(path).read_text())
        s.update({k: bool(v) for k, v in saved.items() if k in s})
    except (OSError, ValueError):
        pass
    return s


def save_settings(settings: dict, path: Path = SETTINGS_FILE) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {k: bool(v) for k, v in settings.items() if k in default_settings()},
        indent=1))


# -- paths ------------------------------------------------------------------

def output_dir(save_folder) -> Path:
    return Path(save_folder) / OUTPUT_DIRNAME


def figures_root(save_folder) -> Path:
    """Parent of the per-stage figure folders; ``manifest.json`` sits here."""
    return output_dir(save_folder) / "figures"


def figures_dir(save_folder, tab: int) -> Path:
    return figures_root(save_folder) / TAB_DIRS[tab]


def data_dir(save_folder, tab: int) -> Path:
    return output_dir(save_folder) / "data" / TAB_DIRS[tab]


# -- writers ----------------------------------------------------------------

def write_trace(save_folder, tab: int, name: str, trace, fps: float,
                roi_ids: Iterable[int], compact: bool = False) -> Path:
    """(T, N) trace, time down / ROIs across. CSV: ``time_s`` then
    ``roi_<suite2p id>`` columns, written in chunks so a 25k x 700 trace
    never needs a second full copy in memory. Compact: ``.npy`` plus a
    ``<name>_roi_ids.npy`` sidecar."""
    d = data_dir(save_folder, tab)
    d.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(list(roi_ids), dtype=int)
    if compact:
        out = d / f"{name}.npy"
        np.save(out, np.asarray(trace, dtype=np.float32))
        np.save(d / f"{name}_roi_ids.npy", ids)
        return out
    out = d / f"{name}.csv"
    T = trace.shape[0]
    with open(out, "w", newline="") as fh:
        fh.write(",".join(["time_s"] + [f"roi_{i}" for i in ids]) + "\n")
        for t0 in range(0, T, 2000):
            t1 = min(T, t0 + 2000)
            block = np.column_stack([np.arange(t0, t1) / float(fps),
                                     np.asarray(trace[t0:t1], dtype=float)])
            np.savetxt(fh, block, delimiter=",", fmt="%.6g")
    return out


def write_table(save_folder, tab: int, name: str, rows) -> Path:
    """A small table (DataFrame / dict of columns / list of dicts) -> CSV."""
    import pandas as pd
    d = data_dir(save_folder, tab)
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{name}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def write_readme(save_folder) -> Path:
    """``output/README.txt``: every file under output/ with the tab that
    wrote it. Regenerated after each stage."""
    root = output_dir(save_folder)
    root.mkdir(parents=True, exist_ok=True)
    names = {v: f"Tab {k}" if k else "NWB" for k, v in TAB_DIRS.items()}
    lines = ["CalLIOPE outputs for this recording.",
             "calliope_summary.xlsx: one workbook with every stage's tables.",
             "Choose what else is saved in Tab 0 > Data to save "
             "(or each tab's 'Data to save' menu).", ""]
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = f.relative_to(root)
        tab = names.get(rel.parts[1], "") if len(rel.parts) > 2 else ""
        if rel.name != "README.txt":
            lines.append(f"{rel.as_posix():60s} {tab}")
    out = root / "README.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


# -- migration of runs made before output/ existed ------------------------

def migrate_run(save_folder, plane0: Optional[Path] = None, log=print) -> list:
    """Move an old run's user files into ``output/`` once:
    ``plane0/calliope_summary.xlsx`` -> ``output/``, and everything in
    ``calliope_figures/`` -> ``output/figures/``. Nothing is overwritten.
    Returns the moves made."""
    save_folder = Path(save_folder)
    plane0 = Path(plane0) if plane0 else (
        save_folder / "detection" / "final" / "suite2p" / "plane0")
    out = output_dir(save_folder)
    moves = []
    old_xlsx = plane0 / "calliope_summary.xlsx"
    if old_xlsx.exists() and not (out / old_xlsx.name).exists():
        moves.append((old_xlsx, out / old_xlsx.name))
    figs = save_folder / "calliope_figures"
    if figs.is_dir():
        for p in figs.iterdir():
            dest = figures_root(save_folder) / p.name
            if not dest.exists():
                moves.append((p, dest))
    for src, dest in moves:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        log(f"[output] moved {src} -> {dest}")
    if figs.is_dir() and not any(figs.iterdir()):
        figs.rmdir()
    if moves:
        write_readme(save_folder)
    return moves


# -- per-stage exports ------------------------------------------------------

# Small tables are CSV copies of workbook sheets: key -> (tab, sheet, file).
_SHEET_EXPORTS = {
    "rois_table": (3, "ROIs", "rois"),
    "event_windows": (5, "EventWindows", "event_windows"),
    "event_onsets": (5, "EventOnsets", "event_onsets"),
    "event_monotonicity": (5, "EventMonotonicity", "event_monotonicity"),
    "cluster_membership": (6, "Clusters", "cluster_membership"),
}
# Big traces read from plane0: key -> (tab, file, output name, is_memmap).
_TRACE_EXPORTS = {
    "trace_f": (3, "F.npy", "F", False),
    "trace_fneu": (3, "Fneu.npy", "Fneu", False),
    "trace_spks": (3, "spks.npy", "spks", False),
    "trace_dff": (3, "r0p7_filtered_dff.memmap.float32", "dff", True),
    "trace_lowpass": (4, "r0p7_filtered_dff_lowpass.memmap.float32",
                      "dff_lowpass", True),
    "trace_dt": (4, "r0p7_filtered_dff_dt.memmap.float32", "dff_dt", True),
}
STAGE_TAB = {"detection": 3, "lowpass": 4, "events": 5, "clustering": 6,
             "xcorr": 7}


def write_stage_exports(stage: str, plane0, settings: Optional[dict] = None,
                        log=print) -> list:
    """Write the ticked data items of one finished stage into
    ``output/data/<stage>/`` (and refresh README.txt). Reads only what
    the stage left on disk, so the GUI and batch call it the same way;
    a missing input skips that item with a log line. Returns the paths
    written."""
    from . import utils
    from .summary_writer import existing_summary_path
    plane0 = Path(plane0)
    save_folder = utils.save_folder_for_plane0(plane0)
    s = settings if settings is not None else load_settings()
    tab = STAGE_TAB[stage]
    written: list = []

    def on(key):
        return s.get(key) and EXPORTS[key][0] == tab

    xlsx = existing_summary_path(plane0)
    for key, (_t, sheet, name) in _SHEET_EXPORTS.items():
        if on(key):
            try:
                import pandas as pd
                written.append(write_table(save_folder, tab, name,
                                           pd.read_excel(xlsx, sheet_name=sheet)))
            except Exception as e:
                log(f"[output] {key} skipped: {e}")

    mask_path = plane0 / "r0p7_cell_mask_bool.npy"
    keep = np.load(mask_path) if mask_path.exists() else None
    fps = utils.ops_fs(plane0) or utils.get_fps_from_notes(str(plane0))
    for key, (_t, fname, name, is_mm) in _TRACE_EXPORTS.items():
        f = plane0 / fname
        if not on(key):
            continue
        if keep is None or not f.exists():
            log(f"[output] {key} skipped: {f.name} or the keep mask is missing")
            continue
        ids = np.flatnonzero(keep)
        if is_mm:
            T = np.load(plane0 / "F.npy", mmap_mode="r").shape[1]
            tr = np.memmap(str(f), dtype="float32", mode="r",
                           shape=(T, ids.size))
        else:  # suite2p (N_total, T): kept rows, transposed to time-down
            tr = np.load(f, mmap_mode="r")[keep].T
        written.append(write_trace(save_folder, tab, name, tr, fps, ids,
                                   compact=bool(s.get(COMPACT_KEY))))
        del tr

    if stage == "detection" and s.get("zdrift"):
        for src, dest in ((plane0 / "zdrift_pc1.csv", data_dir(save_folder, 3)),
                          (plane0 / "zdrift_qc.png", figures_dir(save_folder, 3))):
            if src.exists():
                dest.mkdir(parents=True, exist_ok=True)
                written.append(Path(shutil.copy2(src, dest / src.name)))

    if stage == "xcorr" and s.get("xcorr_pairs"):
        for csv in plane0.glob("*cluster_results/cross_correlation_*/**/*_summary.csv"):
            rel = csv.relative_to(next(p for p in csv.parents
                                       if p.name.startswith("cross_correlation_")).parent)
            dest = data_dir(save_folder, 7) / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            written.append(Path(shutil.copy2(csv, dest)))

    if written:
        write_readme(save_folder)
        log(f"[output] {stage}: wrote {len(written)} file(s) to "
            f"{output_dir(save_folder)}")
    return written


def export_stage_offload(stage: str, plane0_str: str, *, progress_q=None) -> list:
    """Picklable :mod:`calliope.core.offload` target: ``write_stage_exports``
    in a child process (a 700-ROI dF/F CSV is ~175 MB of text)."""
    return [str(p) for p in write_stage_exports(stage, plane0_str)]
