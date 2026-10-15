"""User-facing output layout: one ``<recording>/output/`` folder.

Everything a user is meant to open lives here; the rest of the recording
folder (``detection/…/plane0``, memmaps, caches) is CalLIOPE's own::

    <recording>/output/
        calliope_summary.xlsx
        README.txt            (lists every file and the tab that wrote it)
        manifest.json
        <recording>.nwb       (if exported)
        figures/<n_tab>/...   (PNG + SVG)
        data/<n_tab>/...      (CSV, or .npy for big traces in compact mode)

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

TAB_DIRS = {1: "1_preprocess", 3: "3_detection", 4: "4_lowpass",
            5: "5_events", 6: "6_clustering", 7: "7_xcorr",
            8: "8_propagation", 0: "nwb"}

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


def figures_dir(save_folder, tab: int) -> Path:
    return output_dir(save_folder) / "figures" / TAB_DIRS[tab]


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
    ``plane0/calliope_summary.xlsx`` -> ``output/``, and
    ``calliope_figures/<stage>/`` -> ``output/figures/<n_stage>/`` (its
    ``manifest.json`` -> ``output/``). Nothing is overwritten. Returns
    the moves made."""
    save_folder = Path(save_folder)
    plane0 = Path(plane0) if plane0 else (
        save_folder / "detection" / "final" / "suite2p" / "plane0")
    out = output_dir(save_folder)
    stage_dirs = {"preprocess": 1, "detection": 3, "lowpass": 4,
                  "event_detection": 5, "clustering": 6,
                  "crosscorrelation": 7, "spatial_propagation": 8}
    moves = []
    old_xlsx = plane0 / "calliope_summary.xlsx"
    if old_xlsx.exists() and not (out / old_xlsx.name).exists():
        moves.append((old_xlsx, out / old_xlsx.name))
    figs = save_folder / "calliope_figures"
    if figs.is_dir():
        for p in figs.iterdir():
            if p.is_dir() and p.name in stage_dirs:
                dest = figures_dir(save_folder, stage_dirs[p.name])
            elif p.name == "manifest.json":
                dest = out / p.name
            else:
                dest = out / "figures" / p.name
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
