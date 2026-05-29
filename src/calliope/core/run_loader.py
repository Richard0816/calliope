"""Discover and inventory a completed CalLIOPE run on disk.

Pure, GUI-free helpers behind the "Open run folder…" action: given one
recording folder, work out where each stage's outputs live and which
stages actually completed, so the app can rehydrate every tab in
dependency order and tell the user exactly what loaded.

Reads only what already lives on disk; never recomputes anything. The
GUI orchestration that consumes this lives in
``calliope.pipeline_gui.PipelineApp._on_open_run``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# plane0 sits at a fixed depth under the recording folder (see Tab 3's
# archive layout). Everything downstream is derivable from it.
_PLANE0_REL = ("detection", "final", "suite2p", "plane0")


def plane0_for(run_folder: Path) -> Path:
    """Canonical Suite2p ``plane0`` path for a recording folder."""
    return Path(run_folder).joinpath(*_PLANE0_REL)


def manifest_path_for(run_folder: Path) -> Path:
    """Where the per-run ``manifest.json`` lives (may not exist)."""
    return Path(run_folder) / "calliope_figures" / "manifest.json"


def read_manifest(run_folder: Path) -> Optional[dict]:
    """Parse ``calliope_figures/manifest.json``; ``None`` if absent/unreadable."""
    p = manifest_path_for(run_folder)
    if not p.is_file():
        return None
    try:
        man = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return man if isinstance(man, dict) else None


def manifest_params(run_folder: Path) -> dict:
    """The run's saved ``params`` dict from the manifest (``{}`` if absent).

    Reliable for batch-produced runs (the manifest records the single
    pipeline-wide params dict). For hand-run recordings the manifest is
    overwritten by whichever tab exported last, so it may hold only that
    stage's params -- callers should intersect with the keys they need
    and warn when expected keys are missing.
    """
    man = read_manifest(run_folder)
    if not isinstance(man, dict):
        return {}
    params = man.get("params")
    return params if isinstance(params, dict) else {}


@dataclass
class RunInventory:
    """Which stages a recording folder actually contains."""

    run_folder: Path
    plane0: Path
    has_preprocess: bool = False   # qc.gif + mean.npy
    has_detection: bool = False    # plane0 with F.npy + stat.npy
    has_lowpass: bool = False      # filtered lowpass + derivative memmaps
    has_clusters: bool = False     # a linkage.npy under *cluster_results
    has_xcorr: bool = False        # a cross_correlation_* output dir
    manifest: Optional[dict] = None

    @property
    def any_stage(self) -> bool:
        return any((self.has_preprocess, self.has_detection,
                    self.has_lowpass, self.has_clusters, self.has_xcorr))


def _exists_all(folder: Path, names) -> bool:
    return all((folder / n).exists() for n in names)


def scan_run(run_folder: Path) -> RunInventory:
    """Inspect ``run_folder`` and report which stages are present.

    Stage detection mirrors each tab's own readiness check so the
    inventory matches what a reload can actually populate.
    """
    run_folder = Path(run_folder)
    plane0 = plane0_for(run_folder)
    inv = RunInventory(run_folder=run_folder, plane0=plane0)

    inv.has_preprocess = _exists_all(run_folder, ("qc.gif", "mean.npy"))
    inv.has_detection = plane0.is_dir() and _exists_all(
        plane0, ("F.npy", "stat.npy"))
    # Tab 5's _inputs_ready checks exactly these two memmaps.
    inv.has_lowpass = inv.has_detection and _exists_all(
        plane0,
        ("r0p7_filtered_dff_lowpass.memmap.float32",
         "r0p7_filtered_dff_dt.memmap.float32"))
    # Clustering / xcorr nest under a prefix-named *cluster_results dir.
    # Glob rather than hard-code the prefix so we match whatever the run
    # used (r0p7_, r0p7_filtered_, ...).
    if plane0.is_dir():
        inv.has_clusters = any(plane0.glob("*cluster_results/**/linkage.npy"))
        inv.has_xcorr = any(
            plane0.glob("*cluster_results/**/cross_correlation_*"))
    inv.manifest = read_manifest(run_folder)
    return inv
