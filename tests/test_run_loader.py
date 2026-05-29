"""Tests for the run-inventory logic behind "Open run folder…".

Covers the pure, GUI-free surface: where plane0/manifest live, how
manifest params are read (including the partial-manifest case), and the
per-stage presence detection that drives what a reload populates and
what it warns about.
"""
from __future__ import annotations

import json
from pathlib import Path

from calliope.core import run_loader


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _make_plane0(run: Path) -> Path:
    p = run / "detection" / "final" / "suite2p" / "plane0"
    p.mkdir(parents=True)
    return p


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def _write_manifest(run: Path, params: dict, git_sha: str = "abc123") -> None:
    figs = run / "calliope_figures"
    figs.mkdir(parents=True, exist_ok=True)
    (figs / "manifest.json").write_text(
        json.dumps({"calliope_git_sha": git_sha, "params": params}),
        encoding="utf-8")


# --------------------------------------------------------------------------
# Path derivation
# --------------------------------------------------------------------------

def test_plane0_for_is_fixed_depth(tmp_path):
    run = tmp_path / "rec"
    assert run_loader.plane0_for(run) == (
        run / "detection" / "final" / "suite2p" / "plane0")


def test_manifest_path_for(tmp_path):
    run = tmp_path / "rec"
    assert run_loader.manifest_path_for(run) == (
        run / "calliope_figures" / "manifest.json")


# --------------------------------------------------------------------------
# Manifest params
# --------------------------------------------------------------------------

def test_manifest_params_missing_returns_empty(tmp_path):
    assert run_loader.manifest_params(tmp_path / "rec") == {}


def test_manifest_params_roundtrip(tmp_path):
    run = tmp_path / "rec"
    _write_manifest(run, {"min_prominence": 0.42, "onset_source": "spks"})
    params = run_loader.manifest_params(run)
    assert params["min_prominence"] == 0.42
    assert params["onset_source"] == "spks"


def test_manifest_params_corrupt_returns_empty(tmp_path):
    run = tmp_path / "rec"
    figs = run / "calliope_figures"
    figs.mkdir(parents=True)
    (figs / "manifest.json").write_text("{not json", encoding="utf-8")
    assert run_loader.manifest_params(run) == {}


def test_manifest_params_no_params_key(tmp_path):
    run = tmp_path / "rec"
    figs = run / "calliope_figures"
    figs.mkdir(parents=True)
    (figs / "manifest.json").write_text('{"rec_id": "x"}', encoding="utf-8")
    assert run_loader.manifest_params(run) == {}


# --------------------------------------------------------------------------
# Stage inventory
# --------------------------------------------------------------------------

def test_empty_folder_has_no_stages(tmp_path):
    inv = run_loader.scan_run(tmp_path / "rec")
    assert not inv.any_stage
    assert not inv.has_preprocess and not inv.has_detection


def test_preprocess_only(tmp_path):
    run = tmp_path / "rec"
    run.mkdir()
    _touch(run / "qc.gif")
    _touch(run / "mean.npy")
    inv = run_loader.scan_run(run)
    assert inv.has_preprocess
    assert not inv.has_detection
    assert inv.any_stage


def test_detection_needs_both_f_and_stat(tmp_path):
    run = tmp_path / "rec"
    plane0 = _make_plane0(run)
    _touch(plane0 / "F.npy")          # stat.npy missing
    assert not run_loader.scan_run(run).has_detection
    _touch(plane0 / "stat.npy")
    assert run_loader.scan_run(run).has_detection


def test_lowpass_requires_both_memmaps_and_detection(tmp_path):
    run = tmp_path / "rec"
    plane0 = _make_plane0(run)
    _touch(plane0 / "F.npy")
    _touch(plane0 / "stat.npy")
    _touch(plane0 / "r0p7_filtered_dff_lowpass.memmap.float32")
    # derivative memmap still missing
    assert not run_loader.scan_run(run).has_lowpass
    _touch(plane0 / "r0p7_filtered_dff_dt.memmap.float32")
    assert run_loader.scan_run(run).has_lowpass


def test_clusters_and_xcorr_detected_under_prefix(tmp_path):
    run = tmp_path / "rec"
    plane0 = _make_plane0(run)
    _touch(plane0 / "F.npy")
    _touch(plane0 / "stat.npy")
    cdir = plane0 / "r0p7_filtered_cluster_results" / "gui_recluster"
    _touch(cdir / "linkage.npy")
    _touch(cdir / "cross_correlation_full" / "C0_C1.csv")
    inv = run_loader.scan_run(run)
    assert inv.has_clusters
    assert inv.has_xcorr


def test_load_existing_no_regen_skips_shifted(tmp_path):
    """A viewing reload of an archived run (no shifted TIFF, sidecar
    present) must return immediately without regenerating, pointing
    shifted_tiff at the expected name with shifted_paths empty."""
    import numpy as np
    from calliope.core import preprocessing

    out = tmp_path / "rec"
    out.mkdir()
    (out / "qc.gif").write_bytes(b"GIF89a")
    np.save(out / "mean.npy", np.zeros((4, 4), dtype=float))
    # Sidecar references a raw that does NOT exist -- regeneration would
    # fail/be slow; no-regen mode must not even attempt it.
    (out / "_calliope_raw_paths.json").write_text(
        json.dumps({"raw_paths": [str(tmp_path / "raw_A.tif")]}),
        encoding="utf-8")

    res = preprocessing.load_existing_preprocess(
        str(out), regenerate_shifted=False)

    assert res is not None
    assert res.qc_gif == out / "qc.gif"
    assert res.shifted_paths == []
    assert res.shifted_tiff.name == "shifted_raw_A.tif"
    assert tuple(res.shape_yx) == (4, 4)


def test_full_run_reports_everything(tmp_path):
    run = tmp_path / "rec"
    run.mkdir()
    _touch(run / "qc.gif")
    _touch(run / "mean.npy")
    plane0 = _make_plane0(run)
    for f in ("F.npy", "stat.npy",
              "r0p7_filtered_dff_lowpass.memmap.float32",
              "r0p7_filtered_dff_dt.memmap.float32"):
        _touch(plane0 / f)
    cdir = plane0 / "r0p7_filtered_cluster_results" / "gui_recluster"
    _touch(cdir / "linkage.npy")
    _touch(cdir / "cross_correlation_per_event" / "event_0.csv")
    _write_manifest(run, {"min_prominence": 0.1})
    inv = run_loader.scan_run(run)
    assert all((inv.has_preprocess, inv.has_detection, inv.has_lowpass,
                inv.has_clusters, inv.has_xcorr))
    assert inv.manifest["calliope_git_sha"] == "abc123"
    assert inv.plane0 == plane0
