"""Regression tests for Tab 1 (preprocess) audit fixes.

* GUI preprocessing writes the ``_calliope_raw_paths.json`` sidecar (#101).
* Force rerun overwrites existing grouped shifted TIFFs; a non-forced run
  only reuses one whose page count matches its source (#102).
* Tab 1's status probe never regenerates archived shifted TIFFs on the
  Tk thread, and the regeneration target runs it explicitly (#100).
* Tab 1's QC-gif key list matches core's (#108).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from calliope.core import preprocessing as pp

QC = {"downsample_t": 1, "max_size_px": 16, "playback_fps": 5,
      "clip_low": 1.0, "clip_high": 99.0}


def _raw(path: Path, T: int = 6, fill: int | None = None) -> Path:
    rng = np.random.default_rng(0)
    data = rng.integers(-50, 500, size=(T, 8, 10)).astype(np.int16)
    if fill is not None:
        data[:] = fill
    tifffile.imwrite(str(path), data)
    return path


def test_preprocess_offload_writes_raw_paths_sidecar(tmp_path):
    raw = _raw(tmp_path / "rec_00001.tif")
    res = pp.preprocess_offload([str(raw)], str(tmp_path / "data"),
                                qc_params=QC)
    sidecar = Path(res.out_dir) / pp.RAW_PATHS_SIDECAR
    assert sidecar.is_file()
    assert json.loads(sidecar.read_text())["raw_paths"] == [
        str(raw.resolve())]
    assert pp.read_raw_paths_sidecar(res.out_dir) == [raw.resolve()]


def _group(tmp_path):
    a = _raw(tmp_path / "rec_00001.tif")
    b = _raw(tmp_path / "rec_00002.tif")
    return [str(a), str(b)]


def test_group_force_overwrites_existing_shifted(tmp_path):
    srcs = _group(tmp_path)
    res = pp.preprocess_offload(srcs, str(tmp_path / "data"), group=True,
                                qc_params=QC)
    first = Path(res.shifted_paths[0])
    good = tifffile.imread(str(first))
    # Corrupt the content but keep the page count, so only ``force``
    # (not the page-count guard) can trigger the rewrite.
    tifffile.imwrite(str(first), np.zeros_like(good))

    pp.preprocess_offload(srcs, str(tmp_path / "data"), group=True,
                          qc_params=QC)
    assert not tifffile.imread(str(first)).any(), "non-force reuses"

    pp.preprocess_offload(srcs, str(tmp_path / "data"), group=True,
                          qc_params=QC, force=True)
    np.testing.assert_array_equal(tifffile.imread(str(first)), good)


def test_group_rewrites_truncated_shifted_without_force(tmp_path):
    srcs = _group(tmp_path)
    res = pp.preprocess_offload(srcs, str(tmp_path / "data"), group=True,
                                qc_params=QC)
    first = Path(res.shifted_paths[0])
    good = tifffile.imread(str(first))
    tifffile.imwrite(str(first), good[:2])          # killed mid-write
    res2 = pp.preprocess_offload(srcs, str(tmp_path / "data"), group=True,
                                 qc_params=QC)
    np.testing.assert_array_equal(tifffile.imread(str(first)), good)
    assert res2.n_frames == 12


def _archived(tmp_path):
    raw = _raw(tmp_path / "rec_00001.tif")
    res = pp.preprocess_offload([str(raw)], str(tmp_path / "data"),
                                qc_params=QC)
    for p in res.shifted_paths:
        Path(p).unlink()
    return raw, Path(res.out_dir)


def test_tab1_probe_never_regenerates(tmp_path, monkeypatch):
    tab_mod = pytest.importorskip("calliope.tabs.preprocess.tab")
    raw, out_dir = _archived(tmp_path)

    def boom(*_a, **_k):
        raise AssertionError("probe must not regenerate shifted TIFFs")
    monkeypatch.setattr(pp, "shift_tiff_to_uint16", boom)
    monkeypatch.setattr(pp, "preprocess_tiff_group", boom)

    probe = tab_mod.PreprocessTab._probe_existing
    res = probe(out_dir)
    assert res is not None and res.shifted_paths == []
    assert not list(out_dir.glob("*shifted_*"))

    # Raw gone everywhere -> not regenerable -> Run preprocesses afresh.
    raw.unlink()
    assert probe(out_dir) is None


def test_load_existing_offload_regenerates(tmp_path):
    raw, out_dir = _archived(tmp_path)
    res = pp.load_existing_offload(str(out_dir))
    assert res.shifted_paths and Path(res.shifted_paths[0]).is_file()

    for p in res.shifted_paths:
        Path(p).unlink()
    raw.unlink()
    with pytest.raises(RuntimeError):
        pp.load_existing_offload(str(out_dir))


def test_tab1_qc_keys_match_core():
    tab_mod = pytest.importorskip("calliope.tabs.preprocess.tab")
    assert set(tab_mod.PreprocessTab._qc_keys()) == set(pp._QC_KEYS)
    assert set(tab_mod.PreprocessTab._qc_keys()) == {
        s["name"] for s in tab_mod.PreprocessTab.PARAM_SPEC}
