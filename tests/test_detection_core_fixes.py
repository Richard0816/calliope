"""Regression tests for the detection-core legacy-audit fixes (2026-10-05).

Headless: no suite2p run, no GPU, no real recording.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from calliope.core import calliope_settings as cs
from calliope.core import detection_run as dr
from calliope.core import recording_meta
from calliope.core import sparse_plus_cellpose as spc
from calliope.core import suite2p_pipeline as sp


# --- #47: forced pipeline settings warn instead of silently discarding ---

def test_pipeline_pins_force_values_and_report_discarded_edits():
    settings = cs.build_base_settings()
    settings["detection"]["algorithm"] = "sourcery"
    settings["extraction"]["allow_overlap"] = True
    msgs = sp._apply_pipeline_pins(settings)
    assert settings["detection"]["algorithm"] == "sparsery"
    assert settings["extraction"]["allow_overlap"] is False
    assert settings["detection"]["sparsery_settings"]["spatial_scale"] == 0
    joined = "\n".join(msgs)
    assert "detection.algorithm='sourcery'" in joined
    assert "extraction.allow_overlap=True" in joined
    assert len(msgs) == 2


def test_pipeline_pins_silent_when_nothing_overridden():
    settings = cs.build_base_settings()
    assert sp._apply_pipeline_pins(settings) == []


# --- #49/#204: popout edits win over the legacy path_to_ops file ---

def test_popout_override_wins_over_legacy_ops_file(tmp_path, monkeypatch):
    class _Sizer:
        def pick(self, _folder):
            return 500

    monkeypatch.setattr(sp.utils, "get_adaptive_sizer", lambda: _Sizer())
    monkeypatch.setattr(sp.utils, "change_nbinned_according_to_free_ram",
                        lambda _f: 5000)
    monkeypatch.setattr(sp.utils, "_peek_tiff_frame_shape", lambda _f: None)
    legacy = tmp_path / "legacy_ops.npy"
    np.save(legacy, {"threshold_scaling": 0.5, "nimg_init": 123},
            allow_pickle=True)
    cfg = sp.Suite2pPipelineConfig(
        tiff_folder=str(tmp_path), save_folder=str(tmp_path),
        path_to_ops=str(legacy),
        settings_override={"detection": {"threshold_scaling": 0.9}},
        verbose=False,
    )
    _db, settings = sp.load_base_settings(cfg)
    assert settings["detection"]["threshold_scaling"] == 0.9
    # Keys only the legacy file sets still apply.
    assert settings["registration"]["nimg_init"] == 123


# --- #46/#253: the print-hook hard cap is gone; the note explains it ---

def test_print_hook_hard_cap_retired():
    for name in ("_install_sparsery_roi_cap", "_restore_sparsery_print",
                 "_RoiHardCapExceeded"):
        assert not hasattr(sp, name)


def test_hard_cap_note_only_when_cap_below_max_rois():
    settings = cs.build_base_settings()
    max_rois = settings["detection"]["sparsery_settings"]["max_ROIs"]
    assert sp.hard_cap_note(0, settings) is None
    assert sp.hard_cap_note(max_rois + 1, settings) is None
    note = sp.hard_cap_note(max_rois - 1, settings)
    assert note and "not enforced" in note and str(max_rois) in note


# --- #44/#45: inert sparsery params are flagged ---

def test_inert_sparsery_op_notes():
    base = cs.build_base_settings()
    reg_sigma = base["registration"]["smooth_sigma"]
    notes = spc.inert_sparsery_op_notes(
        {"max_iterations": 2000, "smooth_sigma": reg_sigma + 0.5}, base)
    assert any("max_iterations" in n and "max_ROIs" in n for n in notes)
    assert any("smooth_sigma" in n and "registration" in n for n in notes)
    # A smooth_sigma equal to what registration used is not flagged.
    notes = spc.inert_sparsery_op_notes({"smooth_sigma": reg_sigma}, base)
    assert notes == []
    # The GUI's defaults are not flagged either (they ride every run).
    defaults = {k: spc.SPARSERY_OPS[k]
                for k in ("max_iterations", "smooth_sigma")}
    assert spc.inert_sparsery_op_notes(defaults, base) == []


# --- #226: meta.json detection block records the sparsery-pass values ---

def test_detection_meta_uses_sparsery_pass_settings():
    sp_settings = cs.build_base_settings()
    sp.apply_settings_overrides(sp_settings, {
        "sparse_mode": True, "threshold_scaling": 0.85,
        "spatial_scale": 2, "high_pass": 100,
    })
    reg_view = {"threshold_scaling": 0.8, "spatial_scale": 0}
    meta = spc._detection_meta(sp_settings, {"spatscale_pix": np.array([12])},
                               reg_view)
    assert meta["threshold_scaling"] == 0.85
    assert meta["spatial_scale"] == 2
    assert meta["sparse_mode"] is True
    assert meta["highpass_time"] == 100
    assert meta["spatscale_pix"] == 12


def test_detection_meta_falls_back_to_registration_view():
    meta = spc._detection_meta(None, None,
                               {"threshold_scaling": 0.8, "spatial_scale": 0})
    assert meta["threshold_scaling"] == 0.8
    assert meta["spatial_scale"] == 0


# --- #176: override count is leaves, not top-level keys ---

def test_count_leaves():
    assert spc._count_leaves({"a": 1, "b": {"c": 2, "d": {"e": 3}}}) == 3
    assert (spc._count_leaves(cs.CALLIOPE_BASE_SETTINGS)
            > len(cs.CALLIOPE_BASE_SETTINGS))


# --- #56/#230: importing the module no longer edits sys.path ---

def test_spc_import_does_not_touch_sys_path():
    assert not hasattr(spc, "WORKTREE")
    import inspect
    assert "sys.path.insert" not in inspect.getsource(spc)
    core_dir = str(spc.__file__).rsplit("sparse_plus_cellpose", 1)[0].rstrip("\\/")
    assert all(p.rstrip("\\/") != core_dir for p in sys.path)


def test_main_refuses_placeholder_paths():
    with pytest.raises(SystemExit):
        spc.main()


# --- #60: core prune tuples stay equal to the batch scratch tuples ---

def test_prune_constants_match_batch_logic():
    from calliope.tabs.batch import logic as batch_logic
    assert dr._PRUNE_FILE_PATTERNS == batch_logic.SCRATCH_PRUNE_FILE_PATTERNS
    assert (dr._PRUNE_INTERMEDIATE_DIRS
            == batch_logic.SCRATCH_RECORDING_INTERMEDIATE_DIRS)


# --- #61: headless defaults come from the spc constants ---

def test_headless_detection_defaults_come_from_spc():
    assert dr._build_sparsery_ops({}) == spc.SPARSERY_OPS
    assert dr._build_cellpose_cfg({}) == spc.CELLPOSE_CFG
    ops = dr._build_sparsery_ops({"threshold_scaling": 1.2,
                                  "sparse_mode": False})
    assert ops["threshold_scaling"] == 1.2
    assert ops["sparse_mode"] is True


# --- N4: meta.json baseline_min_sec is seconds, not minutes ---

def test_dff_meta_baseline_is_seconds(tmp_path):
    rng = np.random.default_rng(0)
    T, N = 200, 3
    F = (100 + rng.random((N, T))).astype(np.float32)
    np.save(tmp_path / "F.npy", F)
    np.save(tmp_path / "Fneu.npy", (F * 0.1).astype(np.float32))
    dr.compute_dff_memmaps(
        tmp_path, {"fps_override": 10.0, "use_gpu_dff": False},
        baseline_mode="first_n", baseline_min=0.1)
    meta = recording_meta.read_meta(tmp_path)
    assert meta["dff"]["baseline_min_sec"] == pytest.approx(6.0)


# --- #130: detection writes only the unfiltered r0p7_dff memmap ---

_OLD_TAB3_KEYS = {"default_lowpass_hz": 0.5, "default_sg_win_ms": 200,
                  "default_sg_poly": 3}


def _write_f(plane0, T=200, N=3):
    rng = np.random.default_rng(1)
    F = (100 + 5 * rng.random((N, T))).astype(np.float32)
    np.save(plane0 / "F.npy", F)
    np.save(plane0 / "Fneu.npy", (F * 0.1).astype(np.float32))
    return F, T, N


def _assert_only_dff_written(plane0, T, N):
    assert (plane0 / "r0p7_dff.memmap.float32").stat().st_size == T * N * 4
    assert not (plane0 / "r0p7_dff_lowpass.memmap.float32").exists()
    assert not (plane0 / "r0p7_dff_dt.memmap.float32").exists()
    # the filtered Tab 4 outputs are not this stage's job either
    assert not (plane0 / "r0p7_filtered_dff_lowpass.memmap.float32").exists()


def test_cpu_dff_writes_only_r0p7_dff(tmp_path):
    F, T, N = _write_f(tmp_path)
    info = dr.compute_dff_memmaps(
        tmp_path, {"fps_override": 10.0, "use_gpu_dff": False,
                   **_OLD_TAB3_KEYS},       # stale keys are ignored
        baseline_mode="first_n", baseline_min=0.1)
    assert info["path"] == "cpu"
    _assert_only_dff_written(tmp_path, T, N)
    dff = np.memmap(tmp_path / "r0p7_dff.memmap.float32", dtype="float32",
                    mode="r", shape=(T, N))
    from calliope.core import utils
    ref = utils.first_n_min_df_over_f_1d(
        F[1] - 0.7 * F[1] * 0.1, baseline_min=0.1, perc=10, fps=10.0)
    np.testing.assert_allclose(dff[:, 1], ref, rtol=1e-5, atol=1e-6)
    # no detection-time low-pass stamp any more
    assert recording_meta.read_meta(tmp_path)["lowpass"]["cutoff_hz"] is None


def _numpy_cupy():
    """A numpy-backed stand-in for the CuPy calls _gpu_process_all makes."""
    import types

    pool = types.SimpleNamespace(free_all_blocks=lambda: None)
    return types.SimpleNamespace(
        asarray=lambda x, dtype=None: np.asarray(x, dtype=dtype),
        asnumpy=np.asarray, float32=np.float32, isfinite=np.isfinite,
        mean=np.mean, maximum=np.maximum, full_like=np.full_like,
        get_default_memory_pool=lambda: pool,
        get_default_pinned_memory_pool=lambda: pool,
        cuda=types.SimpleNamespace(Stream=types.SimpleNamespace(
            null=types.SimpleNamespace(synchronize=lambda: None))),
    )


def test_gpu_dff_writes_only_r0p7_dff_and_matches_cpu(tmp_path, monkeypatch):
    from calliope.core import analyze_output_gpu as gpu
    monkeypatch.setattr(gpu, "cp", _numpy_cupy(), raising=False)
    monkeypatch.setattr(gpu, "_CUPY_OK", True, raising=False)
    gpu_dir = tmp_path / "gpu"
    cpu_dir = tmp_path / "cpu"
    gpu_dir.mkdir()
    cpu_dir.mkdir()
    _, T, N = _write_f(gpu_dir)
    _write_f(cpu_dir)
    params = {"fps_override": 10.0, **_OLD_TAB3_KEYS}
    info = dr.compute_dff_memmaps(gpu_dir, {**params, "use_gpu_dff": True},
                                  baseline_mode="first_n", baseline_min=0.1)
    assert info["path"] == "gpu"
    _assert_only_dff_written(gpu_dir, T, N)
    dr.compute_dff_memmaps(cpu_dir, {**params, "use_gpu_dff": False},
                           baseline_mode="first_n", baseline_min=0.1)
    a, b = (np.memmap(d / "r0p7_dff.memmap.float32", dtype="float32",
                      mode="r", shape=(T, N)) for d in (gpu_dir, cpu_dir))
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_removed_tab3_lowpass_keys_are_gone_and_load_harmlessly():
    import types
    from calliope.gui_common import spec_defaults
    from calliope.tabs.batch.logic import build_batch_param_spec
    from calliope.tabs.suite2p.tab import Suite2pTab

    removed = set(_OLD_TAB3_KEYS)
    assert not removed & {e["name"] for e in Suite2pTab.PARAM_SPEC}
    assert not removed & {e["name"] for e in build_batch_param_spec()}
    # an old batch row / saved queue still carrying the keys applies
    # cleanly: they are simply not picked up
    stub = types.SimpleNamespace(
        PARAM_SPEC=Suite2pTab.PARAM_SPEC,
        _params=spec_defaults(Suite2pTab.PARAM_SPEC),
        custom_tau_var=types.SimpleNamespace(set=lambda v: None))
    Suite2pTab.apply_batch_row(stub, {**_OLD_TAB3_KEYS, "perc": 20})
    assert stub._params["perc"] == 20
    assert not removed & set(stub._params)


def test_s2p_open_dff_memmap_needs_only_the_dff_file(tmp_path):
    from calliope.core import utils
    T, N = 7, 4
    np.save(tmp_path / "F.npy", np.zeros((N, T), np.float32))
    data = np.arange(T * N, dtype=np.float32).reshape(T, N)
    data.tofile(tmp_path / "r0p7_dff.memmap.float32")
    dff, t, n = utils.s2p_open_dff_memmap(tmp_path, prefix="r0p7_")
    assert (t, n) == (T, N)
    np.testing.assert_array_equal(np.asarray(dff), data)
