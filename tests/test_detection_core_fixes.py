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
