"""Regression tests for the repo-misc audit fixes (AppState channel
plumbing, the Open-run keep-mask drift warning, and the curation app's
default checkpoint)."""
from __future__ import annotations

import numpy as np

from calliope.gui_common import AppState


# ---- AppState ----

def test_subscribe_methods_return_working_unsubscribe() -> None:
    st = AppState()
    seen: list = []
    for sub, pub, val in (
        (st.subscribe_event_results, st.set_event_results, {"a": 1}),
        (st.subscribe_clusters_ready, st.set_clusters_ready, "x"),
        (st.subscribe_xcorr_ready, st.set_xcorr_ready, "y"),
        (st.subscribe_stage_error, st.set_stage_error, "boom"),
    ):
        unsub = sub(seen.append)
        assert callable(unsub)
        pub(val)
        n = len(seen)
        unsub()
        pub(val)
        assert len(seen) == n, f"{sub.__name__} unsubscribe did not detach"


def test_reset_clears_informational_tab1_paths(tmp_path) -> None:
    st = AppState()
    st.working_dir = tmp_path
    st.data_root = tmp_path
    st.selected_tiff = tmp_path / "a.tif"
    st.reset()
    assert st.working_dir is None
    assert st.data_root is None
    assert st.selected_tiff is None


# ---- Open run folder: keep-mask drift warning ----

def _plane0(tmp_path, live, frozen=None):
    p0 = tmp_path / "plane0"
    p0.mkdir()
    np.save(p0 / "predicted_cell_mask.npy", np.asarray(live, dtype=bool))
    np.save(p0 / "stat.npy", np.array([{} for _ in live], dtype=object),
            allow_pickle=True)
    if frozen is not None:
        np.save(p0 / "r0p7_cell_mask_bool.npy", np.asarray(frozen, dtype=bool))
    return p0


def test_drift_warning_silent_without_frozen_mask(tmp_path) -> None:
    from calliope.pipeline_gui import PipelineApp
    assert PipelineApp._keep_mask_drift_warning(
        _plane0(tmp_path, [1, 1, 0])) == []


def test_drift_warning_silent_when_counts_agree(tmp_path) -> None:
    from calliope.pipeline_gui import PipelineApp
    assert PipelineApp._keep_mask_drift_warning(
        _plane0(tmp_path, [1, 1, 0], frozen=[1, 1, 0])) == []


def test_drift_warning_reports_both_counts(tmp_path) -> None:
    from calliope.pipeline_gui import PipelineApp
    (warn,) = PipelineApp._keep_mask_drift_warning(
        _plane0(tmp_path, [1, 1, 1], frozen=[1, 0, 0]))
    assert "keeps 3 ROIs" in warn and "holds 1" in warn


def test_drift_warning_ignores_wrong_length_frozen_mask(tmp_path) -> None:
    from calliope.pipeline_gui import PipelineApp
    assert PipelineApp._keep_mask_drift_warning(
        _plane0(tmp_path, [1, 1, 1], frozen=[1, 0])) == []


def test_drift_warning_never_raises_on_missing_stat(tmp_path) -> None:
    from calliope.pipeline_gui import PipelineApp
    p0 = tmp_path / "plane0"
    p0.mkdir()
    np.save(p0 / "r0p7_cell_mask_bool.npy", np.ones(3, dtype=bool))
    assert PipelineApp._keep_mask_drift_warning(p0) == []


# ---- standalone curation app checkpoint default ----

def test_curation_ckpt_falls_back_to_bundled(tmp_path) -> None:
    from calliope.scripts import roi_curation_app as app
    got = app._resolve_ckpt(None, checkpoint_dir=tmp_path)
    assert got == app.BUNDLED_CKPT_PATH
    assert got.name == "cellfilter_best.pt" and got.exists()


def test_curation_ckpt_prefers_local_retrain(tmp_path) -> None:
    from calliope.scripts import roi_curation_app as app
    (tmp_path / "best.pt").write_bytes(b"")
    assert app._resolve_ckpt(None, checkpoint_dir=tmp_path) == \
        tmp_path / "best.pt"


def test_curation_ckpt_explicit_arg_wins(tmp_path) -> None:
    from calliope.scripts import roi_curation_app as app
    missing = tmp_path / "nope.pt"
    assert app._resolve_ckpt(str(missing), checkpoint_dir=tmp_path) == missing


# ---- Open run folder: Tab 5 rehydration goes through apply_batch_row ----

def test_rehydrate_events_uses_apply_batch_row() -> None:
    from types import SimpleNamespace
    from calliope.pipeline_gui import PipelineApp

    calls: dict = {}

    class _FakeEv:
        PARAM_SPEC = [{"name": "z_enter"}]
        _suppress_manifest_write = False

        def apply_batch_row(self, params, log=None):
            calls["applied"] = dict(params)

        def _on_render(self):
            calls["rendered"] = True

    ev = _FakeEv()
    fake_app = SimpleNamespace(event_tab=ev)
    assert PipelineApp._rehydrate_events(
        fake_app, {"z_enter": 4.0, "onset_source": "x"}) is True
    assert calls["applied"] == {"z_enter": 4.0, "onset_source": "x"}
    assert calls["rendered"] and ev._suppress_manifest_write is True
    # No PARAM_SPEC keys -> reported as a defaults fallback.
    assert PipelineApp._rehydrate_events(
        fake_app, {"onset_source": "x"}) is False


def test_rehydrate_events_reports_apply_failure() -> None:
    from types import SimpleNamespace
    from calliope.pipeline_gui import PipelineApp

    class _BadEv:
        PARAM_SPEC = [{"name": "z_enter"}]
        _suppress_manifest_write = False

        def apply_batch_row(self, params, log=None):
            raise ValueError("bad z_enter")

        def _on_render(self):
            pass

    fake_app = SimpleNamespace(event_tab=_BadEv())
    assert PipelineApp._rehydrate_events(fake_app, {"z_enter": "x"}) is False
    assert "bad z_enter" in fake_app._rehydrate_error
