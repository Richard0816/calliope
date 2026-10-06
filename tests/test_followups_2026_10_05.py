"""Regression tests for the 2026-10-05 legacy-audit follow-ups."""
import numpy as np

from calliope.core import utils


def _plane0(tmp_path, live, frozen=None):
    p0 = tmp_path / "plane0"
    p0.mkdir()
    np.save(p0 / "predicted_cell_mask.npy", np.asarray(live, dtype=bool))
    if frozen is not None:
        np.save(p0 / "r0p7_cell_mask_bool.npy", np.asarray(frozen, dtype=bool))
    return p0


# ---- utils.pipeline_mask_count (keep-mask drift) ----

def test_pipeline_mask_count_none_without_frozen_mask(tmp_path):
    assert utils.pipeline_mask_count(_plane0(tmp_path, [1, 1, 0]), 3) is None


def test_pipeline_mask_count_none_for_wrong_length(tmp_path):
    p0 = _plane0(tmp_path, [1, 1, 1], frozen=[1, 0])
    assert utils.pipeline_mask_count(p0, 3) is None


def test_pipeline_mask_count_reads_frozen_mask(tmp_path):
    p0 = _plane0(tmp_path, [1, 1, 1], frozen=[1, 0, 0])
    assert utils.pipeline_mask_count(p0, 3) == 1


def test_pipeline_mask_count_anchors_to_memmap_size(tmp_path):
    # Frozen mask says 1 kept, but the memmap on disk holds 2 columns
    # (the predicted mask's count): readers size by the file, so must we.
    p0 = _plane0(tmp_path, [1, 1, 0], frozen=[1, 0, 0])
    T = 5
    np.save(p0 / "F.npy", np.zeros((3, T), dtype=np.float32))
    np.zeros((T, 2), dtype=np.float32).tofile(
        p0 / "r0p7_filtered_dff.memmap.float32")
    assert utils.pipeline_mask_count(p0, 3) == 2


# ---- batch spec carries the xcorr shuffle / partial knobs ----

def test_batch_spec_has_xcorr_shuffle_and_partial():
    from calliope.tabs.batch.logic import build_batch_param_spec
    spec = {e["name"]: e for e in build_batch_param_spec()}
    assert spec["n_shuffles"]["default"] == 500
    assert spec["compute_partial"]["default"] is False
    assert spec["n_shuffles"]["group"] == "6. Cross-correlation"


def test_xcorr_apply_batch_row_parses_string_bools():
    import types
    from calliope.tabs.crosscorrelation.tab import CrossCorrelationTab

    class _V:
        def __init__(self):
            self.v = None

        def set(self, v):
            self.v = v

    s = types.SimpleNamespace(maxlag_var=_V(), zero_lag_var=_V(),
                              gpu_var=_V(), n_shuffles_var=_V(),
                              partial_corr_var=_V())
    CrossCorrelationTab.apply_batch_row(
        s, {"zero_lag": "false", "use_gpu": "False",
            "compute_partial": "true"})
    assert s.zero_lag_var.v is False and s.gpu_var.v is False
    assert s.partial_corr_var.v is True


# ---- GPU dF/F fallback reason reaches the caller's log ----

def test_gpu_dff_reports_why_it_used_cpu(tmp_path):
    from calliope.core import detection_run
    msgs = []
    kw = dict(baseline_min=2.0, fps=15.0, cutoff_hz=1.0, sg_win_ms=333,
              sg_poly=2, r=0.7, progress_cb=msgs.append)
    assert not detection_run._maybe_run_dff_gpu(
        tmp_path, {"use_gpu_dff": False}, baseline_mode="first_n", **kw)
    assert not detection_run._maybe_run_dff_gpu(
        tmp_path, {"use_gpu_dff": True}, baseline_mode="rolling", **kw)
    assert "use_gpu_dff off" in msgs[0]
    assert "rolling" in msgs[1]


def test_gpu_dff_failure_reason_is_reported(tmp_path, monkeypatch):
    from calliope.core import analyze_output_gpu as gpu
    from calliope.core import detection_run

    def boom(*a, **k):
        raise RuntimeError("cuda kaput")

    monkeypatch.setattr(gpu, "_CUPY_OK", True, raising=False)
    monkeypatch.setattr(gpu, "process_suite2p_traces_gpu", boom)
    msgs = []
    F = np.ones((2, 10), dtype=np.float32)
    assert not detection_run._maybe_run_dff_gpu(
        tmp_path, {"use_gpu_dff": True}, baseline_mode="first_n",
        baseline_min=2.0, fps=15.0, cutoff_hz=1.0, sg_win_ms=333,
        sg_poly=2, r=0.7, F=F, Fneu=F, progress_cb=msgs.append)
    assert any("cuda kaput" in m for m in msgs)


# ---- QC "Reload from folder" never regenerates on the Tk thread ----

def test_qc_reload_offloads_archived_regeneration(tmp_path, monkeypatch):
    import types
    from calliope.core import offload
    from calliope.core.preprocessing import PreprocessResult
    from calliope.tabs.qc import tab as qc_tab

    calls = {}
    archived = PreprocessResult(
        out_dir=tmp_path, shifted_tiff=tmp_path / "shifted_a.tif",
        qc_gif=tmp_path / "qc.gif", mean_image_path=tmp_path / "mean.npy",
        n_frames=-1, shape_yx=(4, 4), raw_paths=[tmp_path / "a.tif"],
        shifted_paths=[])

    def fake_load(path, *, regenerate_shifted=True, **_):
        calls["regen"] = regenerate_shifted
        return archived

    monkeypatch.setattr(qc_tab.preprocessing, "load_existing_preprocess",
                        fake_load)
    monkeypatch.setattr(qc_tab.filedialog, "askdirectory",
                        lambda **_: str(tmp_path))
    monkeypatch.setattr(offload, "run_offloaded",
                        lambda *a, **k: calls.setdefault("offloaded", a[2]))
    published = []
    stub = types.SimpleNamespace(
        _reload_job=None, _reload_q=None,
        header_var=types.SimpleNamespace(set=lambda v: None),
        state=types.SimpleNamespace(set_result=published.append))
    qc_tab.QcTab._reload_from_folder(stub)
    assert calls["regen"] is False           # probe only on the Tk thread
    assert calls["offloaded"].__name__ == "load_existing_offload"
    assert published == []                   # published when the job is done

    # Shifted TIFF present -> published straight away, nothing offloaded.
    calls.clear()
    stub._reload_job = None
    archived.shifted_paths = [tmp_path / "shifted_a.tif"]
    qc_tab.QcTab._reload_from_folder(stub)
    assert published == [archived] and "offloaded" not in calls


# ---- standalone curation app: (-1, -1) sentinel is not a flip ----

def test_curation_app_does_not_advance_on_sentinel():
    import types
    from calliope.scripts import roi_curation_app as app
    steps = []
    stub = types.SimpleNamespace(_step=steps.append)
    app.CurationApp._on_flipped(stub, -1, -1)
    assert steps == []
    app.CurationApp._on_flipped(stub, 3, 1)
    assert steps == [1]
