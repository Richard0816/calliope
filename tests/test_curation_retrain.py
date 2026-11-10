"""Curation popout Retrain / re-predict wiring (audit #145, #146, #37, #240)
and the cell-filter train / predict CLI knobs (#145, #207).

The popout is a CTkToplevel, so these tests build it with ``__new__`` and
set only the attributes the methods under test touch -- no Tk display.
"""
from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("customtkinter")

from calliope.core.cellfilter import config as C  # noqa: E402
from calliope.core.cellfilter import predict as P  # noqa: E402
from calliope.core.cellfilter import train as T  # noqa: E402
from calliope.tabs.suite2p import curation_popout as CP  # noqa: E402


class _Var:
    def __init__(self):
        self.value = None

    def set(self, v):
        self.value = v


class _Btn:
    def configure(self, **kw):
        pass


def _bare_popout(tmp_path, **hooks):
    pop = CP.CurationPopout.__new__(CP.CurationPopout)
    pop._plane0 = tmp_path
    pop._rec_id = "rec"
    pop._get_size_outlier_z = hooks.get("get_size_outlier_z")
    pop._get_ckpt_path = hooks.get("get_ckpt_path")
    pop._on_checkpoint_trained = hooks.get("on_checkpoint_trained")
    pop._on_iscell_changed = hooks.get("on_iscell_changed")
    pop._train_status_var = _Var()
    pop._retrain_btn = _Btn()
    pop._train_state_after_id = None
    pop._csv_path = tmp_path / "labels.csv"
    return pop


@pytest.fixture
def fake_train(monkeypatch, tmp_path):
    """Stub train.main / model loading / predict_recording; record calls."""
    calls = {}
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    monkeypatch.setattr(C, "CHECKPOINT_DIR", ckpt_dir)

    def _main(**kw):
        calls["train"] = kw
        return calls.get("auc", 0.9)

    monkeypatch.setattr(T, "main", _main)
    monkeypatch.setattr(P, "load_model_from_checkpoint",
                        lambda path, device: ("model", path))

    def _predict(rec_id, model, device, plane0=None, size_outlier_z=None):
        calls["predict"] = dict(rec_id=rec_id, model=model, plane0=plane0,
                                size_outlier_z=size_outlier_z)

    monkeypatch.setattr(P, "predict_recording", _predict)
    calls["ckpt_dir"] = ckpt_dir
    return calls


def test_retrain_warm_starts_and_repredicts_with_user_size_gate(
        tmp_path, fake_train):
    pop = _bare_popout(tmp_path)
    warm = tmp_path / "bundled.pt"
    pop._train_worker(warm, 0.0)
    assert fake_train["train"] == {"warm_start": warm,
                                   "lr": C.WARM_START_LR}
    # size_outlier_z=0 (user opted out) must reach predict_recording,
    # not be replaced by the 4.0 default.
    assert fake_train["predict"]["size_outlier_z"] == 0.0
    assert fake_train["predict"]["plane0"] == tmp_path
    assert pop._train_result[0] == "ok"
    assert pop._trained_ckpt == fake_train["ckpt_dir"] / "best.pt"


def test_retrain_without_checkpoint_trains_from_scratch(tmp_path, fake_train):
    pop = _bare_popout(tmp_path)
    pop._train_worker(None, None)
    assert fake_train["train"] == {}
    assert fake_train["predict"]["size_outlier_z"] == C.SIZE_OUTLIER_Z


def test_retrain_with_undefined_auc_skips_repredict(tmp_path, fake_train):
    fake_train["auc"] = -1.0          # train.main's "never saved best.pt"
    pop = _bare_popout(tmp_path)
    pop._train_worker(None, 4.0)
    assert "predict" not in fake_train
    assert pop._train_result[0] == "warn"
    assert pop._trained_ckpt is None


def test_hooks_snapshot_size_gate_and_warm_start(tmp_path):
    ckpt = tmp_path / "sel.pt"
    ckpt.write_bytes(b"x")
    pop = _bare_popout(tmp_path, get_size_outlier_z=lambda: 0,
                       get_ckpt_path=lambda: f"  {ckpt}  ")
    assert pop._size_outlier_z() == 0.0
    assert pop._warm_start_ckpt() == ckpt
    missing = _bare_popout(tmp_path, get_ckpt_path=lambda: str(tmp_path / "no.pt"))
    assert missing._warm_start_ckpt() is None
    assert _bare_popout(tmp_path)._size_outlier_z() == C.SIZE_OUTLIER_Z
    assert _bare_popout(tmp_path)._warm_start_ckpt() is None


def test_poll_selects_new_checkpoint_and_fires_sentinel(
        tmp_path, monkeypatch):
    selected, pings = [], []
    pop = _bare_popout(tmp_path, on_checkpoint_trained=selected.append,
                       on_iscell_changed=lambda i, v: pings.append((i, v)))

    class _Done:
        def is_alive(self):
            return False

    pop._train_thread = _Done()
    pop._train_result = ("ok", "done.")
    pop._trained_ckpt = tmp_path / "best.pt"
    pop._load_probs = lambda: None
    pop._refresh_status = lambda: None
    pop._poll_train_state()
    assert selected == [tmp_path / "best.pt"]
    assert pings == [(-1, -1)]
    assert pop._train_thread is None


def test_poll_warn_without_new_checkpoint_selects_nothing(
        tmp_path, monkeypatch):
    selected = []
    pop = _bare_popout(tmp_path, on_checkpoint_trained=selected.append)
    monkeypatch.setattr(CP.messagebox, "showwarning", lambda *a, **k: None)

    class _Done:
        def is_alive(self):
            return False

    pop._train_thread = _Done()
    pop._train_result = ("warn", "no best.pt")
    pop._trained_ckpt = None
    pop._poll_train_state()
    assert selected == []


def test_open_dff_ignores_unprefixed_memmap(tmp_path):
    n, t = 3, 4
    np.zeros((t, n), np.float32).tofile(tmp_path / "dff.memmap.float32")
    assert CP._open_dff(tmp_path, n) == (None, 0)
    np.ones((t, n), np.float32).tofile(tmp_path / "r0p7_dff.memmap.float32")
    dff, T_ = CP._open_dff(tmp_path, n)
    assert T_ == t and dff.shape == (t, n)
    del dff


def test_recording_id_matches_utils(tmp_path):
    from calliope.core import utils
    plane0 = tmp_path / "2025-01-15_00001" / "detection" / "final" / "suite2p" / "plane0"
    plane0.mkdir(parents=True)
    assert (CP.CurationPopout._derive_recording_id(plane0)
            == utils.infer_recording_id(plane0))


def test_predict_cli_passes_size_outlier_z(tmp_path, monkeypatch):
    (tmp_path / "stat.npy").write_bytes(b"")
    seen = {}
    monkeypatch.setattr(P, "load_model_from_checkpoint",
                        lambda path, device: "model")
    monkeypatch.setattr(
        P, "predict_recording",
        lambda rec_id, model, device, plane0=None, size_outlier_z=None:
            seen.setdefault("z", size_outlier_z))
    P.main(["--plane0", str(tmp_path), "--ckpt", str(tmp_path / "x.pt"),
            "--size-outlier-z", "0"])
    assert seen["z"] == 0.0


def _labels(n_rec, cls_per_rec):
    rows = []
    for r in range(n_rec):
        for i, lab in enumerate(cls_per_rec(r)):
            rows.append({"plane0_path": f"p{r}", "recording_ID": f"r{r}",
                         "ROI_number": i, "user_defined_cell": lab,
                         "timestamp_iso": ""})
    return pd.DataFrame(rows)


def test_split_labels_recording_holds_out_whole_recordings():
    df = _labels(10, lambda r: [0, 1, 0, 1])
    train_df, val_df = T._split_labels(df, "recording")
    assert not set(train_df.plane0_path) & set(val_df.plane0_path)
    assert len(val_df) == 8                     # 2 of 10 recordings


def test_split_labels_recording_refuses_single_class_side():
    # Every recording is single-class: whichever lands in val, some side
    # ends up without one of the classes.
    df = _labels(5, lambda r: [1, 1] if r else [0, 0])
    with pytest.raises(ValueError, match="needs both"):
        T._split_labels(df, "recording")


def test_split_labels_rejects_unknown_split():
    with pytest.raises(ValueError):
        T._split_labels(_labels(2, lambda r: [0, 1]), "bogus")


def test_retrain_dialog_describes_threaded_training(tmp_path, monkeypatch):
    seen = {}

    def _ask(info, warm_start):
        seen["text"] = info
        return None                       # cancel: nothing is started

    pop = _bare_popout(tmp_path, get_size_outlier_z=lambda: 0,
                       on_checkpoint_trained=lambda p: None)
    pop._ask_retrain_options = _ask
    pop._train_thread = None
    pop._on_retrain()
    text = seen["text"]
    assert "synchronous" not in text and "block" not in text
    assert "Flip buttons stay enabled" in text
    assert "size_outlier_z=0" in text
    assert "filtered dF/F memmap" in text
    assert pop._train_thread is None


def test_retrain_scratch_choice_drops_warm_start(tmp_path, monkeypatch):
    ckpt = tmp_path / "sel.pt"
    ckpt.write_bytes(b"x")
    pop = _bare_popout(tmp_path, get_ckpt_path=lambda: str(ckpt))
    pop._ask_retrain_options = lambda info, warm: {
        "mode": "scratch", "implicit": True, "reapply_root": " "}
    pop._train_thread = None
    started = {}

    class _Thread:
        def __init__(self, target, args, daemon):
            started["args"] = args

        def start(self):
            pass

    monkeypatch.setattr(CP.threading, "Thread", _Thread)
    pop._poll_train_state = lambda: None
    pop._on_retrain()
    assert started["args"] == (None, C.SIZE_OUTLIER_Z, True, None)


def _plane0(root, name, keep):
    p0 = root / name / "suite2p" / "plane0"
    p0.mkdir(parents=True)
    np.save(p0 / "stat.npy", np.array([{"npix": 1}] * len(keep)),
            allow_pickle=True)
    np.save(p0 / C.PREDICTED_MASK_NAME, np.array(keep, bool))
    (p0 / "r0p7_dff.memmap.float32").write_bytes(b"")
    return p0


def test_implicit_positives_add_only_unlabelled_kept_rois(tmp_path):
    p0 = _plane0(tmp_path, "a", [True, True, False, True])
    df = pd.DataFrame({"plane0_path": [str(p0)], "recording_ID": ["a"],
                       "ROI_number": [0], "user_defined_cell": [0],
                       "timestamp_iso": [""]})
    out = T._add_implicit_positives(df)
    added = out[out.timestamp_iso == "implicit"]
    assert sorted(added.ROI_number) == [1, 3]
    assert (added.user_defined_cell == 1).all()
    assert out[out.ROI_number == 0].user_defined_cell.tolist() == [0]


def test_label_overrides_win_over_prediction(tmp_path):
    p0 = _plane0(tmp_path, "a", [True, False, True])
    labels = pd.DataFrame({"plane0_path": [str(p0), str(p0), "elsewhere"],
                           "ROI_number": [0, 1, 2],
                           "user_defined_cell": [0, 1, 0]})
    assert CP._apply_label_overrides(p0, labels) == 2
    assert np.load(p0 / C.PREDICTED_MASK_NAME).tolist() == [False, True, True]


def test_reapply_repredicts_other_recordings_and_keeps_labels(
        tmp_path, fake_train, monkeypatch):
    from calliope.core import detection_run
    cur = _plane0(tmp_path, "cur", [True])
    other = _plane0(tmp_path, "other", [True, True])
    (other / "r0p7_filtered_dff.memmap.float32").write_bytes(b"")
    nofilt = _plane0(tmp_path, "nofilt", [True])
    rewritten = []
    monkeypatch.setattr(detection_run, "compute_filtered_dff",
                        rewritten.append)
    predicted = []
    monkeypatch.setattr(
        P, "predict_recording",
        lambda rec_id, model, device, plane0=None, size_outlier_z=None:
            predicted.append(plane0))
    labels = pd.DataFrame({"plane0_path": [str(other)], "ROI_number": [1],
                           "user_defined_cell": [0]})
    pop = _bare_popout(cur)
    pop._train_result = ("ok", "done.")
    pop._reapply(tmp_path, "model", "cpu", labels, 4.0)
    assert sorted(predicted) == sorted([other, nofilt])
    assert rewritten == [other]
    assert np.load(other / C.PREDICTED_MASK_NAME).tolist() == [True, False]
    assert pop._train_result[0] == "ok" and "2/2" in pop._train_result[1]
