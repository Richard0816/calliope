"""Tracker #33 Linux Tab 6 P3: a reloaded pre-stamp clustering records
{"filtered": None}; an unchanged rewrite afterwards must not make Export
refuse it, while a real cell-set change still must."""
import numpy as np

from calliope.core import detection_run, generation
from calliope.tabs.clustering.logic import inputs_still_match


def _legacy_plane0(p, keep):
    n, t = len(keep), 12
    np.save(p / "F.npy", np.ones((n, t), np.float32))
    np.save(p / "iscell.npy", np.c_[keep, keep].astype(float))
    np.save(p / "predicted_cell_mask.npy", np.asarray(keep, bool))
    np.memmap(p / "r0p7_dff.memmap.float32", np.float32, "w+",
              shape=(t, n))[:] = np.arange(t * n).reshape(t, n)
    detection_run.compute_filtered_dff(p)
    (p / generation.FILTERED_FILE).unlink()      # pre-stamp run
    return p


def test_unchanged_rewrite_after_legacy_reload(tmp_path):
    p = _legacy_plane0(tmp_path, [1, 0, 1, 1])
    made_from = generation.cluster_input(p, True)
    assert made_from == {"filtered": None}
    detection_run.compute_filtered_dff(p)        # unchanged keep rule
    now = generation.cluster_input(p, True)
    assert now["filtered"] is not None
    assert inputs_still_match(p, made_from, now)


def test_changed_cell_set_still_refused(tmp_path):
    p = _legacy_plane0(tmp_path, [1, 0, 1, 1])
    made_from = generation.cluster_input(p, True)
    np.save(p / "predicted_cell_mask.npy", np.array([1, 1, 1, 1], bool))
    detection_run.compute_filtered_dff(p)
    assert not inputs_still_match(p, made_from, generation.cluster_input(p, True))


def _fake_tab(p, made_from):
    from types import SimpleNamespace
    from calliope.tabs.clustering.tab import ClusteringTab
    from calliope.core import utils
    prefix = "r0p7_filtered_"
    Z = np.array([[0, 1, 1.0, 2], [2, 3, 2.0, 3]])   # 3 leaves
    ids = np.where(np.load(p / "r0p7_cell_mask_bool.npy"))[0]
    fake = SimpleNamespace(
        _Z=Z, _plane0=p, _prefix=prefix, _cluster_inputs=made_from,
        _visual_to_label={1: 1},
        _current_threshold=lambda: 5.0,
        _compute_visual_cluster_map=lambda T: {1: 1},
        _linkage_ids_current=lambda: ids,
        _existing_cluster_dir=lambda plane0, pre: plane0 / f"{pre}cluster_results")
    assert utils.is_filtered_prefix(prefix)
    return ClusteringTab._export_clusters.__get__(fake)


def test_export_after_legacy_reload_and_unchanged_rewrite(tmp_path):
    p = _legacy_plane0(tmp_path, [1, 0, 1, 1])
    export = _fake_tab(p, generation.cluster_input(p, True))
    detection_run.compute_filtered_dff(p)        # unchanged keep rule
    out = export()
    assert (out / "C1_rois.npy").exists()
