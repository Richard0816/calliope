"""Tab 6 must check the *identity* of the kept ROIs behind a linkage, not
just their count.

A Tab 3 promotion can swap kept ROIs at the same count ([0,1,3] ->
[0,2,3]). The old guard compared only the count, so Export / Copy /
Reload mapped the linkage's leaves onto the new ids and silently gave
ROI 1's cluster membership to ROI 2. The ordered ids are now stored with
the linkage (``linkage_roi_ids.npy``) and compared on export, copy,
summary and reload; regenerating the filtered dF/F drops the linkage.

Methods are called unbound on stubs, so no Tk root is needed.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.cluster.hierarchy import linkage

from calliope.core import clustering as core_cl
from calliope.tabs.clustering import logic as cl_logic
from calliope.tabs.clustering import tab as ctab
from calliope.tabs.clustering.tab import ClusteringTab

BUILT_ON = np.array([0, 1, 3])
SWAPPED = np.array([0, 2, 3])


class _Var:
    def __init__(self, value=None):
        self._v = value

    def get(self):
        return self._v

    def set(self, v):
        self._v = v


def _stub(tmp_path):
    s = MagicMock()
    s._Z = linkage(np.random.default_rng(0).normal(size=(3, 4)), "ward")
    s._plane0 = tmp_path
    s._linkage_plane0 = tmp_path
    s._prefix = "r0p7_filtered_"
    s._roi_ids = BUILT_ON.copy()
    s._current_threshold = lambda: 1e9          # one cluster
    s._compute_visual_cluster_map = lambda T: {1: 1}
    s._visual_to_label = {1: 1}
    s._picked_visual_ids = lambda: [1]
    s._linkage_ids_current = lambda: ClusteringTab._linkage_ids_current(s)
    s._existing_cluster_dir = (
        lambda p0, pre: ClusteringTab._existing_cluster_dir(s, p0, pre))
    return s


def test_export_refuses_same_count_different_ids(monkeypatch, tmp_path):
    monkeypatch.setattr(ctab, "linkage_roi_ids", lambda p0, pre: SWAPPED)
    s = _stub(tmp_path)
    with pytest.raises(ValueError, match="Re-run clustering"):
        ClusteringTab._export_clusters(s)
    assert not list(tmp_path.glob("**/C*_rois.npy"))


def test_export_writes_ids_the_linkage_was_built_on(monkeypatch, tmp_path):
    monkeypatch.setattr(ctab, "linkage_roi_ids", lambda p0, pre: BUILT_ON)
    s = _stub(tmp_path)
    out = ClusteringTab._export_clusters(s)
    assert np.load(out / "C1_rois.npy").tolist() == [0, 1, 3]
    assert np.load(out / core_cl.LINKAGE_ROI_IDS_FILE).tolist() == [0, 1, 3]


def test_copy_list_empty_when_ids_swapped(monkeypatch, tmp_path):
    s = _stub(tmp_path)
    monkeypatch.setattr(ctab, "linkage_roi_ids", lambda p0, pre: BUILT_ON)
    assert ClusteringTab._picked_cluster_roi_ids(s) == [0, 1, 3]
    monkeypatch.setattr(ctab, "linkage_roi_ids", lambda p0, pre: SWAPPED)
    assert ClusteringTab._picked_cluster_roi_ids(s) == []


def _write_saved(cluster_dir, Z, ids=None, legacy_rois=None):
    cluster_dir.mkdir(parents=True, exist_ok=True)
    np.save(cluster_dir / "linkage.npy", Z)
    np.save(cluster_dir / "threshold_used.npy", np.array([1.0]))
    if ids is not None:
        np.save(cluster_dir / core_cl.LINKAGE_ROI_IDS_FILE, ids)
    if legacy_rois is not None:
        for i, r in enumerate(legacy_rois, start=1):
            np.save(cluster_dir / f"C{i}_rois.npy", np.asarray(r))
        (cluster_dir / "_indices_are_suite2p").write_text("1")


@pytest.mark.parametrize("layout", ["ids_file", "legacy_marker"])
def test_reload_refuses_swapped_ids(monkeypatch, tmp_path, layout):
    Z = linkage(np.random.default_rng(1).normal(size=(3, 4)), "ward")
    cdir = tmp_path / "r0p7_filtered_cluster_results"
    if layout == "ids_file":
        _write_saved(cdir, Z, ids=BUILT_ON)
    else:
        _write_saved(cdir, Z, legacy_rois=[[0, 3], [1]])
    monkeypatch.setattr(cl_logic, "load_filtered_dff",
                        lambda p0, pre: (None, 10, 3, None))
    monkeypatch.setattr(cl_logic.utils, "load_plane_view",
                        lambda p0: {"Lx": 4, "Ly": 4})
    args = (str(tmp_path), "r0p7_filtered_", str(cdir))

    monkeypatch.setattr(cl_logic, "linkage_roi_ids", lambda p0, pre: SWAPPED)
    with pytest.raises(ValueError, match="different set of kept"):
        cl_logic.reload_clustering_offload(*args)

    monkeypatch.setattr(cl_logic, "linkage_roi_ids",
                        lambda p0, pre: BUILT_ON)
    out = cl_logic.reload_clustering_offload(*args)
    assert out["roi_ids"].tolist() == [0, 1, 3]


def test_release_dff_invalidates_linkage(tmp_path):
    s = _stub(tmp_path)
    s._cluster_popout = None
    s.roi_list_var = _Var("1-3")
    s._invalidate_linkage = lambda: ClusteringTab._invalidate_linkage(s)
    ClusteringTab._release_dff(s)
    assert s._Z is None and s._roi_ids is None
    s.export_btn.configure.assert_called_with(state="disabled")
    assert s.roi_list_var.get() == ""


def test_headless_run_clustering_saves_roi_ids(monkeypatch, tmp_path):
    dff = np.random.default_rng(2).normal(size=(40, 3)).astype(np.float32)
    mask = np.array([1, 0, 1, 1], dtype=bool)
    monkeypatch.setattr(core_cl, "load_filtered_dff",
                        lambda p0, pre: (dff, 40, 3, mask))
    try:
        core_cl.run_clustering(tmp_path, {"threshold": 1e9},
                               write_summary=False)
    except Exception:
        pass  # later steps need real plane0 files; the export comes first
    saved = np.load(tmp_path / "r0p7_filtered_cluster_results"
                    / core_cl.LINKAGE_ROI_IDS_FILE)
    assert saved.tolist() == [0, 2, 3]
