"""Headless regression tests for Tab 6 (ClusteringTab) state handling.

* A Tab 0 batch row's non-zero ``threshold`` must survive the following
  Run (``_on_done`` used to overwrite it with the auto cut while
  ``manual_var`` stayed False, so batch exports always used the auto cut).
* ``_on_reloaded`` must enable "Export figures" (it only enabled the other
  export buttons, leaving figures unreachable after a Reload).

The tab's methods are called unbound on a stub ``self`` so no Tk root is
needed.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
from scipy.cluster.hierarchy import linkage

from calliope.tabs.clustering import tab as cl_tab
from calliope.tabs.clustering.tab import ClusteringTab


class _Var:
    def __init__(self, value=None):
        self._v = value

    def get(self):
        return self._v

    def set(self, v):
        self._v = v


def _stub(zmax=10.0):
    s = MagicMock()
    s.manual_var = _Var(False)
    s._batch_threshold = None
    s._zmax = zmax
    s._auto_threshold = 0.7
    s._manual_T = 0.7
    s._current_threshold = lambda: ClusteringTab._current_threshold(s)
    return s


def test_batch_threshold_reaches_current_cut():
    s = _stub(zmax=10.0)
    ClusteringTab.apply_batch_row(s, {"threshold": 2.5})
    applied = ClusteringTab._apply_initial_cut(s, 0.3)
    assert applied is True
    assert s.manual_var.get() is True
    assert ClusteringTab._current_threshold(s) == 2.5
    # Consumed: the next (interactive) Run starts from the auto pick.
    s.manual_var.set(False)
    assert ClusteringTab._apply_initial_cut(s, 0.3) is False
    assert ClusteringTab._current_threshold(s) == 3.0


def test_batch_threshold_clamped_to_zmax():
    s = _stub(zmax=4.0)
    ClusteringTab.apply_batch_row(s, {"threshold": 9.0})
    ClusteringTab._apply_initial_cut(s, 0.5)
    assert ClusteringTab._current_threshold(s) == 4.0


def test_batch_threshold_zero_resets_to_auto():
    s = _stub(zmax=10.0)
    ClusteringTab.apply_batch_row(s, {"threshold": 2.5})
    ClusteringTab._apply_initial_cut(s, 0.3)          # row 1: manual cut
    ClusteringTab.apply_batch_row(s, {"threshold": 0})  # row 2: auto
    assert s.manual_var.get() is False
    assert ClusteringTab._apply_initial_cut(s, 0.4) is False
    assert ClusteringTab._current_threshold(s) == 4.0


def test_reload_enables_export_figures(monkeypatch, tmp_path):
    rng = np.random.RandomState(0)
    Z = linkage(rng.rand(6, 3), method="ward")
    monkeypatch.setattr(cl_tab, "load_filtered_dff",
                        lambda *a, **k: (np.zeros((5, 6)), 5, 6, None))
    monkeypatch.setattr(cl_tab, "stat_for_prefix",
                        lambda *a, **k: ([{}] * 6, None))
    s = _stub()
    s.path_var = _Var(str(tmp_path))
    s._prefix = "r0p7_"
    data = {"Z": Z, "Lx": 4, "Ly": 4, "zmax": float(Z[:, 2].max()),
            "saved_threshold": float(Z[:, 2].max()) / 2,
            "N": 6, "T": 5, "cluster_dir": str(tmp_path)}
    ClusteringTab._on_reloaded(s, data)
    s.export_figs_btn.configure.assert_called_with(state="normal")


def test_batch_threshold_zero_disables_slider():
    s = _stub(zmax=10.0)
    ClusteringTab.apply_batch_row(s, {"threshold": 0})
    s.threshold_scale.config.assert_called_with(state="disabled")


def test_run_clustering_refuses_untranslatable_ids(monkeypatch, tmp_path):
    # keep mask count != linkage leaves -> refuse rather than export
    # filtered column positions as if they were Suite2p ids.
    import numpy as np
    import pytest
    from calliope.core import clustering

    rng = np.random.default_rng(0)
    dff = rng.normal(size=(50, 6)).astype(np.float32)
    bad_mask = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)  # 3 != 6
    monkeypatch.setattr(clustering, "load_filtered_dff",
                        lambda plane0, prefix: (dff, 50, 6, bad_mask))
    with pytest.raises(ValueError, match="cannot be translated"):
        clustering.run_clustering(tmp_path, {"threshold": 1.0},
                                  write_summary=False)


def test_gui_export_refuses_mismatched_translator(monkeypatch, tmp_path):
    import numpy as np
    import pytest
    from scipy.cluster.hierarchy import linkage
    from calliope.tabs.clustering import tab as ctab

    s = _stub()
    s._Z = linkage(np.random.default_rng(0).normal(size=(4, 3)), "ward")
    s._plane0 = tmp_path
    s._prefix = "r0p7_filtered_"
    s._current_threshold = lambda: 1.0
    monkeypatch.setattr(ctab, "filtered_to_suite2p_indices",
                        lambda p0, pre: np.arange(3))   # 3 != 4 leaves
    with pytest.raises(ValueError, match="Re-run clustering"):
        ctab.ClusteringTab._export_clusters(s)
    assert not list(tmp_path.glob("**/C*_rois.npy"))
