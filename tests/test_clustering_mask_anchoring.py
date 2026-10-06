"""Regression tests: headless clustering must use the memmap-anchored keep mask.

``core.clustering.run_clustering`` used to translate filtered dF/F positions
to Suite2p ids with the *live* ``predicted_cell_mask``/``iscell``. After
curation drift (live mask count != memmap column count) it silently exported
filtered-list positions while still writing the ``_indices_are_suite2p``
marker. ``stat_for_prefix`` likewise read ``r0p7_cell_mask_bool.npy``
without checking it against the memmap's column count.
"""
from __future__ import annotations

import numpy as np

from calliope.core import clustering as core_clustering

N_TOTAL = 6
T = 120
PREFIX = "r0p7_filtered_"


def _make_plane0(tmp_path, frozen_keep, live_keep=None):
    plane0 = tmp_path / "plane0"
    plane0.mkdir()
    rng = np.random.RandomState(0)
    np.save(plane0 / "F.npy", rng.rand(N_TOTAL, T).astype(np.float32))
    stat = np.array(
        [{"ypix": np.array([i]), "xpix": np.array([i]), "roi": i}
         for i in range(N_TOTAL)], dtype=object)
    np.save(plane0 / "stat.npy", stat, allow_pickle=True)
    frozen = np.zeros(N_TOTAL, dtype=bool)
    frozen[list(frozen_keep)] = True
    np.save(plane0 / "r0p7_cell_mask_bool.npy", frozen)
    if live_keep is not None:
        live = np.zeros(N_TOTAL, dtype=bool)
        live[list(live_keep)] = True
        np.save(plane0 / "predicted_cell_mask.npy", live)
    n_kept = len(frozen_keep)
    mm = np.memmap(plane0 / f"{PREFIX}dff.memmap.float32", dtype="float32",
                   mode="w+", shape=(T, n_kept))
    mm[:] = rng.randn(T, n_kept).astype(np.float32)
    mm.flush()
    del mm
    return plane0


def test_run_clustering_exports_suite2p_ids_despite_live_mask_drift(tmp_path):
    frozen_keep = [0, 2, 3, 5]
    # Live mask drifted after the memmap was written (one more ROI kept).
    plane0 = _make_plane0(tmp_path, frozen_keep, live_keep=[0, 1, 2, 3, 5])

    out = core_clustering.run_clustering(
        plane0, {"prefix": PREFIX, "threshold": None}, write_summary=False)

    export_dir = plane0 / f"{PREFIX}cluster_results"
    exported = sorted(
        int(i) for f in export_dir.glob("C*_rois.npy") for i in np.load(f))
    # Suite2p ids of the frozen keep set -- not filtered positions 0..3.
    assert exported == frozen_keep
    assert (export_dir / "_indices_are_suite2p").read_text() == "1"
    assert out["n_clusters"] >= 1


def test_run_clustering_unfiltered_prefix_exports_positions(tmp_path):
    plane0 = _make_plane0(tmp_path, [0, 1, 2, 3, 4, 5])
    (plane0 / f"{PREFIX}dff.memmap.float32").rename(
        plane0 / "r0p7_dff.memmap.float32")
    core_clustering.run_clustering(
        plane0, {"prefix": "r0p7_", "threshold": None}, write_summary=False)
    export_dir = plane0 / "r0p7_cluster_results"
    exported = sorted(
        int(i) for f in export_dir.glob("C*_rois.npy") for i in np.load(f))
    assert exported == list(range(N_TOTAL))


def test_stat_for_prefix_anchors_to_memmap_size(tmp_path):
    plane0 = _make_plane0(tmp_path, [0, 2, 3, 5], live_keep=[0, 2, 3, 5])
    # Stale persisted mask (wrong count); the live mask matches the memmap.
    stale = np.zeros(N_TOTAL, dtype=bool)
    stale[[0, 1, 2]] = True
    np.save(plane0 / "r0p7_cell_mask_bool.npy", stale)

    stat, used = core_clustering.stat_for_prefix(plane0, PREFIX)
    assert [int(s["roi"]) for s in stat] == [0, 2, 3, 5]
    assert used.tolist() == [0, 2, 3, 5]


def test_tab_logic_reexports_core_helpers():
    from calliope.tabs.clustering import logic
    for name in ("ward_linkage", "load_filtered_dff", "stat_for_prefix",
                 "filtered_to_suite2p_indices"):
        assert getattr(logic, name) is getattr(core_clustering, name)
