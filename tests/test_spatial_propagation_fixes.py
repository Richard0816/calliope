"""Regression tests for Tab 8 (spatial propagation) audit fixes.

* ``_load_plane0_meta`` must not serve a stale ``stat.npy`` after a
  re-detection rewrote it in the same session (#127).
* The on-screen ``pix_to_um`` must go through
  ``core.scale.resolve_pix_to_um`` like the export path (#126).
* ``render_spatial_event_figures`` must return ``[]`` for a missing
  ``event_windows`` instead of crashing (#128).
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest


def build_synthetic_plane0(root, *, N=6, Ly=32, Lx=40, write_meta=False):
    """Minimal suite2p plane0: stat.npy + a legacy ops.npy (Ly/Lx)."""
    plane0 = root / "rec" / "detection" / "final" / "suite2p" / "plane0"
    plane0.mkdir(parents=True, exist_ok=True)
    stat = [{"xpix": np.array([i + 1, i + 2]), "ypix": np.array([i, i])}
            for i in range(N)]
    np.save(plane0 / "stat.npy", np.array(stat, dtype=object),
            allow_pickle=True)
    np.save(plane0 / "ops.npy", {"Ly": Ly, "Lx": Lx}, allow_pickle=True)
    return plane0


def _fake_tab():
    return SimpleNamespace(_plane0_cache={})


def _load_meta(fake, plane0):
    tab_mod = pytest.importorskip("calliope.tabs.spatial_propagation.tab")
    return tab_mod.SpatialPropagationTab._load_plane0_meta(fake, plane0)


def test_plane0_cache_invalidated_when_stat_rewritten(tmp_path):
    plane0 = build_synthetic_plane0(tmp_path, N=6, write_meta=False)
    fake = _fake_tab()
    meta1 = _load_meta(fake, plane0)
    assert len(meta1["stat_all"]) == 6
    # Same file again -> served from cache (same object).
    assert _load_meta(fake, plane0) is meta1

    # Simulate a re-detection: stat.npy rewritten with a different ROI
    # count. Force a distinct mtime so the test doesn't depend on the
    # filesystem's timestamp resolution.
    stat_path = plane0 / "stat.npy"
    new_stat = [{"xpix": np.array([1, 2]), "ypix": np.array([3, 4])}
                for _ in range(9)]
    np.save(stat_path, np.array(new_stat, dtype=object), allow_pickle=True)
    st = stat_path.stat()
    os.utime(stat_path, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000_000))

    meta2 = _load_meta(fake, plane0)
    assert len(meta2["stat_all"]) == 9
    # The superseded entry is dropped rather than accumulating.
    assert len(fake._plane0_cache) == 1


@pytest.mark.parametrize("stored, expected", [(0.0, None), (-1.5, None),
                                              (1.25, 1.25)])
def test_on_screen_pix_to_um_uses_resolver(tmp_path, monkeypatch,
                                           stored, expected):
    plane0 = build_synthetic_plane0(tmp_path, write_meta=False)
    tab_mod = pytest.importorskip("calliope.tabs.spatial_propagation.tab")
    monkeypatch.setattr(
        tab_mod.core_utils, "load_plane_view",
        lambda _p: {"Ly": 32, "Lx": 40, "pix_to_um": stored})
    meta = _load_meta(_fake_tab(), plane0)
    assert meta["pix_to_um"] == expected


@pytest.mark.parametrize("event_data", [{}, {"event_windows": None}])
def test_render_spatial_event_figures_missing_windows(tmp_path, event_data):
    from calliope.core.spatial import render_spatial_event_figures
    plane0 = build_synthetic_plane0(tmp_path, write_meta=False)
    out = render_spatial_event_figures(
        plane0, event_data, figures_dir=tmp_path / "figs")
    assert out == []
