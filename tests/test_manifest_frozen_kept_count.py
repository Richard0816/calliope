"""manifest.json n_cells_kept must describe the exported figures, which are
built from the filtered memmaps sized by the FROZEN r0p7_cell_mask_bool.npy,
not the live predicted_cell_mask.npy that curation can change later."""
from __future__ import annotations

import json

import numpy as np

from calliope.core.export_manifest import write_export_manifest


def _kept(figs):
    return json.loads((figs / "manifest.json").read_text("utf-8"))["n_cells_kept"]


def test_frozen_mask_wins_over_drifted_live_mask(tmp_path):
    plane0 = tmp_path / "plane0"
    plane0.mkdir()
    np.save(plane0 / "r0p7_cell_mask_bool.npy",
            np.array([1, 1, 1, 0, 0, 0], dtype=bool))       # frozen: 3
    np.save(plane0 / "predicted_cell_mask.npy",
            np.array([1, 1, 1, 1, 1, 0], dtype=bool))       # live (drifted): 5
    figs = tmp_path / "calliope_figures"
    write_export_manifest(figs, rec_id="r", params={}, plane0=plane0)
    assert _kept(figs) == 3


def test_iscell_fallback_when_no_predicted_mask(tmp_path):
    plane0 = tmp_path / "plane0"
    plane0.mkdir()
    iscell = np.zeros((5, 2), dtype=np.float32)
    iscell[[0, 2], 0] = 1
    np.save(plane0 / "iscell.npy", iscell)
    figs = tmp_path / "calliope_figures"
    write_export_manifest(figs, rec_id="r", params={}, plane0=plane0)
    assert _kept(figs) == 2
