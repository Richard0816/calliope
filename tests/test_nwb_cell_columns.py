"""NWB PlaneSegmentation cell columns: 'iscell_prob' must hold suite2p's
classifier probability (iscell[:, 1]), not the binary accept label in
column 0, and the cell-filter probability + frozen keep mask must be
exported too."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from calliope.core import nwb_export

pytestmark = pytest.mark.skipif(
    not nwb_export.nwb_available(), reason="pynwb not installed")

N, T, LY, LX = 4, 50, 20, 20


def _plane0(root: Path) -> Path:
    plane0 = root / "MyRec" / "detection" / "final" / "suite2p" / "plane0"
    plane0.mkdir(parents=True)
    rng = np.random.default_rng(0)
    np.save(plane0 / "F.npy", rng.normal(100, 5, (N, T)).astype("float32"))
    np.save(plane0 / "Fneu.npy", rng.normal(50, 3, (N, T)).astype("float32"))
    stat = []
    for i in range(N):
        yp = np.array([2 + 4 * i, 3 + 4 * i], dtype="int32")
        xp = np.array([5, 5], dtype="int32")
        stat.append({"ypix": yp, "xpix": xp,
                     "lam": np.ones(2, dtype="float32") / 2})
    np.save(plane0 / "stat.npy", np.array(stat, dtype=object),
            allow_pickle=True)
    np.save(plane0 / "ops.npy", {"Ly": LY, "Lx": LX, "fs": 10.0},
            allow_pickle=True)
    iscell = np.array([[1, 0.9], [0, 0.2], [1, 0.6], [0, 0.4]],
                      dtype="float32")
    np.save(plane0 / "iscell.npy", iscell)
    np.save(plane0 / "predicted_cell_prob.npy",
            np.array([0.8, 0.1, 0.7, 0.3], dtype="float32"))
    np.save(plane0 / "predicted_cell_mask.npy",
            np.array([1, 1, 1, 0], dtype=bool))          # live (drifted)
    np.save(plane0 / "r0p7_cell_mask_bool.npy",
            np.array([1, 0, 1, 0], dtype=bool))          # frozen
    return plane0


def test_cell_columns(tmp_path):
    out = tmp_path / "rec.nwb"
    nwb_export.export_recording_to_nwb(_plane0(tmp_path), out)
    from pynwb import NWBHDF5IO
    with NWBHDF5IO(str(out), "r") as io:
        nwb = io.read()
        ps = nwb.processing["ophys"].data_interfaces[
            "ImageSegmentation"].plane_segmentations["PlaneSegmentation"]
        np.testing.assert_allclose(ps["iscell_prob"].data[:],
                                   [0.9, 0.2, 0.6, 0.4], rtol=1e-6)
        assert list(ps["iscell"].data[:]) == [1, 0, 1, 0]
        np.testing.assert_allclose(ps["cellfilter_prob"].data[:],
                                   [0.8, 0.1, 0.7, 0.3], rtol=1e-6)
        assert list(ps["kept"].data[:]) == [1, 0, 1, 0]
