from types import SimpleNamespace

import numpy as np

from calliope.core import utils
from calliope.tabs.suite2p.curation_popout import CurationPopout


def test_fps_prefers_ops_fs(tmp_path):
    assert utils.get_fps_from_notes(str(tmp_path)) == utils.DEFAULT_FPS
    assert utils.ops_fs(tmp_path) is None
    np.save(tmp_path / "ops.npy", {"fs": 30.0}, allow_pickle=True)
    assert utils.get_fps_from_notes(str(tmp_path)) == 30.0
    np.save(tmp_path / "ops.npy", {"fs": 0.0}, allow_pickle=True)
    assert utils.get_fps_from_notes(str(tmp_path)) == utils.DEFAULT_FPS


def test_load_iscell_reports_reason(tmp_path):
    fake = SimpleNamespace(_plane0=tmp_path, _n_total=3)
    load = lambda: CurationPopout._load_iscell(fake)
    assert load() is None and "No iscell.npy" in fake._iscell_err
    np.save(tmp_path / "iscell.npy", np.ones((2, 2)))
    assert load() is None and "2 ROIs" in fake._iscell_err
    (tmp_path / "iscell.npy").write_bytes(b"junk")
    assert load() is None and "could not be read" in fake._iscell_err
    np.save(tmp_path / "iscell.npy", np.ones((3, 2)))
    assert load().shape == (3, 2)
