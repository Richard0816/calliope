"""find_recording_root fallback for user-named (non date-pattern) recordings.

Regression: the old fallback walked ``p.parents`` only, so passing the
recording folder itself returned its parent, and any path below plane0
(e.g. a cluster results subfolder) returned that subfolder's name.
"""
from __future__ import annotations

from pathlib import Path

from calliope.core import utils


def _plane0(root: Path) -> Path:
    p = root / "MyRec" / "detection" / "final" / "suite2p" / "plane0"
    p.mkdir(parents=True)
    return p


def test_plane0_resolves_to_user_named_recording(tmp_path):
    plane0 = _plane0(tmp_path)
    assert utils.find_recording_root(plane0) == (tmp_path / "MyRec").resolve()
    assert utils.infer_recording_id(plane0) == "MyRec"


def test_path_below_plane0_resolves_to_recording(tmp_path):
    plane0 = _plane0(tmp_path)
    deep = plane0 / "r0p7_filtered_cluster_results" / "C1" / "cross_correlation_full"
    deep.mkdir(parents=True)
    assert utils.infer_recording_id(deep) == "MyRec"


def test_recording_folder_itself(tmp_path):
    rec = tmp_path / "MyRec"
    rec.mkdir()
    assert utils.infer_recording_id(rec) == "MyRec"


def test_date_pattern_still_wins(tmp_path):
    p = tmp_path / "2025-01-15_00001" / "detection" / "final" / "suite2p" / "plane0"
    p.mkdir(parents=True)
    assert utils.infer_recording_id(p / "sub") == "2025-01-15_00001"
