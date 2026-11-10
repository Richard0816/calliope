"""Batch hands write_event_summary_offload numpy arrays, not lists.

The 2026-11-08 in vivo batch crashed here ("truth value of an array ... is
ambiguous") and wrote no event sheets.
"""
import numpy as np

from calliope.core.event_detection_run import write_event_summary_offload


def test_summary_accepts_array_windows(tmp_path, capsys):
    data = {"event_windows": np.array([[1.0, 2.0], [5.0, 6.0]]),
            "onsets_by_roi": [np.array([1.5, 5.5]), np.array([9.0])],
            "fps": 10.0, "T": 100, "N_kept": 2, "kept_idx": np.array([0, 3])}
    write_event_summary_offload(str(tmp_path), data)
    assert "windows=2 onsets=3 EventOnsets rows=2" in capsys.readouterr().out
