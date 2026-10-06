"""merge_gap_s is documented as inert (its merge block was disabled on
2026-05-01). Pin that: changing it must not change detected windows, with or
without the watershed split. If someone revives the merge path, this test
fails and the Tab 5 help text / README must be updated with it.
"""
from __future__ import annotations

import numpy as np
import pytest

from calliope.core import utils


def _onsets(n_rois=30, seed=0):
    rng = np.random.default_rng(seed)
    # Two bursts 0.3 s apart (close enough that walked windows can touch)
    # plus sparse background onsets.
    out = []
    for _ in range(n_rois):
        burst = np.concatenate([
            rng.normal(5.0, 0.03, 1),
            rng.normal(5.3, 0.03, 1),
            rng.uniform(0, 20, 1),
        ])
        out.append(np.sort(np.clip(burst, 0, 19.99)))
    return out


@pytest.mark.parametrize("watershed", [True, False])
def test_merge_gap_s_has_no_effect(watershed):
    onsets = _onsets()
    fps, T = 15.0, 300
    results = []
    for gap in (0.0, 5.0):
        p = utils.EventDetectionParams(
            enable_watershed_split=watershed,
            merge_gap_s=gap,
            auto_min_prominence=False,
        )
        windows, A, first_time = utils.detect_event_windows(
            onsets, T=T, fps=fps, params=p)
        results.append((np.asarray(windows), np.asarray(A)))
    np.testing.assert_array_equal(results[0][0], results[1][0])
    np.testing.assert_array_equal(results[0][1], results[1][1])
