"""Regression tests for the 2026-10-05 legacy-audit follow-ups."""
import numpy as np

from calliope.core import utils


def _plane0(tmp_path, live, frozen=None):
    p0 = tmp_path / "plane0"
    p0.mkdir()
    np.save(p0 / "predicted_cell_mask.npy", np.asarray(live, dtype=bool))
    if frozen is not None:
        np.save(p0 / "r0p7_cell_mask_bool.npy", np.asarray(frozen, dtype=bool))
    return p0


# ---- utils.pipeline_mask_count (keep-mask drift) ----

def test_pipeline_mask_count_none_without_frozen_mask(tmp_path):
    assert utils.pipeline_mask_count(_plane0(tmp_path, [1, 1, 0]), 3) is None


def test_pipeline_mask_count_none_for_wrong_length(tmp_path):
    p0 = _plane0(tmp_path, [1, 1, 1], frozen=[1, 0])
    assert utils.pipeline_mask_count(p0, 3) is None


def test_pipeline_mask_count_reads_frozen_mask(tmp_path):
    p0 = _plane0(tmp_path, [1, 1, 1], frozen=[1, 0, 0])
    assert utils.pipeline_mask_count(p0, 3) == 1


def test_pipeline_mask_count_anchors_to_memmap_size(tmp_path):
    # Frozen mask says 1 kept, but the memmap on disk holds 2 columns
    # (the predicted mask's count): readers size by the file, so must we.
    p0 = _plane0(tmp_path, [1, 1, 0], frozen=[1, 0, 0])
    T = 5
    np.save(p0 / "F.npy", np.zeros((3, T), dtype=np.float32))
    np.zeros((T, 2), dtype=np.float32).tofile(
        p0 / "r0p7_filtered_dff.memmap.float32")
    assert utils.pipeline_mask_count(p0, 3) == 2
