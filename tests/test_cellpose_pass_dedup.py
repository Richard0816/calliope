"""A cell found by both Cellpose passes must not become two ROIs.

Before 2026-11-09 the extra (SAM-on-Vcorr) pass was filtered against Sparsery
only, so a cell the mean-image pass had already added came back as a duplicate.
"""
import numpy as np

from calliope.core.sparse_plus_cellpose import filter_extra_cellpose_passes


def _roi(y0, x0, n=6):
    yy, xx = np.mgrid[y0:y0 + n, x0:x0 + n]
    return {"ypix": yy.ravel(), "xpix": xx.ravel(), "lam": np.ones(n * n)}


def test_extra_pass_drops_rois_already_found_by_mean_image_pass():
    sparsery = [_roi(0, 0)]
    mean_pass_kept = [_roi(20, 20)]
    extra = [_roi(21, 21),          # same cell as the mean-image ROI -> dropped
             _roi(1, 1),            # same cell as a Sparsery ROI -> dropped
             _roi(40, 40)]          # new cell -> kept
    (tag, kept), = filter_extra_cellpose_passes(sparsery, mean_pass_kept, [("cellpose_sam", extra)],
                                                64, 64, max_overlap=0.3, verbose=False)
    assert tag == "cellpose_sam"
    assert [int(r["ypix"].min()) for r in kept] == [40]
