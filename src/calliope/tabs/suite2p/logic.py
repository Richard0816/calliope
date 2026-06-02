"""Logic + calculations for the Suite2p detection tab.

This module hosts the pure / stateless helpers Tab 3 uses around its
Suite2p worker: ROI mask building, label-image rendering, ``iscell``
mask loading, etc. Keeping these out of ``tab.py`` lets the GUI module
focus on widget layout + lifecycle.

It also re-exports a couple of core types that Tab 3's docstring
references (``PreprocessResult``, ``summary_writer``) so the import
surface from inside the tab stays narrow.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...core import summary_writer
from ...core.preprocessing import PreprocessResult

__all__ = [
    "PreprocessResult",
    "summary_writer",
    "count_leaves",
    "candidate_plane0",
    "plane0_has_outputs",
    "build_label_image",
    "build_score_image",
    "load_keep_mask",
    "render_panel",
    "render_score_panel",
]


def count_leaves(d: dict) -> int:
    """Recursively count non-dict leaves in a nested dict.

    Used to summarise the popout's overrides count in the run log.
    """
    n = 0
    for v in d.values():
        if isinstance(v, dict):
            n += count_leaves(v)
        else:
            n += 1
    return n


def candidate_plane0(result: PreprocessResult) -> Path:
    """Resolve the expected plane0 folder for a preprocess result."""
    return result.out_dir / "detection" / "final" / "suite2p" / "plane0"


def plane0_has_outputs(plane0: Path) -> bool:
    """Quick existence check used by the "Load existing panels" path."""
    return (plane0 / "stat.npy").exists() and (plane0 / "ops.npy").exists()


def build_label_image(stat, Ly: int, Lx: int) -> np.ndarray:
    """``(Ly, Lx)`` int32 label image: each ROI's pixel footprint
    painted with its 1-based index. Used by the click-handler to map
    a pixel coordinate back to an ROI.
    """
    label = np.zeros((Ly, Lx), dtype=np.int32)
    for i, s in enumerate(stat, start=1):
        yp = np.asarray(s["ypix"]); xp = np.asarray(s["xpix"])
        ok = (yp >= 0) & (yp < Ly) & (xp >= 0) & (xp < Lx)
        label[yp[ok], xp[ok]] = i
    return label


def build_score_image(stat, keep: np.ndarray, probs: np.ndarray,
                      Ly: int, Lx: int) -> np.ndarray:
    """``(Ly, Lx)`` float32 image of cell-filter scores. Pixels of
    kept ROIs are painted with the ROI's ``predicted_cell_prob``;
    every other pixel stays NaN so the matplotlib overlay masks it.
    """
    img = np.full((Ly, Lx), np.nan, dtype=np.float32)
    for i, s in enumerate(stat):
        if not keep[i]:
            continue
        yp = np.asarray(s["ypix"]); xp = np.asarray(s["xpix"])
        ok = (yp >= 0) & (yp < Ly) & (xp >= 0) & (xp < Lx)
        img[yp[ok], xp[ok]] = float(probs[i])
    return img


def load_keep_mask(plane0: Path, n_total: int) -> np.ndarray:
    """Boolean mask of which ROIs to keep, reflecting current curation
    (``predicted_cell_mask`` -> ``iscell`` -> "keep everything").

    Thin delegate to :func:`calliope.core.utils.resolve_live_mask`, the
    single writer-facing keep-mask resolver.
    """
    from ...core import utils
    return utils.resolve_live_mask(plane0, n_total)


def render_panel(ax, canvas, mean, label_img, vmin, vmax, title) -> None:
    """Paint ``mean`` on ``ax`` and overlay any non-zero pixels of
    ``label_img`` coloured by ``nipy_spectral`` cluster index.
    """
    ax.clear(); ax.set_axis_off()
    ax.imshow(mean, cmap="gray", vmin=vmin, vmax=vmax)
    if label_img.max() > 0:
        overlay = np.ma.masked_where(label_img == 0, label_img)
        ax.imshow(overlay, cmap="nipy_spectral", alpha=0.45,
                  interpolation="nearest")
    ax.set_title(title, fontsize=9)
    canvas.draw_idle()


def render_background_panel(ax, canvas, mean, vmin, vmax, title):
    """Paint just the background image (no ROI overlay) on a freshly
    recreated axis. Mirrors :func:`render_score_panel`'s ``fig.clear()``
    + ``add_subplot`` so toggling away from the score panel can't leave a
    stale colourbar behind. Returns the new axes.
    """
    fig = ax.figure
    fig.clear()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.imshow(mean, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    canvas.draw_idle()
    return ax


def render_score_panel(ax, canvas, mean, score_img, vmin, vmax,
                       title):
    """Same overlay style as :func:`render_panel` but coloured by score
    across ``[0.5, 1.0]`` (the cell-filter pass range). Adds a
    colourbar to the right of the panel. The figure is wiped and the
    axes recreated each call so the colourbar can never accumulate
    shrinkage on the host axes across re-renders. Returns the freshly
    created axes so the caller can update its handle.
    """
    fig = ax.figure
    fig.clear()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.imshow(mean, cmap="gray", vmin=vmin, vmax=vmax)
    overlay = np.ma.masked_invalid(score_img)
    if overlay.count() > 0:
        im = ax.imshow(overlay, cmap="viridis", alpha=0.45,
                       vmin=0.5, vmax=1.0, interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("predicted_cell_prob (>= 0.5 only)",
                     fontsize=8)
        cb.ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9)
    canvas.draw_idle()
    return ax
