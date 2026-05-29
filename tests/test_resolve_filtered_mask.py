"""Tests for utils.resolve_filtered_mask -- the shared keep-mask resolver
that sizes the filtered dF/F memmaps against the file actually on disk.

The bug this guards: the filtered memmaps are column-sliced to whatever
keep mask was live when Tab 3 wrote them (persisted as
``r0p7_cell_mask_bool.npy``). If ``predicted_cell_mask.npy`` / ``iscell.npy``
later drift (re-curation, "Promote to filter mask"), re-resolving the mask
at read time yields the wrong column count, and ``np.memmap`` asks for a
mapping larger than the file -> ``[WinError 8]`` on Windows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from calliope.core import utils


def _save_mask(plane0: Path, name: str, keep: np.ndarray) -> None:
    np.save(plane0 / name, np.asarray(keep, dtype=bool))


def _write_memmap(plane0: Path, name: str, T: int, N: int) -> Path:
    p = plane0 / name
    arr = np.memmap(p, dtype="float32", mode="w+", shape=(T, N))
    arr.flush()
    del arr
    return p


# --------------------------------------------------------------------------

def test_prefers_persisted_mask_when_predicted_has_drifted(tmp_path):
    """The WinError 8 scenario: the memmap was written with the persisted
    mask (3 kept), but predicted_cell_mask has since grown to 7. The
    resolver must return the 3-column mask that matches the file."""
    T, N_total = 5, 10
    persisted = np.zeros(N_total, bool); persisted[:3] = True
    predicted = np.zeros(N_total, bool); predicted[:7] = True
    _save_mask(tmp_path, "r0p7_cell_mask_bool.npy", persisted)
    _save_mask(tmp_path, "predicted_cell_mask.npy", predicted)
    mm = _write_memmap(tmp_path, "r0p7_filtered_dff.memmap.float32", T, 3)

    mask, n_kept = utils.resolve_filtered_mask(
        tmp_path, N_total, memmap_path=mm, T=T)

    assert n_kept == 3
    assert np.array_equal(mask, persisted)


def test_fresh_run_predicted_matches_file(tmp_path):
    """No persisted sidecar, predicted_cell_mask matches the file -> used."""
    T, N_total = 4, 8
    predicted = np.zeros(N_total, bool); predicted[:4] = True
    _save_mask(tmp_path, "predicted_cell_mask.npy", predicted)
    mm = _write_memmap(tmp_path, "r0p7_filtered_dff.memmap.float32", T, 4)

    mask, n_kept = utils.resolve_filtered_mask(
        tmp_path, N_total, memmap_path=mm, T=T)

    assert n_kept == 4
    assert np.array_equal(mask, predicted)


def test_no_memmap_uses_authoritative_first(tmp_path):
    """With no file to size against, the persisted mask wins over a
    drifted predicted mask (authoritative-first order)."""
    N_total = 10
    persisted = np.zeros(N_total, bool); persisted[:3] = True
    predicted = np.zeros(N_total, bool); predicted[:7] = True
    _save_mask(tmp_path, "r0p7_cell_mask_bool.npy", persisted)
    _save_mask(tmp_path, "predicted_cell_mask.npy", predicted)

    mask, n_kept = utils.resolve_filtered_mask(tmp_path, N_total)

    assert n_kept == 3
    assert np.array_equal(mask, persisted)


def test_iscell_fallback_2d(tmp_path):
    """iscell.npy (suite2p 2-D format) used when no cell-filter mask."""
    T, N_total = 3, 6
    ic = np.zeros((N_total, 2), dtype=np.float32)
    ic[:2, 0] = 1.0  # first two ROIs are cells
    np.save(tmp_path / "iscell.npy", ic)
    mm = _write_memmap(tmp_path, "r0p7_filtered_dff.memmap.float32", T, 2)

    mask, n_kept = utils.resolve_filtered_mask(
        tmp_path, N_total, memmap_path=mm, T=T)

    assert n_kept == 2
    assert mask.tolist() == [True, True, False, False, False, False]


def test_all_ones_when_no_mask_files(tmp_path):
    """Nothing on disk and nothing to size against -> keep everything."""
    mask, n_kept = utils.resolve_filtered_mask(tmp_path, 5)
    assert n_kept == 5
    assert mask.all()


def test_raises_clear_error_on_unresolvable_drift(tmp_path):
    """File needs 6 columns but no available mask sums to 6 -> a clear
    ValueError naming the candidate sums, not a cryptic WinError."""
    T, N_total = 4, 10
    persisted = np.zeros(N_total, bool); persisted[:3] = True
    predicted = np.zeros(N_total, bool); predicted[:7] = True
    _save_mask(tmp_path, "r0p7_cell_mask_bool.npy", persisted)
    _save_mask(tmp_path, "predicted_cell_mask.npy", predicted)
    mm = _write_memmap(tmp_path, "r0p7_filtered_dff.memmap.float32", T, 6)

    with pytest.raises(ValueError) as exc:
        utils.resolve_filtered_mask(tmp_path, N_total, memmap_path=mm, T=T)
    msg = str(exc.value)
    assert "6 ROI columns" in msg
    assert "drifted" in msg


def test_filtered_mask_handles_shrink_after_curation(tmp_path):
    """The real WinError repro: the memmap was written at N_kept=7, then
    the user curated DOWN to 3 (predicted_cell_mask shrank). The reader
    must still resolve to the persisted 7 that matches the file."""
    T, N_total = 5, 10
    persisted = np.zeros(N_total, bool); persisted[:7] = True   # at write time
    curated = np.zeros(N_total, bool); curated[:3] = True       # shrank since
    _save_mask(tmp_path, "r0p7_cell_mask_bool.npy", persisted)
    _save_mask(tmp_path, "predicted_cell_mask.npy", curated)
    mm = _write_memmap(tmp_path, "r0p7_filtered_dff.memmap.float32", T, 7)

    mask, n_kept = utils.resolve_filtered_mask(
        tmp_path, N_total, memmap_path=mm, T=T)

    assert n_kept == 7
    assert np.array_equal(mask, persisted)


def test_live_mask_ignores_persisted_to_avoid_fixpoint(tmp_path):
    """The writer-facing resolver must NEVER read r0p7_cell_mask_bool --
    it reflects current curation (predicted), so a re-run shrinks/grows
    correctly instead of reusing its own stale prior output."""
    N_total = 10
    persisted = np.zeros(N_total, bool); persisted[:3] = True
    predicted = np.zeros(N_total, bool); predicted[:7] = True
    _save_mask(tmp_path, "r0p7_cell_mask_bool.npy", persisted)
    _save_mask(tmp_path, "predicted_cell_mask.npy", predicted)

    mask = utils.resolve_live_mask(tmp_path, N_total)

    assert int(mask.sum()) == 7              # follows live predicted, not r0p7
    assert np.array_equal(mask, predicted)


def test_live_mask_iscell_then_ones(tmp_path):
    N_total = 6
    # iscell fallback (2-D suite2p format)
    ic = np.zeros((N_total, 2), dtype=np.float32); ic[:2, 0] = 1.0
    np.save(tmp_path / "iscell.npy", ic)
    assert int(utils.resolve_live_mask(tmp_path, N_total).sum()) == 2
    # all-ones when nothing exists
    assert utils.resolve_live_mask(tmp_path / "nope", N_total).all()


def test_writer_reader_agree_on_fresh_run(tmp_path):
    """Invariant that protects the WinError fix: on a fresh write, the
    mask the writer slices to (resolve_live_mask) must size the memmap
    that the reader (resolve_filtered_mask) later opens."""
    T, N_total = 4, 9
    predicted = np.zeros(N_total, bool); predicted[:5] = True
    _save_mask(tmp_path, "predicted_cell_mask.npy", predicted)

    # Writer side: slice + persist the live mask.
    live = utils.resolve_live_mask(tmp_path, N_total)
    _save_mask(tmp_path, "r0p7_cell_mask_bool.npy", live)
    mm = _write_memmap(tmp_path, "r0p7_filtered_dff.memmap.float32", T, int(live.sum()))

    # Reader side must agree.
    _mask, n_kept = utils.resolve_filtered_mask(tmp_path, N_total, memmap_path=mm, T=T)
    assert n_kept == int(live.sum()) == 5


def test_wrong_length_mask_skipped(tmp_path):
    """A mask whose length != n_total is ignored (stale from a different
    detection run), falling through to the next candidate."""
    T, N_total = 3, 8
    np.save(tmp_path / "predicted_cell_mask.npy",
            np.ones(99, dtype=bool))  # wrong length -> skipped
    good = np.zeros(N_total, bool); good[:5] = True
    _save_mask(tmp_path, "r0p7_cell_mask_bool.npy", good)
    mm = _write_memmap(tmp_path, "r0p7_filtered_dff.memmap.float32", T, 5)

    mask, n_kept = utils.resolve_filtered_mask(
        tmp_path, N_total, memmap_path=mm, T=T)

    assert n_kept == 5
    assert np.array_equal(mask, good)
