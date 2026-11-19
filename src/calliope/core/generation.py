"""Which dF/F generation each derived output was built from.

The filtered dF/F, Tab 4's low-pass / derivative memmaps, Tab 5's event
sheets and Tab 6's clusters are each computed from the stage before. When
an earlier stage is redone (a new full dF/F, or a different cell set after
Promote / Retrain), the later files stay on disk and can still have the
right shape -- e.g. a swapped cell set at the same count -- so a reader
would silently use traces of the wrong ROIs.

Nothing is deleted. Instead each stage records a small JSON stamp and
readers compare stamps:

* ``r0p7_dff_build.json``: a fresh id every time the full dF/F is built.
* ``r0p7_filtered_generation.json``: written with the filtered dF/F; its
  ``generation`` hashes the frozen keep mask and the dF/F build id.
* ``r0p7_lowpass_generation.json``: the filtered generation Tab 4 read.
* ``r0p7_events_generation.json``: the generation Tab 5's sheets used.
* ``generation.json`` in a cluster folder: what Tab 6 clustered.

A reader whose stamp no longer matches raises :class:`StaleOutputError`
naming the stage to re-run. Runs made before stamps existed carry none
and are accepted as before, unless a later stamped rewrite changed the
cell set or the dF/F (``legacy_ok``).
"""
from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

BUILD_FILE = "r0p7_dff_build.json"
FILTERED_FILE = "r0p7_filtered_generation.json"
LOWPASS_FILE = "r0p7_lowpass_generation.json"
EVENTS_FILE = "r0p7_events_generation.json"
CLUSTER_FILE = "generation.json"
FROZEN_MASK = "r0p7_cell_mask_bool.npy"


class StaleOutputError(RuntimeError):
    """A derived output was built from an earlier generation."""


def _read(path: Path) -> Optional[dict]:
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def _write(path: Path, data: dict) -> None:
    # Write-then-rename so a crash never leaves a half-written stamp.
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)


def mask_hash(mask) -> str:
    m = np.asarray(mask, dtype=bool).ravel()
    return hashlib.sha1(m.size.to_bytes(8, "little")
                        + np.packbits(m).tobytes()).hexdigest()


def frozen_mask_hash(plane0: Path) -> Optional[str]:
    try:
        return mask_hash(np.load(Path(plane0) / FROZEN_MASK))
    except (OSError, ValueError):
        return None


def new_dff_build(plane0: Path) -> str:
    """Call before (re)writing the full dF/F: everything after it is now
    from an earlier generation until it is rebuilt."""
    build = uuid.uuid4().hex
    _write(Path(plane0) / BUILD_FILE, {"build_id": build})
    return build


def dff_build_id(plane0: Path) -> Optional[str]:
    d = _read(Path(plane0) / BUILD_FILE)
    return d.get("build_id") if d else None


def stamp_filtered(plane0: Path, mask, prev_mask_hash: Optional[str]) -> str:
    """Call after the filtered dF/F and the frozen mask are written.
    ``prev_mask_hash`` is the frozen mask's hash before this write."""
    plane0 = Path(plane0)
    mh = mask_hash(mask)
    build = dff_build_id(plane0)
    prev = _read(plane0 / FILTERED_FILE)
    if prev is None:
        # First stamped write. Unstamped Tab 4/5/6 outputs from before
        # stay usable only if neither the cells nor the dF/F changed.
        legacy_ok = build is None and prev_mask_hash == mh
    else:
        legacy_ok = (bool(prev.get("legacy_ok")) and prev.get("mask_hash") == mh
                     and prev.get("build_id") == build)
    gen = hashlib.sha1(f"{mh}:{build}".encode()).hexdigest()
    _write(plane0 / FILTERED_FILE, {
        "generation": gen, "mask_hash": mh, "build_id": build,
        "n_kept": int(np.asarray(mask, dtype=bool).sum()),
        "legacy_ok": legacy_ok})
    return gen


def check_filtered(plane0: Path, *, need_cells: bool = True) -> Optional[str]:
    """Generation of the current filtered dF/F (``None`` for an unstamped
    run made before stamps). Raises when it is out of date, or -- with
    ``need_cells`` -- when no ROI survived the cell filter."""
    plane0 = Path(plane0)
    stamp = _read(plane0 / FILTERED_FILE)
    build = dff_build_id(plane0)
    if stamp is None:
        if build is not None:
            raise StaleOutputError(
                f"{plane0}: the dF/F was rebuilt after the filtered dF/F "
                f"was written; re-run Tab 3 (filtered dF/F).")
        return None
    if stamp.get("build_id") != build or (
            stamp.get("mask_hash") != frozen_mask_hash(plane0)):
        raise StaleOutputError(
            f"{plane0}: the filtered dF/F is out of date; re-run Tab 3 "
            f"(filtered dF/F).")
    if need_cells and not stamp.get("n_kept"):
        raise StaleOutputError(
            f"{plane0}: no ROI survives the cell filter; change the keep "
            f"rule in Tab 3 and re-run it.")
    return stamp.get("generation")


def _check_stamp(plane0: Path, stamp: Optional[dict], rerun: str) -> Optional[str]:
    gen = check_filtered(plane0)
    if stamp is None:
        if gen is None:
            return None                     # nothing stamped yet: as before
        if (_read(Path(plane0) / FILTERED_FILE) or {}).get("legacy_ok"):
            return gen                      # pre-stamp output, nothing changed
        raise StaleOutputError(
            f"{plane0}: these results predate a change of cell set or "
            f"dF/F; {rerun}.")
    if stamp.get("generation") != gen:
        raise StaleOutputError(
            f"{plane0}: these results were made for a different cell set "
            f"or dF/F; {rerun}.")
    return gen


def stamp_lowpass(plane0: Path, generation: Optional[str]) -> None:
    _write(Path(plane0) / LOWPASS_FILE, {"generation": generation})


def check_lowpass(plane0: Path) -> Optional[str]:
    """Raise unless Tab 4's low-pass / derivative memmaps match the current
    filtered dF/F. Returns the generation they belong to."""
    return _check_stamp(plane0, _read(Path(plane0) / LOWPASS_FILE),
                        "re-run Tab 4")


def stamp_events(plane0: Path, generation: Optional[str]) -> None:
    _write(Path(plane0) / EVENTS_FILE, {"generation": generation})


def check_events(plane0: Path) -> Optional[str]:
    """Raise unless Tab 5's saved event sheets match the current low-pass
    traces (and through them the filtered dF/F)."""
    check_lowpass(plane0)
    return _check_stamp(plane0, _read(Path(plane0) / EVENTS_FILE),
                        "re-run Tab 4 / Tab 5")


def cluster_input(plane0: Path, filtered: bool) -> dict:
    """What a clustering run reads: the full dF/F build, plus the filtered
    generation for a filtered prefix."""
    plane0 = Path(plane0)
    if filtered:
        return {"filtered": check_filtered(plane0)}
    return {"build_id": dff_build_id(plane0)}


def stamp_clusters(cluster_dir: Path, inputs: dict) -> None:
    _write(Path(cluster_dir) / CLUSTER_FILE, dict(inputs))


def check_clusters(plane0: Path, cluster_dir: Path, filtered: bool) -> None:
    """Raise unless the saved clusters in ``cluster_dir`` were built from
    the current dF/F (and cell set, for a filtered prefix)."""
    plane0 = Path(plane0)
    stamp = _read(Path(cluster_dir) / CLUSTER_FILE)
    if filtered:
        _check_stamp(plane0, None if stamp is None
                     else {"generation": stamp.get("filtered")},
                     "re-run Tab 6 clustering")
        return
    build = dff_build_id(plane0)
    if stamp is None:
        if build is not None and not (
                _read(plane0 / FILTERED_FILE) or {}).get("legacy_ok", True):
            raise StaleOutputError(
                f"{plane0}: these clusters predate a dF/F rebuild; re-run "
                f"Tab 6 clustering.")
        return
    if stamp.get("build_id") != build:
        raise StaleOutputError(
            f"{plane0}: these clusters were made from an earlier dF/F; "
            f"re-run Tab 6 clustering.")
