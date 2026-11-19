"""Which dF/F generation each derived output was built from.

The filtered dF/F, Tab 4's low-pass / derivative memmaps, Tab 5's event
sheets and Tab 6's clusters are each computed from the stage before. When
an earlier stage is redone (a new full dF/F, or a different cell set after
Promote / Retrain), the later files stay on disk and can still have the
right shape -- e.g. a swapped cell set at the same count -- so a reader
would silently use traces of the wrong ROIs.

Nothing is deleted. Instead each stage records a small JSON stamp and
readers compare stamps:

* ``r0p7_dff_build.json``: a fresh id every time the full dF/F (or the
  detection it comes from) is rebuilt, ``complete`` once it is written.
* ``r0p7_filtered_generation.json``: written with the filtered dF/F; its
  ``generation`` hashes the frozen keep mask and the dF/F build id.
* ``r0p7_lowpass_generation.json``: the filtered generation Tab 4 read,
  plus an id of its own for this low-pass run.
* ``r0p7_events_generation.json``: the low-pass run Tab 5's sheets used.
* ``generation.json`` in a cluster folder: what Tab 6 clustered.

A stage marks its stamp ``pending`` once it starts overwriting data and
stamps it when done, so a crash in between is refused, not misread.

A reader whose stamp no longer matches raises :class:`StaleOutputError`
naming the stage to re-run. Runs made before stamps existed carry none
and are accepted as before, until a later stamped rewrite changes their
cell set or dF/F (``legacy_ok``). A stamp file that exists but can't be
read is refused, never taken for a pre-stamp run.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
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


def _retry(fn, tries: int = 20):
    # Windows refuses to open / replace a file another handle is using
    # for a moment (sharing violation): wait briefly and try again.
    for i in range(tries):
        try:
            return fn()
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.05)


def _read(path: Path) -> Optional[dict]:
    """The stamp at ``path``; ``None`` only when there is no such file."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        d = json.loads(_retry(lambda: path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as e:
        raise StaleOutputError(
            f"{path} is unreadable ({e}); re-run the stage that writes it.")
    if not isinstance(d, dict):
        raise StaleOutputError(f"{path} is not a valid stamp; re-run the "
                               f"stage that writes it.")
    return d


def _write(path: Path, data: dict) -> None:
    # Write a uniquely named temp file, then rename over the stamp, so a
    # crash never leaves a half-written stamp and writers never collide.
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    try:
        _retry(lambda: tmp.replace(path))
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


def mask_hash(mask) -> str:
    m = np.asarray(mask, dtype=bool).ravel()
    return hashlib.sha1(m.size.to_bytes(8, "little")
                        + np.packbits(m).tobytes()).hexdigest()


def frozen_mask_hash(plane0: Path) -> Optional[str]:
    try:
        return mask_hash(np.load(Path(plane0) / FROZEN_MASK))
    except (OSError, ValueError):
        return None


# --- full dF/F ------------------------------------------------------------

def new_dff_build(plane0: Path) -> str:
    """Call before overwriting anything the full dF/F depends on (detection
    outputs, or the dF/F memmap itself). Until :func:`finish_dff_build`,
    everything downstream is refused."""
    build = uuid.uuid4().hex
    _write(Path(plane0) / BUILD_FILE, {"build_id": build, "complete": False})
    return build


def finish_dff_build(plane0: Path) -> None:
    p = Path(plane0) / BUILD_FILE
    d = _read(p)
    if d is not None:
        d["complete"] = True
        _write(p, d)


def dff_build_id(plane0: Path) -> Optional[str]:
    d = _read(Path(plane0) / BUILD_FILE)
    if d is None:
        return None
    if not d.get("complete", True):
        raise StaleOutputError(
            f"{plane0}: the dF/F rebuild did not finish; re-run Tab 3.")
    return d.get("build_id")


# --- filtered dF/F ----------------------------------------------------------

def begin_filtered(plane0: Path) -> Optional[dict]:
    """Call right before the filtered dF/F is overwritten. Returns the
    previous stamp for :func:`stamp_filtered`."""
    plane0 = Path(plane0)
    prev = _read(plane0 / FILTERED_FILE)
    _write(plane0 / FILTERED_FILE, {"pending": True,
                                    "legacy_ok": _legacy_ok(prev, plane0)})
    return prev


def _legacy_ok(prev: Optional[dict], plane0: Path) -> bool:
    if prev is None:
        return True        # never stamped: decided by stamp_filtered
    return bool(prev.get("legacy_ok"))


def stamp_filtered(plane0: Path, mask, prev_mask_hash: Optional[str],
                   prev: Optional[dict] = None) -> str:
    """Call after the filtered dF/F and the frozen mask are written.
    ``prev_mask_hash`` is the frozen mask's hash before this write and
    ``prev`` the stamp :func:`begin_filtered` returned (if it was called)."""
    plane0 = Path(plane0)
    mh = mask_hash(mask)
    build = dff_build_id(plane0)
    if prev is None:
        prev = _read(plane0 / FILTERED_FILE)
        if prev is not None and prev.get("pending"):
            prev = {"legacy_ok": prev.get("legacy_ok"), "mask_hash": None}
    if prev is None:
        # First stamped write. Unstamped Tab 4/5/6 outputs from before
        # stay usable only if neither the cells nor the dF/F changed.
        legacy_ok = build is None and prev_mask_hash == mh
    else:
        legacy_ok = (bool(prev.get("legacy_ok"))
                     and prev.get("mask_hash", mh) in (mh, None)
                     and prev_mask_hash == mh
                     and prev.get("build_id") == build)
    gen = hashlib.sha1(f"{mh}:{build}".encode()).hexdigest()
    _write(plane0 / FILTERED_FILE, {
        "generation": gen, "mask_hash": mh, "build_id": build,
        "n_kept": int(np.asarray(mask, dtype=bool).sum()),
        "legacy_ok": legacy_ok})
    return gen


def filtered_n_kept(plane0: Path) -> Optional[int]:
    """Kept-ROI count of a stamped, complete filtered dF/F, else ``None``."""
    try:
        stamp = _read(Path(plane0) / FILTERED_FILE)
    except StaleOutputError:
        return None
    if stamp is None or stamp.get("pending"):
        return None
    n = stamp.get("n_kept")
    return int(n) if n is not None else None


def check_filtered(plane0: Path, *, need_cells: bool = True) -> Optional[str]:
    """Generation of the current filtered dF/F (``None`` for an unstamped
    run made before stamps). Raises when it is out of date, half-written,
    or -- with ``need_cells`` -- when no ROI survived the cell filter."""
    plane0 = Path(plane0)
    stamp = _read(plane0 / FILTERED_FILE)
    build = dff_build_id(plane0)
    if stamp is None:
        if build is not None:
            raise StaleOutputError(
                f"{plane0}: the dF/F was rebuilt after the filtered dF/F "
                f"was written; re-run Tab 3 (filtered dF/F).")
        return None
    if stamp.get("pending"):
        raise StaleOutputError(
            f"{plane0}: writing the filtered dF/F did not finish; re-run "
            f"Tab 3 (filtered dF/F).")
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


def _legacy_accepted(plane0: Path, gen: Optional[str]) -> bool:
    """Whether outputs made before stamps (or by a pre-stamp filtered dF/F)
    still belong to the current filtered dF/F."""
    if gen is None:
        return True
    return bool((_read(Path(plane0) / FILTERED_FILE) or {}).get("legacy_ok"))


# --- Tab 4 -------------------------------------------------------------------

def begin_lowpass(plane0: Path) -> None:
    """Call once Tab 4 starts overwriting its memmaps."""
    _write(Path(plane0) / LOWPASS_FILE, {"pending": True})


def stamp_lowpass(plane0: Path, generation: Optional[str]) -> str:
    """Call after the low-pass / derivative memmaps are written. Returns
    this low-pass run's token (what Tab 5 records)."""
    lp_id = uuid.uuid4().hex
    _write(Path(plane0) / LOWPASS_FILE,
           {"generation": generation, "lowpass_id": lp_id})
    return f"{generation}|{lp_id}"


def check_lowpass(plane0: Path) -> Optional[str]:
    """Raise unless Tab 4's low-pass / derivative memmaps match the current
    filtered dF/F. Returns their token (``None`` for pre-stamp output)."""
    plane0 = Path(plane0)
    gen = check_filtered(plane0)
    stamp = _read(plane0 / LOWPASS_FILE)
    if stamp is not None and stamp.get("pending"):
        raise StaleOutputError(
            f"{plane0}: Tab 4 did not finish writing; re-run Tab 4.")
    if stamp is None:
        if _legacy_accepted(plane0, gen):
            return None
        raise StaleOutputError(
            f"{plane0}: the low-pass traces predate a change of cell set "
            f"or dF/F; re-run Tab 4.")
    made_from = stamp.get("generation")
    ok = (made_from == gen if made_from is not None
          else _legacy_accepted(plane0, gen))     # Tab 4 run on a pre-stamp run
    if not ok:
        raise StaleOutputError(
            f"{plane0}: the low-pass traces were made for a different cell "
            f"set or dF/F; re-run Tab 4.")
    return f"{made_from}|{stamp.get('lowpass_id')}"


# --- Tab 5 ---------------------------------------------------------------------

def stamp_events(plane0: Path, token: Optional[str]) -> None:
    _write(Path(plane0) / EVENTS_FILE, {"lowpass": token})


def check_events(plane0: Path) -> Optional[str]:
    """Raise unless Tab 5's saved event sheets were made from the current
    low-pass run (and through it the current filtered dF/F)."""
    plane0 = Path(plane0)
    token = check_lowpass(plane0)
    stamp = _read(plane0 / EVENTS_FILE)
    if stamp is None:
        if token is None:
            return None          # pre-stamp events on pre-stamp low-pass
        raise StaleOutputError(
            f"{plane0}: the saved events predate the current low-pass "
            f"traces; re-run Tab 5.")
    if stamp.get("lowpass") != token:
        raise StaleOutputError(
            f"{plane0}: the saved events were made from different low-pass "
            f"traces; re-run Tab 5.")
    return token


# --- Tab 6 ---------------------------------------------------------------------

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
        gen = check_filtered(plane0)
        made_from = None if stamp is None else stamp.get("filtered")
        ok = (made_from == gen if made_from is not None
              else _legacy_accepted(plane0, gen))
        if not ok:
            raise StaleOutputError(
                f"{plane0}: these clusters were made for a different cell "
                f"set or dF/F; re-run Tab 6 clustering.")
        return
    build = dff_build_id(plane0)
    if (stamp or {}).get("build_id") != build:
        raise StaleOutputError(
            f"{plane0}: these clusters were made from an earlier dF/F; "
            f"re-run Tab 6 clustering.")
