"""Per-recording ``meta.json`` sidecar.

Every CalLIOPE recording writes its outputs into a ``suite2p/plane0``
folder (``stat.npy``, ``F.npy``, ``spks.npy``, the ``r0p7_*`` memmaps, ...).
Historically the *parameters* that produced those arrays were scattered:
the neuropil coefficient was implicit in the ``r0p7_`` filename prefix, the
OASIS ``tau`` lived only inside ``ops.npy``, the GCaMP variant label was a
GUI-only string that was never persisted, and the dF/F / lowpass settings
were not recorded anywhere downstream consumers could read them.

This module adds a single ``<plane0>/meta.json`` sidecar that records all
per-recording parameters in one well-typed place. It is the single source
of truth the NWB exporter reads for imaging metadata (tau, fps, neuropil
coef, GCaMP variant), and it also carries the z-drift QC flags.

The schema is intentionally permissive: :func:`read_meta` always returns a
fully-populated dict (missing keys filled from :func:`default_meta`), and
:func:`update_meta` deep-merges section dicts so each pipeline stage can
stamp only the fields it owns without clobbering the others. Writes are
atomic (``tempfile`` in the same directory + :func:`os.replace`) so a
crashed run never leaves a half-written ``meta.json``.

The ``schema_version`` field lets future schema changes be detected.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Union

PathLike = Union[str, Path]

#: Bump when the schema below changes incompatibly.
SCHEMA_VERSION = 1

#: The sidecar filename written into every ``plane0`` folder.
META_FILENAME = "meta.json"


def _pipeline_version() -> str:
    """Best-effort CalLIOPE version string for provenance stamping."""
    try:  # avoid a hard import cycle at module import time
        from calliope import __version__  # type: ignore

        return str(__version__)
    except Exception:
        return "unknown"


def default_meta() -> Dict[str, Any]:
    """Return a fresh, fully-populated schema-v1 skeleton.

    Every section/key the rest of the pipeline may stamp is present with a
    ``None`` (or sensible default) placeholder so downstream consumers can
    rely on the keys existing.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "created_iso": None,
        "pipeline_version": None,
        "registration": {
            "block_size": None,
            "smooth_sigma": None,
            "nonrigid": None,
        },
        "detection": {
            "threshold_scaling": None,
            "sparse_mode": None,
            "cellpose_enabled": None,
            "cellpose_model": None,
            "cellpose_flow_threshold": None,
            "enable_sam_vcorr_pass": None,
        },
        "extraction": {
            "neuropil_coef": None,
            "tau_seconds": None,
            "gcamp_variant_label": None,
        },
        "dff": {
            "baseline_mode": None,
            "baseline_percentile": None,
            "baseline_window_sec": None,
            "baseline_min_sec": None,
            "fps": None,
        },
        "lowpass": {
            "cutoff_hz": None,
            "filter_kind": None,
            "order": None,
            "group_delay_samples": None,
        },
        "spike_inference": {
            "backend": "oasis",
        },
        "denoising": {
            "applied": False,
            "model": None,
        },
        "qc": {
            "zdrift_detected": None,
            "zdrift_n_steps": None,
            "zdrift_step_frames": None,
            "zdrift_pc1_var_explained": None,
        },
    }


def meta_path(plane0: PathLike) -> Path:
    """Return ``<plane0>/meta.json`` as a :class:`Path` (no I/O)."""
    return Path(plane0) / META_FILENAME


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overlay`` into ``base`` (in place) and return it.

    Nested dicts are merged key-by-key; any non-dict value in ``overlay``
    replaces the corresponding value in ``base``.
    """
    for key, val in overlay.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(val, dict)
        ):
            _deep_merge(base[key], val)
        else:
            base[key] = val
    return base


def read_meta(plane0: PathLike) -> Dict[str, Any]:
    """Read ``<plane0>/meta.json``, returning a fully-populated dict.

    Missing files or unreadable/legacy JSON yield the
    :func:`default_meta` skeleton. Any keys present on disk but missing
    from the current schema are preserved; any schema keys missing on disk
    are filled with their defaults. The returned dict is always safe to
    index by every key in :func:`default_meta`.
    """
    skeleton = default_meta()
    path = meta_path(plane0)
    if not path.exists():
        return skeleton
    try:
        with open(path, "r", encoding="utf-8") as fh:
            on_disk = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"[recording_meta] could not read {path}: {exc}; using defaults")
        return skeleton
    if not isinstance(on_disk, dict):
        print(f"[recording_meta] {path} is not a JSON object; using defaults")
        return skeleton
    return _deep_merge(skeleton, on_disk)


def write_meta(plane0: PathLike, meta: Dict[str, Any]) -> Path:
    """Atomically write ``meta`` to ``<plane0>/meta.json``.

    Writes to a temp file in the same directory then :func:`os.replace`\\ s
    it into place, so a reader never observes a partially-written file and
    a crash mid-write cannot corrupt an existing sidecar.
    """
    plane0 = Path(plane0)
    plane0.mkdir(parents=True, exist_ok=True)
    path = meta_path(plane0)
    payload = copy.deepcopy(meta)
    payload["schema_version"] = SCHEMA_VERSION
    if not payload.get("created_iso"):
        payload["created_iso"] = datetime.now(timezone.utc).isoformat()
    if not payload.get("pipeline_version"):
        payload["pipeline_version"] = _pipeline_version()

    fd, tmp_name = tempfile.mkstemp(
        dir=str(plane0), prefix=".meta.", suffix=".json.tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=False)
            fh.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        # Clean up the temp file on any failure so we don't litter plane0.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path


def update_meta(plane0: PathLike, **sections: Any) -> Dict[str, Any]:
    """Merge ``sections`` into ``<plane0>/meta.json`` and persist atomically.

    Each keyword argument is a top-level schema section. Dict values are
    deep-merged into the existing section so a stage can stamp only the
    fields it owns; scalar top-level fields (``pipeline_version``,
    ``created_iso``) may also be passed and are set directly.

    Example
    -------
    >>> update_meta(plane0, extraction={"tau_seconds": 0.5,
    ...                                  "neuropil_coef": 0.7},
    ...             dff={"fps": 30.0})

    Returns the merged dict that was written.
    """
    meta = read_meta(plane0)
    _deep_merge(meta, sections)
    write_meta(plane0, meta)
    return meta
