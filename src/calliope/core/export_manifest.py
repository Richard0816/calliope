"""Per-recording export manifest writer.

Drops ``manifest.json`` next to a recording's ``calliope_figures/`` tree
so anyone who picks up the exported figures can answer four questions:
which CalLIOPE commit produced these, what params were used, what's the
indicator + tau + fs, and how many cells survived the cellfilter pass.

Intentionally small surface. Reads only what already lives on disk
(``ops.npy``, ``stat.npy``, ``predicted_cell_mask.npy``); never
re-derives anything that costs compute. Safe to call even on partial
pipelines — every field is optional and falls back to ``None``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np


def _git_sha(repo_hint: Path) -> Optional[str]:
    """HEAD SHA of the calliope checkout, or None when unavailable.

    ``repo_hint`` should point at a directory inside the calliope
    source tree. Returns ``None`` inside Nuitka-frozen binaries (no
    .git) or when the ``git`` binary isn't on PATH.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_hint),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        )
        return out.strip() or None
    except Exception:
        return None


def _sha256_file(p: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _jsonable(x: Any) -> Any:
    """Coerce numpy / Path values into JSON-serialisable forms."""
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return repr(x)


def write_export_manifest(
    figures_root: Path,
    *,
    rec_id: str,
    params: dict,
    plane0: Optional[Path] = None,
    ckpt_path: Optional[str] = None,
) -> Path:
    """Write ``manifest.json`` at ``figures_root / "manifest.json"``.

    ``figures_root`` is the per-recording figures parent (e.g.
    ``<save_folder>/calliope_figures``); the manifest sits next to the
    per-stage subdirectories so all stages share one receipt.

    Reads ``ops.npy`` / ``stat.npy`` / ``predicted_cell_mask.npy`` when
    ``plane0`` is supplied. Pulls ``tau`` / ``fs`` from ``ops`` first,
    leaving them ``None`` when ops are missing.

    Each call *merges* into any existing manifest rather than
    overwriting it: ``params`` accumulates into the union of every
    stage's params (later calls win on key collisions) and prior stage
    metadata (tau/fs/cell counts/ckpt) is carried forward when this call
    doesn't supply it. This is what lets a reload find the
    event-detection params even after a later stage (clustering, xcorr)
    also exported -- without the merge, the last writer's partial params
    were all that survived.
    """
    figures_root = Path(figures_root)
    figures_root.mkdir(parents=True, exist_ok=True)
    out = figures_root / "manifest.json"

    # Load any existing manifest so each stage's export accumulates into
    # one union instead of clobbering the others.
    prev: dict[str, Any] = {}
    if out.is_file():
        try:
            loaded = json.loads(out.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                prev = loaded
        except (OSError, ValueError):
            prev = {}
    prev_params = prev.get("params")
    merged_params = dict(prev_params) if isinstance(prev_params, dict) else {}
    merged_params.update(params or {})

    manifest: dict[str, Any] = {
        "rec_id": rec_id or prev.get("rec_id"),
        "created_at_iso": datetime.now(timezone.utc).isoformat(),
        "calliope_git_sha": (_git_sha(Path(__file__).resolve().parent)
                             or prev.get("calliope_git_sha")),
        "gcamp_variant": (merged_params.get("gcamp_variant")
                          or prev.get("gcamp_variant")),
        # Prior stage metadata carries forward; refreshed below if this
        # call supplies a plane0.
        "tau": prev.get("tau"),
        "fs": prev.get("fs"),
        "pix_to_um": prev.get("pix_to_um"),
        "n_cells_kept": prev.get("n_cells_kept"),
        "n_cells_total": prev.get("n_cells_total"),
        "cellfilter_ckpt_path": ckpt_path or prev.get("cellfilter_ckpt_path"),
        "cellfilter_ckpt_sha256": (
            _sha256_file(Path(ckpt_path)) if ckpt_path
            else prev.get("cellfilter_ckpt_sha256")
        ),
        "params": _jsonable(merged_params),
    }

    if plane0 is not None:
        plane0 = Path(plane0)
        ops_p = plane0 / "ops.npy"
        if ops_p.is_file():
            try:
                ops = np.load(ops_p, allow_pickle=True).item()
                tau = float(ops.get("tau", 0.0) or 0.0)
                fs = float(ops.get("fs", 0.0) or 0.0)
                manifest["tau"] = tau or None
                manifest["fs"] = fs or None
            except Exception:
                pass
        # Pixel calibration (µm/px) so an exported figure's scale bar is
        # reproducible from the manifest alone. Lives in
        # calliope_calibration.npy (falls back to ops['pix_to_um']).
        try:
            from .utils import load_pix_to_um
            pix = load_pix_to_um(plane0)
            if pix:
                manifest["pix_to_um"] = float(pix)
        except Exception:
            pass
        stat_p = plane0 / "stat.npy"
        if stat_p.is_file():
            try:
                stat = np.load(stat_p, allow_pickle=True)
                manifest["n_cells_total"] = int(len(stat))
            except Exception:
                pass
        mask_p = plane0 / "predicted_cell_mask.npy"
        if mask_p.is_file():
            try:
                mask = np.load(mask_p)
                manifest["n_cells_kept"] = int(np.asarray(mask).sum())
            except Exception:
                pass

    out.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out
