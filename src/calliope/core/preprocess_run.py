"""Headless preprocessing wrapper for batch execution and Tab 1.

Why this module exists
----------------------
``calliope.core.preprocessing.preprocess_tiff`` and
``preprocess_tiff_group`` are already pure-function entry points
(no Tk dependency), so this module is a thin convenience wrapper
that picks single vs grouped automatically and exposes a uniform
``run_preprocess(src_tiffs, data_root, params, ...)`` shape that
matches the rest of the ``*_run`` family.

Public surface
--------------
``run_preprocess(src_tiffs, data_root, params, *,
                 recording_name=None, figures_dir=None,
                 progress_cb=None)``
    Run drift correction + QC GIF + blob detection. Returns the
    :class:`PreprocessResult` augmented with a ``shifted_folder``
    convenience path so downstream stages can point detection at
    that folder directly.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Optional, Sequence

from . import preprocessing


_BLOB_KEYS = ("scale_tol", "min_contrast", "min_area_px", "max_area_px")
_QC_KEYS = ("downsample_t", "max_size_px", "playback_fps",
            "clip_low", "clip_high")


def _split_params(params: dict) -> tuple[float, dict, dict]:
    """Pull soma diameter + blob/QC subsets out of the unified params
    dict, mirroring ``PreprocessTab._start_run``.
    """
    soma = float(params.get("soma_diameter_px", 12.0))
    blob = {k: params[k] for k in _BLOB_KEYS if k in params}
    qc = {k: params[k] for k in _QC_KEYS if k in params}
    return soma, blob, qc


def run_preprocess(
    src_tiffs: Sequence[str | Path],
    data_root: str | Path,
    params: dict,
    *,
    recording_name: Optional[str] = None,
    figures_dir: Optional[Path] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
):
    """Run Tab 1 preprocessing on one recording.

    Parameters
    ----------
    src_tiffs : sequence of paths
        One or more raw TIFFs. A single path runs ``preprocess_tiff``;
        multiple paths run ``preprocess_tiff_group`` (concatenates
        them in trailing-index order).
    data_root : str | Path
        Where to write the recording folder
        (``<data_root>/<recording_name>/``).
    params : dict
        Same shape as :class:`PreprocessTab.PARAM_SPEC`:
        ``soma_diameter_px``, plus any of the blob and QC keys.
    recording_name : str | None
        Folder name. Defaults to single-TIFF stem or '+'-joined
        stems for groups (matches Tab 1's
        ``_resolved_identifier`` rule).
    figures_dir : Path | None
        If set, copies ``qc.gif`` (the only image artefact) into
        this directory so the batch report's
        ``calliope_figures/preprocess/`` subfolder picks it up.

    Returns
    -------
    PreprocessResult
        Same dataclass Tab 1 publishes, with a ``shifted_folder``
        attribute attached at runtime so callers can pass it to
        ``detection_run.run_detection`` without a second
        ``Path.parent`` walk.
    """
    srcs = [Path(p) for p in src_tiffs]
    if not srcs:
        raise ValueError("run_preprocess: no source TIFFs supplied")

    soma, blob, qc = _split_params(params)

    if len(srcs) == 1:
        result = preprocessing.preprocess_tiff(
            src_tiff=srcs[0],
            data_root=str(data_root),
            recording_name=recording_name,
            soma_diameter_px=soma,
            progress_cb=progress_cb,
            blob_params=blob,
            qc_params=qc,
        )
    else:
        result = preprocessing.preprocess_tiff_group(
            src_tiffs=srcs,
            data_root=str(data_root),
            recording_name=recording_name,
            soma_diameter_px=soma,
            progress_cb=progress_cb,
            blob_params=blob,
            qc_params=qc,
        )

    # ``shifted_tiff`` lives inside the recording folder; the
    # detection backend takes a folder, so attach a convenience
    # attribute pointing at it.
    shifted_folder = Path(result.shifted_tiff).parent
    setattr(result, "shifted_folder", shifted_folder)

    if figures_dir is not None:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        try:
            qc_gif = Path(result.qc_gif)
            if qc_gif.exists():
                shutil.copy2(qc_gif, figures_dir / qc_gif.name)
        except Exception as e:
            if progress_cb is not None:
                progress_cb(f"[preprocess] qc gif copy failed: {e}")

    return result
