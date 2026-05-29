"""End-to-end batch pipeline orchestrator (Tab 0).

Drives one recording through Tabs 3-8 in sequence:

    detection  -> low-pass  -> event detection  -> clustering
                                                -> cross-correlation
                                                -> spatial figures

All figures from each stage are saved under
``<save_folder>/calliope_figures/<stage>/`` so the user has a full
visual record without opening the GUI.

Public surface
--------------
``run_recording(tiff_folder, save_folder, params, *, ckpt_path=None,
                tau_override=None, baseline_mode='first_n',
                baseline_min=2.0, recording_id=None, progress_cb=None)``
    Run the full pipeline for one recording. Returns a dict whose
    ``status`` is ``"ok"`` (everything succeeded), ``"partial"``
    (some stages failed but detection completed) or ``"failed"``
    (detection itself crashed). Stage-by-stage timing + error
    messages are in ``stages``.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Callable, Optional

from . import (
    preprocessing, detection_run, lowpass_run, event_detection_run,
    clustering, crosscorrelation, spatial,
)
from .export_manifest import write_export_manifest


# Stage names match the figure subfolder layout. Order matters --
# downstream stages depend on artefacts written by upstream stages.
STAGES = ("preprocess", "detection", "lowpass", "event_detection",
          "clustering", "crosscorrelation", "spatial_propagation")


def _figures_dir_for(save_folder: Path, stage: str) -> Path:
    return Path(save_folder) / "calliope_figures" / stage


def run_recording(
    src_tiffs,
    save_folder: str,
    params: dict,
    *,
    path_to_ops: Optional[str] = None,
    ckpt_path: Optional[str] = None,
    tau_override: Optional[float] = None,
    baseline_mode: str = "first_n",
    baseline_min: float = 2.0,
    recording_id: Optional[str] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """Run the full Tabs 1+3-8 pipeline on a single recording.

    Parameters
    ----------
    src_tiffs : sequence of paths
        Raw TIFFs for the recording. A single path runs the
        single-TIFF preprocessing path; multiple paths run the
        grouped path (concatenated in trailing-index order). Inputs
        are interpreted exactly as Tab 1 does.
    save_folder : str
        Where outputs land. Preprocess writes the recording folder at
        ``save_folder`` (Tab 1's data root layout, with
        ``save_folder.parent`` as the data root and ``save_folder.name``
        as the recording name); detection's plane0 ends up at
        ``<save_folder>/detection/final/suite2p/plane0``; figures land
        under ``<save_folder>/calliope_figures/<stage>/``.
    params : dict
        Union of every tab's tunable knobs (Tabs 1 + 3-7).
    progress_cb : callable
        Called with status strings throughout the run.

    Returns
    -------
    dict
        ``{
            recording_id, save_folder, status, plane0,
            stages: {<stage>: {status, duration_s, error|None}},
            figures: {<stage>: [paths]},
        }``
    """
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    if isinstance(src_tiffs, (str, Path)):
        tiff_input = [Path(src_tiffs)]
    else:
        tiff_input = [Path(p) for p in src_tiffs]
    rec_id = recording_id or save_folder.name

    def _emit(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(f"[{rec_id}] {msg}")

    stages: dict[str, dict] = {s: {"status": "skipped", "duration_s": 0.0,
                                   "error": None}
                               for s in STAGES}
    figures: dict[str, list[str]] = {s: [] for s in STAGES}
    plane0: Optional[Path] = None
    overall_status = "ok"

    # ----- Stage 0: preprocess (Tab 1) ------------------------------
    # Tab 1 writes to ``<data_root>/<recording_name>/``. We pass the
    # parent of ``save_folder`` as the data root and the leaf name as
    # the recording name so the preprocessed folder lands at exactly
    # ``save_folder``. Detection then writes its outputs as
    # subfolders under ``save_folder``, matching Tab 1's layout.
    t0 = time.time()
    try:
        _emit("preprocess: starting")
        pre = preprocessing.run_preprocess(
            src_tiffs=tiff_input,
            data_root=str(save_folder.parent),
            params=params,
            recording_name=save_folder.name,
            figures_dir=_figures_dir_for(save_folder, "preprocess"),
            progress_cb=_emit,
        )
        detection_tiff_folder = str(pre.shifted_folder)
        stages["preprocess"].update(
            status="ok", duration_s=time.time() - t0)
        _emit(f"preprocess: done -> {detection_tiff_folder}")
    except Exception as e:
        stages["preprocess"].update(
            status="failed", duration_s=time.time() - t0,
            error=f"{e}\n{traceback.format_exc()}")
        _emit(f"preprocess: FAILED -- {e}")
        return {
            "recording_id": rec_id,
            "save_folder": str(save_folder),
            "status": "failed",
            "plane0": None,
            "stages": stages,
            "figures": figures,
        }

    # ----- Stage 1: detection (Tab 3) -------------------------------
    t0 = time.time()
    try:
        _emit("detection: starting")
        det = detection_run.run_detection(
            tiff_folder=detection_tiff_folder,
            save_folder=str(save_folder),
            params=params,
            path_to_ops=path_to_ops,
            tau_override=tau_override,
            ckpt_path=ckpt_path,
            rec_id=rec_id,
            baseline_mode=baseline_mode,
            baseline_min=baseline_min,
            figures_dir=_figures_dir_for(save_folder, "detection"),
            progress_cb=_emit,
        )
        plane0 = det["plane0"]
        stages["detection"].update(
            status="ok", duration_s=time.time() - t0)
        figures["detection"] = det.get("figures", [])
        _emit(f"detection: done (plane0={plane0})")
        # Drop the manifest as soon as detection lands a plane0 -- any
        # later stage may crash, but we still want a receipt next to
        # the figures that did get written.
        try:
            manifest_path = write_export_manifest(
                save_folder / "calliope_figures",
                rec_id=rec_id, params=params,
                plane0=Path(plane0), ckpt_path=ckpt_path,
            )
            _emit(f"manifest: {manifest_path.name}")
        except Exception as e:
            _emit(f"manifest: skipped ({e})")
    except Exception as e:
        stages["detection"].update(
            status="failed", duration_s=time.time() - t0,
            error=f"{e}\n{traceback.format_exc()}")
        _emit(f"detection: FAILED -- {e}")
        return {
            "recording_id": rec_id,
            "save_folder": str(save_folder),
            "status": "failed",
            "plane0": None,
            "stages": stages,
            "figures": figures,
        }

    # ----- Stage 2: low-pass + derivative (Tab 4) -------------------
    t0 = time.time()
    try:
        _emit("lowpass: starting")
        lp = lowpass_run.run_lowpass(
            plane0, params,
            figures_dir=_figures_dir_for(save_folder, "lowpass"),
            progress_cb=_emit,
        )
        stages["lowpass"].update(
            status="ok", duration_s=time.time() - t0)
        figures["lowpass"] = lp.get("figures", [])
        _emit("lowpass: done")
    except Exception as e:
        stages["lowpass"].update(
            status="failed", duration_s=time.time() - t0,
            error=f"{e}\n{traceback.format_exc()}")
        _emit(f"lowpass: FAILED -- {e}")
        overall_status = "partial"
        return {
            "recording_id": rec_id,
            "save_folder": str(save_folder),
            "status": overall_status,
            "plane0": str(plane0),
            "stages": stages,
            "figures": figures,
        }

    # ----- Stage 3: event detection (Tab 5) -------------------------
    t0 = time.time()
    event_data: Optional[dict] = None
    try:
        _emit("event_detection: starting")
        event_data = event_detection_run.run_event_detection(
            plane0, params,
            figures_dir=_figures_dir_for(save_folder, "event_detection"),
            write_summary=True,
            progress_cb=_emit,
        )
        stages["event_detection"].update(
            status="ok", duration_s=time.time() - t0)
        figures["event_detection"] = event_data.get("figures", [])
        _emit(f"event_detection: done "
              f"(events={len(event_data.get('event_windows') or [])})")
    except Exception as e:
        stages["event_detection"].update(
            status="failed", duration_s=time.time() - t0,
            error=f"{e}\n{traceback.format_exc()}")
        _emit(f"event_detection: FAILED -- {e}")
        overall_status = "partial"

    # ----- Stage 4: clustering (Tab 6) ------------------------------
    t0 = time.time()
    try:
        _emit("clustering: starting")
        clustering.run_clustering(
            plane0, params,
            figures_dir=_figures_dir_for(save_folder, "clustering"),
            write_summary=True,
        )
        stages["clustering"].update(
            status="ok", duration_s=time.time() - t0)
        _emit("clustering: done")
    except Exception as e:
        stages["clustering"].update(
            status="failed", duration_s=time.time() - t0,
            error=f"{e}\n{traceback.format_exc()}")
        _emit(f"clustering: FAILED -- {e}")
        overall_status = "partial"

    # ----- Stage 5: cross-correlation (Tab 7) -----------------------
    t0 = time.time()
    if stages["clustering"]["status"] == "ok":
        try:
            _emit("crosscorrelation: starting")
            xc_params = dict(params)
            xc_params.setdefault("fps", event_data.get("fps")
                                 if event_data is not None else None)
            event_windows = (event_data.get("event_windows")
                             if event_data is not None else None)
            crosscorrelation.run_crosscorrelation(
                plane0, xc_params,
                event_windows=event_windows,
                figures_dir=_figures_dir_for(save_folder,
                                             "crosscorrelation"),
                progress_cb=lambda done, total: _emit(
                    f"crosscorrelation: {done}/{total}"),
            )
            stages["crosscorrelation"].update(
                status="ok", duration_s=time.time() - t0)
            _emit("crosscorrelation: done")
        except Exception as e:
            stages["crosscorrelation"].update(
                status="failed", duration_s=time.time() - t0,
                error=f"{e}\n{traceback.format_exc()}")
            _emit(f"crosscorrelation: FAILED -- {e}")
            overall_status = "partial"
    else:
        stages["crosscorrelation"].update(
            status="skipped",
            error="clustering did not complete; xcorr needs cluster outputs")
        _emit("crosscorrelation: skipped (clustering missing)")

    # ----- Stage 6: spatial propagation figures (Tab 8) -------------
    t0 = time.time()
    if event_data is not None:
        try:
            _emit("spatial_propagation: starting")
            figs = spatial.render_spatial_event_figures(
                plane0, event_data,
                figures_dir=_figures_dir_for(save_folder,
                                             "spatial_propagation"),
            )
            stages["spatial_propagation"].update(
                status="ok", duration_s=time.time() - t0)
            figures["spatial_propagation"] = figs
            _emit(f"spatial_propagation: done ({len(figs)} figures)")
        except Exception as e:
            stages["spatial_propagation"].update(
                status="failed", duration_s=time.time() - t0,
                error=f"{e}\n{traceback.format_exc()}")
            _emit(f"spatial_propagation: FAILED -- {e}")
            overall_status = "partial"
    else:
        stages["spatial_propagation"].update(
            status="skipped",
            error="event detection did not produce event_data")

    return {
        "recording_id": rec_id,
        "save_folder": str(save_folder),
        "status": overall_status,
        "plane0": str(plane0),
        "stages": stages,
        "figures": figures,
    }
