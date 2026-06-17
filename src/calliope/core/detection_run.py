"""Headless Suite2p detection pipeline (Tab 3) for batch execution.

Why this module exists
----------------------
Tab 3's ``_run_pipeline`` chains five steps -- sparse_plus_cellpose
detection, pix_to_um stamping, dF/F + low-pass, cell-filter
prediction, filtered-dF/F slicing -- with all logic living on the
``Suite2pTab`` class. Tab 0's batch runner needs the same pipeline
without a Tk window, so we lift the chain into module-level
functions.

Public surface
--------------
``run_detection(tiff_folder, save_folder, params, *, ckpt_path=None,
                rec_id=None, baseline_mode='first_n', baseline_min=2.0,
                figures_dir=None, progress_cb=None)``
    End-to-end Tab 3 pipeline. Returns ``{plane0, output_files}``.
``compute_dff_memmaps(plane0, params, ...)``
    The dF/F + low-pass + derivative loop.
``compute_filtered_dff(plane0)``
    Slice the cell-filter mask out of the per-ROI dF/F into the
    "filtered" memmap consumed by Tabs 4-8.
``predict_cell_filter(plane0, ckpt_path, rec_id)``
    Run the cell-filter CNN. Optional step.
``stamp_pix_to_um(plane0, params)``
    Resolve ``params['scope_zoom']`` / ``params['um_per_pixel']`` and
    write ``ops['pix_to_um']`` for downstream tabs.
"""

from __future__ import annotations

import gc
import os
import shutil
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import utils
from . import scale as scale_mod
from . import sparse_plus_cellpose as spc
from . import preprocessing as _pre


# Intermediate-binary cleanup. ``sparse_plus_cellpose.run`` leaves two
# ~13 GB ``data.bin`` files behind per recording (one under
# ``_shared_reg/`` from the register-only pass, one under
# ``sparsery_pass/`` from the detection-only pass). ``merge_and_extract``
# has already folded everything into ``final/`` by the time the
# pipeline returns -- the intermediate binaries are pure dead weight.
# Without an explicit prune, an agent that drives detection over many
# recordings (whether through this module or Tab 3's GUI) accumulates
# ~26 GB/recording until the save drive fills (observed at recording
# 13 in the 2026-05-12 batch).
#
# This mirrors ``BatchTab._prune_scratch_tree`` but runs unconditionally
# at the tail of every detection -- including direct-run agents and
# headless drivers that don't route through BatchTab's row state machine.
_PRUNE_FILE_PATTERNS: tuple[str, ...] = (
    "**/data_raw.bin",
    "**/data_raw_chan2.bin",
    "**/*.tmp",
    "**/*.lock",
)
_PRUNE_INTERMEDIATE_DIRS: tuple[str, ...] = (
    "sparsery_pass",
    "cellpose_pass",
)


def _dir_size(p: Path) -> int:
    total = 0
    try:
        for dp, _dn, fn in os.walk(str(p)):
            for f in fn:
                try:
                    total += (Path(dp) / f).stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def prune_detection_intermediates(
    save_folder,
    *,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> tuple[int, int]:
    """Delete the redundant detection binaries left under
    ``<save_folder>/detection/`` after the final extraction has been
    folded into ``final/``.

    Drops ``sparsery_pass/`` + ``cellpose_pass/`` whole, drops
    ``_shared_reg/suite2p/plane0/data.bin`` (~13 GB; regeneratable
    from the shifted TIFF in ~5 min for a manual re-detect), and
    drops stray ``data_raw*.bin`` / ``*.tmp`` / ``*.lock`` artefacts.
    Keeps ``_shared_reg/ops.npy`` + the rest of the registration
    metadata for audit, and keeps everything under ``final/``.

    Returns ``(n_paths, bytes_freed)`` for logging. Errors on
    individual paths are caught + reported via ``progress_cb`` so a
    single file lock doesn't abort the whole cleanup.
    """
    det = Path(save_folder) / "detection"
    if not det.is_dir():
        return 0, 0
    # Release any open suite2p memmap handles so Windows lets unlink
    # / rmtree succeed on the first pass.
    gc.collect()

    n_paths = 0
    bytes_freed = 0

    for pattern in _PRUNE_FILE_PATTERNS:
        for f in det.glob(pattern):
            try:
                if f.is_file() or f.is_symlink():
                    sz = f.stat().st_size if f.is_file() else 0
                    f.unlink()
                    n_paths += 1
                    bytes_freed += sz
            except OSError:
                continue

    for sub in _PRUNE_INTERMEDIATE_DIRS:
        p = det / sub
        if not p.is_dir():
            continue
        sz = _dir_size(p)
        shutil.rmtree(p, ignore_errors=True)
        if not p.exists():
            n_paths += 1
            bytes_freed += sz

    shared_bin = det / "_shared_reg" / "suite2p" / "plane0" / "data.bin"
    if shared_bin.is_file():
        try:
            sz = shared_bin.stat().st_size
            shared_bin.unlink()
            n_paths += 1
            bytes_freed += sz
        except OSError as e:
            if progress_cb is not None:
                progress_cb(
                    f"[detection] cleanup: could not remove "
                    f"{shared_bin}: {e}")

    if progress_cb is not None and n_paths > 0:
        progress_cb(
            f"[detection] pruned {n_paths} intermediate path"
            f"{'' if n_paths == 1 else 's'} "
            f"({bytes_freed / 1e9:.2f} GB freed)")
    return n_paths, bytes_freed


def archive_recording_post_detection(
    rec_root,
    *,
    raw_paths: Optional[list] = None,
    dest_root=None,
    plane0=None,
    compress_raw: bool = True,
    delete_shifted: bool = True,
    delete_final_bin: bool = True,
    delete_external_raw: bool = False,
    level: int = 19,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """Compress raw TIFFs into the recording folder and drop the now-
    redundant shifted TIFFs + ``final/.../data.bin``.

    Order of operations (each step gated by its flag):

    1. Compress every raw listed in ``_calliope_raw_paths.json``
       (written by ``run_preprocess``) into the recording folder
       root, as ``<rec>/<raw.name>.tif``. Verifies byte-equality
       before swap. If the raw already lives in the recording folder
       (Path equal to dst), the compression is in-place.
    2. Delete the top-level ``*shifted_*.tif`` files. Only runs if
       step 1 succeeded for at least one raw (or was disabled).
    3. Delete ``detection/final/suite2p/plane0/data.bin``. (The
       ``_shared_reg/data.bin`` hardlink twin is already removed by
       ``prune_detection_intermediates``; both sides must be gone to
       actually free bytes.)
    4. Optionally delete the user's external raw at its original path
       (off by default; destructive).

    Returns a dict with counters / bytes-freed for logging:
        {compressed: int, shifted_deleted: int, bin_deleted: bool,
         external_raw_deleted: int, bytes_freed: int,
         compressed_bytes_in: int, compressed_bytes_out: int}

    Errors during compression are caught -- on failure of any single
    raw, the archive step aborts BEFORE any deletions so the
    recording stays in a re-runnable state.
    """
    rec_root = Path(rec_root)
    summary = {
        "compressed": 0,
        "shifted_deleted": 0,
        "bin_deleted": False,
        "external_raw_deleted": 0,
        "bytes_freed": 0,
        "compressed_bytes_in": 0,
        "compressed_bytes_out": 0,
    }

    def _log(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    if not rec_root.is_dir():
        return summary

    # ----- Step 1: compress raws into recording folder ---------------------
    compression_ok = True
    compressed_dst_for_src: dict[Path, Path] = {}

    dst_root = Path(dest_root) if dest_root is not None else rec_root
    if compress_raw:
        if raw_paths:
            # Caller passed the raw source paths directly -- the robust
            # path. Avoids re-reading ``_calliope_raw_paths.json`` from a
            # directory that may not hold it: the batch sidecar lives in
            # the recording folder, but the archive was historically
            # called against that folder's PARENT, so the lookup found
            # nothing and silently skipped compression on every run.
            raw_sources = [Path(p) for p in raw_paths]
            _log(f"[archive] {len(raw_sources)} raw path(s) from caller")
        else:
            raw_sources = _pre.read_raw_paths_sidecar(rec_root)
        if not raw_sources:
            # Sidecar absent. This is expected when detection runs on a
            # recording that wasn't preprocessed via CalLIOPE (e.g., the
            # user placed shifted TIFFs in the folder manually, or it
            # was preprocessed by an older version before the sidecar
            # was added). Fall back to: any *.tif/*.tiff in the
            # recording folder that doesn't have ``shifted_`` in its
            # name is a candidate raw to compress in place.
            fallback = [
                p for p in sorted(rec_root.iterdir())
                if p.is_file()
                and p.suffix.lower() in (".tif", ".tiff")
                and "shifted_" not in p.name
            ]
            if fallback:
                _log(
                    f"[archive] no raw-paths sidecar; using "
                    f"{len(fallback)} TIFF(s) found in the recording "
                    f"folder as raw sources"
                )
                raw_sources = fallback
            else:
                _log(
                    "[archive] raw compression skipped (no sidecar and "
                    "no raw TIFFs in recording folder -- run preprocess "
                    "via Tab 0/Batch to enable). Shifted TIFFs and "
                    "data.bin will still be cleaned up."
                )
        if raw_sources:
            for src in raw_sources:
                src = Path(src)
                if not src.exists():
                    _log(
                        f"[archive] raw missing on disk, skipping: {src}"
                    )
                    compression_ok = False
                    continue
                dst = dst_root / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    in_bytes, out_bytes = _pre.compress_raw_tiff(
                        src, dst, level=level, progress_cb=progress_cb,
                    )
                    summary["compressed"] += 1
                    summary["compressed_bytes_in"] += in_bytes
                    summary["compressed_bytes_out"] += out_bytes
                    compressed_dst_for_src[src.resolve()] = dst.resolve()
                except Exception as e:
                    _log(
                        f"[archive] compression FAILED for {src.name}: {e}"
                    )
                    compression_ok = False
                    break

    if not compression_ok:
        _log("[archive] aborting deletions because compression had errors")
        return summary

    # Guard: if compression was requested but produced nothing (raw
    # sources couldn't be located), do NOT delete the shifted TIFFs --
    # that would leave the recording folder with neither a raw nor a
    # compressed copy. Only an explicit compress_raw=False, or a
    # successful compress, may clear them.
    if compress_raw and summary["compressed"] == 0:
        _log("[archive] WARNING: 0 raws compressed -- keeping shifted "
             "TIFFs and skipping cleanup (check raw paths / sidecar).")
        return summary

    # ----- Step 2: delete top-level shifted TIFFs --------------------------
    if delete_shifted:
        for p in sorted(rec_root.iterdir()):
            if (
                p.is_file()
                and p.suffix.lower() in (".tif", ".tiff")
                and "shifted_" in p.name
            ):
                try:
                    sz = p.stat().st_size
                    p.unlink()
                    summary["shifted_deleted"] += 1
                    summary["bytes_freed"] += sz
                except OSError as e:
                    _log(
                        f"[archive] could not unlink shifted "
                        f"{p.name}: {e}"
                    )

    # ----- Step 3: delete final/.../data.bin -------------------------------
    if delete_final_bin:
        # Prefer the real plane0 when the caller passed it: layouts differ
        # (the GUI nests detection under ``<rec>/detection/``; the headless
        # pipeline writes ``<rec>/final/`` directly), so the reconstructed
        # path is wrong for one of them. Fall back to the legacy
        # reconstruction when plane0 wasn't supplied.
        if plane0 is not None:
            final_bin = Path(plane0) / "data.bin"
        else:
            final_bin = (
                rec_root / "detection" / "final" / "suite2p" / "plane0"
                / "data.bin"
            )
        if final_bin.is_file():
            # Windows holds a lock on the file until every memmap /
            # BinaryFile / torch tensor referencing it has been GC'd.
            # suite2p's extraction wrapper opens data.bin via a context
            # manager that releases at scope exit, but PyTorch sparse
            # tensors built off the same buffer survive until the
            # generation finalises. gc.collect() + a short retry loop
            # gives those refs a chance to drop before we unlink.
            sz = final_bin.stat().st_size
            last_err: OSError | None = None
            for attempt in range(5):
                if attempt > 0:
                    gc.collect()
                    time.sleep(0.5)
                try:
                    final_bin.unlink()
                    summary["bin_deleted"] = True
                    summary["bytes_freed"] += sz
                    last_err = None
                    break
                except OSError as e:
                    last_err = e
            if last_err is not None:
                _log(
                    f"[archive] could not unlink final data.bin "
                    f"after 5 attempts: {last_err}"
                )

    # ----- Step 4: optionally delete external raw originals ----------------
    if delete_external_raw and compressed_dst_for_src:
        for src, dst in compressed_dst_for_src.items():
            try:
                if src == dst:
                    # In-place compress; nothing external to delete.
                    continue
                if not dst.is_file():
                    continue
                if not src.is_file():
                    continue
                sz = src.stat().st_size
                src.unlink()
                summary["external_raw_deleted"] += 1
                summary["bytes_freed"] += sz
            except OSError as e:
                _log(
                    f"[archive] could not delete external raw "
                    f"{src}: {e}"
                )

    if progress_cb is not None:
        _log(
            f"[archive] compressed={summary['compressed']} "
            f"shifted_deleted={summary['shifted_deleted']} "
            f"bin_deleted={summary['bin_deleted']} "
            f"external_raw_deleted={summary['external_raw_deleted']} "
            f"freed={summary['bytes_freed'] / 1e9:.2f} GB"
        )
    return summary


def stamp_pix_to_um(plane0: Path, params: dict) -> Optional[float]:
    """Resolve the µm/pixel calibration and write it to
    ``calliope_calibration.npy`` next to the plane.

    Returns the resolved scalar (or None when both inputs were 0 or
    the resolver could not produce a value).
    """
    zoom = params.get("scope_zoom", 0.0) or None
    um_px = params.get("um_per_pixel", 0.0) or None
    fov_um = params.get("fov_um_reference", scale_mod.FOV_UM_REFERENCE) \
        or scale_mod.FOV_UM_REFERENCE
    if zoom is None and um_px is None:
        return None
    view = utils.load_plane_view(plane0)
    if not view:
        return None
    try:
        resolved = scale_mod.resolve_pix_to_um(
            view, zoom=zoom, um_per_pixel=um_px, fov_um=fov_um,
        )
    except (TypeError, ValueError):
        return None
    if resolved is None:
        return None
    utils.save_pix_to_um(plane0, float(resolved))
    return float(resolved)


def _maybe_run_dff_gpu(plane0: Path, params: dict, *,
                      baseline_mode: str, baseline_min: float,
                      fps: float, cutoff_hz: float, sg_win_ms: int,
                      sg_poly: int, r: float) -> bool:
    """GPU dF/F via ``analyze_output_gpu``. Returns True on success.

    Mirrors ``Suite2pTab._maybe_run_dff_gpu``: only kicks in when
    ``use_gpu_dff`` is set, the baseline is ``first_n``, and CuPy is
    importable.
    """
    if not bool(params.get("use_gpu_dff", True)):
        return False
    if baseline_mode != "first_n":
        return False
    try:
        from calliope.core import analyze_output_gpu as gpu
    except Exception:
        return False
    if not getattr(gpu, "_CUPY_OK", False):
        return False

    F = np.load(plane0 / "F.npy", allow_pickle=False)
    Fneu = np.load(plane0 / "Fneu.npy", allow_pickle=False)
    roi_chunk = int(params.get("gpu_roi_chunk", 0)) or None
    try:
        gpu.process_suite2p_traces_gpu(
            F, Fneu, fps=fps, r=r,
            baseline_sec=float(baseline_min) * 60.0,
            cutoff_hz=cutoff_hz, sg_win_ms=sg_win_ms, sg_poly=sg_poly,
            out_dir=str(plane0), prefix="r0p7_",
            roi_chunk=roi_chunk,
        )
        return True
    except Exception:
        return False


def compute_dff_memmaps(
    plane0: Path,
    params: dict,
    *,
    baseline_mode: str = "first_n",
    baseline_min: float = 2.0,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """Compute neuropil-corrected dF/F + low-pass + SG derivative for
    every Suite2p ROI. Writes::

        plane0/r0p7_dff.memmap.float32          (T, N)
        plane0/r0p7_dff_lowpass.memmap.float32  (T, N)
        plane0/r0p7_dff_dt.memmap.float32       (T, N)

    Tries the GPU path first when ``params.use_gpu_dff`` is enabled
    and the baseline is ``first_n``. Falls back to per-ROI CPU.
    """
    plane0 = Path(plane0)
    F = np.load(plane0 / "F.npy", allow_pickle=False)
    Fneu = np.load(plane0 / "Fneu.npy", allow_pickle=False)
    if F.shape != Fneu.shape or F.ndim != 2:
        raise ValueError(
            f"unexpected F/Fneu shapes: {F.shape}, {Fneu.shape}")
    F = np.asarray(F, dtype=np.float32, order="C")
    Fneu = np.asarray(Fneu, dtype=np.float32, order="C")
    N, T = F.shape

    fps_override = float(params.get("fps_override", 0.0) or 0.0)
    fps = (fps_override if fps_override > 0
           else float(utils.get_fps_from_notes(str(plane0))))

    cutoff_hz = float(params.get("default_lowpass_hz", 1.0))
    sg_win_ms = int(params.get("default_sg_win_ms", 333))
    sg_poly = int(params.get("default_sg_poly", 2))
    r = float(params.get("neuropil_coef", 0.7))
    perc = int(params.get("perc", 10))
    win_sec = float(params.get("win_sec", 45.0))

    # Record the dF/F + low-pass provenance in the meta.json sidecar before
    # either the GPU or CPU branch returns, so the NWB exporter and the
    # cross-correlation tooltips can read back which parameters produced
    # these memmaps. A metadata write must never break the pipeline.
    try:
        from . import recording_meta
        recording_meta.update_meta(
            plane0,
            dff={
                "baseline_mode": baseline_mode,
                "baseline_percentile": perc,
                "baseline_window_sec": win_sec,
                "baseline_min_sec": float(baseline_min),
                "fps": fps,
            },
            lowpass={
                "cutoff_hz": cutoff_hz,
                "filter_kind": "iir_butter",
                "order": 2,
                "group_delay_samples": None,
            },
            extraction={"neuropil_coef": r},
        )
    except Exception as _meta_exc:
        print(f"[meta] could not stamp dF/F params: {_meta_exc}")

    if _maybe_run_dff_gpu(
            plane0, params, baseline_mode=baseline_mode,
            baseline_min=baseline_min, fps=fps,
            cutoff_hz=cutoff_hz, sg_win_ms=sg_win_ms,
            sg_poly=sg_poly, r=r):
        return {"fps": fps, "T": T, "N": N, "path": "gpu"}

    shape = (T, N)
    dff_path = plane0 / "r0p7_dff.memmap.float32"
    lp_path = plane0 / "r0p7_dff_lowpass.memmap.float32"
    dt_path = plane0 / "r0p7_dff_dt.memmap.float32"
    dff_mm = np.memmap(str(dff_path), mode="w+", dtype="float32", shape=shape)
    lp_mm = np.memmap(str(lp_path), mode="w+", dtype="float32", shape=shape)
    dt_mm = np.memmap(str(dt_path), mode="w+", dtype="float32", shape=shape)

    # 2D-vectorized chunk size: caps transient memory at ~3 * CHUNK * T
    # float32 buffers regardless of how generous the outer ``batch`` is.
    DFF_CHUNK = 256
    batch = max(8, utils.change_batch_according_to_free_ram() * 20)
    sos = None
    t0 = time.time()
    from scipy.signal import savgol_filter
    for i0 in range(0, N, batch):
        i1 = min(N, i0 + batch)
        for j0 in range(i0, i1, DFF_CHUNK):
            j1 = min(i1, j0 + DFF_CHUNK)
            # Batched neuropil subtraction (F / Fneu are already
            # float32, so this stays float32 without an extra astype).
            traces_chunk = F[j0:j1] - r * Fneu[j0:j1]
            dff_chunk = np.empty_like(traces_chunk)
            lp_chunk = np.empty_like(traces_chunk)
            for k in range(j1 - j0):
                trace = traces_chunk[k]
                if baseline_mode == "first_n":
                    dff = utils.first_n_min_df_over_f_1d(
                        trace, baseline_min=baseline_min,
                        perc=perc, fps=fps)
                else:
                    dff = utils.robust_df_over_f_1d(
                        trace, win_sec=win_sec, perc=perc, fps=fps)
                lp, _, sos = utils.lowpass_causal_1d(
                    dff, fps=fps, cutoff_hz=cutoff_hz,
                    order=2, zi=None, sos=sos)
                dff_chunk[k] = dff
                lp_chunk[k] = lp
            # Batched Savitzky-Golay derivative along the time axis:
            # one savgol_filter call per chunk replaces CHUNK per-ROI
            # calls. ``savgol_filter`` with ``axis=1`` is mathematically
            # identical to per-row, just amortizes the Python overhead.
            T_len = lp_chunk.shape[1]
            win = max(3, int((sg_win_ms / 1000.0) * fps) | 1)
            if win >= T_len:
                win = max(3, (T_len - (1 - T_len % 2)))
            if win < 3 or T_len < 3:
                dt_chunk = np.empty_like(lp_chunk)
                dt_chunk[:, 0] = 0.0
                dt_chunk[:, 1:] = (lp_chunk[:, 1:] - lp_chunk[:, :-1]) * fps
            else:
                dt_chunk = savgol_filter(
                    lp_chunk, window_length=win, polyorder=sg_poly,
                    deriv=1, delta=1.0 / fps, axis=1,
                ).astype(np.float32)
            # Slab writes (one contiguous transposed block per buffer
            # per chunk) replace CHUNK column-by-column updates.
            dff_mm[:, j0:j1] = dff_chunk.T
            lp_mm[:, j0:j1] = lp_chunk.T
            dt_mm[:, j0:j1] = dt_chunk.T
        dff_mm.flush(); lp_mm.flush(); dt_mm.flush()
        if progress_cb is not None:
            progress_cb(
                f"dF/F batch {i0}-{i1 - 1}/{N - 1} "
                f"({time.time() - t0:.1f}s)")
    del dff_mm, lp_mm, dt_mm
    return {"fps": fps, "T": T, "N": N, "path": "cpu"}


def predict_cell_filter(plane0: Path, ckpt_path: str, rec_id: str) -> None:
    """Run the cell-filter CNN. Writes ``predicted_cell_mask.npy`` and
    ``predicted_cell_prob.npy`` into ``plane0``.
    """
    import gc
    import torch
    from .cellfilter.model import CellFilter
    from .cellfilter.predict import predict_recording

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = CellFilter().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    try:
        predict_recording(rec_id, model, device, plane0=plane0)
    finally:
        # Drop model + checkpoint before flushing the allocator so the
        # weights actually return to the driver. Without this, every
        # recording's predict step adds another model copy worth of
        # VRAM to torch's cache, which suite2p's next run inherits.
        del model, ckpt
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _load_keep_mask(plane0: Path, n_total: int) -> np.ndarray:
    """predicted_cell_mask -> iscell -> all-ones (current curation).

    Thin delegate to :func:`utils.resolve_live_mask`, the single
    writer-facing keep-mask resolver. Kept as a module-local name
    because three call sites here reference it.
    """
    return utils.resolve_live_mask(plane0, n_total)


def compute_filtered_dff(plane0: Path) -> Optional[Path]:
    """Slice ``r0p7_dff.memmap.float32`` down to ROIs that pass the
    cell-filter mask and write ``r0p7_filtered_dff.memmap.float32``
    plus the legacy ``r0p7_cell_mask_bool.npy`` keep-mask file.
    Returns the filtered memmap path (or None if no ROIs survive).
    """
    plane0 = Path(plane0)
    F = np.load(plane0 / "F.npy", mmap_mode="r")
    N_total, T = F.shape

    mask = _load_keep_mask(plane0, N_total)
    N_kept = int(mask.sum())
    if N_kept == 0:
        return None

    dff_path = plane0 / "r0p7_dff.memmap.float32"
    filtered_path = plane0 / "r0p7_filtered_dff.memmap.float32"
    dff = np.memmap(str(dff_path), dtype="float32", mode="r",
                    shape=(T, N_total))
    filt = np.memmap(str(filtered_path), dtype="float32", mode="w+",
                     shape=(T, N_kept))
    chunk = max(1, min(T, utils.DFF_TIME_CHUNK))
    for t0 in range(0, T, chunk):
        t1 = min(T, t0 + chunk)
        filt[t0:t1, :] = dff[t0:t1, :][:, mask]
    filt.flush()
    del filt, dff
    np.save(plane0 / "r0p7_cell_mask_bool.npy", mask)
    return filtered_path


def _build_sparsery_ops(params: dict) -> dict:
    """Mirror ``Suite2pTab._run_pipeline`` Sparsery dict assembly."""
    return {
        "high_pass":         params.get("high_pass", 100),
        "preclassify":       params.get("preclassify", 0.0),
        "smooth_sigma":      params.get("smooth_sigma", 1.0),
        "sparse_mode":       True,
        "spatial_scale":     params.get("spatial_scale", 0),
        "threshold_scaling": params.get("threshold_scaling", 0.85),
        "max_iterations":    params.get("max_iterations", 1500),
    }


def _build_cellpose_cfg(params: dict) -> dict:
    return {
        "cellpose_model_type":
            params.get("cellpose_model_type", "cyto2"),
        "cellpose_diameter":
            params.get("cellpose_diameter", 0),
        "cellpose_flow_threshold":
            params.get("cellpose_flow_threshold", 0.8),
        "cellpose_cellprob_threshold":
            params.get("cellpose_cellprob_threshold", -1.0),
        "cellpose_channel_input": "meanImg",
    }


def archive_offload(save_folder_str: str, params: dict, *,
                    raw_paths=None, plane0_str: Optional[str] = None,
                    progress_q=None) -> dict:
    """Picklable :mod:`calliope.core.offload` target for Tab 3's
    post-detection archive (prune intermediates + compress raw TIFFs).

    Runs in a child process so the slow zstd compression doesn't freeze
    the GUI. Safe to offload (unlike detection itself, this step uses no
    suite2p multiprocessing). Mirrors the inline archive Tab 3 used to do
    in its detection worker: prune always, then compress/clean per the
    ``*_post_detection`` params.
    """
    from .offload import make_progress_cb
    cb = make_progress_cb(progress_q)
    save_folder = Path(save_folder_str)
    prune_detection_intermediates(save_folder, progress_cb=cb)
    if not bool(params.get("archive_post_detection", True)):
        return {"archived": False}
    archive_recording_post_detection(
        save_folder.parent,
        raw_paths=raw_paths,
        plane0=(Path(plane0_str) if plane0_str else None),
        compress_raw=bool(params.get("compress_raw_post_detection", True)),
        delete_shifted=bool(params.get("delete_shifted_post_detection", True)),
        delete_final_bin=bool(
            params.get("delete_final_data_bin_post_detection", True)),
        delete_external_raw=bool(
            params.get("delete_external_raw_after_archive", False)),
        level=int(params.get("raw_compression_level", 19)),
        progress_cb=cb,
    )
    return {"archived": True}


def _render_detection_panels(plane0: Path, figures_dir: Path, *,
                             kept_overlay: bool = True) -> list[str]:
    """Save the same detection-overlay panels Tab 3's ``_draw_panels``
    produces (mean-image + ROI overlays) as PNGs. Falls back gracefully
    if any background image is missing.

    ``kept_overlay`` mirrors Tab 3's panel-3 "ROI overlay" toggle: when
    False, the ``kept_rois`` figure is the clean background image with no
    ROI overlay (the all-detected ``all_rois`` panel always keeps its
    overlay). Defaults True so batch runs keep the kept-ROI overlay.
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    view = utils.load_plane_view(plane0)
    stat = np.load(plane0 / "stat.npy", allow_pickle=True)
    n_total = len(stat)
    keep = _load_keep_mask(plane0, n_total)

    bg = None
    for key in ("meanImg", "meanImgE", "max_proj", "Vcorr"):
        img = view.get(key)
        if isinstance(img, np.ndarray) and img.ndim == 2:
            bg = np.asarray(img, dtype=np.float32)
            break
    if bg is None:
        return written

    # Micrometre scale bar, drawn only when the recording carries a pixel
    # calibration (stamp_pix_to_um ran with a zoom / µm-per-pixel set). The
    # panels keep ``axis('off')`` and pixel data coords, so the bar is sized
    # in pixels (length_um / pix_to_um); the ROI scatter needs no rescaling.
    from .figscale import add_scale_bar
    pix_to_um = view.get("pix_to_um")
    Ly, Lx = bg.shape

    for label, mask, color in (
            ("all_rois", np.ones(n_total, dtype=bool), "tab:cyan"),
            ("kept_rois", keep, "tab:orange"),
    ):
        # The kept panel honours the overlay toggle; all-detected always
        # overlays so the raw output stays visible.
        draw_overlay = (label == "all_rois") or kept_overlay
        fig = plt.Figure(figsize=(6, 6), tight_layout=True)
        ax = fig.add_subplot(111)
        lo, hi = np.percentile(
            bg, [utils.DISPLAY_CLIP_LOW_PCT, utils.DISPLAY_CLIP_HIGH_PCT])
        ax.imshow(bg, cmap="gray", vmin=lo, vmax=hi, aspect="equal")
        if draw_overlay:
            for i, s in enumerate(stat):
                if not mask[i]:
                    continue
                ys = np.asarray(s.get("ypix", []))
                xs = np.asarray(s.get("xpix", []))
                if ys.size == 0:
                    continue
                ax.scatter(xs, ys, s=0.4, c=color, alpha=0.5,
                           linewidths=0)
        suffix = "" if draw_overlay else "  (background only)"
        nice = ("Detected ROIs" if label == "all_rois"
                else "Kept ROIs · cell filter ≥ 0.5")
        ax.set_title(f"{nice}  (n={int(mask.sum())}){suffix}")
        ax.set_axis_off()
        add_scale_bar(ax, pix_to_um, axes_in_um=False,
                      span_um=(Lx * pix_to_um if pix_to_um else None),
                      color="white")
        out = figures_dir / f"{label}.png"
        fig.savefig(out, dpi=150)
        fig.savefig(out.with_suffix(".svg"))
        plt.close(fig)
        written.append(str(out))
    return written


def run_detection(
    tiff_folder: str,
    save_folder: str,
    params: dict,
    *,
    path_to_ops: Optional[str] = None,
    tau_override: Optional[float] = None,
    ckpt_path: Optional[str] = None,
    rec_id: Optional[str] = None,
    baseline_mode: str = "first_n",
    baseline_min: float = 2.0,
    figures_dir: Optional[Path] = None,
    raw_paths: Optional[list] = None,
    archive_root: Optional[Path] = None,
    dest_root: Optional[Path] = None,
    write_summary: bool = True,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """Run the full Tab 3 pipeline for one recording.

    Steps:
      1. ``sparse_plus_cellpose.run`` -> plane0.
      2. Stamp ``ops['pix_to_um']`` from ``params['scope_zoom']`` /
         ``params['um_per_pixel']``.
      3. ``compute_dff_memmaps`` -> r0p7_dff* memmaps.
      4. ``predict_cell_filter`` -> predicted_cell_mask.npy / prob.
         (skipped when ``ckpt_path`` is None or empty)
      5. ``compute_filtered_dff`` -> r0p7_filtered_dff.memmap.

    Returns ``{plane0: Path, fps: float, T: int, N: int,
    pix_to_um: float|None, figures: list[str]}``.
    """
    if progress_cb is not None:
        progress_cb("[detection] running sparse_plus_cellpose...")
    sparsery_ops = _build_sparsery_ops(params)
    cellpose_cfg = _build_cellpose_cfg(params)
    hard_cap = int(params.get("hard_cap", 60000))
    max_overlap = float(params.get("max_overlap", 0.3))
    # User's FPS override (0 = unset) so suite2p's deconvolution / spike
    # timing run at the real acquisition rate rather than the baked-in
    # fallback; the same value drives the dF/F fps in compute_dff_memmaps.
    fps_override = float(params.get("fps_override", 0.0) or 0.0)

    # Tier 2 #13: Cellpose-SAM second pass on Vcorr. Same off-by-default
    # pattern as the GUI tab -- callers opt in via params, the runner
    # silently skips if cellpose 4.x isn't installed.
    enable_sam = bool(params.get("enable_sam_vcorr_pass", False))
    sam_cfg = None
    if enable_sam:
        sam_cfg = {
            "cellpose_model_type": "cpsam",
            "cellpose_channel_input": "Vcorr",
            "cellpose_diameter": params.get("cellpose_diameter", 0),
            "cellpose_flow_threshold": float(
                params.get("sam_flow_threshold", 0.8)),
            "cellpose_cellprob_threshold": float(
                params.get("sam_cellprob_threshold", -1.0)),
        }

    plane0 = spc.run(
        tiff_folder=str(tiff_folder),
        save_folder=str(save_folder),
        path_to_ops=path_to_ops,
        sparsery_ops=sparsery_ops,
        cellpose_cfg=cellpose_cfg,
        hard_cap=hard_cap,
        max_overlap=max_overlap,
        tau_override=tau_override,
        fps_override=fps_override,
        enable_sam_vcorr_pass=enable_sam,
        sam_vcorr_cfg=sam_cfg,
        verbose=True,
    )
    plane0 = Path(plane0)

    pix_to_um = stamp_pix_to_um(plane0, params)

    if progress_cb is not None:
        progress_cb("[detection] computing dF/F + low-pass + derivative...")
    dff_info = compute_dff_memmaps(
        plane0, params,
        baseline_mode=baseline_mode, baseline_min=baseline_min,
        progress_cb=progress_cb,
    )

    if ckpt_path:
        if progress_cb is not None:
            progress_cb("[detection] running cell-filter prediction...")
        predict_cell_filter(plane0, ckpt_path, rec_id or plane0.parent.name)

    if progress_cb is not None:
        progress_cb("[detection] writing filtered dF/F...")
    compute_filtered_dff(plane0)

    # Stamp the GCaMP variant label into meta.json when the caller provided
    # it (the extraction stage records tau/neuropil coef but not the label,
    # which is a GUI/headless-supplied string). Best-effort.
    gcamp_label = params.get("gcamp_variant")
    if gcamp_label:
        try:
            from . import recording_meta
            recording_meta.update_meta(
                plane0, extraction={"gcamp_variant_label": str(gcamp_label)})
        except Exception as e:
            if progress_cb is not None:
                progress_cb(f"[detection] meta gcamp stamp failed: {e}")

    # Post-detection z-drift QC: PCA over ROI dF/F, flag PC1 step changes,
    # write zdrift_pc1.csv + zdrift_qc.png and stamp meta.json. Detection
    # only -- no frames are dropped. Best-effort.
    if bool(params.get("zdrift_qc", True)):
        try:
            from . import zdrift
            fps_override = float(params.get("fps_override", 0.0) or 0.0)
            _zfps = (fps_override if fps_override > 0
                     else float(utils.get_fps_from_notes(str(plane0))))
            zinfo = zdrift.detect_zdrift(
                plane0, fps=_zfps, progress_cb=progress_cb)
            if progress_cb is not None and zinfo.get("detected"):
                progress_cb(
                    f"[detection] z-drift QC: {zinfo['n_steps']} step(s) "
                    f"flagged -> zdrift_qc.png")
        except Exception as e:
            if progress_cb is not None:
                progress_cb(f"[detection] z-drift QC failed: {e}")

    prune_detection_intermediates(save_folder, progress_cb=progress_cb)

    # Post-detection archive: compress raws into the recording folder
    # (as <rec>/<raw.name>.tif), then drop the redundant shifted TIFFs
    # and the orphaned final/data.bin. Defaults match the design:
    # auto-on for compression + shifted/final-bin deletion, off for the
    # destructive external-raw delete. Each toggle is a per-job param
    # key so Tab 3 / batch tab can override.
    if bool(params.get("archive_post_detection", True)):
        try:
            # ``archive_root`` is the recording folder (where the shifted
            # TIFFs + raw-paths sidecar live). Batch passes it explicitly;
            # callers that omit it fall back to the legacy
            # ``save_folder.parent`` (correct for the GUI, whose
            # save_folder IS the detection/ subfolder). ``raw_paths`` lets
            # compression skip the sidecar lookup entirely -- the robust
            # source for the batch path, where the sidecar sits in the
            # recording folder, not its parent.
            archive_recording_post_detection(
                archive_root if archive_root is not None
                else Path(save_folder).parent,
                raw_paths=raw_paths,
                dest_root=dest_root,
                plane0=plane0,
                compress_raw=bool(
                    params.get("compress_raw_post_detection", True)),
                delete_shifted=bool(
                    params.get("delete_shifted_post_detection", True)),
                delete_final_bin=bool(
                    params.get("delete_final_data_bin_post_detection", True)),
                delete_external_raw=bool(
                    params.get("delete_external_raw_after_archive", False)),
                level=int(params.get("raw_compression_level", 19)),
                progress_cb=progress_cb,
            )
        except Exception as e:
            if progress_cb is not None:
                progress_cb(f"[archive] step failed: {e}")

    # Mirror the GUI Suite2p tab's ``_write_summary`` so a batch run
    # produces the same ``Recording`` + ``ROIs`` sheets in
    # ``calliope_summary.xlsx`` that an interactive Tab-3 run does.
    # Without this, batch-processed recordings had no ROIs sheet at all
    # and only got a Recording sheet later (if event detection ran).
    if write_summary:
        try:
            from . import summary_writer
            stat = np.load(plane0 / "stat.npy", allow_pickle=True)
            n_total = len(stat)
            keep = _load_keep_mask(plane0, n_total)
            prob_path = plane0 / "predicted_cell_prob.npy"
            p_cell = np.load(prob_path) if prob_path.exists() else None
            iscell_path = plane0 / "iscell.npy"
            iscell_arr = None
            if iscell_path.exists():
                ic = np.load(iscell_path)
                iscell_arr = (ic[:, 0] > 0) if ic.ndim == 2 else (ic > 0)
            summary_writer.update_recording_meta(
                plane0, recording_id=rec_id, fps=dff_info["fps"],
                T=dff_info["T"], N=n_total)
            summary_writer.write_rois_sheet(
                plane0, stat, p_cell=p_cell,
                predicted_mask=keep, iscell=iscell_arr)
            if progress_cb is not None:
                progress_cb("[detection] wrote Recording + ROIs sheets")
        except Exception as e:
            # A summary-sheet failure must never sink an otherwise
            # successful detection run -- surface it and carry on.
            if progress_cb is not None:
                progress_cb(f"[detection] summary sheet write failed: {e}")

    figs: list[str] = []
    if figures_dir is not None:
        try:
            # Batch keeps the kept-ROI overlay (the interactive panel-3
            # "ROI overlay" toggle is a Tab 3 display control only).
            figs = _render_detection_panels(plane0, Path(figures_dir))
        except Exception as e:
            if progress_cb is not None:
                progress_cb(f"[detection] figure render failed: {e}")

    return {
        "plane0": plane0,
        "fps": dff_info["fps"],
        "T": dff_info["T"],
        "N": dff_info["N"],
        "pix_to_um": pix_to_um,
        "figures": figs,
    }
