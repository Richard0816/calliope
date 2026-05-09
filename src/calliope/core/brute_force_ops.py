"""Brute-force ops sweep + Cellpose pass for Suite2p detection.

Two roles for this file
-----------------------
1. **Standalone parameter sweep** (legacy use case): given a grid of
   Suite2p ops combinations, register the movie once and run each
   detection config in turn. Persists a JSON ledger of every config
   tried + its outcome so you can resume after a crash; ranks the
   successful runs by a residual-image metric.

2. **Cellpose pass helper** (current GUI use case): exports
   ``run_cellpose_pass(...)`` which is what
   ``sparse_plus_cellpose.run`` calls to do its second-opinion
   Cellpose detection on the mean image. The standalone sweep above
   is no longer wired into the GUI -- but the Cellpose pass is.

Standalone-sweep flow
---------------------
  1. Register the TIFF stack ONCE (reuses adaptive_detection's
     shared-reg helper and hardlinks data.bin into each pass folder).
  2. For every combination in PARAMETER_GRID, run suite2p detection-
     only with ``max_rois_hard_cap=HARD_CAP`` so noise blowups abort
     mid-pass.
  3. Keep a persistent JSON ledger of every config tried + its
     outcome. Reruns skip configs already in the ledger (delete the
     file to retry).
  4. Delete the pass folder for any run that produced 0 ROIs or hit
     the hard cap. Only keep folders for runs that actually returned
     ROIs.
  5. Rank successful runs by residual-image metric (lower = more
     image content explained by ROIs).

NO blob augmentation. NO residual-seeded ROIs. NO classifier
filtering. Just raw sparsery output ranked by how much of the mean
image is left unmodeled after masking detected ROI pixels.

Usage (standalone):
    python brute_force_ops.py

Edit the CONFIGURATION block below to change the parameter grid, input
paths, or hard cap.
"""

from __future__ import annotations

import sys
import json
import shutil
import itertools
import time
import traceback
from pathlib import Path

import numpy as np

# Make sure we import the *worktree* copy of adaptive_detection, not any
# sibling version that might be on PYTHONPATH.
WORKTREE = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKTREE))

from .adaptive_detection import (
    AdaptiveConfig,
    load_base_settings,
    _get_or_create_shared_registration,
    run_one_pass,
    build_roi_pixel_mask,
    compute_residual_image,
    apply_settings_overrides,
)

# Cellpose is optional. If it's not installed we mark cellpose configs as
# 'skipped_no_cellpose' instead of crashing the whole sweep.
try:
    from cellpose import models as _cp_models  # type: ignore
    _CELLPOSE_AVAILABLE = True
    _CELLPOSE_IMPORT_ERROR = None
except Exception as _e:                                              # pragma: no cover
    _cp_models = None
    _CELLPOSE_AVAILABLE = False
    _CELLPOSE_IMPORT_ERROR = repr(_e)


# ============================================================================
# CONFIGURATION  — edit these
# ============================================================================

TIFF_FOLDER = r'D:\2024-11-20_00003'
SAVE_FOLDER = r'D:\adaptive_runs_bruteforce\2024-11-20_00003'
PATH_TO_OPS = r'updated_settings.npy'

# Abort any sparsery pass once it produces this many ROIs. The cap fires
# at the next 1000-ROI progress boundary (sparsery prints every 1000).
HARD_CAP = None

# Sparsery's outer-loop iteration limit. Kept generous so legitimate
# passes converge early and speckle-prone passes have a chance to hit
# the ROI cap before running out of iterations. Even at ~20 ROIs/iter
# (typical speckle rate), 1500 iterations comfortably reaches 20k.
MAX_ITERATIONS = 1500

# Parameter grids, split per detection backend. Each backend's grid is
# Cartesian-producted independently; the results are concatenated into
# the full config list, each entry tagged with its 'detection_backend'.
#
# Backends:
#   'sparsery' — Suite2p's built-in sparse-PCA detector (default).
#   'cellpose' — Cellpose deep-learning segmentation on the mean image.
#
# Comment out a whole grid to disable that backend for this sweep, or
# reduce an axis to a single value to shrink the sweep.

SPARSERY_GRID = {
    'threshold_scaling':
        [0.75, 0.85],
    'spatial_scale':
        [1, 2],
    'smooth_sigma':
        [1.0],
    'high_pass':
        [100],
    # Uncomment to widen the sweep further:
    'sparse_mode':  [True],
    # 'max_overlap':  [0.5, 0.75],
    # Downstream cell classifier filters specks, so don't pre-filter here.
    'preclassify':  [0.0],
}

# Cellpose-specific knobs. ``cellpose_diameter=0`` asks cellpose to
# auto-estimate; explicit values override. ``cellpose_model_type`` can
# also be 'cyto2' or 'nuclei'. flow_threshold and cellprob_threshold are
# the standard gating knobs — lower cellprob → more cells, lower flow → more cells.
CELLPOSE_GRID = {
    'cellpose_model_type':
        ['cyto', 'cyto2'],
    'cellpose_diameter':
        [0, 8, 12],          # 0 = auto-estimate
    'cellpose_flow_threshold':
        [0.4, 0.8],
    'cellpose_cellprob_threshold':
        [0.0, -1.0],
    'cellpose_channel_input':
        ['meanImg'],         # could add 'meanImgE' or 'max_proj' later
}

# Backward-compat alias; older code/comments may reference PARAMETER_GRID.
PARAMETER_GRID = SPARSERY_GRID

RESULTS_FILE_NAME = 'brute_force_results.json'
LEADERBOARD_FILE_NAME = 'brute_force_leaderboard.txt'

# Keep only this many successful configs on disk. After every new run,
# anything that falls outside the top-N (ranked by residual_mean, ascending)
# gets its folder deleted and is marked 'evicted' in the ledger. The ledger
# row (metrics, config) is retained so we never re-run a config we've
# already scored, and the top-N remains the single source of truth for
# "best runs worth inspecting".
TOP_N_KEEP = 50

# ============================================================================


# -------- helpers --------

_AXIS_SHORT = {
    'detection_backend': 'be',
    'threshold_scaling': 'thr',
    'spatial_scale': 'sc',
    'smooth_sigma': 'ss',
    'high_pass': 'hp',
    'sparse_mode': 'sp',
    'max_overlap': 'ov',
    'preclassify': 'pc',
    'cellpose_model_type': 'cpmodel',
    'cellpose_diameter': 'cpdiam',
    'cellpose_flow_threshold': 'cpflow',
    'cellpose_cellprob_threshold': 'cpprob',
    'cellpose_channel_input': 'cpimg',
}


def config_signature(cfg: dict) -> str:
    """Stable canonical key for a config dict (also the ledger key)."""
    parts = []
    for k in sorted(cfg):
        v = cfg[k]
        if isinstance(v, float):
            parts.append(f'{k}={v:.4f}')
        else:
            parts.append(f'{k}={v}')
    return '__'.join(parts)


def folder_name_for(cfg: dict) -> str:
    """Compact filesystem-safe folder name from a config dict."""
    bits = []
    for k in sorted(cfg):
        short = _AXIS_SHORT.get(k, k)
        v = cfg[k]
        if isinstance(v, float):
            bits.append(f'{short}{v:.2f}')
        elif isinstance(v, bool):
            bits.append(f'{short}{int(v)}')
        else:
            bits.append(f'{short}{v}')
    return '_'.join(bits)


def residual_metrics(stat, ops) -> dict:
    """Compute scalar metrics on the residual image (ROI pixels masked).

    Lower ``residual_mean`` / ``residual_p99`` / ``residual_energy`` means
    the ROI set covers more of the mean-image content → less unmodeled
    signal remaining.
    """
    Ly, Lx = int(ops['Ly']), int(ops['Lx'])
    roi_mask = build_roi_pixel_mask(stat, Ly, Lx)
    residual, _ = compute_residual_image(ops, roi_mask)

    r = residual.astype(np.float32)
    r = np.clip(r, 0.0, None)   # ignore negative excursions

    areas = np.array([len(s['ypix']) for s in stat], dtype=np.float32)

    return {
        'n_rois': int(len(stat)),
        'residual_mean':    float(np.mean(r)),
        'residual_p99':     float(np.quantile(r, 0.99)),
        'residual_p995':    float(np.quantile(r, 0.995)),
        'residual_energy':  float(np.sum(r.astype(np.float64) ** 2)),
        'median_area_px':   float(np.median(areas)),
        'mean_area_px':     float(np.mean(areas)),
    }


def load_ledger(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f'  ! could not read existing ledger ({e}); starting fresh')
        return {}


def save_ledger(path: Path, data: dict) -> None:
    # atomic-ish write so a crash mid-write doesn't corrupt the ledger
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(path)


def delete_folder(path: Path) -> None:
    try:
        if path.exists():
            shutil.rmtree(path)
    except Exception as e:
        print(f'  ! could not delete {path}: {e}')


def write_leaderboard(ledger: dict, outpath: Path) -> None:
    """Write a human-readable leaderboard, ranked by residual_mean ascending.

    Separates entries into three buckets:
      - kept (folder still on disk, status == 'ok')
      - evicted (scored, folder deleted for top-N retention, status == 'evicted')
      - failed (zero_rois / hard_capped / error)
    """
    kept = [(k, v) for k, v in ledger.items() if v.get('status') == 'ok']
    kept.sort(key=lambda kv: kv[1]['metrics']['residual_mean'])

    evicted = [(k, v) for k, v in ledger.items() if v.get('status') == 'evicted']
    evicted.sort(key=lambda kv: kv[1]['metrics']['residual_mean'])

    failed = [(k, v) for k, v in ledger.items()
              if v.get('status') not in ('ok', 'evicted')]

    lines = []
    lines.append(f'Brute-force leaderboard')
    lines.append(f'  kept (top-N, folder on disk): {len(kept)}')
    lines.append(f'  evicted (scored, folder deleted): {len(evicted)}')
    lines.append(f'  failed (zero/capped/error):       {len(failed)}')
    lines.append('')
    lines.append(f'Ranked by residual_mean (lower = more image content explained by ROIs).')
    lines.append(f'{"rank":>4}  {"residual_mean":>13}  {"residual_p99":>13}  '
                 f'{"n_rois":>7}  {"med_area":>8}  config')
    lines.append('-' * 120)
    for i, (sig, r) in enumerate(kept, 1):
        m = r['metrics']
        lines.append(
            f'{i:>4}  {m["residual_mean"]:>13.4f}  '
            f'{m["residual_p99"]:>13.4f}  {m["n_rois"]:>7}  '
            f'{m["median_area_px"]:>8.1f}  {sig}'
        )

    if evicted:
        lines.append('')
        lines.append(f'Evicted (scored but outside top-N; folders deleted):')
        for sig, r in evicted:
            m = r['metrics']
            lines.append(
                f'       {m["residual_mean"]:>13.4f}  '
                f'{m["residual_p99"]:>13.4f}  {m["n_rois"]:>7}  '
                f'{m["median_area_px"]:>8.1f}  {sig}'
            )

    if failed:
        lines.append('')
        lines.append(f'Failed configs ({len(failed)}):')
        for sig, r in failed:
            lines.append(f'  [{r.get("status","?")}]  n_rois={r.get("n_rois","?")}  {sig}')

    outpath.write_text('\n'.join(lines))


def enforce_top_n(ledger: dict, n: int, verbose: bool = True) -> int:
    """Keep only the top-N successful configs by residual_mean; evict rest.

    Evicted entries:
      - folder on disk is deleted
      - ledger entry is kept (so we don't re-run), status becomes 'evicted'
      - metrics are preserved for reference; ``folder`` is set to None

    Already-evicted entries are not re-processed. Returns the number of
    entries newly evicted on this call.
    """
    ok = [(k, v) for k, v in ledger.items() if v.get('status') == 'ok']
    if len(ok) <= n:
        return 0

    ok.sort(key=lambda kv: kv[1]['metrics']['residual_mean'])
    to_evict = ok[n:]
    for sig, rec in to_evict:
        folder = rec.get('folder')
        if folder:
            delete_folder(Path(folder))
        rec['status'] = 'evicted'
        rec['folder'] = None
        if verbose:
            print(f'  [evict] residual_mean='
                  f'{rec["metrics"]["residual_mean"]:.3f}  {sig}')
    return len(to_evict)


def enumerate_configs(grid: dict) -> list[dict]:
    axis_names = list(grid.keys())
    axis_values = [grid[k] for k in axis_names]
    return [dict(zip(axis_names, combo))
            for combo in itertools.product(*axis_values)]


def enumerate_all_configs(sparsery_grid: dict | None,
                          cellpose_grid: dict | None) -> list[dict]:
    """Produce the combined config list, each entry tagged with its backend.

    A grid set to None (or empty dict) disables that backend entirely.
    """
    out: list[dict] = []
    if sparsery_grid:
        for cfg in enumerate_configs(sparsery_grid):
            cfg = dict(cfg)
            cfg['detection_backend'] = 'sparsery'
            out.append(cfg)
    if cellpose_grid:
        for cfg in enumerate_configs(cellpose_grid):
            cfg = dict(cfg)
            cfg['detection_backend'] = 'cellpose'
            out.append(cfg)
    return out


# -------- cellpose backend --------

def _cellpose_masks_to_stat(masks: np.ndarray) -> list[dict]:
    """Convert a cellpose integer label mask into suite2p-style stat dicts.

    Cellpose returns masks as shape (Ly, Lx) with 0=background and 1..N
    as per-cell labels. We emit one dict per label containing the keys
    that ``build_roi_pixel_mask`` / ``residual_metrics`` expect:
    ``ypix``, ``xpix``, ``lam``, ``med``.
    """
    stat: list[dict] = []
    if masks is None:
        return stat
    labels = np.unique(masks)
    labels = labels[labels > 0]
    for lbl in labels:
        ys, xs = np.where(masks == lbl)
        if ys.size == 0:
            continue
        npix = int(ys.size)
        stat.append({
            'ypix': ys.astype(np.int32),
            'xpix': xs.astype(np.int32),
            'lam':  np.ones(npix, dtype=np.float32) / float(npix),
            'med':  [float(np.median(ys)), float(np.median(xs))],
            'npix': npix,
            'radius': float(np.sqrt(npix / np.pi)),
        })
    return stat


def run_cellpose_pass(save_dir: Path, shared_plane0: Path, cfg: dict,
                      verbose: bool = True):
    """Run Cellpose on the mean image from the shared registration.

    Why we do this on the mean image, not the movie
    -----------------------------------------------
    Cellpose is a 2-D segmentation model -- it doesn't know about
    time. We feed it the recording's mean image (a steady spatial
    snapshot of the cells) and treat the resulting masks as
    candidate ROIs. This catches cells that Suite2p's Sparsery
    detector misses because they're real but didn't fire enough
    during the recording.

    Steps
    -----
    1. Load the shared registration's ``meanImg`` (or whichever
       ``cellpose_channel_input`` key the user picked -- e.g.
       ``meanImgE`` for the enhanced version).
    2. Rescale to ``[0, 1]`` based on the 1st / 99.9th percentiles
       so Cellpose's internal normalisation gets reasonable input
       regardless of the raw mean image's brightness scale.
    3. Run ``CellposeModel.eval``. The function feature-detects the
       Cellpose major version and adjusts the call accordingly.
    4. Convert each integer mask label into a Suite2p-style
       ``stat`` dict (ypix / xpix / lam / med / npix) so downstream
       merging code can treat Cellpose ROIs identically to Sparsery
       ones.
    5. Save ``stat.npy`` + ``masks.npy`` for audit.

    Returns ``(stat, ops_out, plane0)`` with the same shape/semantics
    as ``run_one_pass`` so the rest of the sweep loop
    (residual_metrics, eviction, leaderboard) works unchanged.

    The plane0 directory contains:
      - ops.npy       (copy of the shared registered ops)
      - stat.npy      (cellpose-derived suite2p-style stat list)
      - masks.npy     (raw integer-label mask from cellpose, for audit)
    """
    if not _CELLPOSE_AVAILABLE:
        raise RuntimeError(
            f'cellpose not importable: {_CELLPOSE_IMPORT_ERROR}')

    plane0 = Path(save_dir) / 'suite2p' / 'plane0'
    plane0.mkdir(parents=True, exist_ok=True)

    # Load the shared (post-registration) view to get meanImg + Ly/Lx.
    # In suite2p 1.0 these live in db.npy / settings.npy / reg_outputs.npy
    # which load_plane_view stitches into a flat lookup (with a fallback
    # to a legacy ops.npy on older runs).
    from . import utils
    reg_ops = utils.load_plane_view(shared_plane0)
    img_key = cfg.get('cellpose_channel_input', 'meanImg')
    if img_key not in reg_ops or reg_ops[img_key] is None:
        # Fallback: meanImg is always present after registration.
        img_key = 'meanImg'
    img = np.asarray(reg_ops[img_key]).astype(np.float32)

    # Robust rescale to [0, 1] so Cellpose's internal normalization has
    # reasonable input regardless of raw mean-image dynamic range.
    lo = float(np.quantile(img, 0.01))
    hi = float(np.quantile(img, 0.999))
    if hi <= lo:
        hi = float(img.max()); lo = float(img.min())
    if hi > lo:
        img_norm = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    else:
        img_norm = np.zeros_like(img)

    model_type = cfg.get('cellpose_model_type', 'cyto')
    diameter = cfg.get('cellpose_diameter', 0) or None   # 0 → auto
    flow_threshold = float(cfg.get('cellpose_flow_threshold', 0.4))
    cellprob_threshold = float(cfg.get('cellpose_cellprob_threshold', 0.0))

    # Feature-detect cellpose API version.
    #   Cellpose <=3.x: ``models.Cellpose`` wrapper + ``CellposeModel``; eval
    #       accepts ``channels`` and returns (masks, flows, styles, diams).
    #   Cellpose 4.x  : ``Cellpose`` wrapper removed; only ``CellposeModel``;
    #       eval no longer takes ``channels`` and returns (masks, flows, styles).
    #       Also ``model_type`` (cyto/cyto2/nuclei) is gone — one model ('cpsam').
    has_wrapper = hasattr(_cp_models, 'Cellpose')
    api_version = '3x' if has_wrapper else '4x'

    if verbose:
        print(f"  > running cellpose ({api_version}): model={model_type}  "
              f"diameter={diameter}  flow={flow_threshold}  "
              f"cellprob={cellprob_threshold}  img={img_key} "
              f"{img.shape}")

    t0 = time.time()
    if has_wrapper:
        model = _cp_models.Cellpose(model_type=model_type, gpu=True)
        eval_kwargs = dict(
            diameter=diameter,
            channels=[0, 0],
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )
    else:
        # Cellpose 4.x: the ``model_type`` axis no longer discriminates
        # between cyto/cyto2/nuclei — there's only the unified cpsam model
        # here. We still iterate over it so the ledger/folder name records
        # what we asked for, but each model_type produces the same network.
        model = _cp_models.CellposeModel(gpu=True)
        eval_kwargs = dict(
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )

    eval_out = model.eval(img_norm, **eval_kwargs)
    # 3.x -> (masks, flows, styles, diams); 4.x -> (masks, flows, styles).
    masks = eval_out[0]
    diams = eval_out[3] if len(eval_out) >= 4 else None
    if verbose:
        print(f"  > cellpose done in {time.time() - t0:.1f}s  "
              f"(est. diameter={diams})")

    stat = _cellpose_masks_to_stat(np.asarray(masks))

    # Persist outputs in a suite2p-like layout so the rest of the sweep
    # and any downstream audit tools can find them.
    ops_out = dict(reg_ops)
    ops_out['Ly'] = int(img.shape[0])
    ops_out['Lx'] = int(img.shape[1])
    ops_out['cellpose_config'] = {
        'api_version': api_version,
        'model_type': model_type,
        'diameter_requested': cfg.get('cellpose_diameter', 0),
        'diameter_estimated': float(diams) if diams is not None else None,
        'flow_threshold': flow_threshold,
        'cellprob_threshold': cellprob_threshold,
        'channel_input': img_key,
    }
    np.save(plane0 / 'ops.npy', ops_out, allow_pickle=True)
    np.save(plane0 / 'stat.npy', np.array(stat, dtype=object),
            allow_pickle=True)
    np.save(plane0 / 'cellpose_masks.npy', np.asarray(masks))

    return stat, ops_out, plane0


# -------- main --------

def main():
    save_root = Path(SAVE_FOLDER)
    save_root.mkdir(parents=True, exist_ok=True)
    ledger_path = save_root / RESULTS_FILE_NAME

    adaptive_cfg = AdaptiveConfig(
        tiff_folder=TIFF_FOLDER,
        save_folder=SAVE_FOLDER,
        path_to_ops=PATH_TO_OPS,
        # Turn off all the adaptive machinery — we only want its helpers.
        auto_estimate_spatial_scale=False,
        generate_audit_pngs=False,
        augment_with_residual_blobs=False,
        use_cache=False,            # we manage caching via the ledger
        verbose=True,
        max_rois_hard_cap=HARD_CAP,
    )

    print(f'[brute] load base settings from {PATH_TO_OPS}')
    base_db, base_settings = load_base_settings(adaptive_cfg)

    print(f'[brute] acquire/verify shared registration (one-time)')
    t0 = time.time()
    shared_plane0 = _get_or_create_shared_registration(
        adaptive_cfg, base_db, base_settings,
    )
    print(f'[brute] shared reg ready at {shared_plane0}  '
          f'({time.time() - t0:.1f}s)')

    configs = enumerate_all_configs(SPARSERY_GRID, CELLPOSE_GRID)
    n_sparsery = sum(1 for c in configs
                     if c.get('detection_backend') == 'sparsery')
    n_cellpose = sum(1 for c in configs
                     if c.get('detection_backend') == 'cellpose')
    print(f'[brute] {len(configs)} total configs to evaluate '
          f'({n_sparsery} sparsery, {n_cellpose} cellpose)')
    if n_cellpose and not _CELLPOSE_AVAILABLE:
        print(f'[brute] WARNING: cellpose not importable '
              f'({_CELLPOSE_IMPORT_ERROR}); those configs will be skipped.')

    ledger = load_ledger(ledger_path)
    n_prev_ok = sum(1 for v in ledger.values() if v.get('status') == 'ok')
    n_prev_fail = len(ledger) - n_prev_ok
    print(f'[brute] ledger: {n_prev_ok} ok, {n_prev_fail} failed from previous run(s)')

    start_time = time.time()
    n_new = 0

    for i, cfg in enumerate(configs, 1):
        sig = config_signature(cfg)
        #if sig in ledger:
            #status = ledger[sig].get('status', '?')
            #print(f'[{i}/{len(configs)}] SKIP (already {status}): {sig}')
            #continue

        backend = cfg.get('detection_backend', 'sparsery')

        # Early-skip cellpose runs if the package isn't importable, so
        # the sweep doesn't blow up on every cellpose config.
        if backend == 'cellpose' and not _CELLPOSE_AVAILABLE:
            print(f'[{i}/{len(configs)}] SKIP (cellpose unavailable): {sig}')
            ledger[sig] = {
                'status': 'skipped_no_cellpose',
                'reason': _CELLPOSE_IMPORT_ERROR,
                'config': cfg,
            }
            save_ledger(ledger_path, ledger)
            continue

        folder = save_root / folder_name_for(cfg)

        print(f'\n[{i}/{len(configs)}] RUN  [{backend}]  folder={folder.name}')
        print(f'    cfg = {cfg}')

        t_start = time.time()
        try:
            if backend == 'cellpose':
                stat, ops_out, plane0 = run_cellpose_pass(
                    folder, shared_plane0, cfg, verbose=True,
                )
            else:
                # Build per-config (db, settings) by overlaying the cfg
                # knobs (threshold_scaling, spatial_scale, ...) onto a
                # copy of base_settings.
                pass_settings = {
                    k: (dict(v) if isinstance(v, dict) else v)
                    for k, v in base_settings.items()
                }
                overrides = {k: v for k, v in cfg.items()
                             if k != 'detection_backend'}
                overrides['max_iterations'] = MAX_ITERATIONS
                overrides['roidetect'] = True
                apply_settings_overrides(pass_settings, overrides)
                pass_db = dict(base_db)
                stat, ops_out, plane0 = run_one_pass(
                    TIFF_FOLDER, folder, pass_db, pass_settings,
                    verbose=True, use_cache=False,
                    shared_plane0=shared_plane0,
                    hard_cap=HARD_CAP,
                )
        except Exception as e:
            print(f'  ! exception during run: {e}')
            traceback.print_exc()
            delete_folder(folder)
            ledger[sig] = {
                'status': 'error',
                'reason': str(e),
                'config': cfg,
                'elapsed_s': round(time.time() - t_start, 1),
            }
            save_ledger(ledger_path, ledger)
            n_new += 1
            continue

        elapsed = round(time.time() - t_start, 1)
        n_rois = len(stat)
        hard_capped = (Path(plane0) / 'HARD_CAP_ABORTED.txt').exists()

        if hard_capped:
            print(f'  > HARD CAP — deleting folder ({elapsed}s)')
            delete_folder(folder)
            ledger[sig] = {
                'status': 'hard_capped',
                'config': cfg,
                'n_rois': n_rois,
                'elapsed_s': elapsed,
            }
        elif n_rois == 0:
            print(f'  > 0 ROIs — deleting folder ({elapsed}s)')
            delete_folder(folder)
            ledger[sig] = {
                'status': 'zero_rois',
                'config': cfg,
                'n_rois': 0,
                'elapsed_s': elapsed,
            }
        else:
            metrics = residual_metrics(stat, ops_out)
            print(f'  > OK: n_rois={metrics["n_rois"]}  '
                  f'residual_mean={metrics["residual_mean"]:.3f}  '
                  f'residual_p99={metrics["residual_p99"]:.3f}  '
                  f'median_area={metrics["median_area_px"]:.1f}px  '
                  f'({elapsed}s)')
            ledger[sig] = {
                'status': 'ok',
                'config': cfg,
                'folder': str(folder),
                'plane0': str(plane0),
                'metrics': metrics,
                'elapsed_s': elapsed,
            }

        # Enforce top-N retention BEFORE saving so the ledger on disk
        # always reflects the post-eviction state.
        n_evicted = enforce_top_n(ledger, TOP_N_KEEP, verbose=True)
        save_ledger(ledger_path, ledger)
        n_new += 1
        if n_evicted:
            print(f'  [evict] dropped {n_evicted} folder(s) beyond top-{TOP_N_KEEP}')

        # Rolling leaderboard so you can peek during long runs.
        write_leaderboard(ledger, save_root / LEADERBOARD_FILE_NAME)

    total_elapsed = time.time() - start_time
    print(f'\n[brute] sweep complete: {n_new} new configs, '
          f'total {total_elapsed/60:.1f} min')

    # Final top-N pass in case the user lowered TOP_N_KEEP between runs,
    # or a resumed session carried over entries that now need eviction.
    final_evicted = enforce_top_n(ledger, TOP_N_KEEP, verbose=True)
    if final_evicted:
        print(f'[brute] final sweep evicted {final_evicted} entry(ies)')
        save_ledger(ledger_path, ledger)

    # --- final leaderboard ---
    leaderboard_path = save_root / LEADERBOARD_FILE_NAME
    write_leaderboard(ledger, leaderboard_path)
    print(f'[brute] leaderboard: {leaderboard_path}')
    print(f'[brute] ledger:      {ledger_path}')

    # Print top 20 to stdout for immediate inspection.
    ok = [(k, v) for k, v in ledger.items() if v.get('status') == 'ok']
    ok.sort(key=lambda kv: kv[1]['metrics']['residual_mean'])
    print(f'\nTop 20 by residual_mean (lower = more image explained):')
    print(f'{"rank":>4}  {"residual_mean":>13}  {"n_rois":>7}  '
          f'{"med_area":>8}  config')
    print('-' * 100)
    for i, (sig, r) in enumerate(ok[:20], 1):
        m = r['metrics']
        print(f'{i:>4}  {m["residual_mean"]:>13.4f}  {m["n_rois"]:>7}  '
              f'{m["median_area_px"]:>8.1f}  {sig}')


if __name__ == '__main__':
    main()
