"""
Adaptive Suite2p detection pipeline.

Designed as a drop-in replacement for main.py that iteratively runs Suite2p
with progressively more aggressive detection parameters, auditing the mean
image after each pass to decide whether cells are still being missed.

User only needs to specify:
    - tiff_folder: where the raw TIFF stack lives
    - save_folder: where Suite2p output should be written
    - path_to_ops (optional): starting ops.npy to build from; if None, uses
      suite2p.default_ops() with sensible defaults for human 2p GCaMP data

The goal is to wrap this behind a minimal app where the user never touches
the Suite2p GUI or config files directly.

Note: relocated from the project root into calliope.core during the GUI-package self-containment effort.
"""

from __future__ import annotations

import os
import re
import shutil
import numpy as np
import suite2p
from . import utils
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ============================================================================
# Mid-detection ROI hard-cap via sparsery monkey-patch
# ============================================================================
# Suite2p's sparsery() prints "%d ROIs, score=%.2f" every 1000 ROIs (see
# suite2p/detection/sparsedetect.py:446). Python's LEGB scope resolution means
# if we install a module-level `print` onto suite2p.detection.sparsedetect,
# sparsery's call to print(...) picks up *our* version instead of the builtin.
# Our version watches the ROI count and raises _RoiHardCapExceeded once the
# cap is crossed — aborting mid-detection without touching suite2p source.

class _RoiHardCapExceeded(Exception):
    """Raised from inside sparsery when the ROI count crosses the cap."""
    def __init__(self, count: int, cap: int):
        super().__init__(f"sparsery produced {count} ROIs (>= cap={cap}); "
                         f"threshold is almost certainly detecting noise")
        self.count = count
        self.cap = cap


_SPARSERY_PROGRESS_RE = re.compile(r'^\s*(\d+)\s+ROIs')


def _install_sparsery_roi_cap(cap: int):
    """Monkey-patch suite2p.detection.sparsedetect to abort once len(stats)>=cap.

    Returns the original print so the caller can restore it.
    """
    import suite2p.detection.sparsedetect as _sd
    original_print = getattr(_sd, 'print', print)

    def _watching_print(*args, **kwargs):
        if args:
            msg = args[0] if isinstance(args[0], str) else str(args[0])
            m = _SPARSERY_PROGRESS_RE.match(msg)
            if m and int(m.group(1)) >= cap:
                raise _RoiHardCapExceeded(int(m.group(1)), cap)
        return original_print(*args, **kwargs)

    _sd.print = _watching_print
    return original_print


def _restore_sparsery_print(original_print):
    """Undo _install_sparsery_roi_cap."""
    import suite2p.detection.sparsedetect as _sd
    if original_print is print:
        # Was the builtin; just remove our shim so lookups fall through to builtin
        if 'print' in _sd.__dict__:
            del _sd.__dict__['print']
    else:
        _sd.print = original_print


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class AdaptiveConfig:
    """All parameters controlling the adaptive loop in one place."""

    # ---- Required paths ----
    tiff_folder: str = ""                         # folder containing TIFFs
    save_folder: str = ""                         # where per-pass output lives
    path_to_ops: Optional[str] = None             # optional starting ops.npy

    # ---- Sample metadata (used for tau lookup) ----
    aav_info_csv: str = "human_SLE_2p_meta.csv"
    tau_vals: dict = field(default_factory=lambda: {
        "6f": 0.7, "6m": 1.0, "6s": 1.3, "8m": 0.137,
    })

    # ---- Spatial scale pinning ----
    # 0 = defer to auto_estimate_spatial_scale below (our pre-flight blob
    # estimator) or to suite2p's built-in auto-picker if that's disabled.
    # Set to 1 (~6 px), 2 (~12 px), 3 (~24 px), or 4 (~48 px) to hard-pin.
    pass0_spatial_scale: int = 0

    # When pass0_spatial_scale == 0 and this is True, load_base_ops reads
    # the first ~500 frames of the TIFF, averages them, runs LoG blob
    # detection, and maps the median blob diameter to a suite2p scale bin.
    # This catches the common failure mode where suite2p's internal
    # estimator prints "Spatial scale estimation failed. Setting spatial
    # scale to 1 in order to continue." and then misses every real cell.
    auto_estimate_spatial_scale: bool = True
    auto_estimate_n_frames: int = 500

    # Spatial-scale estimator: cap ranking at the top N blobs by integrated
    # brightness (size x intensity). 200 is a soft upper bound on expected
    # cell count for our typical FOVs; blobs beyond this are likely junk or
    # bright punctae that would skew the median diameter downward.
    spatial_scale_top_n: int = 200

    # If True, write a 3-panel QC figure (all blobs / top-N / diameter
    # histogram) alongside the adaptive run. Default path is
    # ``{save_folder}/spatial_scale_qc.png``; override with
    # ``spatial_scale_qc_path``.
    save_spatial_scale_qc: bool = True
    spatial_scale_qc_path: Optional[str] = None

    # ---- Blob-seeded detection ----
    # When True, bypass suite2p's sparsery entirely: run registration only,
    # take the pre-flight LoG estimator's blobs at the winning scale (plus
    # optionally other scales), convert each blob to a circular ROI with
    # Gaussian-weighted ``lam``, and run suite2p's extraction + classifier
    # + deconvolution on those ROIs. Useful when sparsery is missing
    # structurally visible cells that simply aren't firing much.
    blob_seeded_detection: bool = False
    # If True, seed from all 4 scales; if False, only the winning scale.
    blob_seed_all_scales: bool = False
    # Override which scale's blobs to seed from (1-4). 0 = use the
    # estimator winner. Useful when the estimator's winning scale doesn't
    # match the scale sparsery actually runs at (e.g. estimator picks s3
    # but somata are ~8 px diameter so s1 is where the real cells are).
    blob_seed_scale_override: int = 0
    # Keep only the top-N brightest blobs (per scale if multi-scale, or
    # total if single-scale) as ROI seeds. None = keep all.
    blob_seed_max_rois: Optional[int] = None
    # Gaussian weight falloff inside the blob disc: lam = exp(-d^2 /
    # (2*(factor*r)^2)). Lower = more peaked; higher = flatter.
    blob_seed_gaussian_sigma_factor: float = 0.7

    # ---- Under-detection schedule toggle ----
    # When pass 0 already looks good (high classifier acceptance, median
    # ROI area consistent with real cells), lowering the threshold typically
    # finds noise speckles — not more cells. Set this to False to stop at
    # pass 0 and return whatever it found, skipping the iterative decay.
    run_under_detection_schedule: bool = True

    # ---- Mid-schedule speckle guard ----
    # After each under-detection pass, if the NEW pass's median ROI area is
    # below this fraction of pass 0's median area, treat it as a
    # speckle-explosion and do NOT merge it into the accumulated result.
    # Stops the schedule. 0.0 disables the guard.
    speckle_guard_area_ratio: float = 0.5
    # Same idea, absolute floor: any pass whose median ROI area is below
    # this many pixels is rejected outright. 0 disables.
    speckle_guard_min_area_px: float = 15.0

    # ---- Seed from an existing pass 0 ----
    # If set, adaptive_detect skips its own pass-0 loop entirely and uses
    # the stat/ops already saved at this plane0 path as the baseline. Also
    # hardlinks that plane0's ``data.bin`` as the shared registration so
    # subsequent passes reuse it. Point this at (e.g.)
    # ``D:/.../pass00_thr1.00_sc1/suite2p/plane0`` to pick up where a
    # previous run left off without re-registering or re-detecting pass 0.
    seed_pass0_from: Optional[str] = None

    # ---- Registration reuse ----
    # When True, run registration ONCE into ``{save_folder}/_shared_reg/``
    # and hardlink ``data.bin`` into each pass folder. Saves ~8 GB and
    # several minutes per additional pass. Disable only for debugging.
    reuse_registration: bool = True

    # ---- Residual-blob augmentation ----
    # After the sparsery-based schedule finishes, detect blobs in the
    # residual image (mean image with existing ROI pixels masked out),
    # seed ROIs at their centers, extract fluorescence, and let the
    # classifier decide which to keep. This recovers cells that sparsery
    # misses structurally — they're visible in the mean image but either
    # don't fire often enough or don't cross sparsery's threshold.
    augment_with_residual_blobs: bool = False
    # Minimum classifier probability an augmented ROI must reach to be
    # included in the final merged output. Higher = stricter filter.
    # Set to 0.0 to keep all classifier-assigned ROIs.
    augmentation_min_iscell_prob: float = 0.3

    # ---- Audit PNG generation ----
    # When True, adaptive_detect() walks save_folder at the end and writes an
    # audit.png into every suite2p/plane0 subfolder it finds. Each figure is
    # a 3-panel: (1) mean image + detected ROI contours, (2) residual image
    # with ROI pixels masked, (3) mean image + missed-cell blob candidates.
    generate_audit_pngs: bool = True

    # ---- Spatial-scale runtime fallback ----
    # If pass 0 returns 0 ROIs and ``allow_spatial_scale_fallback`` is True,
    # decrement spatial_scale by 1 and retry, up to
    # ``max_spatial_scale_fallbacks`` times. This catches the case where
    # the LoG estimator picks (say) s3 but suite2p's sparsery can't lock
    # onto somata at that scale and needs s2 instead.
    allow_spatial_scale_fallback: bool = True
    max_spatial_scale_fallbacks: int = 3

    # ---- Adaptive threshold (under-detection regime) ----
    # Two modes for choosing threshold_scaling across passes:
    #
    #   'binary_search' (default):
    #     Treats the threshold as a parameter to fit. The speckle guard is
    #     the discriminator: a pass is "safe" if its median ROI area meets
    #     the guard, otherwise "speckle". The search bracket starts at
    #     [min_threshold_scaling, initial_threshold_scaling] and shrinks as
    #     passes classify threshold values. Safe passes get merged in;
    #     speckle passes don't. Stops when the bracket width falls below
    #     ``binary_search_precision`` or a safe pass adds <
    #     ``min_new_rois_per_pass`` new ROIs.
    #
    #   'decay':
    #     Legacy fixed-step schedule:
    #       multiplicative:  thr_next = thr_prev * threshold_decay_factor
    #       additive:        thr_next = thr_prev - threshold_decay_offset
    #     Bounded below by ``min_threshold_scaling``. Aggressive 2x step on
    #     0-ROI passes.
    #
    # Junk suppression is handled by the speckle guard + downstream
    # classifier, not by the threshold itself.
    threshold_schedule_mode: str = 'binary_search'   # or 'decay'
    initial_threshold_scaling: float = 1.0
    threshold_decay_mode: str = 'multiplicative'   # used in 'decay' mode
    threshold_decay_factor: float = 0.5
    threshold_decay_offset: float = 0.25
    min_threshold_scaling: float = 0.1
    max_under_detection_passes: int = 5
    default_max_iterations: int = 200
    # Binary search precision: stop when hi - lo < this fraction of the
    # initial bracket width. 0.05 = stop when the interval is <5% of
    # [min_threshold_scaling, initial_threshold_scaling].
    binary_search_precision: float = 0.05

    # ---- Mid-detection ROI hard cap ----
    # Abort a sparsery pass the moment it crosses this many ROIs. Installed as
    # a print-hook inside suite2p.detection.sparsedetect so we don't have to
    # wait for the pass to finish before rejecting it. Any pass that hits the
    # cap is treated as speckle: its ROIs are discarded and the bracket
    # upper bound is tightened in binary_search mode. The check fires at the
    # nearest 1000-ROI boundary after the cap (sparsery's progress print
    # granularity), so 20000 means ~20000 ROIs, not exactly 20000.
    # Set to 0 to disable.
    max_rois_hard_cap: int = 20000

    # ---- Over-detection fallback (legacy) ----
    # When ``enable_over_detection_fallback`` is True, the pipeline will
    # re-run at different (spatial_scale, threshold, max_overlap) combos
    # if the merged result looks over-detected. Disabled by default
    # because the speckle guard + downstream classifier now handle junk
    # filtering more reliably.
    enable_over_detection_fallback: bool = False
    spatial_scale_schedule: tuple = (
        (2, 1.5, 0.5),   # 12-px cells, conservative
        (2, 1.0, 0.5),   # 12-px cells, default threshold
        (3, 1.2, 0.5),   # 24-px cells, fallback if somata are larger
    )

    # ---- Over-detection diagnosis (triggers fallback schedule) ----
    # If the first pass detects more ROIs than this, assume wrong spatial scale
    over_detection_roi_count: int = 1500
    # Or if the median ROI pixel area is below this, same conclusion
    over_detection_median_area_px: float = 20.0
    # Also require that accepted/total ratio from suite2p classifier is low,
    # since a huge number of real cells is possible in principle. If iscell
    # keeps fewer than this fraction, that confirms the ROIs are mostly junk.
    over_detection_iscell_ratio: float = 0.15

    # ---- Blob detection on residual image ----
    # These defaults are tuned for dense-field 2p recordings where cells are
    # small (~8 px diameter) and closely packed. Loosen (lower contrast,
    # wider tolerance, lower center/surround ratio) for higher recall at the
    # cost of more candidate blobs the classifier then needs to filter.
    soma_diameter_px: float = 12.0
    soma_scale_tolerance: float = 0.7     # sigma range = r*(1±tol)/√2
    blob_min_contrast: float = 0.04       # LoG threshold after robust norm
    blob_min_area_px: int = 10            # don't drop small sparse-firing cells
    blob_max_area_px: int = 400
    blob_center_surround_ratio: float = 1.15  # was 1.5 (too strict for dense fields)
    blob_num_sigma: int = 10              # finer scale resolution

    # ---- Stopping criteria ----
    min_residual_blobs: int = 8          # stop if fewer missed cells than this
    min_new_rois_per_pass: int = 3       # stop if a pass adds fewer than this
    iou_dedup_threshold: float = 0.3     # ROI pairs above this IoU are merged

    # ---- Caching ----
    # When True, each pass checks its save_dir for an existing completed
    # suite2p run with matching ops and reuses it instead of rerunning.
    # Delete the pass directory to force a rerun of just that pass.
    use_cache: bool = True

    # ---- Misc ----
    verbose: bool = True


# ============================================================================
# Pre-flight spatial scale estimation
# ============================================================================

def estimate_spatial_scale_from_tiff(tiff_folder: str,
                                     n_frames: int = 500,
                                     top_n: int = 200,
                                     qc_path: Optional[str] = None,
                                     verbose: bool = True,
                                     return_scale_data: bool = False):
    """
    Probe raw TIFF frames to pick a suite2p spatial_scale bin.

    Uses *per-scale* LoG blob detection: the sigma space is tiled into
    four non-overlapping bins corresponding to suite2p's spatial_scale
    values (1 = 6 px, 2 = 12 px, 3 = 24 px, 4 = 48 px), and at each bin
    we run blob_log, compute integrated brightness inside each blob's
    circular footprint, and sum the top ``top_n`` masses. The scale with
    the highest top-N mass wins.

    Why per-scale and not one big multi-sigma pass: a single blob_log
    call across sigma 1.5-22 preferentially locks onto the bright center
    pixel of each cell at tiny sigma, so 12-24 px somata get reported as
    4 px "blobs". Forcing the detector to work within each scale bin
    avoids this artifact, and the top-N mass comparison directly answers
    "at which scale does the FOV carry the most signal?".

    Background subtraction uses a 75-px median filter — large enough that
    even a 48-px cell centered in the window is not self-subtracted.

    If ``qc_path`` is supplied, writes a 4-panel QC figure (all blobs by
    scale / winner top-N / per-scale mass bars / per-scale counts).

    Returns an int in ``{1, 2, 3, 4}`` on success, or ``0`` if estimation
    fails (no TIFF found, no blobs detected, etc.) so the caller can fall
    back to suite2p's built-in estimator.

    If ``return_scale_data=True``, returns ``(winner, scale_data)`` so the
    caller can reuse the detected blobs (e.g. for blob-seeded detection).
    ``scale_data`` is a dict keyed by scale 1-4 containing 'blobs',
    'scores', 'kept_idx', 'top_mass', 'n_detected', 'n_kept'. On failure
    returns ``(0, None)``.
    """
    _fail = (0, None) if return_scale_data else 0

    try:
        import tifffile
        from skimage.feature import blob_log
        from scipy.ndimage import median_filter
    except ImportError as e:
        if verbose:
            print(f"  spatial-scale estimator: missing dependency ({e}); skipping")
        return _fail

    # Load the first N frames of the first TIFF
    frames = None
    try:
        for name in sorted(os.listdir(tiff_folder)):
            if not name.lower().endswith(('.tif', '.tiff')):
                continue
            with tifffile.TiffFile(os.path.join(tiff_folder, name)) as tf:
                n_avail = len(tf.pages)
                n = min(n_frames, n_avail)
                frames = np.stack([tf.pages[i].asarray() for i in range(n)])
            break
    except Exception as e:
        if verbose:
            print(f"  spatial-scale estimator: could not read TIFF ({e}); skipping")
        return _fail

    if frames is None or frames.size == 0:
        return _fail

    mean_img = frames.mean(axis=0).astype(np.float32)

    # Wide median-filter background: must be larger than the biggest cell
    # we might detect, else bg subtraction erases the cell body itself
    # (a 25-px window centered on a 24-px cell is almost entirely cell).
    bg = median_filter(mean_img, size=75)
    hp = mean_img - bg
    hp[hp < 0] = 0

    hi = float(np.quantile(hp, 0.995))
    if hi <= 0:
        return _fail
    norm = np.clip(hp / hi, 0.0, 1.0)

    # Per-scale blob detection. Sigma bins tile the space with no overlap:
    #   s1: diameter  4-8.5 px  (sigma  1.4-3.0)
    #   s2: diameter  8.5-17 px (sigma  3.0-6.0)
    #   s3: diameter  17-34 px  (sigma  6.0-12.0)
    #   s4: diameter  34-68 px  (sigma  12.0-24.0)
    # diameter = 2*sigma*sqrt(2); bin edges are log-midpoints between
    # suite2p's nominal scale sizes (6, 12, 24, 48 px).
    scale_defs = [
        (1, 1.4, 3.0),
        (2, 3.0, 6.0),
        (3, 6.0, 12.0),
        (4, 12.0, 24.0),
    ]

    Ly_img, Lx_img = norm.shape
    scale_data = {}
    for sp, min_s, max_s in scale_defs:
        blobs = blob_log(norm, min_sigma=min_s, max_sigma=max_s,
                         num_sigma=5, threshold=0.05, overlap=0.5)
        if blobs.size == 0:
            scale_data[sp] = {
                'blobs': np.empty((0, 3), dtype=np.float32),
                'scores': np.array([], dtype=np.float32),
                'kept_idx': np.array([], dtype=int),
                'top_mass': 0.0,
                'n_detected': 0,
                'n_kept': 0,
            }
            continue

        scores = np.zeros(len(blobs), dtype=np.float32)
        for i in range(len(blobs)):
            yc, xc, sigma = blobs[i]
            r = int(np.ceil(sigma * np.sqrt(2.0)))
            y0 = max(0, int(round(yc)) - r)
            y1 = min(Ly_img, int(round(yc)) + r + 1)
            x0 = max(0, int(round(xc)) - r)
            x1 = min(Lx_img, int(round(xc)) + r + 1)
            patch = norm[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            # integrate within the circular footprint, not the bbox, so
            # neuropil in the corners doesn't inflate the score
            yy, xx = np.ogrid[y0:y1, x0:x1]
            mask = ((yy - yc) ** 2 + (xx - xc) ** 2) <= (r ** 2)
            scores[i] = float(patch[mask].sum()) if mask.any() else 0.0

        order = np.argsort(scores)[::-1]
        n_keep = min(top_n, len(scores))
        kept_idx = order[:n_keep]
        scale_data[sp] = {
            'blobs': blobs,
            'scores': scores,
            'kept_idx': kept_idx,
            'top_mass': float(scores[kept_idx].sum()),
            'n_detected': len(blobs),
            'n_kept': int(n_keep),
        }

    # Winner = scale whose top-N integrated mass is largest
    winner = max(scale_data, key=lambda s: scale_data[s]['top_mass'])

    if verbose:
        print(f"  spatial-scale estimator (per-scale top-{top_n} mass):")
        for sp in (1, 2, 3, 4):
            d = scale_data[sp]
            marker = "  <-- winner" if sp == winner else ""
            approx = 6 * 2 ** (sp - 1)
            print(f"    scale {sp} (~{approx:>2d} px): "
                  f"n_detected={d['n_detected']:>5d}  "
                  f"top_mass={d['top_mass']:>10.1f}{marker}")

    if qc_path is not None:
        try:
            _render_spatial_scale_qc(
                qc_path=qc_path, mean_img=mean_img,
                scale_data=scale_data, winner=winner, top_n=top_n,
            )
            if verbose:
                print(f"  spatial-scale estimator: QC figure saved to {qc_path}")
        except Exception as e:
            if verbose:
                print(f"  spatial-scale estimator: QC figure failed ({e})")

    if scale_data[winner]['n_detected'] == 0:
        if verbose:
            print("  spatial-scale estimator: no blobs at any scale; skipping")
        return _fail

    if verbose:
        approx = 6 * 2 ** (winner - 1)
        print(f"  spatial-scale estimator: -> spatial_scale = {winner} "
              f"(~{approx} px)")
    if return_scale_data:
        return winner, scale_data
    return winner


def _render_spatial_scale_qc(qc_path, mean_img, scale_data, winner, top_n):
    """Four-panel QC figure for the per-scale spatial-scale estimator."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.lines import Line2D
    from matplotlib import colormaps, colors
    from matplotlib import cm  # noqa: F401 (ScalarMappable)

    qc_path = Path(qc_path)
    qc_path.parent.mkdir(parents=True, exist_ok=True)

    vmax = float(np.quantile(mean_img, 0.995))
    vmin = float(np.quantile(mean_img, 0.01))

    scale_colors = {1: 'tab:blue', 2: 'tab:green',
                    3: 'tab:orange', 4: 'tab:red'}

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Panel (0,0): all blobs from all scales, color = scale bin
    ax = axes[0, 0]
    ax.imshow(mean_img, cmap='gray', vmin=vmin, vmax=vmax)
    total_all = 0
    for sp in (1, 2, 3, 4):
        d = scale_data[sp]
        total_all += d['n_detected']
        for y, x, sigma in d['blobs']:
            r = sigma * np.sqrt(2.0)
            ax.add_patch(Circle((x, y), r, fill=False, linewidth=0.6,
                                edgecolor=scale_colors[sp], alpha=0.55))
    ax.set_title(f"All blobs across scales (n = {total_all}), color = scale bin")
    ax.axis('off')
    legend_elems = [Line2D([0], [0], marker='o', color=scale_colors[sp],
                           markerfacecolor='none', linestyle='',
                           label=f"s{sp} (~{6 * 2**(sp-1)} px, "
                                 f"n={scale_data[sp]['n_detected']})")
                    for sp in (1, 2, 3, 4)]
    ax.legend(handles=legend_elems, loc='upper right', fontsize=8)

    # Panel (0,1): winning scale, top-N by mass, color = mass
    ax = axes[0, 1]
    ax.imshow(mean_img, cmap='gray', vmin=vmin, vmax=vmax)
    d = scale_data[winner]
    if d['n_kept'] > 0:
        kept_scores = d['scores'][d['kept_idx']]
        k_min = float(kept_scores.min())
        k_max = float(kept_scores.max())
        if k_max <= k_min:
            k_max = k_min + 1.0
        snorm = colors.Normalize(vmin=k_min, vmax=k_max)
        cmap = colormaps['viridis']
        for idx in d['kept_idx']:
            y, x, sigma = d['blobs'][idx]
            r = sigma * np.sqrt(2.0)
            ax.add_patch(Circle((x, y), r, fill=False, linewidth=1.3,
                                edgecolor=cmap(snorm(d['scores'][idx]))))
        sm = cm.ScalarMappable(cmap=cmap, norm=snorm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, label="blob mass")
    ax.set_title(f"Winner: scale {winner} (~{6 * 2**(winner-1)} px), "
                 f"top {d['n_kept']} of {d['n_detected']} by mass")
    ax.axis('off')

    # Panel (1,0): per-scale total top-N mass -- this is the decision metric
    ax = axes[1, 0]
    scales = [1, 2, 3, 4]
    masses = [scale_data[s]['top_mass'] for s in scales]
    xlabels = [f"s{s}\n~{6 * 2**(s-1)} px" for s in scales]
    bars = ax.bar(xlabels, masses, color=[scale_colors[s] for s in scales])
    bars[winner - 1].set_edgecolor('black')
    bars[winner - 1].set_linewidth(3.0)
    ymax = max(masses) if max(masses) > 0 else 1.0
    for bar, m in zip(bars, masses):
        ax.text(bar.get_x() + bar.get_width() / 2,
                m + 0.02 * ymax, f"{m:.0f}",
                ha='center', va='bottom', fontsize=10)
    ax.set_ylabel(f"Top-{top_n} integrated mass (sum)")
    ax.set_title(f"Decision metric: per-scale top-N mass  (winner: s{winner})")

    # Panel (1,1): detection count per scale
    ax = axes[1, 1]
    counts = [scale_data[s]['n_detected'] for s in scales]
    kept_counts = [scale_data[s]['n_kept'] for s in scales]
    x = np.arange(4)
    ax.bar(x - 0.2, counts, 0.4, label='detected',
           color='lightgray', edgecolor='gray')
    ax.bar(x + 0.2, kept_counts, 0.4, label=f'kept (top {top_n})',
           color=[scale_colors[s] for s in scales])
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("n blobs")
    ax.set_title("Per-scale blob counts")
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    fig.savefig(qc_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# Ops loading / preparation
# ============================================================================

def load_base_ops(config: AdaptiveConfig, return_scale_data: bool = False):
    """
    Load starting ops either from a user-supplied .npy or from Suite2p defaults.
    Apply sample-specific adjustments (tau via GCaMP lookup, batch size, etc.)
    following the same convention as main.py.

    If ``return_scale_data=True``, also returns the ``scale_data`` dict from
    the pre-flight estimator (or ``None`` if estimation was skipped / failed).
    This is used by blob-seeded detection to reuse the detected blobs as
    ROI seeds without re-running the estimator.
    """
    if config.path_to_ops is not None and os.path.exists(config.path_to_ops):
        if config.verbose:
            print(f"Loading base ops from {config.path_to_ops}")
        ops = np.load(config.path_to_ops, allow_pickle=True).item()
    else:
        if config.verbose:
            print("No base ops provided, starting from suite2p defaults")
        ops = suite2p.default_ops()
        # Data-specific defaults (can be overridden by a loaded .npy)
        ops.update({
            'fs': 15.07,
            'nchannels': 1,
            'nplanes': 1,
            'high_pass': 100.0,
            'smooth_sigma': 1.3,
        })

    # --- Resolve spatial_scale ---
    # If the user left pass0_spatial_scale at 0 and asked for auto-estimation,
    # probe the raw TIFF to pick a scale. This runs before suite2p touches
    # the data, so a wrong-scale pass (suite2p's internal estimator
    # defaulting to 1) is avoided entirely.
    effective_spatial_scale = config.pass0_spatial_scale
    scale_data = None
    if effective_spatial_scale == 0 and config.auto_estimate_spatial_scale:
        if config.save_spatial_scale_qc:
            qc_path = (config.spatial_scale_qc_path
                       or str(Path(config.save_folder) / 'spatial_scale_qc.png'))
        else:
            qc_path = None
        # Always request scale_data so blob-seeded detection can reuse it
        # without re-running the estimator (and re-writing the QC figure).
        estimated, scale_data = estimate_spatial_scale_from_tiff(
            config.tiff_folder,
            n_frames=config.auto_estimate_n_frames,
            top_n=config.spatial_scale_top_n,
            qc_path=qc_path,
            verbose=config.verbose,
            return_scale_data=True,
        )
        if estimated > 0:
            effective_spatial_scale = estimated

    # --- Pipeline-required ops, applied to BOTH branches ---
    # These are enforced regardless of whether ops came from .npy or defaults,
    # because the adaptive loop's behavior depends on them. preclassify in
    # particular is the junk-culling mechanism; without it, under-detection
    # passes produce thousands of speckle ROIs.
    pipeline_required = {
        'sparse_mode': True,
        'spatial_scale': effective_spatial_scale,  # 0 = suite2p internal auto
        'preclassify': 0.5,       # drop obvious non-cells during detection
        'allow_overlap': False,
    }
    if config.verbose:
        for k, v in pipeline_required.items():
            prev = ops.get(k, '<unset>')
            if prev != v:
                print(f"  enforcing ops['{k}'] = {v}  (was {prev!r})")
    ops.update(pipeline_required)

    # Apply sample-specific tau if metadata is available
    file_name = os.path.basename(os.path.normpath(config.tiff_folder))
    if os.path.exists(config.aav_info_csv):
        try:
            tau = utils.file_name_to_aav_to_dictionary_lookup(
                file_name, config.aav_info_csv, config.tau_vals
            )
            ops['tau'] = tau
            if config.verbose:
                print(f"Set tau={tau} based on AAV lookup for {file_name}")
        except Exception as e:
            if config.verbose:
                print(f"tau lookup failed ({e}); keeping existing tau={ops.get('tau')}")

    # Dynamic batch size (same logic as main.py)
    ops['batch_size'] = utils.change_batch_according_to_free_ram()

    # Set/cap nbinned so sparsery's peak intermediate fits in RAM.
    # If ops didn't specify nbinned (or specified a too-large value), we
    # compute a safe value from available RAM and TIFF frame size.
    safe_nbinned = utils.change_nbinned_according_to_free_ram(config.tiff_folder)
    current_nbinned = ops.get('nbinned', 0)
    if current_nbinned <= 0:
        if config.verbose:
            print(f"Setting nbinned = {safe_nbinned} (auto, based on available RAM)")
        ops['nbinned'] = safe_nbinned
    elif current_nbinned > safe_nbinned:
        if config.verbose:
            print(f"Lowering nbinned {current_nbinned} -> {safe_nbinned} to fit available RAM")
        ops['nbinned'] = safe_nbinned

    if config.verbose:
        summary_keys = ('sparse_mode', 'spatial_scale', 'preclassify',
                        'allow_overlap', 'high_pass', 'smooth_sigma', 'tau',
                        'fs', 'nbinned', 'batch_size')
        summary = ', '.join(f"{k}={ops.get(k)!r}" for k in summary_keys)
        print(f"  effective ops: {summary}")

    if return_scale_data:
        return ops, scale_data
    return ops


# ============================================================================
# ROI mask, residual image, blob detection
# ============================================================================

def build_roi_pixel_mask(stat, Ly: int, Lx: int) -> np.ndarray:
    """Binary mask of all pixels belonging to any detected ROI."""
    mask = np.zeros((Ly, Lx), dtype=bool)
    for s in stat:
        ypix = np.asarray(s['ypix'])
        xpix = np.asarray(s['xpix'])
        valid = (ypix >= 0) & (ypix < Ly) & (xpix >= 0) & (xpix < Lx)
        mask[ypix[valid], xpix[valid]] = True
    return mask


def compute_residual_image(ops: dict, roi_mask: np.ndarray,
                           dilate_px: int = 2, prefer: str = 'meanImg'):
    """
    Residual brightness after subtracting detected ROIs from the mean image.
    Dilates the ROI mask slightly so neuropil rings around detected cells
    don't register as missed somata.

    prefer='meanImg' (default) uses the raw mean, which preserves dynamic
    range. 'meanImgE' uses the enhanced mean but is often clipped at 1.0.
    """
    from scipy.ndimage import binary_dilation

    if prefer == 'meanImgE' and ops.get('meanImgE') is not None:
        img = ops['meanImgE'].astype(np.float32)
    elif 'meanImg' in ops:
        img = ops['meanImg'].astype(np.float32)
    else:
        img = ops['meanImgE'].astype(np.float32)

    dilated = binary_dilation(roi_mask, iterations=dilate_px)
    residual = img.copy()

    if dilated.any() and (~dilated).any():
        bg = float(np.median(img[~dilated]))
    else:
        bg = float(np.median(img))
    residual[dilated] = bg
    return residual, bg


def detect_residual_blobs(residual: np.ndarray, config: AdaptiveConfig):
    """
    Multi-scale Laplacian-of-Gaussian blob detection tuned to soma size.

    The residual image from 2p recordings has a heavy right tail (a handful of
    very bright pixels) and a diffuse neuropil background. We:
      1. Subtract a local median-filter background at roughly soma scale
      2. Robust-rescale to [0,1] using 99.5th percentile
      3. Require each blob to have 1.5x center/surround contrast
    """
    from skimage.feature import blob_log
    from scipy.ndimage import median_filter

    r = config.soma_diameter_px / 2.0
    tol = config.soma_scale_tolerance
    min_sigma = (r * (1.0 - tol)) / np.sqrt(2.0)
    max_sigma = (r * (1.0 + tol)) / np.sqrt(2.0)

    img = residual.astype(np.float32)

    bg_radius = max(3, int(round(config.soma_diameter_px)))
    bg_local = median_filter(img, size=bg_radius)
    hp = img - bg_local
    hp[hp < 0] = 0

    hi = float(np.quantile(hp, 0.995))
    if hi <= 0:
        return []
    norm = np.clip(hp / hi, 0.0, 1.0)

    blobs = blob_log(
        norm,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=int(getattr(config, 'blob_num_sigma', 10)),
        threshold=config.blob_min_contrast,
        overlap=0.5,
    )
    if blobs.size == 0:
        return []

    cs_ratio = float(getattr(config, 'blob_center_surround_ratio', 1.5))
    out = []
    for y, x, sigma in blobs:
        radius = sigma * np.sqrt(2.0)
        area = np.pi * radius ** 2
        if area < config.blob_min_area_px or area > config.blob_max_area_px:
            continue
        yi, xi = int(round(y)), int(round(x))
        yi = max(0, min(img.shape[0] - 1, yi))
        xi = max(0, min(img.shape[1] - 1, xi))

        rr = int(round(radius))
        center_val = float(hp[yi, xi])
        y0, y1 = max(0, yi - 2*rr), min(img.shape[0], yi + 2*rr + 1)
        x0, x1 = max(0, xi - 2*rr), min(img.shape[1], xi + 2*rr + 1)
        surround = float(np.median(hp[y0:y1, x0:x1]))
        if center_val < surround * cs_ratio + 1e-6:
            continue

        out.append((yi, xi, radius, center_val))
    return out


# ============================================================================
# ROI merging across passes
# ============================================================================

def _stat_to_pixel_set(s):
    ypix = np.asarray(s['ypix']).tolist()
    xpix = np.asarray(s['xpix']).tolist()
    return set(zip(ypix, xpix))


def _iou(pixset_a, pixset_b):
    if not pixset_a or not pixset_b:
        return 0.0
    inter = len(pixset_a & pixset_b)
    if inter == 0:
        return 0.0
    return inter / (len(pixset_a) + len(pixset_b) - inter)


def merge_roi_lists(stat_old, stat_new, iou_threshold=0.3):
    """Combine two ROI lists; dedupe by spatial IoU."""
    old_pixsets = [_stat_to_pixel_set(s) for s in stat_old]
    merged = list(stat_old)
    n_added = 0
    for s_new in stat_new:
        ps_new = _stat_to_pixel_set(s_new)
        if any(_iou(ps_new, ps_old) >= iou_threshold for ps_old in old_pixsets):
            continue
        merged.append(s_new)
        old_pixsets.append(ps_new)
        n_added += 1
    return merged, n_added


# ============================================================================
# Single Suite2p pass wrapper
# ============================================================================

# Keys whose values determine whether a cached pass is reusable. RAM-varying
# keys like nbinned and batch_size are deliberately excluded — they don't
# materially change which ROIs get detected.
_CACHE_COMPARE_KEYS = (
    'threshold_scaling', 'max_iterations', 'spatial_scale', 'sparse_mode',
    'high_pass', 'smooth_sigma', 'preclassify', 'allow_overlap', 'tau',
    'max_overlap', 'fs', 'nchannels', 'nplanes',
)


def _load_cached_pass(save_dir: Path, expected_ops: dict, verbose: bool = True):
    """
    If ``save_dir`` already contains a completed suite2p run whose ops match
    ``expected_ops`` on the keys that control detection, return
    ``(stat, cached_ops, plane0)``. Otherwise return None.
    """
    plane0 = Path(save_dir) / 'suite2p' / 'plane0'
    stat_path = plane0 / 'stat.npy'
    ops_path = plane0 / 'ops.npy'
    if not (stat_path.exists() and ops_path.exists()):
        return None

    try:
        cached_ops = np.load(ops_path, allow_pickle=True).item()
    except Exception as e:
        if verbose:
            print(f"  > cache miss at {plane0}: could not load ops.npy ({e})")
        return None

    mismatches = []
    for k in _CACHE_COMPARE_KEYS:
        exp = expected_ops.get(k)
        cached = cached_ops.get(k)
        if exp != cached:
            mismatches.append((k, cached, exp))

    if mismatches:
        if verbose:
            diffs = ', '.join(f"{k}: cached={c!r} vs expected={e!r}"
                              for k, c, e in mismatches)
            print(f"  > cache miss at {Path(save_dir).name}: {diffs}")
        return None

    try:
        stat = list(np.load(stat_path, allow_pickle=True))
    except Exception as e:
        if verbose:
            print(f"  > cache miss at {plane0}: could not load stat.npy ({e})")
        return None

    if verbose:
        print(f"  > reusing cached pass {Path(save_dir).name} ({len(stat)} ROIs)")
    return stat, cached_ops, plane0


def _link_or_copy(src: Path, dst: Path):
    """Hardlink src -> dst (cheap), fall back to copy on cross-device / perm error."""
    if dst.exists():
        return
    try:
        os.link(str(src), str(dst))
    except (OSError, NotImplementedError):
        shutil.copy2(str(src), str(dst))


def _get_or_create_shared_registration(config: 'AdaptiveConfig',
                                       base_ops: dict) -> Path:
    """
    Ensure a single canonical registered binary exists at
    ``{save_folder}/_shared_reg/suite2p/plane0/`` and return that plane0
    path. All subsequent detection passes hardlink ``data.bin`` from here
    instead of re-registering the TIFF each time.

    If a shared registration already exists, returns immediately.
    Otherwise runs ``suite2p.run_s2p`` with ``roidetect=False`` once to
    produce it.
    """
    save_root = Path(config.save_folder)
    shared_dir = save_root / '_shared_reg'
    shared_plane0 = shared_dir / 'suite2p' / 'plane0'

    if (shared_plane0 / 'data.bin').exists() and (shared_plane0 / 'ops.npy').exists():
        if config.verbose:
            print(f"  > reusing shared registration at {shared_plane0}")
        return shared_plane0

    if config.verbose:
        print(f"  > no shared registration yet; running suite2p register-only "
              f"into {shared_dir}")

    reg_ops = dict(base_ops)
    reg_ops['save_path0'] = str(shared_dir)
    reg_ops['save_folder'] = 'suite2p'
    reg_ops['roidetect'] = False
    db = {'data_path': [config.tiff_folder]}
    suite2p.run_s2p(ops=reg_ops, db=db)

    if not (shared_plane0 / 'data.bin').exists():
        raise RuntimeError(
            f"shared registration completed but data.bin was not written to "
            f"{shared_plane0}. Check suite2p output for errors."
        )
    return shared_plane0


def run_one_pass(tiff_folder: str, save_dir: Path, ops: dict,
                 verbose: bool = True, use_cache: bool = True,
                 shared_plane0: Optional[Path] = None,
                 hard_cap: int = 0):
    """
    Run Suite2p once with the given ops, saving to save_dir. Returns the
    loaded stat list, ops dict, and path to the plane0 directory.

    If ``use_cache`` is True and ``save_dir`` already contains a completed
    suite2p run whose ops match this pass's ops, the cached output is
    returned without re-invoking suite2p.

    If ``shared_plane0`` is provided (a plane0 folder that already
    contains a registered ``data.bin`` and registration-complete
    ``ops.npy``), the pass reuses that binary instead of re-registering:
    it hardlinks ``data.bin`` into its own plane0, writes a detection-only
    ops.npy with ``do_registration=0``, and invokes ``suite2p.run_plane``
    directly. This skips the TIFF->binary conversion and registration steps
    that otherwise take the bulk of per-pass runtime.
    """
    ops = dict(ops)
    ops['save_path0'] = str(save_dir)
    ops['save_folder'] = 'suite2p'

    if use_cache:
        cached = _load_cached_pass(save_dir, ops, verbose=verbose)
        if cached is not None:
            return cached

    plane0 = Path(save_dir) / 'suite2p' / 'plane0'

    # --- Registration-reuse path ---------------------------------------
    if shared_plane0 is not None:
        plane0.mkdir(parents=True, exist_ok=True)
        # Hardlink the registered binary; data.bin itself is read-only in
        # downstream use, so hardlinking is safe.
        _link_or_copy(Path(shared_plane0) / 'data.bin', plane0 / 'data.bin')

        # Start from the registered ops (contains Ly, Lx, meanImg, yoff,
        # xoff, nframes, refImg, corrXY, ...) and layer detection params
        # on top.
        reg_ops = np.load(Path(shared_plane0) / 'ops.npy',
                          allow_pickle=True).item()
        pass_ops = dict(reg_ops)
        # Override only the detection-relevant keys from the caller's ops.
        _detection_keys = (
            'threshold_scaling', 'max_iterations', 'spatial_scale',
            'sparse_mode', 'high_pass', 'smooth_sigma', 'preclassify',
            'allow_overlap', 'max_overlap', 'tau', 'fs', 'nbinned',
            'batch_size', 'roidetect',
        )
        for k in _detection_keys:
            if k in ops:
                pass_ops[k] = ops[k]
        pass_ops['do_registration'] = 0
        pass_ops['save_path0'] = str(save_dir)
        pass_ops['save_folder'] = 'suite2p'
        pass_ops['reg_file'] = str(plane0 / 'data.bin')
        pass_ops['ops_path'] = str(plane0 / 'ops.npy')
        # roidetect defaults to True; make it explicit.
        pass_ops.setdefault('roidetect', True)

        np.save(plane0 / 'ops.npy', pass_ops, allow_pickle=True)

        if verbose:
            print(f"  > running suite2p detection-only (shared registration): "
                  f"threshold_scaling={pass_ops.get('threshold_scaling')}  "
                  f"spatial_scale={pass_ops.get('spatial_scale')}  "
                  f"max_iterations={pass_ops.get('max_iterations')}")

        _orig_print = (_install_sparsery_roi_cap(hard_cap)
                       if hard_cap and hard_cap > 0 else None)
        try:
            suite2p.run_plane(pass_ops, ops_path=str(plane0 / 'ops.npy'))
        except _RoiHardCapExceeded as e:
            if verbose:
                print(f"  > ABORT: {e} -- returning empty pass")
            ops_loaded = (np.load(plane0 / 'ops.npy',
                                  allow_pickle=True).item()
                          if (plane0 / 'ops.npy').exists() else pass_ops)
            try:
                np.save(plane0 / 'stat.npy',
                        np.array([], dtype=object), allow_pickle=True)
                # drop a breadcrumb so diagnostic tooling can see why this pass
                # is empty (versus a genuinely noise-free pass that found 0)
                (plane0 / 'HARD_CAP_ABORTED.txt').write_text(
                    f"Sparsery aborted at ~{e.count} ROIs "
                    f"(cap={e.cap}).\n"
                )
            except Exception:
                pass
            return [], ops_loaded, plane0
        except ValueError as e:
            if 'no ROIs' in str(e) or 'ROIs were found' in str(e):
                if verbose:
                    print("  > suite2p found 0 ROIs for these params; "
                          "returning empty pass")
                ops_loaded = (np.load(plane0 / 'ops.npy',
                                      allow_pickle=True).item()
                              if (plane0 / 'ops.npy').exists() else pass_ops)
                try:
                    np.save(plane0 / 'stat.npy',
                            np.array([], dtype=object), allow_pickle=True)
                except Exception:
                    pass
                return [], ops_loaded, plane0
            raise
        finally:
            if _orig_print is not None:
                _restore_sparsery_print(_orig_print)

        stat = list(np.load(plane0 / 'stat.npy', allow_pickle=True))
        ops_loaded = np.load(plane0 / 'ops.npy', allow_pickle=True).item()
        return stat, ops_loaded, plane0

    # --- Full run_s2p path (registers + detects) -----------------------
    db = {'data_path': [tiff_folder]}

    if verbose:
        print(f"  > running suite2p: threshold_scaling={ops['threshold_scaling']}  "
              f"max_iterations={ops['max_iterations']}")

    _orig_print = (_install_sparsery_roi_cap(hard_cap)
                   if hard_cap and hard_cap > 0 else None)
    try:
        suite2p.run_s2p(ops=ops, db=db)
    except _RoiHardCapExceeded as e:
        if verbose:
            print(f"  > ABORT: {e} -- returning empty pass")
        ops_path = plane0 / 'ops.npy'
        ops_loaded = (np.load(ops_path, allow_pickle=True).item()
                      if ops_path.exists() else dict(ops))
        try:
            plane0.mkdir(parents=True, exist_ok=True)
            np.save(plane0 / 'stat.npy',
                    np.array([], dtype=object), allow_pickle=True)
            (plane0 / 'HARD_CAP_ABORTED.txt').write_text(
                f"Sparsery aborted at ~{e.count} ROIs "
                f"(cap={e.cap}).\n"
            )
        except Exception:
            pass
        return [], ops_loaded, plane0
    except ValueError as e:
        # Suite2p raises when detection finds zero ROIs. Registration has
        # already written ops.npy to plane0 at this point, so we can still
        # return useful info — just with an empty stat list.
        if 'no ROIs' in str(e) or 'ROIs were found' in str(e):
            if verbose:
                print(f"  > suite2p found 0 ROIs for these params; returning empty pass")
            ops_path = plane0 / 'ops.npy'
            ops_loaded = (np.load(ops_path, allow_pickle=True).item()
                          if ops_path.exists() else dict(ops))
            # Persist an empty stat.npy so the cache check recognizes this
            # as a completed (if barren) pass and doesn't re-run it.
            try:
                plane0.mkdir(parents=True, exist_ok=True)
                np.save(plane0 / 'stat.npy',
                        np.array([], dtype=object), allow_pickle=True)
            except Exception:
                pass
            return [], ops_loaded, plane0
        raise
    finally:
        if _orig_print is not None:
            _restore_sparsery_print(_orig_print)

    stat = list(np.load(plane0 / 'stat.npy', allow_pickle=True))
    ops_loaded = np.load(plane0 / 'ops.npy', allow_pickle=True).item()
    return stat, ops_loaded, plane0


# ============================================================================
# Blob-seeded detection (bypass sparsery)
# ============================================================================

def _stat_from_blob_centers(blobs_yxr, Ly: int, Lx: int,
                            gaussian_sigma_factor: float = 0.7):
    """
    Build a list of suite2p stat dicts from (y, x, radius) blob tuples.

    Each blob becomes a circular ROI whose ``lam`` weights fall off as a
    Gaussian with std = ``gaussian_sigma_factor * radius``. Weights are
    normalized to sum to 1 per ROI (matching suite2p's convention).
    """
    stat = []
    for y, x, r in blobs_yxr:
        r_int = max(1, int(np.ceil(float(r))))
        y_int = int(round(float(y)))
        x_int = int(round(float(x)))
        y0 = max(0, y_int - r_int)
        y1 = min(Ly, y_int + r_int + 1)
        x0 = max(0, x_int - r_int)
        x1 = min(Lx, x_int + r_int + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d2 = (yy - y) ** 2 + (xx - x) ** 2
        mask = d2 <= (float(r)) ** 2
        if not mask.any():
            continue
        sigma = max(1e-3, gaussian_sigma_factor * float(r))
        lam = np.exp(-d2[mask] / (2.0 * sigma ** 2)).astype(np.float32)
        s = lam.sum()
        if s <= 0:
            continue
        lam /= s
        stat.append({
            'ypix': yy[mask].astype(np.int32),
            'xpix': xx[mask].astype(np.int32),
            'lam': lam,
            'med': [y_int, x_int],
        })
    return stat


def _collect_seed_blobs(scale_data, winner: int,
                        all_scales: bool,
                        max_rois: Optional[int]):
    """
    Extract seed blobs from the estimator's scale_data.

    Returns a list of (y, x, radius) tuples, optionally capped at
    ``max_rois`` by mass-score ranking.
    """
    seeds = []  # list of (y, x, radius, score)

    def _add_from(sp):
        d = scale_data[sp]
        for i, (y, x, sigma) in enumerate(d['blobs']):
            r = float(sigma) * float(np.sqrt(2.0))
            seeds.append((float(y), float(x), r,
                          float(d['scores'][i]) if len(d['scores']) else 0.0))

    if all_scales:
        for sp in (1, 2, 3, 4):
            _add_from(sp)
    else:
        _add_from(winner)

    if max_rois is not None and len(seeds) > max_rois:
        seeds.sort(key=lambda t: t[3], reverse=True)
        seeds = seeds[:max_rois]

    return [(y, x, r) for (y, x, r, _s) in seeds]


def run_blob_seeded_detection(config: AdaptiveConfig, save_dir: Path,
                              base_ops: dict, blobs_yxr):
    """
    Register the data with suite2p, then bypass sparsery: build stat from
    the provided blob list and run suite2p's extraction + classification
    + deconvolution manually.

    Returns (stat, ops, plane0) in the same shape as ``run_one_pass``.
    """
    from suite2p.detection.stats import roi_stats
    from suite2p.extraction import extract, dcnv
    from suite2p import classification

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    plane0 = save_dir / 'suite2p' / 'plane0'

    # Step 1: register only (skip sparsery entirely)
    reg_ops = dict(base_ops)
    reg_ops['save_path0'] = str(save_dir)
    reg_ops['save_folder'] = 'suite2p'
    reg_ops['roidetect'] = False

    if config.verbose:
        print(f"  > blob-seeded detection: running suite2p registration "
              f"only (roidetect=False)")

    db = {'data_path': [config.tiff_folder]}
    suite2p.run_s2p(ops=reg_ops, db=db)

    # Load the ops that suite2p wrote (contains Ly, Lx, reg_file, nframes, ...)
    ops = np.load(plane0 / 'ops.npy', allow_pickle=True).item()
    Ly, Lx = int(ops['Ly']), int(ops['Lx'])

    if len(blobs_yxr) == 0:
        if config.verbose:
            print("  > blob-seeded detection: no blobs to seed; returning empty")
        np.save(plane0 / 'stat.npy',
                np.array([], dtype=object), allow_pickle=True)
        return [], ops, plane0

    # Step 2: blobs -> stat (ypix, xpix, lam, med)
    stat = _stat_from_blob_centers(
        blobs_yxr, Ly, Lx,
        gaussian_sigma_factor=config.blob_seed_gaussian_sigma_factor,
    )
    if not stat:
        if config.verbose:
            print("  > blob-seeded detection: all blobs produced empty masks; "
                  "returning empty")
        np.save(plane0 / 'stat.npy',
                np.array([], dtype=object), allow_pickle=True)
        return [], ops, plane0

    # Step 3: enrich stat with suite2p's derived metrics (radius, compact, ...)
    median_diam = int(round(2.0 * float(np.median([r for (_y, _x, r) in blobs_yxr]))))
    median_diam = max(3, median_diam)
    stat = roi_stats(stat, Ly, Lx, diameter=median_diam)
    stat_arr = np.array(stat, dtype=object)

    if config.verbose:
        print(f"  > blob-seeded detection: {len(stat)} ROIs seeded from blobs; "
              f"extracting traces")

    # Step 4: extract fluorescence traces from registered binary
    stat_arr, F, Fneu, F_chan2, Fneu_chan2 = extract.create_masks_and_extract(
        ops, stat_arr
    )

    # Step 5: deconvolution (OASIS)
    # Matches suite2p's default: baseline-correct F - 0.7*Fneu, high-pass,
    # then oasis.
    tau = float(ops.get('tau', 1.0))
    fs = float(ops.get('fs', 15.0))
    neucoeff = float(ops.get('neucoeff', 0.7))
    F_sub = F - neucoeff * Fneu
    F_pp = dcnv.preprocess(
        F=F_sub, baseline=ops.get('baseline', 'maximin'),
        win_baseline=float(ops.get('win_baseline', 60.0)),
        sig_baseline=float(ops.get('sig_baseline', 10.0)),
        fs=fs,
        prctile_baseline=float(ops.get('prctile_baseline', 8.0)),
    )
    batch_size = int(ops.get('batch_size', 3000))
    spks = dcnv.oasis(F=F_pp, batch_size=batch_size, tau=tau, fs=fs)

    # Step 6: classifier
    classfile = classification.builtin_classfile
    try:
        iscell = classification.classify(stat=stat_arr, classfile=classfile)
    except Exception as e:
        if config.verbose:
            print(f"  > blob-seeded detection: classifier failed ({e}); "
                  f"marking all ROIs as cells")
        iscell = np.ones((len(stat_arr), 2), dtype=np.float32)

    # Step 7: save outputs in the standard suite2p plane0 layout
    np.save(plane0 / 'stat.npy', stat_arr, allow_pickle=True)
    np.save(plane0 / 'F.npy', F)
    np.save(plane0 / 'Fneu.npy', Fneu)
    np.save(plane0 / 'spks.npy', spks)
    np.save(plane0 / 'iscell.npy', iscell)
    # Re-save ops (create_masks_and_extract may have mutated it)
    np.save(plane0 / 'ops.npy', ops, allow_pickle=True)

    if config.verbose:
        n_accept = int((iscell[:, 0] > 0).sum()) if iscell.ndim == 2 else int((iscell > 0).sum())
        print(f"  > blob-seeded detection: {len(stat_arr)} ROIs, "
              f"{n_accept} classified as cells")

    return list(stat_arr), ops, plane0


def run_residual_augmentation(config: AdaptiveConfig, base_stat, base_ops,
                              base_plane0: Path, shared_plane0: Path,
                              save_dir: Path):
    """
    Second detection stage: seed ROIs at blob centers in the residual image
    (mean image minus existing ROI pixels), combine with ``base_stat``,
    extract fluorescence + deconvolve + classify the union, and save all
    outputs to a new plane0.

    This is the final recovery step when sparsery converges but leaves
    obvious soma-like blobs in the residual. Classifier acts as the junk
    filter via ``augmentation_min_iscell_prob``.

    Returns (stat_out, ops_out, aug_plane0).
    """
    from suite2p.detection.stats import roi_stats
    from suite2p.extraction import extract, dcnv
    from suite2p import classification

    save_dir = Path(save_dir)
    aug_plane0 = save_dir / 'suite2p' / 'plane0'
    aug_plane0.mkdir(parents=True, exist_ok=True)

    Ly, Lx = int(base_ops['Ly']), int(base_ops['Lx'])

    # --- 1. Detect blobs in residual image ----------------------------
    roi_mask = build_roi_pixel_mask(base_stat, Ly, Lx)
    residual, _ = compute_residual_image(base_ops, roi_mask)
    res_blobs = detect_residual_blobs(residual, config)  # (y, x, r, val)

    if config.verbose:
        print(f"  > residual-blob augmentation: {len(res_blobs)} candidate "
              f"blobs detected in residual")

    if not res_blobs:
        if config.verbose:
            print("  > no residual blobs; skipping augmentation")
        return list(base_stat), base_ops, base_plane0

    # --- 2. Build aug ops with reused registration --------------------
    _link_or_copy(Path(shared_plane0) / 'data.bin', aug_plane0 / 'data.bin')
    reg_ops = np.load(Path(shared_plane0) / 'ops.npy',
                      allow_pickle=True).item()
    aug_ops = dict(reg_ops)
    # Carry forward sample-specific params from base_ops
    for k in ('tau', 'fs', 'neucoeff', 'baseline', 'win_baseline',
              'sig_baseline', 'prctile_baseline', 'batch_size'):
        if k in base_ops:
            aug_ops[k] = base_ops[k]
    aug_ops['save_path0'] = str(save_dir)
    aug_ops['save_folder'] = 'suite2p'
    aug_ops['reg_file'] = str(aug_plane0 / 'data.bin')
    aug_ops['ops_path'] = str(aug_plane0 / 'ops.npy')
    aug_ops['do_registration'] = 0
    aug_ops['Ly'] = Ly
    aug_ops['Lx'] = Lx

    # --- 3. Build combined stat (base + residual blobs) ---------------
    # Strip base_stat to raw fields so roi_stats can re-enrich the union.
    # Tag each ROI with '_is_base_roi' so we can distinguish base from
    # residual-seeded AFTER suite2p's create_masks_and_extract (which
    # reorders/drops ROIs and would otherwise break an index-based split).
    combined_raw = []
    for s in base_stat:
        combined_raw.append({
            'ypix': np.asarray(s['ypix']).astype(np.int32),
            'xpix': np.asarray(s['xpix']).astype(np.int32),
            'lam': np.asarray(s['lam']).astype(np.float32),
            'med': list(s.get('med', [int(np.median(s['ypix'])),
                                       int(np.median(s['xpix']))])),
            '_is_base_roi': True,
        })
    n_base = len(combined_raw)

    blobs_yxr = [(y, x, r) for (y, x, r, _v) in res_blobs]
    residual_stat = _stat_from_blob_centers(
        blobs_yxr, Ly, Lx,
        gaussian_sigma_factor=config.blob_seed_gaussian_sigma_factor,
    )
    for s in residual_stat:
        s['_is_base_roi'] = False
    combined_raw.extend(residual_stat)
    n_new = len(residual_stat)

    if config.verbose:
        print(f"  > augmented stat: {n_base} base + {n_new} residual-seeded "
              f"= {len(combined_raw)} total")

    # --- 4. Enrich via roi_stats with median diameter -----------------
    median_diam = int(round(2.0 * float(np.median([r for (_y, _x, r)
                                                   in blobs_yxr]))))
    median_diam = max(3, median_diam)
    combined_enriched = roi_stats(combined_raw, Ly, Lx, diameter=median_diam)
    # roi_stats copies dict fields across, so the '_is_base_roi' tag survives.
    # Assert sanity just in case.
    for i, s in enumerate(combined_enriched):
        if '_is_base_roi' not in s:
            s['_is_base_roi'] = (i < n_base)
    stat_arr = np.array(combined_enriched, dtype=object)

    # --- 5. Extract fluorescence traces -------------------------------
    if config.verbose:
        print(f"  > extracting traces for {len(stat_arr)} ROIs")
    stat_arr, F, Fneu, F_chan2, Fneu_chan2 = extract.create_masks_and_extract(
        aug_ops, stat_arr
    )

    # --- 6. Deconvolve (OASIS) ----------------------------------------
    tau = float(aug_ops.get('tau', 1.0))
    fs = float(aug_ops.get('fs', 15.0))
    neucoeff = float(aug_ops.get('neucoeff', 0.7))
    F_sub = F - neucoeff * Fneu
    F_pp = dcnv.preprocess(
        F=F_sub, baseline=aug_ops.get('baseline', 'maximin'),
        win_baseline=float(aug_ops.get('win_baseline', 60.0)),
        sig_baseline=float(aug_ops.get('sig_baseline', 10.0)),
        fs=fs,
        prctile_baseline=float(aug_ops.get('prctile_baseline', 8.0)),
    )
    batch_size = int(aug_ops.get('batch_size', 3000))
    spks = dcnv.oasis(F=F_pp, batch_size=batch_size, tau=tau, fs=fs)

    # --- 7. Classify --------------------------------------------------
    try:
        iscell = classification.classify(
            stat=stat_arr, classfile=classification.builtin_classfile
        )
    except Exception as e:
        if config.verbose:
            print(f"  > classifier failed ({e}); marking all as cells")
        iscell = np.ones((len(stat_arr), 2), dtype=np.float32)

    # --- 8. Apply augmentation filter ---------------------------------
    # Original base ROIs are always kept (they already passed pass-0
    # classification). Only the newly-seeded residual ROIs need to beat
    # the ``augmentation_min_iscell_prob`` threshold. We distinguish via
    # the '_is_base_roi' tag, not index — because create_masks_and_extract
    # can reorder or drop ROIs, making an index-based split unreliable.
    prob_threshold = float(config.augmentation_min_iscell_prob)
    is_base = np.array([bool(s.get('_is_base_roi', False))
                        for s in stat_arr], dtype=bool)
    keep = np.ones(len(stat_arr), dtype=bool)
    if iscell.ndim == 2 and len(iscell) == len(stat_arr):
        probs = iscell[:, 1] if iscell.shape[1] >= 2 else iscell[:, 0]
        drop_mask = (~is_base) & (probs < prob_threshold)
        keep[drop_mask] = False
    n_dropped = int((~keep & ~is_base).sum())
    n_residual_after_extract = int((~is_base).sum())
    if config.verbose:
        print(f"  > post-extract: {int(is_base.sum())} base + "
              f"{n_residual_after_extract} residual = {len(stat_arr)} "
              f"(was {n_base} base + {n_new} residual before extract)")
        print(f"  > augmentation filter (prob >= {prob_threshold}): "
              f"dropped {n_dropped}/{n_residual_after_extract} residual-seeded ROIs")

    stat_arr = stat_arr[keep]
    F = F[keep]
    Fneu = Fneu[keep]
    spks = spks[keep]
    iscell = iscell[keep]

    # --- 9. Save standard suite2p outputs -----------------------------
    np.save(aug_plane0 / 'stat.npy', stat_arr, allow_pickle=True)
    np.save(aug_plane0 / 'F.npy', F)
    np.save(aug_plane0 / 'Fneu.npy', Fneu)
    np.save(aug_plane0 / 'spks.npy', spks)
    np.save(aug_plane0 / 'iscell.npy', iscell)
    np.save(aug_plane0 / 'ops.npy', aug_ops, allow_pickle=True)

    if config.verbose:
        n_accept = int((iscell[:, 0] > 0).sum()) if iscell.ndim == 2 else int((iscell > 0).sum())
        print(f"  > residual-blob augmentation: final {len(stat_arr)} ROIs "
              f"({n_accept} classified as cells)")

    return list(stat_arr), aug_ops, aug_plane0


# ============================================================================
# Regime diagnosis
# ============================================================================

def diagnose_over_detection(stat, plane0: Path, config: AdaptiveConfig):
    """
    Decide whether a pass has locked onto the wrong spatial scale and is
    detecting a flood of tiny non-cell ROIs.

    Triggers when either:
      - Raw ROI count exceeds the configured ceiling, AND the Suite2p
        classifier kept only a small fraction (iscell ratio is low), OR
      - Median ROI pixel area is very small (below configured floor)

    Returns (is_over_detection: bool, reason: str, stats: dict).
    """
    n_total = len(stat)

    areas = np.array([np.asarray(s['xpix']).size for s in stat], dtype=float)
    median_area = float(np.median(areas)) if areas.size else 0.0

    iscell_path = plane0 / 'iscell.npy'
    if iscell_path.exists():
        iscell = np.load(iscell_path, allow_pickle=True)
        if iscell.ndim == 2:
            iscell = iscell[:, 0]
        n_accepted = int((iscell > 0).sum())
        accept_ratio = n_accepted / max(1, n_total)
    else:
        n_accepted = -1
        accept_ratio = -1.0

    diag = {
        'n_total': n_total,
        'n_accepted': n_accepted,
        'accept_ratio': accept_ratio,
        'median_area_px': median_area,
    }

    # Condition 1: huge total count AND classifier rejected most of them
    bad_count = (n_total >= config.over_detection_roi_count and
                 0 <= accept_ratio < config.over_detection_iscell_ratio)
    # Condition 2: median ROI area implausibly small (sub-soma)
    bad_size = (median_area > 0 and median_area < config.over_detection_median_area_px)

    if bad_count and bad_size:
        reason = (f"ROI count {n_total} with only {accept_ratio:.1%} classifier "
                  f"acceptance AND median area {median_area:.0f}px below "
                  f"{config.over_detection_median_area_px}px floor")
        return True, reason, diag
    if bad_count:
        reason = (f"ROI count {n_total} with only {accept_ratio:.1%} classifier "
                  f"acceptance (threshold: >={config.over_detection_roi_count} "
                  f"AND <{config.over_detection_iscell_ratio:.0%})")
        return True, reason, diag
    if bad_size:
        reason = (f"median ROI area {median_area:.0f}px below "
                  f"{config.over_detection_median_area_px}px floor")
        return True, reason, diag

    return False, "within normal ranges", diag


# ============================================================================
# Adaptive loop
# ============================================================================

def _next_threshold(current: float, config: AdaptiveConfig,
                    aggressive: bool = False) -> float:
    """
    Compute the next threshold_scaling from the previous pass's value.

    Multiplicative mode (default): next = current * threshold_decay_factor
    Additive mode:                  next = current - threshold_decay_offset

    ``aggressive=True`` applies the step TWICE — used when the previous
    pass returned 0 ROIs, since the normal step clearly wasn't enough.
    The result is always clamped to ``min_threshold_scaling``.
    """
    steps = 2 if aggressive else 1
    nxt = current
    for _ in range(steps):
        if config.threshold_decay_mode == 'additive':
            nxt = nxt - config.threshold_decay_offset
        else:  # multiplicative (default)
            nxt = nxt * config.threshold_decay_factor
    return max(nxt, config.min_threshold_scaling)


def _is_speckle_pass(config, stat_pass, baseline_stat):
    """
    Return (rejected, reason) describing whether ``stat_pass`` looks like a
    speckle explosion relative to ``baseline_stat`` (typically pass 0).

    Used by both the decay schedule and the binary-search schedule so the
    rejection criterion is identical across modes.
    """
    if len(stat_pass) == 0:
        return False, None
    pass_areas = np.array([len(s['ypix']) for s in stat_pass])
    pass_median_area = float(np.median(pass_areas))
    baseline_areas = (np.array([len(s['ypix']) for s in baseline_stat])
                     if len(baseline_stat) else np.array([]))
    baseline_median = (float(np.median(baseline_areas))
                       if len(baseline_areas) else 0.0)

    if (config.speckle_guard_min_area_px > 0
            and pass_median_area < config.speckle_guard_min_area_px):
        return True, (f"median area {pass_median_area:.1f}px < "
                      f"speckle_guard_min_area_px="
                      f"{config.speckle_guard_min_area_px}")
    if (config.speckle_guard_area_ratio > 0
            and baseline_median > 0
            and pass_median_area
                < config.speckle_guard_area_ratio * baseline_median):
        return True, (f"median area {pass_median_area:.1f}px < "
                      f"{config.speckle_guard_area_ratio} x pass0_median="
                      f"{baseline_median:.1f}px")
    return False, None


def _run_binary_search_schedule(config, base_ops, save_root,
                                initial_stat, initial_ops, initial_plane0,
                                initial_thr, initial_max_iter,
                                shared_plane0: Optional[Path] = None):
    """
    Binary-search threshold refinement.

    Idea: treat ``threshold_scaling`` as a parameter to fit. The speckle
    guard classifies each pass as "safe" (real cells) or "speckle"
    (noise clumps). We maintain a bracket ``[lo, hi]`` where:
      - ``lo`` is the lowest threshold known to produce real cells
        (starts at ``initial_thr``, set from pass 0).
      - ``hi`` is the highest threshold known to produce speckles
        (initially None — unknown).

    Each probe picks the midpoint of the bracket (or one step below ``lo``
    if ``hi`` is None yet). A safe probe tightens ``lo`` up and its ROIs
    get merged in. A speckle probe tightens ``hi`` down and its ROIs are
    discarded.

    Stops when:
      - hi - lo <= precision*(initial_thr - min_thr), OR
      - a safe probe added < min_new_rois_per_pass new ROIs, OR
      - max_under_detection_passes reached, OR
      - residual blob count below min_residual_blobs.
    """
    passes = [{
        'index': 0,
        'regime': 'binary_search',
        'threshold_scaling': initial_thr,
        'max_iterations': initial_max_iter,
        'n_detected_this_pass': len(initial_stat),
        'n_added_after_merge': len(initial_stat),
        'n_merged_total': len(initial_stat),
        'plane0': str(initial_plane0),
        'role': 'baseline',
    }]
    merged_stat = list(initial_stat)
    final_ops = initial_ops
    final_plane0 = initial_plane0

    # Initial residual audit
    Ly, Lx = initial_ops['Ly'], initial_ops['Lx']
    roi_mask = build_roi_pixel_mask(merged_stat, Ly, Lx)
    residual, _ = compute_residual_image(initial_ops, roi_mask)
    blobs = detect_residual_blobs(residual, config)
    passes[0]['n_residual_blobs'] = len(blobs)

    if config.verbose:
        print(f"  > residual blobs after pass 0: {len(blobs)}")

    if len(blobs) <= config.min_residual_blobs:
        if config.verbose:
            print("  > stopping: pass 0 already captures most cells")
        return merged_stat, passes, final_ops, final_plane0

    # --- bracket setup ---
    lo = float(initial_thr)
    hi: Optional[float] = None   # unknown upper (speckle) bound
    min_thr = float(config.min_threshold_scaling)
    bracket_width_initial = max(initial_thr - min_thr, 1e-6)
    precision = config.binary_search_precision * bracket_width_initial

    # First probe: one decay step below lo. If there's no known speckle
    # threshold yet, go aggressive enough to actually find one.
    first_step = max(
        config.threshold_decay_offset,
        (initial_thr - min_thr) * 0.5,
    )
    next_thr = max(min_thr, lo - first_step)

    tried: set = set()  # avoid re-running identical thresholds

    for j in range(1, config.max_under_detection_passes + 1):
        thr = round(float(next_thr), 4)
        if thr <= min_thr + 1e-9 and hi is None:
            if config.verbose:
                print(f"  > stopping: hit min_threshold_scaling ({min_thr}) "
                      f"without finding a speckle boundary")
            break
        if thr in tried:
            if config.verbose:
                print(f"  > stopping: threshold {thr:.3f} already tried "
                      f"(bracket collapsed)")
            break
        tried.add(thr)

        pass_dir = save_root / f"pass{j:02d}_thr{thr:.2f}"
        pass_dir.mkdir(parents=True, exist_ok=True)

        if config.verbose:
            bracket = f"[{lo:.3f}, {'?' if hi is None else f'{hi:.3f}'}]"
            print(f"\n[pass {j}] binary_search probe at thr={thr:.3f}  "
                  f"bracket={bracket}  max_iterations={initial_max_iter}")

        ops = dict(base_ops)
        ops['threshold_scaling'] = thr
        ops['max_iterations'] = initial_max_iter

        stat_pass, ops_out, plane0 = run_one_pass(
            config.tiff_folder, pass_dir, ops,
            verbose=config.verbose, use_cache=config.use_cache,
            shared_plane0=shared_plane0,
            hard_cap=config.max_rois_hard_cap,
        )

        # --- classify this probe via speckle guard ---
        rejected, reason = _is_speckle_pass(config, stat_pass, initial_stat)

        # Check if the pass was aborted by the ROI hard cap. In that case we
        # know it was a noise blowup even if stat_pass came back empty.
        hard_cap_aborted = (plane0 / 'HARD_CAP_ABORTED.txt').exists()

        if rejected or len(stat_pass) == 0:
            # Speckle (or empty or cap-aborted) probe: tighten hi, discard.
            # An empty-ROI result is treated the same as speckle for
            # bracket-tightening purposes — it means this threshold isn't
            # useful, but a slightly higher one might be. A hard-cap abort
            # is the strongest speckle signal possible.
            if hard_cap_aborted:
                tag = 'hard_cap_aborted'
                hard_cap_reason = (plane0 / 'HARD_CAP_ABORTED.txt').read_text().strip()
            elif rejected:
                tag = 'speckle'
            else:
                tag = 'empty'
            hi = thr
            if config.verbose:
                if hard_cap_aborted:
                    print(f"  > HARD CAP HIT: {hard_cap_reason}")
                elif rejected:
                    print(f"  > SPECKLE probe rejected: {reason}")
                else:
                    print(f"  > EMPTY probe (0 ROIs at thr={thr:.3f})")
            passes.append({
                'index': j,
                'regime': 'binary_search',
                'threshold_scaling': thr,
                'max_iterations': initial_max_iter,
                'n_detected_this_pass': len(stat_pass),
                'n_added_after_merge': 0,
                'n_merged_total': len(merged_stat),
                'role': tag,
                'rejected': rejected or hard_cap_aborted,
                'rejection_reason': (hard_cap_reason if hard_cap_aborted
                                     else reason),
                'plane0': str(plane0),
            })
        else:
            # Safe probe: tighten lo, merge ROIs.
            merged_stat, n_added = merge_roi_lists(
                merged_stat, stat_pass,
                iou_threshold=config.iou_dedup_threshold,
            )
            final_ops = ops_out
            final_plane0 = plane0
            lo = thr

            roi_mask = build_roi_pixel_mask(merged_stat, Ly, Lx)
            residual, _ = compute_residual_image(ops_out, roi_mask)
            blobs = detect_residual_blobs(residual, config)

            passes.append({
                'index': j,
                'regime': 'binary_search',
                'threshold_scaling': thr,
                'max_iterations': initial_max_iter,
                'n_detected_this_pass': len(stat_pass),
                'n_added_after_merge': n_added,
                'n_merged_total': len(merged_stat),
                'n_residual_blobs': len(blobs),
                'role': 'safe',
                'plane0': str(plane0),
            })
            if config.verbose:
                print(f"  > SAFE probe: detected={len(stat_pass)}  "
                      f"added={n_added}  total={len(merged_stat)}  "
                      f"residual_blobs={len(blobs)}")

            if len(blobs) <= config.min_residual_blobs:
                if config.verbose:
                    print("  > stopping: residual blob count below tolerance")
                break
            if n_added < config.min_new_rois_per_pass:
                if config.verbose:
                    print("  > stopping: probe added few new ROIs "
                          "(diminishing returns)")
                break

        # --- decide next probe ---
        if hi is not None and hi - lo <= precision:
            if config.verbose:
                print(f"  > stopping: bracket width {hi-lo:.4f} <= "
                      f"precision {precision:.4f}")
            break
        if hi is None:
            # No speckle boundary found yet — keep probing lower.
            next_thr = max(min_thr, lo - first_step)
        else:
            # Midpoint of current bracket.
            next_thr = (lo + hi) / 2.0

    return merged_stat, passes, final_ops, final_plane0


def _run_under_detection_schedule(config, base_ops, save_root,
                                  initial_stat, initial_ops, initial_plane0,
                                  initial_thr, initial_max_iter,
                                  shared_plane0: Optional[Path] = None):
    """
    Under-detection regime: start from the already-completed pass 0 and keep
    lowering threshold_scaling dynamically (multiplicative or additive,
    controlled by AdaptiveConfig) until one of the stopping criteria is
    met:
      - residual blob count below tolerance
      - pass added fewer new ROIs than ``min_new_rois_per_pass``
      - threshold hit its floor (``min_threshold_scaling``)
      - pass index hit ``max_under_detection_passes``
    """
    passes = [{
        'index': 0,
        'regime': 'under_detection',
        'threshold_scaling': initial_thr,
        'max_iterations': initial_max_iter,
        'n_detected_this_pass': len(initial_stat),
        'n_added_after_merge': len(initial_stat),
        'n_merged_total': len(initial_stat),
        'plane0': str(initial_plane0),
    }]
    merged_stat = list(initial_stat)
    final_ops = initial_ops
    final_plane0 = initial_plane0

    # Initial residual audit on pass 0
    Ly, Lx = initial_ops['Ly'], initial_ops['Lx']
    roi_mask = build_roi_pixel_mask(merged_stat, Ly, Lx)
    residual, _ = compute_residual_image(initial_ops, roi_mask)
    blobs = detect_residual_blobs(residual, config)
    passes[0]['n_residual_blobs'] = len(blobs)

    if config.verbose:
        print(f"  > residual blobs after pass 0: {len(blobs)}")

    if len(blobs) <= config.min_residual_blobs:
        if config.verbose:
            print("  > stopping: pass 0 already captures most cells")
        return merged_stat, passes, final_ops, final_plane0

    # Dynamically derive each subsequent threshold from the previous one.
    # Extra kick when the previous pass returned 0 ROIs.
    current_thr = initial_thr
    prev_n_detected = len(initial_stat)

    for j in range(1, config.max_under_detection_passes + 1):
        aggressive = (prev_n_detected == 0)
        next_thr = _next_threshold(current_thr, config, aggressive=aggressive)

        if next_thr >= current_thr - 1e-9:
            if config.verbose:
                print(f"  > stopping: threshold floor "
                      f"({config.min_threshold_scaling}) reached")
            break
        current_thr = next_thr
        max_iter = config.default_max_iterations

        pass_dir = save_root / f"pass{j:02d}_thr{current_thr:.2f}"
        pass_dir.mkdir(parents=True, exist_ok=True)

        if config.verbose:
            step_note = (f"  (aggressive: previous pass = 0 ROIs)"
                         if aggressive else "")
            print(f"\n[pass {j}] under_detection regime  "
                  f"threshold_scaling={current_thr:.4g}  "
                  f"max_iterations={max_iter}{step_note}")

        # base_ops['spatial_scale'] already reflects the pass-0 fallback
        # result, so subsequent passes inherit the scale that actually
        # produced ROIs. No override needed here.
        ops = dict(base_ops)
        ops['threshold_scaling'] = current_thr
        ops['max_iterations'] = max_iter

        stat_pass, ops_out, plane0 = run_one_pass(
            config.tiff_folder, pass_dir, ops,
            verbose=config.verbose, use_cache=config.use_cache,
            shared_plane0=shared_plane0,
            hard_cap=config.max_rois_hard_cap,
        )
        prev_n_detected = len(stat_pass)

        # --- Speckle guard ---
        # If this pass's ROIs are much smaller than pass 0's (classic sign
        # of sparsery locking onto noise clumps at a lowered threshold),
        # reject this pass outright — don't merge, don't update final_ops,
        # and stop the schedule. The merged state before this pass is kept.
        pass_rejected = False
        rejection_reason = None
        if len(stat_pass) > 0:
            pass_areas = np.array([len(s['ypix']) for s in stat_pass])
            pass_median_area = float(np.median(pass_areas))
            baseline_areas = np.array([len(s['ypix']) for s in initial_stat]) \
                if len(initial_stat) else np.array([])
            baseline_median = (float(np.median(baseline_areas))
                               if len(baseline_areas) else 0.0)
            if (config.speckle_guard_min_area_px > 0
                    and pass_median_area < config.speckle_guard_min_area_px):
                pass_rejected = True
                rejection_reason = (
                    f"median area {pass_median_area:.1f}px < "
                    f"speckle_guard_min_area_px={config.speckle_guard_min_area_px}"
                )
            elif (config.speckle_guard_area_ratio > 0
                  and baseline_median > 0
                  and pass_median_area
                      < config.speckle_guard_area_ratio * baseline_median):
                pass_rejected = True
                rejection_reason = (
                    f"median area {pass_median_area:.1f}px < "
                    f"{config.speckle_guard_area_ratio} x pass0_median="
                    f"{baseline_median:.1f}px"
                )

        if pass_rejected:
            passes.append({
                'index': j,
                'regime': 'under_detection',
                'threshold_scaling': current_thr,
                'max_iterations': max_iter,
                'n_detected_this_pass': len(stat_pass),
                'n_added_after_merge': 0,
                'n_merged_total': len(merged_stat),
                'rejected': True,
                'rejection_reason': rejection_reason,
                'plane0': str(plane0),
            })
            if config.verbose:
                print(f"  > REJECTED pass: {rejection_reason}")
                print(f"  > stopping: speckle guard tripped; keeping previous merge")
            break

        final_ops = ops_out
        final_plane0 = plane0

        merged_stat, n_added = merge_roi_lists(
            merged_stat, stat_pass, iou_threshold=config.iou_dedup_threshold
        )

        Ly, Lx = ops_out['Ly'], ops_out['Lx']
        roi_mask = build_roi_pixel_mask(merged_stat, Ly, Lx)
        residual, _ = compute_residual_image(ops_out, roi_mask)
        blobs = detect_residual_blobs(residual, config)

        passes.append({
            'index': j,
            'regime': 'under_detection',
            'threshold_scaling': current_thr,
            'max_iterations': max_iter,
            'n_detected_this_pass': len(stat_pass),
            'n_added_after_merge': n_added,
            'n_merged_total': len(merged_stat),
            'n_residual_blobs': len(blobs),
            'plane0': str(plane0),
        })

        if config.verbose:
            print(f"  > detected: {len(stat_pass)}  added: {n_added}  "
                  f"total: {len(merged_stat)}  residual_blobs: {len(blobs)}")

        if len(blobs) <= config.min_residual_blobs:
            if config.verbose:
                print("  > stopping: residual blob count below tolerance")
            break
        # Only count "few new ROIs" as a stop when the pass actually
        # produced detections — a 0-ROI pass means threshold was too high,
        # which should trigger an aggressive retry, not termination.
        if len(stat_pass) > 0 and n_added < config.min_new_rois_per_pass:
            if config.verbose:
                print("  > stopping: pass added few new ROIs")
            break

    return merged_stat, passes, final_ops, final_plane0


def _run_over_detection_schedule(config, base_ops, save_root,
                                 shared_plane0: Optional[Path] = None):
    """
    Over-detection regime: pass 0 is discarded, and we rerun using explicit
    spatial_scale and tightened max_overlap. Each pass in the schedule is
    kept in isolation; the best pass (fewest ROIs but most accepted by the
    classifier) wins.
    """
    passes = []
    candidates = []  # list of (score, merged_stat, ops, plane0, pass_dict)

    for j, (sp_scale, thr_scale, max_overlap) in enumerate(
        config.spatial_scale_schedule
    ):
        pass_dir = save_root / f"pass_sc{j:02d}_scale{sp_scale}_thr{thr_scale:.2f}"
        pass_dir.mkdir(parents=True, exist_ok=True)

        if config.verbose:
            print(f"\n[spatial pass {j}] over_detection regime  "
                  f"spatial_scale={sp_scale}  threshold_scaling={thr_scale}  "
                  f"max_overlap={max_overlap}")

        ops = dict(base_ops)
        ops['spatial_scale'] = sp_scale
        ops['threshold_scaling'] = thr_scale
        ops['max_overlap'] = max_overlap
        # Keep max_iterations at a moderate value; over-detection doesn't need
        # deep iteration, it needs the right scale
        ops['max_iterations'] = 100

        stat_pass, ops_out, plane0 = run_one_pass(
            config.tiff_folder, pass_dir, ops,
            verbose=config.verbose, use_cache=config.use_cache,
            shared_plane0=shared_plane0,
            hard_cap=config.max_rois_hard_cap,
        )

        # Evaluate this pass
        iscell_path = plane0 / 'iscell.npy'
        if iscell_path.exists():
            iscell = np.load(iscell_path, allow_pickle=True)
            if iscell.ndim == 2:
                iscell = iscell[:, 0]
            n_accepted = int((iscell > 0).sum())
        else:
            n_accepted = len(stat_pass)

        areas = np.array([np.asarray(s['xpix']).size for s in stat_pass],
                         dtype=float)
        median_area = float(np.median(areas)) if areas.size else 0.0

        # Score: accepted count weighted by plausibility of soma-size ROIs.
        # We want high n_accepted AND median_area above the floor.
        size_penalty = 1.0 if median_area >= config.over_detection_median_area_px else 0.25
        score = n_accepted * size_penalty

        pass_info = {
            'index': j,
            'regime': 'over_detection',
            'spatial_scale': sp_scale,
            'threshold_scaling': thr_scale,
            'max_overlap': max_overlap,
            'n_detected_this_pass': len(stat_pass),
            'n_accepted': n_accepted,
            'median_area_px': median_area,
            'score': score,
            'plane0': str(plane0),
        }
        passes.append(pass_info)
        candidates.append((score, list(stat_pass), ops_out, plane0, pass_info))

        if config.verbose:
            print(f"  > detected: {len(stat_pass)}  accepted: {n_accepted}  "
                  f"median_area: {median_area:.0f}px  score: {score:.1f}")

    # Pick the best pass by score
    candidates.sort(key=lambda t: t[0], reverse=True)
    best_score, best_stat, best_ops, best_plane0, best_info = candidates[0]

    if config.verbose:
        print(f"\n  > best over-detection pass: spatial_scale="
              f"{best_info['spatial_scale']} thr={best_info['threshold_scaling']} "
              f"(accepted={best_info['n_accepted']}, "
              f"median_area={best_info['median_area_px']:.0f}px)")

    # Mark winner in passes list
    for p in passes:
        p['selected'] = (p is best_info)

    return best_stat, passes, best_ops, best_plane0


def adaptive_detect(config: AdaptiveConfig):
    """
    Main entry point. Runs the adaptive detection pipeline end-to-end.

    Flow:
      1. Run pass 0 (conservative threshold) and diagnose for over-detection.
      2. Always run the under-detection schedule first, merging ROIs from
         pass 0 and progressively lowering threshold_scaling. This is a
         strict superset of pass 0's output — we never lose cells here.
      3. Re-diagnose the merged result. If it *still* looks over-detected
         (too many sub-soma micro-ROIs, low classifier acceptance), fall
         back to the spatial_scale schedule and keep whichever result is
         more plausible.

    Returns a dict with:
        merged_stat      list of deduplicated ROI stat dicts
        passes           per-pass diagnostics (includes 'regime' key)
        regime           'under_detection' | 'over_detection_fallback' | 'under_detection_kept'
        diagnosis        dict from diagnose_over_detection() on pass 0
        final_ops        ops from the selected Suite2p invocation
        final_plane0     Path to the selected pass's plane0 output
        save_folder      root save folder
    """
    if not config.tiff_folder:
        raise ValueError("config.tiff_folder must be set")
    if not config.save_folder:
        raise ValueError("config.save_folder must be set")

    save_root = Path(config.save_folder)
    save_root.mkdir(parents=True, exist_ok=True)

    base_ops, scale_data = load_base_ops(config, return_scale_data=True)

    # ---------------- shared registration (one-time, reused) ---------------
    # Run registration exactly once into _shared_reg/, then hardlink
    # data.bin into each pass folder instead of re-registering the TIFF.
    # When seed_pass0_from is provided, prefer that plane0's binary as the
    # shared source (so pass 0 is skipped entirely).
    shared_plane0: Optional[Path] = None
    if config.seed_pass0_from:
        seed_plane0 = Path(config.seed_pass0_from)
        if not (seed_plane0 / 'data.bin').exists():
            raise FileNotFoundError(
                f"seed_pass0_from points at {seed_plane0} but no data.bin "
                f"was found there"
            )
        if not (seed_plane0 / 'ops.npy').exists():
            raise FileNotFoundError(
                f"seed_pass0_from points at {seed_plane0} but no ops.npy "
                f"was found there"
            )
        shared_plane0 = seed_plane0
        if config.verbose:
            print(f"  > using seed pass 0 at {seed_plane0} as shared "
                  f"registration source")
    elif config.reuse_registration:
        shared_plane0 = _get_or_create_shared_registration(config, base_ops)

    # ------------------- blob-seeded detection short-circuit ---------------
    # When enabled, bypass sparsery entirely: feed the pre-flight estimator's
    # blobs to suite2p as pre-computed ROI seeds, then run extraction +
    # classification + deconvolution on those. This is the "we already know
    # where the cells are" path — no under/over-detection schedules run.
    if config.blob_seeded_detection:
        if config.verbose:
            print("\n[blob-seeded detection] bypassing sparsery; seeding ROIs "
                  "from pre-flight LoG blobs")

        if scale_data is None:
            if config.verbose:
                print("  > no scale_data available (estimator disabled or "
                      "failed); re-running estimator for blob extraction")
            est_qc_path = (config.spatial_scale_qc_path
                           or str(Path(config.save_folder) / 'spatial_scale_qc.png')) \
                if config.save_spatial_scale_qc else None
            winner, scale_data = estimate_spatial_scale_from_tiff(
                config.tiff_folder,
                n_frames=config.auto_estimate_n_frames,
                top_n=config.spatial_scale_top_n,
                qc_path=est_qc_path,
                verbose=config.verbose,
                return_scale_data=True,
            )
        else:
            winner = base_ops.get('spatial_scale', 0) or 1

        if scale_data is None:
            raise RuntimeError(
                "blob_seeded_detection=True but spatial-scale estimator "
                "returned no scale_data; cannot seed ROIs"
            )

        # Optional: override the winning scale (e.g. estimator picked s3
        # but real cells are ~8 px diameter so we want s1 blobs).
        seed_scale = (config.blob_seed_scale_override
                      if config.blob_seed_scale_override in (1, 2, 3, 4)
                      else winner)
        if config.verbose and seed_scale != winner:
            print(f"  > overriding blob seed scale: winner was {winner}, "
                  f"using {seed_scale} (blob_seed_scale_override)")

        blobs_yxr = _collect_seed_blobs(
            scale_data, winner=seed_scale,
            all_scales=config.blob_seed_all_scales,
            max_rois=config.blob_seed_max_rois,
        )
        if config.verbose:
            print(f"  > {len(blobs_yxr)} blobs selected as ROI seeds "
                  f"(all_scales={config.blob_seed_all_scales}, "
                  f"seed_scale={seed_scale}, max_rois={config.blob_seed_max_rois})")

        seeded_dir = save_root / 'blob_seeded'
        seeded_stat, seeded_ops, seeded_plane0 = run_blob_seeded_detection(
            config, seeded_dir, base_ops, blobs_yxr,
        )

        # Persist a minimal summary mirroring the normal path so downstream
        # tooling can consume the same dict shape.
        merged_out = save_root / 'merged'
        merged_out.mkdir(exist_ok=True)
        np.save(merged_out / 'stat_merged.npy',
                np.array(seeded_stat, dtype=object), allow_pickle=True)
        np.save(merged_out / 'ops_final.npy', seeded_ops, allow_pickle=True)

        import json
        with open(merged_out / 'pass_summary.json', 'w') as f:
            json.dump({
                'regime': 'blob_seeded',
                'n_seeds': len(blobs_yxr),
                'winner_scale': int(winner),
                'blob_seed_all_scales': bool(config.blob_seed_all_scales),
                'blob_seed_max_rois': config.blob_seed_max_rois,
                'n_rois': len(seeded_stat),
            }, f, indent=2, default=str)

        if config.verbose:
            print(f"\nDone. Regime: blob_seeded. ROI count: {len(seeded_stat)}")
            print(f"Final outputs in: {merged_out}")

        if getattr(config, 'generate_audit_pngs', True):
            try:
                generate_audit_pngs_for_save_folder(
                    str(save_root), config=config, verbose=config.verbose,
                )
            except Exception as e:
                if config.verbose:
                    print(f"[audit] generation failed: {e}")

        return {
            'merged_stat': seeded_stat,
            'passes': [],
            'regime': 'blob_seeded',
            'diagnosis': {'n_seeds': len(blobs_yxr), 'winner_scale': int(winner)},
            'final_ops': seeded_ops,
            'final_plane0': seeded_plane0,
            'save_folder': str(save_root),
        }

    # -------------------------- pass 0 (diagnostic) ------------------------
    # If pass 0 returns 0 ROIs at the estimated spatial_scale, decrement and
    # retry. The LoG estimator tells us where the most "mass" lives in the
    # image, but suite2p's sparsery may need a smaller scale to actually
    # seed ROIs (e.g., our s3 estimate vs sparsery needing s2 because real
    # somata are ~15 px, not ~24 px).
    initial_thr = config.initial_threshold_scaling
    initial_max_iter = config.default_max_iterations

    current_scale = base_ops.get('spatial_scale', 0)
    n_fallbacks_used = 0

    stat0, ops0, plane0_dir = None, None, None

    # --- seed pass 0 from an existing plane0 folder ---
    if config.seed_pass0_from:
        seed_plane0 = Path(config.seed_pass0_from)
        stat0 = list(np.load(seed_plane0 / 'stat.npy', allow_pickle=True))
        ops0 = np.load(seed_plane0 / 'ops.npy', allow_pickle=True).item()
        plane0_dir = seed_plane0
        current_scale = ops0.get('spatial_scale', current_scale)
        initial_thr = ops0.get('threshold_scaling', initial_thr)
        initial_max_iter = ops0.get('max_iterations', initial_max_iter)
        if config.verbose:
            print(f"\n[pass 0 / seeded] loaded {len(stat0)} ROIs from "
                  f"{seed_plane0}")
            print(f"  threshold_scaling={initial_thr}  "
                  f"spatial_scale={current_scale}  "
                  f"max_iterations={initial_max_iter}")
    else:
        while True:
            scale_suffix = f"_sc{current_scale}" if n_fallbacks_used > 0 else ""
            pass0_dir = save_root / f"pass00_thr{initial_thr:.2f}{scale_suffix}"
            pass0_dir.mkdir(parents=True, exist_ok=True)

            if config.verbose:
                print(f"\n[pass 0 / diagnostic] threshold_scaling={initial_thr}  "
                      f"max_iterations={initial_max_iter}  "
                      f"spatial_scale={current_scale}"
                      f"{'  (fallback attempt ' + str(n_fallbacks_used) + ')' if n_fallbacks_used > 0 else ''}")

            ops = dict(base_ops)
            ops['spatial_scale'] = current_scale
            ops['threshold_scaling'] = initial_thr
            ops['max_iterations'] = initial_max_iter

            stat0, ops0, plane0_dir = run_one_pass(
                config.tiff_folder, pass0_dir, ops,
                verbose=config.verbose, use_cache=config.use_cache,
                shared_plane0=shared_plane0,
                hard_cap=config.max_rois_hard_cap,
            )

            if len(stat0) > 0:
                break
            if not config.allow_spatial_scale_fallback:
                break
            if n_fallbacks_used >= config.max_spatial_scale_fallbacks:
                if config.verbose:
                    print(f"  > exhausted {config.max_spatial_scale_fallbacks} "
                          f"spatial_scale fallbacks; continuing with 0 ROIs")
                break
            if current_scale <= 1:
                if config.verbose:
                    print("  > already at smallest spatial_scale (1); "
                          "no further fallback available")
                break

            current_scale -= 1
            n_fallbacks_used += 1
            if config.verbose:
                print(f"  > 0 ROIs at spatial_scale={current_scale + 1}; "
                      f"falling back to spatial_scale={current_scale}")

    # Lock the scale that actually worked into base_ops so all subsequent
    # passes inherit it.
    base_ops['spatial_scale'] = current_scale

    is_over, reason, diag = diagnose_over_detection(stat0, plane0_dir, config)

    if config.verbose:
        print(f"\n  pass 0 diagnosis: n_total={diag['n_total']}, "
              f"n_accepted={diag['n_accepted']}, "
              f"accept_ratio={diag['accept_ratio']:.1%}, "
              f"median_area={diag['median_area_px']:.0f}px")
        print(f"  initial regime guess: "
              f"{'OVER_DETECTION' if is_over else 'UNDER_DETECTION'}  ({reason})")

    # -------------------- optional: stop at pass 0 -------------------------
    # When pass 0 already looks good (user has eyeballed it and the
    # classifier accepts most ROIs), the under-detection schedule at
    # lower thresholds typically lets sparsery lock onto noise speckles
    # rather than real cells. Skip the schedule in that case.
    if not config.run_under_detection_schedule:
        if config.verbose:
            print("\nrun_under_detection_schedule=False; stopping at pass 0")
        merged_out = save_root / 'merged'
        merged_out.mkdir(exist_ok=True)
        np.save(merged_out / 'stat_merged.npy',
                np.array(stat0, dtype=object), allow_pickle=True)
        np.save(merged_out / 'ops_final.npy', ops0, allow_pickle=True)

        import json
        with open(merged_out / 'pass_summary.json', 'w') as f:
            json.dump({
                'regime': 'pass0_only',
                'diagnosis_pass0': diag,
                'passes': [{'pass': 0,
                            'threshold_scaling': initial_thr,
                            'spatial_scale': current_scale,
                            'n_rois': len(stat0)}],
            }, f, indent=2, default=str)

        if config.verbose:
            print(f"\nDone. Regime: pass0_only. ROI count: {len(stat0)}")
            print(f"Final outputs in: {merged_out}")

        if getattr(config, 'generate_audit_pngs', True):
            try:
                generate_audit_pngs_for_save_folder(
                    str(save_root), config=config, verbose=config.verbose,
                )
            except Exception as e:
                if config.verbose:
                    print(f"[audit] generation failed: {e}")

        return {
            'merged_stat': stat0,
            'passes': [],
            'regime': 'pass0_only',
            'diagnosis': diag,
            'final_ops': ops0,
            'final_plane0': plane0_dir,
            'save_folder': str(save_root),
        }

    # -------------------- always try under-detection first ------------------
    schedule_mode = getattr(config, 'threshold_schedule_mode', 'binary_search')
    if config.verbose:
        print(f"\nRunning under-detection schedule first (mode={schedule_mode}, "
              f"merging from pass 0)")

    if schedule_mode == 'binary_search':
        under_stat, under_passes, under_ops, under_plane0 = _run_binary_search_schedule(
            config, base_ops, save_root,
            initial_stat=stat0, initial_ops=ops0, initial_plane0=plane0_dir,
            initial_thr=initial_thr, initial_max_iter=initial_max_iter,
            shared_plane0=shared_plane0,
        )
    else:
        under_stat, under_passes, under_ops, under_plane0 = _run_under_detection_schedule(
            config, base_ops, save_root,
            initial_stat=stat0, initial_ops=ops0, initial_plane0=plane0_dir,
            initial_thr=initial_thr, initial_max_iter=initial_max_iter,
            shared_plane0=shared_plane0,
        )

    # Re-diagnose on the merged result
    is_over_after, reason_after, diag_after = diagnose_over_detection(
        under_stat, under_plane0, config
    )

    if config.verbose:
        print(f"\n  merged diagnosis: n_total={diag_after['n_total']}, "
              f"n_accepted={diag_after['n_accepted']}, "
              f"accept_ratio={diag_after['accept_ratio']:.1%}, "
              f"median_area={diag_after['median_area_px']:.0f}px")

    if not is_over_after:
        regime = 'under_detection'
        merged_stat = under_stat
        passes = under_passes
        final_ops = under_ops
        final_plane0 = under_plane0
    else:
        # ------------------ fall back to over-detection schedule -------------
        if not getattr(config, 'enable_over_detection_fallback', False):
            if config.verbose:
                print(f"\n  merged result looks over-detected ({reason_after}), "
                      f"but over-detection fallback is disabled "
                      f"(enable_over_detection_fallback=False). "
                      f"Trusting downstream cell classifier to filter specks; "
                      f"keeping under-detection merge as-is.")
            regime = 'under_detection'
            merged_stat = under_stat
            passes = under_passes
            final_ops = under_ops
            final_plane0 = under_plane0
        else:
            if config.verbose:
                print(f"\n  merged result still over-detected ({reason_after}); "
                      f"falling back to spatial_scale schedule")

            over_stat, over_passes, over_ops, over_plane0 = _run_over_detection_schedule(
                config, base_ops, save_root,
                shared_plane0=shared_plane0,
            )

            # Keep over-detection only if it actually found ROIs; otherwise the
            # under-detection merge is still the best thing we have.
            if len(over_stat) > 0:
                regime = 'over_detection_fallback'
                merged_stat = over_stat
                final_ops = over_ops
                final_plane0 = over_plane0
            else:
                if config.verbose:
                    print("  > all over-detection passes found 0 ROIs; "
                          "keeping under-detection merged result")
                regime = 'under_detection_kept'
                merged_stat = under_stat
                final_ops = under_ops
                final_plane0 = under_plane0
            passes = under_passes + over_passes

    # ---------------- residual-blob augmentation (optional) ----------------
    # After sparsery converges, detect any remaining soma-like blobs in the
    # residual image and seed ROIs at those locations. Let the classifier
    # filter by probability. This is how we recover cells sparsery misses
    # structurally — e.g. silent neurons visible in the mean image.
    augmentation_info = None
    if config.augment_with_residual_blobs and shared_plane0 is not None:
        if config.verbose:
            print(f"\n[residual augmentation] detecting missed cells in "
                  f"residual image and classifying")
        aug_dir = save_root / 'augmented'
        aug_stat, aug_ops, aug_plane0 = run_residual_augmentation(
            config, base_stat=merged_stat, base_ops=final_ops,
            base_plane0=final_plane0, shared_plane0=shared_plane0,
            save_dir=aug_dir,
        )
        if len(aug_stat) >= len(merged_stat):
            merged_stat = aug_stat
            final_ops = aug_ops
            final_plane0 = aug_plane0
            regime = f"{regime}+residual_augmented"
            augmentation_info = {
                'enabled': True,
                'augmented_plane0': str(aug_plane0),
                'final_n_rois': len(aug_stat),
            }
        else:
            if config.verbose:
                print("  > augmentation produced fewer ROIs than base; "
                      "discarding and keeping original merged result")
            augmentation_info = {
                'enabled': True,
                'augmented_plane0': str(aug_plane0),
                'final_n_rois': len(aug_stat),
                'discarded': True,
            }
    elif config.augment_with_residual_blobs and shared_plane0 is None:
        if config.verbose:
            print("  > augment_with_residual_blobs=True but reuse_registration=False "
                  "and no seed_pass0_from; skipping augmentation")

    # -------------------------- persist merged output ----------------------
    merged_out = save_root / 'merged'
    merged_out.mkdir(exist_ok=True)
    np.save(merged_out / 'stat_merged.npy',
            np.array(merged_stat, dtype=object), allow_pickle=True)
    np.save(merged_out / 'ops_final.npy', final_ops, allow_pickle=True)

    import json
    with open(merged_out / 'pass_summary.json', 'w') as f:
        json.dump({
            'regime': regime,
            'diagnosis_pass0': diag,
            'diagnosis_after_under_detection': diag_after,
            'passes': passes,
            'augmentation': augmentation_info,
        }, f, indent=2, default=str)

    if config.verbose:
        print(f"\nDone. Regime: {regime}. Merged ROI count: {len(merged_stat)}")
        print(f"Final outputs in: {merged_out}")

    # ------------------ audit PNGs (per pass + merged) --------------------
    if getattr(config, 'generate_audit_pngs', True):
        try:
            generate_audit_pngs_for_save_folder(
                str(save_root), config=config, verbose=config.verbose,
            )
        except Exception as e:
            if config.verbose:
                print(f"[audit] generation failed: {e}")

    return {
        'merged_stat': merged_stat,
        'passes': passes,
        'regime': regime,
        'diagnosis': diag,
        'final_ops': final_ops,
        'final_plane0': final_plane0,
        'save_folder': str(save_root),
        'augmentation': augmentation_info,
    }


# ============================================================================
# Standalone audit (no Suite2p invocation)
# ============================================================================

def audit_existing_detection(plane0_dir: str,
                             config: Optional[AdaptiveConfig] = None,
                             stat_file: str = 'stat.npy',
                             ops_file: str = 'ops.npy'):
    """Residual-blob audit on an already-completed Suite2p plane0 folder.

    ``stat_file`` / ``ops_file`` default to the standard names. Pass
    ``stat_merged.npy`` / ``ops_final.npy`` for the flat merged output dir.
    """
    if config is None:
        config = AdaptiveConfig()

    plane0 = Path(plane0_dir)
    stat = list(np.load(plane0 / stat_file, allow_pickle=True))
    ops = np.load(plane0 / ops_file, allow_pickle=True).item()

    Ly, Lx = ops['Ly'], ops['Lx']
    roi_mask = build_roi_pixel_mask(stat, Ly, Lx)
    residual, _ = compute_residual_image(ops, roi_mask)
    blobs = detect_residual_blobs(residual, config)

    return {
        'n_rois': len(stat),
        'n_residual_blobs': len(blobs),
        'residual_blobs': blobs,
        'residual_image': residual,
        'roi_mask': roi_mask,
    }


def visualize_audit(plane0_dir: str, config: Optional[AdaptiveConfig] = None,
                    outpath: Optional[str] = None,
                    title_prefix: Optional[str] = None,
                    stat_file: str = 'stat.npy',
                    ops_file: str = 'ops.npy'):
    """Three-panel audit figure:

      1. Mean image with detected-ROI contours overlaid (green)
      2. Residual image after masking out ROI pixels (magma colormap)
      3. Mean image with missed-cell blob candidates (cyan circles)

    Works even if the pass has zero ROIs (aborted / empty pass) — panel 1
    degrades gracefully and panel 3 still shows every residual blob.
    """
    import matplotlib.pyplot as plt
    from matplotlib import patches as mpatches

    plane0 = Path(plane0_dir)
    ops_path = plane0 / ops_file
    if not ops_path.exists():
        if config is not None and getattr(config, 'verbose', False):
            print(f"  > skipping audit for {plane0}: no {ops_file}")
        return None

    ops = np.load(ops_path, allow_pickle=True).item()
    img = ops.get('meanImg')
    if img is None:
        if config is not None and getattr(config, 'verbose', False):
            print(f"  > skipping audit for {plane0}: no meanImg in ops")
        return None

    # Residual computed over whatever ROIs exist (possibly none).
    result = audit_existing_detection(str(plane0), config,
                                      stat_file=stat_file, ops_file=ops_file)
    stat_path = plane0 / stat_file
    stat = (list(np.load(stat_path, allow_pickle=True))
            if stat_path.exists() else [])

    # Check for hard-cap abort breadcrumb so title reflects the reason for 0 ROIs.
    cap_note = ''
    cap_file = plane0 / 'HARD_CAP_ABORTED.txt'
    if cap_file.exists():
        cap_note = f"  [HARD CAP ABORT]"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    vmax = float(np.quantile(img, 0.995))

    # ---- Panel 1: mean image with detected ROI contours ----
    axes[0].imshow(img, cmap='gray', vmax=vmax)
    Ly, Lx = int(ops['Ly']), int(ops['Lx'])
    if stat:
        # Build a label mask (ROI index +1, 0 = background), then draw a
        # contour at the 0.5 iso-level between each label and bg. This is
        # faster and visually cleaner than per-ROI polygon fits.
        label_mask = np.zeros((Ly, Lx), dtype=np.int32)
        for i, s in enumerate(stat, start=1):
            yp = np.asarray(s['ypix']).astype(int)
            xp = np.asarray(s['xpix']).astype(int)
            ok = (yp >= 0) & (yp < Ly) & (xp >= 0) & (xp < Lx)
            label_mask[yp[ok], xp[ok]] = i
        axes[0].contour(label_mask > 0, levels=[0.5],
                        colors='lime', linewidths=0.8)
    axes[0].set_title(
        f"{title_prefix + ' | ' if title_prefix else ''}"
        f"Mean image + ROI contours (n={len(stat)}){cap_note}"
    )
    axes[0].axis('off')

    # ---- Panel 2: residual ----
    rvmax = float(np.quantile(result['residual_image'], 0.995))
    axes[1].imshow(result['residual_image'], cmap='magma', vmax=rvmax)
    axes[1].set_title("Residual (ROI pixels masked out)")
    axes[1].axis('off')

    # ---- Panel 3: mean image with missed-cell candidates ----
    axes[2].imshow(img, cmap='gray', vmax=vmax)
    for (y, x, r, _) in result['residual_blobs']:
        c = mpatches.Circle((x, y), r, color='cyan',
                            fill=False, linewidth=1.5)
        axes[2].add_patch(c)
    axes[2].set_title(
        f"Missed-cell candidates (n = {result['n_residual_blobs']})"
    )
    axes[2].axis('off')

    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
        plt.close(fig)
    else:
        plt.show()
    return result


def generate_audit_pngs_for_save_folder(save_folder: str,
                                        config: Optional[AdaptiveConfig] = None,
                                        verbose: bool = True):
    """Walk ``save_folder`` for detection outputs and write ``audit.png`` for
    each one.

    Covers:
      - Every per-pass folder (standard ``<pass>/suite2p/plane0/`` layout).
      - The flat ``merged/`` folder (``stat_merged.npy`` + ``ops_final.npy``).
      - The ``augmented/`` output from residual-blob augmentation.

    Folders that lack both a ``stat.npy`` and a ``stat_merged.npy`` are
    skipped (e.g. ``_shared_reg/`` is registration-only).
    """
    if config is None:
        config = AdaptiveConfig()

    root = Path(save_folder)

    # Standard plane0 dirs (pass folders, augmented).
    candidates = list(root.glob('**/suite2p/plane0'))
    # Flat "merged" folders (stat_merged.npy + ops_final.npy at the top level).
    for merged_dir in root.glob('**/stat_merged.npy'):
        candidates.append(merged_dir.parent)

    candidates = sorted(set(candidates))
    if verbose:
        print(f"\n[audit] scanning {len(candidates)} candidate folder(s) under {root}")

    n_written = 0
    for cand in candidates:
        # Resolve which stat + ops files apply and skip reg-only folders.
        if (cand / 'stat.npy').exists() and (cand / 'ops.npy').exists():
            stat_path = cand / 'stat.npy'
            ops_path = cand / 'ops.npy'
        elif (cand / 'stat_merged.npy').exists() and (cand / 'ops_final.npy').exists():
            stat_path = cand / 'stat_merged.npy'
            ops_path = cand / 'ops_final.npy'
        else:
            if verbose:
                print(f"  [audit] skip {cand}: no stat/ops pair")
            continue

        out = cand / 'audit.png'
        try:
            rel = cand.relative_to(root).parts[0]
        except ValueError:
            rel = cand.name
        if verbose:
            print(f"  [audit] {rel}: writing {out}")
        try:
            visualize_audit(str(cand), config=config, outpath=str(out),
                            title_prefix=rel,
                            stat_file=stat_path.name,
                            ops_file=ops_path.name)
            n_written += 1
        except Exception as e:
            if verbose:
                print(f"    ! audit failed for {rel}: {e}")

    if verbose:
        print(f"[audit] done — {n_written} figure(s) written")


# ============================================================================
# Script entry point
# ============================================================================

if __name__ == '__main__':
    config = AdaptiveConfig(
        tiff_folder=r'D:\2024-11-20_00003',
        save_folder=r'D:\adaptive_runs\2024-11-20_00003',
        path_to_ops=r"suite2p_2p_ops_240621.npy",  # set to None to use defaults

        # Seed from the already-known-good pass 0 (17 real cells at sc=1,
        # thr=1.0). Reuses its data.bin as the shared registration too, so
        # nothing gets re-registered.
        seed_pass0_from=None,

        # Residual-blob augmentation: after the sparsery schedule finishes,
        # detect remaining soma-like blobs in the residual image, seed ROIs
        # at their centers, extract + classify, and keep the ones that pass
        # the classifier probability threshold.
        # prob=0.15 is deliberately permissive: dense-field cells often
        # score mid-range on the built-in classifier because of overlap
        # with neighbors; we'd rather over-recall and let the user prune.
        augment_with_residual_blobs=True,
        augmentation_min_iscell_prob=0.15,

        # ---- Aggressive blob detection for high recall ----
        # Audit diagnostic on this dataset showed 400-600 cell-like peaks
        # in the mean image vs only 26 at the old defaults. Loosen all
        # three knobs: contrast threshold, scale tolerance, and the
        # center/surround ratio. Expect several hundred residual blobs
        # to be seeded; the classifier does the filtering downstream.
        blob_min_contrast=0.04,
        blob_min_area_px=10,
        soma_scale_tolerance=0.7,
        blob_center_surround_ratio=1.15,
        blob_num_sigma=12,

        # Keep the speckle guard on so any sparsery pass that locks onto
        # noise gets rejected before it can poison the merge.
        speckle_guard_area_ratio=0.5,
        speckle_guard_min_area_px=8.0,

        # Under-detection schedule: binary-search ``threshold_scaling`` over
        # [min_threshold_scaling, initial_threshold_scaling]. Each probe is
        # classified by the speckle guard — safe probes tighten the lower
        # bound and get merged; speckle probes tighten the upper bound and
        # are discarded. Converges to the loosest threshold that still finds
        # real cells, so we recover missed ROIs without inviting noise.
        threshold_schedule_mode='binary_search',
        initial_threshold_scaling=1.0,
        min_threshold_scaling=0.3,
        threshold_decay_offset=0.2,    # first probe step if hi unknown
        binary_search_precision=0.05,  # stop once bracket <=5% of initial range
        max_under_detection_passes=6,
        default_max_iterations=200,

        # Disable the spatial_scale over-detection fallback — the downstream
        # cell classifier (inside run_residual_augmentation) already filters
        # speckles, so spending passes on over-detection is wasted.
        enable_over_detection_fallback=False,

        # Pre-flight estimator: tuned to ~8 px cells (matches the 55-px
        # median area observed in pass00_sc1).
        pass0_spatial_scale=0,
        auto_estimate_spatial_scale=True,
        soma_diameter_px=8.0,
        min_residual_blobs=8,
    )

    result = adaptive_detect(config)
    print(f"\nFinal ROI count: {len(result['merged_stat'])}")
    for p in result['passes']:
        print(p)