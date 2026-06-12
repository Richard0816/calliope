"""Sparsery + Cellpose detection (no adaptive threshold search).

Why this file exists
--------------------
Suite2p ships with a detector called **Sparsery** that finds bright,
sparse, time-varying spots -- it works very well on the kind of cells
that fire often during a recording but misses cells that happen to
stay quiet. To catch the quiet ones we run a second detector,
**Cellpose** (a generalist deep-learning cell segmentation model from
Stringer & Pachitariu), on the recording's *mean image* and merge any
non-overlapping new ROIs into Suite2p's set. The combined ROI list
then goes through Suite2p's normal extraction pipeline (fluorescence
trace + neuropil + deconvolution + iscell classifier).

Pipeline at a glance
--------------------
1. ``run_one_pass``    -- Suite2p's Sparsery detector, fixed ops (no
                          binary search). Returns ``stat`` (a list,
                          one per ROI) and ``ops`` (image dict).
2. ``run_cellpose_pass`` -- Cellpose on ``ops['meanImg']``. Returns
                          its own ``stat`` list.
3. ``filter_non_overlapping_cellpose`` -- drop Cellpose ROIs whose
                          pixels are already covered by a Sparsery
                          ROI. Threshold = ``CELLPOSE_MERGE_MAX_OVERLAP``.
4. ``merge_and_extract`` -- concatenate the two stat lists, run
                          Suite2p ``extract`` to get F / Fneu / spks,
                          run Suite2p's ``classify`` to label
                          cell / not-cell, write everything to
                          ``final_dir/suite2p/plane0``.

Tab 3 calls this module's ``run_full_pipeline`` (defined later in the
file) when the user clicks "Run detection". Inputs:

- The shifted TIFF stack from Tab 1.
- Base settings (calliope's 2-photon defaults) come from
  ``calliope.core.calliope_settings.CALLIOPE_BASE_SETTINGS`` in source.
  Per-session edits flow in via Tab 3's "Edit suite2p settings..."
  popout as a ``settings_override`` dict.
- The recording's tau (GCaMP decay constant) looked up from the AAV
  metadata table.

All suite2p contact happens via :mod:`calliope.core.suite2p_pipeline`;
this file is the orchestrator that wires sparsery + cellpose together
and runs extract/dcnv/classify on the merged ROI list.

Usage:
    python sparse_plus_cellpose.py
"""

from __future__ import annotations

# ``sys.path.insert`` below lets us import sibling files when this
# script is run directly (rather than as part of the calliope package).
import sys
# ``time`` is used for ad-hoc timing prints during long-running steps.
import time
from pathlib import Path

import numpy as np

# Compute *this file's parent folder* and add it to ``sys.path`` so a
# bare ``python sparse_plus_cellpose.py`` invocation can resolve the
# sibling files. Inside the calliope package this is harmless.
WORKTREE = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKTREE))

# ``# noqa: F401`` is a linter hint meaning "don't warn about
# 'imported but unused'". Suite2p's own ``__init__`` does some
# environment setup we want to trigger even though we don't use the
# top-level name directly.
import suite2p   # noqa: F401
# ``roi_stats``: recompute the per-ROI stat dict (npix, lam, ypix...)
# after we mutate the ROI list. Needed because we splice Cellpose
# ROIs into Sparsery's list mid-pipeline.
from suite2p.detection.stats import roi_stats
# ``extract``: pull each ROI's raw fluorescence trace + a surrounding
# "neuropil" ring trace. ``dcnv``: run the OASIS deconvolution that
# produces the spks.npy estimated-firing-rate trace.
from suite2p.extraction import extract, dcnv
# ``classification``: Suite2p's built-in cell/not-cell classifier
# that writes iscell.npy.
from suite2p import classification
# ``default_settings`` is the source of truth for the nested 1.0
# extraction defaults; we deep-merge calliope's overrides on top.
from suite2p.parameters import default_settings

# Helpers re-used from the older adaptive-detection pipeline. Even
# though we skip the binary-search loop here, we still need ops
# loading, the registration cache, and the ROI-mask builder.
from .suite2p_pipeline import (
    Suite2pPipelineConfig,
    load_base_settings,
    _get_or_create_shared_registration,
    _link_or_copy,
    run_one_pass,
    run_cellpose_pass,
    build_roi_pixel_mask,
    apply_settings_overrides,
)
from . import utils


# ============================================================================
# CONFIGURATION
# ============================================================================

# Standalone-harness paths (only used by ``main()`` below when this module
# is run directly; the GUI / batch pipeline imports functions and never
# touches these). Fill in your own input/output folders before running.
TIFF_FOLDER = r'<path-to-tiff-folder>'
SAVE_FOLDER = r'<path-to-save-folder>'
# PATH_TO_OPS is kept None: base settings come from
# calliope.core.calliope_settings.build_base_settings (in-source).
# Set to a .npy path here only if you want to override with a saved
# settings/ops file from outside the repo.
PATH_TO_OPS = None

# Fixed sparsery ops — no binary search, just one pass with these values.
SPARSERY_OPS = {
    'high_pass':          100,
    'preclassify':        0.0,
    'smooth_sigma':       1.0,
    'sparse_mode':        True,
    'spatial_scale':      0,      # 0 = auto
    'threshold_scaling':  0.85,
    'max_iterations':     1500,
}

# Cellpose config (pinned to the brute-force winner).
CELLPOSE_CFG = {
    'cellpose_model_type':         'cyto2',
    'cellpose_diameter':           0,      # 0 = auto
    'cellpose_flow_threshold':     0.8,
    'cellpose_cellprob_threshold': -1.0,
    'cellpose_channel_input':      'meanImg',
}

# Cap sparsery mid-detection if it floods past this many ROIs. None = no cap.
HARD_CAP = 60000

# A cellpose ROI is dropped if this fraction of its pixels is already
# covered by the union of sparsery ROIs.
CELLPOSE_MERGE_MAX_OVERLAP = 0.3

# Tau lookup table by GCaMP variant. Kept here as a reference of the
# values the lab uses; the GUI now passes a per-recording tau scalar
# via ``tau_override`` and there is no automatic filename->AAV lookup.
# Mirrors ``tabs/suite2p/tab.py::GCAMP_OPTIONS`` -- keep the two in sync
# when adding indicators. Sources:
#   - 6f/6m/6s: Chen et al., Nature 2013 (doi:10.1038/nature12354), with
#     6s widened to 1.5 s per the Suite2p / Pachitariu 2018 slow bin
#     (doi:10.1523/JNEUROSCI.3339-17.2018).
#   - 7f: Dana et al., Nat Methods 2019 (doi:10.1038/s41592-019-0435-6).
#   - 8f/8m/8s: Zhang et al., Nature 2023 (doi:10.1038/s41586-023-05828-9)
#     base kinetics, with in-vivo OASIS-optimum tau from Rupprecht et al.
#     bioRxiv 2025 (doi:10.1101/2025.03.03.641129). The 8m value was
#     0.137 s pre-2026-05-26 -- that matches the cultured-neuron half-
#     decay, not the in-vivo OASIS optimum (~0.25 s).
DEFAULT_TAU_VALS: dict = {
    "6f": 0.7,
    "6m": 1.0,
    "6s": 1.5,
    "7f": 0.7,
    "8f": 0.15,
    "8m": 0.25,
    "8s": 0.5,
}


# ============================================================================
# Helpers
# ============================================================================

def filter_non_overlapping_cellpose(sparsery_stat, cellpose_stat,
                                    Ly: int, Lx: int,
                                    max_overlap: float,
                                    verbose: bool = True):
    """Keep cellpose ROIs whose pixels are mostly *not* covered by
    Sparsery's existing ROI footprint.

    Why we need this
    ----------------
    Sparsery and Cellpose often re-find the same cell. We don't want
    duplicate ROIs in the final list -- one cell with two ROIs ends
    up with two near-identical traces and corrupts every downstream
    statistic (clustering, cross-correlation, etc.). The simplest fix
    is to keep Sparsery's ROIs as canonical and drop any Cellpose ROI
    whose pixels are mostly already accounted for.

    Algorithm
    ---------
    1. Build a single ``(Ly, Lx)`` boolean image marking every pixel
       belonging to *any* Sparsery ROI. This is what
       ``build_roi_pixel_mask`` does under the hood.
    2. For each Cellpose ROI, compute the fraction of its pixels that
       fall inside that mask. Keep it if the fraction is below
       ``max_overlap`` (default 0.3 in the GUI), otherwise drop.

    Returns
    -------
    kept : list[dict]
        The Cellpose stat entries that survived filtering. Will be
        concatenated with ``sparsery_stat`` by ``merge_and_extract``.
    """
    # Single union-of-Sparsery-pixels mask, computed once for speed.
    sparsery_mask = build_roi_pixel_mask(sparsery_stat, Ly, Lx)
    kept = []
    dropped = 0
    for s in cellpose_stat:
        # ``np.asarray(...).astype(int)`` defensively coerces whatever
        # Cellpose returns (sometimes lists, sometimes int64) into the
        # int dtype we'll use for indexing.
        yp = np.asarray(s['ypix']).astype(int)
        xp = np.asarray(s['xpix']).astype(int)
        # Bounds check: ROIs near the FOV edge can have one or two
        # pixels rounded slightly out of range; clip those.
        ok = (yp >= 0) & (yp < Ly) & (xp >= 0) & (xp < Lx)
        yp = yp[ok]; xp = xp[ok]
        if yp.size == 0:
            dropped += 1
            continue
        # ``sparsery_mask[yp, xp]`` is fancy indexing -- a vector of
        # 0/1 telling us, for each Cellpose pixel, whether it's
        # already inside the Sparsery footprint. Its mean is the
        # overlap fraction.
        overlap = float(sparsery_mask[yp, xp].sum()) / float(yp.size)
        if overlap < max_overlap:
            kept.append(s)
        else:
            dropped += 1
    if verbose:
        print(f"    cellpose merge: kept {len(kept)} / {len(cellpose_stat)} "
              f"ROIs (dropped {dropped} overlapping sparsery)")
    return kept


def merge_and_extract(sparsery_stat, cellpose_stat,
                      shared_plane0: Path, base_settings: dict,
                      final_dir: Path, max_overlap: float,
                      verbose: bool = True,
                      extra_cellpose_passes: "list[tuple[str, list]] | None" = None,
                      detect_plane0: "Path | None" = None):
    """Combine sparsery + non-overlapping cellpose ROIs, then extract
    fluorescence → dcnv → classify → save into ``final_dir/suite2p/plane0/``.

    This is the heart of Tab 3's "Run detection" worker. Walking
    through the steps:

    1. **Reuse the registered movie.** Suite2p already wrote the
       motion-corrected ``data.bin`` and ``ops.npy`` into the shared
       Sparsery folder. Symlink (or copy) ``data.bin`` over to the
       final folder so we don't have to re-register the movie.
    2. **Drop overlapping Cellpose ROIs** with
       ``filter_non_overlapping_cellpose``. Each Cellpose pass is
       filtered *independently* against the Sparsery mask (i.e. we
       don't filter the SAM pass against the cyto2 pass). The audit
       (Tier 2 #13) calls this out: SAM is meant to recover silent
       cells Sparsery missed, and we want both detectors to have a
       full shot at those even if they happen to agree on some.
    3. **Concatenate** all stat lists into ``combined_raw``, tagging
       each entry with ``_source`` so we can audit later.
    4. **Build the final ops dict** by copying registration-time ops
       and overlaying the trace-extraction knobs from ``base_ops``
       (tau, fs, neucoeff, baseline window, etc.).
    5. **Recompute per-ROI stats** with Suite2p's ``roi_stats`` so
       every dict has the full set of fields (npix, lam normalisation,
       med, ipix, etc.) that ``extract`` expects.
    6. **Extract** F / Fneu using Suite2p's neuropil ring extractor.
    7. **Deconvolve** spks via OASIS (``dcnv``).
    8. **Classify** cell vs not-cell into ``iscell.npy``.
    9. **Save** every output (F, Fneu, spks, stat, ops, iscell) into
       ``final_dir/suite2p/plane0`` -- the same layout Tabs 4-8 expect.

    Parameters
    ----------
    sparsery_stat : list of dict
        ROIs found by Suite2p's Sparsery detector. ``_source`` will
        be set to ``"sparsery"``.
    cellpose_stat : list of dict
        ROIs from the primary Cellpose pass (cyto2 on meanImg in 3.x
        / SAM on meanImg in 4.x). ``_source`` will be set to
        ``"cellpose_cyto2"`` -- the name is historical; in cellpose
        4.x it's actually SAM-on-meanImg, but downstream consumers
        already key off this string.
    extra_cellpose_passes : list of (source_tag, stat) tuples, optional
        Additional Cellpose passes to merge in. The Tier 2 #13
        Cellpose-SAM-on-Vcorr pass arrives here with
        ``source_tag="cellpose_sam"``. Each pass is filtered
        independently against the Sparsery mask before concatenation.

    Returns the path to the populated plane0 folder so callers can
    publish it on ``AppState.plane0``.
    """
    final_plane0 = final_dir / 'suite2p' / 'plane0'
    final_plane0.mkdir(parents=True, exist_ok=True)

    _link_or_copy(Path(shared_plane0) / 'data.bin',
                  final_plane0 / 'data.bin')
    # Load the shared registration's runtime view (Ly, Lx, meanImg, ...)
    # via load_plane_view so we work off the new db.npy / settings.npy /
    # reg_outputs.npy on disk (with a fallback to legacy ops.npy).
    reg_view = utils.load_plane_view(shared_plane0)
    Ly, Lx = int(reg_view['Ly']), int(reg_view['Lx'])

    cellpose_kept = filter_non_overlapping_cellpose(
        sparsery_stat, cellpose_stat, Ly, Lx,
        max_overlap=max_overlap, verbose=verbose,
    )

    # Filter each extra cellpose pass independently against the same
    # Sparsery mask. Doing it independently (instead of chaining each
    # against the previous cumulative mask) is deliberate: the audit
    # plan for Tier 2 #13 explicitly says "Filter each Cellpose output
    # against Sparsery independently (not against each other)" so the
    # SAM pass can recover silent cells the cyto2 pass also flagged.
    # The follow-up de-dup happens implicitly via Suite2p's
    # extract/classify pipeline downstream.
    extra_kept: list[tuple[str, list]] = []
    if extra_cellpose_passes:
        for tag, stat_list in extra_cellpose_passes:
            kept = filter_non_overlapping_cellpose(
                sparsery_stat, stat_list, Ly, Lx,
                max_overlap=max_overlap, verbose=verbose,
            )
            extra_kept.append((str(tag), kept))

    combined_raw = []
    for s in sparsery_stat:
        combined_raw.append({
            'ypix': np.asarray(s['ypix']).astype(np.int32),
            'xpix': np.asarray(s['xpix']).astype(np.int32),
            'lam':  np.asarray(s['lam']).astype(np.float32),
            'med':  list(s.get('med', [int(np.median(s['ypix'])),
                                        int(np.median(s['xpix']))])),
            '_source': 'sparsery',
        })
    for s in cellpose_kept:
        combined_raw.append({
            'ypix': np.asarray(s['ypix']).astype(np.int32),
            'xpix': np.asarray(s['xpix']).astype(np.int32),
            'lam':  np.asarray(s['lam']).astype(np.float32),
            'med':  list(s.get('med', [int(np.median(s['ypix'])),
                                        int(np.median(s['xpix']))])),
            '_source': 'cellpose_cyto2',
        })
    for tag, kept in extra_kept:
        for s in kept:
            combined_raw.append({
                'ypix': np.asarray(s['ypix']).astype(np.int32),
                'xpix': np.asarray(s['xpix']).astype(np.int32),
                'lam':  np.asarray(s['lam']).astype(np.float32),
                'med':  list(s.get('med', [int(np.median(s['ypix'])),
                                            int(np.median(s['xpix']))])),
                '_source': tag,
            })

    n_sp = len(sparsery_stat)
    n_cp = len(cellpose_kept)
    if verbose:
        parts = [f"{n_sp} sparsery", f"{n_cp} cellpose_cyto2"]
        for tag, kept in extra_kept:
            parts.append(f"{len(kept)} {tag}")
        parts_str = " + ".join(parts)
        print(f"    merged stat: {parts_str} = "
              f"{len(combined_raw)} total ROIs")

    if not combined_raw:
        raise RuntimeError('merge produced 0 ROIs — nothing to extract')

    # Build the flat ops dict that suite2p's extract.create_masks_and_extract
    # still consumes. Start from the shared-registration's view and overlay
    # the trace-extraction knobs we care about, pulled from base_settings.
    from .suite2p_pipeline import get_setting
    final_ops = dict(reg_view)
    final_ops['tau'] = float(base_settings.get('tau', final_ops.get('tau', 1.0)))
    final_ops['fs'] = float(base_settings.get('fs', final_ops.get('fs', 15.0)))
    final_ops['neucoeff'] = float(
        get_setting(base_settings, 'neuropil_coefficient',
                    final_ops.get('neucoeff', 0.7))
    )
    for k in ('baseline', 'win_baseline', 'sig_baseline',
              'prctile_baseline'):
        v = get_setting(base_settings, k)
        if v is not None:
            final_ops[k] = v
    bs = get_setting(base_settings, 'batch_size')
    if bs is not None:
        final_ops['batch_size'] = bs
    final_ops['save_path0'] = str(final_dir)
    final_ops['save_folder'] = 'suite2p'
    final_ops['reg_file']    = str(final_plane0 / 'data.bin')
    final_ops['ops_path']    = str(final_plane0 / 'ops.npy')
    final_ops['do_registration'] = 0
    final_ops['Ly'] = Ly
    final_ops['Lx'] = Lx

    # Carry the detection-pass projection images (max_proj / Vcorr /
    # maxImg) into the final ops. Suite2p computes these during Sparsery
    # detection and writes them to the sparsery pass's detect_outputs.npy,
    # but that pass is a pruned intermediate -- and final_ops is built from
    # the registration-only view, which never has them. Without this copy
    # the final plane0 has no max projection / correlation image, so the
    # GUI background dropdown, figure export, and reload all miss them.
    if detect_plane0 is not None:
        try:
            det_view = utils.load_plane_view(Path(detect_plane0))
            for _k in ('max_proj', 'Vcorr', 'maxImg'):
                _v = det_view.get(_k)
                if isinstance(_v, np.ndarray) and _v.ndim == 2:
                    final_ops[_k] = _v
                    if verbose:
                        print(f"    carried {_k} {tuple(_v.shape)} into "
                              f"final ops")
        except Exception as _e:
            if verbose:
                print(f"    could not carry detection projections: {_e}")

    median_diam = max(3, int(round(
        2.0 * float(np.sqrt(np.mean(
            [len(s['ypix']) for s in combined_raw]
        ) / np.pi)),
    )))
    # suite2p 1.0's roi_stats unpacks ``dy, dx = diameter[0], diameter[1]``
    # so a scalar int trips a ``'int' object is not subscriptable``. Pass a
    # 2-element list (square pixels on the lab's 2-photon rig).
    # It also fancy-indexes ``stats[keep_rois]`` with a numpy bool mask, so
    # the input list has to be wrapped as an object ndarray; a bare Python
    # list raises ``TypeError: only integer scalar arrays can be converted
    # to a scalar index``.
    combined_arr = np.array(combined_raw, dtype=object)
    enriched = roi_stats(combined_arr, Ly, Lx,
                         diameter=[median_diam, median_diam])
    for i, s in enumerate(enriched):
        if '_source' not in s:
            s['_source'] = combined_raw[i].get('_source', 'sparsery')
    stat_arr = np.array(enriched, dtype=object)

    if verbose:
        print(f"    extracting traces for {len(stat_arr)} ROIs")
    # suite2p 1.0 dropped ``extract.create_masks_and_extract`` and now exposes
    # ``extract.extraction_wrapper(stat, f_reg, settings=...)``. We open the
    # registered binary as a BinaryFile (memmap-backed; exposes the .shape
    # extraction_wrapper expects) and pass settings.extraction through.
    from suite2p.io import BinaryFile
    import torch
    nframes = int(reg_view.get('nframes') or 0)
    if nframes <= 0:
        # Fallback when load_plane_view didn't surface nframes (e.g. older
        # plane folder without db.npy): infer from file size at int16.
        nframes = (final_plane0 / 'data.bin').stat().st_size // (Ly * Lx * 2)
    extraction_settings = {
        **default_settings()['extraction'],
        **base_settings.get('extraction', {}),
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with BinaryFile(Ly=Ly, Lx=Lx,
                    filename=str(final_plane0 / 'data.bin'),
                    n_frames=nframes, write=False) as f_reg:
        F, Fneu, F_chan2, Fneu_chan2 = extract.extraction_wrapper(
            stat_arr, f_reg,
            settings=extraction_settings,
            device=device,
        )

    tau = float(final_ops.get('tau', 1.0))
    fs = float(final_ops.get('fs', 15.0))
    neucoeff = float(extraction_settings.get('neuropil_coefficient',
                                             final_ops.get('neucoeff', 0.7)))
    F_sub = F - neucoeff * Fneu
    F_pp = dcnv.preprocess(
        F=F_sub,
        baseline=final_ops.get('baseline', 'maximin'),
        win_baseline=float(final_ops.get('win_baseline', 60.0)),
        sig_baseline=float(final_ops.get('sig_baseline', 10.0)),
        fs=fs,
        prctile_baseline=float(final_ops.get('prctile_baseline', 8.0)),
        # suite2p's dcnv.preprocess defaults device=torch.device('cuda');
        # pass the device we resolved above (cpu when torch.cuda is
        # unavailable) so the maximin baseline runs on CPU-only torch
        # builds instead of raising "Torch not compiled with CUDA enabled".
        device=device,
    )
    spks = dcnv.oasis(
        F=F_pp, batch_size=int(final_ops.get('batch_size', 3000)),
        tau=tau, fs=fs,
    )

    try:
        iscell = classification.classify(
            stat=stat_arr, classfile=classification.builtin_classfile,
        )
    except Exception as e:
        if verbose:
            print(f"    classifier failed ({e}); marking all as cells")
        iscell = np.ones((len(stat_arr), 2), dtype=np.float32)

    np.save(final_plane0 / 'stat.npy',   stat_arr,  allow_pickle=True)
    np.save(final_plane0 / 'F.npy',      F)
    np.save(final_plane0 / 'Fneu.npy',   Fneu)
    np.save(final_plane0 / 'spks.npy',   spks)
    np.save(final_plane0 / 'iscell.npy', iscell)
    # Persist the merged-output ops snapshot. Suite2p 1.0's full pipeline
    # would write its own ops.npy (when settings.io.save_ops_orig=True),
    # but here we ran extract/dcnv/classify by hand on the shared
    # registration, so we drop the merged ops ourselves so downstream
    # readers (utils.load_plane_view, Tab 6/7 spatial overlays, the
    # cellfilter dataset loader) see a complete view.
    np.save(final_plane0 / 'ops.npy',    final_ops, allow_pickle=True)

    # Stamp the per-recording parameters known at extraction time into the
    # meta.json sidecar (single source of truth for downstream consumers and
    # the NWB exporter). The GCaMP variant *label* is a GUI/headless string
    # stamped separately by the caller; here we record what is in scope.
    # A metadata write must never break the pipeline.
    try:
        from . import recording_meta
        recording_meta.update_meta(
            final_plane0,
            registration={
                "block_size": final_ops.get("block_size"),
                "smooth_sigma": final_ops.get("smooth_sigma"),
                "nonrigid": final_ops.get("nonrigid"),
            },
            detection={
                "threshold_scaling": final_ops.get("threshold_scaling"),
                "sparse_mode": final_ops.get("sparse_mode"),
            },
            extraction={
                "neuropil_coef": float(neucoeff),
                "tau_seconds": float(tau),
            },
            spike_inference={"backend": "oasis"},
        )
    except Exception as _meta_exc:
        if verbose:
            print(f"    [meta] could not write meta.json: {_meta_exc}")

    if verbose:
        n_accept = (int((iscell[:, 0] > 0).sum())
                    if iscell.ndim == 2 else int((iscell > 0).sum()))
        print(f"    final: {len(stat_arr)} ROIs "
              f"({n_accept} classified as cells)")

    return stat_arr, final_ops, final_plane0


# ============================================================================
# Public API
# ============================================================================

def run(
    tiff_folder: str | Path,
    save_folder: str | Path,
    path_to_ops: str | Path | None = None,
    sparsery_ops: dict | None = None,
    cellpose_cfg: dict | None = None,
    hard_cap: int | None = HARD_CAP,
    max_overlap: float = CELLPOSE_MERGE_MAX_OVERLAP,
    tau_override: float | None = None,
    fps_override: float | None = None,
    settings_override: dict | None = None,
    enable_sam_vcorr_pass: bool = False,
    sam_vcorr_cfg: dict | None = None,
    verbose: bool = True,
) -> Path:
    """Sparsery + cellpose detection. Returns the final ``suite2p/plane0`` path.

    This is the single-call public API Tab 3's worker uses. The full
    flow:

    1. Build a ``Suite2pPipelineConfig`` (input bundle for the
       suite2p invocation helpers in
       :mod:`calliope.core.suite2p_pipeline` -- tau override,
       optional popout settings override, shared-registration cache).
    2. ``load_base_settings`` builds the base 2-photon ``(db, settings)``
       pair from :func:`calliope_settings.build_base_settings` (in source)
       and overlays any per-session ``settings_override`` (from Tab 3's
       popout) plus the user-picked GCaMP-variant tau.
    3. ``_get_or_create_shared_registration`` runs Suite2p's motion-
       correction once and caches ``data.bin`` + ``ops.npy`` so both
       Sparsery and Cellpose can reuse it.
    4. ``run_one_pass`` -- Sparsery detection on the registered movie
       at the fixed ops up top.
    5. ``run_cellpose_pass`` -- Cellpose on the mean image, also using
       the shared registration cache.
    6. ``merge_and_extract`` -- drop overlapping Cellpose ROIs,
       concatenate, run Suite2p extract / dcnv / classify, save into
       ``<save_folder>/final/suite2p/plane0``.

    Parameters
    ----------
    tiff_folder : str | Path
        Folder containing the shifted TIFF(s). Multi-TIFF groups are
        read as one continuous recording.
    save_folder : str | Path
        Where to write ``sparsery_pass``, ``cellpose_pass`` and the
        final merged ``final/suite2p/plane0`` outputs.
    path_to_ops : str | Path, optional
        Override for the base ops .npy file. Defaults to the lab's
        2-photon ops profile.
    sparsery_ops, cellpose_cfg : dict, optional
        Per-detector overrides; fall back to module-level defaults.
    hard_cap : int, optional
        Maximum number of Sparsery ROIs to keep before bailing.
        ``None`` disables the cap.
    max_overlap : float
        Cellpose ROIs whose pixels are >= this fraction inside an
        existing Sparsery ROI are dropped during merge.
    tau_override : float, optional
        Per-recording tau scalar (seconds) used for Suite2p
        deconvolution. The GUI resolves this from the GCaMP-variant
        picker; ``None`` falls back to the in-source default tau.
    verbose : bool
        Print step-by-step progress.

    Returns
    -------
    final_plane0 : Path
        The populated ``suite2p/plane0`` folder. Tab 3 publishes this
        on ``AppState.plane0`` so downstream tabs (lowpass, event
        detection, ...) can pick it up.
    """
    save_root = Path(save_folder)
    save_root.mkdir(parents=True, exist_ok=True)

    sp_ops = dict(SPARSERY_OPS) if sparsery_ops is None else dict(sparsery_ops)
    cp_cfg = dict(CELLPOSE_CFG) if cellpose_cfg is None else dict(cellpose_cfg)
    # ``path_to_ops`` is the legacy back-compat path: when explicit, it
    # wins. When None, the in-source defaults from
    # ``calliope.core.calliope_settings.build_base_settings`` are used.
    ops_path = str(path_to_ops) if path_to_ops else None

    cfg_kwargs = dict(
        tiff_folder=str(tiff_folder),
        save_folder=str(save_root),
        path_to_ops=ops_path,
        use_cache=False,
        verbose=verbose,
    )
    if tau_override is not None:
        cfg_kwargs["tau_override"] = float(tau_override)
    if fps_override is not None and float(fps_override) > 0:
        cfg_kwargs["fs_override"] = float(fps_override)
    if settings_override:
        cfg_kwargs["settings_override"] = dict(settings_override)
    cfg = Suite2pPipelineConfig(**cfg_kwargs)

    if verbose:
        if ops_path:
            print(f'[s+cp] loading base settings from legacy file {ops_path}')
        else:
            from . import calliope_settings as _cs
            n_overrides = len(_cs.CALLIOPE_BASE_SETTINGS)
            tweak = " + popout edits" if settings_override else ""
            print(f'[s+cp] base settings = suite2p defaults + '
                  f'{n_overrides} calliope overrides{tweak}')
    base_db, base_settings = load_base_settings(cfg)

    if verbose:
        print(f'[s+cp] shared registration')
    t0 = time.time()
    shared_plane0 = _get_or_create_shared_registration(
        cfg, base_db, base_settings,
    )
    if verbose:
        print(f'[s+cp] shared reg ready at {shared_plane0}  '
              f'({time.time() - t0:.1f}s)')

    # ---- sparsery (one shot, fixed settings) ----
    if verbose:
        print(f'[s+cp] sparsery pass '
              f'(threshold_scaling={sp_ops["threshold_scaling"]})')
    sparsery_dir = save_root / 'sparsery_pass'
    sp_settings = {
        # _deep_copy via dict comprehension over top-level keys; nested
        # dicts get cloned by apply_settings_overrides as needed.
        k: (dict(v) if isinstance(v, dict) else v)
        for k, v in base_settings.items()
    }
    apply_settings_overrides(sp_settings, sp_ops)
    apply_settings_overrides(sp_settings, {'roidetect': True})
    sp_db = dict(base_db)
    t_sp = time.time()
    sp_stat, _sp_view, _sp_plane0 = run_one_pass(
        str(tiff_folder), sparsery_dir, sp_db, sp_settings,
        verbose=verbose, use_cache=False,
        shared_plane0=shared_plane0,
        hard_cap=(hard_cap or 0),
    )
    if verbose:
        print(f'[s+cp] sparsery: {len(sp_stat)} ROIs '
              f'({time.time() - t_sp:.1f}s)')

    # ---- cellpose (primary pass on meanImg) ----
    if verbose:
        print(f'[s+cp] cellpose pass (meanImg)')
    cp_dir = save_root / 'cellpose_pass'
    t_cp = time.time()
    cp_stat, _cp_ops, _cp_plane0 = run_cellpose_pass(
        cp_dir, shared_plane0, cp_cfg, verbose=verbose,
    )
    if verbose:
        print(f'[s+cp] cellpose: {len(cp_stat)} ROIs '
              f'({time.time() - t_cp:.1f}s)')

    # ---- cellpose-SAM second pass on the correlation image (Vcorr) ----
    # The audit (Tier 2 #13) recommends this as a recovery pass for
    # cells Sparsery misses: Vcorr lights up pixels whose timecourse
    # correlates with their neighbours, which is exactly what a soma
    # looks like even if the cell happens to fire infrequently.
    # Cellpose-SAM is the unified model in cellpose 4.x; in 3.x the
    # check below gates the pass off with a clear warning.
    extra_passes: list[tuple[str, list]] = []
    if enable_sam_vcorr_pass:
        from .suite2p_pipeline import cellpose_sam_available
        if not cellpose_sam_available():
            if verbose:
                print('[s+cp] SAM-on-Vcorr requested but cellpose>=4.0 '
                      'not available; skipping second pass.')
        else:
            # Resolve a Vcorr image and persist it next to shared_plane0
            # so run_cellpose_pass's filesystem fallback can find it.
            # The Sparsery pass we just ran wrote detect_outputs.npy
            # into ``sparsery_dir / suite2p / plane0``; pull from there
            # if available, otherwise stream-compute from data.bin.
            from .correlation_image import load_or_compute_vcorr
            sparsery_plane0 = sparsery_dir / 'suite2p' / 'plane0'
            try:
                vcorr_src = (sparsery_plane0
                             if sparsery_plane0.exists()
                             else shared_plane0)
                # Pull Ly/Lx/nframes from the shared registration view
                # so the compute_vcorr fallback path can do its sums
                # without re-parsing the binary header.
                reg_view = utils.load_plane_view(shared_plane0)
                Ly_v = int(reg_view.get('Ly') or 0)
                Lx_v = int(reg_view.get('Lx') or 0)
                nframes_v = int(reg_view.get('nframes') or 0)
                data_bin_path = Path(shared_plane0) / 'data.bin'
                vcorr = load_or_compute_vcorr(
                    vcorr_src,
                    data_bin_path=data_bin_path,
                    Ly=Ly_v, Lx=Lx_v, nframes=nframes_v,
                )
                # Persist in shared_plane0 so run_cellpose_pass's
                # ``<plane0>/<key>.npy`` filesystem fallback picks it up.
                np.save(Path(shared_plane0) / 'Vcorr.npy', vcorr)
                if verbose:
                    print(f'[s+cp] Vcorr persisted '
                          f'({vcorr.shape}, range '
                          f'[{vcorr.min():.3f}, {vcorr.max():.3f}])')

                sam_cfg = dict(sam_vcorr_cfg) if sam_vcorr_cfg else {}
                # Fill in sensible defaults for the SAM-on-Vcorr pass.
                # ``cellpose_channel_input='Vcorr'`` is what triggers
                # the filesystem-fallback lookup in run_cellpose_pass.
                # ``model_type='cpsam'`` is an audit-tracked label;
                # cellpose 4.x ignores it (only one model exists).
                sam_cfg.setdefault('cellpose_model_type', 'cpsam')
                sam_cfg.setdefault('cellpose_channel_input', 'Vcorr')
                sam_cfg.setdefault('cellpose_diameter',
                                   cp_cfg.get('cellpose_diameter', 0))
                sam_cfg.setdefault('cellpose_flow_threshold',
                                   cp_cfg.get('cellpose_flow_threshold', 0.8))
                sam_cfg.setdefault('cellpose_cellprob_threshold',
                                   cp_cfg.get('cellpose_cellprob_threshold',
                                              -1.0))

                if verbose:
                    print(f'[s+cp] cellpose-SAM pass (Vcorr)')
                t_sam = time.time()
                sam_dir = save_root / 'cellpose_sam_pass'
                sam_stat, _sam_ops, _sam_plane0 = run_cellpose_pass(
                    sam_dir, shared_plane0, sam_cfg, verbose=verbose,
                )
                if verbose:
                    print(f'[s+cp] cellpose-SAM: {len(sam_stat)} ROIs '
                          f'({time.time() - t_sam:.1f}s)')
                extra_passes.append(('cellpose_sam', sam_stat))
            except Exception as e:
                if verbose:
                    print(f'[s+cp] SAM-on-Vcorr failed ({e}); '
                          'continuing with primary cellpose only.')

    # ---- merge + extract ----
    if verbose:
        print(f'[s+cp] merging + extracting')
    final_dir = save_root / 'final'
    stat_arr, _final_ops, final_plane0 = merge_and_extract(
        sp_stat, cp_stat, shared_plane0, base_settings, final_dir,
        max_overlap=max_overlap, verbose=verbose,
        extra_cellpose_passes=extra_passes or None,
        detect_plane0=_sp_plane0,
    )

    if verbose:
        print(f'\n[s+cp] DONE.')
        print(f'  plane0:    {final_plane0}')
        print(f'  stat.npy:  {len(stat_arr)} ROIs')
        print(f'  F/Fneu/spks/iscell saved')
    return final_plane0


# ============================================================================
# Module-globals entrypoint (used by run_full_pipeline.py and CLI execution)
# ============================================================================

def main():
    """Run with the module-level config constants.

    The intent is that lab members can edit ``TIFF_FOLDER``,
    ``SAVE_FOLDER``, ``PATH_TO_OPS`` etc. at the top of this file,
    save, and run ``python sparse_plus_cellpose.py`` for a one-shot
    detection without writing any code. The Tab 3 GUI calls
    ``run(...)`` directly with parameters from its form, so it
    bypasses this function entirely.

    Callers that monkey-patch the module-level constants before
    calling ``main()`` (e.g. ``run_full_pipeline.run_detection``)
    also keep working unchanged.
    """
    run(
        TIFF_FOLDER, SAVE_FOLDER, PATH_TO_OPS,
        sparsery_ops=SPARSERY_OPS,
        cellpose_cfg=CELLPOSE_CFG,
        hard_cap=HARD_CAP,
        max_overlap=CELLPOSE_MERGE_MAX_OVERLAP,
        verbose=True,
    )


# Standard "if this file is the program being run, do this" guard.
# When the GUI imports the module instead, ``__name__`` is the
# import path and ``main()`` doesn't fire.
if __name__ == '__main__':
    main()
