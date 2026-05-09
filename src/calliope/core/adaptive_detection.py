"""Adaptive Suite2p detection pipeline (legacy).

Why this file exists
--------------------
This is the *original* detection pipeline -- ``sparse_plus_cellpose``
is the simpler successor. The adaptive version runs Sparsery in a
binary-search loop, dialling ``threshold_scaling`` up or down until
the number of detected ROIs lands in a sweet spot. It also keeps a
shared registration cache so the search loop doesn't re-register the
movie every iteration.

Suite2p 1.0 adaptation (2026-05)
--------------------------------
Suite2p 1.0.0.1 (PyPI 2026-02-11) replaced the flat ``ops=`` kwarg on
``suite2p.run_s2p`` with a nested ``db=`` + ``settings=`` schema, and
moved registration/detection progress output from ``print()`` to
``logging.getLogger('suite2p')``. CalLIOPE still maintains a flat
legacy-style ``ops`` dict as its internal pipeline state because the
on-disk base ops .npy is shaped that way and downstream code reads
``ops['Ly']`` / ``ops['spatial_scale']`` straight from suite2p's own
output. Translation to the new nested schema happens at the
``run_s2p`` call boundary -- see :func:`_build_db_and_settings`,
:func:`_ensure_s2p_logger`, and :func:`_run_s2p`.

It's still used in CalLIOPE for two reasons:

1. ``AdaptiveConfig`` carries the AAV / tau lookup logic that
   ``sparse_plus_cellpose.run`` depends on.
2. ``load_base_ops``, ``_get_or_create_shared_registration``,
   ``_link_or_copy``, ``run_one_pass``, ``build_roi_pixel_mask`` and
   ``visualize_audit`` are imported by ``sparse_plus_cellpose.py``.

The "adaptive" loop itself is no longer wired into the GUI -- Tab 3
calls ``sparse_plus_cellpose.run`` instead -- but the helpers below
are too useful to remove.

Public-ish surface (used elsewhere in the package)
--------------------------------------------------
``AdaptiveConfig``
    Dataclass bundling the inputs (tiff folder, save folder, base ops
    path, optional AAV info CSV, optional tau lookup).
``load_base_ops(cfg)``
    Read the base ops .npy and overlay the recording's tau (looked up
    by GCaMP variant from the AAV CSV).
``_get_or_create_shared_registration(cfg, base_ops)``
    Run Suite2p registration once; subsequent detection passes re-use
    the cached ``data.bin`` + ``ops.npy`` from the returned plane0.
``_link_or_copy(src, dst)``
    Hard-link if possible (fast, no extra disk), fall back to copy.
``run_one_pass(tiff_folder, out_dir, ops, ...)``
    One Sparsery detection pass on the shared registration. Returns
    ``stat``, ``ops``, plane0 path.
``build_roi_pixel_mask(stat, Ly, Lx)``
    Boolean ``(Ly, Lx)`` image marking every pixel that belongs to
    *any* ROI in ``stat``.
``visualize_audit(plane0, ...)``
    Save a PNG with the mean image and every ROI footprint outlined
    -- the "did detection do something sane?" debugging picture.

User only needs to specify:
    - tiff_folder: where the raw TIFF stack lives
    - save_folder: where Suite2p output should be written
    - path_to_ops (optional): starting ops.npy to build from; if None, uses
      _suite2p_default_ops() with sensible defaults for human 2p GCaMP data

The goal is to wrap this behind a minimal app where the user never touches
the Suite2p GUI or config files directly.
"""

from __future__ import annotations

import os
import re
import shutil
import numpy as np
import suite2p
from . import utils


def _suite2p_default_ops() -> dict:
    """Return suite2p's legacy flat default-ops dict.

    Suite2p 1.0 dropped the package-level ``suite2p.default_ops()``
    shortcut but kept the function inside the ``suite2p.default_ops``
    submodule. We still build flat ops dicts internally and translate
    at the ``run_s2p`` boundary (see :func:`_build_db_and_settings`),
    so we just need a working default-ops factory regardless of
    suite2p version.
    """
    try:
        from suite2p.default_ops import default_ops
        return default_ops()
    except (ImportError, AttributeError):
        # Pre-1.0 suite2p exposed it as a top-level attribute.
        return suite2p.default_ops()  # type: ignore[attr-defined]
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ============================================================================
# Suite2p 1.0 db / settings model
# ============================================================================
#
# Suite2p 1.0.0.1 (uploaded to PyPI 2026-02-11) made two breaking changes:
#
# 1. ``run_s2p`` no longer accepts ``ops=``; it takes a nested
#    ``settings=`` dict plus an enriched ``db=`` dict. Old flat keys
#    now live in different locations -- e.g. ``ops['nbinned']`` ->
#    ``settings['detection']['nbins']``, ``ops['high_pass']`` ->
#    ``settings['detection']['highpass_time']``, ``ops['batch_size']``
#    (a registration knob in the old flat dict) ->
#    ``settings['registration']['batch_size']``. ``roidetect`` /
#    ``spikedetect`` move into ``settings['run']['do_detection']`` /
#    ``settings['run']['do_deconvolution']``.
#
# 2. ``run_s2p`` switched from ``print()`` to Python ``logging``
#    (``logging.getLogger('suite2p')``), but only suite2p's own GUI/CLI
#    entry points install handlers. Programmatic callers must run
#    ``run_s2p.logger_setup`` themselves or registration progress goes
#    nowhere.
#
# CalLIOPE pipeline state now lives natively in the new nested form:
# the GUI tab and ``sparse_plus_cellpose`` build ``(db, settings)``
# pairs and pass them through ``run_one_pass`` /
# ``_get_or_create_shared_registration``. We no longer translate from
# a flat ops dict at the call boundary -- helper utilities below
# (``set_setting``, ``get_setting``, ``apply_settings_overrides``)
# manipulate the nested settings directly.

# Map of legacy flat-ops names to their new nested settings location.
# Used by ``apply_settings_overrides`` so callers can still pass
# user-friendly flat dicts (e.g. {'threshold_scaling': 0.85}) without
# remembering whether each key landed in ``detection`` or
# ``detection.sparsery_settings``.
_SETTINGS_PATHS: dict = {
    # top-level
    'tau':                      (),
    'fs':                       (),
    'diameter':                 (),
    'torch_device':             (),
    # run flags
    'do_registration':          ('run',),
    'multiplane_parallel':      ('run',),
    'do_detection':             ('run',),
    'do_deconvolution':         ('run',),
    # io
    'combined':                 ('io',),
    'save_mat':                 ('io',),
    'save_NWB':                 ('io',),
    'delete_bin':               ('io',),
    'move_bin':                 ('io',),
    'save_ops_orig':            ('io',),
    # registration
    'nimg_init':                ('registration',),
    'maxregshift':              ('registration',),
    'do_bidiphase':             ('registration',),
    'bidiphase':                ('registration',),
    'batch_size':               ('registration',),
    'nonrigid':                 ('registration',),
    'maxregshiftNR':            ('registration',),
    'block_size':               ('registration',),
    'smooth_sigma':             ('registration',),
    'smooth_sigma_time':        ('registration',),
    'spatial_taper':            ('registration',),
    'th_badframes':             ('registration',),
    'norm_frames':              ('registration',),
    'snr_thresh':               ('registration',),
    'subpixel':                 ('registration',),
    'two_step_registration':    ('registration',),
    'reg_tif':                  ('registration',),
    'reg_tif_chan2':            ('registration',),
    # detection
    'denoise':                  ('detection',),
    'threshold_scaling':        ('detection',),
    'max_overlap':              ('detection',),
    'soma_crop':                ('detection',),
    'algorithm':                ('detection',),
    'nbins':                    ('detection',),
    'highpass_time':            ('detection',),
    'chan2_threshold':          ('detection',),
    # detection.sparsery_settings
    'spatial_scale':            ('detection', 'sparsery_settings'),
    # detection.sourcery_settings
    'connected':                ('detection', 'sourcery_settings'),
    'max_iterations':           ('detection', 'sourcery_settings'),
    # detection.cellpose_settings
    'flow_threshold':           ('detection', 'cellpose_settings'),
    'cellprob_threshold':       ('detection', 'cellpose_settings'),
    'cellpose_model':           ('detection', 'cellpose_settings'),
    # classification
    'preclassify':              ('classification',),
    'classifier_path':          ('classification',),
    'use_builtin_classifier':   ('classification',),
    # extraction
    'neuropil_extract':         ('extraction',),
    'inner_neuropil_radius':    ('extraction',),
    'min_neuropil_pixels':      ('extraction',),
    'lam_percentile':           ('extraction',),
    'allow_overlap':            ('extraction',),
    'neuropil_coefficient':     ('extraction',),
    # dcnv preprocess
    'baseline':                 ('dcnv_preprocess',),
    'win_baseline':             ('dcnv_preprocess',),
    'sig_baseline':             ('dcnv_preprocess',),
    'prctile_baseline':         ('dcnv_preprocess',),
}

# Legacy flat-ops names whose suite2p 1.0 nested-settings counterpart
# uses a different key name. Callers can still pass the legacy name
# and ``apply_settings_overrides`` will translate it.
_LEGACY_KEY_ALIASES: dict = {
    'nbinned':          'nbins',
    'high_pass':        'highpass_time',
    'chan2_thres':      'chan2_threshold',
    'neucoeff':         'neuropil_coefficient',
    'pretrained_model': 'cellpose_model',
}

# Db-side keys (runtime registration outputs / paths). These never live
# in ``settings``; they belong on the ``db`` dict.
_DB_KEYS = frozenset({
    'data_path', 'look_one_level_down', 'input_format', 'keep_movie_raw',
    'nplanes', 'nrois', 'nchannels', 'swap_order', 'functional_chan',
    'lines', 'dy', 'dx', 'ignore_flyback', 'subfolders', 'file_list',
    'save_path0', 'fast_disk', 'save_folder', 'h5py_key', 'nwb_driver',
    'nwb_series', 'force_sktiff', 'bruker_bidirectional',
    'nframes', 'Ly', 'Lx', 'reg_file', 'reg_file_chan2',
    'raw_file', 'raw_file_chan2', 'save_path',
})


def _coerce_to_default_type(value, default):
    """Coerce ``value`` to ``type(default)`` when an upstream loader
    handed us a numpy float for what should be a bool/int.

    Suite2p 1.0 does arithmetic / ``range()`` on int settings and
    rejects floats. We only coerce numeric primitives; lists, strings,
    None, and unknown defaults pass through unchanged. Order matters:
    bool is a subclass of int, so check bool first.
    """
    if value is None or default is None:
        return value
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        try:
            return bool(value)
        except (TypeError, ValueError):
            return value
    if isinstance(default, int) and not isinstance(default, bool):
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        try:
            iv = int(value)
            return iv if float(iv) == float(value) else value
        except (TypeError, ValueError):
            return value
    return value


def default_db_settings() -> tuple[dict, dict]:
    """Return a fresh ``(db, settings)`` pair of suite2p 1.0 defaults."""
    from suite2p.parameters import default_db, default_settings
    return default_db(), default_settings()


def set_setting(settings: dict, key: str, value) -> None:
    """Write ``value`` to its canonical nested location in ``settings``.

    ``key`` may be a current suite2p 1.0 name (looked up in
    ``_SETTINGS_PATHS``) or a legacy flat-ops name (looked up in
    ``_LEGACY_KEY_ALIASES`` and translated). Unknown keys are silently
    dropped, matching suite2p's own behaviour.
    """
    real_key = _LEGACY_KEY_ALIASES.get(key, key)
    path = _SETTINGS_PATHS.get(real_key)
    if path is None:
        return
    node = settings
    for p in path:
        if p not in node:
            node[p] = {}
        node = node[p]
    node[real_key] = _coerce_to_default_type(value, node.get(real_key))


def get_setting(settings: dict, key: str, default=None):
    """Read ``settings`` at the nested location for ``key``."""
    real_key = _LEGACY_KEY_ALIASES.get(key, key)
    path = _SETTINGS_PATHS.get(real_key)
    if path is None:
        return default
    node = settings
    for p in path:
        if p not in node:
            return default
        node = node[p]
    return node.get(real_key, default)


def apply_settings_overrides(settings: dict, overrides: dict) -> None:
    """Bulk-apply a flat ``{name: value}`` dict to a nested settings.

    Supports the legacy flag aliases (``roidetect`` ->
    ``run.do_detection``, ``sparse_mode`` /
    ``anatomical_only`` -> ``detection.algorithm``). Unknown names
    are dropped silently.
    """
    for k, v in overrides.items():
        if k == 'roidetect':
            settings.setdefault('run', {})['do_detection'] = bool(v)
        elif k == 'spikedetect':
            settings.setdefault('run', {})['do_deconvolution'] = bool(v)
        elif k == 'anatomical_only' and v:
            settings.setdefault('detection', {})['algorithm'] = 'cellpose'
        elif k == 'sparse_mode':
            settings.setdefault('detection', {})['algorithm'] = (
                'sparsery' if v else 'sourcery'
            )
        else:
            set_setting(settings, k, v)


def apply_db_overrides(db: dict, overrides: dict) -> None:
    """Copy any ``_DB_KEYS`` entries from ``overrides`` into ``db``."""
    for k, v in overrides.items():
        if k in _DB_KEYS:
            db[k] = v
    for k in ('subfolders', 'file_list', 'fast_disk', 'ignore_flyback'):
        if db.get(k) in ([], '', ()):
            db[k] = None


def _ensure_s2p_logger(save_path=None):
    """Route suite2p 1.0's ``logging.getLogger('suite2p')`` output to
    the current ``sys.stderr`` (so Tab 3 picks it up via its
    ``contextlib.redirect_stderr`` capture) and tee a ``run.log`` file
    next to the suite2p output, mirroring suite2p's own
    ``run_s2p.logger_setup``.

    Why we don't just call ``logger_setup`` once
    --------------------------------------------
    Suite2p's helper installs a ``StreamHandler()`` that captures
    ``sys.stderr`` at construction time. Tab 3 redirects ``sys.stderr``
    to a fresh ``QueueWriter`` per run, so a one-shot setup binds the
    handler to a stale stream and the second run's log goes nowhere.
    Instead we maintain our own handler tagged with ``_calliope`` and
    rebind ``handler.stream`` on every call.
    """
    import logging
    import sys
    import pathlib

    s2p = logging.getLogger('suite2p')
    s2p.setLevel(logging.INFO)
    # Don't double-print via the root logger if anything else attaches
    # to it.
    s2p.propagate = False

    stream_h = next(
        (h for h in s2p.handlers
         if getattr(h, '_calliope', None) == 'stream'),
        None,
    )
    if stream_h is None:
        stream_h = logging.StreamHandler(sys.stderr)
        stream_h._calliope = 'stream'
        stream_h.setLevel(logging.INFO)
        stream_h.setFormatter(logging.Formatter("[s2p] %(message)s"))
        s2p.addHandler(stream_h)
    else:
        # Tab 3 swaps sys.stderr per run via contextlib.redirect_stderr;
        # rebind so this run's writer receives the log.
        stream_h.stream = sys.stderr

    if save_path is None:
        return
    save_dir = pathlib.Path(save_path)
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    log_file = save_dir / 'run.log'
    existing = next(
        (h for h in s2p.handlers
         if getattr(h, '_calliope', None) == 'file'),
        None,
    )
    if (existing is not None
            and getattr(existing, '_calliope_path', None) == str(log_file)):
        return
    if existing is not None:
        s2p.removeHandler(existing)
        existing.close()
    try:
        log_file.unlink()
    except FileNotFoundError:
        pass
    file_h = logging.FileHandler(log_file, mode='w')
    file_h._calliope = 'file'
    file_h._calliope_path = str(log_file)
    file_h.setLevel(logging.INFO)
    file_h.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    s2p.addHandler(file_h)


def _invoke_run_s2p(db: dict, settings: dict):
    """Dispatch to suite2p.run_s2p with logger setup and patches in place."""
    _ensure_s2p_logger(db.get('save_path0'))
    _patch_suite2p_estimate_spatial_scale()
    return suite2p.run_s2p(db=db, settings=settings)


def _invoke_run_plane(db: dict, settings: dict):
    """Dispatch to suite2p.run_plane with logger setup and patches in place."""
    _ensure_s2p_logger(db.get('save_path0'))
    _patch_suite2p_estimate_spatial_scale()
    return suite2p.run_plane(db, settings)


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
    """Raised from inside sparsery when the ROI count crosses the cap.

    Why a custom exception class
    ----------------------------
    Python lets you subclass ``Exception`` to carry typed state along
    with the error message. Doing so lets callers do
    ``except _RoiHardCapExceeded as e: ... e.count ...`` to recover
    the diagnostic numbers without parsing the string. Equivalent to
    R's ``simpleCondition`` / ``stop`` with a custom class slot.

    We raise this from inside Suite2p's own detection loop via the
    print-monkey-patch trick below, which lets us abort *mid-detection*
    when ``threshold_scaling`` is too low and Sparsery starts finding
    spurious noise blobs.
    """
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


# ----------------------------------------------------------------------------
# suite2p 1.0 estimate_spatial_scale shim
# ----------------------------------------------------------------------------
# In suite2p 1.0.0.1, ``sparsedetect.estimate_spatial_scale`` calls
# ``scipy.stats.mode(..., keepdims=True)``, which returns a 1-element ndarray
# instead of a scalar. That propagates into ``spatscale_pix = 3 * 2**scale``
# (now also an ndarray) and finally ``int(spatscale_pix * 1.5 // 2 * 2)``,
# which throws ``TypeError: only 0-dimensional arrays can be converted to
# Python scalars``. Triggered whenever ``settings.detection.sparsery_settings.
# spatial_scale == 0`` (auto-estimate).
#
# Wrap the upstream function so its return is coerced to a Python int. Idem-
# potent (only patches once); preserves the original via ``_calliope_orig``
# so we can restore for tests.

def _patch_suite2p_estimate_spatial_scale() -> None:
    """Coerce ``estimate_spatial_scale`` return to a Python int.

    Necessary because suite2p 1.0.0.1 + scipy >= 1.11 returns a (1,)
    ndarray from ``mode(..., keepdims=True)`` and downstream code does
    ``int(spatscale_pix * 1.5 // 2 * 2)`` which fails on non-scalar arrays.
    """
    import suite2p.detection.sparsedetect as _sd
    if getattr(_sd.estimate_spatial_scale, '_calliope_patched', False):
        return
    _orig = _sd.estimate_spatial_scale

    def _wrapped(I):
        result = _orig(I)
        # Return shape may be a Python int, a 0-d ndarray, or a (1,) ndarray
        # depending on suite2p version + scipy version. ``np.asarray(...).
        # item()`` handles all three.
        try:
            return int(np.asarray(result).item())
        except (TypeError, ValueError):
            # Older suite2p that returned a tuple-of-(im, count) shouldn't
            # land here, but if it does, pass through unchanged.
            return result

    _wrapped._calliope_patched = True
    _wrapped._calliope_orig = _orig
    _sd.estimate_spatial_scale = _wrapped


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class AdaptiveConfig:
    """All parameters controlling the adaptive Suite2p detection loop.

    Why this is a dataclass
    -----------------------
    The legacy adaptive pipeline takes ~50 tunable parameters --
    sparsery thresholds, registration reuse flags, residual-blob
    augmentation, spatial-scale fallback, audit-PNG generation, etc.
    Bundling them in a dataclass means:

    * Functions take one ``cfg`` argument instead of 50 keywords.
    * Defaults live with the field declarations so any forgotten
      parameter still has a sensible value.
    * Every field can carry its own inline comment (see below) so a
      reader can scroll through the dataclass to understand what
      each knob does without hunting through call sites.

    What still uses this in CalLIOPE
    --------------------------------
    The full adaptive loop is no longer wired into the GUI -- Tab 3
    calls ``sparse_plus_cellpose.run`` instead -- but the helpers
    that ``run`` reuses (``load_base_ops``, the shared-registration
    cache, ``run_one_pass``) all consume an ``AdaptiveConfig``.

    Most callers will only set the first few fields (paths + AAV
    metadata) and leave the rest at their defaults. The decay-mode
    schedule, residual-blob augmentation, etc. are exposed for
    backward compat with old scripts.
    """

    # ---- Required paths ----
    tiff_folder: str = ""                         # folder containing TIFFs
    save_folder: str = ""                         # where per-pass output lives
    path_to_ops: Optional[str] = None             # optional starting ops.npy

    # ---- Sample metadata (used for tau lookup) ----
    aav_info_csv: str = "human_SLE_2p_meta.csv"
    tau_vals: dict = field(default_factory=lambda: {
        "6f": 0.7, "6m": 1.0, "6s": 1.3, "8m": 0.137,
    })
    # If set, ``load_base_ops`` will use this tau directly and skip the
    # AAV CSV lookup. Tab 3 sets this when the user picks an explicit
    # GCaMP variant from the dropdown ("Auto (from AAV CSV)" leaves it
    # ``None`` so the legacy lookup runs).
    tau_override: Optional[float] = None

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

def load_base_settings(config: AdaptiveConfig, return_scale_data: bool = False):
    """Build a suite2p 1.0 ``(db, settings)`` pair tailored to the recording.

    Sources, in priority order
    --------------------------
    1. ``config.path_to_ops`` (a saved .npy of either a flat legacy
       ops dict or an already-nested settings dict) -- those values
       seed the settings via ``apply_settings_overrides``.
    2. Suite2p 1.0's :func:`suite2p.parameters.default_settings` if
       no override file is supplied.

    Either way the pipeline overlays:

    * **tau** : the GCaMP-variant decay constant. Looked up via
      ``utils.file_name_to_aav_to_dictionary_lookup`` from the AAV
      metadata CSV in ``config.aav_info_csv`` and the dict in
      ``config.tau_vals`` (or ``config.tau_override`` wins outright).
    * **batch_size** : auto-scaled to free RAM by
      ``utils.change_batch_according_to_free_ram``.
    * **spatial_scale** : optionally pre-estimated from the first
      ~500 frames via ``estimate_spatial_scale_from_tiff`` (a LoG
      blob detector that maps median blob diameter to a Suite2p
      scale bin).
    * **nbins** (legacy ``nbinned``) : capped via
      ``utils.change_nbinned_according_to_free_ram`` so the sparsery
      peak intermediate fits in RAM.

    Returns
    -------
    (db, settings) : tuple[dict, dict]
        Suite2p 1.0 nested config pair. ``db`` carries pipeline-input
        keys (``data_path`` etc.); ``settings`` carries everything
        else.
    scale_data : dict, optional
        Only returned when ``return_scale_data=True`` -- the
        per-scale blob inventory from the pre-flight LoG estimator,
        used downstream by ``run_blob_seeded_detection``.
    """
    db, settings = default_db_settings()

    # --- Step 1: seed from optional override file ---
    if config.path_to_ops is not None and os.path.exists(config.path_to_ops):
        if config.verbose:
            print(f"Loading base ops from {config.path_to_ops}")
        loaded = np.load(config.path_to_ops, allow_pickle=True).item()
        # The on-disk file might be either a legacy flat ops dict (from
        # pre-1.0 suite2p) or an already-nested settings/db pair stored
        # together. We treat it as a flat overrides bag either way:
        # apply_settings_overrides routes each known key to the right
        # nested location, apply_db_overrides catches the db-side ones.
        apply_db_overrides(db, loaded)
        apply_settings_overrides(settings, loaded)
    else:
        if config.verbose:
            print("No base ops provided, starting from suite2p defaults")
        # Data-specific defaults for our 2-photon GCaMP recordings.
        apply_settings_overrides(settings, {
            'fs': 15.07,
            'highpass_time': 100.0,
            'smooth_sigma': 1.3,
        })
        db['nchannels'] = 1
        db['nplanes'] = 1

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

    # --- Pipeline-required settings, applied unconditionally. preclassify
    # in particular is the junk-culling mechanism; without it, low-threshold
    # passes produce thousands of speckle ROIs.
    apply_settings_overrides(settings, {
        'sparse_mode':       True,
        'spatial_scale':     effective_spatial_scale,
        'preclassify':       0.5,
        'allow_overlap':     False,
    })

    # --- Sample-specific tau ---
    # Priority order:
    #   1. Explicit tau_override on the config.
    #   2. AAV CSV lookup keyed by recording filename.
    #   3. Whatever tau was already in settings.
    if config.tau_override is not None:
        settings['tau'] = float(config.tau_override)
        if config.verbose:
            print(f"Set tau={settings['tau']} from explicit override "
                  f"(skipping AAV lookup)")
    else:
        file_name = os.path.basename(os.path.normpath(config.tiff_folder))
        if os.path.exists(config.aav_info_csv):
            try:
                tau = utils.file_name_to_aav_to_dictionary_lookup(
                    file_name, config.aav_info_csv, config.tau_vals
                )
                settings['tau'] = tau
                if config.verbose:
                    print(f"Set tau={tau} based on AAV lookup for {file_name}")
            except Exception as e:
                if config.verbose:
                    print(f"tau lookup failed ({e}); keeping existing "
                          f"tau={settings.get('tau')}")

    # --- Dynamic batch size ---
    set_setting(settings, 'batch_size', utils.change_batch_according_to_free_ram())

    # --- nbins (legacy nbinned) capped to RAM budget ---
    safe_nbinned = utils.change_nbinned_according_to_free_ram(config.tiff_folder)
    current_nbinned = get_setting(settings, 'nbins', 0) or 0
    if current_nbinned <= 0:
        if config.verbose:
            print(f"Setting nbins = {safe_nbinned} (auto, based on available RAM)")
        set_setting(settings, 'nbins', safe_nbinned)
    elif current_nbinned > safe_nbinned:
        if config.verbose:
            print(f"Lowering nbins {current_nbinned} -> {safe_nbinned} to fit available RAM")
        set_setting(settings, 'nbins', safe_nbinned)

    if config.verbose:
        summary = (
            f"algorithm={settings['detection'].get('algorithm', '?')} "
            f"spatial_scale={get_setting(settings, 'spatial_scale')} "
            f"preclassify={get_setting(settings, 'preclassify')} "
            f"highpass_time={get_setting(settings, 'highpass_time')} "
            f"smooth_sigma={get_setting(settings, 'smooth_sigma')} "
            f"tau={settings.get('tau')} fs={settings.get('fs')} "
            f"nbins={get_setting(settings, 'nbins')} "
            f"batch_size={get_setting(settings, 'batch_size')}"
        )
        print(f"  effective settings: {summary}")

    if return_scale_data:
        return db, settings, scale_data
    return db, settings


# Backward-compat alias. Old call sites that imported ``load_base_ops``
# get a TypeError if they treat the result as a flat dict; the rename
# is intentional so the (db, settings) tuple can't be silently misused.
load_base_ops = load_base_settings


# ============================================================================
# ROI mask, residual image, blob detection
# ============================================================================

def build_roi_pixel_mask(stat, Ly: int, Lx: int) -> np.ndarray:
    """Binary ``(Ly, Lx)`` mask of all pixels belonging to any ROI in
    ``stat``.

    Used both as a deduplication tool (``filter_non_overlapping_cellpose``
    in ``sparse_plus_cellpose``) and as the foundation for "residual
    image" generation -- the mean image with every detected ROI's
    pixels masked out -- so we can run a follow-up blob detector on
    cells the first pass missed.

    ``stat`` is Suite2p's per-ROI dict list; each entry exposes
    ``ypix`` / ``xpix`` arrays.
    """
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


def _load_cached_pass(save_dir: Path, expected_settings: dict,
                      verbose: bool = True):
    """If ``save_dir`` already contains a completed suite2p run whose
    cached settings match ``expected_settings`` on detection-relevant
    keys, return ``(stat, cached_view, plane0)``. Otherwise return
    None.

    ``cached_view`` is a flat dict synthesized from the on-disk
    db.npy / settings.npy / reg_outputs / detect_outputs (or the
    legacy ops.npy when present) -- the same shape that
    ``utils.load_plane_view`` returns, so callers can read familiar
    keys like ``Ly`` / ``meanImg`` from it.
    """
    plane0 = Path(save_dir) / 'suite2p' / 'plane0'
    stat_path = plane0 / 'stat.npy'
    if not stat_path.exists():
        return None

    cached_view = utils.load_plane_view(plane0)
    if not cached_view:
        if verbose:
            print(f"  > cache miss at {plane0}: no settings/ops on disk")
        return None

    mismatches = []
    for k in _CACHE_COMPARE_KEYS:
        exp = get_setting(expected_settings, k)
        if exp is None:
            exp = expected_settings.get(k)
        cached = cached_view.get(k)
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
    return stat, cached_view, plane0


def _link_or_copy(src: Path, dst: Path):
    """Hardlink src -> dst, falling back to a real copy if the OS
    refuses (cross-device, missing FS support, no permissions, etc.).

    Why hardlink: registered movie files (``data.bin``) are huge (8+
    GiB for a 10-min recording). Copying them per detection pass
    burns time and disk; a hardlink is one inode reference -- no
    extra bytes. ``shutil.copy2`` is the slow safety net.
    """
    if dst.exists():
        return
    try:
        os.link(str(src), str(dst))
    except (OSError, NotImplementedError):
        shutil.copy2(str(src), str(dst))


def _get_or_create_shared_registration(config: 'AdaptiveConfig',
                                       db: dict, settings: dict) -> Path:
    """Ensure a single canonical registered binary exists at
    ``{save_folder}/_shared_reg/suite2p/plane0/`` and return that
    plane0 path.

    Why this exists
    ---------------
    Suite2p's motion correction (registration) is by far the most
    expensive step -- minutes per recording. The adaptive loop and
    ``sparse_plus_cellpose`` both want to run *detection* multiple
    times against the same registered movie (different threshold,
    different detector). Re-registering each time is wasteful, so
    this helper:

    1. Checks for an existing ``_shared_reg/suite2p/plane0/data.bin``.
    2. If present, returns its plane0 path immediately.
    3. Otherwise runs ``suite2p.run_s2p`` once with
       ``settings.run.do_detection=False`` (registration only, no
       detection) to produce the binary.

    Subsequent detection passes get the binary via ``_link_or_copy``
    -- one hardlink per pass folder, no copies.
    """
    save_root = Path(config.save_folder)
    shared_dir = save_root / '_shared_reg'
    shared_plane0 = shared_dir / 'suite2p' / 'plane0'

    # The legacy ops.npy is still written by suite2p when
    # settings.io.save_ops_orig=True (the default), so we still gate
    # on its presence as the "registration ran to completion" marker.
    if (shared_plane0 / 'data.bin').exists() and (shared_plane0 / 'ops.npy').exists():
        if config.verbose:
            print(f"  > reusing shared registration at {shared_plane0}")
        return shared_plane0

    # A previous attempt may have crashed mid-write, leaving a
    # ``plane0/`` with db.npy / settings.npy but no usable data.bin.
    # Suite2p 1.0's ``_find_existing_binaries`` would then match on the
    # partial folder and skip the binary-conversion step entirely,
    # silently reusing whatever stale settings.npy was on disk. Wipe
    # the directory so register-only starts clean.
    if shared_dir.exists():
        if config.verbose:
            print(f"  > clearing partial shared registration at {shared_dir}")
        shutil.rmtree(shared_dir, ignore_errors=True)

    if config.verbose:
        print(f"  > no shared registration yet; running suite2p register-only "
              f"into {shared_dir}")

    reg_db = dict(db)
    reg_db['data_path'] = [config.tiff_folder]
    reg_db['save_path0'] = str(shared_dir)
    reg_db['save_folder'] = 'suite2p'

    reg_settings = _deep_copy_settings(settings)
    reg_settings.setdefault('run', {})['do_detection'] = False

    _invoke_run_s2p(reg_db, reg_settings)

    if not (shared_plane0 / 'data.bin').exists():
        raise RuntimeError(
            f"shared registration completed but data.bin was not written to "
            f"{shared_plane0}. Check suite2p output for errors."
        )
    return shared_plane0


def _deep_copy_settings(settings: dict) -> dict:
    """Recursively copy a settings dict (one level of nesting is enough
    for suite2p's schema). Top-level dict and any nested dict values
    are shallow-copied so mutating the copy doesn't bleed back.
    """
    out = {}
    for k, v in settings.items():
        if isinstance(v, dict):
            out[k] = _deep_copy_settings(v)
        else:
            out[k] = v
    return out


def run_one_pass(tiff_folder: str, save_dir: Path,
                 db: dict, settings: dict,
                 verbose: bool = True, use_cache: bool = True,
                 shared_plane0: Optional[Path] = None,
                 hard_cap: int = 0):
    """Run Suite2p once with the given ``(db, settings)`` and return
    its detection output.

    Steps
    -----
    1. (Optional) compare ``settings`` with any cached settings.npy
       already in ``save_dir``; reuse the cached ``stat.npy`` if the
       detection-controlling keys all match.
    2. If a ``shared_plane0`` was passed (the shared-registration
       cache), hardlink its ``data.bin`` into ``save_dir`` and call
       ``suite2p.run_plane`` (which skips registration entirely).
    3. (Optional) install the ROI-hard-cap monkey-patch on
       ``suite2p.detection.sparsedetect.print`` so a sparsery run
       that finds more than ``hard_cap`` ROIs aborts mid-iteration.
    4. Otherwise dispatch to ``suite2p.run_s2p`` for the full
       TIFF -> binary -> registration -> detection -> extraction flow.

    Returns
    -------
    stat : list[dict]
        The per-ROI detection output.
    view : dict
        A ``utils.load_plane_view``-shaped flat dict of the run's
        on-disk db / settings / reg / detect outputs (so callers can
        still read ``view['Ly']`` / ``view['meanImg']``).
    plane0 : Path
        ``<save_dir>/suite2p/plane0``.
    """
    db = dict(db)
    db['save_path0'] = str(save_dir)
    db['save_folder'] = 'suite2p'

    if use_cache:
        cached = _load_cached_pass(save_dir, settings, verbose=verbose)
        if cached is not None:
            return cached

    plane0 = Path(save_dir) / 'suite2p' / 'plane0'

    # --- Registration-reuse path ---------------------------------------
    if shared_plane0 is not None:
        plane0.mkdir(parents=True, exist_ok=True)
        # Hardlink the registered binary; data.bin is read-only downstream.
        _link_or_copy(Path(shared_plane0) / 'data.bin', plane0 / 'data.bin')

        # Pull the shared registration's runtime fields onto our db so
        # run_plane sees the right Ly/Lx/nframes/etc. We start from the
        # caller's db (has nchannels, nplanes, save_path0, ...), then
        # layer the cached registration outputs on top.
        shared_view = utils.load_plane_view(shared_plane0)
        plane_db = dict(db)
        for k in ('Ly', 'Lx', 'nframes', 'nchannels'):
            if k in shared_view:
                plane_db[k] = shared_view[k]
        plane_db['save_path0'] = str(save_dir)
        plane_db['save_folder'] = 'suite2p'
        plane_db['save_path'] = str(plane0)
        plane_db['reg_file'] = str(plane0 / 'data.bin')
        # raw_file from the shared register-only run points at the
        # _shared_reg folder; we only hardlinked data.bin. Force
        # keep_movie_raw=False so suite2p's raw-file existence check
        # doesn't blow up on the missing path.
        plane_db.pop('raw_file', None)
        plane_db.pop('raw_file_chan2', None)
        plane_db['keep_movie_raw'] = False

        # Ensure we're actually doing detection on this plane.
        plane_settings = _deep_copy_settings(settings)
        plane_settings.setdefault('run', {})['do_registration'] = False
        plane_settings['run'].setdefault('do_detection', True)

        if verbose:
            print(f"  > running suite2p detection-only (shared registration): "
                  f"threshold_scaling={get_setting(plane_settings, 'threshold_scaling')}  "
                  f"spatial_scale={get_setting(plane_settings, 'spatial_scale')}  "
                  f"max_iterations={get_setting(plane_settings, 'max_iterations')}")

        _orig_print = (_install_sparsery_roi_cap(hard_cap)
                       if hard_cap and hard_cap > 0 else None)
        try:
            _invoke_run_plane(plane_db, plane_settings)
        except _RoiHardCapExceeded as e:
            if verbose:
                print(f"  > ABORT: {e} -- returning empty pass")
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
            return [], utils.load_plane_view(plane0), plane0
        except ValueError as e:
            if 'no ROIs' in str(e) or 'ROIs were found' in str(e):
                if verbose:
                    print("  > suite2p found 0 ROIs for these params; "
                          "returning empty pass")
                try:
                    plane0.mkdir(parents=True, exist_ok=True)
                    np.save(plane0 / 'stat.npy',
                            np.array([], dtype=object), allow_pickle=True)
                except Exception:
                    pass
                return [], utils.load_plane_view(plane0), plane0
            raise
        finally:
            if _orig_print is not None:
                _restore_sparsery_print(_orig_print)

        stat = list(np.load(plane0 / 'stat.npy', allow_pickle=True))
        return stat, utils.load_plane_view(plane0), plane0

    # --- Full run_s2p path (registers + detects) -----------------------
    full_db = dict(db)
    full_db['data_path'] = [tiff_folder]

    if verbose:
        print(f"  > running suite2p: "
              f"threshold_scaling={get_setting(settings, 'threshold_scaling')}  "
              f"max_iterations={get_setting(settings, 'max_iterations')}")

    _orig_print = (_install_sparsery_roi_cap(hard_cap)
                   if hard_cap and hard_cap > 0 else None)
    try:
        _invoke_run_s2p(full_db, settings)
    except _RoiHardCapExceeded as e:
        if verbose:
            print(f"  > ABORT: {e} -- returning empty pass")
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
        return [], utils.load_plane_view(plane0), plane0
    except ValueError as e:
        if 'no ROIs' in str(e) or 'ROIs were found' in str(e):
            if verbose:
                print(f"  > suite2p found 0 ROIs for these params; "
                      f"returning empty pass")
            try:
                plane0.mkdir(parents=True, exist_ok=True)
                np.save(plane0 / 'stat.npy',
                        np.array([], dtype=object), allow_pickle=True)
            except Exception:
                pass
            return [], utils.load_plane_view(plane0), plane0
        raise
    finally:
        if _orig_print is not None:
            _restore_sparsery_print(_orig_print)

    stat = list(np.load(plane0 / 'stat.npy', allow_pickle=True))
    return stat, utils.load_plane_view(plane0), plane0


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
                              base_db: dict, base_settings: dict, blobs_yxr):
    """
    Register the data with suite2p, then bypass sparsery: build stat from
    the provided blob list and run suite2p's extraction + classification
    + deconvolution manually.

    Returns ``(stat, view, plane0)`` -- same shape as ``run_one_pass``.
    """
    from suite2p.detection.stats import roi_stats
    from suite2p.extraction import extract, dcnv
    from suite2p import classification

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    plane0 = save_dir / 'suite2p' / 'plane0'

    # Step 1: register only (skip sparsery entirely)
    reg_db = dict(base_db)
    reg_db['data_path'] = [config.tiff_folder]
    reg_db['save_path0'] = str(save_dir)
    reg_db['save_folder'] = 'suite2p'
    reg_settings = _deep_copy_settings(base_settings)
    reg_settings.setdefault('run', {})['do_detection'] = False

    if config.verbose:
        print(f"  > blob-seeded detection: running suite2p registration "
              f"only (do_detection=False)")

    _invoke_run_s2p(reg_db, reg_settings)

    # Load the synthesized view that suite2p produced (Ly, Lx, reg_file, ...)
    ops = utils.load_plane_view(plane0)
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
                              base_plane0, shared_plane0, save_dir):
    """Deferred: residual-blob augmentation pass.

    The implementation pre-suite2p-1.0-refactor mutated a flat ops dict
    in place. It needs to be ported to the new (db, settings) model,
    but isn't wired into the GUI; it's safe to leave deferred until a
    follow-up PR. Callers that depend on the legacy adaptive loop
    (none in the current GUI) get a clear error.
    """
    raise NotImplementedError(
        "run_residual_augmentation is deferred during the suite2p 1.0 "
        "(db, settings) refactor; rewrite to take (base_db, base_settings) "
        "before re-enabling. The active GUI path uses sparse_plus_cellpose."
    )


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


def _run_binary_search_schedule(*args, **kwargs):
    """Deferred during the suite2p 1.0 (db, settings) refactor."""
    raise NotImplementedError(
        "_run_binary_search_schedule is deferred during the suite2p 1.0 "
        "(db, settings) refactor; the binary-search adaptive loop is not "
        "wired into the current GUI. Use sparse_plus_cellpose.run instead."
    )


def _run_under_detection_schedule(*args, **kwargs):
    """Deferred during the suite2p 1.0 (db, settings) refactor."""
    raise NotImplementedError(
        "_run_under_detection_schedule is deferred during the suite2p 1.0 "
        "(db, settings) refactor."
    )


def _run_over_detection_schedule(*args, **kwargs):
    """Deferred during the suite2p 1.0 (db, settings) refactor."""
    raise NotImplementedError(
        "_run_over_detection_schedule is deferred during the suite2p 1.0 "
        "(db, settings) refactor."
    )


def adaptive_detect(config: AdaptiveConfig):
    """Deferred during the suite2p 1.0 (db, settings) refactor.

    The full adaptive binary-search detection loop relied on mutating
    a flat ops dict pervasively (see git history before this commit).
    Porting it to the nested (db, settings) model is straightforward
    but voluminous, and the GUI no longer wires this entry point --
    Tab 3 calls ``sparse_plus_cellpose.run`` instead. This function
    will be re-enabled in a follow-up PR; until then, it raises so
    accidental callers fail fast.
    """
    raise NotImplementedError(
        "adaptive_detect is deferred during the suite2p 1.0 (db, settings) "
        "refactor; use sparse_plus_cellpose.run for the active GUI path."
    )


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
    """Three-panel "did detection do something sane?" audit figure.

    The three panels in order
    -------------------------
    1. Mean image with detected-ROI contours outlined in green --
       answers "are the ROIs we found in plausible cell-shaped
       locations?"
    2. Residual image (mean with every ROI's pixels masked out) on
       a magma colormap -- answers "what did detection miss? Bright
       blobs in the residual are likely cells that need a second
       pass."
    3. Mean image again, with the blobs detected on the residual
       drawn as cyan circles -- the proposed missed cells, ready for
       sanity check.

    Works even if the pass has zero ROIs (aborted / empty pass)
    -- panel 1 degrades gracefully and panel 3 still shows every
    residual blob. ``sparse_plus_cellpose.run`` calls this at the
    end of each detection so the user can scroll through a folder of
    audit PNGs without re-opening matplotlib.
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