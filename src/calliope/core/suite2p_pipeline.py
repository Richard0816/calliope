"""Suite2p 1.0 invocation helpers for CalLIOPE.

This is the single home for everything that talks to suite2p directly.
``sparse_plus_cellpose.py`` (Tab 3's worker) imports from here; nothing
else in the package should import suite2p outside of this module.

The pipeline has three layers:

1. **Settings construction** -- :func:`load_base_settings` builds a
   suite2p 1.0 ``(db, settings)`` pair from
   :func:`calliope.core.calliope_settings.build_base_settings` (the
   in-source defaults), then deep-merges any per-session
   ``settings_override`` from Tab 3's "Edit suite2p settings..."
   popout. Per-recording dynamics (``tau`` from the AAV CSV,
   RAM-tuned ``batch_size`` / ``nbins``) are layered on top.

2. **Suite2p invocation** -- :func:`_invoke_run_s2p` and
   :func:`_invoke_run_plane` are the only call sites for
   ``suite2p.run_s2p`` and ``suite2p.run_plane``. Both install the
   ``logging.getLogger('suite2p')`` stream handler and the
   ``estimate_spatial_scale`` ndarray-return monkey-patch before
   dispatching.

3. **Pipeline orchestration** --
   :func:`_get_or_create_shared_registration` runs suite2p
   register-only once into ``_shared_reg/`` so subsequent detection
   passes can hardlink the registered ``data.bin``;
   :func:`run_one_pass` is the single-pass detector. The cellpose
   side is :func:`run_cellpose_pass`, which segments the mean image
   so we can union its ROIs with sparsery's set.

Settings routing helpers (``set_setting`` / ``get_setting`` /
``apply_settings_overrides`` / ``apply_db_overrides`` /
``default_db_settings``) translate flat legacy ops keys (the format
suite2p 0.x used and the GUI tabs still produce in PARAM_SPEC) to
the nested 1.0 schema.

Two monkey-patches are installed: :func:`_install_sparsery_roi_cap`
(abort sparsery mid-pass once ROI count crosses a cap) and
:func:`_patch_suite2p_estimate_spatial_scale` (coerce suite2p's broken
``mode(..., keepdims=True)`` ndarray return back to a Python int).
"""

from __future__ import annotations

import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import suite2p

from . import utils


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


def _is_cuda_oom(exc: BaseException) -> bool:
    """True if ``exc`` is a CUDA out-of-memory error.

    torch raises this as ``torch.cuda.OutOfMemoryError`` (subclass of
    ``RuntimeError``) but older torch versions raise bare
    ``RuntimeError`` with ``"out of memory"`` in the message. Cover
    both shapes.
    """
    try:
        import torch
    except ImportError:
        return False
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    if isinstance(exc, RuntimeError) and \
            "out of memory" in str(exc).lower():
        return True
    return False


def _clear_cuda_arena() -> None:
    """Best-effort empty the torch CUDA allocator's free-list so the
    OOM retry sees a clean memory map.

    Useful between attempts because the allocator holds onto freed
    blocks for reuse -- without an explicit reset, a retry can hit
    the same fragmentation that caused the first OOM even when the
    smaller batch_size would have fit in a fresh allocator.
    """
    try:
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _invoke_run_s2p(db: dict, settings: dict):
    """Dispatch to suite2p.run_s2p with logger setup and patches in place.

    Wraps the call in :class:`utils.PeakMemoryMonitor` and feeds the
    observed peak back to :class:`utils.AdaptiveBatchSizer` so the next
    recording's batch_size converges on the target RAM fraction.

    CUDA OOM safety net: if registration blows the GPU, halve
    ``settings['registration']['batch_size']``, clear the CUDA arena,
    and retry once. The closed loop's recording-2 update then has
    real feedback to converge from. Up to two retries (8x reduction
    from the initial estimate) before giving up -- past that the
    user has a deeper problem than batch_size tuning.
    """
    _ensure_s2p_logger(db.get('save_path0'))
    _patch_suite2p_estimate_spatial_scale()
    sizer = utils.get_adaptive_sizer()
    used_bs = get_setting(settings, 'batch_size')
    max_oom_retries = 2

    for attempt in range(max_oom_retries + 1):
        mon = utils.PeakMemoryMonitor()
        mon.__enter__()
        try:
            result = suite2p.run_s2p(db=db, settings=settings)
        except BaseException as exc:
            mon.__exit__(type(exc), exc, exc.__traceback__)
            if _is_cuda_oom(exc) and attempt < max_oom_retries:
                new_bs = max(64, used_bs // 2)
                print(f"[adaptive] CUDA OOM at batch_size={used_bs} "
                      f"(attempt {attempt + 1}/{max_oom_retries + 1}); "
                      f"halving to {new_bs} and retrying")
                set_setting(settings, 'batch_size', new_bs)
                # Record the OOM as a "100% GPU peak" data point so
                # the closed loop knows this batch_size was too big
                # even though we recover via retry. Use the just-
                # halved value as the batch_size we'll actually run
                # at on the retry.
                sizer.update_from_peak(
                    cpu_peak=mon.cpu_peak_fraction,
                    gpu_peak=1.0,
                    batch_size=new_bs)
                used_bs = new_bs
                _clear_cuda_arena()
                continue
            # Non-OOM or out of retries: record what we saw and
            # bubble up.
            sizer.update_from_peak(
                cpu_peak=mon.cpu_peak_fraction,
                gpu_peak=mon.gpu_peak_fraction,
                batch_size=used_bs)
            print(f"[adaptive] suite2p run_s2p FAILED: "
                  f"cpu peak={mon.cpu_peak_fraction:.1%} "
                  f"gpu peak={mon.gpu_peak_fraction:.1%} "
                  f"batch_size_used={used_bs}"
                  + (f" (after {attempt} OOM retries)"
                     if attempt > 0 else ""))
            raise
        # Success path.
        mon.__exit__(None, None, None)
        sizer.update_from_peak(
            cpu_peak=mon.cpu_peak_fraction,
            gpu_peak=mon.gpu_peak_fraction,
            batch_size=used_bs)
        print(f"[adaptive] suite2p run_s2p: "
              f"cpu peak={mon.cpu_peak_fraction:.1%} "
              f"gpu peak={mon.gpu_peak_fraction:.1%} "
              f"batch_size_used={used_bs}"
              + (f" (after {attempt} OOM retries)"
                 if attempt > 0 else ""))
        return result


def _invoke_run_plane(db: dict, settings: dict):
    """Dispatch to suite2p.run_plane with logger setup and patches in place.

    Same peak-RAM monitor + adaptive feedback wrapping as
    :func:`_invoke_run_s2p`. ``run_plane`` skips registration so its
    batch_size doesn't matter for memory, but the detection phase still
    benefits from the closed loop on subsequent recordings.
    """
    _ensure_s2p_logger(db.get('save_path0'))
    _patch_suite2p_estimate_spatial_scale()
    sizer = utils.get_adaptive_sizer()
    used_bs = get_setting(settings, 'batch_size')
    with utils.PeakMemoryMonitor() as mon:
        try:
            return suite2p.run_plane(db, settings)
        finally:
            sizer.update_from_peak(
                cpu_peak=mon.cpu_peak_fraction,
                gpu_peak=mon.gpu_peak_fraction,
                batch_size=used_bs)
            print(f"[adaptive] suite2p run_plane: "
                  f"cpu peak={mon.cpu_peak_fraction:.1%} "
                  f"gpu peak={mon.gpu_peak_fraction:.1%} "
                  f"batch_size_used={used_bs}")


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
class Suite2pPipelineConfig:
    """Inputs for the suite2p invocation helpers in this module.

    Consumers:

    * ``load_base_settings`` reads paths, AAV/tau settings, the popout
      override dict, and the legacy ``path_to_ops`` back-compat file.
    * ``_get_or_create_shared_registration`` reads ``tiff_folder`` and
      ``save_folder``.
    * ``run_one_pass`` reads ``use_cache`` and ``verbose``.
    """

    # ---- Required paths ----
    tiff_folder: str = ""                         # folder containing TIFFs
    save_folder: str = ""                         # where per-pass output lives
    # Optional legacy override .npy. Either a flat pre-1.0 ops dict or a
    # nested 1.0 settings dict; ``load_base_settings`` detects and merges
    # appropriately. Leave None to use the in-source CALLIOPE_BASE_SETTINGS.
    path_to_ops: Optional[str] = None

    # ---- Per-session settings overrides (from Tab 3 popout) ----
    # Nested suite2p 1.0 settings dict. Deep-merged on top of the in-source
    # defaults from ``calliope.core.calliope_settings.build_base_settings``
    # by ``load_base_settings``. Tab 3's "Edit suite2p settings..." popout
    # constructs this from the user's edits and stuffs it on the cfg.
    settings_override: Optional[dict] = None

    # ---- Sample metadata (used for tau lookup) ----
    aav_info_csv: str = "human_SLE_2p_meta.csv"
    tau_vals: dict = field(default_factory=lambda: {
        "6f": 0.7, "6m": 1.0, "6s": 1.3, "8m": 0.137,
    })
    # If set, ``load_base_settings`` uses this tau directly and skips the
    # AAV CSV lookup. Tab 3 sets this when the user picks an explicit
    # GCaMP variant from the dropdown ("Auto (from AAV CSV)" leaves it
    # ``None`` so the legacy lookup runs).
    tau_override: Optional[float] = None

    # ---- Caching ----
    # When True, each pass checks its save_dir for an existing completed
    # suite2p run with matching settings and reuses it instead of rerunning.
    # Delete the pass directory to force a rerun of just that pass.
    use_cache: bool = True

    # ---- Misc ----
    verbose: bool = True



# ============================================================================
# Ops loading / preparation
# ============================================================================

# Top-level keys that distinguish a suite2p 1.0 nested settings dict from a
# legacy flat-ops dict. If any of these appear at the top level *and* their
# value is itself a dict, the file is nested.
_NESTED_SETTINGS_TOP_KEYS = frozenset((
    'run', 'io', 'registration', 'detection',
    'classification', 'extraction', 'dcnv_preprocess',
))


def _looks_like_nested_settings(loaded: dict) -> bool:
    """Return True if ``loaded`` is a suite2p 1.0 nested settings dict.

    Heuristic: at least one of the canonical sub-system keys
    (``run``, ``detection``, ``registration``, ...) must be present
    *as a dict*. A legacy flat-ops file has these keys absent (or, in
    the rare case ``detection`` was a scalar, not a dict).
    """
    if not isinstance(loaded, dict):
        return False
    return any(
        isinstance(loaded.get(k), dict)
        for k in _NESTED_SETTINGS_TOP_KEYS
    )


def _deep_merge(dst: dict, src: dict) -> None:
    """In-place deep-merge ``src`` into ``dst``.

    Keys in ``src`` overwrite ``dst``. Nested dicts are merged
    recursively; leaves are replaced wholesale (no list-append etc.).
    """
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v


def load_base_settings(config: Suite2pPipelineConfig):
    """Build a suite2p 1.0 ``(db, settings)`` pair tailored to the recording.

    Base settings come from :mod:`calliope.core.calliope_settings` (the
    in-source hardcoded dict that replaced the legacy
    ``updated_settings.npy``). When the user has tweaked the popout
    "Edit suite2p settings..." dialog in Tab 3, ``config.settings_override``
    carries those tweaks and gets deep-merged on top.

    For backward-compat, ``config.path_to_ops`` still works: if set, it's
    treated as a flat legacy ops dict (pre-1.0 suite2p) or an already-
    nested settings dict, and its values overlay the hardcoded base. New
    code should prefer the in-source dict + popout edits.

    The pipeline then overlays per-recording dynamics:

    * **tau** : the GCaMP-variant decay constant. Looked up via
      ``utils.file_name_to_aav_to_dictionary_lookup`` from the AAV
      metadata CSV in ``config.aav_info_csv`` and the dict in
      ``config.tau_vals`` (or ``config.tau_override`` wins outright).
    * **batch_size** : picked by ``utils.get_adaptive_sizer().pick()``
      -- frame-size aware initial estimate for the first recording in
      a queue, closed-loop scaling against the previous run's observed
      peak RAM thereafter. Target fraction defaults to
      ``utils.RAM_TARGET_FRACTION`` (0.80).
    * **spatial_scale** : pinned to 0 (suite2p auto-pick); the broken
      ``estimate_spatial_scale`` ndarray bug is patched at
      ``_invoke_run_*`` dispatch time via
      :func:`_patch_suite2p_estimate_spatial_scale`.
    * **nbins** (legacy ``nbinned``) : capped via
      ``utils.change_nbinned_according_to_free_ram`` so the sparsery
      peak intermediate fits in RAM.

    Returns
    -------
    (db, settings) : tuple[dict, dict]
        Suite2p 1.0 nested config pair. ``db`` carries pipeline-input
        keys (``data_path`` etc.); ``settings`` carries everything
        else.
    """
    from . import calliope_settings as _cs

    # --- Step 0: seed from the in-source calliope base settings ---
    db = _cs.build_base_db()
    settings = _cs.build_base_settings()
    db['nchannels'] = 1
    db['nplanes'] = 1

    # --- Step 1: optional Tab 3 popout overrides (per-session edits) ---
    settings_override = getattr(config, 'settings_override', None)
    if settings_override:
        _deep_merge(settings, settings_override)

    # --- Step 2: optional legacy override file (back-compat) ---
    if config.path_to_ops is not None and os.path.exists(config.path_to_ops):
        if config.verbose:
            print(f"Loading base ops from {config.path_to_ops}")
        loaded = np.load(config.path_to_ops, allow_pickle=True).item()
        # The on-disk file might be either:
        #   (a) a legacy flat ops dict (pre-1.0 suite2p, e.g.
        #       suite2p_2p_ops_240621.npy) -- treat as flat overrides
        #       and route each known key to its nested location;
        #   (b) an already-nested suite2p 1.0 settings dict (legacy
        #       updated_settings.npy) -- deep-merge directly.
        if _looks_like_nested_settings(loaded):
            _deep_merge(settings, loaded)
        else:
            apply_db_overrides(db, loaded)
            apply_settings_overrides(settings, loaded)

    # --- Pipeline-required settings, applied unconditionally. preclassify
    # in particular is the junk-culling mechanism; without it, low-threshold
    # passes produce thousands of speckle ROIs. spatial_scale=0 lets
    # suite2p auto-pick (calliope's monkey-patch coerces the result to int).
    apply_settings_overrides(settings, {
        'sparse_mode':       True,
        'spatial_scale':     0,
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

    # --- Dynamic batch size (frame-size aware + closed-loop in batch mode) ---
    # First recording in a queue: frame-size-aware initial estimate.
    # Recordings 2..N: scaled by the previous run's observed peak RAM
    # so we converge on ``utils.RAM_TARGET_FRACTION``. Single-shot Tab 3
    # calls just get the initial estimate every time (no feedback loop
    # because there's no recording 2 to apply it to).
    set_setting(settings, 'batch_size',
                utils.get_adaptive_sizer().pick(config.tiff_folder))

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

    return db, settings


# Backward-compat aliases. ``AdaptiveConfig`` was the dataclass name
# back when this module hosted the adaptive binary-search loop;
# ``load_base_ops`` was the pre-1.0 entry point that returned a flat
# ops dict. The new names are ``Suite2pPipelineConfig`` and
# ``load_base_settings``. Aliases let old scripts keep importing.
AdaptiveConfig = Suite2pPipelineConfig
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


def _get_or_create_shared_registration(config: 'Suite2pPipelineConfig',
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
        # data_path needs to be a non-empty list -- run_plane indexes it
        # at [0] for the bad_frames.npy lookup. Prefer the value the
        # shared register-only run wrote to its db.npy; fall back to
        # the caller-supplied tiff_folder.
        shared_data_path = shared_view.get('data_path')
        if isinstance(shared_data_path, (list, tuple)) and shared_data_path:
            plane_db['data_path'] = list(shared_data_path)
        else:
            plane_db['data_path'] = [str(tiff_folder)]
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
# Cellpose pass (segments the mean image; output unioned with sparsery)
# ============================================================================
# Cellpose is optional. If it's not installed, ``run_cellpose_pass`` raises
# at call time with a clear message. The sentinels are checked at import
# time so we can fail fast on a misconfigured environment.

try:
    from cellpose import models as _cp_models  # type: ignore
    _CELLPOSE_AVAILABLE = True
    _CELLPOSE_IMPORT_ERROR = None
except Exception as _e:                                              # pragma: no cover
    _cp_models = None
    _CELLPOSE_AVAILABLE = False
    _CELLPOSE_IMPORT_ERROR = repr(_e)


def _cellpose_masks_to_stat(masks: np.ndarray) -> list[dict]:
    """Convert a cellpose integer label mask into suite2p-style stat dicts.

    Cellpose returns masks as shape (Ly, Lx) with 0=background and 1..N
    as per-cell labels. We emit one dict per label containing the keys
    ``build_roi_pixel_mask`` / extract / classify expect: ``ypix``,
    ``xpix``, ``lam``, ``med``, ``npix``, ``radius``.
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
    3. Run ``CellposeModel.eval``. Feature-detects the Cellpose
       major version and adjusts the call accordingly.
    4. Convert each integer mask label into a Suite2p-style
       ``stat`` dict (ypix / xpix / lam / med / npix) so downstream
       merging code can treat Cellpose ROIs identically to Sparsery
       ones.
    5. Save ``stat.npy`` + ``cellpose_masks.npy`` for audit.

    Returns ``(stat, ops_out, plane0)`` with the same shape/semantics
    as ``run_one_pass`` so downstream merge code works unchanged.

    The plane0 directory contains:
      - ops.npy             (copy of the shared registered ops)
      - stat.npy            (cellpose-derived suite2p-style stat list)
      - cellpose_masks.npy  (raw integer-label mask, for audit)
    """
    if not _CELLPOSE_AVAILABLE:
        raise RuntimeError(
            f"cellpose not importable: {_CELLPOSE_IMPORT_ERROR}")

    plane0 = Path(save_dir) / "suite2p" / "plane0"
    plane0.mkdir(parents=True, exist_ok=True)

    # Load the shared (post-registration) view to get meanImg + Ly/Lx.
    # In suite2p 1.0 these live in db.npy / settings.npy / reg_outputs.npy
    # which load_plane_view stitches into a flat lookup (with a fallback
    # to a legacy ops.npy on older runs).
    reg_ops = utils.load_plane_view(shared_plane0)
    img_key = cfg.get("cellpose_channel_input", "meanImg")
    if img_key not in reg_ops or reg_ops[img_key] is None:
        # Fallback: meanImg is always present after registration.
        img_key = "meanImg"
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

    model_type = cfg.get("cellpose_model_type", "cyto")
    diameter = cfg.get("cellpose_diameter", 0) or None   # 0 -> auto
    flow_threshold = float(cfg.get("cellpose_flow_threshold", 0.4))
    cellprob_threshold = float(cfg.get("cellpose_cellprob_threshold", 0.0))

    # Feature-detect cellpose API version.
    #   Cellpose <=3.x: ``models.Cellpose`` wrapper + ``CellposeModel``;
    #       eval accepts ``channels`` and returns
    #       (masks, flows, styles, diams).
    #   Cellpose 4.x : ``Cellpose`` wrapper removed; only ``CellposeModel``;
    #       eval no longer takes ``channels`` and returns
    #       (masks, flows, styles). Also ``model_type`` is gone -- one
    #       unified model.
    has_wrapper = hasattr(_cp_models, "Cellpose")
    api_version = "3x" if has_wrapper else "4x"

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
        # between cyto/cyto2/nuclei -- there's only the unified cpsam
        # model. We still record the requested type in ops for audit.
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

    # Persist outputs in a suite2p-like layout so the rest of the
    # pipeline + any downstream audit tools can find them.
    ops_out = dict(reg_ops)
    ops_out["Ly"] = int(img.shape[0])
    ops_out["Lx"] = int(img.shape[1])
    ops_out["cellpose_config"] = {
        "api_version": api_version,
        "model_type": model_type,
        "diameter_requested": cfg.get("cellpose_diameter", 0),
        "diameter_estimated": float(diams) if diams is not None else None,
        "flow_threshold": flow_threshold,
        "cellprob_threshold": cellprob_threshold,
        "channel_input": img_key,
    }
    np.save(plane0 / "ops.npy", ops_out, allow_pickle=True)
    np.save(plane0 / "stat.npy", np.array(stat, dtype=object),
            allow_pickle=True)
    np.save(plane0 / "cellpose_masks.npy", np.asarray(masks))

    return stat, ops_out, plane0
