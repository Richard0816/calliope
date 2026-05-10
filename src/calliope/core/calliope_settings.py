"""Hardcoded base suite2p settings for the lab's 2-photon GCaMP rig.

This module replaces the previous ``src/calliope/data/updated_settings.npy``
on-disk file. The data lives in source code now so:

    - Settings changes are version-controlled diffs you can review in PRs
      instead of opaque .npy churn.
    - Editors and IDEs can jump to the value definition.
    - The pipeline has zero filesystem dependency for its base settings; it
      can be imported from a notebook or batch script with no side effects.

Tab 3's "Edit suite2p settings..." popout reads :func:`build_base_settings`
to seed its form, then writes the user's tweaks into the per-run
``settings`` dict before dispatch.

The intent is the same as the legacy
``src/calliope/data/suite2p_2p_ops_240621.npy`` (which was a flat-ops dict
in pre-1.0 suite2p format) but expressed in suite2p 1.0's nested settings
schema. Keys not listed here inherit from
``suite2p.parameters.default_settings()`` at build time.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


# ---------------------------------------------------------------------------
# Calliope's overrides on top of suite2p 1.0 default_settings()
# ---------------------------------------------------------------------------
#
# Edit values in place; they get deep-merged onto suite2p's defaults at
# runtime by :func:`build_base_settings`. To add a new override, place it at
# the correct nested location matching suite2p 1.0's settings schema.
#
# Notes on the chosen values:
#     * ``tau`` defaults to GCaMP6f (0.137 s); per-recording AAV lookup
#       overrides this when the recording is in the AAV CSV.
#     * ``fs`` is the lab's 2-photon scope's nominal frame rate; the tab
#       overrides per recording from the notes.txt sidecar when present.
#     * ``registration.batch_size`` is much larger than suite2p's default
#       because the rig has plenty of RAM for the registration buffer; the
#       runtime ``change_batch_according_to_free_ram`` helper trims it
#       further if needed.
#     * ``detection.threshold_scaling`` 0.8 (vs default 1.0) catches the
#       dimmer somas typical of slice recordings.
#     * ``detection.cellpose_settings.cellpose_model`` is pinned to the
#       legacy ``cyto`` model (the brute-force winner from earlier
#       calibration); suite2p's default ``cpsam`` is too aggressive.

CALLIOPE_BASE_SETTINGS: dict[str, Any] = {
    # Top-level
    "tau":     0.137,
    "fs":      15.07,
    "diameter": 0,

    # Registration overrides
    "registration": {
        "nimg_init":       300,
        "batch_size":      4000,
        "block_size":      [128, 128],
        "spatial_taper":   20.0,
        "smooth_sigma":    1.15,
    },

    # Detection overrides
    "detection": {
        "highpass_time":     50.0,
        "threshold_scaling": 0.8,
        "chan2_threshold":   0.65,
        "cellpose_settings": {
            "cellpose_model":  "cyto",
            "flow_threshold":  1.5,
        },
    },
}


def _deep_merge(dst: dict, src: dict) -> dict:
    """In-place deep-merge ``src`` into ``dst`` and return ``dst``.

    Lists and scalars in ``src`` replace whatever was in ``dst``; nested
    dicts are merged recursively.
    """
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def build_base_settings() -> dict:
    """Return a fresh suite2p 1.0 settings dict seeded with calliope overrides.

    The result is safe to mutate (deep-copied from
    :data:`CALLIOPE_BASE_SETTINGS`). The caller is responsible for any
    additional per-run overlays (recording-specific ``tau``, dynamic
    ``batch_size`` from free RAM, etc.) that
    :func:`suite2p_pipeline.load_base_settings` performs.
    """
    from suite2p.parameters import default_settings
    settings = default_settings()
    _deep_merge(settings, deepcopy(CALLIOPE_BASE_SETTINGS))
    return settings


def build_base_db() -> dict:
    """Return a fresh suite2p 1.0 db dict seeded with calliope defaults.

    Mirror of :func:`build_base_settings` for the db side. Currently just
    returns ``default_db()``; if/when the lab grows db-side overrides
    (multi-plane, multi-channel, etc.), add a ``CALLIOPE_BASE_DB`` dict
    above and merge it here.
    """
    from suite2p.parameters import default_db
    return default_db()


# ---------------------------------------------------------------------------
# Settings-popout helpers (flatten <-> nest)
# ---------------------------------------------------------------------------

# Heading shown in the popout for any leaf at the top level of settings.
_TOP_LEVEL_GROUP = "Top-level"


def _infer_type(value: Any) -> str:
    """Return the AdvancedDialog type tag matching a runtime value.

    Booleans must be checked before ints because ``bool`` is a subclass
    of ``int`` in Python. ``None`` and lists fall through to ``str`` so
    the user can type literals (e.g. ``[128, 128]``) into a free-form
    Entry; :func:`coerce_value_for_path` parses them back.
    """
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    # list / tuple / None / nested unknown -- render as text. The user
    # can edit JSON-ish syntax; we'll ast.literal_eval on commit.
    return "str"


def flatten_settings_to_specs(
    base: dict,
    current: dict | None = None,
) -> tuple[list[dict], dict]:
    """Walk a nested settings dict and produce ``(specs, flat_values)``.

    ``specs`` is an :class:`AdvancedDialog`-compatible list with one
    entry per leaf, using a dot-path as the entry's ``name`` (e.g.
    ``detection.threshold_scaling``) and the top-level subsystem as the
    ``group`` (``Top-level`` for direct root leaves).

    ``flat_values`` is a ``{dot_path: value}`` dict carrying the *current*
    values: ``current`` (deep-merged onto ``base``) when supplied,
    else ``base`` itself. The dialog will mutate this in place; the
    caller pipes it through :func:`nest_flat_values` to recover a nested
    overrides dict.
    """
    merged = deepcopy(base)
    if current:
        _deep_merge(merged, current)

    specs: list[dict] = []
    flat: dict = {}

    def _walk(node: dict, path_parts: tuple[str, ...]) -> None:
        for key, value in node.items():
            new_path = path_parts + (key,)
            if isinstance(value, dict):
                _walk(value, new_path)
                continue
            dot = ".".join(new_path)
            group = (path_parts[0] if path_parts else _TOP_LEVEL_GROUP)
            spec_type = _infer_type(value)
            # Python literals (None, lists) get serialised to repr so the
            # entry box round-trips through ast.literal_eval.
            if isinstance(value, (list, tuple)) or value is None:
                shown_default = repr(value)
            else:
                shown_default = value
            specs.append({
                "name":    dot,
                "label":   key,
                "type":    spec_type,
                "default": shown_default,
                "group":   group,
            })
            flat[dot] = shown_default

    _walk(merged, ())
    return specs, flat


def coerce_value_for_path(raw: Any, base_value: Any) -> Any:
    """Convert a popout-edited value back to the type ``base_value`` uses.

    The :class:`AdvancedDialog` already handles ``int``/``float``/``bool``
    coercion. This helper picks up the freeform-string cases (lists,
    tuples, ``None``) by routing through :func:`ast.literal_eval`.
    """
    import ast
    if isinstance(base_value, bool):
        return bool(raw)
    if isinstance(base_value, int) and not isinstance(base_value, bool):
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            return base_value
    if isinstance(base_value, float):
        try:
            return float(raw)
        except (TypeError, ValueError):
            return base_value
    if isinstance(base_value, (list, tuple)) or base_value is None:
        if isinstance(raw, (list, tuple)) or raw is None:
            return raw
        if isinstance(raw, str):
            stripped = raw.strip()
            if stripped == "" or stripped.lower() == "none":
                return None
            try:
                return ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                return base_value
        return raw
    # Plain strings: pass through.
    return raw


def nest_flat_values(
    flat: dict,
    base: dict,
) -> dict:
    """Turn a ``{dot.path: value}`` dict into a nested overrides dict.

    Only entries whose value differs from the corresponding leaf in
    ``base`` are kept -- that way the override dict carries just the
    user's edits, and downstream callers can deep-merge it back onto a
    fresh ``build_base_settings()`` to reproduce the user-visible state
    on the next run.
    """
    overrides: dict = {}
    for dot, value in flat.items():
        parts = dot.split(".")
        # Look up the corresponding leaf in ``base`` so we can:
        #   (1) coerce string-typed entry back to the right Python type,
        #   (2) skip the entry if the user didn't actually change it.
        node: Any = base
        try:
            for p in parts:
                node = node[p]
        except (KeyError, TypeError):
            node = None
        coerced = coerce_value_for_path(value, node)
        if coerced == node:
            continue
        # Walk down (creating sub-dicts as needed) and store the leaf.
        out_node = overrides
        for p in parts[:-1]:
            if p not in out_node or not isinstance(out_node[p], dict):
                out_node[p] = {}
            out_node = out_node[p]
        out_node[parts[-1]] = coerced
    return overrides
