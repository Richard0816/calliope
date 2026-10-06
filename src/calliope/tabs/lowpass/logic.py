"""Logic + calculations for the Low-pass filter tab.

Re-export shim
--------------
Tab 4's own module only needs four helpers from ``calliope.core.utils``
for loading and previewing: ``resolve_filtered_mask`` (size the keep
mask against the filtered memmap on disk), ``get_fps_from_notes`` /
``ops_fs`` (frame rate and where it came from), and
``lowpass_zero_phase_1d`` (the panel-3 preview filter). The tab imports
this file as ``utils`` so it can call them with their original names.

The causal filter + Savitzky-Golay derivative that Compute writes to
disk (``lowpass_causal_1d`` / ``sg_first_derivative_1d``) run inside
``calliope.core.lowpass_run``, which imports them from ``core.utils``
directly, so they are not re-exported here.

See ``calliope.tabs.preprocess.logic`` for the shim pattern overview.
"""

from __future__ import annotations

from ...core.utils import (
    get_fps_from_notes,
    lowpass_zero_phase_1d,
    ops_fs,
    resolve_filtered_mask,
)

__all__ = [
    "get_fps_from_notes",
    "lowpass_zero_phase_1d",
    "ops_fs",
    "resolve_filtered_mask",
]
