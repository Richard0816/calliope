"""Logic + calculations for the Low-pass filter tab.

Re-export shim
--------------
Tab 4 only needs three signal-processing functions from
``calliope.core.utils``: ``lowpass_causal_1d`` (the Butterworth SOS
filter), ``sg_first_derivative_1d`` (the Savitzky-Golay first
derivative), and ``get_fps_from_notes`` (to look up the frame rate).
The tab module imports this file as ``utils`` so it can call them
with their original names.

See ``calliope.tabs.preprocess.logic`` for the shim pattern overview.
"""

from __future__ import annotations

from ...core.utils import (
    get_fps_from_notes,
    lowpass_causal_1d,
    lowpass_zero_phase_1d,
    sg_first_derivative_1d,
)

__all__ = [
    "get_fps_from_notes",
    "lowpass_causal_1d",
    "lowpass_zero_phase_1d",
    "sg_first_derivative_1d",
]
