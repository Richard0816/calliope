"""Logic + calculations for the Low-pass filter tab.

Re-exports the slice of ``calliope.core.utils`` the tab calls
(``get_fps_from_notes``, ``lowpass_causal_1d``, ``sg_first_derivative_1d``).
The tab module imports this as ``utils`` so its existing call sites
(``utils.lowpass_causal_1d(...)``) keep working.
"""

from __future__ import annotations

from ...core.utils import (
    get_fps_from_notes,
    lowpass_causal_1d,
    sg_first_derivative_1d,
)

__all__ = [
    "get_fps_from_notes",
    "lowpass_causal_1d",
    "sg_first_derivative_1d",
]
