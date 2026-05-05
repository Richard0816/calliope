"""Logic + calculations for the Event detection tab.

Re-exports the slice of ``calliope.core.utils`` and
``calliope.core.summary_writer`` the tab calls. The tab module imports
this as ``utils`` so existing call sites (``utils.detect_event_windows``
etc.) stay unchanged.
"""

from __future__ import annotations

from ...core import summary_writer
from ...core.utils import (
    EventDetectionParams,
    detect_event_windows,
    get_fps_from_notes,
    hysteresis_onsets,
    mad_z,
    plot_event_detection,
    shade_event_windows,
)

__all__ = [
    "EventDetectionParams",
    "detect_event_windows",
    "get_fps_from_notes",
    "hysteresis_onsets",
    "mad_z",
    "plot_event_detection",
    "shade_event_windows",
    "summary_writer",
]
