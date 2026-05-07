"""Logic + calculations for the Event detection tab.

Re-export shim
--------------
Tab 5 needs:

- The per-ROI onset detector (``mad_z``, ``hysteresis_onsets``).
- The population-event detector
  (``EventDetectionParams``, ``detect_event_windows``).
- Diagnostic plotting helpers (``plot_event_detection``,
  ``shade_event_windows``).
- ``get_fps_from_notes`` for reading the recording's frame rate.
- ``summary_writer`` for stamping the EventWindows / EventOnsets
  sheets into ``calliope_summary.xlsx`` after every render.

The tab module imports this as ``utils`` (yes, despite also being
named ``logic`` -- the ``as`` rename keeps the math-y call sites
short).

See ``calliope.tabs.preprocess.logic`` for the shim pattern overview.
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
