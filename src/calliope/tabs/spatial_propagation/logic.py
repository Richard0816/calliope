"""Logic + calculations for the Spatial propagation tab.

Re-export shim
--------------
Tab 8 is a *consumer* of Tab 5's results -- it doesn't run any
detection of its own, just paints the events Tab 5 published via
``AppState.event_results``. To paint, it needs:

- ``spatial`` -- ``CYAN_TO_RED`` colormap, ``order_map_for_event``,
  ``paint_order_map``, ``event_frame_centroids``.
- A handful of ``utils`` symbols only as type re-exports (the
  current Tab 8 implementation doesn't run detection but earlier
  versions did, so the imports are kept for backward compat).

See ``calliope.tabs.preprocess.logic`` for the shim pattern overview.
"""

from __future__ import annotations

from ...core import spatial
from ...core.utils import (
    EventDetectionParams,
    detect_event_windows,
    get_fps_from_notes,
    hysteresis_onsets,
    mad_z,
)

__all__ = [
    "EventDetectionParams",
    "detect_event_windows",
    "get_fps_from_notes",
    "hysteresis_onsets",
    "mad_z",
    "spatial",
]
