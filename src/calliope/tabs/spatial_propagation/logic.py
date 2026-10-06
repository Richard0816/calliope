"""Logic + calculations for the Spatial propagation tab.

Re-export shim
--------------
Tab 8 is a *consumer* of Tab 5's results -- it doesn't run any
detection of its own, just paints the events Tab 5 published via
``AppState.event_results``. To paint, it needs only:

- ``spatial`` -- ``CYAN_TO_RED`` colormap, ``order_map_for_event``,
  ``paint_order_map``, ``event_frame_centroids``,
  ``directional_monotonicity_spearman``.

See ``calliope.tabs.preprocess.logic`` for the shim pattern overview.
"""

from __future__ import annotations

from ...core import spatial

__all__ = ["spatial"]
