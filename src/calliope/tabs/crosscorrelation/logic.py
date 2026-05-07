"""Logic + calculations for the Cross-correlation tab.

Re-export shim
--------------
Tab 7 needs:

- ``crosscorrelation`` (renamed to ``xc`` because the call sites
  use it constantly: ``xc.batch_xcorr_clusters``,
  ``xc.run_cluster_xcorr_full``, etc).
- ``utils`` for ``get_fps_from_notes``.

The ``import ... as ...`` form lets us alias a module at import
time. R analogue: ``library(stats)`` then writing ``stats::lm(...)``
-- only here we abbreviate the prefix.

See ``calliope.tabs.preprocess.logic`` for the shim pattern overview.
"""

from __future__ import annotations

from ...core import crosscorrelation as xc
from ...core import utils

__all__ = ["xc", "utils"]
