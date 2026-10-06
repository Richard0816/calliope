"""Logic + calculations for the Cross-correlation tab.

Re-export shim
--------------
Tab 7 needs:

- ``crosscorrelation`` (renamed to ``xc`` because the call sites
  use it constantly: ``xc.single_pair_xcorr_curve``,
  ``xc.run_cluster_xcorr_offload``, etc).
- ``utils`` for ``get_fps_from_notes`` (called without ``notes_root``,
  so it returns ``ops.npy['fs']`` or ``DEFAULT_FPS``) and
  ``DEFAULT_FPS``.

The ``import ... as ...`` form lets us alias a module at import
time. R analogue: ``library(stats)`` then writing ``stats::lm(...)``
-- only here we abbreviate the prefix.

See ``calliope.tabs.preprocess.logic`` for the shim pattern overview.
"""

from __future__ import annotations

from ...core import crosscorrelation as xc
from ...core import utils

__all__ = ["xc", "utils"]
