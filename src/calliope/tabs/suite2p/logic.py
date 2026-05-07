"""Logic + calculations for the Suite2p detection tab.

Re-export shim
--------------
Tab 3 is the heaviest tab in the GUI -- it pulls in Suite2p, Cellpose
and the cell-filter PyTorch model. To keep import time low for users
who never touch detection (e.g. they're loading an existing recording
to look at clusters), we lazy-import the detection functions inside
the worker thread rather than at module load. This file only needs:

- ``PreprocessResult``: the type hint for ``state.result`` -- Tab 3
  reads ``state.result.shifted_tiff`` to know what to feed to Suite2p.
- ``summary_writer``: writes the per-ROI table to the recording's
  ``calliope_summary.xlsx`` once detection completes.

See ``calliope.tabs.preprocess.logic`` for the shim pattern overview.
"""

from __future__ import annotations

from ...core import summary_writer
from ...core.preprocessing import PreprocessResult

__all__ = ["PreprocessResult", "summary_writer"]
