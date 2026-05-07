"""Logic + calculations for the QC Preview tab.

Re-export shim
--------------
The QC tab itself is mostly a GIF viewer + a few overlay buttons; it
just needs the ``PreprocessResult`` type for type hints (each tab
subscribes to ``state.set_result`` and receives one of these) and
``load_existing_preprocess`` for the "Reload from folder" button.
The actual GIF / mean-image building lives in
``calliope.core.preprocessing``.

See the docstring on ``calliope.tabs.preprocess.logic`` for the full
explanation of the shim pattern.
"""

from __future__ import annotations

from ...core.preprocessing import PreprocessResult, load_existing_preprocess

__all__ = ["PreprocessResult", "load_existing_preprocess"]
