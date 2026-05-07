"""Logic + calculations for the Preprocess tab.

Why this file is so short
-------------------------
This is a *re-export shim*. The tab's UI lives in ``tab.py`` next to
this file; the heavy lifting (intensity-shifting TIFFs, building
mean images, blob-detecting candidate cells, generating QC GIFs)
lives in ``calliope.core.preprocessing``. The shim exists so the
tab module can write::

    from . import logic as preprocessing
    preprocessing.preprocess_tiff(...)

and keep its call sites unchanged even if the underlying functions
move around inside ``core``.

Pattern note
------------
Every tab folder follows the same convention: ``tab.py`` for GUI
widgets, ``logic.py`` as a thin re-export of the calculations it
needs, ``__init__.py`` mostly empty, and a ``README.md`` describing
the tab's role.

The ``__all__`` list at the bottom is Python's way of saying "these
are the public names of this module". When someone does
``from calliope.tabs.preprocess.logic import *`` they get exactly
the names listed; ``__all__`` is only consulted for ``import *`` --
explicit imports like ``from logic import preprocess_tiff`` always
work regardless.
"""

# Type-hint forward-reference shim (see pipeline_gui.py).
from __future__ import annotations

# Three-dot relative import: go up TWO levels from ``tabs/preprocess/``
# to ``calliope/``, then into ``core.preprocessing``. The dots count
# package levels, not directories.
from ...core.preprocessing import (
    PreprocessResult,
    list_tiffs,
    load_existing_preprocess,
    preprocess_tiff,
    preprocess_tiff_group,
)

__all__ = [
    "PreprocessResult",
    "list_tiffs",
    "load_existing_preprocess",
    "preprocess_tiff",
    "preprocess_tiff_group",
]
