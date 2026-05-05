"""Logic + calculations for the Suite2p detection tab.

Re-exports the slice of ``calliope.core`` modules the tab calls
(`PreprocessResult` for state typing, `summary_writer` for writing the
ROIs sheet to ``calliope_summary.xlsx``).

The detection worker also lazy-imports ``sparse_plus_cellpose`` from the
project root inside the worker thread; that import resolves via the
``sys.path`` bootstrap in ``calliope/__init__.py`` so we don't have to
copy its (heavy) cellpose / suite2p dependency graph into the package.
"""

from __future__ import annotations

from ...core import summary_writer
from ...core.preprocessing import PreprocessResult

__all__ = ["PreprocessResult", "summary_writer"]
