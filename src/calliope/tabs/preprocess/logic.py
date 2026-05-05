"""Logic + calculations for the Preprocess tab.

Re-exports the slice of ``calliope.core.preprocessing`` the tab calls,
so the tab module can ``from . import logic as preprocessing`` and keep
its existing call sites unchanged.
"""

from __future__ import annotations

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
