"""Logic + calculations for the Cross-correlation tab.

Re-exports the slice of ``calliope.core.crosscorrelation`` (as ``xc``)
and ``calliope.core.utils`` the tab calls.
"""

from __future__ import annotations

from ...core import crosscorrelation as xc
from ...core import utils

__all__ = ["xc", "utils"]
