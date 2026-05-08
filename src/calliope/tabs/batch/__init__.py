"""Tab 0: batch runner.

Drives the full Tabs 3-8 pipeline over a list of recordings, one
per row. The actual analysis lives in ``calliope.core.batch_pipeline``;
this package only owns the GUI.
"""

from .tab import BatchTab

__all__ = ["BatchTab"]
