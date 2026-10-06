"""Tab 0: batch runner.

Drives the full Tabs 1 + 3-8 pipeline over a list of recordings, one
per row, by driving the GUI tabs themselves (each tab's own run
method, sequenced through AppState completion signals). Non-GUI
helpers (scratch prune / mirror, union PARAM_SPEC) live in
``logic.py``. The separate headless orchestrator
``calliope.core.batch_pipeline`` is not used by this tab.
"""

from .tab import BatchTab

__all__ = ["BatchTab"]
