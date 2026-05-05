"""calliope - GUI package for the calcium imaging pipeline.

Top-level coordinator: ``calliope.pipeline_gui``. Each tab lives in its
own subfolder under ``calliope.tabs.<name>`` with a ``tab`` module
holding the Tk class and a ``logic`` module wrapping the calculations.
Shared backend modules (preprocessing, summary writer, signal-processing
utils, hierarchical-clustering palette helpers, cross-correlation,
cellpose-based detection, and the cell-filter PyTorch model) live under
``calliope.core``. Bundled resource files (suite2p ops, AAV metadata)
ship in ``calliope.data``.

Run the GUI with ``python -m calliope``.
"""

__version__ = "0.1.0"
