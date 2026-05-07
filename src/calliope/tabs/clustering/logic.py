"""Logic + calculations for the Clustering tab.

Re-export shim
--------------
Tab 6 imports three slices of ``calliope.core``:

- ``clustering_cmap`` -- the actual hierarchical clustering
  algorithm (``run_clustering``, ``recluster_branch``,
  ``auto_choose_threshold``) + palette helpers.
- ``summary_writer`` -- writes the Clusters sheet to
  ``calliope_summary.xlsx``.
- ``utils`` -- ``get_fps_from_notes`` and ``paint_spatial`` for the
  spatial map below the dendrogram.

Re-exporting whole modules (rather than picking individual names)
keeps the call sites readable: ``clustering_cmap.run_clustering(...)``
is more navigable than a dozen unbound symbol imports.

See ``calliope.tabs.preprocess.logic`` for the shim pattern overview.
"""

from __future__ import annotations

from ...core import clustering_cmap, summary_writer, utils

__all__ = ["clustering_cmap", "summary_writer", "utils"]
