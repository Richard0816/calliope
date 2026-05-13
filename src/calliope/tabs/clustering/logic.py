"""Logic + calculations for the Clustering tab.

Re-export shim
--------------
Tab 6 imports three slices of ``calliope.core``:

- ``clustering`` -- the actual hierarchical clustering algorithm
  (``run_clustering``, ``auto_choose_threshold``, ``count_clusters``)
  plus palette helpers and the dendrogram / spatial-map plotters.
- ``summary_writer`` -- writes the Clusters sheet to
  ``calliope_summary.xlsx``.
- ``utils`` -- ``get_fps_from_notes`` and ``paint_spatial`` for the
  spatial map below the dendrogram.

Re-exporting whole modules (rather than picking individual names)
keeps the call sites readable: ``clustering.run_clustering(...)``
is more navigable than a dozen unbound symbol imports.

See ``calliope.tabs.preprocess.logic`` for the shim pattern overview.

Backwards-compat alias: ``clustering_cmap`` is exposed as an alias of
``clustering`` so older import paths (``from .logic import
clustering_cmap as cmap_mod``) keep working through the refactor.
"""

from __future__ import annotations

from ...core import clustering, summary_writer, utils

# Backwards-compat alias for the pre-merge module name. The merged
# module is now ``calliope.core.clustering``; the old name was
# ``clustering_cmap``. Tab 6 still imports it under the ``cmap_mod``
# alias so the rebind keeps the call sites working unchanged.
clustering_cmap = clustering

__all__ = ["clustering", "clustering_cmap", "summary_writer", "utils"]
