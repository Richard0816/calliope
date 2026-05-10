"""Pure-Python algorithms shared across the GUI tabs.

Nothing in here knows about Tk or any GUI; every function takes plain
arrays / paths and returns plain arrays / paths. The GUI tabs over in
``calliope.tabs.<name>`` import these helpers and present their
results in figures and dialogs.

If you have an R background: think of ``core`` as the equivalent of an
R package's ``R/`` folder where the actual statistics live, and
``tabs`` as a Shiny app that calls those helpers from event
callbacks. Splitting the two layers means the math can be tested and
re-used outside the GUI.

Module index
------------
- ``preprocessing``       - intensity-shift TIFF stacks, mean image,
                             LoG blob detection, QC GIF.
- ``sparse_plus_cellpose`` - merge Suite2p (``Sparsery``) ROIs with a
                             Cellpose pass on the mean image.
- ``utils``               - dF/F + neuropil correction, lowpass +
                             Savitzky-Golay derivative, MAD-z
                             hysteresis onset detection,
                             density-based event detection,
                             ``paint_spatial`` ROI->image painter,
                             ``EventDetectionParams`` dataclass.
- ``crosscorrelation``    - batched (one matmul per lag) cluster x
                             cluster cross-correlation.
- ``clustering_cmap``     - palettes + auto-threshold helpers for
                             hierarchical clustering.
- ``spatial``             - cyan->blue->red colormap, per-event
                             activation-rank painting, frame-grouped
                             centroids (used by tab 8).
- ``suite2p_pipeline``    - all suite2p contact: ``run_one_pass``,
                             ``run_cellpose_pass``,
                             ``_get_or_create_shared_registration``,
                             ``load_base_settings``, plus the
                             monkey-patches that work around suite2p
                             1.0 quirks. Imported only by
                             ``sparse_plus_cellpose``.
- ``calliope_settings``   - in-source nested defaults for the lab's
                             2-photon rig + popout flatten/nest
                             helpers.
- ``summary_writer``      - writes ``calliope_summary.xlsx``
                             (Recording / EventOnsets / EventWindows
                             sheets) used by tabs 5+.
- ``cellfilter``          - PyTorch CNN that scores each ROI as
                             cell / not-cell on a mix of a 32x32
                             spatial crop and the trace timeseries.
"""
