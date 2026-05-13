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
                             QC GIF, plus the headless
                             ``run_preprocess`` entry point Tab 0's
                             batch runner calls.
- ``sparse_plus_cellpose`` - merge Suite2p (``Sparsery``) ROIs with a
                             Cellpose pass on the mean image.
- ``utils``               - dF/F + neuropil correction, lowpass +
                             Savitzky-Golay derivative, MAD-z
                             hysteresis onset detection,
                             density-based event detection,
                             ``paint_spatial`` ROI->image painter,
                             ``EventDetectionParams`` dataclass.
- ``crosscorrelation``    - batched (one matmul per lag) cluster x
                             cluster cross-correlation + the
                             headless ``run_crosscorrelation``
                             wrapper that drives the full + per-event
                             runs and renders the violin plots.
- ``clustering``          - palettes, auto-threshold helpers,
                             dendrogram + spatial plotters, and the
                             headless ``run_clustering`` orchestrator
                             that exports cluster ROI lists and the
                             Clusters summary sheet.
- ``spatial``             - cyan->blue->red colormap, per-event
                             activation-rank painting, frame-grouped
                             centroids (Tab 8) plus the headless
                             ``render_spatial_event_figures`` entry
                             point Tab 0's batch runner calls.
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
                             / Clusters / EventMonotonicity sheets)
                             used by tabs 5+.
- ``cellfilter``          - PyTorch CNN that scores each ROI as
                             cell / not-cell on a mix of a 32x32
                             spatial crop and the trace timeseries.
- ``detection_run`` / ``lowpass_run`` / ``event_detection_run``
                            -- per-stage headless entry points for
                             Tab 0's batch pipeline (Tabs 3 / 4 / 5).
- ``batch_pipeline``      - end-to-end orchestrator that strings the
                             headless entry points together (Tab 0).
"""
