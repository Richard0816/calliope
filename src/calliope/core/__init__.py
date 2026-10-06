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
                             1.0 quirks. Driven by
                             ``sparse_plus_cellpose``; Tab 0's batch
                             runner also imports it for its OOM-retry
                             settings handling.
- ``calliope_settings``   - in-source nested defaults for the lab's
                             2-photon rig (incl. the ``DEFAULT_FPS``
                             fallback) + popout flatten/nest helpers.
- ``summary_writer``      - writes ``calliope_summary.xlsx``
                             (Recording / ROIs / EventWindows /
                             EventOnsets / RoiEventTimes /
                             EventMonotonicity / Clusters sheets;
                             a FilterParams writer is reserved and
                             currently unused).
- ``cellfilter``          - PyTorch CNN that scores each ROI as
                             cell / not-cell on a mix of a 32x32
                             spatial crop and the trace timeseries.
- ``detection_run`` / ``lowpass_run`` / ``event_detection_run``
                            -- per-stage headless entry points
                             (Tabs 3 / 4 / 5) that Tab 0's batch
                             runner calls.
- ``batch_pipeline``      - headless end-to-end helper
                             (``run_recording``) that strings the
                             stage entry points together, plus small
                             batch guards (``_no_rois_survive``,
                             ``_kept_roi_count``). The GUI's Tab 0 does
                             NOT call ``run_recording``; it drives the
                             stages itself.
- ``analyze_output_gpu``  - CuPy dF/F path
                             (GPU twin of the CPU dF/F writer).
- ``_cuda_import``        - the single place the optional CuPy
                             dependency is imported (and probed).
- ``correlation_image``   - per-pixel temporal correlation image
                             (Vcorr) for the Cellpose-SAM pass.
- ``zdrift``              - PCA-based z-drift QC (flags, never drops).
- ``recording_meta``      - per-recording ``meta.json`` provenance
                             sidecar (atomic write, safe read).
- ``export_manifest``     - ``manifest.json`` written next to exported
                             figures (git SHA, params, tau/fs, cell
                             counts).
- ``nwb_export``          - optional NWB export (``[nwb]`` extra).
- ``scale`` / ``figscale`` - pixel<->µm maths (matplotlib-free) and
                             the matching scale-bar drawing helper.
- ``offload``             - run heavy compute in a child process so
                             the Tk GUI stays responsive.
- ``run_loader``          - discover/inventory a completed run on disk
                             for "Open run folder...".
"""
