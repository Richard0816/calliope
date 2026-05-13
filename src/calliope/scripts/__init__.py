"""Standalone helper scripts that ship with CalLIOPE.

These modules are CLI-callable entry points for tasks the
pipeline GUI doesn't host directly. Run them with
``python -m calliope.scripts.<name>`` so the package import works
without needing an editable install.

Current scripts
---------------
- ``roi_curation_app`` -- a single-window ROI curation UI that
  loads one Suite2p plane0 folder, scores every ROI with the
  current cell-filter checkpoint, and lets the user step through
  ROIs (sorted by classifier uncertainty by default) flipping
  cell / not-cell labels. Writes to the in-project
  ``cellfilter_labels.csv`` and the recording's ``iscell.npy``.
"""
