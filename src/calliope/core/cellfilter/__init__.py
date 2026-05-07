"""Cell-filter CNN: PyTorch model that scores each Suite2p ROI as
"this is a real cell" vs "this is a spurious blob" on a 0..1 scale.

Sub-modules
-----------
- ``config``  - hyper-parameters in one place (training, model size).
- ``model``   - the two-branch network: a small CNN over a 32x32
                 ROI crop + a 1D temporal stream over the dF/F trace.
- ``dataset`` - PyTorch ``Dataset`` that pulls the per-ROI crops and
                 traces into training tensors.
- ``train``   - training loop (BCE + pos-weight, AUROC selection,
                 learning-rate scheduler, checkpointing).
- ``predict`` - inference helper used by Tab 3 to write the per-ROI
                 keep-mask alongside the Suite2p output.

This file is intentionally almost empty -- importing
``calliope.core.cellfilter`` should not pull in PyTorch unless one of
the sub-modules above is also imported. Tabs that don't run the CNN
(everything except Tab 3 and the optional retraining script) therefore
pay no startup cost for the dependency.
"""
