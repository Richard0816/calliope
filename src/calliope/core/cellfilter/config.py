"""Central config for the cell-filter CNN.

Why a separate config file
--------------------------
Every other module in ``cellfilter`` (``model``, ``dataset``,
``train``, ``predict``) imports its hyperparameters from here. That
way training and inference are guaranteed to use exactly the same
patch size, trace length, and architecture -- if you change the
``PATCH_SIZE`` constant below you don't have to remember to update
the model definition in ``model.py``, the dataset loader in
``dataset.py``, etc.

R analogy: think of this as the ``options()`` block at the top of an
analysis script that downstream functions read with ``getOption()``,
except using plain Python module-level constants instead of a global
options list.

Sections
--------
- **Paths** - locations on the lab's F:\\ drive of the curation CSV
  (the human-labelled cell-vs-noise ground truth), the recording
  data, and where to save model checkpoints.
- **Data** - patch size used by both training and inference, plus
  training-only knobs (validation fraction, random seed, trace
  crop length).
- **Model** - architecture sizes for the temporal and spatial
  branches of the CNN.
- **Training** - optimisation knobs (batch size, learning rate,
  weight decay, early-stopping patience).
- **Inference** - decision threshold and the file names the
  ``predict`` step writes into ``plane0/``.
"""
from pathlib import Path

# --- Paths ---
# Curation CSV: one row per (recording, roi_id) with a 0/1 label.
# The training pipeline reads this to decide which ROIs are "cells".
LABELS_CSV = Path(r"F:\roi_curation.csv")
# Where the lab keeps registered TIFF stacks + their Suite2p outputs.
DATA_ROOT = Path(r"F:\data\2p_shifted")  # contains Cx\ and Hip\
# Extra roots searched recursively (any depth) for a folder named <rec_id>
# that contains suite2p/plane0. First match wins. Useful when a
# recording has been moved to a different drive.
EXTRA_DATA_ROOTS = [
    Path(r"D:\data"),
]
# Where ``train.py`` saves model weights at end-of-epoch and at the
# best AUROC seen so far.
CHECKPOINT_DIR = Path(r"F:\cellfilter_checkpoints")

# --- Data ---
DFF_PREFIX = "r0p7_"         # neuropil-corrected dF/F memmap prefix.
PATCH_SIZE = 32              # spatial patch edge, pixels.
TRACE_CROP_LEN = 2000        # random crop length for training (frames).
VAL_FRAC = 0.20              # fraction of recordings held out for validation.
RANDOM_SEED = 0              # reproducible split between train / val.

# --- Model ---
# Channel widths for each conv block in the temporal branch
# (1D ResNet-style on the dF/F trace). Powers of 2 for GPU
# friendliness.
TEMPORAL_CHANNELS = (16, 32, 64)
# Same idea for the 2D spatial branch (the 32x32 patch).
SPATIAL_CHANNELS = (16, 32, 64)
# After each branch, concatenate two 64-d embeddings = 128-d combined
# feature, fed through a small MLP head.
EMBED_DIM = 64               # per-branch output dim.
DENSE_DIM = 64               # hidden dim of the classifier head.
DROPOUT = 0.3                # applied between MLP layers.

# --- Training ---
BATCH_SIZE = 32
LR = 1e-3                    # Adam learning rate.
WEIGHT_DECAY = 1e-5          # L2 regularisation.
EPOCHS = 40                  # max epochs; early-stopping ends sooner.
NUM_WORKERS = 0              # Windows: keep 0 to avoid multiproc headaches.
EARLY_STOP_PATIENCE = 8      # stop if validation AUROC doesn't improve.

# --- Inference ---
THRESHOLD = 0.5              # >= THRESHOLD => cell, < => not cell.
# Filenames the predict step writes into <plane0>/. Tab 3 reads these
# back; downstream tabs key off ``predicted_cell_mask.npy``.
PREDICTED_PROB_NAME = "predicted_cell_prob.npy"
PREDICTED_MASK_NAME = "predicted_cell_mask.npy"
