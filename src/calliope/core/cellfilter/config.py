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
- **Paths** - in-project labels CSV (the human-labelled
  cell-vs-noise ground truth) and an out-of-project checkpoint
  directory under the user's home so binary weights aren't checked
  into version control.
- **Data** - patch size used by both training and inference, plus
  training-only knobs (validation fraction, random seed, trace
  crop length).
- **Model** - architecture sizes for the temporal and spatial
  branches of the CNN.
- **Training** - optimisation knobs (batch size, learning rate,
  weight decay, early-stopping patience).
- **Inference** - decision threshold and the file names the
  ``predict`` step writes into ``plane0/``.

History
-------
Pre-2026-05-12 this module hardcoded ``LABELS_CSV =
F:\\roi_curation.csv``, ``DATA_ROOT = F:\\data\\2p_shifted``, and
``CHECKPOINT_DIR = F:\\cellfilter_checkpoints``. Training was fragile
to drive-letter changes and could only find recordings whose folders
sat in a specific ``Cx/`` or ``Hip/`` layout under ``DATA_ROOT``.

The 2026-05-12 refactor moved labels in-project, switched the CSV
schema to carry the absolute ``plane0_path`` per row (so the trainer
no longer has to search for recordings by name), and parked
checkpoints under ``~/.calliope/cellfilter_checkpoints/`` so the
project tree stays free of large binary artefacts. ``DATA_ROOT`` and
``EXTRA_DATA_ROOTS`` are gone -- the CSV is now self-locating.
"""
from pathlib import Path

# --- Paths ---
# Curation CSV: one row per (plane0_path, ROI_number) with a 0/1
# label. The training pipeline reads this; the Tab 3 popout and the
# standalone ``scripts/roi_curation_app.py`` append to it. Lives
# inside the project (next to the suite2p ops .npy + AAV metadata
# .csv) so a fresh clone of the repo has everything it needs to
# retrain without any external drive present.
#
# Schema: ``plane0_path, recording_ID, ROI_number,
# user_defined_cell, timestamp_iso``. Created lazily on the first
# label append; missing CSV is treated as "no curated labels yet"
# by ``dataset.load_labels``.
LABELS_CSV = (Path(__file__).resolve().parents[2]
              / "data" / "cellfilter_labels.csv")

# Where ``train.py`` saves model weights at end-of-epoch and at the
# best AUROC seen so far. Lives under the user's home so big
# (~50-200 MB) .pt files don't end up tracked by git. ``predict.py``
# loads ``CHECKPOINT_DIR / "best.pt"`` by default. The directory is
# created on first save.
CHECKPOINT_DIR = Path.home() / ".calliope" / "cellfilter_checkpoints"

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


# --- CSV schema constants ---
# Centralised so the popout writer, the standalone curation app,
# and ``dataset.load_labels`` all use the exact same column order.
LABELS_CSV_COLUMNS: tuple[str, ...] = (
    "plane0_path",
    "recording_ID",
    "ROI_number",
    "user_defined_cell",
    "timestamp_iso",
)
