"""Central config for the cell filter model."""
from pathlib import Path

# --- Paths ---
LABELS_CSV = Path(r"F:\roi_curation.csv")
DATA_ROOT = Path(r"F:\data\2p_shifted")  # contains Cx\ and Hip\
# Extra roots searched recursively (any depth) for a folder named <rec_id>
# that contains suite2p/plane0. First match wins.
EXTRA_DATA_ROOTS = [
    Path(r"D:\data"),
]
CHECKPOINT_DIR = Path(r"F:\cellfilter_checkpoints")

# --- Data ---
DFF_PREFIX = "r0p7_"
PATCH_SIZE = 32              # spatial patch edge, pixels
TRACE_CROP_LEN = 2000        # random crop length for training
VAL_FRAC = 0.20              # fraction of recordings held out for validation
RANDOM_SEED = 0

# --- Model ---
TEMPORAL_CHANNELS = (16, 32, 64)
SPATIAL_CHANNELS = (16, 32, 64)
EMBED_DIM = 64               # per-branch output dim
DENSE_DIM = 64
DROPOUT = 0.3

# --- Training ---
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 40
NUM_WORKERS = 0              # Windows: keep 0 to avoid multiproc headaches
EARLY_STOP_PATIENCE = 8

# --- Inference ---
THRESHOLD = 0.5
PREDICTED_PROB_NAME = "predicted_cell_prob.npy"
PREDICTED_MASK_NAME = "predicted_cell_mask.npy"
