# `core/cellfilter` — The Cell-Filter CNN

A small two-branch convolutional neural network that takes one Suite2p ROI and outputs `P(this is a real neuron)`. Used by Tab 3 to replace Suite2p's built-in `iscell.npy` classifier with something tuned on this lab's recordings.

This README explains, in order:

1. The biological problem and why a learned classifier helps.
2. The model architecture (`model.py`).
3. The dataset and how (image, trace) tensors are built per ROI (`dataset.py`).
4. The training loop, loss, and metrics (`train.py`).
5. Inference at prediction time (`predict.py`).
6. Configuration (`config.py`).
7. The curation workflow + how many labels you actually need.
8. A re-implementation checklist.

---

## 1. The problem

Suite2p (Sparsery + Cellpose) returns a few hundred to a few thousand ROIs per recording. Many of them are not neurons:

- **Blood vessels** — elongated bright structures with fluorescence variation that *looks* trace-like but is actually motion or perfusion.
- **Dendrites and processes** — sub-cellular structures Suite2p sometimes splits off as separate ROIs.
- **Bright pixels / hot pixels / debris** — small ROIs with no real signal.
- **Out-of-focus cells** — visible in the mean image but with very dim, slow traces.

Suite2p's `iscell.npy` ships a logistic regression on hand-engineered features (footprint compactness, skewness of the trace, etc.). On this lab's recordings it has a high false-positive rate. A small CNN trained on hand-labelled ROIs from these specific recordings does much better.

The CNN is a **binary classifier**. Output >= 0.5 ⇒ "keep this ROI." Default threshold is `THRESHOLD = 0.5` (`config.py`); change it if you want to be more or less inclusive.

---

## 2. Model architecture (`model.py`)

A `CellFilter` module with **two parallel branches** that look at complementary features, then a small MLP head that combines them.

```
spatial patch (B, 3, 32, 32)            trace (B, 1, T)
        │                                       │
        ▼                                       ▼
 SpatialBranch                            TemporalBranch
 (3 conv2d blocks)                        (3 conv1d blocks)
        │                                       │
        ▼                                       ▼
   z_s ∈ ℝ^64                              z_t ∈ ℝ^64
        │                                       │
        └──────────── concat ─────────────┐
                                          ▼
                              [128] → Linear(64) → ReLU → Dropout(0.3) → Linear(1)
                                          │
                                          ▼
                                      logit ∈ ℝ
                                          │
                                       sigmoid
                                          ▼
                                    P(cell) ∈ [0, 1]
```

### 2.1 Spatial branch

`SpatialBranch` operates on a `(B, 3, 32, 32)` tensor where the three channels per ROI are:

| Channel | What it carries |
|---|---|
| `mean` | The recording's z-scored **mean image**, cropped 32×32 around the ROI centroid. Tells the model what the ROI looks like in average fluorescence — round bright soma, stringy dendrite, elongated vessel. |
| `max_proj` | The z-scored **max-projection image**, same crop. Highlights pixels that ever got bright — picks up cells that don't appear in the mean but flashed during events. |
| `roi_mask` | A binary `(32, 32)` mask of the Suite2p footprint. Tells the model *which pixels* this ROI claims — so the network learns "is the bright thing in the mean/max actually inside the footprint?" |

Each of three `Conv2d → BatchNorm2d → ReLU → MaxPool2d(2)` blocks halves the spatial dimensions and increases channels:

```
(B, 3, 32, 32) → (B, 16, 16, 16) → (B, 32, 8, 8) → (B, 64, 4, 4)
```

Then `AdaptiveAvgPool2d(1)` collapses the 4×4 to a single value per channel → `(B, 64, 1, 1)` → flatten → `(B, 64)`. A `Linear(64, EMBED_DIM=64)` produces the spatial embedding `z_s`.

Conv kernel size is `3`, padding `1` (same-size convolution); pool is `2×2`. Channel widths are `(16, 32, 64)` (`config.SPATIAL_CHANNELS`).

### 2.2 Temporal branch

`TemporalBranch` operates on a `(B, 1, T)` tensor, the per-ROI z-scored dF/F. `T = 2000` during training (random crop) and the full trace length at inference.

Three `Conv1d(k=7, p=3) → BatchNorm1d → ReLU → MaxPool1d(2)` blocks. Kernel of 7 frames at 15 fps spans ~470 ms — wide enough to see one calcium transient. Channel widths `(16, 32, 64)`.

After pooling the time dimension shrinks by `2^3 = 8`. `AdaptiveAvgPool1d(1)` then collapses time entirely → `(B, 64, 1)` → squeeze → `(B, 64)`. A `Linear(64, 64)` produces `z_t`.

The temporal branch's job is to learn what calcium transients look like (sharp rise, exponential decay) versus motion artefacts (slow ramps, flat noise) versus blood vessels (periodic perfusion).

### 2.3 Head

```
z = concat(z_s, z_t)                        # (B, 128)
h = ReLU(Linear(128, 64)(z))
h = Dropout(0.3)(h)
logit = Linear(64, 1)(h).squeeze(-1)        # (B,)
```

`forward(spatial, trace) -> logit`. Use `predict_proba(...)` for `sigmoid(logit)` directly with `@torch.no_grad`.

Total parameter count is ~250k; small enough to fit on any GPU and to train in tens of minutes on a few thousand labelled ROIs.

### 2.4 Why two branches?

Either alone would be fragile:
- **Image alone**: a perfectly round, bright cell silhouette that never fires looks like a great cell. (It's probably dead.)
- **Trace alone**: a vessel pulsing with the cardiac cycle looks transient-like.

Combined, the model can learn rules like "looks like a cell *and* fires sometimes" or "trace looks calcium-y *and* the footprint actually corresponds to a bright spot in the mean image."

---

## 3. Dataset (`dataset.py`)

### 3.1 Sample format

One sample = one `(spatial, trace, label)` triple for one ROI:

- `spatial`: `(3, 32, 32) float32` — `[mean_z, max_z, roi_mask]`.
- `trace`: `(1, T) float32` — z-scored dF/F (T = `TRACE_CROP_LEN = 2000` during training, full length at eval).
- `label`: scalar `0.0` or `1.0` (curated by hand).

### 3.2 The label CSV

`config.LABELS_CSV` lives **inside the project** at `src/calliope/data/cellfilter_labels.csv` (resolved relative to the package). The file is created lazily on the first curation flip and is gitignored — per-curator data, not source code.

Schema (column order canonical, see `config.LABELS_CSV_COLUMNS`):

```
plane0_path, recording_ID, ROI_number, user_defined_cell, timestamp_iso
E:\calliope\2024-07-01_00018\detection\final\suite2p\plane0,  2024-07-01_00018,  0, 1, 2026-05-12T14:03:22
E:\calliope\2024-07-01_00018\detection\final\suite2p\plane0,  2024-07-01_00018,  1, 0, 2026-05-12T14:03:25
E:\calliope\2024-07-01_00018\detection\final\suite2p\plane0,  2024-07-01_00018,  2, 1, 2026-05-12T14:03:28
...
```

- **`plane0_path`** — absolute path to the recording's suite2p plane0 folder. The trainer reads features directly from this path; no more `DATA_ROOT/Cx,Hip/<rec_id>` resolution. The CSV is self-locating, so curators on different machines can share labels without rewriting the recording paths.
- **`recording_ID`** — human-readable id (typically the recording folder name). Kept for auditing; the trainer doesn't use it for lookup.
- **`ROI_number`** — integer suite2p ROI index inside that plane0's `stat.npy`.
- **`user_defined_cell`** — `0` or `1`.
- **`timestamp_iso`** — when the label was written. Updated on every re-label, so the CSV reflects the curator's most recent opinion at a glance.

Writes go through `_upsert_label` in the curation popout (`tabs/suite2p/curation_popout.py`): for each `(plane0_path, ROI_number)` pair the row is **replaced in place** if it exists, appended otherwise. Re-labeling an ROI doesn't accumulate history — the CSV stays as one row per labelled ROI.

`load_labels(csv_path)` reads it, casts types, and drops duplicates keyed on `(plane0_path, ROI_number)` keeping the **last** entry. If the file doesn't exist yet it returns an empty DataFrame with the expected columns so `train.py` can be invoked on a fresh project without crashing.

### 3.3 Recording resolution

There is none, by design. Each row of the CSV carries the absolute `plane0_path`; `dataset.py` opens that path directly. The pre-2026-05-12 `find_recording_root(rec_id)` walker (recursive `EXTRA_DATA_ROOTS` search, falling back to `DATA_ROOT/{Cx,Hip}/<rec_id>`) is gone, along with `DATA_ROOT` and `EXTRA_DATA_ROOTS` themselves.

If a recording is moved on disk, the labels CSV needs its `plane0_path` column rewritten — but no other code path needs updating. A simple `pd.read_csv` + string replace + `to_csv` covers it.

### 3.4 Per-recording cache (`_RecordingCache`)

Loading the same recording's mean/max/dff for every ROI would be wasteful, so `_RecordingCache` lazily loads each recording once and reuses it across all ROIs labelled in that recording. Constructor takes the plane0 path directly:

```python
class _RecordingCache:
    def __init__(self, plane0: Path):
        ...

# In ROIDataset.__getitem__:
plane0_path = str(row["plane0_path"])
rec = self._cache.setdefault(plane0_path, _RecordingCache(Path(plane0_path)))
```

Cache key is the plane0 path string (so two recordings whose `recording_ID` strings happen to collide are still treated as distinct, since their plane0_path is different).

Inside the cache:

```python
stat        = np.load(plane0/'stat.npy')                 # ROI footprints
view        = utils.load_plane_view(plane0)              # ops + reg/detect outputs
mean_img    = view.get('meanImgE') or view.get('meanImg')
max_img     = view.get('max_proj') or view.get('maxImg') or mean_img
mean_img_z  = (mean_img - mean) / std                    # z-score
max_img_z   = (max_img  - mean) / std
dff, _, _, T, N = utils.s2p_open_memmaps(plane0, prefix='r0p7_')
```

If `max_proj` is cropped to Suite2p's `yrange/xrange` (a common quirk), it's padded back to full FOV using those ranges so it spatially aligns with `mean_img`.

### 3.5 `get_patch(roi_idx, size=32)`

```python
s   = stat[roi_idx]
cy  = round(mean(ypix))
cx  = round(mean(xpix))
mean_patch  = _pad_to_patch(mean_img_z, cy, cx, 32)      # zero-pad if near edge
max_patch   = _pad_to_patch(max_img_z,  cy, cx, 32)
mask_full   = zeros(H, W); mask_full[ypix, xpix] = 1
mask_patch  = _pad_to_patch(mask_full, cy, cx, 32)
return stack([mean_patch, max_patch, mask_patch])         # (3, 32, 32)
```

Centroid is computed from the ROI's `xpix/ypix` (unweighted by `lam`). Patch is zero-padded if the ROI is near the FOV edge.

### 3.6 `get_trace(roi_idx)`

```python
trace = dff[:, roi_idx].astype(float32)
trace = (trace - trace.mean()) / max(trace.std(), 1e-6)   # z-score
return trace                                              # (T,)
```

Per-ROI z-scoring removes any DC offset and amplitude differences between cells, so the temporal branch sees comparable shapes regardless of expression level.

### 3.7 Random temporal crop (training augmentation)

`ROIDataset.__getitem__` cuts a random `TRACE_CROP_LEN = 2000` frames out of each trace during training (`random_crop=True`), or the centred `2000` frames at eval (`random_crop=False`). Traces shorter than 2000 frames are zero-padded.

This augments the dataset (different temporal contexts each epoch) and ensures the model handles arbitrary recording lengths at inference.

### 3.8 Splits

Two split helpers — choose based on dataset size:

- `split_by_recording(df, val_frac=0.20)` — preferred once you have ≥5 labelled recordings. Hold out a fraction of *recordings* (not ROIs). Validation never sees ROIs from training-set recordings, so AUROC reflects generalisation across mice/slices, not memorisation of a particular FOV. Uniqueness key is `plane0_path` rather than `recording_ID` (post-2026-05-12 schema) so two recordings whose human-readable ids collide get treated as distinct.
- `split_by_roi(df, val_frac=0.20)` — fallback. Random per-ROI split, stratified on label so val keeps the class balance. Looser test of generalisation but works with one or two labelled recordings.

`train.py` currently calls `split_by_roi` (the lab's labelled set is small enough that recording-level splits leave too few per side). Swap the import in `train.py` once you have enough recordings to support a clean recording-level holdout.

---

## 4. Training (`train.py`)

### 4.1 Loop

```
for epoch in 1..40:
    train: BCEWithLogitsLoss(pos_weight=N_neg/N_pos) + Adam(lr=1e-3, wd=1e-5)
    val:   same loss, no grad
    log    train_loss/acc/auc, val_loss/acc/auc, seconds
    save   last.pt every epoch
    if    val_auc > best_auc: save best.pt; bad_epochs = 0
    else:                      bad_epochs += 1
    if    bad_epochs >= 8:    early stop
```

### 4.2 Loss

`nn.BCEWithLogitsLoss(pos_weight=...)`. The `pos_weight` is `N_neg / N_pos` (clamped to ≥ 1). Without it, the dominant class (usually negatives — most ROIs *aren't* good cells) drives the gradient and the model learns to output ~0 for everyone. The weight scales positive-class gradient up to balance.

### 4.3 Metrics

- **Accuracy** at threshold 0.5.
- **AUROC** computed in-house (`_auroc`) to avoid a sklearn dependency. Uses Mann–Whitney U with average-rank tie correction:

```python
ranks   = avg-rank of scores
U       = sum(ranks where label==1) - n_pos*(n_pos+1)/2
auroc   = U / (n_pos * n_neg)
```

AUROC is a threshold-free measure of class separability and is what `best.pt` is selected on. Accuracy alone would be misleading under heavy class imbalance.

### 4.4 Early stopping

`EARLY_STOP_PATIENCE = 8` epochs without a new best val AUROC stops training. `best.pt` (peak val AUROC) is the checkpoint that goes into Tab 3.

### 4.5 Outputs

- `CHECKPOINT_DIR/best.pt` — best-val-AUROC weights.
- `CHECKPOINT_DIR/last.pt` — most recent weights.
- `CHECKPOINT_DIR/train_log.csv` — per-epoch loss/acc/auc table for plotting.

Each `.pt` is `{model: state_dict, epoch: int, val_auc: float}`.

### 4.6 Determinism

`torch.manual_seed(0)` and `np.random.seed(0)` at the top of `main()`. Splits use `RANDOM_SEED = 0` too. Re-running `train.py` from the same labels produces (modulo CUDA non-determinism) the same `best.pt`.

### 4.7 Windows OpenMP shim

```python
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
```

These three lines must run **before** numpy/torch import. NumPy-MKL and PyTorch each ship their own `libiomp5md.dll`; both being loaded crashes Python on Windows. The first env var lets duplicates coexist; the other two pin OMP to 1 thread to avoid the resulting deadlock.

---

## 5. Inference (`predict.py`)

### 5.1 `predict_recording(rec_id, model, device, plane0=None)`

The hot path called by Tab 3 and the standalone curation app:

```python
if plane0 is None:
    raise ValueError("plane0 is required")  # DATA_ROOT walker removed 2026-05-12
rec = _RecordingCache(Path(plane0))                       # mmap dF/F + load images
N   = rec.N
probs = zeros(N, float32)
for roi in range(N):
    patch    = rec.get_patch(roi, 32)                     # (3, 32, 32)
    trace    = rec.get_trace(roi)                         # (T,)
    spatial  = torch.from_numpy(patch)[None].to(device)
    trace_t  = torch.from_numpy(trace)[None, None].to(device)  # (1, 1, T)
    logit    = model(spatial, trace_t)
    probs[roi] = torch.sigmoid(logit).item()

np.save(plane0/'predicted_cell_prob.npy', probs)
np.save(plane0/'predicted_cell_mask.npy', probs >= 0.5)
```

`rec_id` is now only used for the log line at the end; the actual feature loading is keyed on `plane0`. Callers without a clean recording id can pass `infer_recording_id(plane0)` or `plane0.parent.name`.

Per-ROI loop, batch size 1 (the temporal branch handles variable `T` per recording, so naive batching across recordings would need padding/masking we don't bother with). Empirically takes ~1 s per ROI on CPU and < 0.1 s on GPU.

`load_model_from_checkpoint(ckpt_path, device)` builds a fresh `CellFilter`, calls `torch.load` + `load_state_dict` + `.eval()`, and returns the model. Shared between the CLI here, the Tab 3 popout's auto-repredict-after-retrain path, and the standalone curation app's auto-predict-on-open path so all three paths use the same loader.

### 5.2 Why full traces (no crop) at inference?

The convolutional + global-pool architecture handles arbitrary `T`. Cropping at inference would throw away signal. The `AdaptiveAvgPool1d(1)` at the end of the temporal branch makes the model `T`-agnostic.

### 5.3 CLI usage

```
python -m calliope.core.cellfilter.predict --plane0 D:/data/.../suite2p/plane0 [...]
    # one or more plane0 directories; the checkpoint is loaded once
    # and the model is reused across recordings in the batch
```

Tab 3 invokes the function form (`predict_recording(rec_id, model, device, plane0=plane0)`) not the CLI.

The 2026-05-12 refactor removed the old `--rec ID` mode and the no-arg "scan `DATA_ROOT/{Cx,Hip}/` and predict everything" mode along with `DATA_ROOT` itself. The canonical input is now an explicit plane0 path; `predict_recording` raises `ValueError` if called without one.

### 5.4 Outputs

In each `plane0/`:
- `predicted_cell_prob.npy` — `(N,) float32`, sigmoid score in `[0, 1]`.
- `predicted_cell_mask.npy` — `(N,) bool`, `prob >= 0.5`.

Both are picked up by **every** downstream tab (4–7) via `_load_keep_mask` (`utils._load_keep_mask`), with `iscell.npy` as a final fallback if the predicted files are missing.

---

## 6. Configuration (`config.py`)

| Key | Value | Purpose |
|---|---|---|
| `LABELS_CSV` | `src/calliope/data/cellfilter_labels.csv` (in-project) | Hand-labelled training data. Created lazily on the first curation flip. Gitignored so per-curator labels don't conflict. |
| `LABELS_CSV_COLUMNS` | `(plane0_path, recording_ID, ROI_number, user_defined_cell, timestamp_iso)` | Canonical CSV schema. `plane0_path` is the trainer's source of truth -- no more `DATA_ROOT` resolution. |
| `CHECKPOINT_DIR` | `~/.calliope/cellfilter_checkpoints/` | Where retrained `best.pt`, `last.pt`, `train_log.csv` from `cellfilter.train` land. The default ckpt Tab 3 loads is the bundled `src/calliope/data/cellfilter_best.pt` (small enough to ship in the wheel); users who retrain point Tab 3's Browse... field at their new file here. |
| `DFF_PREFIX` | `r0p7_` | Memmap prefix the cache reads from `plane0`. |
| `PATCH_SIZE` | 32 | Spatial patch edge in pixels. |
| `TRACE_CROP_LEN` | 2000 | Random crop length for training. |
| `VAL_FRAC` | 0.20 | Validation fraction. |
| `RANDOM_SEED` | 0 | Determinism. |
| `TEMPORAL_CHANNELS` | (16, 32, 64) | Conv1d channel widths. |
| `SPATIAL_CHANNELS` | (16, 32, 64) | Conv2d channel widths. |
| `EMBED_DIM` | 64 | Per-branch output dim. |
| `DENSE_DIM` | 64 | Hidden width of the MLP head. |
| `DROPOUT` | 0.3 | Dropout in the head. |
| `BATCH_SIZE` | 32 | Training batch size. |
| `LR` | 1e-3 | Adam learning rate. |
| `WEIGHT_DECAY` | 1e-5 | Adam weight decay. |
| `EPOCHS` | 40 | Max training epochs. |
| `EARLY_STOP_PATIENCE` | 8 | Epochs without val-AUROC improvement before stopping. |
| `NUM_WORKERS` | 0 | DataLoader workers (0 on Windows to dodge multiproc bugs). |
| `THRESHOLD` | 0.5 | Probability threshold for the boolean mask. |

---

## 7. Curation workflow

### 7.1 Two entry points to the same labels CSV

| Entry | When you'd use it | Where it lives |
|---|---|---|
| **Tab 3 popout** | One-off curation while you're already inspecting a recording in the pipeline GUI. Click an ROI on the "All detected ROIs" panel; a child window opens for that ROI. Flip with `1`/`0`, close, move on. | `src/calliope/tabs/suite2p/curation_popout.py` |
| **Standalone curation app** | Bulk training-data collection. Walks many recordings in one sitting, sorts ROIs by classifier uncertainty so the curator hits the most informative ones first. No pipeline-GUI dependency. | `src/calliope/scripts/roi_curation_app.py` |

Both write to the same `cellfilter_labels.csv` via the same `_upsert_label` function. Run them in any combination — labels accumulate in one file.

### 7.2 Standalone app

```bash
python -m calliope.scripts.roi_curation_app [plane0 ...] [--sort MODE] [--repredict] [--ckpt PATH]
```

Tk window with a `Listbox` of plane0 paths plus **Add plane0...** / **Remove** / **Open curation** / **Re-predict selected** / sort dropdown. Add several plane0s up front (or drip-feed via Add plane0 as you go). Selecting an entry:

1. Runs `predict_recording` against it if `predicted_cell_prob.npy` is missing in that plane0 (or `--repredict` was passed).
2. Spawns the `CurationPopout` against that plane0.
3. Binds `Left` / `Right` for Prev / Next sorted ROI; `1` / `0` for cell / not-a-cell; `Esc` to close.

Sort modes:

| `--sort` | Order | Best for |
|---|---|---|
| `unsure` (default) | `|p(cell) - 0.5|` ascending | Active learning — the most-ambiguous ROIs come first, which is where new labels move the model the most. |
| `prob_asc` | lowest p(cell) first | Hunting for false positives (the model said "not cell" but it actually is one). |
| `prob_desc` | highest p(cell) first | Hunting for false negatives in the "cell" set. |
| `index` | raw Suite2p ROI order | When you don't have a checkpoint yet, or just want a deterministic walk. |

After every flip the popout auto-advances to the next sorted ROI — `1`/`0` is enough to label your way through the queue. Switching plane0s in the Listbox closes the current popout and opens a new one; labels CSV is the persistence layer across switches.

### 7.3 Retrain loop (auto re-predict)

The Tab 3 popout's **Retrain cell filter** button (greyed until ≥1 flip in the current session) does:

1. `cellfilter.train.main()` — reads the labels CSV, writes new `best.pt` + `last.pt` to `CHECKPOINT_DIR`.
2. `predict_recording(rec_id, model, device, plane0=self._plane0)` — scores every ROI in the *current* recording against the fresh checkpoint.
3. Refreshes the popout's own `_probs` cache; pings parent Tab 3 via `on_iscell_changed(-1, -1)` so panels 2 + 3 repaint against the new mask without a manual re-detect.

Three result states:

- `("ok", msg)` — both train and re-predict succeeded; new mask on disk and visible.
- `("warn", msg)` — train succeeded but re-predict failed (e.g., the active plane0 was moved). New `best.pt` is still on disk; the curator can run `predict.py --plane0 ...` manually.
- `("err", msg)` — training itself failed; old checkpoint untouched.

### 7.4 How many labels for a robust classifier?

For this two-branch CNN (~250 K params, the existing config), expect:

| Total labels | Recordings | What you get |
|---:|---:|---|
| ~300–500 | 3+ | Bare minimum viable. Will train, will beat Suite2p's built-in classifier on the recordings it saw. Won't generalise to new days/scope settings. |
| ~1000–1500 | 6–10 | "It's working well now." Validation AUROC usually plateaus in here for this model size. The `split_by_recording` validator needs ≥5 recordings to be meaningful. |
| ~2500–4000 | 10–15 | Robust / "stop curating." Past this the curve flattens — extra labels mostly fight measurement noise. Only worth pushing further if you're adding genuinely new conditions (new region, new GCaMP variant, new microscope). |

What bends the curve in your favour:

1. **Recording diversity > per-recording count.** 100 ROIs from each of 10 recordings beats 1000 ROIs from 1 recording. The validator can only catch overfitting if held-out recordings differ from training ones.
2. **Spend keystrokes on ambiguous ROIs.** The default `unsure` sort puts the most-informative ROIs first. One ambiguous label is worth ~3–5 confident ones. Avoid sitting at `p ≈ 0.95` confirming the model.
3. **Iterate, don't batch.** Label ~200–300 → Retrain (auto re-predict rescores everything) → reopen with `unsure` sort → label another ~200–300 on newly-uncertain ROIs. Three or four passes get further than 2000 labels in one sitting.
4. **Class balance handles itself.** `BCEWithLogitsLoss(pos_weight = N_neg / N_pos)` upweights the rare class automatically. Don't artificially skip obvious cases on one side or the other.

For tissue-identity experiments (e.g., the cortex+hippocampus stitch test), get ≥500 labels in the bag *before* trusting the keep-mask for downstream clustering — otherwise mask noise dominates whatever signal you're trying to measure.

---

## 8. Re-implementation checklist

1. **Build the per-ROI sample.**
   - Compute the centroid of `(xpix, ypix)`.
   - Crop a 32×32 patch from the z-scored `meanImgE` (or `meanImg`), z-scored `max_proj` (padded to full FOV if needed), and a binary footprint mask. Zero-pad near edges.
   - Stack into `(3, 32, 32)`.
   - Z-score the per-ROI dF/F trace; pad/crop to `T = 2000` during training, full length at inference.
2. **Build the model.**
   - Spatial: 3× `(Conv2d 3×3 → BN → ReLU → MaxPool 2×2)` with channels `(16, 32, 64)`. `AdaptiveAvgPool2d(1)` → `Linear(64, 64)`.
   - Temporal: 3× `(Conv1d k=7 p=3 → BN → ReLU → MaxPool 2)` with channels `(16, 32, 64)`. `AdaptiveAvgPool1d(1)` → `Linear(64, 64)`.
   - Head: concat 128-d → `Linear(64) → ReLU → Dropout(0.3) → Linear(1)` → squeeze.
3. **Loss.** `BCEWithLogitsLoss(pos_weight = N_neg / N_pos)`. Adam(lr=1e-3, wd=1e-5).
4. **Metrics.** Accuracy at 0.5 + AUROC via Mann–Whitney U with avg-rank tie handling. Pick the checkpoint with the highest **validation AUROC**.
5. **Splits.** Prefer `split_by_recording` when you have >= ~5 labelled recordings; fall back to stratified per-ROI split otherwise.
6. **Training augmentation.** Random temporal crop only — no spatial flip/rotation (the model would have to be equivariant to FOV orientation, which isn't worth the extra training data here).
7. **Inference.** Full traces, batch size 1, `model.eval()` + `@torch.no_grad`. Save `predicted_cell_prob.npy` and `predicted_cell_mask.npy = prob >= 0.5` in `plane0/`.
8. **Windows OpenMP shim.** Set `KMP_DUPLICATE_LIB_OK=TRUE`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` *before* importing numpy/torch when running on Windows with both numpy-MKL and PyTorch installed.

That's the whole filter. Small model, small dataset, high impact downstream — every kept ROI is a feature in the clustering and cross-correlation tabs, so a 10% reduction in false positives here cleans up Tabs 5–7 substantially.
