"""Train the cell-filter CNN.

What this file does
-------------------
This is the offline training script that produces the checkpoint
``predict.py`` later loads. The GUI does not invoke this; lab members
run it from the command line on a GPU box once they've curated a new
batch of ROIs in the labelling spreadsheet.

Steps performed
---------------
1. Load the curation CSV (``LABELS_CSV`` in ``config.py``) and split
   by ROI into train + validation sets (``split_by_roi``, stratified
   on label -- roughly a stratified ``createDataPartition`` in R).
   NOTE: this is a per-ROI split, so ROIs from the same recording can
   land on both sides (recording-level leakage). It is an intentional
   tradeoff for the lab's small labelled set; switch the import to
   ``split_by_recording`` once you have >= ~5 labelled recordings for a
   leak-free holdout (see README).
2. Build two ``ROIDataset`` instances and wrap them in
   ``torch.utils.data.DataLoader`` for batched, shuffled iteration.
3. Instantiate ``CellFilter``, the optimiser (Adam) and the loss
   (``BCEWithLogitsLoss`` with ``pos_weight`` to compensate for
   class imbalance -- typical recordings have ~50% non-cell ROIs).
4. For each epoch:
       - Train pass: forward / loss / backward / optimiser step.
       - Validation pass: collect probabilities + labels, compute
         AUROC.
       - Log epoch metrics to a CSV.
       - Save ``last.pt``; if the AUROC improved, save ``best.pt``.
5. Stop early if AUROC hasn't improved in
   ``EARLY_STOP_PATIENCE`` epochs.

Usage
-----
    python -m src.calliope.core.cellfilter.train

Outputs
-------
    {CHECKPOINT_DIR}/best.pt           best-validation-AUROC checkpoint
    {CHECKPOINT_DIR}/last.pt           last-epoch checkpoint
    {CHECKPOINT_DIR}/train_log.csv     per-epoch metrics
"""
from __future__ import annotations

# --- OpenMP / MKL workaround: both numpy-MKL and pytorch ship their own
# libiomp5md.dll on Windows, which clash at import time. Must be set BEFORE
# numpy / torch are imported.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config as C
from .dataset import ROIDataset, load_labels, split_by_roi
from .model import CellFilter


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U AUROC with tie handling, no sklearn dependency.

    AUROC = "Area Under the ROC curve" = the probability that a
    randomly chosen positive sample gets a higher score than a
    randomly chosen negative one. We compute it analytically from
    the rank sum of positive samples (Mann-Whitney U formulation),
    which is faster and less code than thresholding the ROC curve.

    Returns NaN if either class is empty (AUROC is undefined).
    Tie handling: average ranks are assigned to ties so two samples
    with identical scores don't randomly swap their AUROC contribution.
    """
    if len(np.unique(labels)) < 2:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    sort_s = scores[order]
    i = 0
    while i < len(sort_s):
        j = i
        while j + 1 < len(sort_s) and sort_s[j + 1] == sort_s[i]:
            j += 1
        if j > i:
            avg = ranks[order[i:j + 1]].mean()
            ranks[order[i:j + 1]] = avg
        i = j + 1
    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = ranks[labels == 1].sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def _run_epoch(model, loader, device, optim=None, pos_weight=None):
    """Run one full pass of ``loader`` through ``model``.

    Acts as both the training and the validation loop:
        * If ``optim`` is provided -> training mode: forward, loss,
          backward, optimiser step.
        * If ``optim`` is None -> eval mode: forward + loss only,
          no parameter updates.

    Returns ``(mean_loss, accuracy, auroc)`` for the epoch.

    What ``BCEWithLogitsLoss`` is
    -----------------------------
    Binary cross-entropy applied to raw logits (pre-sigmoid). It
    fuses sigmoid + BCE into one numerically stable op.
    ``pos_weight`` reweights the positive class -- crucial here
    because typical recordings have many more "not cell" ROIs than
    "cell" ones; without reweighting the model could get great
    accuracy by just predicting "not cell" for everything.
    """
    train = optim is not None
    # ``model.train(True)`` enables dropout + BatchNorm running
    # stats updates; ``model.train(False)`` (eval mode) disables
    # them.
    model.train(train)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    total_loss = 0.0
    total_n = 0
    all_scores = []
    all_labels = []

    # PyTorch DataLoader iterates over the Dataset in batches with
    # parallel preloading. Each iteration yields one batch.
    for spatial, trace, label in loader:
        # ``non_blocking=True`` allows asynchronous host-to-GPU copy
        # when the source tensor is in pinned memory -- a small
        # speedup on CUDA boxes.
        spatial = spatial.to(device, non_blocking=True)
        trace = trace.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        # ``torch.set_grad_enabled`` is a context manager that
        # flips autograd on/off. We turn it off in eval mode to
        # save memory.
        with torch.set_grad_enabled(train):
            logit = model(spatial, trace)
            loss = loss_fn(logit, label)

            if train:
                # ``zero_grad(set_to_none=True)`` deletes the
                # gradient buffers (faster than zeroing them).
                # Then backprop and optimiser step.
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()

        bs = label.size(0)
        total_loss += loss.item() * bs
        total_n += bs
        # Stash per-batch predictions for AUROC at the end. ``.detach()``
        # severs the autograd graph; ``.cpu()`` ships back to host
        # RAM so we don't pile up GPU tensors.
        all_scores.append(torch.sigmoid(logit).detach().cpu().numpy())
        all_labels.append(label.detach().cpu().numpy())

    # No batches were seen (empty loader, e.g. an empty val split when a
    # class has a single ROI) -> np.concatenate([]) would raise. Return the
    # zero loss and NaN metrics, which downstream formatting already handles.
    if not all_scores:
        return total_loss / max(1, total_n), float("nan"), float("nan")

    # Concatenate every batch's scores / labels and compute the
    # epoch-level metrics.
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    pred = (scores >= 0.5).astype(np.int32)
    acc = float((pred == labels.astype(np.int32)).mean())
    auc = _auroc(scores, labels)
    return total_loss / max(1, total_n), acc, auc


def _drop_missing_recordings(df):
    """Drop label rows whose recording is no longer on disk.

    A row is usable only if its ``plane0_path`` still holds a
    ``stat.npy`` (what ``_RecordingCache`` needs to slice patches).
    Returns the filtered DataFrame and warns -- loudly -- about every
    recording it dropped, so a partial retrain after data has moved or
    been deleted fails gracefully instead of crashing in the loader.
    """
    paths = df["plane0_path"].unique()
    missing = {p for p in paths if not (Path(p) / "stat.npy").exists()}
    if missing:
        n_rows = int(df["plane0_path"].isin(missing).sum())
        print(f"WARNING: {len(missing)} of {len(paths)} recording(s) "
              f"not found on disk -- dropping {n_rows} labelled ROI(s):")
        for p in sorted(missing):
            print(f"  missing: {p}")
        df = df[~df["plane0_path"].isin(missing)].reset_index(drop=True)
    return df


def main(warm_start: "Path | str | None" = None,
         lr: "float | None" = None,
         ckpt_dir: "Path | None" = None,
         skip_missing: bool = True):
    """Top-level training driver.

    Parameters
    ----------
    warm_start : path or None
        If given, load this checkpoint's weights into the model before
        training (fine-tuning / transfer learning) instead of training
        from random init. Public users with none of the original data
        pass the bundled ``calliope/data/cellfilter_best.pt`` here to
        adapt the shipped model to their own labelled ROIs.
    lr : float or None
        Adam learning rate. Defaults to ``config.LR``; pass something
        smaller (e.g. 1e-4) when warm-starting so fine-tuning doesn't
        blow away the pretrained weights.
    ckpt_dir : Path or None
        Where to write ``best.pt`` / ``last.pt`` / ``train_log.csv``.
        Defaults to ``config.CHECKPOINT_DIR``. Override to compare runs
        without clobbering the live checkpoint.
    skip_missing : bool
        Drop label rows whose recording is gone from disk (default
        True). Lets a retrain proceed on the surviving subset.

    Steps:
        1. Seed RNGs + select GPU/CPU device.
        2. Load + split the curation labels.
        3. Build train/val ``ROIDataset`` + ``DataLoader``.
        4. Instantiate ``CellFilter``, Adam optimiser, BCE loss with
           positive-class weight.
        5. For each epoch: train, validate, log to CSV, save
           checkpoints, early-stop if val AUROC stops improving.

    Returns the best validation AUROC seen (NaN if val never had both
    classes), so a comparison driver can pick the better of two runs.
    """
    ckpt_dir = Path(ckpt_dir) if ckpt_dir is not None else C.CHECKPOINT_DIR
    lr = C.LR if lr is None else float(lr)

    # Seed both PyTorch and NumPy so train/val splits and weight
    # init are reproducible.
    torch.manual_seed(C.RANDOM_SEED)
    np.random.seed(C.RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- data ---
    df = load_labels(C.LABELS_CSV)
    if skip_missing:
        df = _drop_missing_recordings(df)
    if len(df) == 0:
        raise RuntimeError(
            "No labelled ROIs with recordings present on disk -- nothing "
            "to train on. Check cellfilter_labels.csv and the plane0_path "
            "column.")
    print(f"Loaded {len(df)} labeled ROIs across {df['recording_ID'].nunique()} recordings.")
    print(f"  positives: {(df['user_defined_cell']==1).sum()}   "
          f"negatives: {(df['user_defined_cell']==0).sum()}")

    # Stratified per-ROI split. ``split_by_recording`` would be more
    # rigorous but with our small recording count produces empty
    # train sets when the val side gets unlucky.
    train_df, val_df = split_by_roi(df, C.VAL_FRAC, C.RANDOM_SEED)
    print(f"train: {len(train_df)} ROIs  ({train_df['recording_ID'].nunique()} recs)  "
          f"pos={(train_df['user_defined_cell']==1).sum()}  "
          f"neg={(train_df['user_defined_cell']==0).sum()}")
    print(f"val:   {len(val_df)} ROIs  ({val_df['recording_ID'].nunique()} recs)  "
          f"pos={(val_df['user_defined_cell']==1).sum()}  "
          f"neg={(val_df['user_defined_cell']==0).sum()}")

    # Sharing one cache between train and val Datasets means each
    # recording's mean image / max projection / dF/F memmap loads
    # only once, even though both Datasets iterate ROIs from it.
    shared_cache = {}
    # ``augment=True`` enables dihedral-group spatial transforms +
    # additive trace noise on the train set; ``augment=False`` for val
    # so the eval signal is comparable across epochs and against a
    # bundled checkpoint trained without augmentation. Probabilities
    # / noise std are set in ``config.AUG_*``.
    train_ds = ROIDataset(train_df, random_crop=True, augment=True,
                          cache=shared_cache)
    val_ds = ROIDataset(val_df, random_crop=False, augment=False,
                        cache=shared_cache)

    # ``DataLoader`` handles batching, shuffling, multi-process
    # preloading, pinned-memory transfers. ``num_workers=0`` on
    # Windows because Python multiprocessing on Windows + Tk is
    # fragile.
    train_loader = DataLoader(
        train_ds, batch_size=C.BATCH_SIZE, shuffle=True,
        num_workers=C.NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=C.BATCH_SIZE, shuffle=False,
        num_workers=C.NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )

    # --- model ---
    model = CellFilter().to(device)
    # ``p.numel()`` is the number of elements in a tensor; summing
    # across all parameter tensors gives the total parameter count.
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    # Warm start: initialise from an existing checkpoint instead of
    # random weights. The model is fixed-architecture, so a plain
    # ``load_state_dict`` is all transfer learning needs here.
    if warm_start is not None:
        ckpt = torch.load(Path(warm_start), map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        print(f"warm-started from {warm_start} "
              f"(val_auc={ckpt.get('val_auc')}, lr={lr:g})")

    # Class imbalance: pos_weight = (n_negatives / n_positives) tells
    # BCEWithLogitsLoss "treat each positive sample as if it were
    # this many positives". That cancels out the bias toward the
    # majority (not-cell) class.
    n_pos = (train_df["user_defined_cell"] == 1).sum()
    n_neg = (train_df["user_defined_cell"] == 0).sum()
    pos_weight = torch.tensor([max(1.0, n_neg / max(1, n_pos))], device=device)
    print(f"pos_weight: {pos_weight.item():.3f}")

    # Adam: adaptive-learning-rate optimiser; weight_decay = small
    # L2 regulariser to discourage huge weights.
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=C.WEIGHT_DECAY)

    log_path = ckpt_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "train_acc", "train_auc",
             "val_loss", "val_acc", "val_auc", "seconds"]
        )

    best_auc = -1.0
    bad_epochs = 0

    for epoch in range(1, C.EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_auc = _run_epoch(model, train_loader, device, optim, pos_weight)
        va_loss, va_acc, va_auc = _run_epoch(model, val_loader, device, None, pos_weight)
        dt = time.time() - t0

        print(
            f"ep {epoch:3d}  "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} auc {tr_auc:.3f}  |  "
            f"val loss {va_loss:.4f} acc {va_acc:.3f} auc {va_auc:.3f}  "
            f"[{dt:.1f}s]"
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, tr_loss, tr_acc, tr_auc, va_loss, va_acc, va_auc, f"{dt:.2f}"]
            )

        # Always overwrite ``last.pt`` with the most recent epoch
        # so we can resume from anywhere.
        # ``model.state_dict()`` is a dict mapping parameter names
        # to their current tensor values -- the canonical PyTorch
        # serialisation form.
        torch.save(
            {"model": model.state_dict(), "epoch": epoch, "val_auc": va_auc},
            ckpt_dir / "last.pt",
        )

        # Track the best validation AUROC seen so far. ``best.pt``
        # is the checkpoint downstream code (``predict.py`` / Tab 3)
        # actually loads.
        if not np.isnan(va_auc) and va_auc > best_auc:
            best_auc = va_auc
            bad_epochs = 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_auc": va_auc},
                ckpt_dir / "best.pt",
            )
            print(f"  -> new best (val_auc={va_auc:.3f}), checkpoint saved")
        else:
            # Early stopping: bail if val AUROC hasn't improved in
            # ``EARLY_STOP_PATIENCE`` epochs. Saves time and dodges
            # over-fitting.
            bad_epochs += 1
            if bad_epochs >= C.EARLY_STOP_PATIENCE:
                print(f"early stop after {bad_epochs} epochs without improvement")
                break

    print(f"best val auc: {best_auc:.3f}")
    print(f"checkpoints in {ckpt_dir}")
    return best_auc


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--warm-start", metavar="CKPT", default=None,
                    help="checkpoint to initialise from (fine-tune) instead "
                         "of random init; e.g. the bundled cellfilter_best.pt")
    ap.add_argument("--lr", type=float, default=None,
                    help="Adam learning rate (default config.LR; use ~1e-4 "
                         "when warm-starting)")
    ap.add_argument("--ckpt-dir", default=None,
                    help="output dir for best/last/log (default "
                         "config.CHECKPOINT_DIR)")
    ap.add_argument("--keep-missing", action="store_true",
                    help="do NOT drop label rows whose recording is gone "
                         "(default is to skip them)")
    args = ap.parse_args()
    main(warm_start=args.warm_start, lr=args.lr, ckpt_dir=args.ckpt_dir,
         skip_missing=not args.keep_missing)