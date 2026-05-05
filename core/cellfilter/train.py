"""
Train the cell filter.

Usage
-----
    python -m cellfilter.train

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
    """Mann-Whitney U AUROC with tie handling, no sklearn dependency."""
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
    train = optim is not None
    model.train(train)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    total_loss = 0.0
    total_n = 0
    all_scores = []
    all_labels = []

    for spatial, trace, label in loader:
        spatial = spatial.to(device, non_blocking=True)
        trace = trace.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            logit = model(spatial, trace)
            loss = loss_fn(logit, label)

            if train:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()

        bs = label.size(0)
        total_loss += loss.item() * bs
        total_n += bs
        all_scores.append(torch.sigmoid(logit).detach().cpu().numpy())
        all_labels.append(label.detach().cpu().numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    pred = (scores >= 0.5).astype(np.int32)
    acc = float((pred == labels.astype(np.int32)).mean())
    auc = _auroc(scores, labels)
    return total_loss / max(1, total_n), acc, auc


def main():
    torch.manual_seed(C.RANDOM_SEED)
    np.random.seed(C.RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    C.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # --- data ---
    df = load_labels(C.LABELS_CSV)
    print(f"Loaded {len(df)} labeled ROIs across {df['recording_ID'].nunique()} recordings.")
    print(f"  positives: {(df['user_defined_cell']==1).sum()}   "
          f"negatives: {(df['user_defined_cell']==0).sum()}")

    train_df, val_df = split_by_roi(df, C.VAL_FRAC, C.RANDOM_SEED)
    print(f"train: {len(train_df)} ROIs  ({train_df['recording_ID'].nunique()} recs)  "
          f"pos={(train_df['user_defined_cell']==1).sum()}  "
          f"neg={(train_df['user_defined_cell']==0).sum()}")
    print(f"val:   {len(val_df)} ROIs  ({val_df['recording_ID'].nunique()} recs)  "
          f"pos={(val_df['user_defined_cell']==1).sum()}  "
          f"neg={(val_df['user_defined_cell']==0).sum()}")

    shared_cache = {}
    train_ds = ROIDataset(train_df, random_crop=True, cache=shared_cache)
    val_ds = ROIDataset(val_df, random_crop=False, cache=shared_cache)

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
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    # class imbalance: weight positives
    n_pos = (train_df["user_defined_cell"] == 1).sum()
    n_neg = (train_df["user_defined_cell"] == 0).sum()
    pos_weight = torch.tensor([max(1.0, n_neg / max(1, n_pos))], device=device)
    print(f"pos_weight: {pos_weight.item():.3f}")

    optim = torch.optim.Adam(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)

    log_path = C.CHECKPOINT_DIR / "train_log.csv"
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

        torch.save(
            {"model": model.state_dict(), "epoch": epoch, "val_auc": va_auc},
            C.CHECKPOINT_DIR / "last.pt",
        )

        if not np.isnan(va_auc) and va_auc > best_auc:
            best_auc = va_auc
            bad_epochs = 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_auc": va_auc},
                C.CHECKPOINT_DIR / "best.pt",
            )
            print(f"  -> new best (val_auc={va_auc:.3f}), checkpoint saved")
        else:
            bad_epochs += 1
            if bad_epochs >= C.EARLY_STOP_PATIENCE:
                print(f"early stop after {bad_epochs} epochs without improvement")
                break

    print(f"best val auc: {best_auc:.3f}")
    print(f"checkpoints in {C.CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()