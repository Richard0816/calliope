"""Run the trained cell-filter CNN over a recording.

Why this script exists
----------------------
Tab 3's detection step produces a ``stat.npy`` with N candidate
ROIs. Many of those candidates are noise / blood vessels / motion
artefacts. ``train.py`` produces a model checkpoint that scores any
ROI as "real cell" vs "not"; this file is the runtime that loads the
checkpoint and applies it to every ROI in a recording, then writes:

    suite2p/plane0/predicted_cell_prob.npy   float32 (N,)
        Sigmoid-of-logit probabilities, one per ROI.
    suite2p/plane0/predicted_cell_mask.npy   bool (N,)
        ``probs >= THRESHOLD`` -- the actual keep mask Tabs 4-8 use.

The GUI calls ``predict_recording`` after Tab 3's detection finishes;
the CLI in ``main()`` is for batch-running over many recordings on a
GPU box.

Usage
-----
    python -m cellfilter.predict
        --- predicts for every recording folder found under
            DATA_ROOT\\{Cx,Hip}\\

    python -m cellfilter.predict --rec 2024-07-01_00018
        --- predicts for a specific recording

Accepts full traces at inference (no random crop).
"""
from __future__ import annotations

# Set process-level environment variables BEFORE numpy / torch
# import to avoid a known Intel MKL multi-threading conflict on
# Windows. ``setdefault`` only sets the variable if the user hasn't
# already configured it.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ``argparse`` is the stdlib command-line parser. Roughly the
# equivalent of R's ``optparse::OptionParser``.
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from .. import utils

from . import config as C
from .dataset import _RecordingCache, find_recording_root
from .model import CellFilter


@torch.no_grad()                                                  # See model.py
def predict_recording(rec_id: str, model: CellFilter, device: torch.device,
                      plane0: Path | None = None) -> Path:
    """Score every ROI in one recording and write the keep-mask.

    Steps:
        1. Open a ``_RecordingCache`` (lazy-loads mean image, max
           projection, dF/F memmap, stat).
        2. Loop over all N ROIs:
            a. ``rec.get_patch(roi, 32)`` -> (3, H, W) numpy array.
            b. ``rec.get_trace(roi)``     -> (T,) numpy array.
            c. ``torch.from_numpy(...)[None]`` adds a batch
               dimension; ``.to(device)`` ships to GPU if available.
            d. ``model(spatial, trace_t)`` -> raw logit;
               ``torch.sigmoid(...).item()`` -> Python float.
        3. Save ``predicted_cell_prob.npy`` and
           ``predicted_cell_mask.npy`` in plane0.

    The ``@torch.no_grad()`` decorator on the function disables
    autograd for the whole call -- saves memory and time at
    inference.
    """
    rec = _RecordingCache(rec_id, plane0=plane0)
    N = rec.N

    probs = np.zeros(N, dtype=np.float32)
    for roi in range(N):
        patch = rec.get_patch(roi, C.PATCH_SIZE)          # (3, H, W)
        trace = rec.get_trace(roi)                        # (T,)
        # ``[None]`` is NumPy/Torch shorthand for "insert a new axis
        # of size 1 here". The model expects a batch dimension, so
        # we add one to turn (3, H, W) into (1, 3, H, W).
        spatial = torch.from_numpy(patch)[None].to(device)
        # ``[None, None]`` adds two new axes: trace was (T,), now
        # (1, 1, T) which is (batch, channels, time).
        trace_t = torch.from_numpy(trace)[None, None].to(device)
        logit = model(spatial, trace_t)
        # ``sigmoid`` -> probability; ``.item()`` extracts the Python
        # scalar from a 0-D tensor.
        probs[roi] = torch.sigmoid(logit).item()

    out_prob = rec.plane0 / C.PREDICTED_PROB_NAME
    out_mask = rec.plane0 / C.PREDICTED_MASK_NAME
    np.save(out_prob, probs)
    # Boolean mask: NumPy treats ``probs >= C.THRESHOLD`` as a
    # vectorised comparison returning a bool array of the same
    # shape.
    np.save(out_mask, probs >= C.THRESHOLD)
    print(f"{rec_id}: {(probs >= C.THRESHOLD).sum()}/{N} kept   "
          f"-> {out_prob.name}, {out_mask.name}")
    return out_prob


def list_all_recordings(data_root: Path = C.DATA_ROOT) -> list[str]:
    """Walk DATA_ROOT/{Cx, Hip}/ and return every recording id whose
    Suite2p output is on disk.

    "Recording id" = the folder name; we treat anything containing
    ``suite2p/plane0/stat.npy`` as a candidate. Sorted alphabetically
    so the batch order is reproducible.
    """
    rec_ids = []
    for region in ("Cx", "Hip"):
        region_dir = data_root / region
        if not region_dir.exists():
            continue
        for p in region_dir.iterdir():
            if p.is_dir() and (p / "suite2p" / "plane0" / "stat.npy").exists():
                rec_ids.append(p.name)
    return sorted(rec_ids)


def main():
    """CLI entry: predict cell-mask for one recording or every recording.

    Argument structure (parsed via ``argparse``)
    -------------------------------------------
    --rec ID
        Run on a single recording id. Mutually informative with --plane0.
    --plane0 PATH
        Direct path to a ``suite2p/plane0`` folder; bypasses the
        ``DATA_ROOT/{Cx,Hip}/<rec_id>`` resolution above.
    --ckpt PATH
        Override checkpoint path (defaults to
        ``CHECKPOINT_DIR/best.pt``).

    Steps
    -----
    1. Resolve the checkpoint path.
    2. Pick the device (CUDA when available, else CPU).
    3. ``torch.load`` the checkpoint, build a fresh ``CellFilter``,
       restore weights via ``load_state_dict``, switch to eval mode.
    4. Either run on the explicit ``--plane0`` path, the single
       ``--rec`` id, or every recording under DATA_ROOT.
    5. Per-recording errors are caught individually so a single
       missing memmap doesn't kill a batch run.
    """
    # ``argparse`` is the standard CLI parser. ``add_argument`` is
    # the equivalent of declaring an option in R's ``optparse``.
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", type=str, default=None,
                    help="Single recording ID (e.g. 2024-07-01_00018). "
                         "If omitted, predicts every recording under DATA_ROOT.")
    ap.add_argument("--plane0", type=str, default=None,
                    help="Direct path to a suite2p/plane0 directory. Bypasses "
                         "DATA_ROOT/{Cx,Hip}/<rec_id> resolution.")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="Path to checkpoint. Defaults to CHECKPOINT_DIR/best.pt")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt) if args.ckpt else (C.CHECKPOINT_DIR / "best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Train first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}   ckpt: {ckpt_path}")

    # ``torch.load`` reads the checkpoint dict (model state +
    # metadata) we wrote in train.py. ``map_location`` rehomes
    # tensors saved on a different device.
    # ``weights_only=False`` -> allow non-tensor entries (epoch
    # number, val AUROC, etc.) in the checkpoint dict.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = CellFilter().to(device)
    # ``load_state_dict`` overwrites the model's parameters with the
    # saved values.
    model.load_state_dict(ckpt["model"])
    # ``eval()`` flips the model into inference mode (disables
    # dropout, freezes BatchNorm running stats).
    model.eval()

    # Branch 1: caller pointed us at an explicit plane0 path.
    if args.plane0:
        plane0 = Path(args.plane0)
        if not (plane0 / "stat.npy").exists():
            raise FileNotFoundError(f"No stat.npy in {plane0}")
        # Resolve the recording id robustly: the new
        # sparse_plus_cellpose layout puts plane0 deeper than
        # ``<rec>/suite2p/plane0`` so the legacy two-parents-up
        # walk lands on ``final`` instead of the recording id.
        from .. import utils as _u
        rec_id = args.rec or _u.infer_recording_id(plane0)
        predict_recording(rec_id, model, device, plane0=plane0)
        return

    # Branch 2: --rec for one recording, otherwise scan DATA_ROOT.
    if args.rec:
        rec_ids = [args.rec]
    else:
        rec_ids = list_all_recordings()
        print(f"Found {len(rec_ids)} recordings.")

    for rid in rec_ids:
        try:
            predict_recording(rid, model, device)
        except Exception as ex:
            # Per-recording try/except: don't abort the whole batch
            # because one recording is missing a memmap.
            print(f"[skip] {rid}: {ex}")


if __name__ == "__main__":
    main()