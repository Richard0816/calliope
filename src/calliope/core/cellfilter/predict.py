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
    python -m calliope.core.cellfilter.predict --plane0 PATH [PATH ...]
        --- predicts for the listed plane0 directories. Multiple
            paths are batched in one Python process so the model
            checkpoint is loaded only once.

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
from pathlib import Path

import numpy as np
import torch

from .. import utils

from . import config as C
from .dataset import _RecordingCache
from .model import CellFilter


@torch.no_grad()                                                  # See model.py
def predict_recording(rec_id: str, model: CellFilter, device: torch.device,
                      plane0: Path | None = None) -> Path:
    """Score every ROI in one recording and write the keep-mask.

    Parameters
    ----------
    rec_id
        Human-readable recording id (used only for the kept-count
        log line at the end). Pass an empty string if you don't
        have one; ``infer_recording_id(plane0)`` can supply it.
    model, device
        Loaded ``CellFilter`` and target torch device.
    plane0
        Suite2p plane0 folder for the recording. Required as of
        2026-05-12; the old ``DATA_ROOT`` resolution path was
        removed when curation labels moved in-project.

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
    if plane0 is None:
        raise ValueError(
            "predict_recording: plane0 is required (DATA_ROOT-based "
            "recording id resolution was removed 2026-05-12).")
    rec = _RecordingCache(Path(plane0))
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


def load_model_from_checkpoint(ckpt_path: Path,
                               device: torch.device) -> CellFilter:
    """Build a fresh ``CellFilter``, restore the saved weights, and
    switch to eval mode. Shared between the CLI here and the
    standalone curation app which also needs to score every ROI.
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. Train first.")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = CellFilter().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main():
    """CLI entry: predict cell-mask for one recording (or many).

    Argument structure (parsed via ``argparse``)
    -------------------------------------------
    --plane0 PATH [PATH ...]
        One or more ``suite2p/plane0`` directories to score.
        Required.
    --ckpt PATH
        Override checkpoint path (defaults to
        ``CHECKPOINT_DIR/best.pt``).

    Steps
    -----
    1. Resolve the checkpoint path.
    2. Pick the device (CUDA when available, else CPU).
    3. Build the model + load weights via ``load_model_from_checkpoint``.
    4. For each ``--plane0`` argument, call ``predict_recording``.
       Per-recording errors are caught individually so a single
       missing memmap doesn't kill the batch.

    History: pre-2026-05-12 this CLI also accepted ``--rec ID`` and
    a no-arg "predict everything under DATA_ROOT" mode. Both were
    removed when ``DATA_ROOT`` came out of the cellfilter config --
    the canonical input is now an explicit plane0 path.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--plane0", type=str, nargs="+", required=True,
        help="One or more suite2p/plane0 directories to score.")
    ap.add_argument(
        "--ckpt", type=str, default=None,
        help="Path to checkpoint. Defaults to CHECKPOINT_DIR/best.pt")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt) if args.ckpt else (C.CHECKPOINT_DIR / "best.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}   ckpt: {ckpt_path}")
    model = load_model_from_checkpoint(ckpt_path, device)

    from .. import utils as _u
    for plane0_str in args.plane0:
        plane0 = Path(plane0_str)
        if not (plane0 / "stat.npy").exists():
            print(f"[skip] {plane0}: no stat.npy")
            continue
        try:
            rec_id = _u.infer_recording_id(plane0)
        except Exception:
            rec_id = plane0.parent.name
        try:
            predict_recording(rec_id, model, device, plane0=plane0)
        except Exception as ex:
            print(f"[skip] {plane0}: {ex}")


if __name__ == "__main__":
    main()