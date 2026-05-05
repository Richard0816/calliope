"""
Run the trained cell filter over every ROI in a set of recordings and write
    suite2p/plane0/predicted_cell_prob.npy   float32 (N,)   sigmoid scores
    suite2p/plane0/predicted_cell_mask.npy   bool    (N,)   prob >= THRESHOLD

Usage
-----
    python -m cellfilter.predict
        --- predicts for every recording folder found under DATA_ROOT\\{Cx,Hip}\\

    python -m cellfilter.predict --rec 2024-07-01_00018
        --- predicts for a specific recording

Accepts full traces at inference (no random crop).
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from .. import utils

from . import config as C
from .dataset import _RecordingCache, find_recording_root
from .model import CellFilter


@torch.no_grad()
def predict_recording(rec_id: str, model: CellFilter, device: torch.device,
                      plane0: Path | None = None) -> Path:
    rec = _RecordingCache(rec_id, plane0=plane0)
    N = rec.N

    probs = np.zeros(N, dtype=np.float32)
    for roi in range(N):
        patch = rec.get_patch(roi, C.PATCH_SIZE)          # (3, H, W)
        trace = rec.get_trace(roi)                        # (T,)
        spatial = torch.from_numpy(patch)[None].to(device)
        trace_t = torch.from_numpy(trace)[None, None].to(device)  # (1, 1, T)
        logit = model(spatial, trace_t)
        probs[roi] = torch.sigmoid(logit).item()

    out_prob = rec.plane0 / C.PREDICTED_PROB_NAME
    out_mask = rec.plane0 / C.PREDICTED_MASK_NAME
    np.save(out_prob, probs)
    np.save(out_mask, probs >= C.THRESHOLD)
    print(f"{rec_id}: {(probs >= C.THRESHOLD).sum()}/{N} kept   "
          f"-> {out_prob.name}, {out_mask.name}")
    return out_prob


def list_all_recordings(data_root: Path = C.DATA_ROOT) -> list[str]:
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

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = CellFilter().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if args.plane0:
        plane0 = Path(args.plane0)
        if not (plane0 / "stat.npy").exists():
            raise FileNotFoundError(f"No stat.npy in {plane0}")
        rec_id = args.rec or plane0.parent.parent.name
        predict_recording(rec_id, model, device, plane0=plane0)
        return

    if args.rec:
        rec_ids = [args.rec]
    else:
        rec_ids = list_all_recordings()
        print(f"Found {len(rec_ids)} recordings.")

    for rid in rec_ids:
        try:
            predict_recording(rid, model, device)
        except Exception as ex:
            print(f"[skip] {rid}: {ex}")


if __name__ == "__main__":
    main()