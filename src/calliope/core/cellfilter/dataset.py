"""Dataset builder for the cell-filter CNN.

What this file does
-------------------
Defines a PyTorch ``Dataset`` (``RoiDataset``) that yields one
training sample per ROI. PyTorch's training loop calls
``len(dataset)`` to know how many samples there are and
``dataset[i]`` to fetch the i-th sample; the framework handles
batching and shuffling automatically.

One sample is the (spatial_patch, trace, label) triple the model
needs:

    spatial_patch : (3, H, W)  float32 - [mean, max_proj, roi_mask]
    trace         : (1, T)     float32 - per-ROI z-scored dF/F
    label         : 0 or 1     - human curation: real cell vs noise

How a sample is built
---------------------
1. The dataset reads the curation CSV (one row per labelled ROI)
   and stores ``(rec_id, roi_id, label)`` tuples internally.
2. On ``dataset[i]``, we look up the recording on disk, slice out
   the 32x32 patch around the ROI's median centroid, build the 3-
   channel image, z-score the dF/F trace, and return the tensors.
3. Per-recording tensors (mean image, max projection, dF/F memmap,
   stat) are cached in ``_REC_TENSOR_CACHE`` so repeated ROIs from
   the same recording don't reload from disk.

R analogy: think of this as a custom data-frame iterator that
encapsulates "given a row id, fetch the appropriate features from a
collection of large files and return them as a list of arrays". The
PyTorch ``DataLoader`` plays the role of ``data.table::split`` +
``parallel::mclapply`` -- it batches and parallelises ``__getitem__``
calls under the hood.

Recordings are resolved by searching ``{DATA_ROOT}\\Cx\\<rec_id>``
and ``{DATA_ROOT}\\Hip\\<rec_id>``. Per-recording tensors (mean,
max, normalized traces, stat) are cached so repeated ROIs from the
same recording don't reload from disk.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .. import utils

from . import config as C


# ---------------- helpers ----------------

_REC_ROOT_CACHE: dict[str, Path] = {}


def _scan_extra_root(extra_root: Path, rec_id: str) -> Optional[Path]:
    """Recursively walk `extra_root` for a directory named `rec_id` that has
    suite2p/plane0. Returns the first match, or None."""
    if not extra_root.exists():
        return None
    for path in extra_root.rglob(rec_id):
        if path.is_dir() and (path / "suite2p" / "plane0").is_dir():
            return path
    return None


def find_recording_root(rec_id: str, data_root: Path = C.DATA_ROOT) -> Path:
    """Resolve a recording id to its root folder.

    Search order:
      1. C.EXTRA_DATA_ROOTS (recursive walk)
      2. data_root\\Cx\\<rec_id>, data_root\\Hip\\<rec_id>
    """
    if rec_id in _REC_ROOT_CACHE:
        return _REC_ROOT_CACHE[rec_id]

    for extra in getattr(C, "EXTRA_DATA_ROOTS", ()):
        hit = _scan_extra_root(Path(extra), rec_id)
        if hit is not None:
            _REC_ROOT_CACHE[rec_id] = hit
            return hit

    for region in ("Cx", "Hip"):
        cand = data_root / region / rec_id
        if cand.exists():
            _REC_ROOT_CACHE[rec_id] = cand
            return cand

    extras = ", ".join(str(p) for p in getattr(C, "EXTRA_DATA_ROOTS", ()))
    raise FileNotFoundError(
        f"Recording not found in extra roots [{extras}] or in {data_root}/Cx,Hip: {rec_id}"
    )


def _znorm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m = x.mean()
    s = x.std()
    return (x - m) / max(s, eps)


def _pad_to_patch(img: np.ndarray, cy: int, cx: int, size: int) -> tuple[np.ndarray, int, int]:
    """
    Crop a (size, size) patch from `img` centered at (cy, cx), zero-padding
    when the window falls outside the image bounds. Returns (patch, y0, x0)
    where (y0, x0) is the top-left corner in image coordinates.
    """
    h, w = img.shape
    half = size // 2
    y0, x0 = cy - half, cx - half
    y1, x1 = y0 + size, x0 + size

    # compute slices with clipping
    sy0, sy1 = max(0, y0), min(h, y1)
    sx0, sx1 = max(0, x0), min(w, x1)

    patch = np.zeros((size, size), dtype=np.float32)
    py0 = sy0 - y0
    px0 = sx0 - x0
    patch[py0:py0 + (sy1 - sy0), px0:px0 + (sx1 - sx0)] = img[sy0:sy1, sx0:sx1]
    return patch, y0, x0


# ---------------- per-recording cache ----------------

class _RecordingCache:
    """One recording's worth of arrays, loaded once and re-used.

    Why we cache
    ------------
    Training visits each ROI of a recording multiple times across
    epochs. ROIs from the same recording all need the same mean image,
    max projection, and dF/F memmap; loading those from disk on every
    ``__getitem__`` would be wasteful. ``_RecordingCache``
    consolidates them, exposes lazy-but-ready getters for the patch
    and trace, and is shared across ROIs by ``ROIDataset``'s
    ``_cache`` dict.

    Attributes
    ----------
    stat       : ndarray of suite2p stat dicts.
    mean_img_z : (H, W) float32 mean image, z-normalised.
    max_img_z  : (H, W) float32 max projection, z-normalised. Padded
                 to match the mean image's shape if Suite2p cropped
                 it.
    dff        : (T, N) float32 memmap of neuropil-corrected dF/F.
    T, N       : trace length and ROI count.
    H, W       : FOV dimensions (pixels).
    """
    def __init__(self, rec_id: str, plane0: Optional[Path] = None):
        self.rec_id = rec_id
        # Two ways to specify a recording: an explicit ``plane0``
        # path (used by ``predict_recording`` from the GUI) or a
        # bare ``rec_id`` that gets resolved through ``find_recording_root``
        # (used by training).
        if plane0 is not None:
            self.plane0 = Path(plane0)
            # ``plane0`` is ``<root>/suite2p/plane0`` -- two
            # ``.parent`` calls reach the recording root.
            self.root = self.plane0.parent.parent
        else:
            self.root = find_recording_root(rec_id)
            self.plane0 = self.root / "suite2p" / "plane0"

        stat = np.load(self.plane0 / "stat.npy", allow_pickle=True)
        view = utils.load_plane_view(self.plane0)

        mean_img = view.get("meanImgE", None)
        if mean_img is None:
            mean_img = view.get("meanImg")
        mean_img = np.asarray(mean_img, dtype=np.float32)

        max_img = view.get("max_proj", None)
        if max_img is None:
            max_img = view.get("maxImg", None)
        if max_img is None:
            max_img = mean_img
        max_img = np.asarray(max_img, dtype=np.float32)

        # If max_proj is cropped (common with suite2p), pad it back to full FOV
        if max_img.shape != mean_img.shape:
            H, W = mean_img.shape
            y0 = int(view.get("yrange", [0, H])[0])
            x0 = int(view.get("xrange", [0, W])[0])
            padded = np.zeros_like(mean_img)
            mh, mw = max_img.shape
            padded[y0:y0 + mh, x0:x0 + mw] = max_img
            max_img = padded

        # dF/F memmap
        dff, _, _, T, N = utils.s2p_open_memmaps(self.plane0, prefix=C.DFF_PREFIX)

        self.stat = stat
        self.mean_img_z = _znorm(mean_img)
        self.max_img_z = _znorm(max_img)
        self.dff = dff
        self.T = T
        self.N = N
        self.H, self.W = mean_img.shape

    def get_patch(self, roi_idx: int, size: int) -> np.ndarray:
        """Return (3, size, size) float32 patch: [mean, max, mask]."""
        s = self.stat[roi_idx]
        xpix = s["xpix"]
        ypix = s["ypix"]
        cy = int(round(float(ypix.mean())))
        cx = int(round(float(xpix.mean())))

        mean_patch, y0, x0 = _pad_to_patch(self.mean_img_z, cy, cx, size)
        max_patch, _, _ = _pad_to_patch(self.max_img_z, cy, cx, size)

        mask_full = np.zeros((self.H, self.W), dtype=np.float32)
        mask_full[ypix, xpix] = 1.0
        mask_patch, _, _ = _pad_to_patch(mask_full, cy, cx, size)

        return np.stack([mean_patch, max_patch, mask_patch], axis=0)

    def get_trace(self, roi_idx: int) -> np.ndarray:
        """Return (T,) float32 per-ROI z-scored trace."""
        trace = np.asarray(self.dff[:, roi_idx], dtype=np.float32)
        return _znorm(trace)


# ---------------- dataset ----------------

class ROIDataset(Dataset):
    """PyTorch Dataset adapter for the cell-filter training pipeline.

    Why subclass ``torch.utils.data.Dataset``
    -----------------------------------------
    PyTorch's training infrastructure (``DataLoader``, batching,
    shuffling, multiprocessing) all rely on the Dataset *protocol*:
    a class that exposes ``__len__`` (how many samples) and
    ``__getitem__(i)`` (fetch the i-th sample). Once you implement
    those two, ``DataLoader(dataset, batch_size=32, shuffle=True)``
    handles batching, shuffling and parallel preloading
    automatically.

    Each ``__getitem__`` returns:
      spatial : (3, H, W) torch.Tensor
          (mean, max-projection, ROI mask).
      trace   : (1, T_crop) torch.Tensor during training,
                 (1, T_full) during eval.
      label   : torch.Tensor, scalar 0.0 or 1.0
          The human curation: 1 = real cell, 0 = noise.

    Constructor parameters
    ----------------------
    labels_df : pd.DataFrame
        One row per ROI with columns ``recording_ID``, ``ROI_number``,
        ``user_defined_cell``.
    patch_size : int
        Size of the spatial patch around the ROI centroid.
    trace_crop : int or None
        If set, return a randomly-cropped sub-trace this long during
        training. ``None`` -> always return the full trace (for
        inference / validation).
    random_crop : bool
        ``True`` for random training crops, ``False`` for centred
        deterministic crops (validation).
    cache : dict, optional
        Pre-existing ``rec_id -> _RecordingCache`` dict. Pass the
        same one to train and val Datasets so per-recording arrays
        load only once across both.
    """
    def __init__(
        self,
        labels_df: pd.DataFrame,
        *,
        patch_size: int = C.PATCH_SIZE,
        trace_crop: Optional[int] = C.TRACE_CROP_LEN,
        random_crop: bool = True,
        cache: Optional[dict] = None,
    ):
        self.df = labels_df.reset_index(drop=True)
        self.patch_size = patch_size
        self.trace_crop = trace_crop
        self.random_crop = random_crop
        self._cache = cache if cache is not None else {}

    def _get_rec(self, rec_id: str) -> _RecordingCache:
        if rec_id not in self._cache:
            self._cache[rec_id] = _RecordingCache(rec_id)
        return self._cache[rec_id]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        rec_id = str(row["recording_ID"])
        roi = int(row["ROI_number"])
        label = float(row["user_defined_cell"])

        rec = self._get_rec(rec_id)
        patch = rec.get_patch(roi, self.patch_size)    # (3, H, W)
        trace = rec.get_trace(roi)                     # (T,)

        if self.trace_crop is not None and trace.shape[0] >= self.trace_crop:
            if self.random_crop:
                start = np.random.randint(0, trace.shape[0] - self.trace_crop + 1)
            else:
                start = (trace.shape[0] - self.trace_crop) // 2
            trace = trace[start:start + self.trace_crop]
        elif self.trace_crop is not None and trace.shape[0] < self.trace_crop:
            # pad end with zeros if shorter than crop length
            pad = self.trace_crop - trace.shape[0]
            trace = np.concatenate([trace, np.zeros(pad, dtype=np.float32)])

        return (
            torch.from_numpy(patch),
            torch.from_numpy(trace[None, :]),
            torch.tensor(label, dtype=torch.float32),
        )


# ---------------- splits ----------------

def load_labels(csv_path: Path = C.LABELS_CSV) -> pd.DataFrame:
    """Load the curation CSV and clean it up.

    The CSV is the human-labelled ground truth: lab members open
    each recording's ROIs in a Suite2p-like browser and tick
    "cell" / "not cell" based on visual inspection. This function:
        1. Reads the CSV.
        2. Coerces dtypes (recording id as string, others as int).
        3. Drops duplicates -- keeps the LAST occurrence so a later
           re-labelling overrides an earlier mistake.

    Returns a DataFrame ready to feed to ``ROIDataset``.
    """
    df = pd.read_csv(csv_path)
    df["recording_ID"] = df["recording_ID"].astype(str)
    df["ROI_number"] = df["ROI_number"].astype(int)
    df["user_defined_cell"] = df["user_defined_cell"].astype(int)
    # drop duplicates, keeping last (so corrections win)
    df = df.drop_duplicates(subset=["recording_ID", "ROI_number"], keep="last")
    return df.reset_index(drop=True)


def split_by_recording(
    df: pd.DataFrame,
    val_frac: float = C.VAL_FRAC,
    seed: int = C.RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/val split that keeps **whole recordings** in one or the
    other set.

    Why do it this way: ROIs from the same recording share the
    background mean image, motion artefacts, focus level, etc.
    Splitting at the ROI level (a random 80/20) would let
    information leak between train and val -- the model could learn
    "this is a cell because it's bright relative to the rest of
    *this exact recording*" rather than a general rule. Splitting at
    the recording level forces it to generalise across slices.
    """
    rng = np.random.default_rng(seed)
    # ``np.array`` + ``rng.shuffle`` shuffles in-place. ``sorted``
    # first so the same seed gives the same split regardless of
    # input dataframe ordering.
    recs = np.array(sorted(df["recording_ID"].unique()))
    rng.shuffle(recs)
    n_val = max(1, int(round(len(recs) * val_frac)))
    # Slice the first n_val shuffled recordings for validation.
    val_recs = set(recs[:n_val].tolist())
    # ``df.isin(set)`` returns a boolean Series; ``~`` negates.
    train_df = df[~df["recording_ID"].isin(val_recs)].reset_index(drop=True)
    val_df = df[df["recording_ID"].isin(val_recs)].reset_index(drop=True)
    return train_df, val_df


def split_by_roi(
    df: pd.DataFrame,
    val_frac: float = C.VAL_FRAC,
    seed: int = C.RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random per-ROI split, stratified by label.

    Use this when there's only one (or very few) recordings -- the
    recording-level split would either drop every positive ROI into
    val (and leave training with none) or vice versa.
    Stratification guarantees both classes appear in train and val
    in roughly the same proportion as the source df.
    """
    rng = np.random.default_rng(seed)
    parts_train, parts_val = [], []
    for label, sub in df.groupby("user_defined_cell"):
        idx = np.arange(len(sub))
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac))) if len(idx) > 1 else 0
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        parts_val.append(sub.iloc[val_idx])
        parts_train.append(sub.iloc[train_idx])
    train_df = pd.concat(parts_train, ignore_index=True).sample(
        frac=1.0, random_state=seed
    ).reset_index(drop=True)
    val_df = pd.concat(parts_val, ignore_index=True).reset_index(drop=True)
    return train_df, val_df
