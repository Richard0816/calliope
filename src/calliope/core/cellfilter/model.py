"""Two-branch cell-filter CNN.

What this network does
----------------------
Each ROI Suite2p detects gets fed two parallel inputs:

    spatial : (B, 3, H, W)
        A 32x32 patch of the FOV centred on the ROI. The 3 channels
        carry different views of the same patch:
            channel 0 = mean image (long-term average brightness)
            channel 1 = max projection (peak brightness over time)
            channel 2 = the ROI's own pixel mask
        The network learns "does the soma in the centre of this
        patch look like a real cell, given the surrounding context?"

    trace : (B, 1, T)
        The z-scored dF/F trace. T is variable -- training crops
        random T = TRACE_CROP_LEN (2000) frames; inference can pass
        the whole recording. The network learns "does this trace
        look like calcium-imaging activity (transient bumps with
        slow rise + slow decay) or noise / motion artefact?"

Each branch is a small ResNet-style CNN (1-D for the trace, 2-D for
the patch). Their outputs are concatenated and passed through a tiny
MLP that emits a single logit. Sigmoid -> probability of "this is a
real cell".

R analogy: this is a feature-engineered classifier that takes two
heterogeneous inputs (a 2-D image and a 1-D timeseries), squeezes
each through its own pipeline of convolutions and pools, then merges
them. Conceptually similar to a multi-modal ``randomForest`` that
takes engineered features from each input -- only here the
"features" are learned end-to-end.

Class layout
------------
``_conv1d_block`` / ``_conv2d_block``
    Tiny helpers: Conv -> BatchNorm -> ReLU -> MaxPool. Used to build
    each branch.
``TemporalBranch``
    Stack of 1-D conv blocks over the trace, ending in a global
    average pool and a linear projection to a 64-d embedding.
``SpatialBranch``
    Same idea but 2-D over the (3, 32, 32) patch.
``CellFilter``
    Brings the two branches together. ``forward(spatial, trace)``
    returns the raw logit; ``predict_proba`` wraps it in sigmoid +
    no_grad for inference.

Inputs / outputs
----------------
spatial : (B, 3, H, W)   channels = [mean, max_proj, roi_mask]
trace   : (B, 1, T)      z-scored dF/F, variable length at inference

Output  (forward)
-----------------
logit   : (B,)           pre-sigmoid cell score. Use torch.sigmoid to
                          get P(cell) or call predict_proba.
"""
from __future__ import annotations

# PyTorch is the deep-learning library. ``torch`` is the array
# backend; ``torch.nn`` holds the layer / module classes used to
# build networks.
import torch
import torch.nn as nn

# Centralised hyperparameters (channel widths, embedding dim,
# dropout). See cellfilter/config.py.
from . import config as C


def _conv1d_block(in_c: int, out_c: int) -> nn.Sequential:
    """One 1-D conv block: Conv1d -> BN -> ReLU -> MaxPool.

    Returns an ``nn.Sequential`` -- a pre-built mini-network that
    runs each layer in order. The MaxPool halves the time dimension
    so a stack of three blocks reduces a 2000-sample trace to ~250.
    """
    return nn.Sequential(
        # ``kernel_size=7`` covers ~7 frames at a time; ``padding=3``
        # keeps the time dimension the same after the conv (only the
        # MaxPool below shrinks it).
        nn.Conv1d(in_c, out_c, kernel_size=7, padding=3),
        # BatchNorm normalises activations across the batch ->
        # speeds up training and reduces sensitivity to weight init.
        nn.BatchNorm1d(out_c),
        # ReLU = non-linearity, ``inplace=True`` re-uses the input
        # buffer to save memory.
        nn.ReLU(inplace=True),
        # Down-sample by 2 along time.
        nn.MaxPool1d(2),
    )


def _conv2d_block(in_c: int, out_c: int) -> nn.Sequential:
    """One 2-D conv block: Conv2d -> BN -> ReLU -> MaxPool. Same
    pattern as the 1-D version but with 3x3 spatial kernels."""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class TemporalBranch(nn.Module):
    """1-D CNN over the dF/F trace -> 64-d embedding.

    Why we need a separate branch
    -----------------------------
    Real calcium-imaging traces have a characteristic shape: fast
    rise, slow exponential decay, transient bumps roughly every few
    seconds. A 1-D conv stack over the trace can learn to recognise
    that shape, regardless of the absolute amplitude. This is what
    distinguishes a real cell from e.g. a constant-bright pixel of
    a blood vessel that happens to overlap with an ROI.
    """

    def __init__(self, channels=C.TEMPORAL_CHANNELS, out_dim=C.EMBED_DIM):
        # ``super().__init__()`` initialises the parent ``nn.Module``
        # bookkeeping (parameter registration, GPU moves, etc.).
        super().__init__()
        prev = 1   # input has one channel (the raw trace).
        # Build a list of conv blocks, doubling channels each time.
        blocks = []
        for c in channels:
            blocks.append(_conv1d_block(prev, c))
            prev = c
        # ``Sequential(*blocks)`` packages them into one runnable
        # module. ``*blocks`` unpacks the list as positional args.
        self.blocks = nn.Sequential(*blocks)
        # Adaptive avg pool to length 1: collapses time to a single
        # value per channel regardless of the input trace length.
        self.pool = nn.AdaptiveAvgPool1d(1)
        # Linear projection: per-channel mean -> 64-d embedding.
        self.proj = nn.Linear(prev, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ``x`` arrives shaped (batch, 1, time).
        h = self.blocks(x)
        # ``squeeze(-1)`` removes the now-size-1 time axis so we end
        # up with (B, C).
        h = self.pool(h).squeeze(-1)
        return self.proj(h)


class SpatialBranch(nn.Module):
    """2-D CNN over the (mean, max-proj, roi-mask) patch -> 64-d
    embedding.

    Why three channels rather than just the mean image
    --------------------------------------------------
    - **mean image** captures the steady appearance: a real cell
      should be slightly brighter than its surroundings on average.
    - **max projection** captures whether the spot ever lights up
      transiently -- a quiet but real cell may not stand out in the
      mean but does in the max-proj.
    - **roi mask** tells the network exactly which pixels Suite2p
      thinks belong to this candidate, so the network can compare
      "what's inside the proposed footprint" vs "what's outside".
    """

    def __init__(self, in_ch=3, channels=C.SPATIAL_CHANNELS, out_dim=C.EMBED_DIM):
        super().__init__()
        prev = in_ch
        blocks = []
        for c in channels:
            blocks.append(_conv2d_block(prev, c))
            prev = c
        self.blocks = nn.Sequential(*blocks)
        # 2-D adaptive average pool to (1, 1) -> per-channel scalar.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(prev, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        h = self.blocks(x)
        # ``flatten(1)`` collapses the trailing (1, 1) spatial dims
        # to (B, C).
        h = self.pool(h).flatten(1)
        return self.proj(h)


class CellFilter(nn.Module):
    """The full classifier. Combines the two branches and emits one
    logit per ROI.

    When this gets called
    ---------------------
    - **Training** (``cellfilter/train.py``): one ``forward`` call
      per minibatch, followed by ``BCEWithLogitsLoss`` -> backward
      pass -> Adam step.
    - **Inference** (``cellfilter/predict.py`` -> Tab 3): one call
      per ROI on a freshly-loaded recording. The returned probability
      is thresholded at ``config.THRESHOLD`` to produce the
      ``predicted_cell_mask.npy`` Tab 3 saves into ``plane0/``.
    """

    def __init__(self):
        super().__init__()
        # Each branch is a separate sub-module so PyTorch knows about
        # their parameters when we call ``.parameters()`` for Adam.
        self.temporal = TemporalBranch()
        self.spatial = SpatialBranch()
        # Classifier head: (128) -> (64) -> (1). Dropout in between
        # for regularisation; the final 1-d output is the raw logit.
        self.head = nn.Sequential(
            nn.Linear(2 * C.EMBED_DIM, C.DENSE_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(C.DROPOUT),
            nn.Linear(C.DENSE_DIM, 1),
        )

    def forward(self, spatial: torch.Tensor, trace: torch.Tensor) -> torch.Tensor:
        # Run each branch independently...
        z_s = self.spatial(spatial)
        z_t = self.temporal(trace)
        # ...then concatenate along the feature axis (B, 64+64=128)
        # before the classifier head.
        z = torch.cat([z_s, z_t], dim=1)
        # ``squeeze(-1)`` drops the trailing size-1 axis so the
        # output is shape (B,) instead of (B, 1) -- nicer for
        # downstream loss functions.
        return self.head(z).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, spatial: torch.Tensor, trace: torch.Tensor) -> torch.Tensor:
        """Inference convenience: returns sigmoid(forward(...)).

        The ``@torch.no_grad()`` decorator turns off gradient tracking
        for the whole call, saving memory and CPU when we don't need
        the backward pass -- always the case at inference time.
        """
        return torch.sigmoid(self.forward(spatial, trace))
