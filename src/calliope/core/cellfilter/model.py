"""
Two-branch cell filter model.

Inputs
------
spatial : (B, 3, H, W)   channels = [mean, max_proj, roi_mask]
trace   : (B, 1, T)      z-scored dF/F, variable length at inference

Output
------
logit   : (B,)           pre-sigmoid cell score. Use torch.sigmoid to get P(cell).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import config as C


def _conv1d_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_c, out_c, kernel_size=7, padding=3),
        nn.BatchNorm1d(out_c),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(2),
    )


def _conv2d_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class TemporalBranch(nn.Module):
    def __init__(self, channels=C.TEMPORAL_CHANNELS, out_dim=C.EMBED_DIM):
        super().__init__()
        prev = 1
        blocks = []
        for c in channels:
            blocks.append(_conv1d_block(prev, c))
            prev = c
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(prev, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T)
        h = self.blocks(x)
        h = self.pool(h).squeeze(-1)  # (B, C)
        return self.proj(h)


class SpatialBranch(nn.Module):
    def __init__(self, in_ch=3, channels=C.SPATIAL_CHANNELS, out_dim=C.EMBED_DIM):
        super().__init__()
        prev = in_ch
        blocks = []
        for c in channels:
            blocks.append(_conv2d_block(prev, c))
            prev = c
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(prev, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        h = self.blocks(x)
        h = self.pool(h).flatten(1)  # (B, C)
        return self.proj(h)


class CellFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal = TemporalBranch()
        self.spatial = SpatialBranch()
        self.head = nn.Sequential(
            nn.Linear(2 * C.EMBED_DIM, C.DENSE_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(C.DROPOUT),
            nn.Linear(C.DENSE_DIM, 1),
        )

    def forward(self, spatial: torch.Tensor, trace: torch.Tensor) -> torch.Tensor:
        z_s = self.spatial(spatial)
        z_t = self.temporal(trace)
        z = torch.cat([z_s, z_t], dim=1)
        return self.head(z).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, spatial: torch.Tensor, trace: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(spatial, trace))
