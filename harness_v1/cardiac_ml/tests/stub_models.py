"""Stub models used by reusability-proof tests (Step 5.3).

The DiffusionResNetStub is not a real diffusion implementation — it's a
trivial residual CNN that exists to prove the harness handles a second
structurally distinct consumer (conv vs. ODE) without Trainer changes.
"""
from __future__ import annotations

import torch
from torch import nn


class DiffusionResNetStub(nn.Module):
    """Trivial residual 2-layer CNN over (B, C, H, W) float64 inputs."""

    def __init__(self, in_ch: int = 1, hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, in_ch, 3, padding=1),
        ).double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)
