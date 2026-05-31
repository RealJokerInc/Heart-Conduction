"""Test-only model + dataset classes for the end-to-end synthetic smoke test.

Living in cardiac_ml/tests/ so Hydra `_target_` paths resolve without
polluting the main package. The end-to-end synthetic_smoke experiment
references these via Hydra `_target_: cardiac_ml.tests.synthetic_harness_fixtures.*`.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class SyntheticMLP(nn.Module):
    """Pure 4→4 linear layer. Converges quickly on SyntheticLinear targets."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class SyntheticLinear(Dataset):
    """y = x @ W + b. Fixed (W, b) via param_seed; X samples via sample_seed."""

    def __init__(self, n: int = 128, sample_seed: int = 0, param_seed: int = 999) -> None:
        gp = torch.Generator().manual_seed(param_seed)
        gs = torch.Generator().manual_seed(sample_seed)
        self.W = torch.randn(4, 4, generator=gp, dtype=torch.float64)
        self.b = torch.randn(4, generator=gp, dtype=torch.float64)
        self.X = torch.randn(n, 4, generator=gs, dtype=torch.float64)
        self.Y = self.X @ self.W + self.b

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[i], self.Y[i]


def make_dataloader(dataset: Dataset, batch_size: int = 16) -> DataLoader:
    """Factory function (Hydra _target_ instantiation calls this directly)."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
