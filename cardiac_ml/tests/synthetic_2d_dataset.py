"""Synthetic 32x32 noise → target dataset for Step 5.3 diffusion stub test."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


class Synthetic2DDataset(Dataset):
    """x: Gaussian noise. y = 2*x - 0.5 (elementwise). Fixed seed → reproducible."""

    def __init__(self, n: int = 32, hw: int = 32, seed: int = 0) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.X = torch.randn(n, 1, hw, hw, generator=gen, dtype=torch.float64)
        self.Y = 2.0 * self.X - 0.5

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[i], self.Y[i]


def make_dataloader(dataset: Dataset, batch_size: int = 8) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
