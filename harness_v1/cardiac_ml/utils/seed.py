"""Deterministic seeding across torch, numpy, python random, CUDA.

Called by Trainer.__init__ once per fit. Reproducibility (NFR-1) depends
on this covering every RNG source any training step might touch.
"""
from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed torch, numpy, python random, and CUDA (if available).

    Does NOT call `torch.use_deterministic_algorithms(True)` — that would
    disable cuDNN fast paths and is a separate opt-in at the training
    config level if bit-exact reproducibility across runs is required.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
