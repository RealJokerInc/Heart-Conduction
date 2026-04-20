"""Tests for cardiac_ml/utils/seed.py."""
from __future__ import annotations

import random

import numpy as np
import torch

from cardiac_ml.utils.seed import seed_everything


def test_seed_deterministic_torch():
    """Same seed → bitwise-equal torch samples."""
    seed_everything(42)
    a = torch.randn(10)
    seed_everything(42)
    b = torch.randn(10)
    assert torch.equal(a, b)


def test_seed_deterministic_numpy():
    """Same seed → bitwise-equal numpy samples."""
    seed_everything(42)
    a = np.random.randn(10)
    seed_everything(42)
    b = np.random.randn(10)
    assert np.array_equal(a, b)


def test_seed_deterministic_python_random():
    """Same seed → bitwise-equal python random samples."""
    seed_everything(42)
    a = [random.random() for _ in range(10)]
    seed_everything(42)
    b = [random.random() for _ in range(10)]
    assert a == b
