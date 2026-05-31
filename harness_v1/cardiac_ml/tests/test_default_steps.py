"""Tests for cardiac_ml/training/default_steps.py."""
from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from cardiac_ml.training.default_steps import (
    teacher_forced_step,
    teacher_forced_val_step,
)


def _toy_trainer():
    """Synthetic trainer stub with just the .model attribute step fns need."""
    model = nn.Linear(4, 4).double()
    return SimpleNamespace(model=model)


def test_teacher_forced_returns_dict_with_loss():
    """Return dict has scalar float64 loss with requires_grad=True."""
    trainer = _toy_trainer()
    x = torch.randn(8, 4, dtype=torch.float64)
    y = torch.randn(8, 4, dtype=torch.float64)
    out = teacher_forced_step(trainer, (x, y))
    assert "loss" in out
    loss = out["loss"]
    assert loss.dim() == 0
    assert loss.dtype == torch.float64
    assert loss.requires_grad is True


def test_val_step_no_grad():
    """Val step returns loss with requires_grad=False."""
    trainer = _toy_trainer()
    x = torch.randn(8, 4, dtype=torch.float64)
    y = torch.randn(8, 4, dtype=torch.float64)
    out = teacher_forced_val_step(trainer, (x, y))
    assert out["loss"].requires_grad is False
    assert out["val_loss"].requires_grad is False
    # Same tensor reused as both "loss" and "val_loss".
    assert torch.equal(out["loss"], out["val_loss"])


def test_neither_sets_protocol_flags():
    """Default steps don't use _backward_done or _on_after_backward."""
    trainer = _toy_trainer()
    x = torch.randn(4, 4, dtype=torch.float64)
    y = torch.randn(4, 4, dtype=torch.float64)
    out = teacher_forced_step(trainer, (x, y))
    assert "_backward_done" not in out
    assert "_on_after_backward" not in out
