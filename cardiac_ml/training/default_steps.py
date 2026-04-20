"""Default train_step_fn / val_step_fn for the teacher-forced case.

Used by conf/training/teacher_forced.yaml. The Trainer imports this
directly (not via Hydra) when cfg.training.train_step_fn is absent.

Protocol keys the Trainer recognizes on the returned dict (all optional —
omit if unused):

  "_backward_done": bool
      True if the step called `loss.backward()` itself (adjoint case).
      Trainer skips its default backward.
  "_on_after_backward": Callable[[], None]
      Post-backward cleanup (e.g., IonicNODE's `clear_v_trajectory`).
      Trainer invokes on train AND val paths so stateful caches are
      cleaned regardless of gradient mode.

These default step functions use neither flag.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def teacher_forced_step(trainer: Any, batch: Any) -> dict:
    """Default train_step_fn: forward → MSE → return dict.

    Expects `batch` to unpack as `(x, y)`. Loss has `requires_grad=True`
    because it's part of the model's forward graph.
    """
    x, y = batch
    pred = trainer.model(x)
    loss = F.mse_loss(pred, y)
    return {"loss": loss}


def teacher_forced_val_step(trainer: Any, batch: Any) -> dict:
    """Default val_step_fn: no-grad forward → MSE.

    Returns both "loss" (Trainer protocol requires the key) and "val_loss"
    (same value, used for early-stopping monitors and for mlflow metric
    prefixing).
    """
    with torch.no_grad():
        x, y = batch
        pred = trainer.model(x)
        loss = F.mse_loss(pred, y)
    return {"loss": loss, "val_loss": loss}
