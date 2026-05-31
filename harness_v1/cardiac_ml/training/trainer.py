"""cardiac_ml.Trainer — single flexible training loop for all consumers.

The Trainer accepts a Hydra DictConfig, instantiates model / data / optimizer
via `_target_`, loads `train_step_fn` / `val_step_fn` via Hydra's
`hydra.utils.get_method`, and runs a standard epoch/batch loop with callback
hooks. Model-specific training logic lives in pure step functions injected
via config — see `cardiac_ml/training/default_steps.py` for the default
teacher-forced pair.

Protocol keys the Trainer recognizes on each step's return dict:

  "_backward_done": bool
      True if the step called `loss.backward()` itself (adjoint case).
      Trainer skips its default backward. Asserts loss is graph-attached.
  "_on_after_backward": Callable[[], None]
      Post-backward cleanup (e.g., IonicNODE's `clear_v_trajectory`).
      Invoked on BOTH train and val paths so stateful caches are cleared
      regardless of gradient mode. Wrapped in try/except that zero-grads
      and re-raises — prevents silent stale-grad corruption on hook failure.

Round-3 fixes applied:
- M-1: no `torch.set_default_dtype(...)` (process-global); dtype cast via
       `.to(device, dtype)` on the model only.
- M-2 / M-7: `log_artifact` / `log_figure` escape hatches dispatch through
       `self._logger` (Callback-base proxy methods). Trainer never imports
       mlflow directly.
- M-4: batch cast via `_to_device_and_dtype` handles heterogeneous dict
       batches (float-only promotion, non-tensor pass-through).
- M-9: `_on_after_backward` wrapped in try/except that zero-grads and
       re-raises with context on failure.
- H-1: `_backward_done` + `requires_grad` preconditions asserted.
- HIGH-2: logger dedup — if user config already includes an
          MLflowLoggerCallback / NullLogger, Trainer reuses it.
- C-4: `cfg.experiment.name` fallback handled in the logger itself.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Iterable

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from cardiac_ml.training.callbacks import Callback
from cardiac_ml.training.default_steps import (
    teacher_forced_step,
    teacher_forced_val_step,
)
from cardiac_ml.training.mlflow_logger import MLflowLoggerCallback
from cardiac_ml.training.callbacks import NullLogger
from cardiac_ml.utils.seed import seed_everything


def _to_device_and_dtype(batch: Any, device: torch.device, dtype: torch.dtype) -> Any:
    """Cast tensor-leaves to `device`; cast floating-point leaves to `dtype`.

    Non-floating tensors (int32 indices, bool masks) keep their dtype.
    Non-tensor leaves pass through unchanged (string keys, pathlib.Path, None).
    Recurses into dict / list / plain-tuple containers. `namedtuple` is NOT
    handled — use plain tuple or dict in custom `collate_fn`s.
    """
    if torch.is_tensor(batch):
        if batch.is_floating_point():
            return batch.to(device=device, dtype=dtype)
        return batch.to(device=device)
    if isinstance(batch, dict):
        return {k: _to_device_and_dtype(v, device, dtype) for k, v in batch.items()}
    if isinstance(batch, list):
        return [_to_device_and_dtype(v, device, dtype) for v in batch]
    if isinstance(batch, tuple):
        # Refuse namedtuples — their constructor doesn't accept a generator.
        if type(batch) is tuple:
            return tuple(_to_device_and_dtype(v, device, dtype) for v in batch)
        return batch  # namedtuple / custom tuple — pass through unchanged
    return batch  # unknown type — pass through


class Trainer:
    """Single training loop. One class, overridable via config, never subclass."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        seed_everything(cfg.training.seed)
        self.dtype = getattr(torch, cfg.training.dtype)
        self.device = torch.device(cfg.training.device)
        self.model = instantiate(cfg.model).to(device=self.device, dtype=self.dtype)
        self.train_loader = instantiate(cfg.data.train) if cfg.data.get("train") else None
        self.val_loader = instantiate(cfg.data.val) if cfg.data.get("val") else None
        self.optimizer = instantiate(
            cfg.training.optimizer, params=list(self.model.parameters())
        )
        self._train_step_fn: Callable[..., dict] = (
            instantiate(cfg.training.train_step_fn)
            if cfg.training.get("train_step_fn")
            else teacher_forced_step
        )
        self._val_step_fn: Callable[..., dict] = (
            instantiate(cfg.training.val_step_fn)
            if cfg.training.get("val_step_fn")
            else teacher_forced_val_step
        )
        self.callbacks: list = [
            instantiate(c) for c in cfg.training.get("callbacks", [])
        ]
        # Round-3 HIGH-2: logger dedup. If user config already includes an
        # MLflow/NullLogger, reuse it instead of appending a second.
        existing = next(
            (cb for cb in self.callbacks
             if isinstance(cb, (MLflowLoggerCallback, NullLogger))),
            None,
        )
        if existing is not None:
            self._logger = existing
        else:
            if cfg.tracking.get("enabled", True):
                self._logger = MLflowLoggerCallback(
                    cfg.tracking.get("experiment_name", "default"),
                    tracking_uri=cfg.tracking.get("tracking_uri", "./mlruns"),
                )
            else:
                self._logger = NullLogger()
            self.callbacks.append(self._logger)

        self.current_epoch: int = 0
        self.should_stop: bool = False

    # ------------------------------------------------------------------ fit
    def fit(self) -> None:
        for cb in self.callbacks:
            cb.on_fit_start(self)
        try:
            for epoch in range(self.cfg.training.epochs):
                if self.should_stop:
                    break
                self.current_epoch = epoch
                for cb in self.callbacks:
                    cb.on_epoch_start(self, epoch)

                train_metrics = (
                    self._run_epoch(train=True) if self.train_loader else {}
                )
                val_metrics = (
                    self._run_epoch(train=False) if self.val_loader else {}
                )
                metrics = {**train_metrics, **val_metrics}
                for cb in self.callbacks:
                    cb.on_epoch_end(self, epoch, metrics)
        finally:
            for cb in self.callbacks:
                cb.on_fit_end(self)

    def _run_epoch(self, train: bool) -> dict:
        step_fn = self._train_step_fn if train else self._val_step_fn
        loader: Iterable = self.train_loader if train else self.val_loader
        self.model.train(train)
        accum: dict = defaultdict(list)

        for batch_idx, batch in enumerate(loader):
            batch = _to_device_and_dtype(batch, self.device, self.dtype)
            out = step_fn(self, batch)
            backward_done = out.get("_backward_done", False)

            if train:
                loss = out["loss"]
                if backward_done:
                    # H-1 / M-9: fresh-detached tensor means weights won't update.
                    assert loss.requires_grad or loss.grad_fn is not None, (
                        "_backward_done=True but loss is not attached to any "
                        "compute graph. Did you return a constant tensor "
                        "instead of the loss used in backward()?"
                    )
                else:
                    assert loss.requires_grad, (
                        "train_step_fn returned loss with requires_grad=False"
                    )
                    loss.backward()
                # M-9: post-hook wrapped — a failing hook clears grads + re-raises.
                post_hook = out.get("_on_after_backward")
                if post_hook is not None:
                    try:
                        post_hook()
                    except Exception as e:
                        self.optimizer.zero_grad(set_to_none=True)
                        raise RuntimeError(
                            f"_on_after_backward hook raised at epoch "
                            f"{self.current_epoch} batch {batch_idx}: "
                            f"{type(e).__name__}: {e}"
                        ) from e
                self.optimizer.step()
                self.optimizer.zero_grad()
            else:
                # Val path: flag misuse of _backward_done.
                assert not backward_done, (
                    "_backward_done=True on val path — flag is meaningless here"
                )
                # Val path still honors the cleanup hook.
                post_hook = out.get("_on_after_backward")
                if post_hook is not None:
                    post_hook()

            # Per-key metric accumulation (skip protocol flags).
            for k, v in out.items():
                if k.startswith("_"):
                    continue
                if torch.is_tensor(v):
                    accum[k].append(float(v.detach().cpu()))
                else:
                    accum[k].append(float(v) if isinstance(v, (int, float)) else 0.0)

            # Per-batch callbacks
            for cb in self.callbacks:
                (cb.on_train_batch_end if train else cb.on_val_batch_end)(
                    self, batch_idx, out
                )

        prefix = "train_" if train else "val_"
        # Default reduction is mean across batches. M-3: reserved suffixes
        # `_sum` / `_last` are NOT implemented yet — flag if seen.
        return {
            f"{prefix}{k}" if not k.startswith(prefix) else k: float(np.mean(v))
            for k, v in accum.items()
        }

    # Escape hatches — dispatch through logger callback (M-2 / M-7).
    def log_artifact(self, path: str) -> None:
        self._logger.log_artifact(path)

    def log_figure(self, fig: Any, name: str) -> None:
        self._logger.log_figure(fig, name)
