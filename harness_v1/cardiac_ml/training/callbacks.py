"""Callback base class + core callbacks for cardiac_ml.Trainer.

Callbacks isolate logger, checkpointing, and early-stopping concerns from
the Trainer core. Adding a new callback = new subclass, no Trainer changes.

The base class defines 6 lifecycle hooks (all no-ops by default) PLUS
`log_artifact` and `log_figure` proxy methods (Round-2 C-3): Trainer
dispatches escape-hatch calls (`trainer.log_artifact(path)`) through its
registered logger callback, so Trainer itself never imports mlflow.
`NullLogger` inherits the base no-ops; `MLflowLoggerCallback` overrides.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import torch


class Callback:
    """Base class for all cardiac_ml callbacks.

    All six lifecycle hooks are no-ops by default — subclasses override
    only the ones they care about. `log_artifact` and `log_figure` are
    proxy methods (dispatched by Trainer.log_artifact / Trainer.log_figure);
    MLflowLoggerCallback overrides both, NullLogger inherits the no-op.
    """

    # Lifecycle hooks --------------------------------------------------------
    def on_fit_start(self, trainer: Any) -> None: ...
    def on_epoch_start(self, trainer: Any, epoch: int) -> None: ...
    def on_train_batch_end(self, trainer: Any, batch_idx: int, outputs: dict) -> None: ...
    def on_val_batch_end(self, trainer: Any, batch_idx: int, outputs: dict) -> None: ...
    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict) -> None: ...
    def on_fit_end(self, trainer: Any) -> None: ...

    # Proxy methods (override in MLflowLoggerCallback) -----------------------
    def log_artifact(self, path: str) -> None: ...
    def log_figure(self, fig: Any, name: str) -> None: ...


class EarlyStopping(Callback):
    """Stop training if `monitor` hasn't improved for `patience` epochs.

    Sets `trainer.should_stop = True` to request a clean shutdown at the
    next epoch boundary. Trainer.fit() checks this flag each epoch.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        mode: str = "min",
    ):
        assert mode in ("min", "max"), f"mode must be 'min' or 'max', got {mode!r}"
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self._best: float = math.inf if mode == "min" else -math.inf
        self._bad_epochs: int = 0

    def _improved(self, value: float) -> bool:
        return (self.mode == "min" and value < self._best) or (
            self.mode == "max" and value > self._best
        )

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict) -> None:
        if self.monitor not in metrics:
            return  # metric not reported this epoch — don't penalize
        value = float(metrics[self.monitor])
        if self._improved(value):
            self._best = value
            self._bad_epochs = 0
        else:
            self._bad_epochs += 1
            if self._bad_epochs > self.patience:
                trainer.should_stop = True


class ModelCheckpoint(Callback):
    """Write state-dict checkpoints to the Hydra working dir and MLflow.

    - `best.pt`: written whenever `monitor` improves (always).
    - `last.pt`: written at the end of every epoch AND `on_fit_end` so a
      final checkpoint exists regardless of early-stop.
    - `epoch_{N}.pt`: written every `every_n_epochs` epochs.

    All three go through `trainer.log_artifact(...)` so MLflow picks them
    up (or NullLogger no-ops if tracking=off).
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        every_n_epochs: int = 50,
        save_last: bool = True,
    ):
        assert mode in ("min", "max"), f"mode must be 'min' or 'max', got {mode!r}"
        self.monitor = monitor
        self.mode = mode
        self.every_n_epochs = every_n_epochs
        self.save_last = save_last
        self._best: float = math.inf if mode == "min" else -math.inf

    @staticmethod
    def _workdir() -> Path:
        """Hydra working dir if under @hydra.main, else cwd."""
        try:
            from hydra.core.hydra_config import HydraConfig
            return Path(HydraConfig.get().runtime.output_dir)
        except Exception:
            return Path.cwd()

    def _save_and_log(self, trainer: Any, filename: str) -> None:
        path = self._workdir() / filename
        torch.save(trainer.model.state_dict(), path)
        trainer.log_artifact(str(path))

    def _improved(self, value: float) -> bool:
        return (self.mode == "min" and value < self._best) or (
            self.mode == "max" and value > self._best
        )

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict) -> None:
        # Best
        if self.monitor in metrics:
            v = float(metrics[self.monitor])
            if self._improved(v):
                self._best = v
                self._save_and_log(trainer, "best.pt")
        # Periodic
        if self.every_n_epochs and (epoch + 1) % self.every_n_epochs == 0:
            self._save_and_log(trainer, f"epoch_{epoch + 1}.pt")
        # Last (every epoch)
        if self.save_last:
            self._save_and_log(trainer, "last.pt")

    def on_fit_end(self, trainer: Any) -> None:
        # Guarantee a final `last.pt` even if early-stop fired before any
        # on_epoch_end ran with save_last enabled.
        if self.save_last:
            self._save_and_log(trainer, "last.pt")


class GradNormMonitor(Callback):
    """Log the L2 norm of model gradients after each train batch.

    Stored directly on the batch `outputs` dict so the Trainer's default
    per-key metric accumulation surfaces it as `train_grad_norm`.
    """

    def on_train_batch_end(self, trainer: Any, batch_idx: int, outputs: dict) -> None:
        total = 0.0
        for p in trainer.model.parameters():
            if p.grad is not None:
                total += float(p.grad.detach().pow(2).sum().item())
        outputs["grad_norm"] = math.sqrt(total)


class LRSchedulerStep(Callback):
    """Optional LR scheduler that steps after each epoch.

    Resolves the scheduler via Hydra `_target_` on `on_fit_start` (late-
    bound so the optimizer exists). If `scheduler_target` is None, the
    callback is a no-op.
    """

    def __init__(self, scheduler_target: Optional[Any] = None):
        self.scheduler_target = scheduler_target
        self._scheduler: Any = None

    def on_fit_start(self, trainer: Any) -> None:
        if self.scheduler_target is None:
            return
        # Resolve via Hydra if spec-like dict; else assume it's a partial.
        try:
            from hydra.utils import instantiate
            self._scheduler = instantiate(self.scheduler_target, optimizer=trainer.optimizer)
        except Exception:
            self._scheduler = None

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict) -> None:
        if self._scheduler is not None:
            self._scheduler.step()


class NullLogger(Callback):
    """All methods inherit Callback base no-ops. Used when tracking.enabled=false
    so Trainer can dispatch uniformly without checking tracking state.

    Round-2 C-3 / Round-3 C-4 fix: intentionally empty subclass — `log_artifact`
    and `log_figure` are already no-ops on the base class.
    """
    pass
