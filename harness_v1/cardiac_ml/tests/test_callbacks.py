"""Tests for cardiac_ml/training/callbacks.py."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from cardiac_ml.training.callbacks import (
    Callback,
    EarlyStopping,
    GradNormMonitor,
    LRSchedulerStep,
    ModelCheckpoint,
    NullLogger,
)


def _stub_trainer(tmp_path: Path) -> SimpleNamespace:
    """Minimal trainer stub with model + logged-artifacts list."""
    model = nn.Linear(4, 4).double()
    logged = []
    trainer = SimpleNamespace(
        model=model,
        should_stop=False,
        log_artifact=lambda p: logged.append(str(p)),
        _workdir=tmp_path,
    )
    trainer.logged = logged
    return trainer


# ----------------------------- EarlyStopping --------------------------------


def test_early_stopping_fires_after_patience_exceeded():
    es = EarlyStopping(monitor="val_loss", patience=2, mode="min")
    trainer = SimpleNamespace(should_stop=False)
    # Improving
    es.on_epoch_end(trainer, 0, {"val_loss": 1.0})
    assert not trainer.should_stop
    # Getting worse: epoch 1 bad=1, epoch 2 bad=2 (== patience), epoch 3 bad=3 → stop
    es.on_epoch_end(trainer, 1, {"val_loss": 1.1})
    es.on_epoch_end(trainer, 2, {"val_loss": 1.2})
    assert not trainer.should_stop
    es.on_epoch_end(trainer, 3, {"val_loss": 1.3})
    assert trainer.should_stop


def test_early_stopping_ignores_missing_metric():
    es = EarlyStopping(monitor="val_loss", patience=1)
    trainer = SimpleNamespace(should_stop=False)
    # Metric dict has no val_loss — must not count as a bad epoch.
    for _ in range(5):
        es.on_epoch_end(trainer, 0, {"other_metric": 1.0})
    assert not trainer.should_stop


def test_early_stopping_mode_max():
    es = EarlyStopping(monitor="acc", patience=1, mode="max")
    trainer = SimpleNamespace(should_stop=False)
    es.on_epoch_end(trainer, 0, {"acc": 0.8})  # improving (first)
    es.on_epoch_end(trainer, 1, {"acc": 0.75})  # bad 1
    es.on_epoch_end(trainer, 2, {"acc": 0.7})  # bad 2 > patience
    assert trainer.should_stop


# ----------------------------- ModelCheckpoint ------------------------------


def test_model_checkpoint_saves_best_on_improvement(tmp_path, monkeypatch):
    """best.pt is written (and log_artifact called) only when monitor improves."""
    # Force checkpoint to use tmp_path as the workdir.
    monkeypatch.setattr(ModelCheckpoint, "_workdir", staticmethod(lambda: tmp_path))

    mc = ModelCheckpoint(monitor="val_loss", mode="min", every_n_epochs=0, save_last=False)
    trainer = _stub_trainer(tmp_path)
    mc.on_epoch_end(trainer, 0, {"val_loss": 1.0})  # improvement (from inf)
    mc.on_epoch_end(trainer, 1, {"val_loss": 0.5})  # improvement
    mc.on_epoch_end(trainer, 2, {"val_loss": 0.6})  # no improvement
    # Exactly 2 best.pt saves.
    best_logs = [p for p in trainer.logged if p.endswith("best.pt")]
    assert len(best_logs) == 2
    assert (tmp_path / "best.pt").exists()


def test_model_checkpoint_periodic_and_last(tmp_path, monkeypatch):
    monkeypatch.setattr(ModelCheckpoint, "_workdir", staticmethod(lambda: tmp_path))
    mc = ModelCheckpoint(monitor="val_loss", every_n_epochs=2, save_last=True)
    trainer = _stub_trainer(tmp_path)
    for ep in range(4):
        mc.on_epoch_end(trainer, ep, {"val_loss": 1.0 - ep * 0.1})
    # every_n_epochs=2 triggers at epoch 1 (epoch+1=2) and epoch 3 (epoch+1=4).
    periodics = [p for p in trainer.logged if "epoch_" in p]
    assert len(periodics) == 2
    # save_last=True writes last.pt every epoch end.
    last_logs = [p for p in trainer.logged if p.endswith("last.pt")]
    assert len(last_logs) == 4


def test_model_checkpoint_on_fit_end_writes_last(tmp_path, monkeypatch):
    monkeypatch.setattr(ModelCheckpoint, "_workdir", staticmethod(lambda: tmp_path))
    mc = ModelCheckpoint(save_last=True)
    trainer = _stub_trainer(tmp_path)
    mc.on_fit_end(trainer)
    assert any(p.endswith("last.pt") for p in trainer.logged)


# ----------------------------- GradNormMonitor ------------------------------


def test_grad_norm_logged():
    model = nn.Linear(4, 4).double()
    # Run a forward/backward so grads exist.
    x = torch.randn(8, 4, dtype=torch.float64)
    y = torch.randn(8, 4, dtype=torch.float64)
    pred = model(x)
    loss = ((pred - y) ** 2).mean()
    loss.backward()
    trainer = SimpleNamespace(model=model)
    gn = GradNormMonitor()
    out = {}
    gn.on_train_batch_end(trainer, 0, out)
    assert "grad_norm" in out
    assert out["grad_norm"] > 0.0


def test_grad_norm_handles_zero_grads():
    """With no backward pass, grad=None — monitor reports 0.0 without crash."""
    model = nn.Linear(4, 4).double()
    trainer = SimpleNamespace(model=model)
    gn = GradNormMonitor()
    out = {}
    gn.on_train_batch_end(trainer, 0, out)
    assert out["grad_norm"] == 0.0


# ----------------------------- LRSchedulerStep ------------------------------


def test_lr_scheduler_noop_when_target_none():
    cb = LRSchedulerStep(scheduler_target=None)
    trainer = SimpleNamespace(optimizer=None)
    cb.on_fit_start(trainer)  # must not crash
    cb.on_epoch_end(trainer, 0, {})


# ----------------------------- Callback base + NullLogger ---------------------


def test_base_callback_log_methods_are_noops():
    cb = Callback()
    cb.log_artifact("nonexistent.pt")  # must not raise
    cb.log_figure(None, "nonexistent.png")


def test_null_logger_inherits_no_op_proxies():
    cb = NullLogger()
    cb.log_artifact("nonexistent.pt")
    cb.log_figure(None, "nonexistent.png")
    # Also a Callback
    assert isinstance(cb, Callback)
