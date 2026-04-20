"""Trainer integration tests against a synthetic MLP + linear dataset.

Exercises: basic fit convergence, protocol-flag assertions (_backward_done
+ requires_grad preconditions), dtype preservation (no global mutation),
escape-hatch dispatch, and logger dedup.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader

from cardiac_ml.training.callbacks import Callback, NullLogger
from cardiac_ml.training.mlflow_logger import MLflowLoggerCallback
from cardiac_ml.training.trainer import Trainer, _to_device_and_dtype


# ----------------------------- fixtures ------------------------------------


class _LinearDataset(Dataset):
    """y = x @ W + b. param_seed fixes (W, b); sample_seed varies X samples.

    Train and val share param_seed so they have the SAME underlying target
    function, but different sample_seeds so they see different X points.
    """
    def __init__(self, n: int = 128, sample_seed: int = 0, param_seed: int = 999):
        gp = torch.Generator().manual_seed(param_seed)
        gs = torch.Generator().manual_seed(sample_seed)
        self.W = torch.randn(4, 4, generator=gp, dtype=torch.float64)
        self.b = torch.randn(4, generator=gp, dtype=torch.float64)
        self.X = torch.randn(n, 4, generator=gs, dtype=torch.float64)
        self.Y = self.X @ self.W + self.b

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        return self.X[i], self.Y[i]


def _linear_dataset_factory(n: int = 128, sample_seed: int = 0, param_seed: int = 999) -> Dataset:
    """Factory for Hydra _target_ instantiation (classes on module level)."""
    return _LinearDataset(n=n, sample_seed=sample_seed, param_seed=param_seed)


def _dataloader_factory(dataset, batch_size: int = 16) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


class _TinyMLP(torch.nn.Module):
    """Pure linear layer — sufficient to fit the synthetic y = Wx + b target
    and prove the Trainer can actually drive convergence. A 2-layer MLP+GELU
    would converge more slowly than the 20-epoch budget here; we're testing
    the harness, not the approximator."""

    def __init__(self, hidden: int = 8):  # hidden kept for API compat (unused)
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)


def _make_cfg(tracking_uri: str) -> OmegaConf:
    """Minimal Hydra cfg for Trainer.__init__. Float64 + CPU for test portability."""
    return OmegaConf.create({
        "model": {"_target_": f"{__name__}._TinyMLP", "hidden": 8},
        "data": {
            "train": {
                "_target_": f"{__name__}._dataloader_factory",
                "dataset": {
                    "_target_": f"{__name__}._linear_dataset_factory",
                    "n": 128,
                    "sample_seed": 0,
                    "param_seed": 999,
                },
                "batch_size": 16,
            },
            "val": {
                "_target_": f"{__name__}._dataloader_factory",
                "dataset": {
                    "_target_": f"{__name__}._linear_dataset_factory",
                    "n": 64,
                    "sample_seed": 1,
                    "param_seed": 999,
                },
                "batch_size": 16,
            },
        },
        "training": {
            "epochs": 50,
            "seed": 42,
            "device": "cpu",
            "dtype": "float64",
            "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.1},
            "callbacks": [],
        },
        "tracking": {
            "enabled": True,
            "experiment_name": "trainer_tests",
            "tracking_uri": tracking_uri,
        },
        "experiment": "trainer_synthetic",
    })


# ----------------------------- _to_device_and_dtype ---------------------------


def test_to_device_and_dtype_float_only_promotion():
    """Floating tensors promoted; int tensors keep dtype."""
    batch = {
        "x": torch.randn(2, 2, dtype=torch.float32),
        "idx": torch.tensor([0, 1], dtype=torch.int32),
    }
    out = _to_device_and_dtype(batch, torch.device("cpu"), torch.float64)
    assert out["x"].dtype == torch.float64
    assert out["idx"].dtype == torch.int32


def test_to_device_and_dtype_pass_through_non_tensors():
    batch = {"x": torch.randn(2, 2), "meta": "hello", "idx": None}
    out = _to_device_and_dtype(batch, torch.device("cpu"), torch.float64)
    assert out["meta"] == "hello"
    assert out["idx"] is None


def test_to_device_and_dtype_list_and_tuple():
    batch = [torch.randn(2), (torch.randn(2), "meta")]
    out = _to_device_and_dtype(batch, torch.device("cpu"), torch.float64)
    assert isinstance(out, list)
    assert isinstance(out[1], tuple)
    assert out[0].dtype == torch.float64
    assert out[1][1] == "meta"


# ----------------------------- fit convergence --------------------------------


def test_fit_converges_on_linear_task(tmp_path):
    """50 epochs of Adam lr=0.1 on a 128-sample y=Wx+b task should get
    val_loss below 0.01 easily (standalone baseline hits ~0.0 in 50 epochs)."""
    cfg = _make_cfg(tracking_uri=f"file:{tmp_path}")
    trainer = Trainer(cfg)
    trainer.fit()
    val_metrics = trainer._run_epoch(train=False)
    assert val_metrics["val_loss"] < 0.01, f"val_loss={val_metrics['val_loss']}"


# ----------------------------- dtype invariant --------------------------------


def test_model_dtype_is_float64_without_global_mutation(tmp_path):
    """Trainer must NOT mutate torch.get_default_dtype (Round-3 M-1)."""
    before = torch.get_default_dtype()
    cfg = _make_cfg(tracking_uri=f"file:{tmp_path}")
    # Instantiate but don't fit
    trainer = Trainer(cfg)
    assert torch.get_default_dtype() == before, "global default_dtype was mutated"
    p = next(trainer.model.parameters())
    assert p.dtype == torch.float64


# ----------------------------- protocol assertions ----------------------------


def _zero_grad_train_step(trainer: Any, batch: Any) -> dict:
    """Returns a detached loss with _backward_done=True — should raise."""
    x, y = batch
    pred = trainer.model(x)
    loss = torch.nn.functional.mse_loss(pred, y)
    return {"loss": loss.detach(), "_backward_done": True}


def test_backward_done_with_detached_loss_raises(tmp_path):
    """Round-3 M-9 assertion: fresh-detached tensor trips AssertionError."""
    cfg = _make_cfg(tracking_uri=f"file:{tmp_path}")
    cfg.training.train_step_fn = None  # use custom below
    trainer = Trainer(cfg)
    trainer._train_step_fn = _zero_grad_train_step
    with pytest.raises(AssertionError, match="not attached to any compute graph"):
        trainer._run_epoch(train=True)


def _val_with_backward_done(trainer, batch):
    x, y = batch
    pred = trainer.model(x)
    loss = torch.nn.functional.mse_loss(pred, y)
    return {"loss": loss, "_backward_done": True}


def test_backward_done_on_val_path_raises(tmp_path):
    """_backward_done on val path is meaningless — Trainer asserts against it."""
    cfg = _make_cfg(tracking_uri=f"file:{tmp_path}")
    trainer = Trainer(cfg)
    trainer._val_step_fn = _val_with_backward_done
    with pytest.raises(AssertionError, match="meaningless"):
        trainer._run_epoch(train=False)


def _no_grad_train_step(trainer, batch):
    x, y = batch
    with torch.no_grad():
        pred = trainer.model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
    return {"loss": loss}


def test_loss_requires_grad_on_train_path(tmp_path):
    """Without _backward_done, loss MUST have requires_grad=True."""
    cfg = _make_cfg(tracking_uri=f"file:{tmp_path}")
    trainer = Trainer(cfg)
    trainer._train_step_fn = _no_grad_train_step
    with pytest.raises(AssertionError, match="requires_grad=False"):
        trainer._run_epoch(train=True)


# ----------------------------- _on_after_backward -----------------------------


_hook_calls = {"count": 0}


def _hook_train_step(trainer, batch):
    x, y = batch
    pred = trainer.model(x)
    loss = torch.nn.functional.mse_loss(pred, y)
    def _hook():
        _hook_calls["count"] += 1
    return {"loss": loss, "_on_after_backward": _hook}


def test_on_after_backward_invoked_train(tmp_path):
    """_on_after_backward fires after backward on train path."""
    _hook_calls["count"] = 0
    cfg = _make_cfg(tracking_uri=f"file:{tmp_path}")
    cfg.training.epochs = 1
    trainer = Trainer(cfg)
    trainer._train_step_fn = _hook_train_step
    trainer._run_epoch(train=True)
    # n=128, batch_size=16 → 8 batches per epoch
    assert _hook_calls["count"] == 8


def _hook_raise_step(trainer, batch):
    x, y = batch
    pred = trainer.model(x)
    loss = torch.nn.functional.mse_loss(pred, y)
    def _bad_hook():
        raise ValueError("synthetic failure")
    return {"loss": loss, "_on_after_backward": _bad_hook}


def test_on_after_backward_failure_zero_grads_and_reraises(tmp_path):
    """M-9: hook failure clears grads + raises RuntimeError with context."""
    cfg = _make_cfg(tracking_uri=f"file:{tmp_path}")
    cfg.training.epochs = 1
    trainer = Trainer(cfg)
    trainer._train_step_fn = _hook_raise_step
    with pytest.raises(RuntimeError, match="_on_after_backward hook raised"):
        trainer._run_epoch(train=True)
    # After the raise, gradients should be None (zero_grad(set_to_none=True)).
    for p in trainer.model.parameters():
        assert p.grad is None


# ----------------------------- escape hatches ---------------------------------


def test_log_artifact_routes_through_logger(tmp_path):
    """Trainer.log_artifact calls self._logger.log_artifact — no direct mlflow."""
    cfg = _make_cfg(tracking_uri=f"file:{tmp_path}")
    trainer = Trainer(cfg)

    seen: list = []
    trainer._logger = SimpleNamespace(
        log_artifact=lambda p: seen.append(p),
        log_figure=lambda f, n: None,
    )
    trainer.log_artifact("some/path.pt")
    assert seen == ["some/path.pt"]


def test_log_artifact_with_null_logger_no_op(tmp_path):
    """tracking=false → NullLogger → no filesystem side effects."""
    cfg = _make_cfg(tracking_uri=f"file:{tmp_path}")
    cfg.tracking.enabled = False
    trainer = Trainer(cfg)
    # No exception, no files written.
    trainer.log_artifact("nonexistent.pt")
    trainer.log_figure(None, "nonexistent.png")
    assert isinstance(trainer._logger, NullLogger)


# ----------------------------- logger dedup -----------------------------------


def test_logger_dedup_reuses_user_supplied(tmp_path):
    """If user config already contains a NullLogger, Trainer reuses it."""
    cfg = _make_cfg(tracking_uri=f"file:{tmp_path}")
    cfg.training.callbacks = [
        {"_target_": "cardiac_ml.training.callbacks.NullLogger"}
    ]
    cfg.tracking.enabled = True  # would normally install MLflowLoggerCallback
    trainer = Trainer(cfg)
    # Despite tracking.enabled=True, Trainer found the user's NullLogger
    # and reused it — no MLflowLoggerCallback appended.
    loggers = [
        cb for cb in trainer.callbacks
        if isinstance(cb, (NullLogger, MLflowLoggerCallback))
    ]
    assert len(loggers) == 1
    assert isinstance(loggers[0], NullLogger)
    assert trainer._logger is loggers[0]
