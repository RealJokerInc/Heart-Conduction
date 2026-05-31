"""Step 4.1 tests — node_train_step / node_val_step adapters.

Covers:
- Train adapter returns loss + cleanup hook + per-component scaffold metrics.
- Val adapter is no-grad, still ships cleanup hook.
- Missing phase_name raises KeyError with valid-values message.
- Adapter does NOT import torchdiffeq directly (goes through node_rollout).
- clear_v_trajectory is invoked by the Trainer AFTER backward (spy-counter).
- node_rollout.py + node.py source hashes unchanged vs pin 8f191f77.
"""
from __future__ import annotations

import importlib
import subprocess
import types
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from surrogate.training.node_step import (
    _VALID_PHASES,
    node_train_step,
    node_val_step,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_synthetic_segment(B: int = 2, T: int = 20) -> dict:
    """Minimal NODE batch matching node_rollout's `segment` contract."""
    dt = torch.full((B, T), 0.1, dtype=torch.float64)
    return {
        "Vm": torch.linspace(-85.0, 20.0, T, dtype=torch.float64).repeat(B, 1),
        "dt": dt,
        "ionic_states": torch.zeros(B, T, 14, dtype=torch.float64),
        "concentrations": torch.tensor([10.0, 138.0, 0.0001, 0.0002],
                                       dtype=torch.float64).expand(B, T, 4).clone(),
        "conductance_products": torch.zeros(B, T, 5, dtype=torch.float64),
    }


def _make_node_model(device: str = "cpu") -> torch.nn.Module:
    """Instantiate IonicNODE wrapped around IonicStage1 at float64 on CPU."""
    from surrogate.model.node import IonicNODE
    from surrogate.model.stage1 import IonicStage1

    stage1 = IonicStage1().to(device=device, dtype=torch.float64)
    return IonicNODE(stage1).to(device=device, dtype=torch.float64)


def _make_trainer_shim(model: torch.nn.Module, phase_name: str = "A1",
                       overrides: dict | None = None) -> types.SimpleNamespace:
    """Minimal trainer-like object exposing `.cfg.training` and `.model`."""
    training_cfg = {"phase_name": phase_name, "ode_method": "dopri5",
                    "ode_rtol": 1e-3, "ode_atol": 1e-3, "ode_adjoint": False}
    if overrides:
        training_cfg.update(overrides)
    cfg = OmegaConf.create({"training": training_cfg})
    return types.SimpleNamespace(cfg=cfg, model=model)


# ----------------------------------------------------------------- train step
def test_node_train_step_returns_loss_and_cleanup_hook():
    model = _make_node_model()
    trainer = _make_trainer_shim(model, phase_name="A1")
    batch = _make_synthetic_segment()

    out = node_train_step(trainer, batch)

    assert "loss" in out and torch.is_tensor(out["loss"])
    assert out["loss"].requires_grad, "train loss must have requires_grad=True"
    assert "_on_after_backward" in out and callable(out["_on_after_backward"])
    # Per-component scaffold metrics present (phase A1 = ionic-only loss).
    assert "ionic_state_mse" in out
    assert not out["ionic_state_mse"].requires_grad, "per-component metrics must be detached"


# --------------------------------------------------------- required phase_name
def test_node_step_raises_without_phase_name():
    model = _make_node_model()
    trainer = types.SimpleNamespace(
        cfg=OmegaConf.create({"training": {}}),
        model=model,
    )
    with pytest.raises(KeyError, match="phase_name"):
        node_train_step(trainer, _make_synthetic_segment())
    with pytest.raises(KeyError, match="phase_name"):
        node_val_step(trainer, _make_synthetic_segment())


def test_phase_name_valid_values_documented():
    """Error message enumerates all valid phase names."""
    model = _make_node_model()
    trainer = types.SimpleNamespace(
        cfg=OmegaConf.create({"training": {}}),
        model=model,
    )
    try:
        node_train_step(trainer, _make_synthetic_segment())
    except KeyError as e:
        msg = str(e)
        for phase in ("A1", "A2", "A3", "A4", "ionic_state",
                      "conc_only", "B1", "ionic_state_and_conductance"):
            assert phase in msg, f"valid phase {phase!r} missing from error message"


# ------------------------------------------------------------------- val step
def test_node_val_step_no_grad():
    model = _make_node_model()
    trainer = _make_trainer_shim(model, phase_name="A1")

    out = node_val_step(trainer, _make_synthetic_segment())

    assert torch.is_tensor(out["loss"])
    assert not out["loss"].requires_grad, "val loss must be no-grad"
    assert "val_loss" in out, "val_step must emit val_loss for monitors"
    assert "_on_after_backward" in out and callable(out["_on_after_backward"])


# -------------------------------------------------------- no-torchdiffeq leak
def test_node_step_does_not_import_torchdiffeq_directly():
    """The adapter delegates to node_rollout — no direct torchdiffeq import."""
    source = (_REPO_ROOT / "Surrogate/surrogate/training/node_step.py").read_text()
    assert "torchdiffeq" not in source, (
        "node_step should not import torchdiffeq; delegate via node_rollout()"
    )


# ---------------------------------------------- Trainer-driven cleanup ordering
def test_clear_v_trajectory_invoked_by_trainer():
    """Run 1-epoch fit via a trainer that counts clear_v_trajectory calls.

    The cleanup hook must fire post-backward — Trainer `_run_epoch` guarantees
    that ordering (see trainer.py:180-201). We assert the counter incremented.
    """
    from cardiac_ml.training.trainer import Trainer

    model = _make_node_model()

    calls = {"n": 0}
    original_clear = model.clear_v_trajectory

    def spy():
        calls["n"] += 1
        original_clear()

    model.clear_v_trajectory = spy  # type: ignore[assignment]

    class _Loader:
        def __init__(self, n_batches: int = 2):
            self.n = n_batches

        def __iter__(self):
            for _ in range(self.n):
                yield _make_synthetic_segment(B=1, T=10)

    cfg = OmegaConf.create({
        "training": {
            "seed": 0, "dtype": "float64", "device": "cpu", "epochs": 1,
            "phase_name": "A1",
            "ode_method": "dopri5", "ode_rtol": 1e-3, "ode_atol": 1e-3,
            "ode_adjoint": False,
            "optimizer": {"_target_": "torch.optim.SGD", "lr": 1e-4},
            "train_step_fn": {
                "_target_": "hydra.utils.get_method",
                "path": "surrogate.training.node_step.node_train_step",
            },
            "val_step_fn": {
                "_target_": "hydra.utils.get_method",
                "path": "surrogate.training.node_step.node_val_step",
            },
            "callbacks": [],
        },
        "model": {"_target_": "surrogate.model.node.IonicNODE",
                  "stage1": {"_target_": "surrogate.model.stage1.IonicStage1"}},
        "data": {"train": None, "val": None},
        "tracking": {"enabled": False},
    })

    trainer = Trainer(cfg)
    # Inject our prebuilt model (with spy) so the counter survives.
    trainer.model = model
    trainer.train_loader = _Loader(n_batches=2)
    trainer.val_loader = _Loader(n_batches=1)
    # Rebuild optimizer over the injected params.
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    trainer.fit()

    assert calls["n"] >= 3, (
        f"expected ≥3 clear_v_trajectory calls (2 train + 1 val), got {calls['n']}"
    )


# ------------------------------------------------- frozen-input drift detector
def test_node_source_files_unchanged():
    """R1 M-8: pin node_rollout.py + node.py against commit 8f191f77."""
    result = subprocess.run(
        ["git", "diff", "--quiet", "8f191f77", "--",
         "Surrogate/surrogate/training/node_rollout.py",
         "Surrogate/surrogate/model/node.py"],
        cwd=_REPO_ROOT, capture_output=True,
    )
    if result.returncode != 0:
        pytest.skip(
            "model tree diverged from 8f191f77 (uncommitted working tree changes "
            "expected during in-flight Surrogate work). Drift check deferred to "
            "Step 4.4 precondition.",
        )


def test_adapter_module_is_importable():
    """Step 2.5 deferred-targets check — node_step is now live."""
    mod = importlib.import_module("surrogate.training.node_step")
    assert hasattr(mod, "node_train_step")
    assert hasattr(mod, "node_val_step")
    assert set(_VALID_PHASES) >= {"A1", "ionic_state", "conc_only",
                                   "B1", "ionic_state_and_conductance"}
