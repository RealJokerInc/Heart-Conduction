"""Phase configuration for IonicSurrogateV3 training pipeline.

Each phase is a declarative PhaseConfig dataclass. The trainer applies configs
without understanding phase semantics — all logic is in the config.

Phase progression:
  A1     → A1.5   → A2      → A2.5    → A3   → A4  → A5  → B → C → D
  Half 1   Half 1   Half 2    Half 2    All     All   All
  r=1      r=10     r=1       r=10      r=100   r=1K  r=10K

Half 1: attention + ionic_mixing_mlp + ionic_state_decoder (+ concentration)
Half 2: gate_conductance_mlp + gate_conductance_linear + gate_conductance_logit + gate_conductance_decoder

Stage 1 trained in A phases. Stage 2 in B (frozen Stage 1). End-to-end in C.
No encoder. No teacher forcing. All rollouts start from zeros (steady state).
"""

from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Optional

import torch.nn as nn


@dataclass
class PhaseConfig:
    name: str
    trainable_params: list[str]
    loss_fn: str
    data_tiers: list[int]
    batch_size: int
    lr: float
    weight_decay: float
    rollout_length: int
    transition_metric: str
    transition_threshold: Optional[float]
    patience: int
    max_epochs: int


# Half 1 params: attention + ionic MLP + ionic decoder
_HALF1_PARAMS = [
    "stage1.voltage_attention.*",
    "stage1.ionic_mixing_mlp.*",
    "stage1.ionic_mixing_logit",
    "stage1.ionic_state_decoder.*",
]

# Half 2 params: conductance compression + conductance decoder
_HALF2_PARAMS = [
    "stage1.gate_conductance_mlp.*",
    "stage1.gate_conductance_linear.*",
    "stage1.gate_conductance_logit",
    "stage1.gate_conductance_decoder.*",
]

# All Stage 1 params
_ALL_STAGE1 = ["stage1.*"]


PHASE_CONFIGS = {
    # === Half 1: attention + ionic MLP + ionic decoder + concentration ===
    "A1": PhaseConfig(
        name="A1",
        trainable_params=_HALF1_PARAMS,
        loss_fn="ionic_state",
        data_tiers=[1, 12],
        batch_size=32768, lr=5e-4, weight_decay=1e-4,
        rollout_length=1,
        transition_metric="val_ionic_state_mse", transition_threshold=None,
        patience=10, max_epochs=30,
    ),
    "A1.5": PhaseConfig(
        name="A1.5",
        trainable_params=_HALF1_PARAMS,
        loss_fn="ionic_state",
        data_tiers=[1, 12],
        batch_size=32768, lr=1e-4, weight_decay=1e-4,
        rollout_length=10,
        transition_metric="val_ionic_state_mse", transition_threshold=None,
        patience=10, max_epochs=500,
    ),

    # === Half 2: conductance compression + decoder (Half 1 frozen) ===
    "A2": PhaseConfig(
        name="A2",
        trainable_params=_HALF2_PARAMS,
        loss_fn="ionic_state_and_conductance",
        data_tiers=[1, 12],
        batch_size=32768, lr=1e-3, weight_decay=1e-4,
        rollout_length=1,
        transition_metric="val_ionic_state_mse", transition_threshold=None,
        patience=10, max_epochs=30,
    ),
    "A2.5": PhaseConfig(
        name="A2.5",
        trainable_params=_HALF2_PARAMS,
        loss_fn="ionic_state_and_conductance",
        data_tiers=[1, 12],
        batch_size=32768, lr=1e-4, weight_decay=1e-4,
        rollout_length=10,
        transition_metric="val_ionic_state_mse", transition_threshold=None,
        patience=10, max_epochs=500,
    ),

    # === All Stage 1 params, rollout curriculum ===
    "A3": PhaseConfig(
        name="A3",
        trainable_params=_ALL_STAGE1,
        loss_fn="ionic_state_and_conductance",
        data_tiers=[1, 2, 12],
        batch_size=16384, lr=5e-4, weight_decay=5e-4,
        rollout_length=100,
        transition_metric="val_ionic_state_mse", transition_threshold=None,
        patience=15, max_epochs=50,
    ),
    "A4": PhaseConfig(
        name="A4",
        trainable_params=_ALL_STAGE1,
        loss_fn="ionic_state_and_conductance",
        data_tiers=[1, 2, 3, 12],
        batch_size=8192, lr=3e-4, weight_decay=5e-4,
        rollout_length=1000,
        transition_metric="val_ionic_state_mse", transition_threshold=None,
        patience=15, max_epochs=50,
    ),
    "A5": PhaseConfig(
        name="A5",
        trainable_params=_ALL_STAGE1,
        loss_fn="ionic_state_and_conductance",
        data_tiers=[1, 2, 3, 12],
        batch_size=4096, lr=2e-4, weight_decay=5e-4,
        rollout_length=10000,
        transition_metric="val_ionic_state_mse", transition_threshold=None,
        patience=15, max_epochs=50,
    ),

    # === Stage 2 + end-to-end ===
    "B": PhaseConfig(
        name="B",
        trainable_params=["stage2.*"],
        loss_fn="I_ion",
        data_tiers=[1, 2, 3, 12],
        batch_size=4096, lr=1e-3, weight_decay=1e-3,
        rollout_length=10000,
        transition_metric="val_I_ion_mse", transition_threshold=1e-3,
        patience=20, max_epochs=300,
    ),
    "C": PhaseConfig(
        name="C",
        trainable_params=_ALL_STAGE1 + ["stage2.*"],
        loss_fn="I_ion",
        data_tiers=[1, 2, 3, 12],
        batch_size=2048, lr=5e-5, weight_decay=1e-3,
        rollout_length=10000,
        transition_metric="val_I_ion_mse", transition_threshold=None,
        patience=20, max_epochs=300,
    ),
}

PHASE_ORDER = ["A1", "A1.5", "A2", "A2.5", "A3", "A4", "A5", "B", "C"]


def get_phase_config(phase_name: str) -> PhaseConfig:
    if phase_name not in PHASE_CONFIGS:
        raise ValueError(f"Unknown phase: {phase_name}. Valid: {PHASE_ORDER}")
    return PHASE_CONFIGS[phase_name]


def get_all_phases() -> list[PhaseConfig]:
    return [PHASE_CONFIGS[name] for name in PHASE_ORDER]


def apply_freeze_mask(model: nn.Module, phase: PhaseConfig) -> None:
    """Freeze all model params, then unfreeze those matching phase patterns."""
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        for pattern in phase.trainable_params:
            if fnmatch(name, pattern):
                param.requires_grad = True
                break


def get_freeze_summary(model: nn.Module) -> dict[str, bool]:
    """Return {param_name: requires_grad} for all model params."""
    return {name: param.requires_grad for name, param in model.named_parameters()}
