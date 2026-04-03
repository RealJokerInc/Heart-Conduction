"""Phase configuration for IonicSurrogateV3 training pipeline.

Each phase is a declarative PhaseConfig dataclass. The trainer applies configs
without understanding phase semantics — all logic is in the config.

Phase progression: A2 -> B1 -> B2 -> B3 -> B4 -> B5 -> C -> D -> E
"""

from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Optional

import torch.nn as nn


@dataclass
class PhaseConfig:
    name: str
    trainable_params: list[str]    # fnmatch patterns for unfrozen params
    loss_fn: str                   # "concentration", "ionic_state", "ionic_state_and_conductance", "I_ion"
    data_tiers: list[int]
    batch_size: int
    lr: float
    weight_decay: float
    rollout_length: int
    transition_metric: str
    transition_threshold: Optional[float]  # None = no threshold, rely on patience
    patience: int
    max_epochs: int


PHASE_CONFIGS = {
    "A2": PhaseConfig(
        name="A2",
        trainable_params=["stage1.voltage_attention.*"],
        loss_fn="concentration",
        data_tiers=[1],
        batch_size=2048, lr=1e-3, weight_decay=1e-4,
        rollout_length=1,
        transition_metric="val_conc_mse", transition_threshold=1e-6,
        patience=10, max_epochs=100,
    ),
    "B1": PhaseConfig(
        name="B1",
        trainable_params=[
            "stage1.voltage_attention.*",
            "stage1.ionic_mixing_mlp.*",
            "stage1.ionic_mixing_logit",
            "stage1.ionic_state_decoder.*",
        ],
        loss_fn="ionic_state",
        data_tiers=[1, 12],
        batch_size=1024, lr=5e-4, weight_decay=1e-4,
        rollout_length=1,
        transition_metric="val_ionic_state_mse", transition_threshold=None,
        patience=15, max_epochs=200,
    ),
    "B2": PhaseConfig(
        name="B2",
        trainable_params=[
            "stage1.voltage_attention.*",
            "stage1.ionic_mixing_mlp.*",
            "stage1.ionic_mixing_logit",
            "stage1.ionic_state_decoder.*",
        ],
        loss_fn="ionic_state",
        data_tiers=[1, 12],
        batch_size=512, lr=5e-4, weight_decay=1e-4,
        rollout_length=10,
        transition_metric="val_ionic_state_mse", transition_threshold=None,
        patience=15, max_epochs=200,
    ),
    "B3": PhaseConfig(
        name="B3",
        trainable_params=[
            "stage1.voltage_attention.*",
            "stage1.ionic_mixing_mlp.*",
            "stage1.ionic_mixing_logit",
            "stage1.ionic_state_decoder.*",
            "stage1.gate_conductance_mlp.*",
            "stage1.gate_conductance_linear.*",
            "stage1.gate_conductance_logit",
            "stage1.gate_conductance_decoder.*",
        ],
        loss_fn="ionic_state_and_conductance",
        data_tiers=[1, 2, 12],
        batch_size=256, lr=5e-4, weight_decay=5e-4,
        rollout_length=100,
        transition_metric="val_ionic_state_mse", transition_threshold=None,
        patience=15, max_epochs=200,
    ),
    "B4": PhaseConfig(
        name="B4",
        trainable_params=[
            "stage1.voltage_attention.*",
            "stage1.ionic_mixing_mlp.*",
            "stage1.ionic_mixing_logit",
            "stage1.ionic_state_decoder.*",
            "stage1.gate_conductance_mlp.*",
            "stage1.gate_conductance_linear.*",
            "stage1.gate_conductance_logit",
            "stage1.gate_conductance_decoder.*",
        ],
        loss_fn="ionic_state_and_conductance",
        data_tiers=[1, 2, 3, 12],
        batch_size=128, lr=3e-4, weight_decay=5e-4,
        rollout_length=1000,
        transition_metric="val_ionic_state_mse", transition_threshold=None,
        patience=15, max_epochs=200,
    ),
    "B5": PhaseConfig(
        name="B5",
        trainable_params=[
            "stage1.voltage_attention.*",
            "stage1.ionic_mixing_mlp.*",
            "stage1.ionic_mixing_logit",
            "stage1.ionic_state_decoder.*",
            "stage1.gate_conductance_mlp.*",
            "stage1.gate_conductance_linear.*",
            "stage1.gate_conductance_logit",
            "stage1.gate_conductance_decoder.*",
        ],
        loss_fn="ionic_state_and_conductance",
        data_tiers=[1, 2, 3, 12],
        batch_size=64, lr=2e-4, weight_decay=5e-4,
        rollout_length=10000,
        transition_metric="val_ionic_state_mse", transition_threshold=None,
        patience=15, max_epochs=200,
    ),
    "C": PhaseConfig(
        name="C",
        trainable_params=["stage1.*"],
        loss_fn="concentration_rollout",
        data_tiers=[1, 2, 3, 12],
        batch_size=64, lr=1e-4, weight_decay=5e-4,
        rollout_length=10000,
        transition_metric="val_conc_mse", transition_threshold=1e-5,
        patience=20, max_epochs=300,
    ),
    "D": PhaseConfig(
        name="D",
        trainable_params=["stage2.*"],
        loss_fn="I_ion",
        data_tiers=[1, 2, 3, 12],  # T4 added via shard streaming when available
        batch_size=64, lr=1e-3, weight_decay=1e-3,
        rollout_length=10000,
        transition_metric="val_I_ion_mse", transition_threshold=1e-3,
        patience=20, max_epochs=300,
    ),
    "E": PhaseConfig(
        name="E",
        trainable_params=["stage1.*", "stage2.*"],
        loss_fn="I_ion",
        data_tiers=[1, 2, 3, 12],  # T4 added via shard streaming when available
        batch_size=32, lr=5e-5, weight_decay=1e-3,
        rollout_length=10000,
        transition_metric="val_I_ion_mse", transition_threshold=None,
        patience=20, max_epochs=300,
    ),
}

PHASE_ORDER = ["A2", "B1", "B2", "B3", "B4", "B5", "C", "D", "E"]


def get_phase_config(phase_name: str) -> PhaseConfig:
    if phase_name not in PHASE_CONFIGS:
        raise ValueError(f"Unknown phase: {phase_name}. Valid: {PHASE_ORDER}")
    return PHASE_CONFIGS[phase_name]


def get_all_phases() -> list[PhaseConfig]:
    return [PHASE_CONFIGS[name] for name in PHASE_ORDER]


def apply_freeze_mask(model: nn.Module, phase: PhaseConfig) -> None:
    """Freeze all model params, then unfreeze those matching phase patterns."""
    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze matching patterns
    for name, param in model.named_parameters():
        for pattern in phase.trainable_params:
            if fnmatch(name, pattern):
                param.requires_grad = True
                break


def get_freeze_summary(model: nn.Module) -> dict[str, bool]:
    """Return {param_name: requires_grad} for all model params."""
    return {name: param.requires_grad for name, param in model.named_parameters()}
