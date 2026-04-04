"""Phase configuration for IonicSurrogateV3 training pipeline.

dt curriculum: fix temporal coverage at ~300ms (one full AP), vary dt.
Subsample existing T1 data (dt=0.01ms) by stride N → effective dt = N*0.01ms.

A phases: Half 1 (attention + ionic MLP + ionic decoder + concentration)
B phases: Half 2 (conductance compression + decoder), Half 1 frozen
C: Stage 2 (frozen Stage 1)
D: End-to-end
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
    subsample: int              # take every Nth timestep, effective dt = N * 0.01ms
    transition_metric: str
    transition_threshold: Optional[float]
    patience: int
    max_epochs: int
    tbptt_window: int = 0           # truncated BPTT: backprop only last N steps (0 = all)


# Half 1 params
_HALF1_PARAMS = [
    "stage1.voltage_attention.*",
    "stage1.ionic_mixing_mlp.*",
    "stage1.ionic_mixing_logit",
    "stage1.ionic_state_decoder.*",
]

# Half 2 params
_HALF2_PARAMS = [
    "stage1.gate_conductance_mlp.*",
    "stage1.gate_conductance_linear.*",
    "stage1.gate_conductance_logit",
    "stage1.gate_conductance_decoder.*",
]

_ALL_STAGE1 = ["stage1.*"]


PHASE_CONFIGS = {
    # === A: Half 1, dt curriculum ===
    # dt=3.0ms (subsample=300), rollout=100, covers 300ms
    "A1": PhaseConfig(
        name="A1",
        trainable_params=_HALF1_PARAMS,
        loss_fn="ionic_state",
        data_tiers=[1],
        batch_size=128, lr=5e-4, weight_decay=1e-4,
        rollout_length=100, subsample=300,
        transition_metric="val_ionic_state_mse", transition_threshold=None,
        patience=50, max_epochs=200,
    ),
    # dt=1.0ms (subsample=100), rollout=300, covers 300ms
    "A2": PhaseConfig(
        name="A2",
        trainable_params=_HALF1_PARAMS,
        loss_fn="ionic_state",
        data_tiers=[1],
        batch_size=128, lr=1e-4, weight_decay=1e-4,
        rollout_length=300, subsample=100,
        transition_metric="val_ionic_state_mse", transition_threshold=None,
        patience=50, max_epochs=500,
    ),
    # dt=0.1ms (subsample=10), rollout=3000, covers 300ms
    "A3": PhaseConfig(
        name="A3",
        trainable_params=_HALF1_PARAMS,
        loss_fn="ionic_state",
        data_tiers=[1],
        batch_size=128, lr=1e-4, weight_decay=1e-4,
        rollout_length=3000, subsample=10,
        transition_metric="val_ionic_state_mse", transition_threshold=None,
        patience=50, max_epochs=500,
    ),
    # dt=0.01ms (subsample=1), rollout=30000, covers 300ms — original resolution
    # tbptt_window=500: only backprop through last 500 steps (5ms), forward pass runs all 30K
    "A4": PhaseConfig(
        name="A4",
        trainable_params=_HALF1_PARAMS,
        loss_fn="ionic_state",
        data_tiers=[1],
        batch_size=128, lr=1e-4, weight_decay=1e-4,
        rollout_length=30000, subsample=1,
        transition_metric="val_ionic_state_mse", transition_threshold=None,
        tbptt_window=500,
        patience=50, max_epochs=1000,
    ),

    # === B: Half 2, same dt curriculum (Half 1 frozen) ===
    "B1": PhaseConfig(
        name="B1",
        trainable_params=_HALF2_PARAMS,
        loss_fn="ionic_state_and_conductance",
        data_tiers=[1],
        batch_size=128, lr=1e-3, weight_decay=1e-4,
        rollout_length=100, subsample=300,
        transition_metric="val_ionic_state_mse", transition_threshold=None,
        patience=50, max_epochs=200,
    ),
    "B2": PhaseConfig(
        name="B2",
        trainable_params=_HALF2_PARAMS,
        loss_fn="ionic_state_and_conductance",
        data_tiers=[1],
        batch_size=128, lr=1e-4, weight_decay=1e-4,
        rollout_length=300, subsample=100,
        transition_metric="val_ionic_state_mse", transition_threshold=None,
        patience=50, max_epochs=500,
    ),
    "B3": PhaseConfig(
        name="B3",
        trainable_params=_HALF2_PARAMS,
        loss_fn="ionic_state_and_conductance",
        data_tiers=[1],
        batch_size=128, lr=1e-4, weight_decay=1e-4,
        rollout_length=3000, subsample=10,
        transition_metric="val_ionic_state_mse", transition_threshold=None,
        patience=50, max_epochs=500,
    ),
    "B4": PhaseConfig(
        name="B4",
        trainable_params=_HALF2_PARAMS,
        loss_fn="ionic_state_and_conductance",
        data_tiers=[1],
        batch_size=128, lr=1e-4, weight_decay=1e-4,
        rollout_length=30000, subsample=1,
        transition_metric="val_ionic_state_mse", transition_threshold=None,
        patience=50, max_epochs=1000,
        tbptt_window=500,
    ),

    # === C: Stage 2, D: end-to-end ===
    "C": PhaseConfig(
        name="C",
        trainable_params=["stage2.*"],
        loss_fn="I_ion",
        data_tiers=[1],
        batch_size=128, lr=1e-3, weight_decay=1e-3,
        rollout_length=3000, subsample=10,
        transition_metric="val_I_ion_mse", transition_threshold=1e-3,
        patience=20, max_epochs=300,
    ),
    "D": PhaseConfig(
        name="D",
        trainable_params=_ALL_STAGE1 + ["stage2.*"],
        loss_fn="I_ion",
        data_tiers=[1],
        batch_size=2048, lr=5e-5, weight_decay=1e-3,
        rollout_length=3000, subsample=10,
        transition_metric="val_I_ion_mse", transition_threshold=None,
        patience=20, max_epochs=300,
    ),
}

PHASE_ORDER = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "C", "D"]


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
