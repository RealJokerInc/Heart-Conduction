"""Structured config schemas registered with Hydra's ConfigStore.

Applies only to `training` and `tracking` groups per OPEN-3 resolution.
`model`, `data`, `experiment` stay free-form YAML — too diverse for a
single schema.

Round-3 MED-6: `_register()` is called from `scripts/train.py` before
`@hydra.main`, NOT from `cardiac_ml/__init__.py` (which would force
unconditional Hydra import at module load, breaking the PEP 562 lazy
pattern).

Round-3 MED-7: TrainingConfig includes NODE-specific fields
(phase_name, ode_*) as Optional so teacher-forced configs still validate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class TrainingConfig:
    """Schema for cfg.training. Required: epochs, optimizer. Everything else
    has a default; NODE-specific fields are Optional."""
    epochs: int = 1
    optimizer: Any = None            # _target_ spec — free-form within this
    train_step_fn: Optional[Any] = None  # _target_: hydra.utils.get_method spec
    val_step_fn: Optional[Any] = None    # _target_: hydra.utils.get_method spec
    callbacks: List[Any] = field(default_factory=list)
    seed: int = 42
    device: str = "cuda"
    dtype: str = "float64"

    # NODE-specific (all Optional — teacher-forced configs omit these)
    phase_name: Optional[str] = None
    ode_method: str = "dopri5"
    ode_rtol: float = 1e-3
    ode_atol: float = 1e-3
    ode_adjoint: bool = False


@dataclass
class TrackingConfig:
    """Schema for cfg.tracking. Default path is file-backed local ./mlruns/."""
    enabled: bool = True
    experiment_name: str = "default"
    tracking_uri: str = "./mlruns"
    checkpoint_every: int = 50


def _register() -> None:
    """Register structured configs with Hydra's ConfigStore singleton.

    Call once from `scripts/train.py` before `@hydra.main`. Idempotent —
    re-registering the same group/name is a no-op.
    """
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    cs.store(group="training", name="schema", node=TrainingConfig)
    cs.store(group="tracking", name="schema", node=TrackingConfig)
