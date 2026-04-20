"""Trainer-shape adapter for the existing node_rollout() function.

Pure function `node_train_step(trainer, batch) -> dict` that wraps the oracle
`node_rollout(node, segment, phase_name)` call and returns a dict matching the
cardiac_ml Trainer protocol:

    {"loss": tensor,
     "_on_after_backward": callable,      # invoked post-backward for V_traj cleanup
     "ionic_state_mse": tensor, ...}      # per-component metrics, detached

phase_name is REQUIRED — no silent default. Valid values per node_rollout.py:160:
    A1 | A2 | A3 | A4 | ionic_state | conc_only
    B1 | B2 | B3 | B4 | ionic_state_and_conductance

The cleanup hook calls `trainer.model.clear_v_trajectory()` AFTER `loss.backward()`
completes. node_rollout sets the V trajectory internally; adjoint/non-adjoint both
require the trajectory to persist through backward.

t_eval construction — matches run_multi_bcl.py:189-192 when `_bcl` is on the
batch (full-resolution 0.1-ms grid over the beat). Falls back to NODE_T_EVAL_MS
(20-landmark grid) when `_bcl` is absent.
"""
from __future__ import annotations

import torch

from .node_rollout import node_rollout


_VALID_PHASES = (
    "A1", "A2", "A3", "A4", "ionic_state",
    "conc_only",
    "B1", "B2", "B3", "B4", "ionic_state_and_conductance",
)


def _phase_from_cfg(trainer) -> str:
    pn = trainer.cfg.training.get("phase_name")
    if pn is None:
        raise KeyError(
            "cfg.training.phase_name is required for NODE training "
            f"(valid: {'|'.join(_VALID_PHASES)}). See node_rollout.py:160-191."
        )
    return pn


def _ode_kwargs(trainer) -> dict:
    t = trainer.cfg.training
    return dict(
        method=t.get("ode_method", "dopri5"),
        rtol=t.get("ode_rtol", 1e-3),
        atol=t.get("ode_atol", 1e-3),
        adjoint=t.get("ode_adjoint", False),
    )


def _t_eval_from_batch(batch: dict, device: torch.device) -> torch.Tensor | None:
    """Build full-resolution t_eval matching run_multi_bcl.py:189-192.

    If the batch carries `_bcl` metadata (int ms) — as produced by the multi-BCL
    loader — return a linspace from 0 to T_ms at 0.1 ms spacing. Otherwise
    return None so node_rollout falls back to NODE_T_EVAL_MS.
    """
    bcl = batch.get("_bcl")
    if bcl is None:
        return None
    if isinstance(bcl, (list, tuple)):
        bcl = bcl[0]
    T_ms = float(bcl)
    n_pts = int(T_ms / 0.1) + 1
    return torch.linspace(0.0, T_ms, n_pts, dtype=torch.float64, device=device)


def node_train_step(trainer, batch) -> dict:
    """Train-step adapter: call node_rollout, surface loss + metrics + cleanup hook."""
    device = batch["Vm"].device
    result = node_rollout(
        node=trainer.model,
        segment=batch,
        phase_name=_phase_from_cfg(trainer),
        t_eval_ms=_t_eval_from_batch(batch, device),
        **_ode_kwargs(trainer),
    )
    loss = result["loss"]
    model = trainer.model

    def _clear():
        model.clear_v_trajectory()

    extra = {k: v.detach() for k, v in result.items() if k != "loss"}
    return {"loss": loss, "_on_after_backward": _clear, **extra}


def node_val_step(trainer, batch) -> dict:
    """Val-step adapter: no-grad variant. Still clears V_traj between batches."""
    device = batch["Vm"].device
    with torch.no_grad():
        result = node_rollout(
            node=trainer.model,
            segment=batch,
            phase_name=_phase_from_cfg(trainer),
            t_eval_ms=_t_eval_from_batch(batch, device),
            **_ode_kwargs(trainer),
        )
    loss = result["loss"]
    model = trainer.model

    def _clear():
        model.clear_v_trajectory()

    extra = {k: v.detach() for k, v in result.items() if k != "loss"}
    return {"loss": loss, "val_loss": loss, "_on_after_backward": _clear, **extra}
