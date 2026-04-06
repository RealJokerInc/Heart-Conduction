# ARCHIVED: 2026-04-06 — Discrete autoregressive training pipeline.
# Superseded by Neural ODE pivot (node_rollout.py).
# A4 (dt=0.01ms, 30K steps) failed after 155+ epochs — val stuck at ~720.
# Root cause: error compounding over long discrete rollouts. Not a hyperparameter issue.
# Kept for historical reference. Do NOT import in production code.
# See KNOWLEDGE.md Section 5b for full analysis.

"""Autoregressive rollout engine.

Executes the model step-by-step over a segment, accumulating per-step losses.
All rollouts are purely autoregressive — the model always feeds its own output.

Key convention: Stage 2 reads PREVIOUS step's conductance_latent and concentrations
(operator splitting — I_ion(t) depends on state(t), while Stage 1 computes state(t+1)).
"""

from typing import Optional

import torch
from torch import Tensor

from ..model.ionic_surrogate_v3 import IonicSurrogateV3
from .loss_normalization import LossNormalizer

# Default initial concentrations (resting values, Layer 0 physics)
INIT_CONC = torch.tensor([10.0, 138.0, 0.0001, 0.0002], dtype=torch.float64)

# Shared normalizer instance for all rollout calls
_normalizer = LossNormalizer()


def compute_phase_loss(
    phase_name: str,
    model_out: dict[str, Tensor],
    segment: dict[str, Tensor],
    t: int,
) -> dict[str, Tensor]:
    """Compute single-step loss components for a given phase.

    Returns dict with 'loss' (total) and per-component losses.
    """
    losses = {}

    if phase_name in ("A1", "A2", "A3", "A4") or phase_name == "ionic_state":
        losses['ionic_state_mse'] = _normalizer.normalized_mse(
            model_out['ionic_state_pred'], segment['ionic_states'][:, t, :], 'ionic_states')
        losses['conc_mse'] = _normalizer.normalized_mse(
            model_out['concentrations'], segment['concentrations'][:, t, :], 'concentrations')
        losses['loss'] = losses['ionic_state_mse'] + losses['conc_mse']

    elif phase_name in ("B1", "B2", "B3", "B4") or phase_name == "ionic_state_and_conductance":
        losses['ionic_state_mse'] = _normalizer.normalized_mse(
            model_out['ionic_state_pred'], segment['ionic_states'][:, t, :], 'ionic_states')
        losses['conc_mse'] = _normalizer.normalized_mse(
            model_out['concentrations'], segment['concentrations'][:, t, :], 'concentrations')
        losses['conductance_mse'] = _normalizer.normalized_mse(
            model_out['conductance_pred'], segment['conductance_products'][:, t, :], 'conductance_products')
        losses['loss'] = losses['ionic_state_mse'] + losses['conc_mse'] + losses['conductance_mse']

    elif phase_name in ("C", "D") or phase_name == "I_ion":
        losses['I_ion_mse'] = torch.nn.functional.mse_loss(
            model_out['I_ion'], segment['I_ion'][:, t])
        losses['loss'] = losses['I_ion_mse']

    else:
        raise ValueError(f"Unknown phase for loss computation: {phase_name}")

    return losses


def rollout(
    model: IonicSurrogateV3,
    segment: dict[str, Tensor],
    phase_name: str = "A1",
    device: Optional[torch.device] = None,
    tbptt_window: int = 0,
) -> dict[str, Tensor]:
    """Execute autoregressive rollout over a segment, accumulating loss.

    Supports truncated BPTT: the model runs the full T-step forward pass
    (sees all error compounding), but gradients only flow through the last
    `tbptt_window` steps. Earlier steps are detached. This keeps the gradient
    chain short and stable while the model still learns from late-rollout errors.

    Args:
        model: IonicSurrogateV3 instance.
        segment: Dict of (B, T, ...) tensors from SegmentDataset + DataLoader.
        phase_name: Determines which loss function to use.
        device: Device for initial state tensors.
        tbptt_window: If > 0, only backprop through the last N steps.
            Steps before T-N are detached (forward pass still runs).
            If 0 (default), backprop through all steps (original behavior).

    Returns:
        dict with:
            'loss': scalar mean loss over rollout steps (or over tbptt window)
            'per_step_losses': (T,) individual step losses (detached for monitoring)
    """
    B = segment['Vm'].shape[0]
    T = segment['Vm'].shape[1]

    if device is None:
        device = segment['Vm'].device

    # Initialize state: ionic latent = zeros, concentrations = resting values
    carried = torch.zeros(B, model.carried_dim, dtype=torch.float64, device=device)
    carried[:, model.ionic_dim:] = INIT_CONC.to(device)
    cond_lat_prev = torch.zeros(B, model.cond_dim, dtype=torch.float64, device=device)
    conc_prev = INIT_CONC.to(device).unsqueeze(0).expand(B, -1).clone()

    # Determine where to start accumulating gradients
    if tbptt_window > 0 and tbptt_window < T:
        grad_start = T - tbptt_window
    else:
        grad_start = 0

    per_step_losses = []
    component_sums: dict[str, float] = {}

    for t in range(T):
        Vm_t = segment['Vm'][:, t]
        dt_t = segment['dt'][:, t]

        # Detach state before the gradient window — forward pass continues
        # but gradients don't flow back through early steps
        if t == grad_start and grad_start > 0:
            carried = carried.detach()
            cond_lat_prev = cond_lat_prev.detach()
            conc_prev = conc_prev.detach()

        # Forward pass
        out = model(carried, Vm_t, dt_t, cond_lat_prev, conc_prev)

        # Compute per-step loss components
        step_losses = compute_phase_loss(phase_name, out, segment, t)

        # Only accumulate loss for gradient steps (or all if no truncation)
        if t >= grad_start:
            per_step_losses.append(step_losses['loss'])

        # Accumulate component losses for monitoring (all steps, detached)
        for k, v in step_losses.items():
            if k != 'loss':
                component_sums[k] = component_sums.get(k, 0.0) + v.detach()

        # Update prev-step state for next iteration's Stage 2
        cond_lat_prev = out['conductance_latent']
        conc_prev = out['concentrations']

        # Autoregressive: always use model's own prediction
        carried = out['carried_state']

    per_step_losses = torch.stack(per_step_losses)
    mean_loss = per_step_losses.mean()

    result = {
        'loss': mean_loss,
        'per_step_losses': per_step_losses,
    }
    # Add mean component losses (over ALL steps for monitoring)
    for k, v in component_sums.items():
        result[k] = v / T

    return result
