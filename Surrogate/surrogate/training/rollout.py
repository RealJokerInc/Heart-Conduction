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

# Default initial concentrations (resting values, Layer 0 physics)
INIT_CONC = torch.tensor([10.0, 138.0, 0.0001, 0.0002], dtype=torch.float64)


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

    if phase_name in ("B1", "B2") or phase_name == "ionic_state":
        losses['ionic_state_mse'] = torch.nn.functional.mse_loss(
            model_out['ionic_state_pred'], segment['ionic_states'][:, t, :])
        losses['conc_mse'] = torch.nn.functional.mse_loss(
            model_out['concentrations'], segment['concentrations'][:, t, :])
        losses['loss'] = losses['ionic_state_mse'] + losses['conc_mse']

    elif phase_name in ("B3", "B4", "B5") or phase_name == "ionic_state_and_conductance":
        losses['ionic_state_mse'] = torch.nn.functional.mse_loss(
            model_out['ionic_state_pred'], segment['ionic_states'][:, t, :])
        losses['conc_mse'] = torch.nn.functional.mse_loss(
            model_out['concentrations'], segment['concentrations'][:, t, :])
        losses['conductance_mse'] = torch.nn.functional.mse_loss(
            model_out['conductance_pred'], segment['conductance_products'][:, t, :])
        losses['loss'] = losses['ionic_state_mse'] + losses['conc_mse'] + losses['conductance_mse']

    elif phase_name == "C" or phase_name == "concentration_rollout":
        losses['conc_mse'] = torch.nn.functional.mse_loss(
            model_out['concentrations'], segment['concentrations'][:, t, :])
        losses['loss'] = losses['conc_mse']

    elif phase_name in ("D", "E") or phase_name == "I_ion":
        losses['I_ion_mse'] = torch.nn.functional.mse_loss(
            model_out['I_ion'], segment['I_ion'][:, t])
        losses['loss'] = losses['I_ion_mse']

    else:
        raise ValueError(f"Unknown phase for loss computation: {phase_name}")

    return losses


def rollout(
    model: IonicSurrogateV3,
    segment: dict[str, Tensor],
    phase_name: str = "B1",
    device: Optional[torch.device] = None,
) -> dict[str, Tensor]:
    """Execute autoregressive rollout over a segment, accumulating loss.

    Args:
        model: IonicSurrogateV3 instance.
        segment: Dict of (B, T, ...) tensors from SegmentDataset + DataLoader.
        phase_name: Determines which loss function to use.
        device: Device for initial state tensors.

    Returns:
        dict with:
            'loss': scalar mean loss over rollout steps
            'per_step_losses': (T,) individual step losses
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

    per_step_losses = []
    component_sums: dict[str, float] = {}

    for t in range(T):
        Vm_t = segment['Vm'][:, t]
        dt_t = segment['dt'][:, t]

        # Forward pass
        out = model(carried, Vm_t, dt_t, cond_lat_prev, conc_prev)

        # Compute per-step loss components
        step_losses = compute_phase_loss(phase_name, out, segment, t)
        per_step_losses.append(step_losses['loss'])

        # Accumulate component losses
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
    # Add mean component losses
    for k, v in component_sums.items():
        result[k] = v / T

    return result
