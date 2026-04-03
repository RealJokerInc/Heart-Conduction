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
) -> Tensor:
    """Compute single-step loss for a given phase. Single MSE, no weighting.

    Dispatch:
        ionic_state (B1-B2): MSE(ionic_state_pred, true ionic states at t)
        ionic_state_and_conductance (B3-B5): ionic_state MSE + conductance MSE
        concentration_rollout (C): MSE(predicted conc, true conc at t)
        I_ion (D/E): MSE(predicted I_ion, true I_ion at t)
    """
    if phase_name in ("B1", "B2") or phase_name == "ionic_state":
        ionic_loss = torch.nn.functional.mse_loss(
            model_out['ionic_state_pred'], segment['ionic_states'][:, t, :])
        conc_loss = torch.nn.functional.mse_loss(
            model_out['concentrations'], segment['concentrations'][:, t, :])
        return ionic_loss + conc_loss

    elif phase_name in ("B3", "B4", "B5") or phase_name == "ionic_state_and_conductance":
        ionic_loss = torch.nn.functional.mse_loss(
            model_out['ionic_state_pred'], segment['ionic_states'][:, t, :])
        conc_loss = torch.nn.functional.mse_loss(
            model_out['concentrations'], segment['concentrations'][:, t, :])
        cond_loss = torch.nn.functional.mse_loss(
            model_out['conductance_pred'], segment['conductance_products'][:, t, :])
        return ionic_loss + conc_loss + cond_loss

    elif phase_name == "C" or phase_name == "concentration_rollout":
        pred = model_out['concentrations']
        target = segment['concentrations'][:, t, :]
        return torch.nn.functional.mse_loss(pred, target)

    elif phase_name in ("D", "E") or phase_name == "I_ion":
        pred = model_out['I_ion']
        target = segment['I_ion'][:, t]
        return torch.nn.functional.mse_loss(pred, target)

    else:
        raise ValueError(f"Unknown phase for loss computation: {phase_name}")


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

    for t in range(T):
        Vm_t = segment['Vm'][:, t]
        dt_t = segment['dt'][:, t]

        # Forward pass
        out = model(carried, Vm_t, dt_t, cond_lat_prev, conc_prev)

        # Compute per-step loss
        step_loss = compute_phase_loss(phase_name, out, segment, t)
        per_step_losses.append(step_loss)

        # Update prev-step state for next iteration's Stage 2
        cond_lat_prev = out['conductance_latent']
        conc_prev = out['concentrations']

        # Autoregressive: always use model's own prediction
        carried = out['carried_state']

    per_step_losses = torch.stack(per_step_losses)
    mean_loss = per_step_losses.mean()

    return {
        'loss': mean_loss,
        'per_step_losses': per_step_losses,
    }
