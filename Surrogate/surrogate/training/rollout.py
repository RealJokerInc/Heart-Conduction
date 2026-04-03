"""Autoregressive rollout engine with scheduled sampling.

Executes the model step-by-step over a segment, accumulating per-step losses.
Handles teacher forcing (Phase B) via the temporary encoder, and autoregressive
execution (Phase C-E).

Key convention: Stage 2 reads PREVIOUS step's conductance_latent and concentrations
(operator splitting — I_ion(t) depends on state(t), while Stage 1 computes state(t+1)).
"""

import random as pyrandom
from typing import Optional

import torch
from torch import Tensor

from ..model.ionic_surrogate_v3 import IonicSurrogateV3
from ..model.stage1 import interpolate
from .encoder import TemporaryEncoder

# Default initial concentrations (resting values, Layer 0 physics)
INIT_CONC = torch.tensor([10.0, 138.0, 0.0001, 0.0002], dtype=torch.float64)


def _recompute_conductance(model: IonicSurrogateV3, carried_state: Tensor) -> Tensor:
    """Recompute conductance latent from carried_state using Stage 1's compression.

    Replicates the inline logic from IonicStage1.forward() lines 212-215:
    linear + nonlinear + interpolate. Used during teacher forcing to reset
    cond_lat_prev from ground-truth state.
    """
    s1 = model.stage1
    linear_path = s1.gate_conductance_linear(carried_state)
    nonlinear_path = s1.gate_conductance_mlp(carried_state)
    return interpolate(linear_path, nonlinear_path, s1.gate_conductance_logit)


def compute_phase_loss(
    phase_name: str,
    model_out: dict[str, Tensor],
    segment: dict[str, Tensor],
    t: int,
) -> Tensor:
    """Compute single-step loss for a given phase. Single MSE, no weighting.

    Dispatch:
        autoencoder (A1): handled outside rollout (not temporal)
        concentration (A2): MSE(attention output conc, true conc at t+1)
        conductance (A3): MSE(decoded conductance, true products at t)
        ionic_state (B): MSE(ionic_state_pred, true ionic states at t)
        concentration_rollout (C): MSE(predicted conc, true conc at t)
        I_ion (D/E): MSE(predicted I_ion, true I_ion at t)
    """
    if phase_name in ("B1", "B2", "B3", "B4", "B5") or phase_name.startswith("ionic_state"):
        pred = model_out['ionic_state_pred']
        target = segment['ionic_states'][:, t, :]
        return torch.nn.functional.mse_loss(pred, target)

    elif phase_name == "C" or phase_name == "concentration_rollout":
        pred = model_out['concentrations']
        target = segment['concentrations'][:, t, :]
        return torch.nn.functional.mse_loss(pred, target)

    elif phase_name in ("D", "E") or phase_name == "I_ion":
        pred = model_out['I_ion']
        target = segment['I_ion'][:, t]
        return torch.nn.functional.mse_loss(pred, target)

    elif phase_name == "A2" or phase_name == "concentration":
        # Pairs: predict next-step concentration
        pred = model_out['concentrations']
        if t + 1 < segment['concentrations'].shape[1]:
            target = segment['concentrations'][:, t + 1, :]
        else:
            target = segment['concentrations'][:, t, :]
        return torch.nn.functional.mse_loss(pred, target)

    elif phase_name == "A3" or phase_name == "conductance":
        pred = model_out['conductance_pred']
        target = segment['conductance_products'][:, t, :]
        return torch.nn.functional.mse_loss(pred, target)

    else:
        raise ValueError(f"Unknown phase for loss computation: {phase_name}")


def rollout(
    model: IonicSurrogateV3,
    segment: dict[str, Tensor],
    encoder: Optional[TemporaryEncoder] = None,
    scheduled_sampling_p: float = 1.0,
    phase_name: str = "B1",
    device: Optional[torch.device] = None,
) -> dict[str, Tensor]:
    """Execute autoregressive rollout over a segment, accumulating loss.

    Args:
        model: IonicSurrogateV3 instance.
        segment: Dict of (B, T, ...) tensors from SegmentDataset + DataLoader.
        encoder: Temporary encoder for teacher forcing (Phase B only).
        scheduled_sampling_p: Probability of using model's own prediction.
            0.0 = all teacher forcing, 1.0 = all autoregressive.
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

        # Scheduled sampling: use model output or teacher forcing?
        if encoder is not None and pyrandom.random() > scheduled_sampling_p:
            # Teacher forcing: replace carried_state with ground truth
            true_ionic = segment['ionic_states'][:, t, :]
            true_conc = segment['concentrations'][:, t, :]
            latent = encoder(true_ionic)
            carried = torch.cat([latent, true_conc], dim=-1)
            # Recompute conductance from teacher-forced state
            cond_lat_prev = _recompute_conductance(model, carried)
            conc_prev = true_conc
        else:
            # Autoregressive: use model's own prediction
            carried = out['carried_state']

    per_step_losses = torch.stack(per_step_losses)
    mean_loss = per_step_losses.mean()

    return {
        'loss': mean_loss,
        'per_step_losses': per_step_losses,
    }
