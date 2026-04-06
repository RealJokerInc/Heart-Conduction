"""NODE training rollout via odeint_adjoint.

Replaces discrete rollout.py for Neural ODE training.
Loss computed at AP landmark t_eval points; adjoint backprops through full trajectory.
Original rollout.py preserved in archive/ for reference.

IMPORTANT: Do NOT call clear_v_trajectory() before loss.backward(). The adjoint
method re-calls node.forward() during the backward pass, which needs V(t).
Clear after backward completes.
"""

import torch
from torch import Tensor
from typing import Optional

from ..model.node import IonicNODE
from .loss_normalization import LossNormalizer

# Resting concentrations (Layer 0 physics) — duplicated from rollout.py to avoid
# dependency on discrete training code. Same values: [Na_i, K_i, Ca_i, Ca_ss].
INIT_CONC = torch.tensor([10.0, 138.0, 0.0001, 0.0002], dtype=torch.float64)

# AP landmark evaluation times (ms) — dense during upstroke where dynamics are stiff.
# 10 points in first 5ms (upstroke), 10 in plateau/repol/diastole.
NODE_T_EVAL_MS = torch.tensor(
    [0.0, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0,   # dense upstroke
     10.0, 20.0, 40.0, 80.0,                                 # plateau
     120.0, 160.0, 200.0, 240.0, 270.0, 300.0],              # repol + diastole
    dtype=torch.float64,
)

_normalizer = LossNormalizer()


def build_t_grid(segment_dt: Tensor) -> Tensor:
    """Build cumulative time grid from dt values.

    Args:
        segment_dt: (B, T) or (T,) in ms.

    Returns:
        (T+1,) cumulative times starting at 0.

    NOTE: Uses first batch element's dt — assumes uniform dt across batch.
    If dt varies per batch element, this must be revised.
    """
    dt_1d = segment_dt[0] if segment_dt.dim() == 2 else segment_dt
    return torch.cat([
        torch.zeros(1, dtype=torch.float64, device=dt_1d.device),
        dt_1d.double().cumsum(0),
    ])


def node_rollout(
    node: IonicNODE,
    segment: dict,
    phase_name: str = "A1",
    device: Optional[torch.device] = None,
    t_eval_ms: Optional[Tensor] = None,
    z0_noise_sigma: float = 0.0,
) -> dict:
    """NODE training rollout: integrate z via odeint_adjoint, compute loss at landmarks.

    V trajectory is set before integration and must NOT be cleared until after
    loss.backward() completes (adjoint re-calls forward during backward pass).

    Args:
        node: IonicNODE instance (wraps stage1.dzdt).
        segment: dict with Vm (B,T), dt (B,T), ionic_states (B,T,14), etc.
        phase_name: A1/A2/A3/A4/B1/B2/B3/B4/C/D — same as discrete rollout.
        device: target device.
        t_eval_ms: override landmark times (default: NODE_T_EVAL_MS).
        z0_noise_sigma: Gaussian noise std on z0 for attractor basin widening (0=off).

    Returns:
        dict with 'loss' and per-component losses (same keys as discrete rollout).
        Caller MUST call node.clear_v_trajectory() after loss.backward().
    """
    if device is None:
        device = segment['Vm'].device

    B = segment['Vm'].shape[0]
    T = segment['Vm'].shape[1]

    # Build time grid and t_eval
    t_grid = build_t_grid(segment['dt'].to(device))  # (T+1,)
    T_max = t_grid[-1]

    t_eval = (t_eval_ms if t_eval_ms is not None else NODE_T_EVAL_MS).to(device)
    t_eval = t_eval[t_eval <= T_max]  # clamp to trajectory length
    if len(t_eval) == 0 or t_eval[0] > 0:
        t_eval = torch.cat([torch.zeros(1, dtype=torch.float64, device=device), t_eval])

    # Initialize state: ionic=zeros, conc=resting
    z0 = torch.zeros(B, node.stage1.carried_dim, dtype=torch.float64, device=device)
    z0[:, node.stage1.ionic_dim:] = INIT_CONC.to(device)

    # Attractor basin widening: add Gaussian noise to z0 during training
    if z0_noise_sigma > 0 and node.training:
        z0 = z0 + z0_noise_sigma * torch.randn_like(z0)

    # Set V trajectory for interpolation during ODE solve.
    # NOTE: V_traj has T points, t_grid has T+1 points. _interpolate_V handles this.
    # Do NOT clear until after loss.backward() — adjoint re-calls forward.
    node.set_v_trajectory(segment['Vm'].double().to(device), t_grid)

    z_traj = node.integrate(z0, t_eval)  # (N_eval, B, carried_dim)

    # Compute loss at each t_eval point
    losses_per_eval = []
    component_sums: dict = {}

    for i, t_i in enumerate(t_eval):
        # Find nearest segment index for ground truth
        idx = int((torch.searchsorted(t_grid, t_i) - 1).clamp(0, T - 1).item())
        z_pred = z_traj[i]  # (B, carried_dim)

        step_losses = _compute_node_loss(phase_name, z_pred, segment, idx, node)
        losses_per_eval.append(step_losses['loss'])

        for k, v in step_losses.items():
            if k != 'loss':
                component_sums[k] = component_sums.get(k, 0.0) + v.detach()

    mean_loss = torch.stack(losses_per_eval).mean()

    result = {'loss': mean_loss}
    N_eval = len(t_eval)
    for k, v in component_sums.items():
        result[k] = v / N_eval
    return result


def _compute_node_loss(
    phase_name: str,
    z_pred: Tensor,
    segment: dict,
    idx: int,
    node: IonicNODE,
) -> dict:
    """Loss at one t_eval point. Mirrors compute_phase_loss from rollout.py."""
    losses = {}
    ionic_pred = z_pred[:, :node.stage1.ionic_dim]
    conc_pred = z_pred[:, node.stage1.ionic_dim:]

    # Scaffold predictions via decoders (if present)
    ionic_state_pred = None
    conductance_pred = None
    if hasattr(node.stage1, 'ionic_state_decoder'):
        ionic_state_pred = node.stage1.ionic_state_decoder(ionic_pred)
    if hasattr(node.stage1, 'gate_conductance_decoder'):
        cond_lat = node.stage1._compress(z_pred)
        conductance_pred = node.stage1.gate_conductance_decoder(cond_lat)

    if phase_name in ("A1", "A2", "A3", "A4", "ionic_state"):
        losses['ionic_state_mse'] = _normalizer.normalized_mse(
            ionic_state_pred, segment['ionic_states'][:, idx, :], 'ionic_states')
        losses['conc_mse'] = _normalizer.normalized_mse(
            conc_pred, segment['concentrations'][:, idx, :], 'concentrations')
        losses['loss'] = losses['ionic_state_mse'] + losses['conc_mse']

    elif phase_name in ("B1", "B2", "B3", "B4", "ionic_state_and_conductance"):
        losses['ionic_state_mse'] = _normalizer.normalized_mse(
            ionic_state_pred, segment['ionic_states'][:, idx, :], 'ionic_states')
        losses['conc_mse'] = _normalizer.normalized_mse(
            conc_pred, segment['concentrations'][:, idx, :], 'concentrations')
        losses['conductance_mse'] = _normalizer.normalized_mse(
            conductance_pred, segment['conductance_products'][:, idx, :],
            'conductance_products')
        losses['loss'] = (losses['ionic_state_mse'] + losses['conc_mse']
                          + losses['conductance_mse'])

    elif phase_name in ("C", "D", "I_ion"):
        raise NotImplementedError(
            "I_ion phase requires Stage2 in node_rollout — "
            "wire IonicNODE with full surrogate first"
        )
    else:
        raise ValueError(f"Unknown phase: {phase_name}")

    return losses
