"""Validation metrics: APD90, dVm/dt_max, per-phase metric computation."""

from typing import Union

import torch
from torch import Tensor


def compute_apd90(Vm_trace: Tensor, dt: Union[Tensor, float] = 0.01) -> Tensor:
    """Compute APD90 from a Vm trace.

    Args:
        Vm_trace: (T,) membrane voltage trace in mV.
        dt: Scalar or (T,) tensor of timestep sizes in ms.

    Returns:
        APD90 in ms. Returns NaN if no clean AP detected.
    """
    if Vm_trace.dim() != 1:
        raise ValueError(f"Expected 1D Vm_trace, got shape {Vm_trace.shape}")

    threshold = -40.0  # mV, upstroke detection threshold
    Vm = Vm_trace.detach()

    # Find upstroke: first crossing above threshold
    above = Vm > threshold
    if not above.any():
        return torch.tensor(float('nan'), dtype=Vm.dtype)

    # Find first upstroke index
    crossings = torch.where(above[1:] & ~above[:-1])[0]
    if len(crossings) == 0:
        return torch.tensor(float('nan'), dtype=Vm.dtype)
    t_up = crossings[0].item() + 1

    # Find Vm_max after upstroke
    Vm_after = Vm[t_up:]
    if len(Vm_after) == 0:
        return torch.tensor(float('nan'), dtype=Vm.dtype)

    Vm_max = Vm_after.max().item()
    Vm_rest = Vm[:t_up].mean().item() if t_up > 0 else Vm[0].item()

    # 90% repolarization level
    repol_level = Vm_max - 0.9 * (Vm_max - Vm_rest)

    # Find first crossing below repol_level after Vm_max
    idx_max = t_up + Vm_after.argmax().item()
    Vm_repol = Vm[idx_max:]
    below = Vm_repol < repol_level
    if not below.any():
        return torch.tensor(float('nan'), dtype=Vm.dtype)

    t_repol_rel = torch.where(below)[0][0].item()
    t_repol = idx_max + t_repol_rel

    # Compute APD
    if isinstance(dt, Tensor) and dt.dim() > 0:
        apd = dt[t_up:t_repol].sum()
    else:
        apd = (t_repol - t_up) * float(dt)

    return torch.tensor(apd, dtype=Vm.dtype)


def compute_dvdt_max(Vm_trace: Tensor, dt: Union[Tensor, float] = 0.01) -> Tensor:
    """Compute maximum upstroke velocity dVm/dt.

    Args:
        Vm_trace: (T,) membrane voltage trace in mV.
        dt: Scalar or (T,) tensor of timestep sizes in ms.

    Returns:
        Max dVm/dt in mV/ms.
    """
    if Vm_trace.dim() != 1:
        raise ValueError(f"Expected 1D Vm_trace, got shape {Vm_trace.shape}")

    dVm = Vm_trace[1:] - Vm_trace[:-1]

    if isinstance(dt, Tensor) and dt.dim() > 0:
        dt_vals = dt[:-1]
    else:
        dt_vals = float(dt)

    dVdt = dVm / dt_vals
    return dVdt.max()


def compute_phase_metrics(phase_name: str, predictions: dict, targets: dict) -> dict:
    """Compute all relevant metrics for a phase.

    Returns dict of {metric_name: value}.
    """
    metrics = {}

    if phase_name == "A1":
        metrics['recon_mse'] = torch.nn.functional.mse_loss(
            predictions['decoded'], targets['ionic_states']
        ).item()

    elif phase_name in ("B1", "B2", "B3", "B4", "B5"):
        metrics['ionic_state_mse'] = torch.nn.functional.mse_loss(
            predictions['ionic_state_pred'], targets['ionic_states']
        ).item()

    elif phase_name in ("D", "E"):
        metrics['I_ion_mse'] = torch.nn.functional.mse_loss(
            predictions['I_ion'], targets['I_ion']
        ).item()

    return metrics
