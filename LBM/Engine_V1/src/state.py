"""LBM simulation state container and initialization."""

from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass
class LBMState:
    """Container for all LBM simulation state.

    Attributes:
        f: (Q, Nx, Ny) distribution functions
        V: (Nx, Ny) macroscopic voltage = sum(f_i)
        ionic_states: (Nx*Ny, n_states) ionic model state variables
        mask: (Nx, Ny) bool domain mask (True = inside domain)
        t: current simulation time (ms)
    """
    f: Tensor
    V: Tensor
    ionic_states: Tensor
    mask: Tensor
    t: float


def create_lbm_state(Nx: int, Ny: int, lattice, V_init: float = -86.0,
                     n_ionic_states: int = 0,
                     mask: Tensor = None,
                     device: torch.device = None,
                     dtype: torch.dtype = torch.float64) -> LBMState:
    """Create an initialized LBM state.

    Distributions are set to equilibrium: f_i = w_i * V_init.

    Args:
        Nx, Ny: grid dimensions
        lattice: D2Q5 or D2Q9 instance
        V_init: initial voltage (mV), default resting potential
        n_ionic_states: number of ionic state variables (0 = no ionic model)
        mask: optional domain mask, default all True
        device: torch device
        dtype: torch dtype
    """
    w = torch.tensor(lattice.w, device=device, dtype=dtype)

    V = torch.full((Nx, Ny), V_init, device=device, dtype=dtype)
    f = w[:, None, None] * V[None, :, :]  # equilibrium

    if mask is None:
        mask = torch.ones(Nx, Ny, device=device, dtype=torch.bool)

    ionic_states = torch.zeros(Nx * Ny, n_ionic_states, device=device, dtype=dtype)

    return LBMState(f=f, V=V, ionic_states=ionic_states, mask=mask, t=0.0)


def recover_voltage(f: Tensor) -> Tensor:
    """Recover macroscopic voltage from distributions: V = sum(f_i).

    Args:
        f: (Q, Nx, Ny) distributions

    Returns:
        V: (Nx, Ny) voltage
    """
    return f.sum(dim=0)
