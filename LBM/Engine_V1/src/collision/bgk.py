"""BGK (single-relaxation-time) collision operator.

f*_i = f_i - (1/tau)(f_i - w_i*V) + dt*w_i*R

Layer 2: pure function, no state. torch.compile compatible.
"""

import torch
from torch import Tensor


def bgk_collide(f: Tensor, V: Tensor, R: Tensor, dt: float,
                omega: float, w: Tensor) -> Tensor:
    """BGK collision with source term.

    Args:
        f: (Q, Nx, Ny) distribution functions
        V: (Nx, Ny) macroscopic voltage = sum(f_i)
        R: (Nx, Ny) source term = -(I_ion + I_stim) / Cm
        dt: time step
        omega: relaxation frequency = 1/tau
        w: (Q,) lattice weights

    Returns:
        f_star: (Q, Nx, Ny) post-collision distributions
    """
    # w reshaped to (Q, 1, 1) for broadcasting
    w_3d = w[:, None, None]

    # Equilibrium: f_eq_i = w_i * V
    f_eq = w_3d * V.unsqueeze(0)

    # Collision: f* = f - omega*(f - f_eq) + dt*w_i*R
    f_star = f - omega * (f - f_eq) + dt * w_3d * R.unsqueeze(0)
    return f_star
