"""Absorbing boundary condition — equilibrium incoming distributions.

Formula:
    f[opp[a]](x) = w[opp[a]] * V(x, t)

Sets incoming distributions to equilibrium based on the current local voltage.
Minimizes reflections at domain boundaries.

WARNING: Non-conservative — total sum(f) is not preserved. This is intentional
for open/absorbing boundaries.

Layer 2: pure functions, torch.compile compatible.
"""

import torch
from torch import Tensor


def apply_absorbing_d2q5(f: Tensor, bounce_masks: dict[int, Tensor],
                         V: Tensor, w: Tensor) -> Tensor:
    """Apply absorbing BC for D2Q5.

    Args:
        f: (5, Nx, Ny) post-streaming distributions
        bounce_masks: dict from precompute_bounce_masks
        V: (Nx, Ny) previous timestep voltage
        w: (5,) lattice weights

    Returns:
        f with incoming boundary distributions set to equilibrium
    """
    # D2Q5 opposite: (0, 2, 1, 4, 3)
    # At east wall (bounce_masks[1]), incoming direction is W (2)
    f[2] = torch.where(bounce_masks[1], w[2] * V, f[2])
    f[1] = torch.where(bounce_masks[2], w[1] * V, f[1])
    f[4] = torch.where(bounce_masks[3], w[4] * V, f[4])
    f[3] = torch.where(bounce_masks[4], w[3] * V, f[3])
    return f


def apply_absorbing_d2q9(f: Tensor, bounce_masks: dict[int, Tensor],
                         V: Tensor, w: Tensor) -> Tensor:
    """Apply absorbing BC for D2Q9.

    Args:
        f: (9, Nx, Ny) post-streaming distributions
        bounce_masks: dict from precompute_bounce_masks
        V: (Nx, Ny) previous timestep voltage
        w: (9,) lattice weights

    Returns:
        f with incoming boundary distributions set to equilibrium
    """
    # D2Q9 opposite: (0, 2, 1, 4, 3, 7, 8, 5, 6)
    # Cardinals
    f[2] = torch.where(bounce_masks[1], w[2] * V, f[2])
    f[1] = torch.where(bounce_masks[2], w[1] * V, f[1])
    f[4] = torch.where(bounce_masks[3], w[4] * V, f[4])
    f[3] = torch.where(bounce_masks[4], w[3] * V, f[3])
    # Diagonals
    f[7] = torch.where(bounce_masks[5], w[7] * V, f[7])
    f[8] = torch.where(bounce_masks[6], w[8] * V, f[8])
    f[5] = torch.where(bounce_masks[7], w[5] * V, f[5])
    f[6] = torch.where(bounce_masks[8], w[6] * V, f[6])
    return f
