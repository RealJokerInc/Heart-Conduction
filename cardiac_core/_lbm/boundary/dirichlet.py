"""Dirichlet (fixed-value) boundary condition via anti-bounce-back.

Formula (Inamuro et al. 1995):
    f[opp[a]](x) = -f_star[a](x) + 2 * w[a] * V_D

Enforces V = V_D at the midpoint between the boundary node and the wall.

WARNING: Dirichlet on V is NOT the Kleber boundary speedup. It acts as a current
sink and SLOWS conduction. See BOUNDARY_SPEEDUP_ANALYSIS.md S4 for details.
Use for: manufactured solutions, stimulus injection, testing.
NOT for: modeling tissue boundaries (use Neumann instead).

Layer 2: pure functions, torch.compile compatible.
"""

import torch
from torch import Tensor


def apply_dirichlet_d2q5(f: Tensor, f_star: Tensor,
                         bounce_masks: dict[int, Tensor],
                         V_bc: Tensor, w: Tensor) -> Tensor:
    """Apply Dirichlet BC (anti-bounce-back) for D2Q5.

    Args:
        f: (5, Nx, Ny) post-streaming distributions
        f_star: (5, Nx, Ny) pre-streaming distributions
        bounce_masks: dict from precompute_bounce_masks
        V_bc: scalar or (Nx, Ny) boundary voltage
        w: (5,) lattice weights

    Returns:
        f with boundary distributions corrected
    """
    # D2Q5 opposite: (0, 2, 1, 4, 3)
    f[2] = torch.where(bounce_masks[1], -f_star[1] + 2 * w[1] * V_bc, f[2])
    f[1] = torch.where(bounce_masks[2], -f_star[2] + 2 * w[2] * V_bc, f[1])
    f[4] = torch.where(bounce_masks[3], -f_star[3] + 2 * w[3] * V_bc, f[4])
    f[3] = torch.where(bounce_masks[4], -f_star[4] + 2 * w[4] * V_bc, f[3])
    return f


def apply_dirichlet_d2q9(f: Tensor, f_star: Tensor,
                         bounce_masks: dict[int, Tensor],
                         V_bc: Tensor, w: Tensor) -> Tensor:
    """Apply Dirichlet BC (anti-bounce-back) for D2Q9.

    Args:
        f: (9, Nx, Ny) post-streaming distributions
        f_star: (9, Nx, Ny) pre-streaming distributions
        bounce_masks: dict from precompute_bounce_masks
        V_bc: scalar or (Nx, Ny) boundary voltage
        w: (9,) lattice weights

    Returns:
        f with boundary distributions corrected
    """
    # D2Q9 opposite: (0, 2, 1, 4, 3, 7, 8, 5, 6)
    # Cardinals
    f[2] = torch.where(bounce_masks[1], -f_star[1] + 2 * w[1] * V_bc, f[2])
    f[1] = torch.where(bounce_masks[2], -f_star[2] + 2 * w[2] * V_bc, f[1])
    f[4] = torch.where(bounce_masks[3], -f_star[3] + 2 * w[3] * V_bc, f[4])
    f[3] = torch.where(bounce_masks[4], -f_star[4] + 2 * w[4] * V_bc, f[3])
    # Diagonals
    f[7] = torch.where(bounce_masks[5], -f_star[5] + 2 * w[5] * V_bc, f[7])
    f[8] = torch.where(bounce_masks[6], -f_star[6] + 2 * w[6] * V_bc, f[8])
    f[5] = torch.where(bounce_masks[7], -f_star[7] + 2 * w[7] * V_bc, f[5])
    f[6] = torch.where(bounce_masks[8], -f_star[8] + 2 * w[8] * V_bc, f[6])
    return f
