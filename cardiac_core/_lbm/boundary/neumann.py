"""Neumann (no-flux) boundary condition via bounce-back.

Campos et al. (2016) Eq. 17:
    f[opp[a]](x) = f_star[a](x)

Where f_star is the pre-streaming (post-collision) distribution.

Layer 2: pure functions, torch.compile compatible.
Uses torch.where for kernel fusion (not boolean indexing).
"""

import torch
from torch import Tensor


def apply_neumann_d2q5(f: Tensor, f_star: Tensor,
                       bounce_masks: dict[int, Tensor]) -> Tensor:
    """Apply Neumann BC (bounce-back) for D2Q5.

    Args:
        f: (5, Nx, Ny) post-streaming distributions
        f_star: (5, Nx, Ny) pre-streaming (post-collision) distributions
        bounce_masks: dict from precompute_bounce_masks, keys 1-4

    Returns:
        f with boundary distributions corrected
    """
    # D2Q5 opposite: (0, 2, 1, 4, 3)
    # Dir 1 (E) bounces to 2 (W)
    f[2] = torch.where(bounce_masks[1], f_star[1], f[2])
    # Dir 2 (W) bounces to 1 (E)
    f[1] = torch.where(bounce_masks[2], f_star[2], f[1])
    # Dir 3 (N) bounces to 4 (S)
    f[4] = torch.where(bounce_masks[3], f_star[3], f[4])
    # Dir 4 (S) bounces to 3 (N)
    f[3] = torch.where(bounce_masks[4], f_star[4], f[3])
    return f


def apply_neumann_d2q9(f: Tensor, f_star: Tensor,
                       bounce_masks: dict[int, Tensor]) -> Tensor:
    """Apply Neumann BC (bounce-back) for D2Q9.

    Args:
        f: (9, Nx, Ny) post-streaming distributions
        f_star: (9, Nx, Ny) pre-streaming distributions
        bounce_masks: dict from precompute_bounce_masks, keys 1-8

    Returns:
        f with boundary distributions corrected
    """
    # D2Q9 opposite: (0, 2, 1, 4, 3, 7, 8, 5, 6)
    # Cardinals
    f[2] = torch.where(bounce_masks[1], f_star[1], f[2])   # E -> W
    f[1] = torch.where(bounce_masks[2], f_star[2], f[1])   # W -> E
    f[4] = torch.where(bounce_masks[3], f_star[3], f[4])   # N -> S
    f[3] = torch.where(bounce_masks[4], f_star[4], f[3])   # S -> N
    # Diagonals
    f[7] = torch.where(bounce_masks[5], f_star[5], f[7])   # NE -> SW
    f[8] = torch.where(bounce_masks[6], f_star[6], f[8])   # NW -> SE
    f[5] = torch.where(bounce_masks[7], f_star[7], f[5])   # SW -> NE
    f[6] = torch.where(bounce_masks[8], f_star[8], f[6])   # SE -> NW
    return f
