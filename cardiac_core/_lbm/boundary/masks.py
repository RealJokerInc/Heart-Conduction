"""Boundary mask pre-computation for LBM bounce-back.

Computed once at simulation init, stored and passed to compiled step functions.
"""

import torch
from torch import Tensor


def precompute_bounce_masks(domain_mask: Tensor, lattice) -> dict[int, Tensor]:
    """Compute per-direction boundary masks.

    For each direction a, identifies nodes where direction a is "outgoing"
    (the neighbor at x + e_a is outside the domain).

    Args:
        domain_mask: (Nx, Ny) bool tensor, True = inside domain
        lattice: object with .e (velocity vectors) and .Q (number of directions)

    Returns:
        bounce_masks: dict mapping direction index -> (Nx, Ny) bool tensor
            bounce_masks[a] is True where direction a points outside the domain.
            Only non-rest directions (a >= 1) are included.
    """
    bounce_masks = {}
    for a in range(1, lattice.Q):
        ex, ey = lattice.e[a]
        # Shift domain mask in direction a: does the neighbor exist?
        neighbor_mask = torch.roll(
            torch.roll(domain_mask, shifts=-ex, dims=0),
            shifts=-ey, dims=1
        )
        # Outgoing = inside domain AND neighbor is outside
        bounce_masks[a] = domain_mask & ~neighbor_mask
    return bounce_masks
