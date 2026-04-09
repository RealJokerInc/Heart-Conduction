"""B-spline KAN (Kolmogorov-Arnold Network) layer.

Each edge learns a univariate function via B-spline basis.
Output_j = sum_i φ_ij(x_i) — no cross-communication between inputs.

Used for concentration rate prediction: each concentration's rate is a sum of
independent nonlinear functions of ionic latent dims and Vm.
"""

import torch
import torch.nn as nn
from torch import Tensor


def _bspline_basis(x: Tensor, grid: Tensor, order: int) -> Tensor:
    """Evaluate B-spline basis functions at x.

    Args:
        x: (...,) input values
        grid: (n_knots,) knot positions, must be sorted
        order: spline order (0=constant, 1=linear, 3=cubic)

    Returns:
        (..., n_basis) basis values where n_basis = n_knots - order - 1
    """
    # Cox-de Boor recursion
    n_knots = grid.shape[0]
    # Order 0: indicator functions
    bases = ((x.unsqueeze(-1) >= grid[:-1]) & (x.unsqueeze(-1) < grid[1:])).to(x.dtype)

    # Handle right endpoint: last basis should include right boundary
    bases[..., -1] = torch.where(
        x == grid[-1], torch.ones_like(x), bases[..., -1]
    )

    for k in range(1, order + 1):
        n = n_knots - k - 1
        left_num = x.unsqueeze(-1) - grid[:n]
        left_den = grid[k:k + n] - grid[:n]
        left = left_num / left_den.clamp(min=1e-12) * bases[..., :n]

        right_num = grid[k + 1:k + 1 + n] - x.unsqueeze(-1)
        right_den = grid[k + 1:k + 1 + n] - grid[1:1 + n]
        right = right_num / right_den.clamp(min=1e-12) * bases[..., 1:1 + n]

        bases = left + right

    return bases


class KANLayer(nn.Module):
    """Single KAN layer: n_in → n_out via learnable univariate B-spline functions.

    Each (input_i, output_j) pair has its own B-spline function φ_ij.
    Output_j = sum_i φ_ij(x_i).

    Also includes a residual linear path (SiLU-weighted) per the efficient KAN
    formulation: output_j = sum_i [φ_ij(x_i) + w_ij * SiLU(x_i)].

    Args:
        n_in: number of input dimensions
        n_out: number of output dimensions
        grid_size: number of intervals in the B-spline grid
        order: B-spline order (3 = cubic)
        grid_range: (min, max) for the spline grid knots
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        grid_size: int = 5,
        order: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.order = order
        self.grid_size = grid_size

        # Build extended knot vector (grid_size + 2*order + 1 knots)
        n_knots = grid_size + 2 * order + 1
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(
            grid_range[0] - order * h,
            grid_range[1] + order * h,
            n_knots,
        )
        self.register_buffer('grid', grid)

        # Number of basis functions = n_knots - order - 1 = grid_size + order
        n_basis = grid_size + order

        # Spline coefficients: (n_in, n_out, n_basis) — init near zero
        self.spline_weight = nn.Parameter(
            torch.randn(n_in, n_out, n_basis) * 0.1 / n_basis
        )

        # Residual linear weight: (n_in, n_out)
        self.base_weight = nn.Parameter(
            torch.randn(n_in, n_out) * (1.0 / (n_in * n_out) ** 0.5)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (..., n_in)
        Returns:
            (..., n_out)
        """
        # B-spline basis: (..., n_in, n_basis)
        bases = _bspline_basis(x, self.grid, self.order)

        # Spline path: sum over basis -> (..., n_in, n_out)
        spline_out = torch.einsum('...ib,iob->...io', bases, self.spline_weight)

        # Residual linear path with SiLU: (..., n_in, n_out)
        base_out = torch.nn.functional.silu(x).unsqueeze(-1) * self.base_weight

        # Sum over input dims: (..., n_out)
        return (spline_out + base_out).sum(dim=-2)
