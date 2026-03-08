"""
BidomainDiffusionSolver Abstract Base Class

Solves the bidomain diffusion step. Updates both Vm and phi_e in-place.

Ref: improvement.md L916-948
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from ...discretization.base import BidomainSpatialDiscretization
    from ...state import BidomainState


class BidomainDiffusionSolver(ABC):
    """
    Abstract base class for bidomain diffusion solvers.

    Solves the bidomain diffusion step, updating both Vm and phi_e.
    Concrete strategies: Gauss-Seidel, semi-implicit, Jacobi, coupled, etc.

    Owns:
    - BidomainSpatialDiscretization (provides L_i, L_e stencils)
    - LinearSolver(s) for sub-problems (strategy-dependent)
    """

    def __init__(self, spatial: 'BidomainSpatialDiscretization', dt: float):
        self._spatial = spatial
        self._dt = dt

    @abstractmethod
    def step(self, state: 'BidomainState', dt: float) -> None:
        """
        Advance diffusion by dt.
        Modifies state.Vm AND state.phi_e in-place.
        """
        pass

    def rebuild_operators(self, spatial: 'BidomainSpatialDiscretization', dt: float) -> None:
        """Rebuild operators when dt changes (adaptive time stepping)."""
        self._spatial = spatial
        self._dt = dt

    @staticmethod
    def apply_elliptic_pinning(A, pin_node):
        """Enforce phi_e(pin_node) = 0 by modifying the elliptic matrix.

        For all-Neumann BCs the elliptic operator -(L_i + L_e) has a constant
        null space. Pinning one node removes the singularity.

        Returns the pinned matrix (may be a new sparse tensor).
        """
        if A.is_sparse:
            A_coal = A.coalesce()
            indices = A_coal.indices()
            values = A_coal.values()

            row_mask = indices[0] != pin_node
            col_mask = indices[1] != pin_node
            keep = row_mask & col_mask

            new_row = [indices[0, keep], indices[0].new_tensor([pin_node])]
            new_col = [indices[1, keep], indices[1].new_tensor([pin_node])]
            new_val = [values[keep], values.new_tensor([1.0])]

            new_indices = torch.stack([torch.cat(new_row), torch.cat(new_col)])
            new_values = torch.cat(new_val)
            return torch.sparse_coo_tensor(
                new_indices, new_values, A.shape).coalesce()
        else:
            A[pin_node, :] = 0
            A[:, pin_node] = 0
            A[pin_node, pin_node] = 1.0
            return A
