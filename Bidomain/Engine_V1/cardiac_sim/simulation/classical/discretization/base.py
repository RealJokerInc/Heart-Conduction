"""
BidomainSpatialDiscretization Abstract Base Class

Provides two Laplacians (L_i, L_e) and operators for the decoupled
parabolic + elliptic solves.

Ref: improvement.md L658-735
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple
import torch

if TYPE_CHECKING:
    from ....tissue_builder.mesh.structured import StructuredGrid


class BidomainSpatialDiscretization(ABC):
    """
    Bidomain spatial discretization provides two Laplacians (L_i, L_e)
    and operators for the decoupled parabolic + elliptic solves.

    The grid and its BoundarySpec are accessible via self.grid.
    """

    @property
    @abstractmethod
    def n_dof(self) -> int:
        """Number of degrees of freedom."""
        ...

    @property
    @abstractmethod
    def grid(self) -> 'StructuredGrid':
        """The mesh, including boundary_spec."""
        ...

    @property
    @abstractmethod
    def coordinates(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """(x, y) coordinates for stimulus evaluation."""
        ...

    # --- Operator application (matrix-free compatible) ---

    @abstractmethod
    def apply_L_i(self, V: torch.Tensor) -> torch.Tensor:
        """Apply intracellular Laplacian: L_i * V = div(D_i * grad(V))."""
        ...

    @abstractmethod
    def apply_L_e(self, V: torch.Tensor) -> torch.Tensor:
        """Apply extracellular Laplacian: L_e * V = div(D_e * grad(V))."""
        ...

    @abstractmethod
    def apply_L_ie(self, V: torch.Tensor) -> torch.Tensor:
        """Apply combined Laplacian: (L_i + L_e) * V."""
        ...

    # --- Decoupled solver operators ---

    @abstractmethod
    def get_parabolic_operators(self, dt: float, theta: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build (A_para, B_para) for: A_para * Vm^{n+1} = B_para * Vm^n + coupling."""
        ...

    @abstractmethod
    def get_elliptic_operator(self) -> torch.Tensor:
        """Build A_ellip = -(L_i + L_e) for: A_ellip * phi_e = L_i * Vm."""
        ...

    # --- Raw matrices (for preconditioner setup, etc.) ---

    @property
    @abstractmethod
    def L_i(self) -> torch.Tensor:
        """Intracellular Laplacian matrix (sparse)."""
        ...

    @property
    @abstractmethod
    def L_e(self) -> torch.Tensor:
        """Extracellular Laplacian matrix (sparse)."""
        ...
