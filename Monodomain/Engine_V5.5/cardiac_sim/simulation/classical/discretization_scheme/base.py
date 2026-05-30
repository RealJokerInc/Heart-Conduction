"""
SpatialDiscretization Abstract Base Class

Defines the interface that all spatial discretization methods must implement.
Discretization schemes provide spatial OPERATORS — they do NOT control time stepping.

Ref: improvement.md:L761-806
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Tuple
import torch


class MassType(Enum):
    """Type of mass matrix produced by the discretization."""
    IDENTITY = "identity"    # FDM: no mass matrix (M = I)
    DIAGONAL = "diagonal"    # FVM: diagonal volume matrix
    SPARSE = "sparse"        # FEM: consistent mass matrix


@dataclass
class DiffusionOperators:
    """
    Pre-built matrices for implicit time stepping.

    These are constructed by SpatialDiscretization.get_diffusion_operators()
    and consumed by DiffusionSolver.

    Ref: improvement.md:L964-973
    """
    A_lhs: torch.Tensor           # LHS matrix (e.g., M + 0.5*dt*K for CN)
    B_rhs: torch.Tensor           # RHS matrix (e.g., M - 0.5*dt*K for CN)
    apply_mass: Callable          # M*f operation (identity/diagonal/sparse)


def sparse_mv(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Sparse matrix-vector multiplication.

    Parameters
    ----------
    A : torch.Tensor
        Sparse COO matrix
    x : torch.Tensor
        Dense vector

    Returns
    -------
    torch.Tensor
        A @ x
    """
    if A.is_sparse:
        return torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
    else:
        return A @ x


class SpatialDiscretization(ABC):
    """
    Abstract base class for spatial discretization methods.

    Discretization schemes provide spatial OPERATORS.
    They do NOT control time stepping method.

    Subclasses must implement:
    - n_dof: number of degrees of freedom
    - coordinates: (x, y) coordinate arrays for stimulus evaluation
    - mass_type: type of mass matrix (IDENTITY, DIAGONAL, SPARSE)
    - get_diffusion_operators(dt, scheme): build time-step-dependent operators
    - apply_diffusion(V): compute div(D*grad(V)) directly

    Ref: improvement.md:L761-806
    """

    @property
    @abstractmethod
    def n_dof(self) -> int:
        """Number of degrees of freedom (nodes for FEM, cells for FDM/FVM)."""
        ...

    @property
    @abstractmethod
    def coordinates(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """(x, y) coordinates for stimulus evaluation. Each shape (n_dof,)."""
        ...

    @property
    @abstractmethod
    def mass_type(self) -> MassType:
        """IDENTITY (FDM), DIAGONAL (FVM), or SPARSE (FEM)."""
        ...

    @abstractmethod
    def get_diffusion_operators(self, dt: float, scheme: str) -> DiffusionOperators:
        """
        Build operators for implicit time stepping.

        Parameters
        ----------
        dt : float
            Time step size (ms)
        scheme : str
            Time stepping scheme: "CN" (Crank-Nicolson), "BDF1", "BDF2"

        Returns
        -------
        DiffusionOperators
            A_lhs, B_rhs, apply_mass callable
        """
        ...

    @abstractmethod
    def apply_diffusion(self, V: torch.Tensor) -> torch.Tensor:
        """
        Compute div(D * grad(V)) directly for explicit methods.

        Parameters
        ----------
        V : torch.Tensor
            Voltage field, shape (n_dof,)

        Returns
        -------
        torch.Tensor
            Laplacian result, shape (n_dof,)
        """
        ...
