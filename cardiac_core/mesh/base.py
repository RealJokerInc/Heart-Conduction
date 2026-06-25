"""
Mesh Abstract Base Class

Defines the common interface for all mesh types:
- TriangularMesh (unstructured, FEM)
- StructuredGrid (Cartesian, FDM/FVM/LBM)

The Mesh ABC provides the minimal contract needed by SimulationState
and SpatialDiscretization to work with any mesh type.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import torch


class Mesh(ABC):
    """
    Abstract base class for computational meshes.

    All mesh types must provide:
    - Number of degrees of freedom (nodes or cells)
    - Coordinate arrays for stimulus evaluation and visualization
    - Device and dtype for tensor compatibility
    """

    @property
    @abstractmethod
    def n_dof(self) -> int:
        """Number of degrees of freedom (nodes for FEM, cells for FDM/FVM)."""
        ...

    @property
    @abstractmethod
    def coordinates(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Flat coordinate arrays for all DOFs.

        Returns
        -------
        x : torch.Tensor
            Shape (n_dof,) — x-coordinates
        y : torch.Tensor
            Shape (n_dof,) — y-coordinates
        """
        ...

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Device where mesh tensors are stored."""
        ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """Data type for floating point tensors."""
        ...

    @abstractmethod
    def to(self, device: torch.device) -> 'Mesh':
        """Move mesh to specified device."""
        ...
