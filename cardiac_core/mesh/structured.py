"""
Structured Cartesian Grid

Provides a rectangular grid for FDM, FVM, and LBM discretizations.
Supports domain masks (for irregular geometries), fiber angle fields,
and BoundarySpec for bidomain simulations.

Ref: improvement.md L414-453
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import torch

from .base import Mesh
from .boundary import BoundarySpec, BCType, Edge, EdgeBC


@dataclass
class StructuredGrid(Mesh):
    """
    Structured Cartesian grid for FDM/FVM/LBM.

    Node-centered: nodes at grid intersections (including boundaries).
    Grid has Nx nodes in x and Ny nodes in y, with spacing dx and dy.

    Attributes
    ----------
    Nx : int
        Number of nodes in x-direction
    Ny : int
        Number of nodes in y-direction
    Lx : float
        Domain length in x (cm)
    Ly : float
        Domain length in y (cm)
    dx : float
        Grid spacing in x (cm)
    dy : float
        Grid spacing in y (cm)
    domain_mask : torch.Tensor or None
        Boolean mask (Nx, Ny) — True for active nodes. None = all active.
    fiber_angles : torch.Tensor or None
        Fiber angle field (Nx, Ny) in radians. None = isotropic.
    _device : torch.device
        Device for tensors
    _dtype : torch.dtype
        Data type for float tensors
    """
    Nx: int
    Ny: int
    Lx: float
    Ly: float
    dx: float = 0.0  # computed in __post_init__
    dy: float = 0.0  # computed in __post_init__
    domain_mask: Optional[torch.Tensor] = None
    fiber_angles: Optional[torch.Tensor] = None
    boundary_spec: BoundarySpec = None  # Initialized in __post_init__
    _device: torch.device = None
    _dtype: torch.dtype = torch.float64

    def __post_init__(self):
        if self._device is None:
            self._device = torch.device('cpu')
        if self.boundary_spec is None:
            self.boundary_spec = BoundarySpec.insulated()
        self.dx = self.Lx / (self.Nx - 1)
        self.dy = self.Ly / (self.Ny - 1)

        # Build coordinate tensors
        x_1d = torch.linspace(0, self.Lx, self.Nx, device=self._device, dtype=self._dtype)
        y_1d = torch.linspace(0, self.Ly, self.Ny, device=self._device, dtype=self._dtype)
        xx, yy = torch.meshgrid(x_1d, y_1d, indexing='ij')
        self._xx = xx
        self._yy = yy

        # Flat coordinate arrays
        if self.domain_mask is not None:
            self._flat_x = xx[self.domain_mask]
            self._flat_y = yy[self.domain_mask]
        else:
            self._flat_x = xx.flatten()
            self._flat_y = yy.flatten()

    # === Mesh ABC implementation ===

    @property
    def n_dof(self) -> int:
        if self.domain_mask is not None:
            return int(self.domain_mask.sum().item())
        return self.Nx * self.Ny

    @property
    def coordinates(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._flat_x, self._flat_y

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def to(self, device: torch.device) -> 'StructuredGrid':
        """Move grid to specified device."""
        mask = self.domain_mask.to(device) if self.domain_mask is not None else None
        fibers = self.fiber_angles.to(device) if self.fiber_angles is not None else None
        return StructuredGrid(
            Nx=self.Nx, Ny=self.Ny, Lx=self.Lx, Ly=self.Ly,
            domain_mask=mask, fiber_angles=fibers,
            boundary_spec=self.boundary_spec,
            _device=device, _dtype=self._dtype
        )

    # === Factory methods ===

    @classmethod
    def create_rectangle(
        cls,
        Lx: float,
        Ly: float,
        Nx: int,
        Ny: int,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float64
    ) -> 'StructuredGrid':
        """
        Create a uniform rectangular grid.

        Parameters
        ----------
        Lx, Ly : float
            Domain dimensions (cm)
        Nx, Ny : int
            Number of nodes in each direction
        device : str
            Device for tensors
        dtype : torch.dtype
            Float data type
        """
        return cls(
            Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly,
            _device=torch.device(device), _dtype=dtype
        )

    @classmethod
    def from_mask(
        cls,
        mask: torch.Tensor,
        dx: float,
        dy: float,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float64
    ) -> 'StructuredGrid':
        """
        Create grid from a boolean mask.

        Parameters
        ----------
        mask : torch.Tensor
            Boolean mask of shape (Nx, Ny), True = active
        dx, dy : float
            Grid spacing (cm)
        """
        dev = torch.device(device)
        Nx, Ny = mask.shape
        Lx = dx * (Nx - 1)
        Ly = dy * (Ny - 1)
        return cls(
            Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly,
            domain_mask=mask.to(dev),
            _device=dev, _dtype=dtype
        )

    # === Grid-specific helpers ===

    def flat_to_grid(self, flat: torch.Tensor) -> torch.Tensor:
        """
        Reshape flat DOF array to (Nx, Ny) grid.

        If domain_mask is set, fills masked-out locations with fill_value.
        """
        if self.domain_mask is not None:
            grid = torch.zeros(self.Nx, self.Ny, device=self._device, dtype=flat.dtype)
            grid[self.domain_mask] = flat
            return grid
        return flat.reshape(self.Nx, self.Ny)

    def grid_to_flat(self, grid: torch.Tensor) -> torch.Tensor:
        """Extract flat DOF array from (Nx, Ny) grid."""
        if self.domain_mask is not None:
            return grid[self.domain_mask]
        return grid.flatten()

    @property
    def boundary_mask(self) -> torch.Tensor:
        """Boolean mask (Nx, Ny) for boundary nodes."""
        mask = torch.zeros(self.Nx, self.Ny, device=self._device, dtype=torch.bool)
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
        if self.domain_mask is not None:
            mask = mask & self.domain_mask
        return mask

    @property
    def edge_masks(self) -> Dict[Edge, torch.Tensor]:
        """Per-edge boolean masks (Nx, Ny). Precomputed."""
        masks = {}
        masks[Edge.LEFT] = torch.zeros(self.Nx, self.Ny, dtype=torch.bool, device=self._device)
        masks[Edge.LEFT][0, :] = True
        masks[Edge.RIGHT] = torch.zeros_like(masks[Edge.LEFT])
        masks[Edge.RIGHT][-1, :] = True
        masks[Edge.BOTTOM] = torch.zeros_like(masks[Edge.LEFT])
        masks[Edge.BOTTOM][:, 0] = True
        masks[Edge.TOP] = torch.zeros_like(masks[Edge.LEFT])
        masks[Edge.TOP][:, -1] = True
        return masks

    @property
    def dirichlet_mask_phi_e(self) -> torch.Tensor:
        """Combined mask of all Dirichlet-BC nodes for phi_e. (Nx, Ny) bool."""
        mask = torch.zeros(self.Nx, self.Ny, dtype=torch.bool, device=self._device)
        em = self.edge_masks
        for edge, bc in self.boundary_spec.phi_e.items():
            if bc.bc_type == BCType.DIRICHLET:
                mask |= em[edge]
        return mask

    @property
    def neumann_mask_phi_e(self) -> torch.Tensor:
        """Combined mask of all Neumann-BC nodes for phi_e."""
        return self.boundary_mask & ~self.dirichlet_mask_phi_e

    def __repr__(self) -> str:
        n = self.n_dof
        masked = " (masked)" if self.domain_mask is not None else ""
        return f"StructuredGrid({self.Nx}x{self.Ny}, dx={self.dx:.4f}, dy={self.dy:.4f}, n_dof={n}{masked})"
