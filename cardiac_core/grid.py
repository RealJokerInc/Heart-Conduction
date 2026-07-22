"""
Grid — the single structured-grid geometry descriptor for cardiac_core.

Structured grids are the ONLY geometry standard (decision 2026-06-24: FEM/unstructured dropped —
no ``TriangularMesh``, no flat ``(n_dof,)`` secondary path). ``Grid`` holds ``Nx, Ny, dx, dy`` and an
optional boolean ``mask``, derives ``Lx/Ly/coordinates/n_dof``, and lazily builds the engines'
``StructuredGrid`` when a factory needs it (Phase 4).

Coordinate convention matches the engine ``StructuredGrid`` exactly:
``x_1d = linspace(0, Lx, Nx)``, ``y_1d = linspace(0, Ly, Ny)``, ``meshgrid(..., indexing='ij')`` →
``x[i,j] = x_1d[i]`` (varies along axis 0), ``y[i,j] = y_1d[j]`` (varies along axis 1). So
``x[0,0] == 0`` and ``x[-1,0] == Lx``.
"""

from typing import Optional, Tuple

import torch


class Grid:
    """Structured rectangular grid: ``Nx*Ny`` nodes, spacing ``dx`` (and ``dy``, defaulting to ``dx``).

    Parameters
    ----------
    Nx, Ny : int
        Node counts along x and y (``(Nx, Ny)`` ij convention).
    dx : float
        Grid spacing along x (cm).
    dy : float, optional
        Grid spacing along y (cm). Defaults to ``dx``.
    mask : array-like of bool, optional
        ``(Nx, Ny)`` active-node mask. ``None`` = full domain.
    boundary_mode : str, optional
        Ghost/mirror edge rule for the diffusion stencil + analysis field ops
        (``"face_mirror"`` ≙ scipy ``reflect`` no-flux). Default ``"face_mirror"``.
    device, dtype :
        Tensor device / float dtype (float64 by default).
    """

    def __init__(self, Nx: int, Ny: int, dx: float, dy: Optional[float] = None, *,
                 mask=None, boundary_mode: str = "face_mirror",
                 device: str = "cpu", dtype: torch.dtype = torch.float64):
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.dx = float(dx)
        self.dy = float(dy) if dy is not None else float(dx)
        self.boundary_mode = boundary_mode
        self.device = device
        self.dtype = dtype

        if mask is not None:
            mask = torch.as_tensor(mask, dtype=torch.bool, device=torch.device(device))
            if tuple(mask.shape) != (self.Nx, self.Ny):
                raise ValueError(
                    f"mask shape {tuple(mask.shape)} != (Nx, Ny) = ({self.Nx}, {self.Ny})"
                )
        self.mask = mask
        self._sg = None  # cached engine StructuredGrid (built lazily)

    @property
    def Lx(self) -> float:
        return self.dx * (self.Nx - 1)

    @property
    def Ly(self) -> float:
        return self.dy * (self.Ny - 1)

    @property
    def coordinates(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(x, y)`` meshgrid tensors of shape ``(Nx, Ny)``, ``ij`` indexing.

        Matches the engine ``StructuredGrid`` orientation (``x[-1,0] == Lx``).
        """
        dev = torch.device(self.device)
        x_1d = torch.linspace(0.0, self.Lx, self.Nx, device=dev, dtype=self.dtype)
        y_1d = torch.linspace(0.0, self.Ly, self.Ny, device=dev, dtype=self.dtype)
        xx, yy = torch.meshgrid(x_1d, y_1d, indexing='ij')
        return xx, yy

    @property
    def n_dof(self) -> int:
        """Active node count: ``mask.sum()`` if masked, else ``Nx*Ny``."""
        if self.mask is not None:
            return int(self.mask.sum().item())
        return self.Nx * self.Ny

    def _structured_grid(self):
        """Build (and cache) the shared ``StructuredGrid`` for factory construction."""
        if self._sg is None:
            from .mesh.structured import StructuredGrid
            if self.mask is not None:
                self._sg = StructuredGrid.from_mask(
                    self.mask, self.dx, self.dy, device=self.device, dtype=self.dtype
                )
            else:
                self._sg = StructuredGrid.create_rectangle(
                    self.Lx, self.Ly, self.Nx, self.Ny, device=self.device, dtype=self.dtype
                )
        return self._sg

    def __repr__(self) -> str:
        return (f"Grid(Nx={self.Nx}, Ny={self.Ny}, dx={self.dx}, dy={self.dy}, "
                f"boundary_mode={self.boundary_mode!r}, n_dof={self.n_dof})")
