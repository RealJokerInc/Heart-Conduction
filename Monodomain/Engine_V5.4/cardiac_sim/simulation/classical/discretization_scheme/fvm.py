"""
Finite Volume Method Discretization

Cell-centered FVM on structured Cartesian grids.
Uses Two-Point Flux Approximation (TPFA) with harmonic mean
for interface conductivity at material discontinuities.

Node-centered layout: FVM cells are centered on grid nodes.
Each cell has volume dx*dy. Flux is computed at faces between cells.

Supports anisotropic D via D_field=(D_xx, D_yy) and masked grids.

Ref: improvement.md:L901-935
Ref: Research/02_openCARP:L200-250
Ref: Research/00_Research_Summary:L90 (harmonic mean)
"""

from typing import Optional, Tuple, Union
import torch
import numpy as np

from .base import SpatialDiscretization, MassType, DiffusionOperators, sparse_mv
from ....tissue_builder.mesh.structured import StructuredGrid


def _speye(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create sparse identity matrix."""
    idx = torch.arange(n, device=device)
    indices = torch.stack([idx, idx])
    values = torch.ones(n, device=device, dtype=dtype)
    return torch.sparse_coo_tensor(indices, values, size=(n, n)).coalesce()


class FVMDiscretization(SpatialDiscretization):
    """
    Finite Volume Method spatial discretization.

    Cell-centered on a StructuredGrid. Each node is a cell center
    with volume dx*dy. Face fluxes use harmonic mean for interface
    conductivity, giving correct behavior at material boundaries.

    The monodomain equation is:
        χ·Cm·∂V/∂t = ∇·(D·∇V)  (diffusion only, ionic handled separately)

    This discretizes to (per cell):
        χ·Cm·Vol·dV/dt = F·V

    where F is the flux operator with diffusion D built in.

    Supports masked grids (domain_mask on StructuredGrid): only active nodes
    are included in the system. Matrix size = n_active × n_active.

    Parameters
    ----------
    grid : StructuredGrid
        The computational grid
    D : float
        Scalar diffusion coefficient (cm^2/ms) for isotropic case.
        Ignored if D_field is provided.
    chi : float
        Surface-to-volume ratio (cm^-1). Default 1400.
    Cm : float
        Membrane capacitance (µF/cm²). Default 1.0.
    D_field : torch.Tensor or tuple of (D_xx, D_yy), optional
        Single tensor (Nx, Ny): isotropic per-node D.
        Tuple of two tensors (Nx, Ny): anisotropic (D_xx for x-faces, D_yy for y-faces).
    """

    def __init__(
        self,
        grid: StructuredGrid,
        D: float = 0.001,
        chi: float = 1400.0,
        Cm: float = 1.0,
        D_field: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        self._grid = grid
        self._nx = grid.Nx
        self._ny = grid.Ny
        self._dx = grid.dx
        self._dy = grid.dy
        self._device = grid.device
        self._dtype = grid.dtype
        self._n_dof = grid.n_dof
        self._chi = chi
        self._Cm = Cm

        x, y = grid.coordinates
        self._x = x
        self._y = y

        # Per-node diffusion fields (Nx, Ny)
        if D_field is not None:
            if isinstance(D_field, tuple):
                self._Dxx_grid, self._Dyy_grid = D_field
            else:
                self._Dxx_grid = D_field
                self._Dyy_grid = D_field
        else:
            self._Dxx_grid = torch.full(
                (self._nx, self._ny), D,
                device=self._device, dtype=self._dtype
            )
            self._Dyy_grid = torch.full(
                (self._nx, self._ny), D,
                device=self._device, dtype=self._dtype
            )

        # Cell volumes (uniform grid, active cells only)
        self._volumes = torch.full(
            (self._n_dof,), self._dx * self._dy,
            device=self._device, dtype=self._dtype
        )

        # Scaled mass: χ·Cm·Vol
        self._scaled_mass = self._chi * self._Cm * self._volumes

        # Build flux operator
        self.F = self._build_flux_matrix(grid.domain_mask)

    @property
    def n_dof(self) -> int:
        return self._n_dof

    @property
    def coordinates(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._x, self._y

    @property
    def mass_type(self) -> MassType:
        return MassType.DIAGONAL

    @property
    def grid(self) -> StructuredGrid:
        return self._grid

    @property
    def volumes(self) -> torch.Tensor:
        return self._volumes

    def get_diffusion_operators(self, dt: float, scheme: str) -> DiffusionOperators:
        """
        Build operators for implicit time stepping.

        For FVM: χ·Cm·Vol·dV/dt = F*V
        - CN: (χ·Cm·Vol - 0.5*dt*F)*V^{n+1} = (χ·Cm·Vol + 0.5*dt*F)*V^n
        - BDF1: (χ·Cm·Vol - dt*F)*V^{n+1} = χ·Cm·Vol*V^n

        This matches the FEM formulation where M ~ χ·Cm·Vol and K ~ D.
        """
        scheme = scheme.upper()
        n = self._n_dof

        # Build diagonal scaled mass matrix: χ·Cm·Vol
        idx = torch.arange(n, device=self._device)
        indices = torch.stack([idx, idx])
        M = torch.sparse_coo_tensor(
            indices, self._scaled_mass,
            size=(n, n), dtype=self._dtype, device=self._device
        ).coalesce()

        if scheme == "CN":
            A = (M - 0.5 * dt * self.F).coalesce()
            B = (M + 0.5 * dt * self.F).coalesce()
        elif scheme == "BDF1":
            A = (M - dt * self.F).coalesce()
            B = M
        elif scheme == "BDF2":
            A = (3.0 * M - 2.0 * dt * self.F).coalesce()
            B = (4.0 * M).coalesce()
        else:
            raise ValueError(f"Unknown scheme: {scheme}")

        scaled_mass = self._scaled_mass

        def apply_mass(f: torch.Tensor) -> torch.Tensor:
            return f * scaled_mass

        return DiffusionOperators(A_lhs=A, B_rhs=B, apply_mass=apply_mass)

    def apply_diffusion(self, V: torch.Tensor) -> torch.Tensor:
        """
        Compute F*V / (χ·Cm·Vol) for explicit time stepping.

        Returns dV/dt from diffusion: dV/dt = (1/(χ·Cm)) * ∇·(D·∇V)
        """
        return sparse_mv(self.F, V) / self._scaled_mass

    # === Internal: Flux Matrix Assembly ===

    def _build_flux_matrix(
        self,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Assemble the flux matrix F such that F*V gives the net flux into each cell.

        Face conductivity uses harmonic mean:
        D_face = 2*D_left*D_right / (D_left + D_right)

        x-faces use D_xx, y-faces use D_yy (anisotropic support).

        Neumann BC: no flux through domain boundaries (boundary faces omitted).
        Inactive neighbors are skipped (zero-flux at active/inactive boundary).
        """
        nx, ny = self._nx, self._ny
        dx, dy = self._dx, self._dy
        device = self._device
        dtype = self._dtype

        # Convert to numpy for fast Python-level access
        Dxx_np = self._Dxx_grid.detach().cpu().numpy()
        Dyy_np = self._Dyy_grid.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy() if mask is not None else None

        # Build active-node index mapping
        if mask_np is not None:
            active_map = np.full((nx, ny), -1, dtype=np.int64)
            count = 0
            for i in range(nx):
                for j in range(ny):
                    if mask_np[i, j]:
                        active_map[i, j] = count
                        count += 1
            N = count
        else:
            N = nx * ny

        def _is_active(i, j):
            if i < 0 or i >= nx or j < 0 or j >= ny:
                return False
            if mask_np is not None:
                return bool(mask_np[i, j])
            return True

        def _idx(i, j):
            if mask_np is not None:
                return int(active_map[i, j])
            return i * ny + j

        rows_list = []
        cols_list = []
        vals_list = []

        def _add(r, c, v):
            rows_list.append(r)
            cols_list.append(c)
            vals_list.append(v)

        def _harm(a, b):
            s = a + b
            return 2.0 * a * b / s if s > 0 else 0.0

        for i in range(nx):
            for j in range(ny):
                if mask_np is not None and not mask_np[i, j]:
                    continue

                k = _idx(i, j)
                Dxx_c = float(Dxx_np[i, j])
                Dyy_c = float(Dyy_np[i, j])
                center = 0.0

                # East face: between (i,j) and (i+1,j) — uses D_xx
                if _is_active(i + 1, j):
                    D_face = _harm(Dxx_c, float(Dxx_np[i + 1, j]))
                    coeff = D_face * dy / dx
                    _add(k, _idx(i + 1, j), coeff)
                    center -= coeff

                # West face: between (i,j) and (i-1,j) — uses D_xx
                if _is_active(i - 1, j):
                    D_face = _harm(Dxx_c, float(Dxx_np[i - 1, j]))
                    coeff = D_face * dy / dx
                    _add(k, _idx(i - 1, j), coeff)
                    center -= coeff

                # North face: between (i,j) and (i,j+1) — uses D_yy
                if _is_active(i, j + 1):
                    D_face = _harm(Dyy_c, float(Dyy_np[i, j + 1]))
                    coeff = D_face * dx / dy
                    _add(k, _idx(i, j + 1), coeff)
                    center -= coeff

                # South face: between (i,j) and (i,j-1) — uses D_yy
                if _is_active(i, j - 1):
                    D_face = _harm(Dyy_c, float(Dyy_np[i, j - 1]))
                    coeff = D_face * dx / dy
                    _add(k, _idx(i, j - 1), coeff)
                    center -= coeff

                # Center = -sum(off-diagonals) => conservation
                _add(k, k, center)

        rows = torch.tensor(rows_list, dtype=torch.long, device=device)
        cols = torch.tensor(cols_list, dtype=torch.long, device=device)
        vals = torch.tensor(vals_list, dtype=dtype, device=device)

        F = torch.sparse_coo_tensor(
            torch.stack([rows, cols]), vals,
            size=(N, N), dtype=dtype, device=device
        ).coalesce()

        return F
