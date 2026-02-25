"""
Finite Difference Method Discretization

9-point anisotropic stencil on structured Cartesian grids.
Reduces to standard 5-point stencil when Dxy=0 (isotropic or axis-aligned).
Neumann (no-flux) BC via modified boundary stencils.
Cardinal directions use harmonic mean at interfaces for correct scar/heterogeneity handling.

Ref: improvement.md:L848-899
Ref: Research/01_FDM (stencil coefficients, Neumann BC, harmonic mean)
"""

from typing import Optional, Tuple
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


class FDMDiscretization(SpatialDiscretization):
    """
    Finite Difference Method spatial discretization.

    Uses 9-point stencil for anisotropic diffusion, reducing to 5-point
    for isotropic case. Node-centered on a StructuredGrid.

    The monodomain equation is:
        χ·Cm·∂V/∂t = ∇·(D·∇V)  (diffusion only, ionic handled separately)

    This discretizes to:
        χ·Cm·dV/dt = L·V

    where L is the Laplacian operator with diffusion D built in.

    Cardinal stencil uses harmonic mean at interfaces:
        D_face(i+1/2,j) = 2·Dxx(i,j)·Dxx(i+1,j) / (Dxx(i,j) + Dxx(i+1,j))

    This ensures zero flux at D=0 boundaries (scar, background).
    For uniform D, harmonic mean = D, so results are identical.

    Supports masked grids (domain_mask on StructuredGrid): only active nodes
    are included in the system. Matrix size = n_active × n_active.

    Parameters
    ----------
    grid : StructuredGrid
        The computational grid (may have domain_mask for irregular domains)
    D : float
        Scalar diffusion coefficient (cm^2/ms) for isotropic case.
        Ignored if D_field is provided.
    chi : float
        Surface-to-volume ratio (cm^-1). Default 1400.
    Cm : float
        Membrane capacitance (µF/cm²). Default 1.0.
    D_field : tuple of (Dxx, Dxy, Dyy), optional
        Tensor fields, each shape (Nx, Ny), for anisotropic/heterogeneous diffusion.
    """

    def __init__(
        self,
        grid: StructuredGrid,
        D: float = 0.001,
        chi: float = 1400.0,
        Cm: float = 1.0,
        D_field: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
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

        # Coordinate arrays (active nodes only if masked)
        x, y = grid.coordinates
        self._x = x
        self._y = y

        # Diffusion tensor components (full grid)
        if D_field is not None:
            Dxx, Dxy, Dyy = D_field
        else:
            Dxx = torch.full((self._nx, self._ny), D, device=self._device, dtype=self._dtype)
            Dxy = torch.zeros(self._nx, self._ny, device=self._device, dtype=self._dtype)
            Dyy = torch.full((self._nx, self._ny), D, device=self._device, dtype=self._dtype)

        # Build sparse Laplacian (contains D, but NOT chi*Cm)
        self.L = self._build_laplacian(Dxx, Dxy, Dyy, grid.domain_mask)

    @property
    def n_dof(self) -> int:
        return self._n_dof

    @property
    def coordinates(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._x, self._y

    @property
    def mass_type(self) -> MassType:
        return MassType.IDENTITY

    @property
    def grid(self) -> StructuredGrid:
        return self._grid

    @property
    def nx(self) -> int:
        return self._nx

    @property
    def ny(self) -> int:
        return self._ny

    def get_diffusion_operators(self, dt: float, scheme: str) -> DiffusionOperators:
        """
        Build operators for implicit time stepping.

        For FDM: χ·Cm·dV/dt = L*V
        - CN:   (χ·Cm·I - 0.5*dt*L)*V^{n+1} = (χ·Cm·I + 0.5*dt*L)*V^n
        - BDF1: (χ·Cm·I - dt*L)*V^{n+1} = χ·Cm*V^n

        This matches the FEM formulation where M ~ χ·Cm and K ~ D.
        """
        scheme = scheme.upper()
        n = self._n_dof
        I = _speye(n, self._device, self._dtype)
        chi_Cm = self._chi * self._Cm

        if scheme == "CN":
            A = (chi_Cm * I - 0.5 * dt * self.L).coalesce()
            B = (chi_Cm * I + 0.5 * dt * self.L).coalesce()
        elif scheme == "BDF1":
            A = (chi_Cm * I - dt * self.L).coalesce()
            B = chi_Cm * I
        elif scheme == "BDF2":
            A = (3.0 * chi_Cm * I - 2.0 * dt * self.L).coalesce()
            B = (4.0 * chi_Cm * I).coalesce()
        else:
            raise ValueError(f"Unknown scheme: {scheme}")

        def apply_mass(f: torch.Tensor) -> torch.Tensor:
            return chi_Cm * f

        return DiffusionOperators(A_lhs=A, B_rhs=B, apply_mass=apply_mass)

    def apply_diffusion(self, V: torch.Tensor) -> torch.Tensor:
        """
        Compute (1/(χ·Cm)) * L*V for explicit time stepping.

        Returns dV/dt from diffusion: dV/dt = (1/(χ·Cm)) * ∇·(D·∇V)
        """
        return sparse_mv(self.L, V) / (self._chi * self._Cm)

    # === Internal: Sparse Laplacian Assembly ===

    def _build_laplacian(
        self,
        Dxx: torch.Tensor,
        Dxy: torch.Tensor,
        Dyy: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Assemble the 9-point Laplacian as a sparse matrix.

        Cardinal directions use harmonic mean at interfaces:
            D_face = 2·D_L·D_R / (D_L + D_R)
        This ensures zero flux at D=0 interfaces (scar/background).

        Neumann BC at rectangle boundary via ghost-node elimination.
        Active/inactive boundary within domain: skip (zero-flux).

        If mask is provided, only active nodes are assembled. Matrix size
        is n_active × n_active. Inactive neighbors are skipped.

        Parameters
        ----------
        Dxx, Dxy, Dyy : torch.Tensor
            Diffusion tensor components, shape (Nx, Ny).
        mask : torch.Tensor or None
            Boolean mask (Nx, Ny). True = active. None = all active.
        """
        nx, ny = self._nx, self._ny
        dx, dy = self._dx, self._dy
        device = self._device
        dtype = self._dtype

        # Convert to numpy for fast Python-level access
        dxx = Dxx.detach().cpu().numpy()
        dxy = Dxy.detach().cpu().numpy()
        dyy = Dyy.detach().cpu().numpy()
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

        cx = 1.0 / (dx * dx)
        cy = 1.0 / (dy * dy)
        cxy = 1.0 / (4.0 * dx * dy)

        for i in range(nx):
            for j in range(ny):
                if mask_np is not None and not mask_np[i, j]:
                    continue

                k = _idx(i, j)
                d_xx = float(dxx[i, j])
                d_xy = float(dxy[i, j])
                d_yy = float(dyy[i, j])

                center = 0.0

                # --- Cardinal directions with harmonic mean ---
                # Ghost node Neumann at rectangle boundary.
                # Skip at active/inactive boundary (zero-flux).

                # East (i+1, j)
                if _is_active(i + 1, j):
                    D_face = _harm(d_xx, float(dxx[i + 1, j]))
                    w = D_face * cx
                    center -= w
                    _add(k, _idx(i + 1, j), w)
                elif i + 1 >= nx:
                    # Rectangle boundary: ghost V[nx,j] = V[nx-2,j]
                    # Mirror is (i-1, j)
                    if _is_active(i - 1, j):
                        D_face = _harm(d_xx, float(dxx[i - 1, j]))
                        w = D_face * cx
                        center -= w
                        _add(k, _idx(i - 1, j), w)
                    # else: mirror doesn't exist or inactive → skip

                # West (i-1, j)
                if _is_active(i - 1, j):
                    D_face = _harm(d_xx, float(dxx[i - 1, j]))
                    w = D_face * cx
                    center -= w
                    _add(k, _idx(i - 1, j), w)
                elif i - 1 < 0:
                    # Rectangle boundary: ghost V[-1,j] = V[1,j]
                    # Mirror is (i+1, j)
                    if _is_active(i + 1, j):
                        D_face = _harm(d_xx, float(dxx[i + 1, j]))
                        w = D_face * cx
                        center -= w
                        _add(k, _idx(i + 1, j), w)

                # North (i, j+1)
                if _is_active(i, j + 1):
                    D_face = _harm(d_yy, float(dyy[i, j + 1]))
                    w = D_face * cy
                    center -= w
                    _add(k, _idx(i, j + 1), w)
                elif j + 1 >= ny:
                    if _is_active(i, j - 1):
                        D_face = _harm(d_yy, float(dyy[i, j - 1]))
                        w = D_face * cy
                        center -= w
                        _add(k, _idx(i, j - 1), w)

                # South (i, j-1)
                if _is_active(i, j - 1):
                    D_face = _harm(d_yy, float(dyy[i, j - 1]))
                    w = D_face * cy
                    center -= w
                    _add(k, _idx(i, j - 1), w)
                elif j - 1 < 0:
                    if _is_active(i, j + 1):
                        D_face = _harm(d_yy, float(dyy[i, j + 1]))
                        w = D_face * cy
                        center -= w
                        _add(k, _idx(i, j + 1), w)

                # --- Diagonal directions (9-point, anisotropic) ---
                # Use center Dxy. Skip inactive diagonal neighbors.
                # At rectangle boundary, diagonal ghosts are omitted
                # (acceptable: Dxy is a small correction term).

                # NE (i+1, j+1)
                if _is_active(i + 1, j + 1):
                    w = -d_xy * cxy
                    _add(k, _idx(i + 1, j + 1), w)
                    center -= w

                # NW (i-1, j+1)
                if _is_active(i - 1, j + 1):
                    w = d_xy * cxy
                    _add(k, _idx(i - 1, j + 1), w)
                    center -= w

                # SE (i+1, j-1)
                if _is_active(i + 1, j - 1):
                    w = d_xy * cxy
                    _add(k, _idx(i + 1, j - 1), w)
                    center -= w

                # SW (i-1, j-1)
                if _is_active(i - 1, j - 1):
                    w = -d_xy * cxy
                    _add(k, _idx(i - 1, j - 1), w)
                    center -= w

                _add(k, k, center)

        rows = torch.tensor(rows_list, dtype=torch.long, device=device)
        cols = torch.tensor(cols_list, dtype=torch.long, device=device)
        vals = torch.tensor(vals_list, dtype=dtype, device=device)

        L = torch.sparse_coo_tensor(
            torch.stack([rows, cols]), vals,
            size=(N, N), dtype=dtype, device=device
        ).coalesce()

        return L
