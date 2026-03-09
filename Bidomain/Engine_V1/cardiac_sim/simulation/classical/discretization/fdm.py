"""
Bidomain FDM Discretization

9-point anisotropic stencil on structured Cartesian grids.
Builds two symmetric Laplacians (L_i, L_e) using face-based stencils.

DESIGN DECISION — Symmetric Face-Based Stencil (differs from V5.4)

V5.4 uses ghost-node mirror for Neumann BCs: at boundary node i=N-1,
the ghost at i+1 mirrors to i-1, doubling the connection weight.
This gives L[N-1, N-2] = 2w but L[N-2, N-1] = w — asymmetric.

For monodomain, asymmetry is harmless because A = chi_Cm/dt * I - theta*L
is dominated by the identity (ratio ~10^6). But the bidomain elliptic
operator A_ellip = -(L_i + L_e) has NO identity term, so asymmetry
makes the matrix non-SPD and PCG fails.

Solution: face-based stencil where each interior face contributes
equally to both adjacent nodes. Out-of-domain faces are skipped
(zero flux = Neumann). This gives symmetric L with zero row sum.

Trade-off: at boundary nodes, the stiffness form gives d²u/dx² / 2
(half the strong-form value). This is physically correct in the
variational sense (half control volume at boundary), and cancels
in the bidomain because both LHS and RHS use the same stiffness form.

Interior accuracy: O(h^2). Global PDE convergence: O(h^2).

Formulation B (diffusivity-based): L contains D = sigma/(chi*Cm).
Parabolic operator uses 1/dt (NOT chi*Cm/dt). Chi does not appear
anywhere in this module. Cm is stored only for source term scaling.

Dirichlet enforcement is NOT baked into L_i or L_e. It is applied
only in get_elliptic_operator() via symmetric row+column elimination.

Ref: improvement.md L737-812 (FDM concrete impl)
Ref: improvement.md L481-527 (FDM Stencil at Boundary Nodes)
Ref: V5.4 fdm.py (original ghost-node approach)
"""

from typing import Optional, Tuple
import torch
import numpy as np

from .base import BidomainSpatialDiscretization
from ....tissue_builder.mesh.structured import StructuredGrid
from ....tissue_builder.mesh.boundary import BoundarySpec, BCType, Edge
from ....tissue_builder.tissue.conductivity import BidomainConductivity


def _speye(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create sparse identity matrix."""
    idx = torch.arange(n, device=device)
    indices = torch.stack([idx, idx])
    values = torch.ones(n, device=device, dtype=dtype)
    return torch.sparse_coo_tensor(indices, values, size=(n, n)).coalesce()


def _sparse_mv(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Sparse matrix-vector multiplication."""
    if A.is_sparse:
        return torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
    return A @ x


class BidomainFDMDiscretization(BidomainSpatialDiscretization):
    """
    FDM bidomain discretization on structured grid.

    Builds two 9-pt stencil Laplacians L_i and L_e from BidomainConductivity.
    Supports scalar D, per-node D_field, and fiber-based anisotropy.

    The Laplacian contains D (diffusivity = sigma/(chi*Cm)). Chi and Cm do
    NOT appear anywhere in the operators. The parabolic operator uses 1/dt
    (Formulation B). Cm is stored for source term scaling (R = -I_ion/Cm).

    Parameters
    ----------
    grid : StructuredGrid
        Computational grid with boundary_spec
    conductivity : BidomainConductivity
        Paired intra/extracellular conductivity
    Cm : float
        Membrane capacitance (uF/cm^2). Default 1.0. Used only for
        source term scaling. D values in conductivity already contain
        chi*Cm scaling (D = sigma/(chi*Cm)).
    chi : float
        Deprecated. Ignored. Chi is already absorbed into D values.
        Kept for backward compatibility — will be removed in V2.
    """

    def __init__(
        self,
        grid: StructuredGrid,
        conductivity: BidomainConductivity,
        Cm: float = 1.0,
        chi: float = None,
    ):
        if chi is not None and chi != 1.0:
            import warnings
            warnings.warn(
                f"chi={chi} is ignored. D values already contain chi*Cm scaling. "
                "Pass chi via ConductivityConfig (D = sigma/(chi*Cm)) instead.",
                DeprecationWarning, stacklevel=2
            )
        self._grid = grid
        self._conductivity = conductivity
        self._nx = grid.Nx
        self._ny = grid.Ny
        self._dx = grid.dx
        self._dy = grid.dy
        self._device = grid.device
        self._dtype = grid.dtype
        self._n_dof = grid.n_dof
        self._Cm = Cm

        # Coordinate arrays
        x, y = grid.coordinates
        self._x = x
        self._y = y

        # Build diffusion tensor components for each domain
        Dxx_i, Dxy_i, Dyy_i = self._get_D_components(
            conductivity.D_i, conductivity.D_i_field,
            conductivity.D_i_fiber, conductivity.D_i_cross,
            conductivity.theta
        )
        Dxx_e, Dxy_e, Dyy_e = self._get_D_components(
            conductivity.D_e, conductivity.D_e_field,
            conductivity.D_e_fiber, conductivity.D_e_cross,
            conductivity.theta
        )

        # Build two Laplacians using symmetric face-based stencils.
        # Both use the same stencil construction (Neumann-like face-based).
        # Dirichlet BCs are enforced in get_elliptic_operator() only.
        self._L_i_mat = self._build_laplacian(Dxx_i, Dxy_i, Dyy_i)
        self._L_e_mat = self._build_laplacian(Dxx_e, Dxy_e, Dyy_e)
        self._L_ie_mat = None  # Lazy: built on first use
        self._A_ellip = None   # Lazy: built on first use

    # === ABC implementation ===

    @property
    def n_dof(self) -> int:
        return self._n_dof

    @property
    def grid(self) -> StructuredGrid:
        return self._grid

    @property
    def coordinates(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._x, self._y

    def apply_L_i(self, V: torch.Tensor) -> torch.Tensor:
        return _sparse_mv(self._L_i_mat, V)

    def apply_L_e(self, V: torch.Tensor) -> torch.Tensor:
        return _sparse_mv(self._L_e_mat, V)

    def apply_L_ie(self, V: torch.Tensor) -> torch.Tensor:
        if self._L_ie_mat is None:
            self._L_ie_mat = (self._L_i_mat + self._L_e_mat).coalesce()
        return _sparse_mv(self._L_ie_mat, V)

    @property
    def Cm(self) -> float:
        """Membrane capacitance (uF/cm^2) for source term scaling."""
        return self._Cm

    @property
    def conductivity(self) -> 'BidomainConductivity':
        """Conductivity configuration."""
        return self._conductivity

    def get_parabolic_operators(
        self, dt: float, theta: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build (A_para, B_para) for the decoupled parabolic Vm solve.

        Formulation B (diffusivity-based): L contains D = sigma/(chi*Cm).
        The mass term uses 1/dt (NOT chi*Cm/dt).

        A_para * Vm^{n+1} = B_para * Vm^n + L_i * phi_e^n

        A_para = 1/dt * I - theta * L_i
        B_para = 1/dt * I + (1-theta) * L_i
        """
        n = self._n_dof
        I = _speye(n, self._device, self._dtype)
        A_para = (1.0 / dt * I - theta * self._L_i_mat).coalesce()
        B_para = (1.0 / dt * I + (1 - theta) * self._L_i_mat).coalesce()
        return A_para, B_para

    def get_elliptic_operator(self) -> torch.Tensor:
        """
        Build A_ellip = -(L_i + L_e) for: A_ellip * phi_e = L_i * Vm.

        For Dirichlet boundary nodes of phi_e, enforces identity rows
        (symmetric elimination) so that A_ellip remains SPD.
        """
        if self._A_ellip is not None:
            return self._A_ellip

        if self._L_ie_mat is None:
            self._L_ie_mat = (self._L_i_mat + self._L_e_mat).coalesce()

        A = (-self._L_ie_mat).coalesce()

        # Apply Dirichlet enforcement for phi_e boundary nodes
        A = self._enforce_dirichlet(A)

        self._A_ellip = A
        return self._A_ellip

    @property
    def L_i(self) -> torch.Tensor:
        return self._L_i_mat

    @property
    def L_e(self) -> torch.Tensor:
        return self._L_e_mat

    # === Internal: Diffusion tensor construction ===

    def _get_D_components(
        self,
        D_scalar: float,
        D_field: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        D_fiber: Optional[float],
        D_cross: Optional[float],
        theta: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get (Dxx, Dxy, Dyy) tensor components."""
        nx, ny = self._nx, self._ny
        device, dtype = self._device, self._dtype

        if D_field is not None:
            return D_field

        if D_fiber is not None and D_cross is not None and theta is not None:
            # Rotate fiber conductivity to Cartesian
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            Dxx = D_fiber * cos_t**2 + D_cross * sin_t**2
            Dxy = (D_fiber - D_cross) * cos_t * sin_t
            Dyy = D_fiber * sin_t**2 + D_cross * cos_t**2
            return Dxx, Dxy, Dyy

        # Scalar isotropic
        Dxx = torch.full((nx, ny), D_scalar, device=device, dtype=dtype)
        Dxy = torch.zeros(nx, ny, device=device, dtype=dtype)
        Dyy = torch.full((nx, ny), D_scalar, device=device, dtype=dtype)
        return Dxx, Dxy, Dyy

    # === Internal: Dirichlet enforcement ===

    def _enforce_dirichlet(self, A: torch.Tensor) -> torch.Tensor:
        """
        Symmetric Dirichlet enforcement on sparse matrix.

        For each Dirichlet boundary node of phi_e:
        - Zero out row and column
        - Set diagonal to 1
        """
        dmask = self._grid.dirichlet_mask_phi_e  # (Nx, Ny) bool
        if not dmask.any():
            return A

        # Get flat indices of Dirichlet nodes
        dir_flat = self._grid.grid_to_flat(
            dmask.to(self._dtype)
        ).nonzero(as_tuple=False).squeeze(1)

        if dir_flat.numel() == 0:
            return A

        A = A.coalesce()
        indices = A.indices()  # (2, nnz)
        values = A.values()    # (nnz,)

        # Remove all entries in rows OR columns of Dirichlet nodes
        row_ok = ~torch.isin(indices[0], dir_flat)
        col_ok = ~torch.isin(indices[1], dir_flat)
        keep = row_ok & col_ok

        new_indices = indices[:, keep]
        new_values = values[keep]

        # Add identity entries for Dirichlet nodes
        id_indices = torch.stack([dir_flat, dir_flat])
        id_values = torch.ones(dir_flat.shape[0], device=self._device, dtype=self._dtype)

        all_indices = torch.cat([new_indices, id_indices], dim=1)
        all_values = torch.cat([new_values, id_values])

        return torch.sparse_coo_tensor(
            all_indices, all_values, A.shape,
            device=self._device, dtype=self._dtype
        ).coalesce()

    # === Internal: Sparse Laplacian Assembly ===

    def _build_laplacian(
        self,
        Dxx: torch.Tensor,
        Dxy: torch.Tensor,
        Dyy: torch.Tensor,
    ) -> torch.Tensor:
        """
        Assemble the 9-point Laplacian as a symmetric sparse matrix.

        Uses face-based stencil: each interior face connects two nodes
        symmetrically. Out-of-domain faces are skipped (zero flux).
        This produces a symmetric negative semi-definite matrix with
        zero row sum — equivalent to Neumann BC everywhere.

        Dirichlet BC enforcement is applied separately in
        get_elliptic_operator() via _enforce_dirichlet().

        Cardinal directions use harmonic mean at interfaces for
        correct handling of heterogeneous diffusion.

        Parameters
        ----------
        Dxx, Dxy, Dyy : torch.Tensor
            Diffusion tensor components, shape (Nx, Ny).
        """
        nx, ny = self._nx, self._ny
        dx, dy = self._dx, self._dy
        device = self._device
        dtype = self._dtype
        mask = self._grid.domain_mask

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
        # Factor of 2 from div(D·∇V) = Dxx·V_xx + 2·Dxy·V_xy + Dyy·V_yy.
        # The cross-derivative d²V/dxdy ≈ (V_NE - V_NW - V_SE + V_SW)/(4·dx·dy),
        # multiplied by the factor 2 gives cxy = 2/(4·dx·dy) = 1/(2·dx·dy).
        cxy = 1.0 / (2.0 * dx * dy)

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
                # Face-based symmetric stencil: each face contributes
                # equally to both adjacent nodes. Out-of-domain faces
                # are skipped (Neumann: zero flux) or handled via
                # Dirichlet row elimination above.

                # East (i+1, j)
                if _is_active(i + 1, j):
                    D_face = _harm(d_xx, float(dxx[i + 1, j]))
                    w = D_face * cx
                    center -= w
                    _add(k, _idx(i + 1, j), w)
                # else: outside domain → skip (no flux through boundary)

                # West (i-1, j)
                if _is_active(i - 1, j):
                    D_face = _harm(d_xx, float(dxx[i - 1, j]))
                    w = D_face * cx
                    center -= w
                    _add(k, _idx(i - 1, j), w)

                # North (i, j+1)
                if _is_active(i, j + 1):
                    D_face = _harm(d_yy, float(dyy[i, j + 1]))
                    w = D_face * cy
                    center -= w
                    _add(k, _idx(i, j + 1), w)

                # South (i, j-1)
                if _is_active(i, j - 1):
                    D_face = _harm(d_yy, float(dyy[i, j - 1]))
                    w = D_face * cy
                    center -= w
                    _add(k, _idx(i, j - 1), w)

                # --- Diagonal directions (9-point, anisotropic) ---
                # Cross-derivative: 2*Dxy * d²V/dxdy
                # d²V/dxdy ≈ (V_NE - V_SE - V_NW + V_SW) / (4*dx*dy)
                # Combined with cxy = 1/(2*dx*dy), the weights are:
                #   NE: +Dxy*cxy, NW: -Dxy*cxy, SE: -Dxy*cxy, SW: +Dxy*cxy

                # NE (i+1, j+1)
                if _is_active(i + 1, j + 1):
                    w = d_xy * cxy
                    _add(k, _idx(i + 1, j + 1), w)
                    center -= w

                # NW (i-1, j+1)
                if _is_active(i - 1, j + 1):
                    w = -d_xy * cxy
                    _add(k, _idx(i - 1, j + 1), w)
                    center -= w

                # SE (i+1, j-1)
                if _is_active(i + 1, j - 1):
                    w = -d_xy * cxy
                    _add(k, _idx(i + 1, j - 1), w)
                    center -= w

                # SW (i-1, j-1)
                if _is_active(i - 1, j - 1):
                    w = d_xy * cxy
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

    def __repr__(self) -> str:
        bs = self._grid.boundary_spec
        bc_str = "insulated" if bs.phi_e_has_null_space else "bath-coupled"
        return (f"BidomainFDMDiscretization("
                f"{self._nx}x{self._ny}, "
                f"dx={self._dx:.4f}, dy={self._dy:.4f}, "
                f"n_dof={self._n_dof}, "
                f"phi_e_bc={bc_str})")
