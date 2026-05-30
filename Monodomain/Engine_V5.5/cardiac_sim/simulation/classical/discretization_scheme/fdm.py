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
    boundary_mode : str
        Discrete Neumann ghost choice at the rectangle wall. Options:
          - 'face_mirror' (DEFAULT, 2026-04-29): ghost = boundary cell itself
            (V[i,-1]=V[i,0]). Wall placed at y=-h/2 (face-centered). Off-grid
            neighbor flux is identically zero for any V; no entry written.
            L_y at j=0 = (V[i,1]-V[i,0]).  Genuine no-flux Neumann.
          - 'node_mirror_existing' (LEGACY pre-2026-04-29): ghost = sub-edge cell
            (V[i,-1]=V[i,1]). Wall placed AT the boundary node; cardinal stencil
            writes mirror off-diagonal that combines with the interior cardinal
            entry to give 2w after coalesce. L_y at j=0 = 2*(V[i,1]-V[i,0]).
            Amplifies any column-wise gradient at the wall by 2x — root cause of
            storage-tank "camel-toe" / "crescent" boundary artifacts. Kept for
            bit-exact reproduction of pre-flip experiments only.
          - 'zero_pad': ghost = 0 (no Neumann; Dirichlet-to-zero outside).
            Off-grid contributes -w to diagonal only; matrix stays SPD.
            L_y at j=0 = (V[i,1]-V[i,0]) - V[i,0].
          - 'rest_pad': ghost = pad_value (Dirichlet-to-constant outside).
            Same matrix as zero_pad, but the constant is subtracted from V
            before applying L. L_y at j=0 = (V[i,1]-V[i,0]) - (V[i,0]-pad_value).
            For cardiac use, set pad_value = ionic_model.V_rest so the wall
            is silent at rest, but the boundary still gets clamped during the
            AP (peak V at wall ~ 27 mV vs face_mirror ~ 54 mV under TTP06).
            Source-term plumbing means apply_diffusion works out of the box;
            CN/BDF would need extra wiring (not done).
    pad_value : float
        Constant ghost value used by 'rest_pad' (mV). Ignored otherwise.
    """

    # Order: default first, then legacy, then non-Neumann modes.
    # face_mirror_iso (added 2026-04-30): diagonal-aware reflection — for diagonal
    #   off-grid pipes, mirror only the off-grid axis (ghost(i+di,-1)=V[i+di,0]).
    #   For cardinal4 stencil, degenerates to face_mirror (no diagonals).
    #   For moore8_uniform/moore8_iso, eliminates the boundary deficit in
    #   y-uniform fields (LBM bounce-back analog).
    BOUNDARY_MODES = ('face_mirror', 'face_mirror_iso', 'node_mirror_existing',
                      'zero_pad', 'rest_pad')

    # Stencil options (added 2026-04-30):
    #   cardinal4      — 5-point cardinal Laplacian + Dxy cross-derivative diagonals
    #                    (existing legacy behavior; full anisotropic D support)
    #   moore8_uniform — 8-neighbour uniform weights w_card=w_diag=1/(3·h²)
    #                    (isotropic D only; raises NotImplementedError if Dxy != 0)
    #   moore8_iso     — Patra-Kaluza isotropic 9-pt: w_card=4/(6·h²), w_diag=1/(6·h²)
    #                    (4th-order accurate; isotropic D only)
    STENCILS = ('cardinal4', 'moore8_uniform', 'moore8_iso')

    def __init__(
        self,
        grid: StructuredGrid,
        D: float = 0.001,
        chi: float = 1400.0,
        Cm: float = 1.0,
        D_field: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        boundary_mode: str = 'face_mirror',
        pad_value: float = 0.0,
        stencil: str = 'cardinal4',
    ):
        if boundary_mode not in self.BOUNDARY_MODES:
            raise ValueError(
                f"boundary_mode must be one of {self.BOUNDARY_MODES}, "
                f"got {boundary_mode!r}"
            )
        if stencil not in self.STENCILS:
            raise ValueError(
                f"stencil must be one of {self.STENCILS}, got {stencil!r}"
            )
        self._boundary_mode = boundary_mode
        self._stencil = stencil
        self._pad_value = float(pad_value)
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
    def Cm(self) -> float:
        """Membrane capacitance (uF/cm^2) — single source of truth, shared with reaction."""
        return self._Cm

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

        For 'rest_pad' mode, the ghost value is `pad_value` (typically V_rest).
        Equivalent to applying zero_pad to the shifted field U = V - pad_value:
        constants are in the null space of the interior stencil, and at the
        boundary cell's row in L_zero_pad, (L * const_c)[k] = -w_total * c,
        so apply_diffusion(V) - apply_diffusion(const) = (L * (V - const))/χ·Cm.
        """
        if self._boundary_mode == 'rest_pad' and self._pad_value != 0.0:
            V_shifted = V - self._pad_value
            return sparse_mv(self.L, V_shifted) / (self._chi * self._Cm)
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
        Top-level dispatch — delegates to per-stencil builder.

        cardinal4      -> _build_laplacian_cardinal (5-pt + Dxy diagonals;
                                                      full anisotropic D support)
        moore8_uniform -> _build_laplacian_moore8 with weighting='uniform'
        moore8_iso     -> _build_laplacian_moore8 with weighting='iso'

        Moore-8 stencils require isotropic scalar D and dx == dy.
        """
        if self._stencil == 'cardinal4':
            return self._build_laplacian_cardinal(Dxx, Dxy, Dyy, mask)

        # Moore-8 stencils — validate isotropic D and square grid first.
        # Predicate for "no anisotropy": Dxy is None, OR every entry is 0.
        # Default-constructed Dxy is zeros(nx, ny) when D_field=None, so the
        # magnitude check is the real protection; the None check is belt-and-
        # suspenders.
        has_anisotropy = (
            Dxy is not None
            and torch.as_tensor(Dxy).abs().max().item() > 0.0
        )
        if has_anisotropy:
            raise NotImplementedError(
                f"Moore-8 stencils currently support isotropic scalar D only "
                f"(stencil={self._stencil!r}); for anisotropic Dxy, use "
                f"stencil='cardinal4'."
            )
        if abs(self._dx - self._dy) > 1e-12:
            raise NotImplementedError(
                f"Moore-8 stencils require dx == dy "
                f"(got dx={self._dx}, dy={self._dy}); use stencil='cardinal4' "
                f"for non-square grids."
            )
        weighting = 'uniform' if self._stencil == 'moore8_uniform' else 'iso'
        return self._build_laplacian_moore8(Dxx, Dyy, mask, weighting=weighting)

    def _build_laplacian_moore8(
        self,
        Dxx: torch.Tensor,
        Dyy: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weighting: str = 'uniform',
    ) -> torch.Tensor:
        """
        Assemble the 9-point Laplacian using a Moore-8 (8-neighbour) stencil,
        with either uniform or Patra-Kaluza isotropic 4:1 weights.

        weighting='uniform':
            All 8 directions weight 1/(3·h²). y-uniform interior recovers
            (V_E + V_W − 2·V_C)/h² (matches continuum). Boundary deficit
            with face_mirror BC: 1/3 (boundary cells lose 1 inflow + 1 outflow
            diagonal pipe each).
        weighting='iso':
            Patra-Kaluza isotropic 9-pt: ∇²V ≈ (1/6h²)·[4·cards + diags − 20·V_C].
            Cardinals weight 4/(6·h²), diagonals 1/(6·h²). The 1/6 prefactor IS
            the canonical normalisation — without it, D_eff = 6k violates the
            2D-explicit CFL limit of 0.25 and produces grid-scale mosaic
            instability (see IDEALOG.md "Bug fix — iso weights need 1/6
            normalisation" for the failure we hit on John's tanks).
            Boundary deficit with face_mirror BC: 1/6 (smaller than uniform
            because cardinals are weighted more heavily).

        Boundary modes:
            face_mirror      : ghost = self for ALL off-grid (cardinal AND
                               diagonal). Off-grid pipes contribute 0. This
                               is the John-equivalent — boundary cells genuinely
                               have fewer effective neighbours, deficit is REAL.
            face_mirror_iso  : ghost = self for cardinals (same as face_mirror).
                               For diagonals: mirror only the off-grid axis to
                               the in-grid cell at the boundary row/col. Gives
                               ZERO deficit in y-uniform fields when paired
                               with iso weights — LBM bounce-back analog.
            node_mirror_*    : geometrically reflect the off-grid index back
                               to in-grid (across the wall plane y=0).
            zero_pad         : ghost = 0 → flux = -w·V_self (diagonal only).
            rest_pad         : same matrix as zero_pad; constant handled in
                               apply_diffusion via V-shift.

        ASSUMES: dx == dy and isotropic D (Dxx == Dyy). Validated by the
        dispatcher before this method is called.
        """
        nx, ny = self._nx, self._ny
        h = self._dx
        h2 = h * h
        device = self._device
        dtype = self._dtype

        if weighting == 'uniform':
            w_card_base = 1.0 / (3.0 * h2)
            w_diag_base = 1.0 / (3.0 * h2)
        elif weighting == 'iso':
            # Patra-Kaluza 4:1 weights with mandatory 1/6 normalisation.
            w_card_base = 4.0 / (6.0 * h2)
            w_diag_base = 1.0 / (6.0 * h2)
        else:
            raise ValueError(f"unknown weighting: {weighting!r}")

        # Convert to numpy for fast Python-level access.
        # For isotropic D, Dxx == Dyy; we use Dxx as the single field.
        dxx = Dxx.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy() if mask is not None else None

        # Build active-node index mapping (same as cardinal).
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

        # 8 Moore-neighbour offsets.
        MOORE_8 = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)
                   if (di, dj) != (0, 0)]

        for i in range(nx):
            for j in range(ny):
                if mask_np is not None and not mask_np[i, j]:
                    continue

                k = _idx(i, j)
                d_self = float(dxx[i, j])
                center = 0.0

                for di, dj in MOORE_8:
                    is_cardinal = (di == 0) ^ (dj == 0)
                    w_base = w_card_base if is_cardinal else w_diag_base
                    ni, nj = i + di, j + dj

                    if _is_active(ni, nj):
                        # In-grid neighbour: standard harmonic-mean averaging.
                        D_face = _harm(d_self, float(dxx[ni, nj]))
                        w = D_face * w_base
                        center -= w
                        _add(k, _idx(ni, nj), w)
                    else:
                        # Off-grid neighbour — boundary mode dispatch.
                        bm = self._boundary_mode
                        if bm == 'face_mirror':
                            # ghost = self -> flux = 0. No contribution.
                            # Faithful John-equivalent: cell genuinely has
                            # fewer effective neighbours.
                            pass
                        elif bm == 'face_mirror_iso':
                            if is_cardinal:
                                # Cardinal off-grid: same as face_mirror.
                                pass
                            else:
                                # Diagonal off-grid: mirror only the off-grid
                                # axis to the in-grid cell at the boundary row.
                                # E.g., NE at (i+1, -1) -> ghost = V[i+1, 0].
                                ni_m = ni if 0 <= ni < nx else i
                                nj_m = nj if 0 <= nj < ny else j
                                if (ni_m, nj_m) == (i, j):
                                    # Corner: both axes off-grid -> ghost = self.
                                    pass
                                elif _is_active(ni_m, nj_m):
                                    D_face = _harm(d_self,
                                                   float(dxx[ni_m, nj_m]))
                                    w = D_face * w_base
                                    center -= w
                                    _add(k, _idx(ni_m, nj_m), w)
                        elif bm == 'node_mirror_existing':
                            # Reflect across the wall plane(s):
                            # x-off-grid -> ni_m = i - di (mirror x).
                            # y-off-grid -> nj_m = j - dj (mirror y).
                            # Corner (both off-grid) -> mirror both.
                            ni_m = ni if 0 <= ni < nx else i - di
                            nj_m = nj if 0 <= nj < ny else j - dj
                            if (ni_m, nj_m) == (i, j):
                                pass  # shouldn't happen for MOORE_8 offsets
                            elif _is_active(ni_m, nj_m):
                                D_face = _harm(d_self,
                                               float(dxx[ni_m, nj_m]))
                                w = D_face * w_base
                                center -= w
                                _add(k, _idx(ni_m, nj_m), w)
                        elif bm in ('zero_pad', 'rest_pad'):
                            # ghost = const -> diagonal-only modification.
                            # Use cell-local D since there's no "neighbour" cell.
                            w = d_self * w_base
                            center -= w
                        else:
                            raise ValueError(
                                f"Moore-8 builder: unhandled boundary_mode {bm!r}"
                            )

                _add(k, k, center)

        rows = torch.tensor(rows_list, dtype=torch.long, device=device)
        cols = torch.tensor(cols_list, dtype=torch.long, device=device)
        vals = torch.tensor(vals_list, dtype=dtype, device=device)

        L = torch.sparse_coo_tensor(
            torch.stack([rows, cols]), vals,
            size=(N, N), dtype=dtype, device=device
        ).coalesce()

        return L

    def _build_laplacian_cardinal(
        self,
        Dxx: torch.Tensor,
        Dxy: torch.Tensor,
        Dyy: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Assemble the 9-point Laplacian as a sparse matrix (cardinal-4 stencil
        + optional Dxy cross-derivative diagonals).

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
                    # Rectangle boundary: BC choice controls ghost value.
                    if self._boundary_mode == 'node_mirror_existing':
                        # ghost = V[i-1,j]; combines with cardinal-west to 2w.
                        if _is_active(i - 1, j):
                            D_face = _harm(d_xx, float(dxx[i - 1, j]))
                            w = D_face * cx
                            center -= w
                            _add(k, _idx(i - 1, j), w)
                    elif self._boundary_mode == 'face_mirror':
                        # ghost = V[i,j]; flux = D(V_C - V_C)/h^2 = 0. Skip.
                        pass
                    elif self._boundary_mode == 'face_mirror_iso':
                        # cardinal4 has no diagonals -> face_mirror_iso degenerates
                        # to face_mirror. Explicit pass to avoid silent fallthrough.
                        pass
                    elif self._boundary_mode in ('zero_pad', 'rest_pad'):
                        # ghost = const; matrix gets -w on diagonal. The const
                        # itself is handled in apply_diffusion via V-shift.
                        # No mirror neighbor -> no harmonic mean; use cell D.
                        w = d_xx * cx
                        center -= w

                # West (i-1, j)
                if _is_active(i - 1, j):
                    D_face = _harm(d_xx, float(dxx[i - 1, j]))
                    w = D_face * cx
                    center -= w
                    _add(k, _idx(i - 1, j), w)
                elif i - 1 < 0:
                    if self._boundary_mode == 'node_mirror_existing':
                        if _is_active(i + 1, j):
                            D_face = _harm(d_xx, float(dxx[i + 1, j]))
                            w = D_face * cx
                            center -= w
                            _add(k, _idx(i + 1, j), w)
                    elif self._boundary_mode == 'face_mirror':
                        pass
                    elif self._boundary_mode == 'face_mirror_iso':
                        pass
                    elif self._boundary_mode in ('zero_pad', 'rest_pad'):
                        w = d_xx * cx
                        center -= w

                # North (i, j+1)
                if _is_active(i, j + 1):
                    D_face = _harm(d_yy, float(dyy[i, j + 1]))
                    w = D_face * cy
                    center -= w
                    _add(k, _idx(i, j + 1), w)
                elif j + 1 >= ny:
                    if self._boundary_mode == 'node_mirror_existing':
                        if _is_active(i, j - 1):
                            D_face = _harm(d_yy, float(dyy[i, j - 1]))
                            w = D_face * cy
                            center -= w
                            _add(k, _idx(i, j - 1), w)
                    elif self._boundary_mode == 'face_mirror':
                        pass
                    elif self._boundary_mode == 'face_mirror_iso':
                        pass
                    elif self._boundary_mode in ('zero_pad', 'rest_pad'):
                        w = d_yy * cy
                        center -= w

                # South (i, j-1)
                if _is_active(i, j - 1):
                    D_face = _harm(d_yy, float(dyy[i, j - 1]))
                    w = D_face * cy
                    center -= w
                    _add(k, _idx(i, j - 1), w)
                elif j - 1 < 0:
                    if self._boundary_mode == 'node_mirror_existing':
                        if _is_active(i, j + 1):
                            D_face = _harm(d_yy, float(dyy[i, j + 1]))
                            w = D_face * cy
                            center -= w
                            _add(k, _idx(i, j + 1), w)
                    elif self._boundary_mode == 'face_mirror':
                        pass
                    elif self._boundary_mode == 'face_mirror_iso':
                        pass
                    elif self._boundary_mode in ('zero_pad', 'rest_pad'):
                        w = d_yy * cy
                        center -= w

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
