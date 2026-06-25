"""
Chebyshev Polynomial Linear Solver

Zero-sync polynomial iteration for SPD systems.
No inner products or global reductions during iteration,
making it ideal for GPU execution.

Uses Gershgorin bounds for eigenvalue estimation.
Requires SPD matrix (all diffusion operators are SPD).

Bug fixes from LINEAR_SOLVER_IMPLEMENTATION.md:
  CH-1: Preconditioned Gershgorin bounds for D^{-1}A (was using bounds for A)
  CH-2: Guard theta > 0 (guaranteed for SPD but now asserted)
  CH-3: Clamp rho_new to prevent overflow
  CH-4: Warm start support (x0 parameter in solve())

Ref: Research/03_GPU_Linear:L39-65 (algorithm)
Ref: Research/03_GPU_Linear:L77-106 (Gershgorin bounds)
"""

import torch
from typing import Optional, Tuple

from .base import LinearSolver


def _gershgorin_bounds(A: torch.Tensor, safety_margin: float = 0.1) -> Tuple[float, float]:
    """
    Estimate eigenvalue bounds of A via Gershgorin circle theorem.

    Parameters
    ----------
    A : torch.Tensor
        Sparse COO matrix (SPD)
    safety_margin : float
        Fraction to expand bounds (default 10%)

    Returns
    -------
    lam_min, lam_max : float
        Estimated eigenvalue bounds
    """
    A = A.coalesce()
    indices = A.indices()
    values = A.values()
    n = A.size(0)
    device = A.device
    dtype = A.dtype

    diag = torch.zeros(n, device=device, dtype=dtype)
    off_diag_sum = torch.zeros(n, device=device, dtype=dtype)

    rows = indices[0]
    cols = indices[1]

    diag_mask = rows == cols
    diag.scatter_add_(0, rows[diag_mask], values[diag_mask])

    off_diag_mask = ~diag_mask
    off_diag_sum.scatter_add_(0, rows[off_diag_mask], values[off_diag_mask].abs())

    lam_min_raw = (diag - off_diag_sum).min().item()
    lam_max_raw = (diag + off_diag_sum).max().item()

    lam_min = max(lam_min_raw * (1 - safety_margin), 1e-10)
    lam_max = lam_max_raw * (1 + safety_margin)

    return lam_min, lam_max


def _gershgorin_bounds_preconditioned(
    A: torch.Tensor, diag_inv: torch.Tensor, safety_margin: float = 0.1
) -> Tuple[float, float]:
    """
    Gershgorin bounds for D^{-1}A where D = diag(A).

    For the preconditioned system, each row of D^{-1}A has:
      - Center = 1.0 (diagonal entry a_{ii}/a_{ii})
      - Radius = sum(|a_{ij}/a_{ii}|, j != i)
    """
    A = A.coalesce()
    indices = A.indices()
    values = A.values()
    n = A.size(0)
    device = A.device
    dtype = A.dtype

    rows = indices[0]
    cols = indices[1]

    off_diag = rows != cols
    # |a_{ij}| / a_{ii} for each off-diagonal entry
    scaled_abs = values[off_diag].abs() * diag_inv[rows[off_diag]]

    radii = torch.zeros(n, device=device, dtype=dtype)
    radii.scatter_add_(0, rows[off_diag], scaled_abs)

    # Centers are all 1.0 for D^{-1}A
    lam_min_raw = (1.0 - radii).min().item()
    lam_max_raw = (1.0 + radii).max().item()

    lam_min = max(lam_min_raw * (1 - safety_margin), 1e-10)
    lam_max = lam_max_raw * (1 + safety_margin)

    return lam_min, lam_max


class ChebyshevSolver(LinearSolver):
    """
    Chebyshev polynomial linear solver.

    Uses 3-term recurrence for polynomial acceleration without
    inner products. Ideal for GPU: no sync points during iteration.

    The algorithm requires eigenvalue bounds [lam_min, lam_max].
    These are estimated via Gershgorin circles on first solve,
    then cached for subsequent solves with the same matrix.

    Parameters
    ----------
    max_iters : int
        Fixed number of iterations (no convergence check during iteration)
    tol : float
        Not used during iteration (fixed count), but stored for API
    safety_margin : float
        Gershgorin bounds expansion factor (default 10%)
    use_jacobi_precond : bool
        Apply Jacobi (diagonal) preconditioning (default True)
    """

    def __init__(
        self,
        max_iters: int = 50,
        tol: float = 1e-8,
        safety_margin: float = 0.1,
        use_jacobi_precond: bool = True
    ):
        self.max_iters = max_iters
        self.tol = tol
        self.safety_margin = safety_margin
        self.use_jacobi_precond = use_jacobi_precond

        self._lam_min: Optional[float] = None
        self._lam_max: Optional[float] = None
        self._A_id: Optional[int] = None

        self._x: Optional[torch.Tensor] = None
        self._r: Optional[torch.Tensor] = None
        self._z: Optional[torch.Tensor] = None
        self._d: Optional[torch.Tensor] = None
        self._diag_inv: Optional[torch.Tensor] = None

    def _ensure_workspace(self, n: int, device: torch.device, dtype: torch.dtype) -> None:
        """Allocate workspace tensors if needed."""
        if self._x is None or self._x.shape[0] != n:
            self._x = torch.zeros(n, device=device, dtype=dtype)
            self._r = torch.zeros(n, device=device, dtype=dtype)
            self._z = torch.zeros(n, device=device, dtype=dtype)
            self._d = torch.zeros(n, device=device, dtype=dtype)

    def _extract_diag_inv(self, A: torch.Tensor) -> torch.Tensor:
        """Extract inverse diagonal for Jacobi preconditioning."""
        A = A.coalesce()
        indices = A.indices()
        values = A.values()
        n = A.size(0)

        diag = torch.zeros(n, device=A.device, dtype=A.dtype)
        diag_mask = indices[0] == indices[1]
        diag.scatter_add_(0, indices[0][diag_mask], values[diag_mask])

        # Guard against zero diagonal
        diag = diag.clamp(min=1e-15)
        return 1.0 / diag

    # Sentinel: if _A_id is this object, manual bounds are set — skip estimation
    _MANUAL_BOUNDS = object()

    def _estimate_eigenvalues(self, A: torch.Tensor) -> None:
        """Estimate eigenvalue bounds if not cached for this matrix."""
        if self._A_id is self._MANUAL_BOUNDS:
            # Manual bounds set via set_eigenvalue_bounds(); skip auto-estimation
            # but still update preconditioner if needed
            if self.use_jacobi_precond and self._diag_inv is None:
                self._diag_inv = self._extract_diag_inv(A)
            return

        A_id = A.values().data_ptr() if A.is_sparse else A.data_ptr()

        if self._A_id != A_id:
            self._diag_inv = self._extract_diag_inv(A) if self.use_jacobi_precond else None

            # CH-1 FIX: Use preconditioned bounds when Jacobi is enabled
            if self.use_jacobi_precond:
                self._lam_min, self._lam_max = _gershgorin_bounds_preconditioned(
                    A, self._diag_inv, self.safety_margin)
            else:
                self._lam_min, self._lam_max = _gershgorin_bounds(
                    A, self.safety_margin)

            self._A_id = A_id

    def solve(self, A: torch.Tensor, b: torch.Tensor,
              x0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Solve Ax = b using Chebyshev iteration.

        Parameters
        ----------
        A : torch.Tensor
            Sparse SPD system matrix
        b : torch.Tensor
            Right-hand side vector
        x0 : torch.Tensor, optional
            Initial guess for warm start (CH-4 fix)

        Returns
        -------
        x : torch.Tensor
            Approximate solution
        """
        n = b.shape[0]
        device = b.device
        dtype = b.dtype

        self._ensure_workspace(n, device, dtype)
        self._estimate_eigenvalues(A)

        lam_min = self._lam_min
        lam_max = self._lam_max

        # Chebyshev parameters
        theta = (lam_max + lam_min) / 2.0
        delta = (lam_max - lam_min) / 2.0

        # CH-2 FIX: Guard theta and delta
        assert theta > 0, f"Chebyshev theta={theta} <= 0 (not SPD?)"
        assert delta > 0, f"Chebyshev delta={delta} <= 0 (single eigenvalue?)"

        sigma = theta / delta

        x = self._x
        r = self._r
        z = self._z
        d = self._d

        # CH-4 FIX: Warm start support
        if x0 is not None:
            x.copy_(x0)
        else:
            x.zero_()

        # r = b - A*x
        if x0 is not None:
            Ax = torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
            r.copy_(b - Ax)
        else:
            r.copy_(b)

        # z = M^{-1} * r
        if self.use_jacobi_precond:
            z.copy_(r * self._diag_inv)
        else:
            z.copy_(r)

        # First iteration
        rho = 1.0 / sigma
        d.copy_(z / theta)
        x.add_(d)

        # Main iteration loop
        for _ in range(1, self.max_iters):
            Ax = torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
            r.copy_(b - Ax)

            if self.use_jacobi_precond:
                z.copy_(r * self._diag_inv)
            else:
                z.copy_(r)

            # CH-3 FIX: Guard against overflow
            denom = 2.0 * sigma - rho
            if abs(denom) < 1e-15:
                break  # Degenerate — stop iterating
            rho_new = 1.0 / denom
            rho_new = max(min(rho_new, 1e15), -1e15)  # Clamp

            d.mul_(rho_new * rho).add_(z, alpha=2.0 * rho_new / delta)
            x.add_(d)
            rho = rho_new

        return x.clone()

    def set_eigenvalue_bounds(self, lam_min: float, lam_max: float) -> None:
        """
        Manually set eigenvalue bounds.

        Useful when bounds are known from problem structure or previous
        estimation (avoids Gershgorin cost).
        """
        self._lam_min = lam_min
        self._lam_max = lam_max
        self._A_id = self._MANUAL_BOUNDS  # Skip auto-estimation on next solve
