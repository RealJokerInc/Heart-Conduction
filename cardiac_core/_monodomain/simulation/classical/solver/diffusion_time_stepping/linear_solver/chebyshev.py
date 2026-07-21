"""
Chebyshev Polynomial Linear Solver

Zero-sync polynomial iteration for SPD systems.
No inner products or global reductions during iteration,
making it ideal for GPU execution.

Uses Gershgorin bounds for eigenvalue estimation.
Requires SPD matrix (all diffusion operators are SPD).

Ref: Research/03_GPU_Linear:L39-65 (algorithm)
Ref: Research/03_GPU_Linear:L77-106 (Gershgorin bounds)
"""

import torch
from typing import Optional, Tuple

from .base import LinearSolver, warn_nonconvergence


def _gershgorin_bounds(A: torch.Tensor, safety_margin: float = 0.1) -> Tuple[float, float]:
    """
    Estimate eigenvalue bounds of A via Gershgorin circle theorem.

    For Jacobi-preconditioned system, discs are centered at ~1.
    This implementation works directly on the sparse matrix.

    Parameters
    ----------
    A : torch.Tensor
        Sparse COO matrix
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

    # Extract diagonal and compute off-diagonal row sums
    diag = torch.zeros(n, device=device, dtype=dtype)
    off_diag_sum = torch.zeros(n, device=device, dtype=dtype)

    rows = indices[0]
    cols = indices[1]

    # Diagonal entries
    diag_mask = rows == cols
    diag.scatter_add_(0, rows[diag_mask], values[diag_mask])

    # Off-diagonal absolute sums
    off_diag_mask = ~diag_mask
    off_diag_sum.scatter_add_(0, rows[off_diag_mask], values[off_diag_mask].abs())

    # Gershgorin radii (relative to center = diag)
    # Each eigenvalue lies in [diag_i - r_i, diag_i + r_i]
    centers = diag
    radii = off_diag_sum

    # Conservative bounds
    lam_min_raw = (centers - radii).min().item()
    lam_max_raw = (centers + radii).max().item()

    # Apply safety margin
    lam_min = max(lam_min_raw * (1 - safety_margin), 1e-10)
    lam_max = lam_max_raw * (1 + safety_margin)

    return lam_min, lam_max


def _gershgorin_bounds_preconditioned(
    A: torch.Tensor, diag_inv: torch.Tensor, safety_margin: float = 0.1
) -> Tuple[float, float]:
    """Gershgorin bounds for the PRECONDITIONED operator ``D^{-1}A`` (``D = diag(A)``).

    The Chebyshev iteration actually acts on ``D^{-1}A`` when Jacobi preconditioning is on,
    so its eigenvalue interval must be measured there — not on the raw ``A`` (whose spectrum
    is ~``chi*Cm`` times wider, producing a polynomial tuned for the wrong interval that barely
    damps the error, i.e. up to ~46% silent error at large diffusion number). Each row of
    ``D^{-1}A`` is centered at 1 with radius ``sum_j!=i |a_ij| / a_ii``.

    Ported from the bidomain ChebyshevSolver ("CH-1 fix"); closes CODE_AUDIT 2026-07-02 M1.
    """
    A = A.coalesce()
    indices = A.indices()
    values = A.values()
    n = A.size(0)

    rows = indices[0]
    cols = indices[1]

    off_diag = rows != cols
    scaled_abs = values[off_diag].abs() * diag_inv[rows[off_diag]]

    radii = torch.zeros(n, device=A.device, dtype=A.dtype)
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

    The algorithm requires eigenvalue bounds [λ_min, λ_max].
    These are estimated via Gershgorin circles on first solve,
    then cached for subsequent solves with the same matrix.

    Parameters
    ----------
    max_iters : int
        Fixed number of iterations (no convergence check during iteration)
    tol : float
        Not used during iteration (fixed count), but stored for API consistency
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

        # Cached eigenvalue bounds
        self._lam_min: Optional[float] = None
        self._lam_max: Optional[float] = None
        self._A_id: Optional[int] = None  # For cache invalidation

        # Workspace (lazy allocation)
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

        # Guard against a zero diagonal
        diag = diag.clamp(min=1e-15)
        return 1.0 / diag

    # Sentinel: if _A_id is this object, manual bounds are set — skip estimation
    _MANUAL_BOUNDS = object()

    def _estimate_eigenvalues(self, A: torch.Tensor) -> None:
        """Estimate eigenvalue bounds if not cached for this matrix."""
        if self._A_id is self._MANUAL_BOUNDS:
            # Manual bounds set via set_eigenvalue_bounds(); skip auto-estimation
            # but still build the preconditioner so Jacobi stays active.
            if self.use_jacobi_precond and self._diag_inv is None:
                self._diag_inv = self._extract_diag_inv(A)
            return

        # Use data_ptr for cache invalidation (works for both sparse and dense)
        A_id = A.values().data_ptr() if A.is_sparse else A.data_ptr()

        if self._A_id != A_id:
            self._diag_inv = self._extract_diag_inv(A) if self.use_jacobi_precond else None

            # CH-1 FIX (CODE_AUDIT 2026-07-02 M1): bound the PRECONDITIONED operator D^{-1}A
            # that the iteration actually acts on, not the raw A.
            if self.use_jacobi_precond:
                self._lam_min, self._lam_max = _gershgorin_bounds_preconditioned(
                    A, self._diag_inv, self.safety_margin)
            else:
                self._lam_min, self._lam_max = _gershgorin_bounds(A, self.safety_margin)

            self._A_id = A_id

    def solve(self, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Solve Ax = b using Chebyshev iteration.

        Parameters
        ----------
        A : torch.Tensor
            Sparse SPD system matrix
        b : torch.Tensor
            Right-hand side vector

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
        sigma = theta / delta

        # Initialize x = 0
        x = self._x
        x.zero_()

        # r = b - A*x = b (since x=0)
        r = self._r
        r.copy_(b)

        # z = precond(r)
        z = self._z
        if self.use_jacobi_precond:
            z.copy_(r * self._diag_inv)
        else:
            z.copy_(r)

        # First iteration (special case)
        rho = 1.0 / sigma
        d = self._d
        d.copy_(z / theta)
        x.add_(d)

        # Main iteration loop
        for _ in range(1, self.max_iters):
            # r = b - A*x
            Ax = torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
            r.copy_(b - Ax)

            # z = precond(r)
            if self.use_jacobi_precond:
                z.copy_(r * self._diag_inv)
            else:
                z.copy_(r)

            # Chebyshev update
            rho_new = 1.0 / (2.0 * sigma - rho)
            d.mul_(rho_new * rho).add_(z, alpha=2.0 * rho_new / delta)
            x.add_(d)
            rho = rho_new

        # Chebyshev runs a FIXED iteration count with no in-loop convergence test, so a
        # too-loose count / bad bounds silently returns a wrong answer. Do ONE end-of-solve
        # residual check per run (an extra SpMV + sync): the operator A is fixed across a
        # run and Chebyshev's relative residual bound is b-independent, so if the first solve
        # converges every solve does — skip the check (and its sync) for the rest of the run.
        # The per-run reset (CardiacSimulation._reset_solver_diagnostics) re-arms it each run.
        if self.tol is not None and not getattr(self, '_nonconv_warned', False):
            Ax = torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
            b_norm = torch.norm(b)
            r_norm = torch.norm(b - Ax)
            if b_norm > 0 and (r_norm / b_norm) > self.tol:
                warn_nonconvergence('Chebyshev', self.max_iters,
                                    r_norm.item(), b_norm.item(), self.tol,
                                    reason="fixed iteration count")
            self._nonconv_warned = True   # checked once this run (converged or not)

        return x.clone()

    def set_eigenvalue_bounds(self, lam_min: float, lam_max: float) -> None:
        """
        Manually set eigenvalue bounds.

        Useful when bounds are known from problem structure or previous
        estimation (avoids Gershgorin cost).

        Parameters
        ----------
        lam_min, lam_max : float
            Eigenvalue bounds
        """
        self._lam_min = lam_min
        self._lam_max = lam_max
        # Mark manual so the next solve keeps these bounds AND still builds the Jacobi
        # preconditioner (a plain None would let the next solve clobber the bounds and
        # silently disable preconditioning).
        self._A_id = self._MANUAL_BOUNDS
