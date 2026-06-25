"""
Preconditioned Conjugate Gradient (PCG) Solver

Solves Ax = b for symmetric positive-definite systems using
PCG with Jacobi (diagonal) preconditioning.

Migrated from V5.3 solver/linear.py with lazy workspace allocation.

Ref: V5.3/IMPLEMENTATION.md:L966-1019
"""

from dataclasses import dataclass
from typing import Optional
import torch

from .base import LinearSolver


@dataclass
class SolverStats:
    """Statistics from PCG solve."""
    converged: bool
    iterations: int
    residual_norm: float
    initial_residual_norm: float


def sparse_mv(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Sparse matrix-vector multiplication."""
    if A.is_sparse:
        return torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
    else:
        return A @ x


def extract_diagonal(A: torch.Tensor) -> torch.Tensor:
    """Extract diagonal from sparse COO matrix."""
    if A.is_sparse:
        A_coal = A.coalesce() if not A.is_coalesced() else A
        indices = A_coal.indices()
        values = A_coal.values()
        mask = indices[0] == indices[1]
        diag = torch.zeros(A.shape[0], dtype=A.dtype, device=A.device)
        diag[indices[0, mask]] = values[mask]
        return diag
    else:
        return A.diag()


class PCGSolver(LinearSolver):
    """
    Preconditioned Conjugate Gradient solver with Jacobi preconditioning.

    Uses lazy workspace allocation to avoid allocation per step.
    Supports warm start using previous solution as initial guess.

    Parameters
    ----------
    max_iters : int
        Maximum number of iterations
    tol : float
        Convergence tolerance on relative residual norm
    use_warm_start : bool
        If True, use previous solution as initial guess
    """

    def __init__(
        self,
        max_iters: int = 500,
        tol: float = 1e-8,
        use_warm_start: bool = True
    ):
        self.max_iters = max_iters
        self.tol = tol
        self.use_warm_start = use_warm_start

        # Workspace (lazy-allocated on first call)
        self._r: Optional[torch.Tensor] = None
        self._z: Optional[torch.Tensor] = None
        self._p: Optional[torch.Tensor] = None
        self._Ap: Optional[torch.Tensor] = None
        self._x: Optional[torch.Tensor] = None
        self._M_inv: Optional[torch.Tensor] = None
        self._cached_A_id: Optional[int] = None

        # For warm start
        self._last_solution: Optional[torch.Tensor] = None

        # Stats from last solve
        self.last_stats: Optional[SolverStats] = None

    def _allocate_workspace(
        self,
        n: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> None:
        """Allocate workspace buffers if needed."""
        if self._r is None or self._r.shape[0] != n or self._r.device != device:
            self._r = torch.zeros(n, device=device, dtype=dtype)
            self._z = torch.zeros(n, device=device, dtype=dtype)
            self._p = torch.zeros(n, device=device, dtype=dtype)
            self._Ap = torch.zeros(n, device=device, dtype=dtype)
            self._x = torch.zeros(n, device=device, dtype=dtype)

    def _update_preconditioner(self, A: torch.Tensor) -> None:
        """Update Jacobi preconditioner if A changed."""
        A_id = id(A)
        if self._cached_A_id != A_id:
            diag = extract_diagonal(A)
            # Prevent division by zero
            diag = torch.where(diag.abs() < 1e-14, torch.ones_like(diag), diag)
            self._M_inv = 1.0 / diag
            self._cached_A_id = A_id

    def solve(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        return_stats: bool = False
    ) -> torch.Tensor:
        """
        Solve Ax = b using PCG with Jacobi preconditioning.

        Parameters
        ----------
        A : torch.Tensor
            System matrix (sparse COO, symmetric positive-definite)
        b : torch.Tensor
            Right-hand side vector
        return_stats : bool
            If True, return (x, stats) tuple

        Returns
        -------
        x : torch.Tensor
            Solution vector
        stats : SolverStats, optional
            Solver statistics if return_stats=True
        """
        n = b.shape[0]
        device = b.device
        dtype = b.dtype

        # Allocate workspace
        self._allocate_workspace(n, device, dtype)

        # Update preconditioner
        self._update_preconditioner(A)

        # Initial guess
        x = self._x
        if self.use_warm_start and self._last_solution is not None:
            x.copy_(self._last_solution)
        else:
            x.zero_()

        # r = b - A*x
        r = self._r
        r.copy_(b)
        if x.abs().sum() > 0:  # Only subtract if x is nonzero
            r.sub_(sparse_mv(A, x))

        # Check for convergence
        r_norm = torch.norm(r)
        b_norm = torch.norm(b)
        r0_norm = r_norm.item()

        # Handle zero RHS
        if b_norm < 1e-14:
            self._last_solution = x.clone()
            self.last_stats = SolverStats(True, 0, r_norm.item(), r0_norm)
            if return_stats:
                return x.clone(), self.last_stats
            return x.clone()

        # z = M^{-1} * r
        z = self._z
        z.copy_(self._M_inv * r)

        # p = z
        p = self._p
        p.copy_(z)

        # rz = r^T * z
        rz = torch.dot(r, z)

        converged = False
        k = 0

        for k in range(self.max_iters):
            # Ap = A * p
            Ap = self._Ap
            Ap.copy_(sparse_mv(A, p))

            # alpha = rz / (p^T * Ap)
            pAp = torch.dot(p, Ap)
            if pAp.abs() < 1e-30:
                break
            alpha = rz / pAp

            # x = x + alpha * p
            x.add_(p, alpha=alpha)

            # r = r - alpha * Ap
            r.sub_(Ap, alpha=alpha)

            # Check convergence
            r_norm = torch.norm(r)
            if r_norm / b_norm < self.tol:
                converged = True
                break

            # z = M^{-1} * r
            z.copy_(self._M_inv * r)

            # beta = (r^T * z)_new / (r^T * z)_old
            rz_new = torch.dot(r, z)
            beta = rz_new / rz

            # p = z + beta * p
            p.mul_(beta).add_(z)

            rz = rz_new

        # Store for warm start
        self._last_solution = x.clone()

        # Store stats
        self.last_stats = SolverStats(
            converged=converged,
            iterations=k + 1,
            residual_norm=r_norm.item(),
            initial_residual_norm=r0_norm
        )

        if return_stats:
            return x.clone(), self.last_stats
        return x.clone()

    def reset_warm_start(self) -> None:
        """Clear warm start state."""
        self._last_solution = None
