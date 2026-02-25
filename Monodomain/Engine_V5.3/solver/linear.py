"""
Linear Solvers for FEM Systems

Provides Preconditioned Conjugate Gradient (PCG) solver with Jacobi
preconditioner for symmetric positive-definite systems arising from
FEM discretization.

The solver is optimized for:
- Sparse COO tensors from FEM assembly
- GPU acceleration via PyTorch
- Warm-start capabilities for time-stepping
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch


@dataclass
class SolverStats:
    """Statistics from PCG solve."""
    converged: bool
    iterations: int
    residual_norm: float
    initial_residual_norm: float


def sparse_mv(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Sparse matrix-vector multiplication.

    Parameters
    ----------
    A : torch.Tensor
        Sparse COO matrix
    x : torch.Tensor
        Dense vector

    Returns
    -------
    torch.Tensor
        A @ x
    """
    if A.is_sparse:
        return torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
    else:
        return A @ x


def extract_diagonal(A: torch.Tensor) -> torch.Tensor:
    """
    Extract diagonal from sparse COO matrix.

    Parameters
    ----------
    A : torch.Tensor
        Sparse COO matrix

    Returns
    -------
    torch.Tensor
        Diagonal elements as dense vector
    """
    if A.is_sparse:
        # Ensure tensor is coalesced before accessing indices
        A_coal = A.coalesce() if not A.is_coalesced() else A
        indices = A_coal.indices()
        values = A_coal.values()
        mask = indices[0] == indices[1]
        diag = torch.zeros(A.shape[0], dtype=A.dtype, device=A.device)
        diag[indices[0, mask]] = values[mask]
        return diag
    else:
        return A.diag()


def pcg_solve(
    A: torch.Tensor,
    b: torch.Tensor,
    x0: Optional[torch.Tensor] = None,
    tol: float = 1e-8,
    max_iter: int = 500,
    M_inv: Optional[torch.Tensor] = None,
    return_stats: bool = False
) -> Tuple[torch.Tensor, Optional[SolverStats]]:
    """
    Preconditioned Conjugate Gradient solver for Ax = b.

    Solves symmetric positive-definite systems using PCG with
    Jacobi (diagonal) preconditioning by default.

    Parameters
    ----------
    A : torch.Tensor
        System matrix (sparse COO, symmetric positive-definite)
    b : torch.Tensor
        Right-hand side vector
    x0 : torch.Tensor, optional
        Initial guess (default: zero)
    tol : float
        Convergence tolerance on relative residual norm
    max_iter : int
        Maximum number of iterations
    M_inv : torch.Tensor, optional
        Preconditioner (inverse of M). If None, uses Jacobi (1/diag(A))
    return_stats : bool
        If True, return solver statistics

    Returns
    -------
    x : torch.Tensor
        Solution vector
    stats : SolverStats, optional
        Solver statistics (if return_stats=True)

    Notes
    -----
    The Jacobi preconditioner M = diag(A) is simple but effective for
    well-conditioned FEM systems. For ill-conditioned problems, consider
    ILU or AMG preconditioners (not implemented here).

    Convergence criterion: ||r|| / ||b|| < tol
    """
    n = b.shape[0]
    device = b.device
    dtype = b.dtype

    # Initial guess
    if x0 is not None:
        x = x0.clone()
    else:
        x = torch.zeros(n, dtype=dtype, device=device)

    # Jacobi preconditioner (default)
    if M_inv is None:
        diag = extract_diagonal(A)
        # Prevent division by zero
        diag = torch.where(diag.abs() < 1e-14, torch.ones_like(diag), diag)
        M_inv = 1.0 / diag

    # Initial residual: r = b - Ax
    r = b - sparse_mv(A, x)

    # Check if already converged
    r_norm = torch.norm(r)
    b_norm = torch.norm(b)
    r0_norm = r_norm.item()

    # Handle zero RHS
    if b_norm < 1e-14:
        if return_stats:
            return x, SolverStats(True, 0, r_norm.item(), r0_norm)
        return x

    # Preconditioned residual: z = M^{-1} r
    z = M_inv * r

    # Search direction
    p = z.clone()

    # r^T z
    rz = torch.dot(r, z)

    converged = False
    k = 0

    for k in range(max_iter):
        # Matrix-vector product
        Ap = sparse_mv(A, p)

        # Step size: alpha = (r^T z) / (p^T A p)
        pAp = torch.dot(p, Ap)

        # Check for breakdown
        if pAp.abs() < 1e-30:
            break

        alpha = rz / pAp

        # Update solution: x = x + alpha * p
        x = x + alpha * p

        # Update residual: r = r - alpha * A p
        r = r - alpha * Ap

        # Check convergence
        r_norm = torch.norm(r)
        if r_norm / b_norm < tol:
            converged = True
            break

        # Preconditioned residual
        z = M_inv * r

        # Beta for conjugate direction
        rz_new = torch.dot(r, z)
        beta = rz_new / rz

        # Update search direction
        p = z + beta * p
        rz = rz_new

    if return_stats:
        stats = SolverStats(
            converged=converged,
            iterations=k + 1,
            residual_norm=r_norm.item(),
            initial_residual_norm=r0_norm
        )
        return x, stats

    return x


def apply_dirichlet_bc(
    A: torch.Tensor,
    b: torch.Tensor,
    boundary_nodes: torch.Tensor,
    boundary_values: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Dirichlet boundary conditions by modifying system.

    Uses the penalty method: set A[i,i] = 1e30 and b[i] = 1e30 * value
    for boundary nodes.

    Parameters
    ----------
    A : torch.Tensor
        System matrix (sparse COO)
    b : torch.Tensor
        Right-hand side vector
    boundary_nodes : torch.Tensor
        Indices of boundary nodes
    boundary_values : torch.Tensor
        Values to impose at boundary nodes

    Returns
    -------
    A_bc : torch.Tensor
        Modified system matrix
    b_bc : torch.Tensor
        Modified RHS vector

    Notes
    -----
    For homogeneous Dirichlet (value=0), this is equivalent to
    zeroing rows/columns and setting diagonal to 1.
    """
    penalty = 1e30

    # Copy b
    b_bc = b.clone()

    # Modify RHS for boundary nodes
    b_bc[boundary_nodes] = penalty * boundary_values

    # For sparse matrices, we need to add penalty to diagonal
    if A.is_sparse:
        # Get existing diagonal
        diag = extract_diagonal(A)

        # Create penalty addition
        n_boundary = len(boundary_nodes)
        penalty_indices = torch.stack([
            boundary_nodes,
            boundary_nodes
        ])
        penalty_values = torch.full(
            (n_boundary,),
            penalty,
            dtype=A.dtype,
            device=A.device
        )

        # Create penalty matrix
        A_penalty = torch.sparse_coo_tensor(
            penalty_indices,
            penalty_values,
            size=A.shape,
            dtype=A.dtype,
            device=A.device
        )

        # Add to original (this adds to existing diagonal entries)
        A_bc = (A + A_penalty).coalesce()
    else:
        A_bc = A.clone()
        A_bc[boundary_nodes, boundary_nodes] += penalty

    return A_bc, b_bc


def cg_solve_dense(
    A: torch.Tensor,
    b: torch.Tensor,
    x0: Optional[torch.Tensor] = None,
    tol: float = 1e-8,
    max_iter: int = 500
) -> torch.Tensor:
    """
    Simple CG solver for dense matrices (testing/reference).

    Parameters
    ----------
    A : torch.Tensor
        Dense system matrix (symmetric positive-definite)
    b : torch.Tensor
        Right-hand side vector
    x0 : torch.Tensor, optional
        Initial guess
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations

    Returns
    -------
    torch.Tensor
        Solution vector
    """
    n = b.shape[0]
    device = b.device
    dtype = b.dtype

    x = x0 if x0 is not None else torch.zeros(n, dtype=dtype, device=device)

    r = b - A @ x
    p = r.clone()
    rs_old = torch.dot(r, r)

    for _ in range(max_iter):
        Ap = A @ p
        alpha = rs_old / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)

        if torch.sqrt(rs_new) < tol:
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x
