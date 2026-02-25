"""
Bidomain Block Preconditioners for Cardiac Simulation Engine V5.4

This module provides reference implementations for block preconditioners used
to solve the 2×2 bidomain linear system efficiently. The bidomain system is
typically ill-conditioned, particularly when spatial resolution is fine.

Block system structure:
    [ A11  A12 ] [ u1 ]     [ f1 ]
    [ A21  A22 ] [ u2 ]  =  [ f2 ]

where:
    A11 = M/dt + θ*Ki (parabolic part, u1 = Vm)
    A12 = θ*Ki (coupling)
    A21 = Ki (coupling)
    A22 = -(Ki + Ke) (elliptic part, u2 = φe, singular)

References:
    [1] Pennacchio & Simoncini "Efficient Algebraic Solution of the First-Order
        Regularization Problems in Cardiac Electrical Activity" (2012)
    [2] Murillo & Cai "Block Preconditioners for Saddle Point Systems with a
        Penalty Parameter" (2014)
    [3] Schöberl et al. "Efficient Solvers for Nonlinear Time-Dependent PDE
        Systems" (2009)
    [4] Saad "Iterative Methods for Sparse Linear Systems" (2003)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Callable, Dict, Any
from abc import ABC, abstractmethod
import numpy as np


# ============================================================================
# Abstract Base Classes
# ============================================================================

class BlockPreconditioner(ABC):
    """Base class for block preconditioners."""

    @abstractmethod
    def apply(self, b: torch.Tensor) -> torch.Tensor:
        """Apply preconditioner: solve M @ x = b approximately."""
        pass

    @abstractmethod
    def apply_transpose(self, b: torch.Tensor) -> torch.Tensor:
        """Apply transposed preconditioner."""
        pass


class IterativeSolver(ABC):
    """Base class for iterative linear solvers."""

    @abstractmethod
    def solve(
        self,
        A: Callable[[torch.Tensor], torch.Tensor],
        b: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
        tol: float = 1e-6,
        maxiter: int = 1000
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Solve A @ x = b iteratively."""
        pass


# ============================================================================
# Utility Functions
# ============================================================================

def sparse_matvec(A: torch.sparse.FloatTensor, x: torch.Tensor) -> torch.Tensor:
    """Efficient sparse matrix-vector product."""
    return torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)


def form_block_matrix(
    A11: torch.sparse.FloatTensor,
    A12: torch.sparse.FloatTensor,
    A21: torch.sparse.FloatTensor,
    A22: torch.sparse.FloatTensor,
    device: torch.device = torch.device('cpu')
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Form a block matrix multiplication function.

    Args:
        A11, A12, A21, A22: Sparse block matrices
        device: torch device

    Returns:
        Function that computes block matrix-vector products
    """
    def matvec(x: torch.Tensor) -> torch.Tensor:
        n = A11.shape[0]
        u1 = x[:n]
        u2 = x[n:]

        res1 = sparse_matvec(A11, u1) + sparse_matvec(A12, u2)
        res2 = sparse_matvec(A21, u1) + sparse_matvec(A22, u2)

        return torch.cat([res1, res2])

    return matvec


# ============================================================================
# Block Diagonal Preconditioner
# ============================================================================

class BlockDiagonalPreconditioner(BlockPreconditioner):
    """
    Block diagonal preconditioner.

    Structure:
        M = [ A11      0   ]
            [   0    A22*  ]

    where A11 is solved exactly and A22* is an approximation to A22.

    This preconditioner is cheap but may be less effective than coupled approaches.
    Used as a base or in combination with other techniques.
    """

    def __init__(
        self,
        A11: torch.sparse.FloatTensor,
        A22_approx: torch.sparse.FloatTensor,
        solver_A11: Optional[IterativeSolver] = None,
        solver_A22: Optional[IterativeSolver] = None,
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            A11: Intracellular diffusion + mass matrix
            A22_approx: Approximation to extracellular system (can be lumped diagonal)
            solver_A11: Solver for A11 (if None, use diagonal approximation)
            solver_A22: Solver for A22 (if None, use diagonal approximation)
            device: torch device
        """
        self.A11 = A11.to(device)
        self.A22_approx = A22_approx.to(device)
        self.device = device
        self.n = A11.shape[0]

        # Precompute diagonal approximations
        self._compute_diagonal_approximations()

        self.solver_A11 = solver_A11
        self.solver_A22 = solver_A22

    def _compute_diagonal_approximations(self):
        """Compute diagonal approximations for cheap solves."""
        # Extract diagonal of A11
        indices = self.A11.coalesce().indices()
        values = self.A11.coalesce().values()

        diag_A11 = torch.zeros(self.n, device=self.device)
        mask = indices[0] == indices[1]
        diag_A11[indices[0][mask]] = values[mask]
        diag_A11 = torch.clamp(diag_A11, min=1e-8)  # Avoid division by zero
        self.diag_A11_inv = 1.0 / diag_A11

        # Extract diagonal of A22
        indices_22 = self.A22_approx.coalesce().indices()
        values_22 = self.A22_approx.coalesce().values()

        diag_A22 = torch.zeros(self.n, device=self.device)
        mask_22 = indices_22[0] == indices_22[1]
        diag_A22[indices_22[0][mask_22]] = values_22[mask_22]
        diag_A22 = torch.clamp(diag_A22, min=1e-8)
        self.diag_A22_inv = 1.0 / diag_A22

    def apply(self, b: torch.Tensor) -> torch.Tensor:
        """
        Apply block diagonal preconditioner.

        Solves:
            [ A11    0  ] [ x1 ]   [ b1 ]
            [  0   A22* ] [ x2 ] = [ b2 ]

        approximately using diagonal approximations.
        """
        b1 = b[:self.n]
        b2 = b[self.n:]

        # Solve A11 diag ≈ b1 (diagonal approximation)
        x1 = b1 * self.diag_A11_inv

        # Solve A22 diag ≈ b2
        x2 = b2 * self.diag_A22_inv

        return torch.cat([x1, x2])

    def apply_transpose(self, b: torch.Tensor) -> torch.Tensor:
        """Apply transposed preconditioner (same as apply for symmetric case)."""
        return self.apply(b)


# ============================================================================
# Block Triangular (LDU) Preconditioner
# ============================================================================

class BlockTriangularPreconditioner(BlockPreconditioner):
    """
    Block triangular (LDU) preconditioner.

    Structure (lower triangular):
        M_L = [ A11      0  ]
              [ A21   S_A   ]

    where S_A = A22 + A21 * A11^{-1} * A12 is the Schur complement.

    The preconditioner applies:
        1. Solve A11 @ y1 = b1
        2. Compute b2' = b2 - A21 @ y1
        3. Solve S_A @ y2 = b2'

    This couples the two systems and typically provides better convergence
    than block diagonal, especially when coupling is strong.
    """

    def __init__(
        self,
        A11: torch.sparse.FloatTensor,
        A12: torch.sparse.FloatTensor,
        A21: torch.sparse.FloatTensor,
        A22: torch.sparse.FloatTensor,
        schur_approx: str = "mass-lumped",
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            A11, A12, A21, A22: Block matrix components
            schur_approx: How to approximate Schur complement:
                - "mass-lumped": diagonal approximation
                - "full": compute exactly (expensive)
                - "inexact": use sparse approximate inverse
            device: torch device
        """
        self.A11 = A11.to(device)
        self.A12 = A12.to(device)
        self.A21 = A21.to(device)
        self.A22 = A22.to(device)
        self.schur_approx = schur_approx
        self.device = device
        self.n = A11.shape[0]

        # Precompute diagonal approximations for A11 and Schur
        self._compute_approximations()

    def _compute_approximations(self):
        """Compute approximations needed for the preconditioner."""
        # Diagonal of A11
        indices = self.A11.coalesce().indices()
        values = self.A11.coalesce().values()
        diag_A11 = torch.zeros(self.n, device=self.device)
        mask = indices[0] == indices[1]
        if mask.any():
            diag_A11[indices[0][mask]] = values[mask]
        diag_A11 = torch.clamp(diag_A11, min=1e-8)
        self.diag_A11_inv = 1.0 / diag_A11

        # Schur complement approximation
        if self.schur_approx == "mass-lumped":
            # S ≈ A22 (diagonal is easiest)
            indices_22 = self.A22.coalesce().indices()
            values_22 = self.A22.coalesce().values()
            diag_A22 = torch.zeros(self.n, device=self.device)
            mask_22 = indices_22[0] == indices_22[1]
            if mask_22.any():
                diag_A22[indices_22[0][mask_22]] = values_22[mask_22]
            diag_A22 = torch.clamp(diag_A22, min=1e-8)
            self.schur_inv = 1.0 / diag_A22

    def apply(self, b: torch.Tensor) -> torch.Tensor:
        """
        Apply block triangular preconditioner via block forward substitution.
        """
        b1 = b[:self.n]
        b2 = b[self.n:]

        # Step 1: Solve A11 @ y1 = b1 (approximately)
        y1 = b1 * self.diag_A11_inv

        # Step 2: Compute b2' = b2 - A21 @ y1
        A21_y1 = sparse_matvec(self.A21, y1)
        b2_prime = b2 - A21_y1

        # Step 3: Solve Schur @ y2 = b2' (approximately)
        y2 = b2_prime * self.schur_inv

        return torch.cat([y1, y2])

    def apply_transpose(self, b: torch.Tensor) -> torch.Tensor:
        """Apply transposed preconditioner (block backward substitution)."""
        b1 = b[:self.n]
        b2 = b[self.n:]

        # Transpose corresponds to upper triangular solving
        # Step 1: Solve Schur^T @ y2 = b2
        y2 = b2 * self.schur_inv

        # Step 2: Compute b1' = b1 - A12^T @ y2
        A12_T_y2 = sparse_matvec(self.A12.t(), y2)
        b1_prime = b1 - A12_T_y2

        # Step 3: Solve A11^T @ y1 = b1'
        y1 = b1_prime * self.diag_A11_inv

        return torch.cat([y1, y2])


# ============================================================================
# Approximate Schur Complement Preconditioner
# ============================================================================

class SchurComplementPreconditioner(BlockPreconditioner):
    """
    Approximate Schur complement preconditioner.

    The exact Schur complement is: S = A22 - A21 * A11^{-1} * A12

    For the bidomain system, A11 is parabolic (well-behaved) while S is elliptic
    and typically better-conditioned. This preconditioner approximates S.

    Structure:
        M = [ I         0    ]  [ A11    0  ]  [ I    A11^{-1}*A12 ]^{-1}
            [ 0    S^{-1}   ]  [ 0      S  ]  [ 0         I       ]

    which is equivalent to:
        M^{-1} @ [u1; u2] = [u1 - A11^{-1}*A12*u2; S^{-1}*u2]

    This is particularly useful for saddle-point and mixed systems.
    """

    def __init__(
        self,
        A11: torch.sparse.FloatTensor,
        A12: torch.sparse.FloatTensor,
        A21: torch.sparse.FloatTensor,
        A22: torch.sparse.FloatTensor,
        approximate_schur_diag: bool = True,
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            A11, A12, A21, A22: Block matrix components
            approximate_schur_diag: If True, use diagonal(A22) as Schur approx
            device: torch device
        """
        self.A11 = A11.to(device)
        self.A12 = A12.to(device)
        self.A21 = A21.to(device)
        self.A22 = A22.to(device)
        self.device = device
        self.n = A11.shape[0]

        # Precompute inverses
        self._compute_approximations(approximate_schur_diag)

    def _compute_approximations(self, approximate_schur_diag: bool):
        """Compute diagonal approximations."""
        # A11 diagonal inverse
        indices = self.A11.coalesce().indices()
        values = self.A11.coalesce().values()
        diag_A11 = torch.zeros(self.n, device=self.device)
        mask = indices[0] == indices[1]
        if mask.any():
            diag_A11[indices[0][mask]] = values[mask]
        diag_A11 = torch.clamp(diag_A11, min=1e-8)
        self.diag_A11_inv = 1.0 / diag_A11

        # Approximate Schur: S ≈ A22 or diag(A22)
        if approximate_schur_diag:
            indices_22 = self.A22.coalesce().indices()
            values_22 = self.A22.coalesce().values()
            diag_A22 = torch.zeros(self.n, device=self.device)
            mask_22 = indices_22[0] == indices_22[1]
            if mask_22.any():
                diag_A22[indices_22[0][mask_22]] = values_22[mask_22]
            diag_A22 = torch.clamp(torch.abs(diag_A22), min=1e-8)
            self.schur_inv = 1.0 / diag_A22
        else:
            self.schur_inv = None

    def apply(self, b: torch.Tensor) -> torch.Tensor:
        """
        Apply Schur complement preconditioner.

        Computes:
            x1 = A11^{-1} @ b1 - A11^{-1} @ A12 @ (S^{-1} @ b2)
            x2 = S^{-1} @ b2
        """
        b1 = b[:self.n]
        b2 = b[self.n:]

        # Approximate: S^{-1} @ b2
        x2 = b2 * self.schur_inv

        # A11^{-1} @ A12 @ x2
        A12_x2 = sparse_matvec(self.A12, x2)
        A11_inv_A12_x2 = A12_x2 * self.diag_A11_inv

        # A11^{-1} @ b1
        A11_inv_b1 = b1 * self.diag_A11_inv

        # x1 = A11^{-1} @ b1 - A11^{-1} @ A12 @ x2
        x1 = A11_inv_b1 - A11_inv_A12_x2

        return torch.cat([x1, x2])

    def apply_transpose(self, b: torch.Tensor) -> torch.Tensor:
        """Apply transposed preconditioner."""
        return self.apply(b)  # Symmetric case


# ============================================================================
# MINRES Solver for Symmetric Indefinite Systems
# ============================================================================

class MINRESSolver(IterativeSolver):
    """
    Minimal Residual (MINRES) solver for symmetric indefinite systems.

    The bidomain system after block assembly is symmetric but indefinite
    (especially the Schur complement block). MINRES is appropriate for
    such systems, unlike CG which requires positive definiteness.

    References:
        Paige & Saunders "Solution of Sparse Indefinite Systems of Linear Equations"
        SIAM J. Numer. Anal. (1975)
    """

    def solve(
        self,
        A: Callable[[torch.Tensor], torch.Tensor],
        b: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
        tol: float = 1e-6,
        maxiter: int = 1000,
        preconditioner: Optional[BlockPreconditioner] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Solve A @ x = b using MINRES with optional preconditioner.

        Args:
            A: Matrix-vector product function
            b: Right-hand side
            x0: Initial guess (default: zero)
            tol: Relative residual tolerance
            maxiter: Maximum iterations
            preconditioner: Block preconditioner (default: none)

        Returns:
            (solution, info_dict)
        """
        device = b.device
        n = b.shape[0]

        if x0 is None:
            x = torch.zeros_like(b)
        else:
            x = x0.clone()

        # Initial residual
        r = b - A(x)
        rhs_norm = torch.linalg.norm(b)

        if rhs_norm < 1e-14:
            return x, {"converged": True, "iterations": 0, "residuals": [0.0]}

        residuals = []
        device_type = b.device.type

        for k in range(maxiter):
            res_norm = torch.linalg.norm(r)
            residuals.append(res_norm.item())

            if res_norm / rhs_norm < tol:
                return x, {
                    "converged": True,
                    "iterations": k,
                    "residuals": residuals,
                    "final_residual": res_norm.item()
                }

            # Apply preconditioner
            if preconditioner is not None:
                y = preconditioner.apply(r)
            else:
                y = r.clone()

            p = y.clone()

            # Standard MINRES iteration (simplified 2-term recurrence)
            Ap = A(p)
            alpha = torch.dot(r, y) / torch.dot(p, Ap)

            x = x + alpha * p
            r_new = r - alpha * Ap

            # Reorthogonalization (optional, for stability)
            if k % 10 == 0 and k > 0:
                r_new = b - A(x)

            r = r_new

        return x, {
            "converged": False,
            "iterations": maxiter,
            "residuals": residuals,
            "final_residual": residuals[-1] if residuals else 1.0
        }


# ============================================================================
# Preconditioned Conjugate Gradient (PCG) for Elliptic Subproblem
# ============================================================================

class PCGSolver(IterativeSolver):
    """
    Preconditioned Conjugate Gradient solver.

    Used for the elliptic problem (φe solve) in operator splitting approaches.
    Requires symmetric positive-definite matrix (after null-space treatment).
    """

    def solve(
        self,
        A: Callable[[torch.Tensor], torch.Tensor],
        b: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
        tol: float = 1e-6,
        maxiter: int = 1000,
        preconditioner: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Solve A @ x = b using CG with optional preconditioner.

        Args:
            A: Matrix-vector product function (must be SPD)
            b: Right-hand side
            x0: Initial guess
            tol: Relative residual tolerance
            maxiter: Maximum iterations
            preconditioner: Preconditioner function M^{-1}

        Returns:
            (solution, info_dict)
        """
        device = b.device

        if x0 is None:
            x = torch.zeros_like(b)
        else:
            x = x0.clone()

        r = b - A(x)
        rhs_norm = torch.linalg.norm(b)

        if rhs_norm < 1e-14:
            return x, {"converged": True, "iterations": 0}

        # Apply preconditioner
        if preconditioner is not None:
            y = preconditioner(r)
        else:
            y = r.clone()

        p = y.clone()
        residuals = []

        for k in range(maxiter):
            res_norm = torch.linalg.norm(r)
            residuals.append(res_norm.item())

            if res_norm / rhs_norm < tol:
                return x, {
                    "converged": True,
                    "iterations": k,
                    "residuals": residuals
                }

            Ap = A(p)
            rDoty = torch.dot(r, y)
            alpha = rDoty / torch.dot(p, Ap)

            x = x + alpha * p
            r = r - alpha * Ap

            # Apply preconditioner to new residual
            if preconditioner is not None:
                y_new = preconditioner(r)
            else:
                y_new = r.clone()

            beta = torch.dot(r, y_new) / rDoty
            p = y_new + beta * p
            y = y_new

        return x, {
            "converged": False,
            "iterations": maxiter,
            "residuals": residuals
        }


# ============================================================================
# Main Demo/Test
# ============================================================================

def main():
    """Demonstration of block preconditioners."""
    print("=" * 80)
    print("BIDOMAIN BLOCK PRECONDITIONERS - REFERENCE IMPLEMENTATION")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Create small test system
    n = 100
    print(f"\nCreating test system (n={n})...")

    # Simple 2D Laplacian for each block
    def create_laplacian(n, alpha=1.0):
        """Create sparse 1D Laplacian."""
        rows, cols, vals = [], [], []
        for i in range(n):
            rows.append(i)
            cols.append(i)
            vals.append(2.0 * alpha)

            if i > 0:
                rows.append(i)
                cols.append(i - 1)
                vals.append(-alpha)

            if i < n - 1:
                rows.append(i)
                cols.append(i + 1)
                vals.append(-alpha)

        indices = torch.tensor([rows, cols], dtype=torch.long, device=device)
        vals = torch.tensor(vals, dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(indices, vals, (n, n), device=device).coalesce()

    # Create block matrices
    M = create_laplacian(n, alpha=1.0)
    Ki = create_laplacian(n, alpha=2.0)
    Ke = create_laplacian(n, alpha=1.0)

    # Block system:
    # [ M + 0.5*Ki    0.5*Ki  ]
    # [ Ki           -(Ki+Ke)]

    dt = 0.01
    theta = 0.5

    A11 = M / dt + theta * Ki
    A12 = theta * Ki
    A21 = Ki
    A22 = -(Ki + Ke)

    # Create block matrix-vector product
    A_matvec = form_block_matrix(A11, A12, A21, A22, device=device)

    # Create test RHS
    b = torch.randn(2*n, device=device)

    print(f"System size: {2*n} × {2*n}")
    print(f"RHS norm: {torch.linalg.norm(b):.6f}")

    # Test Block Diagonal Preconditioner
    print("\n" + "=" * 80)
    print("Block Diagonal Preconditioner")
    print("=" * 80)

    bd_prec = BlockDiagonalPreconditioner(A11, A22, device=device)
    print("Block diagonal preconditioner created")

    # Test application
    x_prec = bd_prec.apply(b)
    print(f"Preconditioner output norm: {torch.linalg.norm(x_prec):.6f}")

    # Test Block Triangular Preconditioner
    print("\n" + "=" * 80)
    print("Block Triangular Preconditioner")
    print("=" * 80)

    bt_prec = BlockTriangularPreconditioner(A11, A12, A21, A22, device=device)
    print("Block triangular preconditioner created")

    x_bt = bt_prec.apply(b)
    print(f"Preconditioner output norm: {torch.linalg.norm(x_bt):.6f}")

    # Test Schur Complement Preconditioner
    print("\n" + "=" * 80)
    print("Schur Complement Preconditioner")
    print("=" * 80)

    sc_prec = SchurComplementPreconditioner(A11, A12, A21, A22, device=device)
    print("Schur complement preconditioner created")

    x_sc = sc_prec.apply(b)
    print(f"Preconditioner output norm: {torch.linalg.norm(x_sc):.6f}")

    # Test MINRES Solver
    print("\n" + "=" * 80)
    print("MINRES Solver (no preconditioner)")
    print("=" * 80)

    minres_solver = MINRESSolver()
    x_minres, info = minres_solver.solve(
        A_matvec, b, tol=1e-6, maxiter=500,
        preconditioner=None
    )

    print(f"Converged: {info['converged']}")
    print(f"Iterations: {info['iterations']}")
    if info['iterations'] > 0:
        print(f"Final residual norm: {info['residuals'][-1]:.6e}")

    # Verify solution
    residual = torch.linalg.norm(b - A_matvec(x_minres))
    print(f"Verification residual: {residual:.6e}")

    # Test MINRES with preconditioner
    print("\n" + "=" * 80)
    print("MINRES Solver (with Block Diagonal Preconditioner)")
    print("=" * 80)

    x_minres_prec, info_prec = minres_solver.solve(
        A_matvec, b, tol=1e-6, maxiter=500,
        preconditioner=bd_prec
    )

    print(f"Converged: {info_prec['converged']}")
    print(f"Iterations: {info_prec['iterations']}")
    if info_prec['iterations'] > 0:
        print(f"Final residual norm: {info_prec['residuals'][-1]:.6e}")

    residual_prec = torch.linalg.norm(b - A_matvec(x_minres_prec))
    print(f"Verification residual: {residual_prec:.6e}")

    # Test PCG for elliptic solve
    print("\n" + "=" * 80)
    print("PCG Solver (for elliptic φe subproblem)")
    print("=" * 80)

    # Use -(Ki + Ke) but force to be SPD by adding identity
    A_elliptic = Ki + Ke + 0.1 * M

    def A_elliptic_matvec(x):
        return sparse_matvec(A_elliptic, x)

    b_elliptic = torch.randn(n, device=device)

    pcg_solver = PCGSolver()
    x_elliptic, info_elliptic = pcg_solver.solve(
        A_elliptic_matvec, b_elliptic,
        tol=1e-6, maxiter=500,
        preconditioner=None
    )

    print(f"Converged: {info_elliptic['converged']}")
    print(f"Iterations: {info_elliptic['iterations']}")

    residual_elliptic = torch.linalg.norm(b_elliptic - A_elliptic_matvec(x_elliptic))
    print(f"Verification residual: {residual_elliptic:.6e}")

    print("\n" + "=" * 80)
    print("Convergence comparison")
    print("=" * 80)
    print(f"MINRES (no prec): {info['iterations']} iterations")
    print(f"MINRES (BD prec): {info_prec['iterations']} iterations")
    speedup = info['iterations'] / max(info_prec['iterations'], 1)
    print(f"Speedup factor: {speedup:.2f}x")

    print("\n" + "=" * 80)
    print("Reference implementation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
