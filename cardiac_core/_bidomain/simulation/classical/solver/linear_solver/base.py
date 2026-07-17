"""
LinearSolver Abstract Base Class

Solves Ax = b. Owns workspace buffers for zero-allocation per step.

Ref: improvement.md:L1076-1109
"""

from abc import ABC, abstractmethod
import warnings
import torch


class SolverConvergenceWarning(UserWarning):
    """A linear solver returned before meeting its convergence tolerance.

    Emitted (not raised) by default so a silently under-solved solve becomes visible
    instead of propagating a wrong result. Callers who want it fatal can escalate::

        import warnings
        warnings.filterwarnings("error", category=SolverConvergenceWarning)
    """


def warn_nonconvergence(solver, iterations, residual_norm, b_norm, tol, reason="max_iters"):
    """Emit a :class:`SolverConvergenceWarning` describing a non-converged solve."""
    rel = residual_norm / b_norm if b_norm > 0 else float("inf")
    warnings.warn(
        f"{solver} did not converge ({reason}): {iterations} iters, relative residual "
        f"{rel:.2e} > tol {tol:.1e}. The solve result may be inaccurate — increase "
        f"max_iters, loosen tol, or choose a different solver.",
        SolverConvergenceWarning,
        stacklevel=3,
    )


class LinearSolver(ABC):
    """
    Abstract base class for linear system solvers.

    Solves Ax = b where A is a sparse matrix from diffusion time stepping.
    Owns workspace buffers to avoid allocation per step.
    """

    @abstractmethod
    def solve(self, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Solve Ax = b.

        Parameters
        ----------
        A : torch.Tensor
            Sparse system matrix (SPD for diffusion)
        b : torch.Tensor
            Right-hand side vector

        Returns
        -------
        x : torch.Tensor
            Solution vector
        """
        pass
