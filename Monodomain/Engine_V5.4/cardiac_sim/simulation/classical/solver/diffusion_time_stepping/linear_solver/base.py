"""
LinearSolver Abstract Base Class

Solves Ax = b. Owns workspace buffers for zero-allocation per step.

Ref: improvement.md:L1076-1109
"""

from abc import ABC, abstractmethod
import torch


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
