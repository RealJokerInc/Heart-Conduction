"""
Diffusion Time Stepping Solvers

Integrators for the diffusion PDE:
- explicit/: Forward Euler (no linear solve, CFL-limited)
- implicit/: Crank-Nicolson, BDF1 (linear solve required)
- linear_solver/: Linear system solvers (PCG)

Ref: improvement.md:L1033-1074
"""

from .base import DiffusionSolver

__all__ = ['DiffusionSolver']
