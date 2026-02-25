"""
Linear Solvers

Solvers for Ax = b arising from implicit diffusion time stepping.

Available solvers:
- PCGSolver: Preconditioned Conjugate Gradient (general sparse SPD)
- ChebyshevSolver: Polynomial iteration, zero sync (GPU-optimized)
- DCTSolver: Spectral solver for structured grids + Neumann BCs
- FFTSolver: Spectral solver for structured grids + periodic BCs
"""

from .base import LinearSolver
from .pcg import PCGSolver
from .chebyshev import ChebyshevSolver
from .fft import DCTSolver, FFTSolver

__all__ = ['LinearSolver', 'PCGSolver', 'ChebyshevSolver', 'DCTSolver', 'FFTSolver']
