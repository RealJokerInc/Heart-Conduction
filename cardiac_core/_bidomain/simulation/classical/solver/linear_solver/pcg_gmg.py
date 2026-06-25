"""
PCG with Geometric Multigrid Preconditioner (Stub)

Tier 3: PCG with GMG preconditioner for heterogeneous/mixed-BC problems.
Full implementation deferred — requires GeometricMultigridPreconditioner.

Ref: improvement.md L1390-1509
"""

from .base import LinearSolver


class EllipticPCGMGSolver(LinearSolver):
    """
    Tier 3: PCG with geometric multigrid preconditioner.

    STUB — not yet implemented. For heterogeneous or mixed-BC problems
    where spectral methods are not applicable.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions
    dx, dy : float
        Grid spacing
    D : float
        Average diffusion coefficient
    n_levels : int
        Number of multigrid levels
    max_iters : int
        Maximum PCG iterations
    tol : float
        Relative residual tolerance
    """

    def __init__(self, nx, ny, dx, dy, D, n_levels=4,
                 max_iters=100, tol=1e-8):
        raise NotImplementedError(
            "EllipticPCGMGSolver is a stub. "
            "Use SpectralSolver (Tier 1) or PCGSpectralSolver (Tier 2) instead."
        )

    def solve(self, A, b):
        raise NotImplementedError
