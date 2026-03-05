"""
PCG with Spectral Preconditioner (Tier 2)

Preconditioned Conjugate Gradient where the preconditioner is the
SpectralSolver (exact inverse of the isotropic constant-coefficient Laplacian).

For mildly anisotropic problems the spectral preconditioner gives condition
number close to 1, converging in 1-5 iterations.

Ref: improvement.md L1055-1242
"""

import torch
from .base import LinearSolver
from .spectral import SpectralSolver
from .pcg import sparse_mv


class PCGSpectralSolver(LinearSolver):
    """
    Tier 2: PCG with spectral (DCT/DST/FFT) preconditioner.

    Handles anisotropic or mildly heterogeneous problems where
    the spectral solver alone would be inaccurate but provides
    an excellent preconditioner.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions
    dx, dy : float
        Grid spacing (cm)
    D : float
        Average diffusion coefficient for preconditioner (D_i + D_e)
    bc_type : str
        'neumann', 'dirichlet', or 'periodic'
    max_iters : int
        Maximum PCG iterations
    tol : float
        Relative residual tolerance
    """

    def __init__(self, nx, ny, dx, dy, D, bc_type='neumann',
                 max_iters=50, tol=1e-8):
        self.max_iters = max_iters
        self.tol = tol
        self._precond = SpectralSolver(nx, ny, dx, dy, D, bc_type)

        # Workspace (lazy)
        self._r = None
        self._z = None
        self._p = None
        self._Ap = None
        self._x = None
        self._last_solution = None
        self.last_iters = 0

    def _alloc(self, n, device, dtype):
        if self._r is None or self._r.shape[0] != n:
            self._r = torch.zeros(n, device=device, dtype=dtype)
            self._z = torch.zeros(n, device=device, dtype=dtype)
            self._p = torch.zeros(n, device=device, dtype=dtype)
            self._Ap = torch.zeros(n, device=device, dtype=dtype)
            self._x = torch.zeros(n, device=device, dtype=dtype)

    def solve(self, A, b):
        """Solve Ax = b using PCG with spectral preconditioner."""
        n = b.shape[0]
        self._alloc(n, b.device, b.dtype)

        x = self._x
        if self._last_solution is not None:
            x.copy_(self._last_solution)
        else:
            x.zero_()

        r = self._r
        r.copy_(b)
        if x.abs().sum() > 0:
            r.sub_(sparse_mv(A, x))

        b_norm = torch.norm(b)
        if b_norm < 1e-14:
            self._last_solution = x.clone()
            self.last_iters = 0
            return x.clone()

        # Preconditioner: z = M^{-1} r via spectral solve
        z = self._z
        z.copy_(self._precond.solve(A, r))

        p = self._p
        p.copy_(z)
        rz = torch.dot(r, z)

        for k in range(self.max_iters):
            Ap = self._Ap
            Ap.copy_(sparse_mv(A, p))

            pAp = torch.dot(p, Ap)
            if pAp.abs() < 1e-30:
                break
            alpha = rz / pAp

            x.add_(p, alpha=alpha)
            r.sub_(Ap, alpha=alpha)

            r_norm = torch.norm(r)
            if r_norm / b_norm < self.tol:
                self.last_iters = k + 1
                break

            z.copy_(self._precond.solve(A, r))
            rz_new = torch.dot(r, z)
            beta = rz_new / rz
            p.mul_(beta).add_(z)
            rz = rz_new
        else:
            self.last_iters = self.max_iters

        self._last_solution = x.clone()
        return x.clone()
