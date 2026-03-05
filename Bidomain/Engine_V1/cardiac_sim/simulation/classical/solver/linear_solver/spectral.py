"""
Unified Spectral Solver (Tier 1)

Direct spectral solve for the elliptic equation on structured grids:
    -D * Laplacian(u) = b

O(N log N). No iterations. Only valid for isotropic, uniform-grid
problems with homogeneous BCs of a single type on all edges.

Supports three BC types via different transforms:
- 'neumann'   -> DCT-II/III (insulated boundaries)
- 'dirichlet' -> DST-I via FFT (bath-coupled phi_e = 0)
- 'periodic'  -> FFT (periodic domain)

For the bidomain elliptic solve: D = D_i + D_e.
For Kleber validation (bath-coupled): bc_type='dirichlet' uses DST.

Ref: improvement.md L1055-1242 (SpectralSolver spec)
"""

import torch
import torch_dct
from .base import LinearSolver


# === GPU-native DST-I via torch.fft ===

def _dst1_1d(x, dim):
    """DST-I along one dimension via odd-extension FFT."""
    N = x.shape[dim]
    M = 2 * (N + 1)

    shape = list(x.shape)
    shape[dim] = M
    ext = x.new_zeros(shape)

    def sl(d, s, e):
        slices = [slice(None)] * len(shape)
        slices[d] = slice(s, e)
        return tuple(slices)

    ext[sl(dim, 1, N + 1)] = x
    ext[sl(dim, N + 2, M)] = -torch.flip(x, [dim])

    fft_ext = torch.fft.fft(ext, dim=dim)
    return -fft_ext[sl(dim, 1, N + 1)].imag


def dst1_2d(x):
    """2D DST-I forward via torch.fft. GPU-native."""
    return _dst1_1d(_dst1_1d(x, dim=1), dim=0)


def idst1_2d(X):
    """2D DST-I inverse. DST-I is self-inverse up to normalization."""
    mx, my = X.shape
    return dst1_2d(X) / (4.0 * (mx + 1) * (my + 1))


class SpectralSolver(LinearSolver):
    """
    Tier 1: Direct spectral solve for constant-coefficient Laplacian.

    Solves -D * Laplacian(u) = b via spectral transform.
    Matrix A in solve(A, b) is ignored — eigenvalues are precomputed
    from grid parameters.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions (full grid, including boundary nodes)
    dx, dy : float
        Grid spacing (cm)
    D : float
        Diffusion coefficient (cm^2/ms). For elliptic: D_i + D_e.
    bc_type : str
        'neumann' (DCT), 'dirichlet' (DST), or 'periodic' (FFT)
    """

    def __init__(self, nx, ny, dx, dy, D, bc_type='neumann'):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.D = D
        self.bc_type = bc_type
        self._eigenvalues = None

    def _compute_eigenvalues(self, device, dtype):
        """Compute Laplacian eigenvalues for the selected BC type."""
        if self.bc_type == 'neumann':
            i_idx = torch.arange(self.nx, device=device, dtype=dtype)
            j_idx = torch.arange(self.ny, device=device, dtype=dtype)
            lam_x = (2.0 / self.dx**2) * (1.0 - torch.cos(torch.pi * i_idx / self.nx))
            lam_y = (2.0 / self.dy**2) * (1.0 - torch.cos(torch.pi * j_idx / self.ny))
            LAM_X, LAM_Y = torch.meshgrid(lam_x, lam_y, indexing='ij')
            self._eigenvalues = self.D * (LAM_X + LAM_Y)
            self._eigenvalues[0, 0] = 1.0  # Avoid division by zero (null space)

        elif self.bc_type == 'dirichlet':
            # Interior grid: (Nx-2) x (Ny-2)
            mx, my = self.nx - 2, self.ny - 2
            i_idx = torch.arange(mx, device=device, dtype=dtype)
            j_idx = torch.arange(my, device=device, dtype=dtype)
            lam_x = (2.0 / self.dx**2) * (1.0 - torch.cos(torch.pi * (i_idx + 1) / (mx + 1)))
            lam_y = (2.0 / self.dy**2) * (1.0 - torch.cos(torch.pi * (j_idx + 1) / (my + 1)))
            LAM_X, LAM_Y = torch.meshgrid(lam_x, lam_y, indexing='ij')
            self._eigenvalues = self.D * (LAM_X + LAM_Y)
            # All eigenvalues > 0 — no null space

        elif self.bc_type == 'periodic':
            kx = torch.fft.fftfreq(self.nx, d=self.dx, device=device, dtype=dtype) * 2 * torch.pi
            ky = torch.fft.fftfreq(self.ny, d=self.dy, device=device, dtype=dtype) * 2 * torch.pi
            KX, KY = torch.meshgrid(kx, ky, indexing='ij')
            self._eigenvalues = self.D * (KX**2 + KY**2)
            self._eigenvalues[0, 0] = 1.0

    def solve(self, A, b):
        """Solve -D*Lap*u = b. Matrix A is ignored."""
        if self._eigenvalues is None:
            self._compute_eigenvalues(b.device, b.dtype)

        if self.bc_type == 'neumann':
            return self._solve_neumann(b)
        elif self.bc_type == 'dirichlet':
            return self._solve_dirichlet(b)
        elif self.bc_type == 'periodic':
            return self._solve_periodic(b)

    def _solve_neumann(self, b):
        rhs = b.reshape(self.nx, self.ny)
        rhs_hat = torch_dct.dct_2d(rhs, norm='ortho')
        u_hat = rhs_hat / self._eigenvalues
        u_hat[0, 0] = 0.0  # Null space: set mean to zero
        u = torch_dct.idct_2d(u_hat, norm='ortho')
        return u.flatten()

    def _solve_dirichlet(self, b):
        rhs_full = b.reshape(self.nx, self.ny)
        rhs_int = rhs_full[1:-1, 1:-1].contiguous()
        rhs_hat = dst1_2d(rhs_int)
        u_hat = rhs_hat / self._eigenvalues
        u_int = idst1_2d(u_hat)
        u_full = b.new_zeros(self.nx, self.ny)
        u_full[1:-1, 1:-1] = u_int
        return u_full.flatten()

    def _solve_periodic(self, b):
        rhs = b.reshape(self.nx, self.ny)
        rhs_hat = torch.fft.fft2(rhs)
        u_hat = rhs_hat / self._eigenvalues
        u_hat[0, 0] = 0.0
        u = torch.fft.ifft2(u_hat).real
        return u.flatten()

