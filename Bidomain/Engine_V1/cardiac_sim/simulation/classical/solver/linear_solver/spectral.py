"""
Unified Spectral Solver (Tier 1)

Direct spectral solve for the elliptic equation on structured grids:
    -D * Laplacian(u) = b

O(N log N). No iterations. Only valid for isotropic, uniform-grid
problems with per-axis homogeneous BCs.

Per-axis BC types and their transforms:
- 'neumann'   -> DCT-II/III (insulated boundary)
- 'dirichlet' -> DST-I via FFT (bath-coupled, u = 0)
- 'periodic'  -> FFT

Supports any combination of per-axis BCs: e.g. Neumann in x (DCT),
Dirichlet in y (DST) for bath_coupled_edges([TOP, BOTTOM]).

For the bidomain elliptic solve: D = D_i + D_e.

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


# === Per-axis transform helpers ===

def _fwd(x, dim, bc):
    """Forward spectral transform along one dimension."""
    if bc == 'neumann':
        # torch_dct.dct works on last dim; transpose for dim=0
        if dim == 0:
            return torch_dct.dct(x.T, norm='ortho').T
        return torch_dct.dct(x, norm='ortho')
    elif bc == 'dirichlet':
        return _dst1_1d(x, dim=dim)
    elif bc == 'periodic':
        return torch.fft.fft(x, dim=dim)


def _inv(x, dim, bc):
    """Inverse spectral transform along one dimension."""
    if bc == 'neumann':
        if dim == 0:
            return torch_dct.idct(x.T, norm='ortho').T
        return torch_dct.idct(x, norm='ortho')
    elif bc == 'dirichlet':
        N = x.shape[dim]
        return _dst1_1d(x, dim=dim) / (2.0 * (N + 1))
    elif bc == 'periodic':
        return torch.fft.ifft(x, dim=dim)


class SpectralSolver(LinearSolver):
    """
    Tier 1: Direct spectral solve for constant-coefficient Laplacian.

    Solves -D * Laplacian(u) = b via spectral transform.
    Matrix A in solve(A, b) is ignored — eigenvalues are precomputed
    from grid parameters.

    Supports per-axis BCs: any combination of neumann/dirichlet/periodic
    along x (dim=0) and y (dim=1).

    Parameters
    ----------
    nx, ny : int
        Grid dimensions (full grid, including boundary nodes)
    dx, dy : float
        Grid spacing (cm)
    D : float
        Diffusion coefficient (cm^2/ms). For elliptic: D_i + D_e.
    bc_type : str
        Uniform BC: 'neumann', 'dirichlet', or 'periodic' (sets both axes)
    bc_x, bc_y : str, optional
        Per-axis override. If given, bc_type is ignored for that axis.
    """

    def __init__(self, nx, ny, dx, dy, D, bc_type='neumann',
                 bc_x=None, bc_y=None):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.D = D
        self.bc_x = bc_x if bc_x is not None else bc_type
        self.bc_y = bc_y if bc_y is not None else bc_type
        # Keep bc_type for backward compat (used by some callers)
        self.bc_type = bc_type
        self._eigenvalues = None

    def _compute_eigenvalues(self, device, dtype):
        """Compute separable Laplacian eigenvalues per axis."""
        lam_x = self._axis_eigenvalues(
            self.bc_x, self.nx, self.dx, device, dtype)
        lam_y = self._axis_eigenvalues(
            self.bc_y, self.ny, self.dy, device, dtype)

        LAM_X, LAM_Y = torch.meshgrid(lam_x, lam_y, indexing='ij')
        self._eigenvalues = self.D * (LAM_X + LAM_Y)

        # Null space: only when both axes are neumann (or periodic)
        self._has_null = (self.bc_x in ('neumann', 'periodic') and
                          self.bc_y in ('neumann', 'periodic'))
        if self._has_null:
            self._eigenvalues[0, 0] = 1.0  # avoid div-by-zero

        # Working grid dimensions (Dirichlet strips boundary nodes)
        self._nx_work = lam_x.shape[0]
        self._ny_work = lam_y.shape[0]

    @staticmethod
    def _axis_eigenvalues(bc, n, dx, device, dtype):
        """Eigenvalues for one axis given its BC type."""
        if bc == 'neumann':
            k = torch.arange(n, device=device, dtype=dtype)
            return (2.0 / dx**2) * (1.0 - torch.cos(torch.pi * k / n))
        elif bc == 'dirichlet':
            m = n - 2  # interior nodes
            k = torch.arange(m, device=device, dtype=dtype)
            return (2.0 / dx**2) * (1.0 - torch.cos(
                torch.pi * (k + 1) / (m + 1)))
        elif bc == 'periodic':
            freq = torch.fft.fftfreq(n, d=dx, device=device,
                                      dtype=dtype) * 2 * torch.pi
            return freq ** 2

    def solve(self, A, b):
        """Solve -D*Lap*u = b. Matrix A is ignored."""
        if self._eigenvalues is None:
            self._compute_eigenvalues(b.device, b.dtype)

        rhs_full = b.reshape(self.nx, self.ny)

        # Extract working region (strip boundary for Dirichlet axes)
        x_sl = slice(1, -1) if self.bc_x == 'dirichlet' else slice(None)
        y_sl = slice(1, -1) if self.bc_y == 'dirichlet' else slice(None)
        rhs_work = rhs_full[x_sl, y_sl].contiguous()

        # Forward transforms
        rhs_hat = _fwd(rhs_work, dim=0, bc=self.bc_x)
        rhs_hat = _fwd(rhs_hat, dim=1, bc=self.bc_y)

        # Spectral division
        u_hat = rhs_hat / self._eigenvalues
        if self._has_null:
            u_hat[0, 0] = 0.0

        # Inverse transforms
        u_work = _inv(u_hat, dim=0, bc=self.bc_x)
        u_work = _inv(u_work, dim=1, bc=self.bc_y)

        # Extract real part (periodic inverse leaves complex intermediates)
        if torch.is_complex(u_work):
            u_work = u_work.real

        # Pad back to full grid (zeros at Dirichlet boundaries)
        if self.bc_x == 'dirichlet' or self.bc_y == 'dirichlet':
            u_full = b.new_zeros(self.nx, self.ny)
            u_full[x_sl, y_sl] = u_work
            return u_full.flatten()
        return u_work.flatten()
