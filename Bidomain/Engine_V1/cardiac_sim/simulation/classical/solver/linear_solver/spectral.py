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
from .base import LinearSolver


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
        rhs_hat = self._dct2(rhs)
        u_hat = rhs_hat / self._eigenvalues
        u_hat[0, 0] = 0.0  # Null space: set mean to zero
        u = self._idct2(u_hat)
        return u.flatten()

    def _solve_dirichlet(self, b):
        rhs_full = b.reshape(self.nx, self.ny)
        # Extract interior nodes only
        rhs_int = rhs_full[1:-1, 1:-1].contiguous()
        mx, my = rhs_int.shape
        # Forward DST-I
        rhs_hat = self._dst1_2d_forward(rhs_int)
        # Spectral division
        u_hat = rhs_hat / self._eigenvalues
        # Inverse DST-I
        u_int = self._dst1_2d_inverse(u_hat)
        # Pad with zeros (Dirichlet boundary value = 0)
        u_full = torch.zeros(self.nx, self.ny, device=b.device, dtype=b.dtype)
        u_full[1:-1, 1:-1] = u_int
        return u_full.flatten()

    def _solve_periodic(self, b):
        rhs = b.reshape(self.nx, self.ny)
        rhs_hat = torch.fft.fft2(rhs)
        u_hat = rhs_hat / self._eigenvalues
        u_hat[0, 0] = 0.0
        u = torch.fft.ifft2(u_hat).real
        return u.flatten()

    # === DCT-II/III via FFT (Neumann) ===

    def _dct2(self, x):
        """2D DCT-II via FFT."""
        nx, ny = x.shape
        # DCT along dim 0
        x_mirror = torch.cat([x, x.flip(0)], dim=0)
        fft_x = torch.fft.rfft(x_mirror, dim=0)
        k = torch.arange(nx, device=x.device, dtype=x.dtype)
        phase = torch.exp(-1j * torch.pi * k / (2 * nx))
        dct_x = (fft_x[:nx] * phase.unsqueeze(1)).real * (2.0 / nx) ** 0.5
        dct_x[0] *= 0.5 ** 0.5
        # DCT along dim 1
        dct_x_mirror = torch.cat([dct_x, dct_x.flip(1)], dim=1)
        fft_y = torch.fft.rfft(dct_x_mirror, dim=1)
        k = torch.arange(ny, device=x.device, dtype=x.dtype)
        phase = torch.exp(-1j * torch.pi * k / (2 * ny))
        dct_xy = (fft_y[:, :ny] * phase.unsqueeze(0)).real * (2.0 / ny) ** 0.5
        dct_xy[:, 0] *= 0.5 ** 0.5
        return dct_xy

    def _idct2(self, x):
        """2D IDCT (DCT-III) via FFT."""
        nx, ny = x.shape
        x_scaled = x.clone()
        x_scaled[0, :] *= 2.0 ** 0.5
        x_scaled[:, 0] *= 2.0 ** 0.5
        x_scaled *= (nx * ny / 4.0) ** 0.5
        # IDCT along dim 1
        k = torch.arange(ny, device=x.device, dtype=x.dtype)
        phase = torch.exp(1j * torch.pi * k / (2 * ny))
        x_complex = x_scaled * phase.unsqueeze(0)
        x_padded = torch.cat([x_complex, torch.zeros_like(x_complex)], dim=1)
        idct_y = torch.fft.irfft(x_padded, n=2 * ny, dim=1)[:, :ny]
        # IDCT along dim 0
        k = torch.arange(nx, device=x.device, dtype=x.dtype)
        phase = torch.exp(1j * torch.pi * k / (2 * nx))
        idct_y_complex = idct_y * phase.unsqueeze(1)
        idct_y_padded = torch.cat([idct_y_complex, torch.zeros_like(idct_y_complex)], dim=0)
        result = torch.fft.irfft(idct_y_padded, n=2 * nx, dim=0)[:nx, :]
        return result.real

    # === DST-I via FFT (Dirichlet) ===

    def _dst1_1d(self, x, dim):
        """1D DST-I via FFT along specified dimension. Returns 2*DST-I(x)."""
        N = x.shape[dim]
        M = 2 * (N + 1)

        # Build odd extension along dim
        shape = list(x.shape)
        shape[dim] = M
        ext = torch.zeros(shape, device=x.device, dtype=x.dtype)

        # Slice helpers
        def _slice(dim, start, end):
            s = [slice(None)] * len(shape)
            s[dim] = slice(start, end)
            return tuple(s)

        ext[_slice(dim, 1, N + 1)] = x
        ext[_slice(dim, N + 2, M)] = -torch.flip(x, [dim])

        # FFT along dim
        fft_ext = torch.fft.fft(ext, dim=dim)

        # Extract -imag of indices 1..N
        result = -fft_ext[_slice(dim, 1, N + 1)].imag

        return result

    def _dst1_2d_forward(self, x):
        """2D DST-I forward transform. Returns 4*DST-I-2D(x)."""
        y = self._dst1_1d(x, dim=1)  # 2*DST-I along dim 1
        y = self._dst1_1d(y, dim=0)  # 2*DST-I along dim 0
        return y  # = 4 * DST-I-2D(x)

    def _dst1_2d_inverse(self, X):
        """2D DST-I inverse transform. Consistent with _dst1_2d_forward."""
        mx, my = X.shape
        raw = self._dst1_2d_forward(X)
        return raw / (4.0 * (mx + 1) * (my + 1))
