"""
FFT/DCT Spectral Linear Solver [DEPRECATED]

DEPRECATED: Use spectral.py instead. This module has chi/Cm baked into
eigenvalues (incompatible with Formulation B) and a DCT fallback with
wrong normalization. See LINEAR_SOLVER_IMPLEMENTATION.md § 2.4.

Direct spectral solve for structured Cartesian grids.
O(N log N) complexity via FFT/DCT.

Ref: Research/03_GPU_Linear:L169-261
"""

import torch
from typing import Optional, Tuple

from .base import LinearSolver


class DCTSolver(LinearSolver):
    """
    DCT-based direct solver for structured grids with Neumann BCs.

    Uses Discrete Cosine Transform (Type II/III) to diagonalize the
    discrete Laplacian operator. O(N log N) solve time.

    This solver is specialized for FDM/FVM on structured Cartesian grids.
    It ignores the sparse matrix A and instead uses pre-computed Laplacian
    eigenvalues derived from grid parameters.

    IMPORTANT: Only use with FDMDiscretization or FVMDiscretization on
    StructuredGrid. Will give wrong results with FEM or unstructured meshes.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions
    dx, dy : float
        Grid spacing (cm)
    dt : float
        Time step (ms)
    D : float
        Diffusion coefficient (cm²/ms)
    chi : float
        Surface-to-volume ratio (cm⁻¹)
    Cm : float
        Membrane capacitance (µF/cm²)
    scheme : str
        Time stepping scheme: 'CN', 'BDF1', 'BDF2'

    Note
    ----
    Requires `torch-dct` package: pip install torch-dct
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float,
        dy: float,
        dt: float,
        D: float,
        chi: float = 1400.0,
        Cm: float = 1.0,
        scheme: str = 'CN'
    ):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.D = D
        self.chi = chi
        self.Cm = Cm
        self.scheme = scheme.upper()

        # Eigenvalues computed lazily on first solve
        self._eigenvalues: Optional[torch.Tensor] = None
        self._denom: Optional[torch.Tensor] = None
        self._device: Optional[torch.device] = None
        self._dtype: Optional[torch.dtype] = None

    def _compute_eigenvalues(self, device: torch.device, dtype: torch.dtype) -> None:
        """
        Compute Laplacian eigenvalues for Neumann BCs.

        λ_{i,j} = (2/dx²)(cos(πi/Nx) - 1) + (2/dy²)(cos(πj/Ny) - 1)

        All eigenvalues are ≤ 0 (Laplacian is negative semi-definite).
        """
        i_idx = torch.arange(self.nx, device=device, dtype=dtype)
        j_idx = torch.arange(self.ny, device=device, dtype=dtype)

        # Eigenvalues of discrete Laplacian with Neumann BCs
        lam_x = (2.0 / self.dx**2) * (torch.cos(torch.pi * i_idx / self.nx) - 1.0)
        lam_y = (2.0 / self.dy**2) * (torch.cos(torch.pi * j_idx / self.ny) - 1.0)

        LAM_X, LAM_Y = torch.meshgrid(lam_x, lam_y, indexing='ij')
        self._eigenvalues = LAM_X + LAM_Y  # All ≤ 0

        # Build denominator based on scheme
        # For FDM: chi*Cm*dV/dt = D*Δ*V (where Δ is Laplacian, L = D*Δ in FDM)
        # CN: (chi*Cm - 0.5*dt*D*Δ)*V^{n+1} = (chi*Cm + 0.5*dt*D*Δ)*V^n
        # In spectral: (chi*Cm - 0.5*dt*D*λ)*V_hat = rhs_hat
        # Note: λ = eigenvalues of Δ (all ≤ 0), so D*λ ≤ 0

        chi_Cm = self.chi * self.Cm
        D_lambda = self.D * self._eigenvalues  # D times Laplacian eigenvalues

        if self.scheme == 'CN':
            # A_lhs = chi*Cm*I - 0.5*dt*L = chi*Cm*I - 0.5*dt*D*Δ
            self._denom = chi_Cm - 0.5 * self.dt * D_lambda
        elif self.scheme == 'BDF1':
            # A_lhs = chi*Cm*I - dt*L
            self._denom = chi_Cm - self.dt * D_lambda
        elif self.scheme == 'BDF2':
            # A_lhs = 3*chi*Cm*I - 2*dt*L
            self._denom = 3.0 * chi_Cm - 2.0 * self.dt * D_lambda
        else:
            raise ValueError(f"Unknown scheme: {self.scheme}")

        # Handle zero eigenvalue at (0,0) for pure Neumann
        # Set to 1 to avoid division by zero, result will be set to 0
        if self._denom[0, 0].abs() < 1e-10:
            self._denom[0, 0] = 1.0

        self._device = device
        self._dtype = dtype

    def solve(self, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Solve Ax = b using DCT.

        Note: The matrix A is ignored. The solver uses pre-computed
        eigenvalues from grid parameters.

        Parameters
        ----------
        A : torch.Tensor
            Sparse system matrix (ignored)
        b : torch.Tensor
            Right-hand side vector, shape (nx*ny,)

        Returns
        -------
        x : torch.Tensor
            Solution vector, shape (nx*ny,)
        """
        device = b.device
        dtype = b.dtype

        # Lazy eigenvalue computation
        if self._eigenvalues is None or self._device != device:
            self._compute_eigenvalues(device, dtype)

        # Reshape to grid
        rhs_grid = b.reshape(self.nx, self.ny)

        # Forward 2D DCT
        try:
            import torch_dct as dct
            # DCT along each dimension
            rhs_dct = dct.dct(dct.dct(rhs_grid, norm='ortho').transpose(-1, -2),
                              norm='ortho').transpose(-1, -2)
        except ImportError:
            # Fallback: implement DCT via FFT
            rhs_dct = self._dct2_via_fft(rhs_grid)

        # Solve in spectral space
        u_dct = rhs_dct / self._denom

        # Handle zero frequency (Neumann singularity)
        # Set mean to zero or preserve it
        u_dct[0, 0] = rhs_dct[0, 0] / self._denom[0, 0] if self._denom[0, 0].abs() > 1e-10 else 0.0

        # Inverse 2D DCT
        try:
            import torch_dct as dct
            u_grid = dct.idct(dct.idct(u_dct, norm='ortho').transpose(-1, -2),
                              norm='ortho').transpose(-1, -2)
        except ImportError:
            u_grid = self._idct2_via_fft(u_dct)

        return u_grid.flatten()

    def _dct2_via_fft(self, x: torch.Tensor) -> torch.Tensor:
        """
        2D DCT-II via FFT (fallback if torch-dct not available).

        DCT-II can be computed from FFT by mirroring and taking real part.
        """
        nx, ny = x.shape

        # DCT along first dimension
        x_mirror = torch.cat([x, x.flip(0)], dim=0)
        fft_x = torch.fft.rfft(x_mirror, dim=0)
        # Extract DCT coefficients
        k = torch.arange(nx, device=x.device, dtype=x.dtype)
        phase = torch.exp(-1j * torch.pi * k / (2 * nx))
        dct_x = (fft_x[:nx] * phase.unsqueeze(1)).real * (2.0 / nx) ** 0.5
        dct_x[0] *= (0.5) ** 0.5

        # DCT along second dimension
        dct_x_mirror = torch.cat([dct_x, dct_x.flip(1)], dim=1)
        fft_y = torch.fft.rfft(dct_x_mirror, dim=1)
        k = torch.arange(ny, device=x.device, dtype=x.dtype)
        phase = torch.exp(-1j * torch.pi * k / (2 * ny))
        dct_xy = (fft_y[:, :ny] * phase.unsqueeze(0)).real * (2.0 / ny) ** 0.5
        dct_xy[:, 0] *= (0.5) ** 0.5

        return dct_xy

    def _idct2_via_fft(self, x: torch.Tensor) -> torch.Tensor:
        """
        2D IDCT (DCT-III) via FFT (fallback if torch-dct not available).
        """
        nx, ny = x.shape

        # Scale for inverse
        x_scaled = x.clone()
        x_scaled[0, :] *= (2.0) ** 0.5
        x_scaled[:, 0] *= (2.0) ** 0.5
        x_scaled *= (nx * ny / 4.0) ** 0.5

        # IDCT along second dimension
        k = torch.arange(ny, device=x.device, dtype=x.dtype)
        phase = torch.exp(1j * torch.pi * k / (2 * ny))
        x_complex = x_scaled * phase.unsqueeze(0)
        x_padded = torch.cat([x_complex, torch.zeros_like(x_complex)], dim=1)
        idct_y = torch.fft.irfft(x_padded, n=2*ny, dim=1)[:, :ny]

        # IDCT along first dimension
        k = torch.arange(nx, device=x.device, dtype=x.dtype)
        phase = torch.exp(1j * torch.pi * k / (2 * nx))
        idct_y_complex = idct_y * phase.unsqueeze(1)
        idct_y_padded = torch.cat([idct_y_complex, torch.zeros_like(idct_y_complex)], dim=0)
        result = torch.fft.irfft(idct_y_padded, n=2*nx, dim=0)[:nx, :]

        return result.real


class FFTSolver(LinearSolver):
    """
    FFT-based direct solver for structured grids with periodic BCs.

    Uses Fast Fourier Transform to diagonalize the discrete Laplacian.
    O(N log N) solve time.

    IMPORTANT: Only use with periodic boundary conditions. For cardiac
    simulation, DCTSolver with Neumann BCs is usually more appropriate.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions
    dx, dy : float
        Grid spacing (cm)
    dt : float
        Time step (ms)
    D : float
        Diffusion coefficient (cm²/ms)
    chi : float
        Surface-to-volume ratio (cm⁻¹)
    Cm : float
        Membrane capacitance (µF/cm²)
    scheme : str
        Time stepping scheme: 'CN', 'BDF1', 'BDF2'
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float,
        dy: float,
        dt: float,
        D: float,
        chi: float = 1400.0,
        Cm: float = 1.0,
        scheme: str = 'CN'
    ):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.D = D
        self.chi = chi
        self.Cm = Cm
        self.scheme = scheme.upper()

        self._eigenvalues: Optional[torch.Tensor] = None
        self._denom: Optional[torch.Tensor] = None
        self._device: Optional[torch.device] = None

    def _compute_eigenvalues(self, device: torch.device, dtype: torch.dtype) -> None:
        """
        Compute Laplacian eigenvalues for periodic BCs.

        λ_{i,j} = -4sin²(πi/Nx)/dx² - 4sin²(πj/Ny)/dy²

        Using FFT wavenumber formulation:
        k_i = 2π * fftfreq(N) / dx
        λ = -k² for Laplacian
        """
        # Wavenumber grids
        kx = torch.fft.fftfreq(self.nx, d=self.dx, device=device) * 2 * torch.pi
        ky = torch.fft.fftfreq(self.ny, d=self.dy, device=device) * 2 * torch.pi

        KX, KY = torch.meshgrid(kx, ky, indexing='ij')
        K2 = KX**2 + KY**2

        # Laplacian eigenvalues: -k² (negative semi-definite)
        self._eigenvalues = -K2

        chi_Cm = self.chi * self.Cm
        D_lambda = self.D * self._eigenvalues  # D times Laplacian eigenvalues

        if self.scheme == 'CN':
            self._denom = chi_Cm - 0.5 * self.dt * D_lambda
        elif self.scheme == 'BDF1':
            self._denom = chi_Cm - self.dt * D_lambda
        elif self.scheme == 'BDF2':
            self._denom = 3.0 * chi_Cm - 2.0 * self.dt * D_lambda
        else:
            raise ValueError(f"Unknown scheme: {self.scheme}")

        # Handle zero frequency
        if self._denom[0, 0].abs() < 1e-10:
            self._denom[0, 0] = 1.0

        self._device = device

    def solve(self, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Solve Ax = b using FFT.

        Parameters
        ----------
        A : torch.Tensor
            Sparse system matrix (ignored)
        b : torch.Tensor
            Right-hand side vector, shape (nx*ny,)

        Returns
        -------
        x : torch.Tensor
            Solution vector, shape (nx*ny,)
        """
        device = b.device
        dtype = b.dtype

        if self._eigenvalues is None or self._device != device:
            self._compute_eigenvalues(device, dtype)

        rhs_grid = b.reshape(self.nx, self.ny)

        # Forward 2D FFT
        rhs_fft = torch.fft.fft2(rhs_grid)

        # Solve in spectral space
        u_fft = rhs_fft / self._denom

        # Handle zero frequency
        u_fft[0, 0] = 0.0  # Set mean to zero

        # Inverse 2D FFT
        u_grid = torch.fft.ifft2(u_fft).real

        return u_grid.flatten()
