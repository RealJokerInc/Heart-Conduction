"""
2D Anisotropic Diffusion Module
===============================

Numba-accelerated diffusion operators for cardiac tissue.

Supports:
- Isotropic diffusion (D scalar)
- Anisotropic diffusion (D tensor with fiber orientation)
- Neumann (no-flux) boundary conditions
- Pre-scaling by dt for operator splitting

The diffusion equation: ∂u/∂t = ∇·(D∇u)

For anisotropic case with fiber angle θ:
    D = R(θ) @ diag(D_parallel, D_perp) @ R(θ)^T

This gives tensor components:
    D_xx = D_parallel * cos²(θ) + D_perp * sin²(θ)
    D_yy = D_parallel * sin²(θ) + D_perp * cos²(θ)
    D_xy = (D_parallel - D_perp) * cos(θ) * sin(θ)

References:
- Keener & Sneyd, Mathematical Physiology
- Sundnes et al., Computing the Electrical Activity in the Heart

Author: Generated with Claude Code
Date: 2025-12-10
"""

from __future__ import annotations
import numpy as np
import numba
from typing import Tuple, Optional
from dataclasses import dataclass

from parameters import SpatialParams, default_spatial_params


# =============================================================================
# Numba-Accelerated Diffusion Kernels
# =============================================================================

@numba.jit(nopython=True, cache=True)
def diffusion_step_isotropic_kernel(
    u: np.ndarray,
    D_dt_dx2: float,
    D_dt_dy2: float,
) -> np.ndarray:
    """
    Isotropic diffusion step with Neumann BC.

    Computes: u_new = u + dt * D * ∇²u

    Parameters are pre-scaled:
        D_dt_dx2 = D * dt / dx²
        D_dt_dy2 = D * dt / dy²

    Parameters
    ----------
    u : np.ndarray (ny, nx)
        Current field
    D_dt_dx2 : float
        Pre-scaled diffusion coefficient in x
    D_dt_dy2 : float
        Pre-scaled diffusion coefficient in y

    Returns
    -------
    u_new : np.ndarray (ny, nx)
        Updated field after diffusion
    """
    ny, nx = u.shape
    u_new = np.copy(u)

    # Interior points (standard 5-point stencil)
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            lap_x = u[i, j+1] - 2.0 * u[i, j] + u[i, j-1]
            lap_y = u[i+1, j] - 2.0 * u[i, j] + u[i-1, j]
            u_new[i, j] = u[i, j] + D_dt_dx2 * lap_x + D_dt_dy2 * lap_y

    # Neumann BC: ∂u/∂n = 0 (no flux at boundaries)
    # Left boundary (j=0): u[-1] = u[1] (ghost point)
    for i in range(1, ny - 1):
        lap_x = 2.0 * (u[i, 1] - u[i, 0])  # Forward difference doubled
        lap_y = u[i+1, 0] - 2.0 * u[i, 0] + u[i-1, 0]
        u_new[i, 0] = u[i, 0] + D_dt_dx2 * lap_x + D_dt_dy2 * lap_y

    # Right boundary (j=nx-1)
    for i in range(1, ny - 1):
        lap_x = 2.0 * (u[i, nx-2] - u[i, nx-1])  # Backward difference doubled
        lap_y = u[i+1, nx-1] - 2.0 * u[i, nx-1] + u[i-1, nx-1]
        u_new[i, nx-1] = u[i, nx-1] + D_dt_dx2 * lap_x + D_dt_dy2 * lap_y

    # Bottom boundary (i=0)
    for j in range(1, nx - 1):
        lap_x = u[0, j+1] - 2.0 * u[0, j] + u[0, j-1]
        lap_y = 2.0 * (u[1, j] - u[0, j])
        u_new[0, j] = u[0, j] + D_dt_dx2 * lap_x + D_dt_dy2 * lap_y

    # Top boundary (i=ny-1)
    for j in range(1, nx - 1):
        lap_x = u[ny-1, j+1] - 2.0 * u[ny-1, j] + u[ny-1, j-1]
        lap_y = 2.0 * (u[ny-2, j] - u[ny-1, j])
        u_new[ny-1, j] = u[ny-1, j] + D_dt_dx2 * lap_x + D_dt_dy2 * lap_y

    # Corners
    # (0, 0)
    lap_x = 2.0 * (u[0, 1] - u[0, 0])
    lap_y = 2.0 * (u[1, 0] - u[0, 0])
    u_new[0, 0] = u[0, 0] + D_dt_dx2 * lap_x + D_dt_dy2 * lap_y

    # (0, nx-1)
    lap_x = 2.0 * (u[0, nx-2] - u[0, nx-1])
    lap_y = 2.0 * (u[1, nx-1] - u[0, nx-1])
    u_new[0, nx-1] = u[0, nx-1] + D_dt_dx2 * lap_x + D_dt_dy2 * lap_y

    # (ny-1, 0)
    lap_x = 2.0 * (u[ny-1, 1] - u[ny-1, 0])
    lap_y = 2.0 * (u[ny-2, 0] - u[ny-1, 0])
    u_new[ny-1, 0] = u[ny-1, 0] + D_dt_dx2 * lap_x + D_dt_dy2 * lap_y

    # (ny-1, nx-1)
    lap_x = 2.0 * (u[ny-1, nx-2] - u[ny-1, nx-1])
    lap_y = 2.0 * (u[ny-2, nx-1] - u[ny-1, nx-1])
    u_new[ny-1, nx-1] = u[ny-1, nx-1] + D_dt_dx2 * lap_x + D_dt_dy2 * lap_y

    return u_new


@numba.jit(nopython=True, cache=True)
def diffusion_step_anisotropic_kernel(
    u: np.ndarray,
    Dxx_dt_dx2: float,
    Dyy_dt_dy2: float,
    Dxy_dt_4dxdy: float,
) -> np.ndarray:
    """
    Anisotropic diffusion step with Neumann BC.

    Computes: ∂u/∂t = ∂/∂x(Dxx ∂u/∂x) + ∂/∂y(Dyy ∂u/∂y) + 2·∂/∂x(Dxy ∂u/∂y)

    For uniform tensor, this simplifies to:
        = Dxx · ∂²u/∂x² + Dyy · ∂²u/∂y² + 2·Dxy · ∂²u/∂x∂y

    Pre-scaled parameters:
        Dxx_dt_dx2 = Dxx * dt / dx²
        Dyy_dt_dy2 = Dyy * dt / dy²
        Dxy_dt_4dxdy = Dxy * dt / (4 * dx * dy)

    Parameters
    ----------
    u : np.ndarray (ny, nx)
        Current field
    Dxx_dt_dx2 : float
        Pre-scaled Dxx coefficient
    Dyy_dt_dy2 : float
        Pre-scaled Dyy coefficient
    Dxy_dt_4dxdy : float
        Pre-scaled Dxy coefficient (for cross-derivative)

    Returns
    -------
    u_new : np.ndarray (ny, nx)
        Updated field after diffusion
    """
    ny, nx = u.shape
    u_new = np.copy(u)

    # Interior points
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            # Second derivatives
            d2u_dx2 = u[i, j+1] - 2.0 * u[i, j] + u[i, j-1]
            d2u_dy2 = u[i+1, j] - 2.0 * u[i, j] + u[i-1, j]

            # Cross derivative: ∂²u/∂x∂y using central difference
            d2u_dxdy = (u[i+1, j+1] - u[i+1, j-1] - u[i-1, j+1] + u[i-1, j-1])

            u_new[i, j] = u[i, j] + (
                Dxx_dt_dx2 * d2u_dx2 +
                Dyy_dt_dy2 * d2u_dy2 +
                Dxy_dt_4dxdy * d2u_dxdy
            )

    # Boundaries: Use Neumann BC with isotropic approximation for simplicity
    # (Cross-derivative terms are zero at boundaries due to symmetry)

    # Left boundary (j=0)
    for i in range(1, ny - 1):
        d2u_dx2 = 2.0 * (u[i, 1] - u[i, 0])
        d2u_dy2 = u[i+1, 0] - 2.0 * u[i, 0] + u[i-1, 0]
        u_new[i, 0] = u[i, 0] + Dxx_dt_dx2 * d2u_dx2 + Dyy_dt_dy2 * d2u_dy2

    # Right boundary (j=nx-1)
    for i in range(1, ny - 1):
        d2u_dx2 = 2.0 * (u[i, nx-2] - u[i, nx-1])
        d2u_dy2 = u[i+1, nx-1] - 2.0 * u[i, nx-1] + u[i-1, nx-1]
        u_new[i, nx-1] = u[i, nx-1] + Dxx_dt_dx2 * d2u_dx2 + Dyy_dt_dy2 * d2u_dy2

    # Bottom boundary (i=0)
    for j in range(1, nx - 1):
        d2u_dx2 = u[0, j+1] - 2.0 * u[0, j] + u[0, j-1]
        d2u_dy2 = 2.0 * (u[1, j] - u[0, j])
        u_new[0, j] = u[0, j] + Dxx_dt_dx2 * d2u_dx2 + Dyy_dt_dy2 * d2u_dy2

    # Top boundary (i=ny-1)
    for j in range(1, nx - 1):
        d2u_dx2 = u[ny-1, j+1] - 2.0 * u[ny-1, j] + u[ny-1, j-1]
        d2u_dy2 = 2.0 * (u[ny-2, j] - u[ny-1, j])
        u_new[ny-1, j] = u[ny-1, j] + Dxx_dt_dx2 * d2u_dx2 + Dyy_dt_dy2 * d2u_dy2

    # Corners (pure Neumann, no cross-derivative)
    # (0, 0)
    d2u_dx2 = 2.0 * (u[0, 1] - u[0, 0])
    d2u_dy2 = 2.0 * (u[1, 0] - u[0, 0])
    u_new[0, 0] = u[0, 0] + Dxx_dt_dx2 * d2u_dx2 + Dyy_dt_dy2 * d2u_dy2

    # (0, nx-1)
    d2u_dx2 = 2.0 * (u[0, nx-2] - u[0, nx-1])
    d2u_dy2 = 2.0 * (u[1, nx-1] - u[0, nx-1])
    u_new[0, nx-1] = u[0, nx-1] + Dxx_dt_dx2 * d2u_dx2 + Dyy_dt_dy2 * d2u_dy2

    # (ny-1, 0)
    d2u_dx2 = 2.0 * (u[ny-1, 1] - u[ny-1, 0])
    d2u_dy2 = 2.0 * (u[ny-2, 0] - u[ny-1, 0])
    u_new[ny-1, 0] = u[ny-1, 0] + Dxx_dt_dx2 * d2u_dx2 + Dyy_dt_dy2 * d2u_dy2

    # (ny-1, nx-1)
    d2u_dx2 = 2.0 * (u[ny-1, nx-2] - u[ny-1, nx-1])
    d2u_dy2 = 2.0 * (u[ny-2, nx-1] - u[ny-1, nx-1])
    u_new[ny-1, nx-1] = u[ny-1, nx-1] + Dxx_dt_dx2 * d2u_dx2 + Dyy_dt_dy2 * d2u_dy2

    return u_new


@numba.jit(nopython=True, cache=True, parallel=True)
def diffusion_step_isotropic_inplace(
    u: np.ndarray,
    u_new: np.ndarray,
    D_dt_dx2: float,
    D_dt_dy2: float,
) -> None:
    """
    Isotropic diffusion step (parallel, writes to u_new).

    Same as diffusion_step_isotropic_kernel but with parallel loops
    and pre-allocated output buffer for better performance.
    """
    ny, nx = u.shape

    # Interior points (parallel)
    for i in numba.prange(1, ny - 1):
        for j in range(1, nx - 1):
            lap_x = u[i, j+1] - 2.0 * u[i, j] + u[i, j-1]
            lap_y = u[i+1, j] - 2.0 * u[i, j] + u[i-1, j]
            u_new[i, j] = u[i, j] + D_dt_dx2 * lap_x + D_dt_dy2 * lap_y

    # Boundaries (serial - small number of points)
    # Left (j=0)
    for i in range(1, ny - 1):
        lap_x = 2.0 * (u[i, 1] - u[i, 0])
        lap_y = u[i+1, 0] - 2.0 * u[i, 0] + u[i-1, 0]
        u_new[i, 0] = u[i, 0] + D_dt_dx2 * lap_x + D_dt_dy2 * lap_y

    # Right (j=nx-1)
    for i in range(1, ny - 1):
        lap_x = 2.0 * (u[i, nx-2] - u[i, nx-1])
        lap_y = u[i+1, nx-1] - 2.0 * u[i, nx-1] + u[i-1, nx-1]
        u_new[i, nx-1] = u[i, nx-1] + D_dt_dx2 * lap_x + D_dt_dy2 * lap_y

    # Bottom (i=0)
    for j in range(1, nx - 1):
        lap_x = u[0, j+1] - 2.0 * u[0, j] + u[0, j-1]
        lap_y = 2.0 * (u[1, j] - u[0, j])
        u_new[0, j] = u[0, j] + D_dt_dx2 * lap_x + D_dt_dy2 * lap_y

    # Top (i=ny-1)
    for j in range(1, nx - 1):
        lap_x = u[ny-1, j+1] - 2.0 * u[ny-1, j] + u[ny-1, j-1]
        lap_y = 2.0 * (u[ny-2, j] - u[ny-1, j])
        u_new[ny-1, j] = u[ny-1, j] + D_dt_dx2 * lap_x + D_dt_dy2 * lap_y

    # Corners
    u_new[0, 0] = u[0, 0] + D_dt_dx2 * 2.0 * (u[0, 1] - u[0, 0]) + D_dt_dy2 * 2.0 * (u[1, 0] - u[0, 0])
    u_new[0, nx-1] = u[0, nx-1] + D_dt_dx2 * 2.0 * (u[0, nx-2] - u[0, nx-1]) + D_dt_dy2 * 2.0 * (u[1, nx-1] - u[0, nx-1])
    u_new[ny-1, 0] = u[ny-1, 0] + D_dt_dx2 * 2.0 * (u[ny-1, 1] - u[ny-1, 0]) + D_dt_dy2 * 2.0 * (u[ny-2, 0] - u[ny-1, 0])
    u_new[ny-1, nx-1] = u[ny-1, nx-1] + D_dt_dx2 * 2.0 * (u[ny-1, nx-2] - u[ny-1, nx-1]) + D_dt_dy2 * 2.0 * (u[ny-2, nx-1] - u[ny-1, nx-1])


# =============================================================================
# Diffusion Operator Class
# =============================================================================

class DiffusionOperator:
    """
    Diffusion operator for 2D cardiac tissue.

    Handles:
    - Parameter pre-scaling by dt, dx, dy
    - Isotropic and anisotropic diffusion
    - Fiber angle rotation
    - Stability checking
    """

    def __init__(
        self,
        spatial: SpatialParams = None,
        dt: float = 0.02,
    ):
        """
        Initialize diffusion operator.

        Parameters
        ----------
        spatial : SpatialParams
            Spatial configuration
        dt : float
            Time step [ms]
        """
        self.spatial = spatial or default_spatial_params()
        self.dt = dt

        # Compute diffusion tensor
        self.Dxx, self.Dyy, self.Dxy = self.spatial.diffusion_tensor()

        # Pre-scale coefficients
        self._prescale()

        # Check stability
        self._check_stability()

        # Work buffer for in-place operations
        self._u_buffer = None

    def _prescale(self) -> None:
        """Pre-compute scaled diffusion coefficients."""
        dx = self.spatial.dx
        dy = self.spatial.dy
        dt = self.dt

        # Isotropic (uses average diffusion)
        D_avg = (self.spatial.D_parallel + self.spatial.D_perp) / 2
        self.D_dt_dx2 = D_avg * dt / (dx * dx)
        self.D_dt_dy2 = D_avg * dt / (dy * dy)

        # Anisotropic
        self.Dxx_dt_dx2 = self.Dxx * dt / (dx * dx)
        self.Dyy_dt_dy2 = self.Dyy * dt / (dy * dy)
        self.Dxy_dt_4dxdy = self.Dxy * dt / (4 * dx * dy)

    def _check_stability(self) -> None:
        """Check CFL stability condition."""
        stability = self.spatial.check_stability(self.dt)
        self.cfl_number = stability['r']
        self.is_stable = stability['stable']

        if not self.is_stable:
            print(f"WARNING: Diffusion may be unstable!")
            print(f"  CFL number r = {self.cfl_number:.4f} (should be < 0.25)")
            print(f"  Suggested dt_max = {stability['dt_max']:.4f} ms")

    def set_dt(self, dt: float) -> None:
        """Update time step and re-prescale."""
        self.dt = dt
        self._prescale()
        self._check_stability()

    def step(self, u: np.ndarray, anisotropic: bool = True) -> np.ndarray:
        """
        Perform one diffusion step.

        Parameters
        ----------
        u : np.ndarray (ny, nx)
            Current voltage field
        anisotropic : bool
            Use anisotropic (True) or isotropic (False) diffusion

        Returns
        -------
        u_new : np.ndarray (ny, nx)
            Updated field
        """
        if anisotropic:
            return diffusion_step_anisotropic_kernel(
                u, self.Dxx_dt_dx2, self.Dyy_dt_dy2, self.Dxy_dt_4dxdy
            )
        else:
            return diffusion_step_isotropic_kernel(
                u, self.D_dt_dx2, self.D_dt_dy2
            )

    def step_inplace(self, u: np.ndarray) -> np.ndarray:
        """
        Perform diffusion step with parallel kernel.

        Uses pre-allocated buffer and parallel loops.

        Parameters
        ----------
        u : np.ndarray (ny, nx)
            Current voltage field

        Returns
        -------
        u_new : np.ndarray (ny, nx)
            Updated field (from buffer)
        """
        # Allocate buffer if needed
        if self._u_buffer is None or self._u_buffer.shape != u.shape:
            self._u_buffer = np.empty_like(u)

        diffusion_step_isotropic_inplace(
            u, self._u_buffer, self.D_dt_dx2, self.D_dt_dy2
        )

        return self._u_buffer

    def compute_cv_theoretical(self) -> float:
        """
        Estimate theoretical conduction velocity.

        For FK model with diffusion D:
            CV ≈ √(D / τ_d) * foot_factor

        This is a rough estimate; actual CV depends on
        ionic model parameters.

        Returns
        -------
        cv : float
            Estimated CV [mm/ms]
        """
        # Simple estimate using parallel diffusion and typical tau_d
        D = self.spatial.D_parallel
        tau_d = 0.25  # Typical FK tau_d

        # Empirical foot factor (accounts for wavefront shape)
        foot_factor = 2.0

        cv = np.sqrt(D / tau_d) * foot_factor
        return cv

    def summary(self) -> str:
        """Generate diffusion operator summary."""
        lines = [
            "Diffusion Operator Configuration",
            "=" * 40,
            "",
            "Spatial Parameters:",
            f"  Domain: {self.spatial.Lx} × {self.spatial.Ly} mm",
            f"  Grid: {self.spatial.nx} × {self.spatial.ny} points",
            f"  dx = {self.spatial.dx} mm, dy = {self.spatial.dy} mm",
            "",
            "Diffusion Coefficients:",
            f"  D_parallel = {self.spatial.D_parallel} mm²/ms",
            f"  D_perp = {self.spatial.D_perp} mm²/ms",
            f"  Anisotropy = {self.spatial.anisotropy_ratio:.1f}:1",
            f"  Fiber angle = {self.spatial.fiber_angle}°",
            "",
            "Diffusion Tensor:",
            f"  Dxx = {self.Dxx:.4f} mm²/ms",
            f"  Dyy = {self.Dyy:.4f} mm²/ms",
            f"  Dxy = {self.Dxy:.4f} mm²/ms",
            "",
            "Pre-scaled Coefficients (dt = {:.4f} ms):".format(self.dt),
            f"  Dxx·dt/dx² = {self.Dxx_dt_dx2:.6f}",
            f"  Dyy·dt/dy² = {self.Dyy_dt_dy2:.6f}",
            f"  Dxy·dt/(4dx·dy) = {self.Dxy_dt_4dxdy:.6f}",
            "",
            "Stability:",
            f"  CFL number = {self.cfl_number:.4f}",
            f"  Status: {'✓ STABLE' if self.is_stable else '✗ UNSTABLE'}",
            "",
            f"Theoretical CV ≈ {self.compute_cv_theoretical():.2f} mm/ms",
        ]
        return "\n".join(lines)


# =============================================================================
# Factory Function
# =============================================================================

def create_diffusion_operator(
    domain_size: float = 80.0,
    resolution: float = 0.5,
    D_parallel: float = 1.0,
    D_perp: float = 0.5,
    fiber_angle: float = 0.0,
    dt: float = 0.02,
) -> DiffusionOperator:
    """
    Create diffusion operator with specified parameters.

    Parameters
    ----------
    domain_size : float
        Square domain side [mm]
    resolution : float
        Grid spacing [mm]
    D_parallel : float
        Diffusion along fibers [mm²/ms]
    D_perp : float
        Diffusion perpendicular to fibers [mm²/ms]
    fiber_angle : float
        Fiber orientation [degrees]
    dt : float
        Time step [ms]

    Returns
    -------
    diffusion : DiffusionOperator
        Configured operator
    """
    spatial = SpatialParams(
        Lx=domain_size,
        Ly=domain_size,
        dx=resolution,
        dy=resolution,
        D_parallel=D_parallel,
        D_perp=D_perp,
        fiber_angle=fiber_angle,
    )

    return DiffusionOperator(spatial=spatial, dt=dt)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("DIFFUSION OPERATOR TEST")
    print("=" * 60)

    # Create operator
    diffusion = create_diffusion_operator(
        domain_size=40.0,
        resolution=0.5,
        D_parallel=1.0,
        D_perp=0.5,
        fiber_angle=0.0,
        dt=0.02,
    )

    print(diffusion.summary())

    # Test: Gaussian blob diffusion
    print("\n" + "-" * 40)
    print("Test: Gaussian Blob Diffusion")
    print("-" * 40)

    ny, nx = diffusion.spatial.ny, diffusion.spatial.nx
    x = np.linspace(0, diffusion.spatial.Lx, nx)
    y = np.linspace(0, diffusion.spatial.Ly, ny)
    X, Y = np.meshgrid(x, y)

    # Initial Gaussian centered at domain center
    x0, y0 = diffusion.spatial.Lx / 2, diffusion.spatial.Ly / 2
    sigma = 2.0  # mm
    u0 = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))

    print(f"Initial peak: {np.max(u0):.4f}")
    print(f"Initial integral: {np.sum(u0) * diffusion.spatial.dx * diffusion.spatial.dy:.4f}")

    # Run diffusion
    u = u0.copy()
    n_steps = 1000
    for _ in range(n_steps):
        u = diffusion.step(u, anisotropic=True)

    t_final = n_steps * diffusion.dt
    print(f"\nAfter {t_final:.1f} ms ({n_steps} steps):")
    print(f"Final peak: {np.max(u):.4f}")
    print(f"Final integral: {np.sum(u) * diffusion.spatial.dx * diffusion.spatial.dy:.4f}")
    print(f"Mass conservation: {np.sum(u) / np.sum(u0) * 100:.2f}%")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(u0, origin='lower', extent=[0, diffusion.spatial.Lx, 0, diffusion.spatial.Ly],
                          cmap='hot', vmin=0, vmax=1)
    axes[0].set_title(f'Initial (t = 0 ms)', fontsize=12)
    axes[0].set_xlabel('x [mm]')
    axes[0].set_ylabel('y [mm]')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(u, origin='lower', extent=[0, diffusion.spatial.Lx, 0, diffusion.spatial.Ly],
                          cmap='hot', vmin=0, vmax=1)
    axes[1].set_title(f'After {t_final:.1f} ms (anisotropic)', fontsize=12)
    axes[1].set_xlabel('x [mm]')
    axes[1].set_ylabel('y [mm]')
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.savefig('images/diffusion_test.png', dpi=150)
    print(f"\nPlot saved: images/diffusion_test.png")

    # Test performance
    print("\n" + "-" * 40)
    print("Performance Test")
    print("-" * 40)

    import time

    u_test = np.random.rand(161, 161).astype(np.float64)

    # Warm up JIT
    _ = diffusion.step(u_test, anisotropic=True)
    _ = diffusion.step(u_test, anisotropic=False)

    # Time isotropic
    n_iter = 1000
    start = time.perf_counter()
    for _ in range(n_iter):
        u_test = diffusion.step(u_test, anisotropic=False)
    elapsed_iso = time.perf_counter() - start

    # Time anisotropic
    start = time.perf_counter()
    for _ in range(n_iter):
        u_test = diffusion.step(u_test, anisotropic=True)
    elapsed_aniso = time.perf_counter() - start

    print(f"Isotropic: {elapsed_iso/n_iter*1000:.3f} ms/step")
    print(f"Anisotropic: {elapsed_aniso/n_iter*1000:.3f} ms/step")
    print(f"Grid size: 161×161 = 25,921 points")

    print("\nDiffusion test complete!")
