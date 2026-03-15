"""
2D Anisotropic Diffusion Module
===============================

Numba-accelerated diffusion operators for cardiac tissue.

Supports:
- Isotropic diffusion (D scalar)
- Anisotropic diffusion (D tensor with fiber orientation)
- Neumann (no-flux) boundary conditions
- Pre-scaling by dt for operator splitting

The diffusion equation: dV/dt = (1/Cm) * div(D * grad(V))

For anisotropic case with fiber angle theta:
    D = R(theta) @ diag(D_parallel, D_perp) @ R(theta)^T

This gives tensor components:
    D_xx = D_parallel * cos^2(theta) + D_perp * sin^2(theta)
    D_yy = D_parallel * sin^2(theta) + D_perp * cos^2(theta)
    D_xy = (D_parallel - D_perp) * cos(theta) * sin(theta)

NOTE: For the monodomain equation, diffusion is applied to voltage V [mV]:
    dV/dt = div(D*grad(V)) - I_ion/Cm

References:
- Keener & Sneyd, Mathematical Physiology
- Sundnes et al., Computing the Electrical Activity in the Heart

Author: Generated with Claude Code
Date: 2025-12-11
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

    Computes: u_new = u + dt * D * Laplacian(u)

    Parameters are pre-scaled:
        D_dt_dx2 = D * dt / dx^2
        D_dt_dy2 = D * dt / dy^2

    Parameters
    ----------
    u : np.ndarray (ny, nx)
        Current field (voltage in mV for cardiac)
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

    # Neumann BC: du/dn = 0 (no flux at boundaries)
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

    Computes: du/dt = d/dx(Dxx du/dx) + d/dy(Dyy du/dy) + 2*d/dx(Dxy du/dy)

    For uniform tensor, this simplifies to:
        = Dxx * d^2u/dx^2 + Dyy * d^2u/dy^2 + 2*Dxy * d^2u/dxdy

    Pre-scaled parameters:
        Dxx_dt_dx2 = Dxx * dt / dx^2
        Dyy_dt_dy2 = Dyy * dt / dy^2
        Dxy_dt_4dxdy = Dxy * dt / (4 * dx * dy)

    Parameters
    ----------
    u : np.ndarray (ny, nx)
        Current field (voltage in mV)
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

            # Cross derivative: d^2u/dxdy using central difference
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
        dt: float = 0.005,
    ):
        """
        Initialize diffusion operator.

        Parameters
        ----------
        spatial : SpatialParams
            Spatial configuration
        dt : float
            Time step [ms] - default 0.005 ms for LRd94
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
            Current voltage field [mV]
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

        For monodomain with diffusion D and upstroke velocity:
            CV ~ sqrt(D * dV/dt_max)

        This is a rough estimate; actual CV depends on ionic model.

        Returns
        -------
        cv : float
            Estimated CV [mm/ms]
        """
        D = self.spatial.D_parallel
        # For LRd94, dV/dt_max ~ 400 mV/ms, but we use simplified estimate
        # CV ~ 0.5 mm/ms for typical parameters
        cv = np.sqrt(D) * 1.5  # Empirical factor
        return cv

    def summary(self) -> str:
        """Generate diffusion operator summary."""
        lines = [
            "Diffusion Operator Configuration",
            "=" * 40,
            "",
            "Spatial Parameters:",
            f"  Domain: {self.spatial.Lx} x {self.spatial.Ly} mm",
            f"  Grid: {self.spatial.nx} x {self.spatial.ny} points",
            f"  dx = {self.spatial.dx} mm, dy = {self.spatial.dy} mm",
            "",
            "Diffusion Coefficients:",
            f"  D_parallel = {self.spatial.D_parallel} mm^2/ms",
            f"  D_perp = {self.spatial.D_perp} mm^2/ms",
            f"  Anisotropy = {self.spatial.anisotropy_ratio:.1f}:1",
            f"  Fiber angle = {self.spatial.fiber_angle} deg",
            "",
            "Diffusion Tensor:",
            f"  Dxx = {self.Dxx:.4f} mm^2/ms",
            f"  Dyy = {self.Dyy:.4f} mm^2/ms",
            f"  Dxy = {self.Dxy:.4f} mm^2/ms",
            "",
            f"Pre-scaled Coefficients (dt = {self.dt:.4f} ms):",
            f"  Dxx*dt/dx^2 = {self.Dxx_dt_dx2:.6f}",
            f"  Dyy*dt/dy^2 = {self.Dyy_dt_dy2:.6f}",
            f"  Dxy*dt/(4dx*dy) = {self.Dxy_dt_4dxdy:.6f}",
            "",
            "Stability:",
            f"  CFL number = {self.cfl_number:.4f}",
            f"  Status: {'STABLE' if self.is_stable else 'UNSTABLE'}",
            "",
            f"Theoretical CV ~ {self.compute_cv_theoretical():.2f} mm/ms",
        ]
        return "\n".join(lines)


# =============================================================================
# Factory Function
# =============================================================================

def create_diffusion_operator(
    domain_size: float = 80.0,
    resolution: float = 0.5,
    D_parallel: float = 0.1,
    D_perp: float = 0.05,
    fiber_angle: float = 0.0,
    dt: float = 0.005,
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
        Diffusion along fibers [mm^2/ms]
    D_perp : float
        Diffusion perpendicular to fibers [mm^2/ms]
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
    print("=" * 60)
    print("DIFFUSION OPERATOR TEST")
    print("=" * 60)

    # Create operator with LRd94 time step
    diffusion = create_diffusion_operator(
        domain_size=40.0,
        resolution=0.5,
        D_parallel=0.1,
        D_perp=0.05,
        fiber_angle=0.0,
        dt=0.005,  # LRd94 time step
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

    # Initial Gaussian centered at domain center (in voltage units - mV)
    x0, y0 = diffusion.spatial.Lx / 2, diffusion.spatial.Ly / 2
    sigma = 2.0  # mm
    # Start at resting potential with Gaussian perturbation
    V_rest = -84.0
    V_peak = 40.0
    u0 = V_rest + (V_peak - V_rest) * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))

    print(f"Initial V range: [{np.min(u0):.1f}, {np.max(u0):.1f}] mV")

    # Run diffusion
    u = u0.copy()
    n_steps = 1000
    for _ in range(n_steps):
        u = diffusion.step(u, anisotropic=True)

    t_final = n_steps * diffusion.dt
    print(f"\nAfter {t_final:.1f} ms ({n_steps} steps):")
    print(f"Final V range: [{np.min(u):.1f}, {np.max(u):.1f}] mV")

    # Test performance
    print("\n" + "-" * 40)
    print("Performance Test")
    print("-" * 40)

    import time

    u_test = np.random.rand(161, 161).astype(np.float64) * 100 - 84  # Random voltages

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
    print(f"Grid size: 161x161 = 25,921 points")

    print("\nDiffusion test complete!")
