"""
2D Cardiac Tissue Mesh Module
==============================

Rectangular mesh for cardiac electrophysiology simulations.

Key Features:
- Rectangular domain with uniform spacing
- Anisotropic diffusion tensor (2:1 longitudinal:transverse)
- Fiber orientation field (left-to-right aligned)
- Storage for voltage V(x,y,t) and recovery w(x,y,t)

Physical Units:
- Space: mm
- Diffusion: mm²/ms
- Voltage: mV (when converted from normalized)

Author: Generated with Claude Code
Date: 2025-12-05
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

try:
    from .parameters import SpatialScaling, PhysicalConstants
except ImportError:
    from parameters import SpatialScaling, PhysicalConstants


@dataclass
class MeshConfiguration:
    """
    Configuration for 2D cardiac mesh.
    """
    # Domain size [mm]
    Lx: float = 80.0  # Horizontal extent (200% of ~40mm infarct)
    Ly: float = 80.0  # Vertical extent

    # Mesh resolution [mm]
    dx: float = 0.5  # Grid spacing in x
    dy: float = 0.5  # Grid spacing in y

    # Diffusion coefficients [mm²/ms]
    D_longitudinal: float = 1.0   # Along fiber direction
    D_transverse: float = 0.5     # Perpendicular to fibers

    # Fiber orientation (uniform for now)
    fiber_angle: float = 0.0  # Angle in radians (0 = left-to-right)

    def anisotropy_ratio(self) -> float:
        """Return ratio of longitudinal to transverse diffusion."""
        return self.D_longitudinal / self.D_transverse

    def grid_dimensions(self) -> Tuple[int, int]:
        """Calculate number of grid points (nx, ny)."""
        nx = int(np.round(self.Lx / self.dx)) + 1
        ny = int(np.round(self.Ly / self.dy)) + 1
        return nx, ny

    def estimate_time_step(self, safety_factor: float = 0.25) -> float:
        """
        Estimate stable time step for explicit diffusion.

        CFL condition for 2D: dt < safety * dx² / (4 * D_max)

        Parameters
        ----------
        safety_factor : float
            Safety margin (< 0.5 for 2D)

        Returns
        -------
        dt : float
            Suggested time step [ms]
        """
        dx_min = min(self.dx, self.dy)
        D_max = max(self.D_longitudinal, self.D_transverse)

        # 2D requires more conservative time step
        dt_crit = safety_factor * dx_min**2 / (4.0 * D_max)

        return dt_crit

    def validate(self):
        """Check configuration is physically reasonable."""
        assert self.Lx > 0, "Lx must be positive"
        assert self.Ly > 0, "Ly must be positive"
        assert self.dx > 0, "dx must be positive"
        assert self.dy > 0, "dy must be positive"
        assert self.D_longitudinal > 0, "D_longitudinal must be positive"
        assert self.D_transverse > 0, "D_transverse must be positive"
        assert self.dx <= self.Lx, "dx too large for domain"
        assert self.dy <= self.Ly, "dy too large for domain"


class CardiacMesh2D:
    """
    2D rectangular mesh for cardiac tissue simulation.

    Stores voltage V(x,y) and recovery w(x,y) fields.
    Handles anisotropic diffusion with fiber orientation.
    """

    def __init__(self, config: MeshConfiguration):
        """
        Initialize 2D cardiac mesh.

        Parameters
        ----------
        config : MeshConfiguration
            Mesh configuration parameters
        """
        config.validate()
        self.config = config

        # Grid dimensions
        self.nx, self.ny = config.grid_dimensions()

        # Physical coordinates
        self.x = np.linspace(0, config.Lx, self.nx)
        self.y = np.linspace(0, config.Ly, self.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        # State variables (normalized voltage and recovery)
        self.V = np.zeros((self.nx, self.ny), dtype=np.float64)  # Voltage [0, 1]
        self.w = np.zeros((self.nx, self.ny), dtype=np.float64)  # Recovery variable

        # Fiber orientation field (uniform for now)
        self.fiber_angle = np.full((self.nx, self.ny), config.fiber_angle)

        # Precompute diffusion tensor components
        self._compute_diffusion_tensors()

    def _compute_diffusion_tensors(self):
        """
        Precompute anisotropic diffusion tensor components.

        For fiber angle θ:
        D_tensor = R(θ) @ diag(D_long, D_trans) @ R(θ)^T

        where R(θ) is rotation matrix.

        This gives:
        D_xx = D_long * cos²(θ) + D_trans * sin²(θ)
        D_yy = D_long * sin²(θ) + D_trans * cos²(θ)
        D_xy = (D_long - D_trans) * cos(θ) * sin(θ)
        """
        D_long = self.config.D_longitudinal
        D_trans = self.config.D_transverse

        cos_theta = np.cos(self.fiber_angle)
        sin_theta = np.sin(self.fiber_angle)
        cos2_theta = cos_theta**2
        sin2_theta = sin_theta**2

        # Diffusion tensor components
        self.D_xx = D_long * cos2_theta + D_trans * sin2_theta
        self.D_yy = D_long * sin2_theta + D_trans * cos2_theta
        self.D_xy = (D_long - D_trans) * cos_theta * sin_theta

    def reset_state(self, V_init: float = 0.0, w_init: float = 0.0):
        """
        Reset mesh to uniform state.

        Parameters
        ----------
        V_init : float
            Initial normalized voltage [0, 1]
        w_init : float
            Initial recovery variable
        """
        self.V.fill(V_init)
        self.w.fill(w_init)

    def set_initial_condition(
        self,
        V_func: Optional[callable] = None,
        w_func: Optional[callable] = None
    ):
        """
        Set initial conditions using spatial functions.

        Parameters
        ----------
        V_func : callable or None
            Function V(x, y) returning normalized voltage
        w_func : callable or None
            Function w(x, y) returning recovery variable
        """
        if V_func is not None:
            for i in range(self.nx):
                for j in range(self.ny):
                    self.V[i, j] = V_func(self.X[i, j], self.Y[i, j])

        if w_func is not None:
            for i in range(self.nx):
                for j in range(self.ny):
                    self.w[i, j] = w_func(self.X[i, j], self.Y[i, j])

    def find_nearest_point(self, x_target: float, y_target: float) -> Tuple[int, int]:
        """
        Find grid indices closest to physical coordinates.

        Parameters
        ----------
        x_target : float
            Target x coordinate [mm]
        y_target : float
            Target y coordinate [mm]

        Returns
        -------
        i, j : int
            Grid indices
        """
        i = np.argmin(np.abs(self.x - x_target))
        j = np.argmin(np.abs(self.y - y_target))
        return i, j

    def apply_stimulus(
        self,
        x_center: float,
        y_center: float,
        amplitude: float,
        radius: float = 2.0
    ) -> np.ndarray:
        """
        Create localized stimulus at specified location.

        Parameters
        ----------
        x_center : float
            Stimulus x coordinate [mm]
        y_center : float
            Stimulus y coordinate [mm]
        amplitude : float
            Stimulus amplitude (dimensionless)
        radius : float
            Stimulus radius [mm]

        Returns
        -------
        I_stim : np.ndarray (nx, ny)
            Stimulus current array
        """
        I_stim = np.zeros((self.nx, self.ny))

        # Distance from stimulus center
        dist = np.sqrt((self.X - x_center)**2 + (self.Y - y_center)**2)

        # Apply Gaussian stimulus
        I_stim = amplitude * np.exp(-(dist / radius)**2)

        return I_stim

    def compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        """
        Compute anisotropic Laplacian: ∇·(D∇field)

        For anisotropic diffusion:
        ∇·(D∇V) = ∂/∂x(D_xx ∂V/∂x + D_xy ∂V/∂y) + ∂/∂y(D_xy ∂V/∂x + D_yy ∂V/∂y)

        Uses central differences with Neumann boundary conditions (no-flux).

        Parameters
        ----------
        field : np.ndarray (nx, ny)
            Field to compute Laplacian of (e.g., V)

        Returns
        -------
        laplacian : np.ndarray (nx, ny)
            Anisotropic Laplacian
        """
        dx = self.config.dx
        dy = self.config.dy

        # Allocate output
        laplacian = np.zeros_like(field)

        # Interior points (central differences)
        # ∂V/∂x at (i,j)
        dV_dx = np.zeros_like(field)
        dV_dx[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2.0 * dx)

        # ∂V/∂y at (i,j)
        dV_dy = np.zeros_like(field)
        dV_dy[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2.0 * dy)

        # Neumann BC: zero derivative at boundaries
        dV_dx[0, :] = (field[1, :] - field[0, :]) / dx
        dV_dx[-1, :] = (field[-1, :] - field[-2, :]) / dx
        dV_dy[:, 0] = (field[:, 1] - field[:, 0]) / dy
        dV_dy[:, -1] = (field[:, -1] - field[:, -2]) / dy

        # Flux components: F_x = D_xx * ∂V/∂x + D_xy * ∂V/∂y
        #                   F_y = D_xy * ∂V/∂x + D_yy * ∂V/∂y
        F_x = self.D_xx * dV_dx + self.D_xy * dV_dy
        F_y = self.D_xy * dV_dx + self.D_yy * dV_dy

        # Divergence: ∂F_x/∂x + ∂F_y/∂y
        laplacian[1:-1, :] += (F_x[2:, :] - F_x[:-2, :]) / (2.0 * dx)
        laplacian[:, 1:-1] += (F_y[:, 2:] - F_y[:, :-2]) / (2.0 * dy)

        # Boundaries (one-sided differences)
        laplacian[0, :] += (F_x[1, :] - F_x[0, :]) / dx
        laplacian[-1, :] += (F_x[-1, :] - F_x[-2, :]) / dx
        laplacian[:, 0] += (F_y[:, 1] - F_y[:, 0]) / dy
        laplacian[:, -1] += (F_y[:, -1] - F_y[:, -2]) / dy

        return laplacian

    def summary(self) -> str:
        """Generate human-readable mesh summary."""
        lines = [
            "=" * 70,
            "2D CARDIAC MESH SUMMARY",
            "=" * 70,
            "",
            "DOMAIN:",
            f"  Size: {self.config.Lx} × {self.config.Ly} mm",
            f"  Grid: {self.nx} × {self.ny} points",
            f"  Spacing: dx = {self.config.dx} mm, dy = {self.config.dy} mm",
            f"  Total points: {self.nx * self.ny:,}",
            "",
            "ANISOTROPY:",
            f"  D_longitudinal = {self.config.D_longitudinal} mm²/ms",
            f"  D_transverse = {self.config.D_transverse} mm²/ms",
            f"  Anisotropy ratio = {self.config.anisotropy_ratio():.1f}:1",
            f"  Fiber angle = {np.degrees(self.config.fiber_angle):.1f}°",
            "",
            "STABILITY:",
            f"  Suggested dt ≤ {self.config.estimate_time_step():.4f} ms",
            f"  With safety factor 0.25",
            "",
            "STATE:",
            f"  V range: [{np.min(self.V):.4f}, {np.max(self.V):.4f}]",
            f"  w range: [{np.min(self.w):.4f}, {np.max(self.w):.4f}]",
            "=" * 70,
        ]
        return "\n".join(lines)


def create_standard_mesh(
    domain_size: float = 80.0,
    resolution: float = 0.5,
    anisotropy: float = 2.0
) -> CardiacMesh2D:
    """
    Create standard cardiac mesh with default settings.

    Parameters
    ----------
    domain_size : float
        Square domain side length [mm]
    resolution : float
        Grid spacing [mm]
    anisotropy : float
        Ratio D_longitudinal / D_transverse

    Returns
    -------
    mesh : CardiacMesh2D
        Configured mesh
    """
    config = MeshConfiguration(
        Lx=domain_size,
        Ly=domain_size,
        dx=resolution,
        dy=resolution,
        D_longitudinal=1.0,
        D_transverse=1.0 / anisotropy,
        fiber_angle=0.0
    )

    mesh = CardiacMesh2D(config)
    return mesh


if __name__ == "__main__":
    print("Testing 2D Cardiac Mesh Module...\n")

    # Create mesh
    mesh = create_standard_mesh(domain_size=80.0, resolution=0.5, anisotropy=2.0)

    print(mesh.summary())

    # Test stimulus
    print("\nTesting stimulus at center:")
    x_stim = mesh.config.Lx / 2
    y_stim = mesh.config.Ly / 2
    I_stim = mesh.apply_stimulus(x_stim, y_stim, amplitude=2.0, radius=2.0)
    print(f"  Stimulus center: ({x_stim}, {y_stim}) mm")
    print(f"  Max stimulus amplitude: {np.max(I_stim):.4f}")
    print(f"  Stimulus points above 0.1: {np.sum(I_stim > 0.1)}")

    # Test Laplacian
    print("\nTesting Laplacian computation:")
    mesh.V = np.random.rand(mesh.nx, mesh.ny) * 0.1  # Small random field
    lap = mesh.compute_laplacian(mesh.V)
    print(f"  Field range: [{np.min(mesh.V):.4f}, {np.max(mesh.V):.4f}]")
    print(f"  Laplacian range: [{np.min(lap):.4f}, {np.max(lap):.4f}]")
    print(f"  Laplacian mean: {np.mean(lap):.6f} (should be ≈0)")

    print("\nMesh module test complete!")
