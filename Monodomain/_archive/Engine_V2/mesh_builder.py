"""
Mesh Builder for Advanced Cardiac Simulations
===============================================

Flexible mesh generation system for cardiac electrophysiology with:
- 2D structured meshes with configurable resolution
- Fiber direction fields (uniform, rotational, or custom)
- Infarct/scar masks (circular, rectangular, or custom shapes)
- Proper boundary condition handling

This module extends the basic mesh_2d.py with tools for creating
complex geometries needed for reentry, spiral waves, and infarct studies.

Key Features:
-------------
1. MeshGeometry: Combines mesh grid + fiber directions + tissue masks
2. Fiber field generators: uniform, rotational (transmural), custom
3. Infarct masks: circular, rectangular, multiple regions
4. Boundary conditions: no-flux (Neumann) at all borders

Physical Units:
---------------
- Space: mm (millimeters)
- Fiber angles: radians
- Diffusion: mm²/ms

Author: Generated with Claude Code
Date: 2025-12-09
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Callable, List
from dataclasses import dataclass
import matplotlib.pyplot as plt

try:
    from .mesh_2d import CardiacMesh2D, MeshConfiguration
except ImportError:
    from mesh_2d import CardiacMesh2D, MeshConfiguration


@dataclass
class MeshGeometry:
    """
    Complete mesh geometry with fibers and tissue masks.

    Attributes
    ----------
    mesh : CardiacMesh2D
        Base 2D cardiac mesh
    fiber_angles : np.ndarray (ny, nx)
        Fiber direction at each point [radians]
        0 = horizontal (left to right), π/2 = vertical
    tissue_mask : np.ndarray (ny, nx), bool
        True = healthy tissue, False = infarct/scar (non-conductive)
    Dxx, Dyy, Dxy : np.ndarray (ny, nx)
        Anisotropic diffusion tensor components [mm²/ms]
        Automatically computed from fiber_angles and tissue_mask
    """
    mesh: CardiacMesh2D
    fiber_angles: np.ndarray
    tissue_mask: np.ndarray
    Dxx: np.ndarray
    Dyy: np.ndarray
    Dxy: np.ndarray

    def summary(self) -> str:
        """Generate summary of mesh geometry."""
        healthy_fraction = np.sum(self.tissue_mask) / self.tissue_mask.size
        angle_range = (np.degrees(np.min(self.fiber_angles)),
                      np.degrees(np.max(self.fiber_angles)))

        lines = [
            "=" * 70,
            "MESH GEOMETRY SUMMARY",
            "=" * 70,
            "",
            f"Domain: {self.mesh.config.Lx} × {self.mesh.config.Ly} mm",
            f"Grid: {self.mesh.nx} × {self.mesh.ny} points",
            f"Spacing: dx={self.mesh.config.dx}, dy={self.mesh.config.dy} mm",
            "",
            "TISSUE:",
            f"  Healthy tissue: {np.sum(self.tissue_mask):,} / {self.tissue_mask.size:,} points ({healthy_fraction*100:.1f}%)",
            f"  Infarct/scar: {np.sum(~self.tissue_mask):,} points ({(1-healthy_fraction)*100:.1f}%)",
            "",
            "FIBERS:",
            f"  Angle range: [{angle_range[0]:.1f}°, {angle_range[1]:.1f}°]",
            f"  Mean angle: {np.degrees(np.mean(self.fiber_angles)):.1f}°",
            "",
            "DIFFUSION:",
            f"  D_parallel: {self.mesh.config.D_longitudinal} mm²/ms",
            f"  D_perpendicular: {self.mesh.config.D_transverse} mm²/ms",
            f"  Anisotropy ratio: {self.mesh.config.anisotropy_ratio():.1f}:1",
            "=" * 70,
        ]
        return "\n".join(lines)

    def visualize(self, save_path: Optional[str] = None):
        """
        Visualize mesh geometry (tissue mask and fiber directions).

        Parameters
        ----------
        save_path : str or None
            Path to save figure (if None, display only)
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Panel 1: Tissue mask
        ax = axes[0]
        im1 = ax.imshow(
            self.tissue_mask.T,  # Transpose for proper orientation
            origin='lower',
            extent=[0, self.mesh.config.Lx, 0, self.mesh.config.Ly],
            cmap='RdYlGn',
            alpha=0.8,
            aspect='equal'
        )
        ax.set_xlabel('x (mm)', fontsize=12)
        ax.set_ylabel('y (mm)', fontsize=12)
        ax.set_title('Tissue Mask (Green=Healthy, Red=Infarct)', fontsize=12, fontweight='bold')
        fig.colorbar(im1, ax=ax, label='Conductive (1=yes, 0=no)')

        # Panel 2: Fiber directions
        ax = axes[1]

        # Show fiber angles as colored background
        im2 = ax.imshow(
            np.degrees(self.fiber_angles.T),
            origin='lower',
            extent=[0, self.mesh.config.Lx, 0, self.mesh.config.Ly],
            cmap='twilight',
            alpha=0.6,
            aspect='equal'
        )

        # Overlay with arrow field (subsampled)
        skip = max(self.mesh.nx // 20, 1)
        X, Y = np.meshgrid(
            np.linspace(0, self.mesh.config.Lx, self.mesh.nx),
            np.linspace(0, self.mesh.config.Ly, self.mesh.ny)
        )
        U = np.cos(self.fiber_angles)
        V = np.sin(self.fiber_angles)

        ax.quiver(
            X[::skip, ::skip], Y[::skip, ::skip],
            U[::skip, ::skip], V[::skip, ::skip],
            color='black', alpha=0.7, scale=20
        )

        ax.set_xlabel('x (mm)', fontsize=12)
        ax.set_ylabel('y (mm)', fontsize=12)
        ax.set_title('Fiber Directions', fontsize=12, fontweight='bold')
        fig.colorbar(im2, ax=ax, label='Fiber angle (degrees)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved geometry visualization to {save_path}")

        return fig


class MeshBuilder:
    """
    Builder class for creating complex cardiac mesh geometries.

    Provides methods to construct meshes with custom fiber fields
    and infarct patterns.
    """

    def __init__(self, config: MeshConfiguration):
        """
        Initialize mesh builder.

        Parameters
        ----------
        config : MeshConfiguration
            Base mesh configuration (domain size, resolution, diffusion)
        """
        self.config = config
        self.nx = int(np.round(config.Lx / config.dx)) + 1
        self.ny = int(np.round(config.Ly / config.dy)) + 1

        # Coordinate grids
        self.x = np.linspace(0, config.Lx, self.nx)
        self.y = np.linspace(0, config.Ly, self.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

    # =========================================================================
    # Fiber Direction Generators
    # =========================================================================

    def uniform_fibers(self, angle: float = 0.0) -> np.ndarray:
        """
        Create uniform fiber direction field.

        Parameters
        ----------
        angle : float
            Fiber angle in radians (0 = horizontal left-to-right)

        Returns
        -------
        fiber_angles : np.ndarray (nx, ny)
            Fiber angle at each grid point [radians]
        """
        return np.full((self.nx, self.ny), angle)

    def rotational_fibers(
        self,
        angle_min: float = -60.0,  # degrees
        angle_max: float = 60.0,   # degrees
        rotation_axis: str = 'y'   # 'x' or 'y'
    ) -> np.ndarray:
        """
        Create linearly rotating fiber field (transmural-like).

        Mimics transmural fiber rotation seen in ventricular wall.

        Parameters
        ----------
        angle_min : float
            Minimum fiber angle [degrees]
        angle_max : float
            Maximum fiber angle [degrees]
        rotation_axis : str
            Axis along which fibers rotate ('x' or 'y')

        Returns
        -------
        fiber_angles : np.ndarray (nx, ny)
            Fiber angles [radians]
        """
        # Convert to radians
        theta_min = np.radians(angle_min)
        theta_max = np.radians(angle_max)

        fiber_angles = np.zeros((self.nx, self.ny))

        if rotation_axis == 'y':
            # Rotate along y-axis (vertical)
            for j in range(self.ny):
                frac = j / (self.ny - 1)
                theta = theta_min + frac * (theta_max - theta_min)
                fiber_angles[:, j] = theta
        elif rotation_axis == 'x':
            # Rotate along x-axis (horizontal)
            for i in range(self.nx):
                frac = i / (self.nx - 1)
                theta = theta_min + frac * (theta_max - theta_min)
                fiber_angles[i, :] = theta
        else:
            raise ValueError("rotation_axis must be 'x' or 'y'")

        return fiber_angles

    def custom_fibers(self, fiber_func: Callable[[float, float], float]) -> np.ndarray:
        """
        Create custom fiber field from user function.

        Parameters
        ----------
        fiber_func : callable
            Function theta(x, y) returning fiber angle [radians]

        Returns
        -------
        fiber_angles : np.ndarray (nx, ny)
            Fiber angles [radians]
        """
        fiber_angles = np.zeros((self.nx, self.ny))

        for i in range(self.nx):
            for j in range(self.ny):
                fiber_angles[i, j] = fiber_func(self.X[i, j], self.Y[i, j])

        return fiber_angles

    # =========================================================================
    # Tissue Mask Generators (Infarct Patterns)
    # =========================================================================

    def healthy_tissue(self) -> np.ndarray:
        """
        Create mask with all tissue healthy (no infarcts).

        Returns
        -------
        tissue_mask : np.ndarray (nx, ny), bool
            All True (fully conductive)
        """
        return np.ones((self.nx, self.ny), dtype=bool)

    def circular_infarct(
        self,
        center_x: float,
        center_y: float,
        radius: float
    ) -> np.ndarray:
        """
        Create circular infarct (non-conductive region).

        Parameters
        ----------
        center_x, center_y : float
            Center coordinates [mm]
        radius : float
            Infarct radius [mm]

        Returns
        -------
        tissue_mask : np.ndarray (nx, ny), bool
            True = healthy, False = infarct
        """
        # Distance from center
        dist = np.sqrt((self.X - center_x)**2 + (self.Y - center_y)**2)

        # Mask: True outside infarct, False inside
        tissue_mask = dist > radius

        return tissue_mask

    def rectangular_infarct(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float
    ) -> np.ndarray:
        """
        Create rectangular infarct region.

        Parameters
        ----------
        x_min, x_max : float
            x-extent of infarct [mm]
        y_min, y_max : float
            y-extent of infarct [mm]

        Returns
        -------
        tissue_mask : np.ndarray (nx, ny), bool
            True = healthy, False = infarct
        """
        tissue_mask = np.ones((self.nx, self.ny), dtype=bool)

        # Find indices in rectangular region
        in_region = (
            (self.X >= x_min) & (self.X <= x_max) &
            (self.Y >= y_min) & (self.Y <= y_max)
        )

        tissue_mask[in_region] = False

        return tissue_mask

    def custom_infarct(self, mask_func: Callable[[float, float], bool]) -> np.ndarray:
        """
        Create custom infarct pattern from user function.

        Parameters
        ----------
        mask_func : callable
            Function is_healthy(x, y) returning bool
            True = healthy tissue, False = infarct

        Returns
        -------
        tissue_mask : np.ndarray (nx, ny), bool
        """
        tissue_mask = np.zeros((self.nx, self.ny), dtype=bool)

        for i in range(self.nx):
            for j in range(self.ny):
                tissue_mask[i, j] = mask_func(self.X[i, j], self.Y[i, j])

        return tissue_mask

    def combine_masks(self, masks: List[np.ndarray], operation: str = 'and') -> np.ndarray:
        """
        Combine multiple tissue masks.

        Parameters
        ----------
        masks : list of np.ndarray
            List of boolean tissue masks to combine
        operation : str
            'and' = tissue healthy only if all masks True (multiple infarcts)
            'or' = tissue healthy if any mask True (union of healthy regions)

        Returns
        -------
        combined_mask : np.ndarray (nx, ny), bool
        """
        if not masks:
            return self.healthy_tissue()

        combined = masks[0].copy()

        for mask in masks[1:]:
            if operation == 'and':
                combined = combined & mask
            elif operation == 'or':
                combined = combined | mask
            else:
                raise ValueError("operation must be 'and' or 'or'")

        return combined

    # =========================================================================
    # Diffusion Tensor Computation
    # =========================================================================

    def compute_diffusion_tensor(
        self,
        fiber_angles: np.ndarray,
        tissue_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute anisotropic diffusion tensor from fiber angles and tissue mask.

        For fiber angle θ at each point:
        D = R(θ) @ diag(D_parallel, D_perp) @ R(θ)^T

        where R(θ) is rotation matrix.

        In infarct regions (tissue_mask=False), all diffusion coefficients set to zero.

        Parameters
        ----------
        fiber_angles : np.ndarray (nx, ny)
            Fiber directions [radians]
        tissue_mask : np.ndarray (nx, ny), bool
            True = healthy, False = infarct

        Returns
        -------
        Dxx, Dyy, Dxy : np.ndarray (nx, ny)
            Diffusion tensor components [mm²/ms]
        """
        D_long = self.config.D_longitudinal
        D_trans = self.config.D_transverse

        cos_theta = np.cos(fiber_angles)
        sin_theta = np.sin(fiber_angles)

        # Tensor components (standard rotation formula)
        Dxx = D_long * cos_theta**2 + D_trans * sin_theta**2
        Dyy = D_long * sin_theta**2 + D_trans * cos_theta**2
        Dxy = (D_long - D_trans) * cos_theta * sin_theta

        # Zero out infarct regions
        Dxx = np.where(tissue_mask, Dxx, 0.0)
        Dyy = np.where(tissue_mask, Dyy, 0.0)
        Dxy = np.where(tissue_mask, Dxy, 0.0)

        return Dxx, Dyy, Dxy

    # =========================================================================
    # Complete Geometry Builder
    # =========================================================================

    def build(
        self,
        fiber_angles: np.ndarray,
        tissue_mask: np.ndarray
    ) -> MeshGeometry:
        """
        Build complete mesh geometry.

        Parameters
        ----------
        fiber_angles : np.ndarray (nx, ny)
            Fiber direction field [radians]
        tissue_mask : np.ndarray (nx, ny), bool
            Tissue conductivity mask

        Returns
        -------
        geometry : MeshGeometry
            Complete mesh geometry with diffusion tensors
        """
        # Create base mesh
        mesh = CardiacMesh2D(self.config)

        # Compute diffusion tensors
        Dxx, Dyy, Dxy = self.compute_diffusion_tensor(fiber_angles, tissue_mask)

        # Create geometry object
        geometry = MeshGeometry(
            mesh=mesh,
            fiber_angles=fiber_angles,
            tissue_mask=tissue_mask,
            Dxx=Dxx,
            Dyy=Dyy,
            Dxy=Dxy
        )

        return geometry


# =============================================================================
# Convenience Functions
# =============================================================================

def create_infarct_mesh(
    domain_size: float = 80.0,
    resolution: float = 0.5,
    infarct_radius: float = 10.0,
    infarct_center: Optional[Tuple[float, float]] = None,
    D_parallel: float = 1.0,
    D_perp: float = 0.5
) -> MeshGeometry:
    """
    Create mesh with circular infarct and uniform horizontal fibers.

    Parameters
    ----------
    domain_size : float
        Square domain side length [mm]
    resolution : float
        Grid spacing [mm]
    infarct_radius : float
        Radius of circular infarct [mm]
    infarct_center : (x, y) or None
        Center of infarct (if None, uses domain center)
    D_parallel : float
        Longitudinal diffusivity [mm²/ms]
    D_perp : float
        Transverse diffusivity [mm²/ms]

    Returns
    -------
    geometry : MeshGeometry
        Mesh with circular infarct
    """
    # Configuration
    config = MeshConfiguration(
        Lx=domain_size,
        Ly=domain_size,
        dx=resolution,
        dy=resolution,
        D_longitudinal=D_parallel,
        D_transverse=D_perp,
        fiber_angle=0.0
    )

    # Builder
    builder = MeshBuilder(config)

    # Fibers (uniform, horizontal)
    fibers = builder.uniform_fibers(angle=0.0)

    # Infarct mask
    if infarct_center is None:
        infarct_center = (domain_size / 2.0, domain_size / 2.0)

    tissue_mask = builder.circular_infarct(
        center_x=infarct_center[0],
        center_y=infarct_center[1],
        radius=infarct_radius
    )

    # Build geometry
    geometry = builder.build(fibers, tissue_mask)

    return geometry


def create_spiral_mesh(
    domain_size: float = 80.0,
    resolution: float = 0.5,
    D_parallel: float = 1.0,
    D_perp: float = 0.5,
    fiber_rotation: bool = False
) -> MeshGeometry:
    """
    Create mesh suitable for spiral wave simulations.

    Parameters
    ----------
    domain_size : float
        Square domain side length [mm]
    resolution : float
        Grid spacing [mm]
    D_parallel : float
        Longitudinal diffusivity [mm²/ms]
    D_perp : float
        Transverse diffusivity [mm²/ms]
    fiber_rotation : bool
        If True, use rotational fibers; if False, uniform horizontal

    Returns
    -------
    geometry : MeshGeometry
        Clean mesh for spiral waves (no infarcts)
    """
    config = MeshConfiguration(
        Lx=domain_size,
        Ly=domain_size,
        dx=resolution,
        dy=resolution,
        D_longitudinal=D_parallel,
        D_transverse=D_perp,
        fiber_angle=0.0
    )

    builder = MeshBuilder(config)

    # Fibers
    if fiber_rotation:
        fibers = builder.rotational_fibers(
            angle_min=-60.0,
            angle_max=60.0,
            rotation_axis='y'
        )
    else:
        fibers = builder.uniform_fibers(angle=0.0)

    # No infarcts (all healthy)
    tissue_mask = builder.healthy_tissue()

    # Build
    geometry = builder.build(fibers, tissue_mask)

    return geometry


if __name__ == "__main__":
    print("Testing Mesh Builder Module...\n")

    # Test 1: Infarct mesh
    print("=" * 70)
    print("TEST 1: Circular Infarct Mesh")
    print("=" * 70)

    geom_infarct = create_infarct_mesh(
        domain_size=80.0,
        resolution=0.5,
        infarct_radius=10.0,
        infarct_center=None  # Center
    )

    print(geom_infarct.summary())
    geom_infarct.visualize(save_path='test_infarct_geometry.png')

    # Test 2: Spiral mesh
    print("\n" + "=" * 70)
    print("TEST 2: Spiral Wave Mesh")
    print("=" * 70)

    geom_spiral = create_spiral_mesh(
        domain_size=80.0,
        resolution=0.5,
        fiber_rotation=False
    )

    print(geom_spiral.summary())
    geom_spiral.visualize(save_path='test_spiral_geometry.png')

    print("\n✓ Mesh builder tests complete!")
