"""
Bidomain Finite Difference Method (FDM) Matrix Assembly for Cardiac Simulation Engine V5.4

This module provides reference implementations for assembling the intracellular (Ki) and
extracellular (Ke) Laplacian matrices using finite difference discretization on structured grids.

Discrete Laplacian operators discretize:
    L[Vm] = ∇·(σ·∇Vm)

on a structured 2D/3D grid. The module supports:
    - Anisotropic conductivity tensors
    - Per-node conductivity values
    - Multiple stencil types (5-point, 9-point in 2D)
    - Neumann boundary conditions
    - Periodic boundary conditions

References:
    [1] Sundnes et al. "Computing the Electrical Activity in the Heart" (2006)
    [2] LeVeque "Finite Difference Methods for Ordinary and Partial Differential
        Equations: Steady-State and Time-Dependent Problems" (2007)
    [3] Clayton et al. "Electromechanical Re-Entry Near Fixed Points in the Heart" (2005)
    [4] Colli Franzone et al. "Mathematical and Numerical Methods for the Forward and
        Inverse Electrocardiography Problem" (2014)
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
import math


# ============================================================================
# Type Definitions
# ============================================================================

@dataclass
class ConductivityTensor:
    """Anisotropic conductivity tensor in material frame."""
    # Longitudinal, transverse, normal directions (ms/cm)
    sigma_l: float = 0.3    # Along fiber
    sigma_t: float = 0.05   # Transverse
    sigma_n: float = 0.05   # Normal (out of plane)

    # Fiber orientation (Euler angles in degrees)
    theta_x: float = 0.0    # Rotation around x-axis (elevation)
    theta_y: float = 0.0    # Rotation around y-axis
    theta_z: float = 0.0    # Rotation around z-axis (in-plane rotation)

    def rotation_matrix(self) -> torch.Tensor:
        """
        Compute 3×3 rotation matrix from Euler angles (ZYX convention).

        Returns:
            (3, 3) rotation matrix
        """
        # Convert degrees to radians
        rx = math.radians(self.theta_x)
        ry = math.radians(self.theta_y)
        rz = math.radians(self.theta_z)

        # Rotation matrices
        Rx = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, math.cos(rx), -math.sin(rx)],
            [0.0, math.sin(rx), math.cos(rx)]
        ])

        Ry = torch.tensor([
            [math.cos(ry), 0.0, math.sin(ry)],
            [0.0, 1.0, 0.0],
            [-math.sin(ry), 0.0, math.cos(ry)]
        ])

        Rz = torch.tensor([
            [math.cos(rz), -math.sin(rz), 0.0],
            [math.sin(rz), math.cos(rz), 0.0],
            [0.0, 0.0, 1.0]
        ])

        # Combine: R = Rz @ Ry @ Rx
        return Rz @ Ry @ Rx

    def conductivity_matrix(self) -> torch.Tensor:
        """
        Compute 3×3 conductivity matrix in global frame.

        σ_global = R @ σ_local @ R^T

        Returns:
            (3, 3) symmetric positive-definite conductivity tensor
        """
        R = self.rotation_matrix()

        # Diagonal conductivity in material frame
        sigma_diag = torch.diag(torch.tensor([self.sigma_l, self.sigma_t, self.sigma_n]))

        # Transform to global frame
        sigma_global = R @ sigma_diag @ R.T

        return sigma_global


@dataclass
class GridParams:
    """Structured grid parameters."""
    nx: int = 64          # Number of nodes in x
    ny: int = 64          # Number of nodes in y
    nz: int = 1           # Number of nodes in z (1 for 2D)
    dx: float = 0.01      # Grid spacing in x (cm)
    dy: float = 0.01      # Grid spacing in y (cm)
    dz: float = 0.01      # Grid spacing in z (cm)

    def total_nodes(self) -> int:
        """Total number of nodes."""
        return self.nx * self.ny * self.nz


# ============================================================================
# Abstract Base Classes
# ============================================================================

class MatrixAssembler(ABC):
    """Base class for matrix assembly."""

    @abstractmethod
    def assemble(self) -> torch.sparse.FloatTensor:
        """Assemble and return sparse matrix."""
        pass


# ============================================================================
# 2D FDM Laplacian Assembly
# ============================================================================

class FDM2DLaplacian(MatrixAssembler):
    """
    Assemble 2D finite difference Laplacian matrices.

    Supports:
        - 5-point stencil (standard)
        - 9-point stencil (higher accuracy)
        - Per-node anisotropic conductivity
        - Boundary conditions (Neumann, periodic)
    """

    def __init__(
        self,
        grid: GridParams,
        conductivity: Optional[Dict[str, torch.Tensor]] = None,
        stencil_type: str = "5-point",
        boundary_type: str = "neumann",
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            grid: Grid configuration
            conductivity: Dict with keys 'sigma_xx', 'sigma_xy', 'sigma_yy'
                         If None, isotropic (σ = 1.0)
            stencil_type: "5-point" or "9-point"
            boundary_type: "neumann" or "periodic"
            device: torch device
        """
        self.grid = grid
        self.conductivity = conductivity
        self.stencil_type = stencil_type
        self.boundary_type = boundary_type
        self.device = device

        # Validate
        assert grid.nz == 1, "FDM2D: nz must be 1 for 2D"

        self.n = grid.nx * grid.ny
        self.nx = grid.nx
        self.ny = grid.ny
        self.dx = grid.dx
        self.dy = grid.dy

        # Default to isotropic conductivity if not provided
        if conductivity is None:
            self.conductivity = {
                'sigma_xx': torch.ones((grid.ny, grid.nx), device=device),
                'sigma_xy': torch.zeros((grid.ny, grid.nx), device=device),
                'sigma_yy': torch.ones((grid.ny, grid.nx), device=device)
            }
        else:
            self.conductivity = {
                k: v.to(device) if isinstance(v, torch.Tensor) else torch.tensor(v, device=device)
                for k, v in conductivity.items()
            }

    def node_index(self, i: int, j: int) -> int:
        """Convert (i, j) grid coordinates to linear index."""
        return i * self.nx + j

    def assemble(self) -> torch.sparse.FloatTensor:
        """
        Assemble 2D FDM Laplacian matrix.

        For anisotropic case:
            ∫∫ ∇u · σ · ∇v dxdy

        Finite difference approximation on a structured grid.
        """
        if self.stencil_type == "5-point":
            return self._assemble_5point()
        elif self.stencil_type == "9-point":
            return self._assemble_9point()
        else:
            raise ValueError(f"Unknown stencil type: {self.stencil_type}")

    def _assemble_5point(self) -> torch.sparse.FloatTensor:
        """
        Assemble 5-point stencil (center + 4 neighbors).

        For isotropic σ=1:
            L[u]_i,j = (u_{i-1,j} - 2*u_{i,j} + u_{i+1,j})/dx²
                     + (u_{i,j-1} - 2*u_{i,j} + u_{i,j+1})/dy²

        For anisotropic σ, use weighted differences.
        """
        rows, cols, vals = [], [], []

        sigma_xx = self.conductivity.get('sigma_xx')
        sigma_xy = self.conductivity.get('sigma_xy')
        sigma_yy = self.conductivity.get('sigma_yy')

        for i in range(self.ny):
            for j in range(self.nx):
                idx_center = self.node_index(i, j)

                # Get local conductivity
                sig_xx = sigma_xx[i, j].item() if sigma_xx is not None else 1.0
                sig_xy = sigma_xy[i, j].item() if sigma_xy is not None else 0.0
                sig_yy = sigma_yy[i, j].item() if sigma_yy is not None else 1.0

                # Coefficients
                cx_main = -2.0 * sig_xx / (self.dx**2)
                cy_main = -2.0 * sig_yy / (self.dy**2)
                cx_side = sig_xx / (self.dx**2)
                cy_side = sig_yy / (self.dy**2)
                cxy = sig_xy / (self.dx * self.dy)

                # Center coefficient
                coeff_center = cx_main + cy_main

                # Contributions
                rows.append(idx_center)
                cols.append(idx_center)
                vals.append(coeff_center)

                # Right neighbor (j+1)
                if j < self.nx - 1:
                    idx_right = self.node_index(i, j + 1)
                    rows.append(idx_center)
                    cols.append(idx_right)
                    vals.append(cx_side + cxy * 0.25)
                elif self.boundary_type == "periodic":
                    idx_right = self.node_index(i, 0)
                    rows.append(idx_center)
                    cols.append(idx_right)
                    vals.append(cx_side + cxy * 0.25)

                # Left neighbor (j-1)
                if j > 0:
                    idx_left = self.node_index(i, j - 1)
                    rows.append(idx_center)
                    cols.append(idx_left)
                    vals.append(cx_side - cxy * 0.25)
                elif self.boundary_type == "periodic":
                    idx_left = self.node_index(i, self.nx - 1)
                    rows.append(idx_center)
                    cols.append(idx_left)
                    vals.append(cx_side - cxy * 0.25)

                # Top neighbor (i-1)
                if i > 0:
                    idx_top = self.node_index(i - 1, j)
                    rows.append(idx_center)
                    cols.append(idx_top)
                    vals.append(cy_side - cxy * 0.25)
                elif self.boundary_type == "periodic":
                    idx_top = self.node_index(self.ny - 1, j)
                    rows.append(idx_center)
                    cols.append(idx_top)
                    vals.append(cy_side - cxy * 0.25)

                # Bottom neighbor (i+1)
                if i < self.ny - 1:
                    idx_bottom = self.node_index(i + 1, j)
                    rows.append(idx_center)
                    cols.append(idx_bottom)
                    vals.append(cy_side + cxy * 0.25)
                elif self.boundary_type == "periodic":
                    idx_bottom = self.node_index(0, j)
                    rows.append(idx_center)
                    cols.append(idx_bottom)
                    vals.append(cy_side + cxy * 0.25)

        # Create sparse tensor
        indices = torch.tensor([rows, cols], dtype=torch.long, device=self.device)
        values = torch.tensor(vals, dtype=torch.float32, device=self.device)

        matrix = torch.sparse_coo_tensor(
            indices, values, (self.n, self.n),
            device=self.device
        ).coalesce()

        return matrix

    def _assemble_9point(self) -> torch.sparse.FloatTensor:
        """
        Assemble 9-point stencil (center + 8 neighbors).

        Uses higher-order approximations for better accuracy.
        For isotropic case, standard 9-point stencil is:

            1  4  1
           4 -20 4    / (6*dx²)
            1  4  1

        For anisotropic, extend with mixed derivatives.
        """
        rows, cols, vals = [], [], []

        sigma_xx = self.conductivity.get('sigma_xx')
        sigma_xy = self.conductivity.get('sigma_xy')
        sigma_yy = self.conductivity.get('sigma_yy')

        for i in range(self.ny):
            for j in range(self.nx):
                idx_center = self.node_index(i, j)

                sig_xx = sigma_xx[i, j].item() if sigma_xx is not None else 1.0
                sig_xy = sigma_xy[i, j].item() if sigma_xy is not None else 0.0
                sig_yy = sigma_yy[i, j].item() if sigma_yy is not None else 1.0

                # 9-point stencil coefficients
                scale = 1.0 / (6.0 * self.dx**2)

                # Cardinal directions (weight 4)
                c_cardinal = 4.0 * scale

                # Diagonal directions (weight 1)
                c_diag = 1.0 * scale

                # Center (weight -20)
                c_center = -20.0 * scale

                # Apply to all 9 points
                stencil_points = [
                    (i, j, c_center, 0, 0),           # center
                    (i, j + 1, c_cardinal, 1, 0),     # right
                    (i, j - 1, c_cardinal, -1, 0),    # left
                    (i + 1, j, c_cardinal, 0, 1),     # bottom
                    (i - 1, j, c_cardinal, 0, -1),    # top
                    (i + 1, j + 1, c_diag, 1, 1),     # bottom-right
                    (i + 1, j - 1, c_diag, -1, 1),    # bottom-left
                    (i - 1, j + 1, c_diag, 1, -1),    # top-right
                    (i - 1, j - 1, c_diag, -1, -1)    # top-left
                ]

                for i_n, j_n, coeff, di, dj in stencil_points:
                    # Handle boundaries
                    i_n_bc = i_n
                    j_n_bc = j_n

                    if self.boundary_type == "periodic":
                        i_n_bc = i_n_bc % self.ny
                        j_n_bc = j_n_bc % self.nx
                    else:
                        if i_n_bc < 0 or i_n_bc >= self.ny or j_n_bc < 0 or j_n_bc >= self.nx:
                            continue  # Skip boundary points (Neumann BC)

                    idx_neighbor = self.node_index(i_n_bc, j_n_bc)

                    # Anisotropy correction
                    if di != 0:
                        coeff *= sig_xx
                    elif dj != 0:
                        coeff *= sig_yy
                    else:
                        coeff *= (2.0 * sig_xx + 2.0 * sig_yy)  # Center

                    rows.append(idx_center)
                    cols.append(idx_neighbor)
                    vals.append(coeff)

        # Create sparse tensor
        indices = torch.tensor([rows, cols], dtype=torch.long, device=self.device)
        values = torch.tensor(vals, dtype=torch.float32, device=self.device)

        matrix = torch.sparse_coo_tensor(
            indices, values, (self.n, self.n),
            device=self.device
        ).coalesce()

        return matrix


# ============================================================================
# Bidomain FDM Assembly
# ============================================================================

class BidomainFDMAssembler:
    """
    Assemble intracellular (Ki) and extracellular (Ke) matrices for bidomain using FDM.
    """

    def __init__(
        self,
        grid: GridParams,
        intracellular_conductivity: ConductivityTensor,
        extracellular_conductivity: ConductivityTensor,
        stencil_type: str = "5-point",
        boundary_type: str = "neumann",
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            grid: Grid parameters
            intracellular_conductivity: σi tensor
            extracellular_conductivity: σe tensor
            stencil_type: "5-point" or "9-point"
            boundary_type: "neumann" or "periodic"
            device: torch device
        """
        self.grid = grid
        self.sigma_i = intracellular_conductivity
        self.sigma_e = extracellular_conductivity
        self.stencil_type = stencil_type
        self.boundary_type = boundary_type
        self.device = device

        self._Ki = None
        self._Ke = None

    def assemble_intracellular(self) -> torch.sparse.FloatTensor:
        """Assemble intracellular Laplacian Ki."""
        if self._Ki is not None:
            return self._Ki

        sigma_matrix = self.sigma_i.conductivity_matrix()

        # Extract 2D components (for 2D case)
        conductivity_2d = {
            'sigma_xx': sigma_matrix[0, 0].item(),
            'sigma_xy': sigma_matrix[0, 1].item(),
            'sigma_yy': sigma_matrix[1, 1].item()
        }

        assembler = FDM2DLaplacian(
            grid=self.grid,
            conductivity=conductivity_2d,
            stencil_type=self.stencil_type,
            boundary_type=self.boundary_type,
            device=self.device
        )

        self._Ki = assembler.assemble()
        return self._Ki

    def assemble_extracellular(self) -> torch.sparse.FloatTensor:
        """Assemble extracellular Laplacian Ke."""
        if self._Ke is not None:
            return self._Ke

        sigma_matrix = self.sigma_e.conductivity_matrix()

        conductivity_2d = {
            'sigma_xx': sigma_matrix[0, 0].item(),
            'sigma_xy': sigma_matrix[0, 1].item(),
            'sigma_yy': sigma_matrix[1, 1].item()
        }

        assembler = FDM2DLaplacian(
            grid=self.grid,
            conductivity=conductivity_2d,
            stencil_type=self.stencil_type,
            boundary_type=self.boundary_type,
            device=self.device
        )

        self._Ke = assembler.assemble()
        return self._Ke


# ============================================================================
# Main Demo/Test
# ============================================================================

def main():
    """Demonstration of bidomain FDM assembly."""
    print("=" * 80)
    print("BIDOMAIN FDM MATRIX ASSEMBLY - REFERENCE IMPLEMENTATION")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Setup
    print("\n" + "=" * 80)
    print("Test 1: Isotropic Conductivity (5-point stencil)")
    print("=" * 80)

    grid = GridParams(nx=32, ny=32, dx=0.01, dy=0.01)
    print(f"Grid: {grid.nx}×{grid.ny}, dx={grid.dx}, dy={grid.dy}")
    print(f"Total nodes: {grid.total_nodes()}")

    # Isotropic conductivity
    sigma_i_iso = ConductivityTensor(sigma_l=0.3, sigma_t=0.3, sigma_n=0.3)
    sigma_e_iso = ConductivityTensor(sigma_l=0.2, sigma_t=0.2, sigma_n=0.2)

    print(f"\nIntracellular conductivity (isotropic): {sigma_i_iso.sigma_l} mS/cm")
    print(f"Extracellular conductivity (isotropic): {sigma_e_iso.sigma_l} mS/cm")

    assembler_iso = BidomainFDMAssembler(
        grid=grid,
        intracellular_conductivity=sigma_i_iso,
        extracellular_conductivity=sigma_e_iso,
        stencil_type="5-point",
        boundary_type="neumann",
        device=device
    )

    Ki_iso = assembler_iso.assemble_intracellular()
    Ke_iso = assembler_iso.assemble_extracellular()

    print(f"\nIntracellular matrix Ki:")
    print(f"  Shape: {Ki_iso.shape}")
    print(f"  Nonzeros: {Ki_iso._nnz()}")
    print(f"  Density: {Ki_iso._nnz() / (grid.total_nodes()**2) * 100:.2f}%")
    print(f"  Condition number (approx): {Ki_iso.to_dense().abs().max() / Ki_iso.to_dense().abs().min():.2e}")

    print(f"\nExtracellular matrix Ke:")
    print(f"  Shape: {Ke_iso.shape}")
    print(f"  Nonzeros: {Ke_iso._nnz()}")

    # Test matrix-vector product
    x_test = torch.ones(grid.total_nodes(), device=device)
    y_Ki = torch.sparse.mm(Ki_iso, x_test.unsqueeze(1)).squeeze(1)
    y_Ke = torch.sparse.mm(Ke_iso, x_test.unsqueeze(1)).squeeze(1)

    print(f"\nMatrix-vector product test:")
    print(f"  ||Ki @ ones||: {torch.linalg.norm(y_Ki):.6e}")
    print(f"  ||Ke @ ones||: {torch.linalg.norm(y_Ke):.6e}")

    # Test 2: Anisotropic conductivity
    print("\n" + "=" * 80)
    print("Test 2: Anisotropic Conductivity (with fiber orientation)")
    print("=" * 80)

    sigma_i_aniso = ConductivityTensor(
        sigma_l=0.3, sigma_t=0.05, sigma_n=0.05,
        theta_z=45.0  # Rotate 45° in-plane
    )
    sigma_e_aniso = ConductivityTensor(
        sigma_l=0.2, sigma_t=0.1, sigma_n=0.1,
        theta_z=45.0
    )

    print(f"Intracellular: l={sigma_i_aniso.sigma_l}, t={sigma_i_aniso.sigma_t}, "
          f"θ_z={sigma_i_aniso.theta_z}°")

    # Rotation matrix check
    R_i = sigma_i_aniso.rotation_matrix()
    print(f"Rotation matrix determinant (should be ≈1): {torch.det(R_i):.6f}")

    assembler_aniso = BidomainFDMAssembler(
        grid=grid,
        intracellular_conductivity=sigma_i_aniso,
        extracellular_conductivity=sigma_e_aniso,
        stencil_type="5-point",
        boundary_type="periodic",
        device=device
    )

    Ki_aniso = assembler_aniso.assemble_intracellular()
    print(f"\nAnisotropic Ki:")
    print(f"  Nonzeros: {Ki_aniso._nnz()}")
    print(f"  Density: {Ki_aniso._nnz() / (grid.total_nodes()**2) * 100:.2f}%")

    # Test 3: Different stencils
    print("\n" + "=" * 80)
    print("Test 3: Stencil Comparison (5-point vs 9-point)")
    print("=" * 80)

    assembler_5pt = BidomainFDMAssembler(
        grid=grid,
        intracellular_conductivity=sigma_i_iso,
        extracellular_conductivity=sigma_e_iso,
        stencil_type="5-point",
        device=device
    )

    assembler_9pt = BidomainFDMAssembler(
        grid=grid,
        intracellular_conductivity=sigma_i_iso,
        extracellular_conductivity=sigma_e_iso,
        stencil_type="9-point",
        device=device
    )

    Ki_5pt = assembler_5pt.assemble_intracellular()
    Ki_9pt = assembler_9pt.assemble_intracellular()

    print(f"5-point stencil Ki:")
    print(f"  Nonzeros: {Ki_5pt._nnz()}")

    print(f"9-point stencil Ki:")
    print(f"  Nonzeros: {Ki_9pt._nnz()}")

    # Difference between stencils
    Ki_5pt_dense = Ki_5pt.to_dense()
    Ki_9pt_dense = Ki_9pt.to_dense()
    diff = torch.linalg.norm(Ki_5pt_dense - Ki_9pt_dense)

    print(f"Difference between 5-point and 9-point: {diff:.6e}")

    # Spectral properties
    print("\n" + "=" * 80)
    print("Spectral Properties")
    print("=" * 80)

    # Compute eigenvalues of diagonal block (small test)
    if grid.total_nodes() <= 256:
        Ki_dense = Ki_iso.to_dense()
        try:
            eigenvalues = torch.linalg.eigvalsh(Ki_dense)
            print(f"Smallest eigenvalue: {eigenvalues.min():.6e}")
            print(f"Largest eigenvalue: {eigenvalues.max():.6e}")
            print(f"Condition number (exact): {eigenvalues.max() / torch.clamp(eigenvalues.min(), min=1e-10):.6e}")
        except:
            print("Could not compute eigenvalues (matrix may be singular)")

    # Symmetry check
    Ki_dense = Ki_iso.to_dense()
    symmetry_error = torch.norm(Ki_dense - Ki_dense.T) / torch.norm(Ki_dense)
    print(f"Symmetry error ||A - A^T||/||A||: {symmetry_error:.6e}")

    # Performance benchmark
    print("\n" + "=" * 80)
    print("Performance Benchmark")
    print("=" * 80)

    x_test = torch.randn(grid.total_nodes(), device=device)
    num_trials = 100

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    import time
    t_start = time.perf_counter()

    for _ in range(num_trials):
        y = torch.sparse.mm(Ki_iso, x_test.unsqueeze(1)).squeeze(1)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_end = time.perf_counter()

    t_per_iter = (t_end - t_start) / num_trials * 1000

    flops = Ki_iso._nnz() * 2  # Multiply-add per nonzero
    throughput = flops / (t_per_iter * 1e-3) / 1e9

    print(f"Matrix-vector product time: {t_per_iter:.4f} ms")
    print(f"Throughput: {throughput:.2f} GFLOP/s")

    print("\n" + "=" * 80)
    print("Reference implementation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
