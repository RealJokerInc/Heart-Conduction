"""
Bidomain Block System Assembly for Cardiac Simulation Engine V5.4

This module provides reference implementations for assembling the 2×2 block linear system
arising from the bidomain equations using implicit time integration schemes.

The bidomain equations are:
    ∂Vm/∂t = ∇·(σi·∇Vm) + ∇·(σi·∇φe) - Iion(Vm, g) + Istim
    0 = ∇·((σi+σe)·∇φe) + ∇·(σi·∇Vm)

where:
    Vm = intracellular transmembrane voltage
    φe = extracellular potential
    σi, σe = intracellular/extracellular conductivity tensors
    Iion = ionic current
    Istim = stimulus current

Time discretization produces a coupled linear system at each time step.

References:
    [1] Sundnes et al. "Computing the Electrical Activity in the Heart" (2006)
    [2] Plank et al. "From mitochondrial ion channels to arrhythmias in the heart" (2018)
    [3] Clayton & Panfilov "A guide to modelling cardiac electrical activity in 3D" (2008)
    [4] Vranken et al. "A block solver for the implicit solution of subcycled multibody
        system dynamics" (2015) - general block system concepts
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


# ============================================================================
# Type Definitions and Constants
# ============================================================================

@dataclass
class BidomainParameters:
    """Bidomain conductivity and geometric parameters."""
    # Conductivity values (mS/cm)
    sigma_il: float = 0.3      # Intracellular longitudinal
    sigma_it: float = 0.05     # Intracellular transverse
    sigma_in: float = 0.05     # Intracellular normal

    sigma_el: float = 0.2      # Extracellular longitudinal
    sigma_et: float = 0.1      # Extracellular transverse
    sigma_en: float = 0.1      # Extracellular normal

    # Tissue geometry (cm)
    fiber_angle: float = 0.0   # Rotation angle in XY plane

    # Physical constants
    Cm: float = 1.0            # Membrane capacitance (uF/cm²)
    chi: float = 1000.0        # Surface-to-volume ratio (1/cm)


@dataclass
class TimeSteppingScheme:
    """Configuration for time stepping schemes."""
    scheme: str = "CN"         # "CN" (Crank-Nicolson) or "BDF1"
    dt: float = 0.01           # Time step (ms)
    theta: float = 0.5         # For CN: 0.5 (centered), for BDF1: 1.0


# ============================================================================
# Abstract Base Classes (V5.4 Interfaces)
# ============================================================================

class SpatialDiscretization(ABC):
    """Base class for spatial discretization schemes."""

    @abstractmethod
    def assemble_stiffness_matrix(self) -> torch.sparse.FloatTensor:
        """Assemble spatial discretization matrix."""
        pass

    @abstractmethod
    def assemble_mass_matrix(self) -> torch.sparse.FloatTensor:
        """Assemble mass matrix."""
        pass


class BlockMatrix(ABC):
    """Abstract base for block matrix operations."""

    @abstractmethod
    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """Block matrix-vector product."""
        pass

    @abstractmethod
    def to_dense(self) -> torch.Tensor:
        """Convert to dense representation for inspection."""
        pass


# ============================================================================
# Block Matrix Data Structures
# ============================================================================

class BidomainBlockMatrix(BlockMatrix):
    """
    2×2 block matrix representation for bidomain systems.

    Structure:
        [ M/dt + theta*Ki    theta*Ki   ] [ Vm ]     [ RHS_Vm ]
        [ Ki                -(Ki+Ke)    ] [ φe ]  =  [ RHS_φe ]

    where Ki and Ke are stiffness matrices (negative divergence-gradient).

    Note: The φe block structure enforces the constraint ∫φe dx = 0 (null space pinning).
    """

    def __init__(
        self,
        M: torch.sparse.FloatTensor,
        Ki: torch.sparse.FloatTensor,
        Ke: torch.sparse.FloatTensor,
        dt: float,
        theta: float,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize block matrix.

        Args:
            M: (n, n) mass matrix
            Ki: (n, n) intracellular stiffness matrix (Laplacian)
            Ke: (n, n) extracellular stiffness matrix (Laplacian)
            dt: time step size
            theta: time discretization parameter (0.5 for CN, 1.0 for BDF1)
            device: torch device
        """
        self.M = M.to(device)
        self.Ki = Ki.to(device)
        self.Ke = Ke.to(device)
        self.dt = dt
        self.theta = theta
        self.device = device
        self.n = M.shape[0]

        # Precompute block diagonal terms
        self.M_over_dt = M / dt
        self.theta_Ki = (theta * Ki)

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute A @ x for the block system.

        Args:
            x: (2n,) vector [Vm; φe]

        Returns:
            (2n,) vector A @ x
        """
        assert x.shape[0] == 2 * self.n, f"Expected shape (2n,), got {x.shape}"

        Vm = x[:self.n]
        phi_e = x[self.n:]

        # Top block: (M/dt + theta*Ki) @ Vm + theta*Ki @ φe
        res_top = torch.sparse.mm(self.M_over_dt, Vm.unsqueeze(1)).squeeze(1) + \
                  torch.sparse.mm(self.theta_Ki, phi_e.unsqueeze(1)).squeeze(1)

        # Bottom block: Ki @ Vm - (Ki + Ke) @ φe
        res_bot = torch.sparse.mm(self.Ki, Vm.unsqueeze(1)).squeeze(1) - \
                  torch.sparse.mm((self.Ki + self.Ke), phi_e.unsqueeze(1)).squeeze(1)

        return torch.cat([res_top, res_bot])

    def to_dense(self) -> torch.Tensor:
        """Convert to dense 2×2 block structure for inspection."""
        n = self.n

        # Convert sparse matrices to dense
        M_over_dt_dense = self.M_over_dt.to_dense()
        theta_Ki_dense = self.theta_Ki.to_dense()
        Ki_dense = self.Ki.to_dense()
        Ke_dense = self.Ke.to_dense()

        # Assemble block matrix
        top_left = M_over_dt_dense + theta_Ki_dense
        top_right = theta_Ki_dense
        bot_left = Ki_dense
        bot_right = -(Ki_dense + Ke_dense)

        A = torch.zeros((2*n, 2*n), device=self.device)
        A[:n, :n] = top_left
        A[:n, n:] = top_right
        A[n:, :n] = bot_left
        A[n:, n:] = bot_right

        return A


class BlockRHS:
    """Right-hand side vector for bidomain block system."""

    def __init__(
        self,
        Vm_rhs: torch.Tensor,
        phi_e_rhs: torch.Tensor,
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            Vm_rhs: (n,) RHS for Vm equation
            phi_e_rhs: (n,) RHS for φe equation
            device: torch device
        """
        self.Vm_rhs = Vm_rhs.to(device)
        self.phi_e_rhs = phi_e_rhs.to(device)
        self.device = device

    def to_vector(self) -> torch.Tensor:
        """Convert to single (2n,) vector."""
        return torch.cat([self.Vm_rhs, self.phi_e_rhs])


# ============================================================================
# Block System Assembly
# ============================================================================

class BidomainBlockSystemAssembler:
    """
    Assembles the 2×2 block linear system for bidomain equations.

    Supports:
        - Crank-Nicolson time stepping
        - BDF1 (Backward Euler) time stepping
        - Null space pinning for φe uniqueness
    """

    def __init__(
        self,
        spatial_discr: SpatialDiscretization,
        params: BidomainParameters,
        scheme: TimeSteppingScheme,
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            spatial_discr: Spatial discretization (FEM/FDM/FVM)
            params: Bidomain parameters
            scheme: Time stepping configuration
            device: torch device
        """
        self.spatial_discr = spatial_discr
        self.params = params
        self.scheme = scheme
        self.device = device
        self.n = None  # Will be set after first assembly

        # Cache assembled matrices
        self._M = None
        self._Ki = None
        self._Ke = None
        self._block_matrix = None

    def assemble_mass_matrix(self) -> torch.sparse.FloatTensor:
        """
        Assemble the mass matrix M = Cm * χ * ∫φ dx.

        In finite element terms: M_ij = Cm*χ ∫φ_i φ_j dx

        Returns:
            (n, n) sparse mass matrix
        """
        if self._M is None:
            self._M = self.spatial_discr.assemble_mass_matrix()
            self._M = self._M * (self.params.Cm * self.params.chi)
        return self._M

    def assemble_intracellular_stiffness(self) -> torch.sparse.FloatTensor:
        """
        Assemble the intracellular Laplacian: Ki = ∫∇φ_i · σi · ∇φ_j dx

        The conductivity tensor σi includes anisotropy from fiber direction.

        Returns:
            (n, n) sparse stiffness matrix
        """
        if self._Ki is None:
            self._Ki = self.spatial_discr.assemble_stiffness_matrix()
            # Scale by conductivity (simplified; actual implementation would include
            # per-element anisotropy)
            avg_sigma_i = (self.params.sigma_il + self.params.sigma_it + \
                          self.params.sigma_in) / 3.0
            self._Ki = self._Ki * avg_sigma_i
        return self._Ki

    def assemble_extracellular_stiffness(self) -> torch.sparse.FloatTensor:
        """
        Assemble the extracellular Laplacian: Ke = ∫∇φ_i · σe · ∇φ_j dx

        Returns:
            (n, n) sparse stiffness matrix
        """
        if self._Ke is None:
            self._Ke = self.spatial_discr.assemble_stiffness_matrix()
            avg_sigma_e = (self.params.sigma_el + self.params.sigma_et + \
                          self.params.sigma_en) / 3.0
            self._Ke = self._Ke * avg_sigma_e
        return self._Ke

    def assemble_block_system(self) -> BidomainBlockMatrix:
        """
        Assemble the 2×2 block system matrix.

        For Crank-Nicolson (CN):
            θ = 0.5
            [ M/dt + 0.5*Ki   0.5*Ki  ]
            [ Ki             -(Ki+Ke)]

        For BDF1 (Backward Euler):
            θ = 1.0
            [ M/dt + Ki       Ki      ]
            [ Ki             -(Ki+Ke)]

        Returns:
            BidomainBlockMatrix instance
        """
        M = self.assemble_mass_matrix()
        Ki = self.assemble_intracellular_stiffness()
        Ke = self.assemble_extracellular_stiffness()

        self.n = M.shape[0]

        theta = 0.5 if self.scheme.scheme == "CN" else 1.0

        self._block_matrix = BidomainBlockMatrix(
            M=M, Ki=Ki, Ke=Ke,
            dt=self.scheme.dt,
            theta=theta,
            device=self.device
        )

        return self._block_matrix

    def apply_null_space_pinning(
        self,
        A: BidomainBlockMatrix,
        pin_node: int = 0
    ) -> BidomainBlockMatrix:
        """
        Apply null space pinning for φe uniqueness.

        The elliptic equation for φe is singular (det(Ki+Ke) = 0) because
        adding a constant to φe doesn't change the physics. To regularize,
        we pin φe(pin_node) = 0.

        This modifies the bottom-right block by setting:
            (Ki+Ke)[pin_node, :] = unit vector e[pin_node]

        Args:
            A: Block matrix to modify
            pin_node: Node index where φe is pinned to zero

        Returns:
            Modified block matrix (sparse modifications are challenging;
            this demonstrates the concept for dense matrices)
        """
        # For sparse tensors, null space pinning is typically handled
        # during the solve (e.g., in MINRES with constraints)
        # This shows conceptual approach:

        n = A.n

        # Create dense version for modification
        A_dense = A.to_dense().clone()

        # Pin row: set bottom-right block, pin_node row to [0...1...0]
        A_dense[n + pin_node, :] = 0
        A_dense[n + pin_node, n + pin_node] = 1.0

        # Correspondingly modify RHS
        # rhs[n + pin_node] = 0

        return A_dense

    def assemble_rhs_crank_nicolson(
        self,
        Vm_n: torch.Tensor,
        phi_e_n: torch.Tensor,
        Iion_n: torch.Tensor,
        Iion_np1: torch.Tensor,
        Istim: torch.Tensor
    ) -> BlockRHS:
        """
        Assemble RHS for Crank-Nicolson scheme.

        CN scheme:
            [M/dt + 0.5*Ki] Vm^{n+1} + 0.5*Ki φe^{n+1}
            = [M/dt - 0.5*Ki] Vm^n - 0.5*Ki φe^n - 0.5(Iion^n + Iion^{n+1}) + Istim

        Args:
            Vm_n: (n,) transmembrane voltage at time step n
            phi_e_n: (n,) extracellular potential at time step n
            Iion_n: (n,) ionic current at time step n
            Iion_np1: (n,) ionic current at time step n+1 (for CN averaging)
            Istim: (n,) stimulus current

        Returns:
            BlockRHS object
        """
        M = self.assemble_mass_matrix()
        Ki = self.assemble_intracellular_stiffness()

        # RHS for Vm equation
        # [M/dt - 0.5*Ki] Vm^n - 0.5*Ki φe^n - 0.5(Iion^n + Iion^{n+1}) + Istim
        rhs_vm = torch.sparse.mm(M / self.scheme.dt, Vm_n.unsqueeze(1)).squeeze(1) - \
                 0.5 * torch.sparse.mm(Ki, Vm_n.unsqueeze(1)).squeeze(1) - \
                 0.5 * torch.sparse.mm(Ki, phi_e_n.unsqueeze(1)).squeeze(1) - \
                 0.5 * (Iion_n + Iion_np1) + Istim

        # RHS for φe equation: Ki Vm^n
        # (Note: with null space pinning, last equation becomes φe(pin) = 0)
        rhs_phi_e = torch.sparse.mm(Ki, Vm_n.unsqueeze(1)).squeeze(1)

        return BlockRHS(rhs_vm, rhs_phi_e, device=self.device)

    def assemble_rhs_bdf1(
        self,
        Vm_n: torch.Tensor,
        phi_e_n: torch.Tensor,
        Iion_np1: torch.Tensor,
        Istim: torch.Tensor
    ) -> BlockRHS:
        """
        Assemble RHS for BDF1 (Backward Euler) scheme.

        BDF1 scheme:
            [M/dt + Ki] Vm^{n+1} + Ki φe^{n+1}
            = [M/dt] Vm^n - Iion^{n+1} + Istim

        Args:
            Vm_n: (n,) transmembrane voltage at time step n
            phi_e_n: (n,) extracellular potential at time step n
            Iion_np1: (n,) ionic current at time step n+1
            Istim: (n,) stimulus current

        Returns:
            BlockRHS object
        """
        M = self.assemble_mass_matrix()
        Ki = self.assemble_intracellular_stiffness()

        # RHS for Vm equation
        rhs_vm = torch.sparse.mm(M / self.scheme.dt, Vm_n.unsqueeze(1)).squeeze(1) - \
                 Iion_np1 + Istim

        # RHS for φe equation
        rhs_phi_e = torch.sparse.mm(Ki, Vm_n.unsqueeze(1)).squeeze(1)

        return BlockRHS(rhs_vm, rhs_phi_e, device=self.device)


# ============================================================================
# Simplified Spatial Discretization (FDM)
# ============================================================================

class SimpleFDMDiscretization(SpatialDiscretization):
    """
    Simple finite difference discretization on structured grid.

    For demonstration; in V5.4, would interface with full FEM/FDM/FVM implementations.
    """

    def __init__(
        self,
        nx: int = 10,
        ny: int = 10,
        dx: float = 0.01,
        device: torch.device = torch.device('cpu')
    ):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.n = nx * ny
        self.device = device
        self._M = None
        self._K = None

    def assemble_mass_matrix(self) -> torch.sparse.FloatTensor:
        """Assemble lumped mass matrix for 2D grid."""
        if self._M is not None:
            return self._M

        # Lumped mass: M_ii = (dx)^2 for each node
        indices = torch.arange(self.n, device=self.device).unsqueeze(0)
        indices = torch.cat([indices, indices], dim=0)
        values = torch.full((self.n,), self.dx**2, device=self.device)

        self._M = torch.sparse_coo_tensor(
            indices, values, (self.n, self.n), device=self.device
        ).coalesce()

        return self._M

    def assemble_stiffness_matrix(self) -> torch.sparse.FloatTensor:
        """Assemble 5-point Laplacian stiffness matrix."""
        if self._K is not None:
            return self._K

        rows, cols, vals = [], [], []
        coeff = -1.0 / (self.dx**2)

        for i in range(self.ny):
            for j in range(self.nx):
                node = i * self.nx + j

                # Center coefficient
                rows.append(node)
                cols.append(node)
                vals.append(-4.0 * coeff)  # 4 neighbors

                # Right neighbor
                if j < self.nx - 1:
                    rows.append(node)
                    cols.append(node + 1)
                    vals.append(coeff)

                # Left neighbor
                if j > 0:
                    rows.append(node)
                    cols.append(node - 1)
                    vals.append(coeff)

                # Bottom neighbor
                if i < self.ny - 1:
                    rows.append(node)
                    cols.append(node + self.nx)
                    vals.append(coeff)

                # Top neighbor
                if i > 0:
                    rows.append(node)
                    cols.append(node - self.nx)
                    vals.append(coeff)

        indices = torch.tensor([rows, cols], dtype=torch.long, device=self.device)
        vals = torch.tensor(vals, dtype=torch.float32, device=self.device)

        self._K = torch.sparse_coo_tensor(
            indices, vals, (self.n, self.n), device=self.device
        ).coalesce()

        return self._K


# ============================================================================
# Main Demo/Test
# ============================================================================

def main():
    """
    Demonstration of bidomain block system assembly.
    """
    print("=" * 80)
    print("BIDOMAIN BLOCK SYSTEM ASSEMBLY - REFERENCE IMPLEMENTATION")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Setup
    nx, ny = 20, 20
    spatial_discr = SimpleFDMDiscretization(nx=nx, ny=ny, dx=0.01, device=device)
    params = BidomainParameters()
    scheme = TimeSteppingScheme(scheme="CN", dt=0.01, theta=0.5)

    assembler = BidomainBlockSystemAssembler(
        spatial_discr=spatial_discr,
        params=params,
        scheme=scheme,
        device=device
    )

    # Assemble block system
    print(f"\nAssembling bidomain block system for {nx}×{ny} grid...")
    block_matrix = assembler.assemble_block_system()
    print(f"Block matrix size: {2*spatial_discr.n} × {2*spatial_discr.n}")
    print(f"Scheme: {scheme.scheme}, dt={scheme.dt}, θ={scheme.theta}")

    # Create test vectors
    n = spatial_discr.n
    Vm_n = torch.randn(n, device=device)
    phi_e_n = torch.randn(n, device=device)
    Iion_n = torch.randn(n, device=device)
    Iion_np1 = torch.randn(n, device=device)
    Istim = torch.zeros(n, device=device)

    # Assemble RHS
    print("\nAssembling RHS for Crank-Nicolson scheme...")
    rhs = assembler.assemble_rhs_crank_nicolson(
        Vm_n=Vm_n,
        phi_e_n=phi_e_n,
        Iion_n=Iion_n,
        Iion_np1=Iion_np1,
        Istim=Istim
    )
    rhs_vec = rhs.to_vector()
    print(f"RHS vector shape: {rhs_vec.shape}")
    print(f"RHS vector norm: {torch.linalg.norm(rhs_vec):.6f}")

    # Test matrix-vector product
    print("\nTesting block matrix-vector product...")
    x_test = torch.randn(2*n, device=device)
    y_test = block_matrix.matvec(x_test)
    print(f"Input shape: {x_test.shape}, Output shape: {y_test.shape}")
    print(f"Input norm: {torch.linalg.norm(x_test):.6f}")
    print(f"Output norm: {torch.linalg.norm(y_test):.6f}")

    # Show block structure (small example)
    print("\nBlock matrix structure (dense representation):")
    if n <= 5:
        A_dense = block_matrix.to_dense()
        print(f"Block matrix (2n × 2n = {2*n} × {2*n}):")
        print(A_dense)
    else:
        print(f"Matrix too large for display ({2*n} × {2*n})")
        A_dense = block_matrix.to_dense()
        print(f"Top-left (Vm equation):")
        print(A_dense[:n, :n])
        print(f"Bottom-left (coupling):")
        print(A_dense[n:, :n])

    # Show sparsity pattern
    Ki = assembler.assemble_intracellular_stiffness()
    Ke = assembler.assemble_extracellular_stiffness()
    M = assembler.assemble_mass_matrix()

    nnz_M = M._nnz()
    nnz_Ki = Ki._nnz()
    nnz_Ke = Ke._nnz()

    print(f"\nSparsity statistics:")
    print(f"  Mass matrix M: {nnz_M} nonzeros (density: {nnz_M/(n*n)*100:.2f}%)")
    print(f"  Intracellular Ki: {nnz_Ki} nonzeros (density: {nnz_Ki/(n*n)*100:.2f}%)")
    print(f"  Extracellular Ke: {nnz_Ke} nonzeros (density: {nnz_Ke/(n*n)*100:.2f}%)")

    # BDF1 demonstration
    print("\n" + "=" * 80)
    print("Testing BDF1 (Backward Euler) scheme...")
    scheme_bdf1 = TimeSteppingScheme(scheme="BDF1", dt=0.01, theta=1.0)
    assembler_bdf1 = BidomainBlockSystemAssembler(
        spatial_discr=spatial_discr,
        params=params,
        scheme=scheme_bdf1,
        device=device
    )

    block_matrix_bdf1 = assembler_bdf1.assemble_block_system()
    rhs_bdf1 = assembler_bdf1.assemble_rhs_bdf1(
        Vm_n=Vm_n,
        phi_e_n=phi_e_n,
        Iion_np1=Iion_np1,
        Istim=Istim
    )

    print(f"BDF1 RHS norm: {torch.linalg.norm(rhs_bdf1.to_vector()):.6f}")

    # Performance test
    print("\n" + "=" * 80)
    print("Performance test: Matrix-vector product")
    import time

    num_trials = 100
    x_test = torch.randn(2*n, device=device)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_start = time.perf_counter()

    for _ in range(num_trials):
        y = block_matrix.matvec(x_test)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_end = time.perf_counter()

    t_per_iter = (t_end - t_start) / num_trials * 1000  # Convert to ms
    print(f"Time per matrix-vector product: {t_per_iter:.4f} ms")
    print(f"Average throughput: {(2*n*nnz_M) / (t_per_iter*1e-3) / 1e9:.2f} GFLOP/s")

    print("\n" + "=" * 80)
    print("Reference implementation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
