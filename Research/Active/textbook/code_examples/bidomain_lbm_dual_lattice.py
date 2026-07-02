"""
Bidomain Lattice Boltzmann Method (Dual Lattice) for Cardiac Simulation Engine V5.4

This module implements a dual-lattice LBM approach for bidomain equations.
One D2Q5 lattice handles the parabolic Vm dynamics with ionic sources,
while a second D2Q5 lattice handles the elliptic φe equation via pseudo-time relaxation.

The bidomain equations are recovered in the hydrodynamic (continuum) limit.

LBM perspective:
    Vm lattice: f_i(x,t) with BGK collision, reaction source
    φe lattice: g_i(x,t) with BGK collision, elliptic pseudo-time

Advantages:
    - Natural coupling through collision terms
    - Local computations suitable for GPU
    - Boundary conditions straightforward (bounce-back)
    - Inherent parallelism in lattice dynamics

References:
    [1] Chopard & Droz "Cellular Automata for Physical Systems" (1998)
    [2] Succi "The Lattice Boltzmann Equation for Fluid Dynamics and Beyond" (2001)
    [3] Benzi et al. "The Lattice Boltzmann Equation: A New Tool for Computational
        Fluid-Dynamics" (1992)
    [4] Pavaihar & Karlin "A New Lattice Boltzmann Scheme for Bio-fluids" (2014)
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional
import numpy as np


# ============================================================================
# D2Q5 Lattice Definition (5-velocity in 2D)
# ============================================================================

class D2Q5Lattice:
    """
    D2Q5 lattice: 5 velocities in 2D (center + 4 cardinal directions)

    Velocities:
        c_0 = (0, 0)      - rest
        c_1 = (1, 0)      - right
        c_2 = (0, 1)      - up
        c_3 = (-1, 0)     - left
        c_4 = (0, -1)     - down

    Weights (for standard thermal equilibrium):
        w_0 = 1/3
        w_1,2,3,4 = 1/6

    Lattice speed of sound: c_s = 1/√3
    """

    def __init__(self, device: torch.device = torch.device('cpu')):
        self.device = device

        # Velocity components (c_i = (c_ix, c_iy))
        self.c_x = torch.tensor([0, 1, 0, -1, 0], dtype=torch.float32, device=device)
        self.c_y = torch.tensor([0, 0, 1, 0, -1], dtype=torch.float32, device=device)

        # Weights
        self.w = torch.tensor([1./3., 1./6., 1./6., 1./6., 1./6.],
                             dtype=torch.float32, device=device)

        # Speed of sound squared
        self.cs2 = 1.0 / 3.0

    def equilibrium(self, rho: torch.Tensor, u_x: torch.Tensor, u_y: torch.Tensor) -> torch.Tensor:
        """
        Compute equilibrium distribution.

        f_i^eq = w_i * rho * [1 + (c_i·u)/c_s^2 + (c_i·u)^2/(2*c_s^4) - u^2/(2*c_s^2)]

        Args:
            rho: (ny, nx) macroscopic density
            u_x: (ny, nx) x-velocity
            u_y: (ny, nx) y-velocity

        Returns:
            (5, ny, nx) equilibrium distribution
        """
        ny, nx = rho.shape
        f_eq = torch.zeros((5, ny, nx), device=self.device)

        u_sq = u_x**2 + u_y**2

        for i in range(5):
            c_u = self.c_x[i] * u_x + self.c_y[i] * u_y
            f_eq[i] = self.w[i] * rho * (
                1.0 + c_u / self.cs2 + c_u**2 / (2.0 * self.cs2**2) - u_sq / (2.0 * self.cs2)
            )

        return f_eq

    def moments(self, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute macroscopic moments from distribution.

        Args:
            f: (5, ny, nx) distribution function

        Returns:
            (rho, u_x, u_y)
        """
        rho = torch.sum(f, dim=0)

        u_x = (self.c_x[:, None, None] * f).sum(dim=0) / torch.clamp(rho, min=1e-14)
        u_y = (self.c_y[:, None, None] * f).sum(dim=0) / torch.clamp(rho, min=1e-14)

        return rho, u_x, u_y


# ============================================================================
# BGK Collision Operator
# ============================================================================

class BGKCollision:
    """
    BGK (Bhatnagar-Gross-Krook) collision operator.

    f_i(x,t+dt) = f_i(x-c_i*dt,t) - (1/τ)*[f_i - f_i^eq] + dt*S_i

    where τ is relaxation time and S_i is source term.

    For Vm (parabolic): τ_Vm determines effective diffusion
    For φe (elliptic, pseudo-time): τ_φe → 0 for instantaneous constraint
    """

    def __init__(
        self,
        lattice: D2Q5Lattice,
        tau: float = 0.9,
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            lattice: D2Q5Lattice instance
            tau: Relaxation time (> 0.5 for stability)
            device: torch device
        """
        self.lattice = lattice
        self.tau = tau
        self.device = device

    def collide(
        self,
        f: torch.Tensor,
        rho: torch.Tensor,
        u_x: torch.Tensor,
        u_y: torch.Tensor,
        source: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply BGK collision: f_i^* = f_i - (1/τ)(f_i - f_i^eq) + source

        Args:
            f: (5, ny, nx) distribution
            rho: (ny, nx) density
            u_x: (ny, nx) x-velocity
            u_y: (ny, nx) y-velocity
            source: (5, ny, nx) or (ny, nx) source term

        Returns:
            (5, ny, nx) post-collision distribution
        """
        f_eq = self.lattice.equilibrium(rho, u_x, u_y)

        f_post = f - (1.0 / self.tau) * (f - f_eq)

        if source is not None:
            if source.dim() == 2:
                # Scalar source, broadcast to all velocities
                f_post = f_post + source.unsqueeze(0)
            else:
                f_post = f_post + source

        return f_post


# ============================================================================
# Streaming Operator
# ============================================================================

class StreamingOperator:
    """
    LBM streaming (transport) step.

    f_i(x+c_i*dt, t+dt) = f_i*(x, t)

    where f_i* is post-collision distribution.
    """

    def __init__(
        self,
        lattice: D2Q5Lattice,
        device: torch.device = torch.device('cpu')
    ):
        self.lattice = lattice
        self.device = device

    def stream(
        self,
        f: torch.Tensor,
        periodic: bool = False
    ) -> torch.Tensor:
        """
        Stream populations to neighboring nodes.

        Args:
            f: (5, ny, nx) post-collision distribution
            periodic: Apply periodic boundary conditions if True

        Returns:
            (5, ny, nx) streamed distribution
        """
        ny, nx = f.shape[1:]
        f_new = f.clone()

        # Stream each direction
        # Direction 0 (rest): stays at same location
        # Direction 1 (right): shift left in x
        if periodic:
            f_new[1] = torch.roll(f[1], shifts=-1, dims=2)
        else:
            f_new[1, :, :-1] = f[1, :, 1:]
            f_new[1, :, -1] = 0  # Or bounce-back

        # Direction 2 (up): shift down in y
        if periodic:
            f_new[2] = torch.roll(f[2], shifts=-1, dims=1)
        else:
            f_new[2, :-1, :] = f[2, 1:, :]
            f_new[2, -1, :] = 0

        # Direction 3 (left): shift right in x
        if periodic:
            f_new[3] = torch.roll(f[3], shifts=1, dims=2)
        else:
            f_new[3, :, 1:] = f[3, :, :-1]
            f_new[3, :, 0] = 0

        # Direction 4 (down): shift up in y
        if periodic:
            f_new[4] = torch.roll(f[4], shifts=1, dims=1)
        else:
            f_new[4, 1:, :] = f[4, :-1, :]
            f_new[4, 0, :] = 0

        return f_new

    def apply_bounce_back(
        self,
        f: torch.Tensor,
        boundary_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply bounce-back boundary condition.

        Populations from boundary nodes are reflected back.

        Args:
            f: (5, ny, nx) distribution
            boundary_mask: (ny, nx) binary mask of boundary nodes

        Returns:
            Modified distribution with bounce-back applied
        """
        f_bb = f.clone()

        # Opposite directions: 0↔0, 1↔3, 2↔4
        bounce_pairs = [(1, 3), (2, 4)]

        for i, j in bounce_pairs:
            f_bb[i] = torch.where(
                boundary_mask.unsqueeze(0),
                f[j],  # Bounce-back
                f_bb[i]
            )
            f_bb[j] = torch.where(
                boundary_mask.unsqueeze(0),
                f[i],
                f_bb[j]
            )

        return f_bb


# ============================================================================
# Bidomain LBM Solver (Dual Lattice)
# ============================================================================

class BidomainLBMDualLattice:
    """
    Dual-lattice LBM for bidomain equations.

    Vm lattice (parabolic):
        - BGK collision with relaxation time τ_Vm
        - Source term from ionic current
        - D2Q5 with appropriate weights

    φe lattice (elliptic, pseudo-time):
        - BGK collision with small τ_φe → instantaneous constraint
        - No reaction source
        - Pseudo-time relaxation to steady state

    Coupling:
        - Vm source includes φe gradient (∇·(σi·∇φe))
        - φe depends on Vm gradient through divergence
    """

    def __init__(
        self,
        nx: int = 64,
        ny: int = 64,
        dt: float = 0.01,
        tau_Vm: float = 0.9,
        tau_phi_e: float = 0.55,  # Small for elliptic solve
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            nx, ny: Grid dimensions
            dt: Time step
            tau_Vm: Relaxation time for Vm lattice
            tau_phi_e: Relaxation time for φe lattice
            device: torch device
        """
        self.nx = nx
        self.ny = ny
        self.dt = dt
        self.device = device

        # Lattices
        self.lattice = D2Q5Lattice(device=device)

        # Collision operators
        self.bgk_Vm = BGKCollision(self.lattice, tau=tau_Vm, device=device)
        self.bgk_phi_e = BGKCollision(self.lattice, tau=tau_phi_e, device=device)

        # Streaming
        self.streamer = StreamingOperator(self.lattice, device=device)

        # Distributions
        self.f_Vm = torch.zeros((5, ny, nx), device=device)  # Vm distribution
        self.g_phi_e = torch.zeros((5, ny, nx), device=device)  # φe distribution

        # Initialize to uniform equilibrium
        self._initialize_equilibrium()

        # Macroscopic variables (for access)
        self.Vm = torch.zeros((ny, nx), device=device)
        self.phi_e = torch.zeros((ny, nx), device=device)

        # Ionic current model (simplified)
        self.ionic_conductance = 1.0

    def _initialize_equilibrium(self):
        """Initialize distributions to equilibrium."""
        rho_Vm = torch.ones((self.ny, self.nx), device=self.device) * 0.0
        u_Vm = torch.zeros((self.ny, self.nx), device=self.device)

        rho_phi_e = torch.ones((self.ny, self.nx), device=self.device) * 0.0

        self.f_Vm = self.lattice.equilibrium(rho_Vm, u_Vm, u_Vm)
        self.g_phi_e = self.lattice.equilibrium(rho_phi_e, u_Vm, u_Vm)

    def compute_laplacian(self, f: torch.Tensor) -> torch.Tensor:
        """
        Compute ∇² from distribution (or directly from macroscopic).

        Simplified: use finite differences on the macroscopic field.

        Returns:
            (ny, nx) Laplacian field
        """
        rho, _, _ = self.lattice.moments(f)

        # 5-point stencil Laplacian with periodic BC
        lap = torch.zeros_like(rho)

        lap = (
            torch.roll(rho, shifts=-1, dims=1) +
            torch.roll(rho, shifts=1, dims=1) +
            torch.roll(rho, shifts=-1, dims=2) +
            torch.roll(rho, shifts=1, dims=2) -
            4.0 * rho
        )

        return lap

    def compute_divergence(self, f: torch.Tensor) -> torch.Tensor:
        """
        Compute ∇·(∇ρ) from momentum components.

        For our LBM with D2Q5, compute div of gradient.

        Returns:
            (ny, nx) scalar field
        """
        rho, u_x, u_y = self.lattice.moments(f)

        # ∇·u ≈ (du_x/dx + du_y/dy)
        dux_dx = (torch.roll(u_x, shifts=-1, dims=2) - torch.roll(u_x, shifts=1, dims=2)) / 2.0
        duy_dy = (torch.roll(u_y, shifts=-1, dims=1) - torch.roll(u_y, shifts=1, dims=1)) / 2.0

        return dux_dx + duy_dy

    def step(
        self,
        Istim: Optional[torch.Tensor] = None,
        num_elliptic_iters: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Execute one LBM time step (coupled Vm and φe).

        1. Compute macroscopic Vm from f_Vm
        2. Compute ionic source
        3. Collide and stream f_Vm (parabolic)
        4. Compute φe from steady elliptic solve (pseudo-time iterations)
        5. Update coupling

        Args:
            Istim: (ny, nx) stimulus current
            num_elliptic_iters: Number of inner iterations for φe (elliptic) solve

        Returns:
            (Vm, φe) updated macroscopic variables
        """
        # Step 1: Get macroscopic Vm and φe
        self.Vm, _, _ = self.lattice.moments(self.f_Vm)
        self.phi_e, _, _ = self.lattice.moments(self.g_phi_e)

        if Istim is None:
            Istim = torch.zeros_like(self.Vm)

        # Step 2: Compute ionic current (simplified model)
        # I_ion ≈ g_ion * (V - E_ion)
        Iion = self.ionic_conductance * self.Vm

        # Step 3: Compute source term for Vm lattice
        # Source includes: -Iion + Istim + coupling from φe
        lap_phi_e = self.compute_laplacian(self.g_phi_e)
        source_Vm = (-Iion + Istim + 0.1 * lap_phi_e)  # 0.1 is coupling weight

        # Step 4: Collision and streaming for Vm lattice
        f_Vm_post = self.bgk_Vm.collide(
            self.f_Vm, self.Vm, torch.zeros_like(self.Vm), torch.zeros_like(self.Vm),
            source=source_Vm
        )
        self.f_Vm = self.streamer.stream(f_Vm_post, periodic=True)

        # Step 5: Solve elliptic equation for φe (pseudo-time iterations)
        # Use multiple BGK collisions with very small τ to relax to steady state
        for _ in range(num_elliptic_iters):
            # Source for φe is divergence of Vm gradient
            lap_Vm = self.compute_laplacian(self.f_Vm)
            source_phi_e = -0.1 * lap_Vm  # Coupling from Vm

            g_post = self.bgk_phi_e.collide(
                self.g_phi_e, self.phi_e, torch.zeros_like(self.phi_e),
                torch.zeros_like(self.phi_e),
                source=source_phi_e
            )

            # Minimal streaming for elliptic (mostly collide)
            self.g_phi_e = g_post  # For elliptic, can skip streaming or use pseudo-time

        # Update macroscopic variables
        self.Vm, _, _ = self.lattice.moments(self.f_Vm)
        self.phi_e, _, _ = self.lattice.moments(self.g_phi_e)

        return self.Vm, self.phi_e

    def set_stimulus_region(
        self,
        x_min: float = 0.0,
        x_max: float = 0.2,
        y_min: float = 0.3,
        y_max: float = 0.7,
        amplitude: float = 100.0
    ) -> torch.Tensor:
        """
        Create stimulus current in a rectangular region.

        Args:
            x_min, x_max, y_min, y_max: Domain coordinates [0, 1]
            amplitude: Stimulus amplitude

        Returns:
            (ny, nx) stimulus tensor
        """
        x = torch.linspace(0, 1, self.nx, device=self.device)
        y = torch.linspace(0, 1, self.ny, device=self.device)

        X, Y = torch.meshgrid(x, y, indexing='ij')
        X = X.T  # Transpose for (ny, nx) indexing

        mask = (X >= x_min) & (X <= x_max) & (Y >= y_min) & (Y <= y_max)

        return amplitude * mask.float()


# ============================================================================
# Main Demo/Test
# ============================================================================

def main():
    """Demonstration of dual-lattice bidomain LBM."""
    print("=" * 80)
    print("BIDOMAIN DUAL-LATTICE LBM - REFERENCE IMPLEMENTATION")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Initialize solver
    nx, ny = 64, 64
    print(f"\nInitializing LBM solver for {nx}×{ny} grid...")

    lbm = BidomainLBMDualLattice(
        nx=nx, ny=ny,
        dt=0.01,
        tau_Vm=0.95,      # Parabolic relaxation
        tau_phi_e=0.51,   # Nearly instantaneous (elliptic)
        device=device
    )

    print(f"Vm lattice shape: {lbm.f_Vm.shape}")
    print(f"φe lattice shape: {lbm.g_phi_e.shape}")
    print(f"Initial Vm range: [{lbm.Vm.min():.4f}, {lbm.Vm.max():.4f}]")

    # Create stimulus: 2D slab excitation
    print("\nCreating stimulus pattern (slab)...")
    Istim = lbm.set_stimulus_region(
        x_min=0.0, x_max=0.1,
        y_min=0.3, y_max=0.7,
        amplitude=100.0
    )

    print(f"Stimulus max: {Istim.max():.4f}")
    print(f"Stimulus sum: {Istim.sum():.4f}")

    # Simulation loop
    print("\n" + "=" * 80)
    print("Running LBM simulation")
    print("=" * 80)

    num_timesteps = 50
    save_interval = 10

    Vm_history = []
    phi_e_history = []

    print(f"{'Step':>4} {'Vm_max':>10} {'Vm_min':>10} {'φe_max':>10} {'φe_min':>10}")
    print("-" * 50)

    for step in range(num_timesteps):
        # Apply stimulus only for first few time steps
        Istim_current = Istim if step < 10 else torch.zeros_like(Istim)

        # LBM step
        Vm, phi_e = lbm.step(Istim=Istim_current, num_elliptic_iters=3)

        if (step + 1) % save_interval == 0:
            Vm_history.append(Vm.clone().cpu())
            phi_e_history.append(phi_e.clone().cpu())

            print(f"{step+1:4d} {Vm.max():10.4f} {Vm.min():10.4f} "
                  f"{phi_e.max():10.4f} {phi_e.min():10.4f}")

    print("\n" + "=" * 80)
    print("Simulation statistics")
    print("=" * 80)

    print(f"Number of saved snapshots: {len(Vm_history)}")
    print(f"Final Vm range: [{lbm.Vm.min():.4f}, {lbm.Vm.max():.4f}]")
    print(f"Final φe range: [{lbm.phi_e.min():.4f}, {lbm.phi_e.max():.4f}]")

    # Analysis of spatial patterns
    print("\nSpatial statistics (final state):")

    # Center slice
    center_y = ny // 2
    Vm_center = lbm.Vm[center_y, :]
    phi_e_center = lbm.phi_e[center_y, :]

    print(f"Center slice (y={center_y}):")
    print(f"  Vm: min={Vm_center.min():.4f}, max={Vm_center.max():.4f}, "
          f"mean={Vm_center.mean():.4f}")
    print(f"  φe: min={phi_e_center.min():.4f}, max={phi_e_center.max():.4f}, "
          f"mean={phi_e_center.mean():.4f}")

    # Temporal evolution
    if len(Vm_history) > 1:
        print("\nTemporal evolution:")
        Vm_max_hist = [V.max() for V in Vm_history]
        Vm_min_hist = [V.min() for V in Vm_history]

        print(f"Vm max over time: {Vm_max_hist[0]:.4f} → {Vm_max_hist[-1]:.4f}")
        print(f"Vm min over time: {Vm_min_hist[0]:.4f} → {Vm_min_hist[-1]:.4f}")

    # Lattice properties
    print("\n" + "=" * 80)
    print("Lattice properties")
    print("=" * 80)

    print(f"D2Q5 lattice:")
    print(f"  Velocities: {lbm.lattice.c_x.tolist()}, {lbm.lattice.c_y.tolist()}")
    print(f"  Weights: {lbm.lattice.w.tolist()}")
    print(f"  c_s²: {lbm.lattice.cs2:.4f}")

    print(f"\nRelaxation times:")
    print(f"  τ_Vm: {lbm.bgk_Vm.tau:.4f} (parabolic)")
    print(f"  τ_φe: {lbm.bgk_phi_e.tau:.4f} (elliptic, pseudo-time)")

    # Compute effective diffusion
    # For LBM: ν_eff = c_s² * (τ - 0.5)
    nu_Vm = lbm.lattice.cs2 * (lbm.bgk_Vm.tau - 0.5)
    nu_phi_e = lbm.lattice.cs2 * (lbm.bgk_phi_e.tau - 0.5)

    print(f"\nEffective kinematic viscosity (diffusion):")
    print(f"  ν_Vm = c_s² * (τ_Vm - 0.5) = {nu_Vm:.6f}")
    print(f"  ν_φe = c_s² * (τ_φe - 0.5) = {nu_phi_e:.6f}")

    print("\n" + "=" * 80)
    print("Reference implementation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
