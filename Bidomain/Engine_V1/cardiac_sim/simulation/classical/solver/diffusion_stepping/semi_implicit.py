"""
Semi-Implicit Bidomain Diffusion Solver

Forward Euler for the parabolic equation (no linear solve),
implicit solve for the elliptic equation only.

Step 1 (Parabolic — explicit Forward Euler):
    Vm^{n+1} = Vm^n + dt * L_i * (Vm^n + phi_e^n)
    Cost: one SpMV (no linear solver)

Step 2 (Elliptic — implicit):
    -(L_i + L_e) * phi_e^{n+1} = L_i * Vm^{n+1}
    Cost: one linear solve (same as Gauss-Seidel elliptic)

CFL constraint: dt < dx^2 / (4 * D_i)
For dx=0.025 cm, D_i=0.00124 cm^2/ms: dt_max ~ 0.126 ms

Ref: DIFFUSION_SPLITTING_DESIGN.md § Strategy 2
"""

import warnings
from typing import TYPE_CHECKING

from .base import BidomainDiffusionSolver

if TYPE_CHECKING:
    from ...discretization.base import BidomainSpatialDiscretization
    from ...state import BidomainState
    from ..linear_solver.base import LinearSolver


class SemiImplicitSolver(BidomainDiffusionSolver):
    """
    Semi-implicit bidomain diffusion solver.

    Eliminates the parabolic linear solve by using Forward Euler for Vm.
    Only one linear solve per step (elliptic for phi_e).

    Parameters
    ----------
    spatial : BidomainSpatialDiscretization
        Provides L_i, L_e stencils and elliptic operator.
    dt : float
        Time step (ms). Must satisfy CFL: dt < dx^2 / (4*D_i).
    elliptic_solver : LinearSolver
        Solver for phi_e sub-problem.
    pin_node : int
        Node for null space pinning (Neumann only).
    """

    def __init__(self, spatial, dt, elliptic_solver, pin_node=0):
        super().__init__(spatial, dt)
        self.elliptic_solver = elliptic_solver

        # CFL check
        D_i = spatial.conductivity.D_i
        dx = spatial.grid.dx
        dt_cfl = dx ** 2 / (4.0 * D_i)
        if dt > dt_cfl:
            raise ValueError(
                f"Semi-implicit CFL violated: dt={dt:.4f} > dt_max={dt_cfl:.4f} ms. "
                f"Use 'gauss_seidel' (implicit) or reduce dt.")
        self._cfl_ratio = dt / dt_cfl

        # Elliptic operator + pinning
        self._needs_pinning = spatial.grid.boundary_spec.phi_e_has_null_space
        self._pin_node = pin_node
        self._build_operators(spatial)

    def _build_operators(self, spatial):
        """Build elliptic operator (no parabolic operator needed)."""
        self.A_ellip = spatial.get_elliptic_operator()
        if self._needs_pinning:
            self.A_ellip = self.apply_elliptic_pinning(
                self.A_ellip, self._pin_node)

    def step(self, state, dt):
        """
        Advance Vm and phi_e by one diffusion time step.

        Step 1: Forward Euler for Vm (one SpMV, no linear solve)
        Step 2: Implicit elliptic solve for phi_e
        """
        # --- Step 1: Explicit parabolic (Vm) ---
        # dVm/dt = L_i * (Vm + phi_e)  =>  Vm += dt * L_i * (Vm + phi_e)
        Vm_plus_phi = state.Vm + state.phi_e
        LiSum = self._spatial.apply_L_i(Vm_plus_phi)
        Vm_new = state.Vm + dt * LiSum

        # --- Step 2: Elliptic (phi_e) ---
        rhs_ellip = self._spatial.apply_L_i(Vm_new)
        if self._needs_pinning:
            rhs_ellip[self._pin_node] = 0.0
        phi_e_new = self.elliptic_solver.solve(self.A_ellip, rhs_ellip)

        if self._needs_pinning:
            phi_e_new = phi_e_new - phi_e_new[self._pin_node]

        # --- Update state in-place ---
        state.Vm.copy_(Vm_new)
        state.phi_e.copy_(phi_e_new)

    def rebuild_operators(self, spatial, dt):
        """Rebuild operators when dt changes."""
        # Re-check CFL
        D_i = spatial.conductivity.D_i
        dx = spatial.grid.dx
        dt_cfl = dx ** 2 / (4.0 * D_i)
        if dt > dt_cfl:
            raise ValueError(
                f"Semi-implicit CFL violated: dt={dt:.4f} > dt_max={dt_cfl:.4f}")
        self._cfl_ratio = dt / dt_cfl
        self._spatial = spatial
        self._dt = dt
        self._build_operators(spatial)
