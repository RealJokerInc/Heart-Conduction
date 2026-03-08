"""
Decoupled Jacobi Bidomain Diffusion Solver

Jacobi splitting: both parabolic and elliptic use OLD timestep data.
Unlike Gauss-Seidel which feeds Vm^{n+1} into the elliptic solve,
Jacobi uses Vm^n for the elliptic RHS. This makes both solves
independent and parallelizable on GPU.

Step 1 (Parabolic):
    A_para * Vm^{n+1} = B_para * Vm^n + L_i * phi_e^n

Step 2 (Elliptic — uses Vm^n, NOT Vm^{n+1}):
    -(L_i + L_e) * phi_e^{n+1} = L_i * Vm^n

Both steps depend only on state^n, so they can run concurrently.

Ref: DIFFUSION_SPLITTING_DESIGN.md § Strategy 3
Ref: Fernandez & Zemzemi 2010 (energy stability proof)
"""

from typing import TYPE_CHECKING

from .base import BidomainDiffusionSolver
from ..linear_solver.pcg import sparse_mv

if TYPE_CHECKING:
    from ...discretization.base import BidomainSpatialDiscretization
    from ...state import BidomainState
    from ..linear_solver.base import LinearSolver


class DecoupledJacobiSolver(BidomainDiffusionSolver):
    """
    Decoupled (Jacobi splitting) bidomain diffusion solver.

    Both solves use only old-timestep data, making them independent
    and suitable for GPU parallelism.

    Parameters
    ----------
    spatial : BidomainSpatialDiscretization
        Provides L_i, L_e stencils and parabolic/elliptic operators.
    dt : float
        Time step (ms)
    parabolic_solver : LinearSolver
        Solver for Vm sub-problem
    elliptic_solver : LinearSolver
        Solver for phi_e sub-problem
    theta : float
        Implicitness (0.5 = CN, 1.0 = BDF1)
    pin_node : int
        Node for null space pinning (Neumann only)
    """

    def __init__(self, spatial, dt, parabolic_solver, elliptic_solver,
                 theta=0.5, pin_node=0):
        super().__init__(spatial, dt)
        self.theta = theta
        self.parabolic_solver = parabolic_solver
        self.elliptic_solver = elliptic_solver

        self._needs_pinning = spatial.grid.boundary_spec.phi_e_has_null_space
        self._pin_node = pin_node

        self._build_operators(spatial, dt)

    def _build_operators(self, spatial, dt):
        """Build parabolic and elliptic operators."""
        self.A_para, self.B_para = spatial.get_parabolic_operators(dt, self.theta)
        self.A_ellip = spatial.get_elliptic_operator()

        if self._needs_pinning:
            self.A_ellip = self.apply_elliptic_pinning(
                self.A_ellip, self._pin_node)

    def step(self, state, dt):
        """
        Advance Vm and phi_e by one diffusion time step.

        Both solves use old-timestep data only (Jacobi splitting).

        Parameters
        ----------
        state : BidomainState
        dt : float
            Must match the dt used at construction (operators encode 1/dt).
        """
        if abs(dt - self._dt) > 1e-14 * self._dt:
            raise ValueError(
                f"Jacobi: step dt={dt} != constructor dt={self._dt}. "
                f"Call rebuild_operators() first.")
        # --- Build RHS for both using old state ---
        rhs_para = sparse_mv(self.B_para, state.Vm) \
                   + self._spatial.apply_L_i(state.phi_e)  # phi_e^n
        rhs_ellip = self._spatial.apply_L_i(state.Vm)       # Vm^n

        if self._needs_pinning:
            rhs_ellip[self._pin_node] = 0.0

        # --- Solve both (independent — could be parallel on GPU) ---
        Vm_new = self.parabolic_solver.solve(self.A_para, rhs_para)
        phi_e_new = self.elliptic_solver.solve(self.A_ellip, rhs_ellip)

        if self._needs_pinning:
            phi_e_new = phi_e_new - phi_e_new[self._pin_node]

        # --- Update state in-place ---
        state.Vm.copy_(Vm_new)
        state.phi_e.copy_(phi_e_new)

    def rebuild_operators(self, spatial, dt):
        """Rebuild operators when dt changes."""
        self._spatial = spatial
        self._dt = dt
        self._build_operators(spatial, dt)
