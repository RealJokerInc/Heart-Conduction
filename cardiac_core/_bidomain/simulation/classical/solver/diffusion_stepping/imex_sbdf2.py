"""
IMEX SBDF2 Bidomain Diffusion Solver

Second-order BDF time discretization for the diffusion step.
Self-starts with BDF1 (backward Euler) for the first step.

BDF1 (step 0):
    (1/dt * I - L_i) * Vm^1 = (1/dt) * Vm^0 + L_i * phi_e^0
    -(L_i + L_e) * phi_e^1 = L_i * Vm^1

BDF2 (steps 1+):
    (3/(2dt) * I - L_i) * Vm^{n+1}
        = (2/dt) * Vm^n - (1/(2dt)) * Vm^{n-1} + L_i * (2*phi_e^n - phi_e^{n-1})
    -(L_i + L_e) * phi_e^{n+1} = L_i * Vm^{n+1}

The explicit coupling term uses the SBDF2 2nd-order extrapolation (2*phi_e^n - phi_e^{n-1});
lagging it at just phi_e^n adds an O(dt) error to the explicit term (audit 2026-07-16, Lane C1).
NOTE (verified on the real solver): fixing the extrapolation ~halves the temporal error but does
NOT by itself restore full 2nd order — the decoupled parabolic->elliptic *staggering* (phi_e is
slaved to Vm within the step and lagged across steps) imposes its own O(dt) splitting floor.
True 2nd order would require a monolithic Vm/phi_e solve. This fix removes the extrapolation
error term; the staggering floor is a separate, deeper limitation left documented.

Properties:
  - 2nd order in time (vs 1st order for GS/semi-implicit)
  - L-stable (no spurious oscillations unlike CN)
  - Unconditionally stable (no CFL constraint)
  - Stores one extra N-vector (Vm^{n-1})

Ref: DIFFUSION_SPLITTING_DESIGN.md § Strategy 5
"""

from typing import TYPE_CHECKING

from .base import BidomainDiffusionSolver

if TYPE_CHECKING:
    from ...discretization.base import BidomainSpatialDiscretization
    from ...state import BidomainState
    from ..linear_solver.base import LinearSolver


class IMEXSBDF2Solver(BidomainDiffusionSolver):
    """
    IMEX SBDF2 bidomain diffusion solver.

    Uses BDF2 for 2nd-order temporal accuracy. Self-starts with
    BDF1 (backward Euler) on the first step.

    Parameters
    ----------
    spatial : BidomainSpatialDiscretization
        Provides L_i, L_e stencils and operators.
    dt : float
        Time step (ms)
    parabolic_solver : LinearSolver
        Solver for Vm sub-problem
    elliptic_solver : LinearSolver
        Solver for phi_e sub-problem
    pin_node : int
        Node for null space pinning (Neumann only)
    """

    def __init__(self, spatial, dt, parabolic_solver, elliptic_solver,
                 pin_node=0):
        super().__init__(spatial, dt)
        self.parabolic_solver = parabolic_solver
        self.elliptic_solver = elliptic_solver

        self._needs_pinning = spatial.grid.boundary_spec.phi_e_has_null_space
        self._pin_node = pin_node

        # Workspace for BDF2 history
        self._Vm_prev = None      # Vm^{n-1}, None until first step completes
        self._phi_e_prev = None   # phi_e^{n-1}, for the 2nd-order coupling extrapolation
        self._step_count = 0

        self._build_operators(spatial, dt)

    def _build_operators(self, spatial, dt):
        """Build BDF1 and BDF2 parabolic operators + elliptic."""
        # BDF1: A = 1/dt * I - L_i (same as backward Euler, theta=1.0)
        self.A_bdf1, _ = spatial.get_parabolic_operators(dt, theta=1.0)

        # BDF2: A = 3/(2dt) * I - L_i
        # This equals get_parabolic_operators(2*dt/3, theta=1.0)[0]
        self.A_bdf2, _ = spatial.get_parabolic_operators(2.0 * dt / 3.0, theta=1.0)

        # Elliptic (same for all steps)
        self.A_ellip = spatial.get_elliptic_operator()
        if self._needs_pinning:
            self.A_ellip = self.apply_elliptic_pinning(
                self.A_ellip, self._pin_node)

    def step(self, state, dt):
        """
        Advance Vm and phi_e by one diffusion time step.

        Uses BDF1 for the first step, BDF2 thereafter.

        Parameters
        ----------
        state : BidomainState
        dt : float
            Must match the dt used at construction (operators encode 1/dt).
        """
        if abs(dt - self._dt) > 1e-14 * self._dt:
            raise ValueError(
                f"IMEX-SBDF2: step dt={dt} != constructor dt={self._dt}. "
                f"Call rebuild_operators() first.")
        if self._step_count == 0:
            self._step_bdf1(state, dt)
        else:
            self._step_bdf2(state, dt)
        self._step_count += 1

    def _step_bdf1(self, state, dt):
        """First step: backward Euler (BDF1)."""
        # Save current Vm, phi_e for BDF2 history (phi_e^0 seeds the first extrapolation)
        self._Vm_prev = state.Vm.clone()
        self._phi_e_prev = state.phi_e.clone()

        # RHS: (1/dt) * Vm^0 + L_i * phi_e^0
        rhs_para = (1.0 / dt) * state.Vm + self._spatial.apply_L_i(state.phi_e)
        Vm_new = self.parabolic_solver.solve(self.A_bdf1, rhs_para)

        # Elliptic solve
        self._solve_elliptic(state, Vm_new)

    def _step_bdf2(self, state, dt):
        """Subsequent steps: BDF2 (2nd order)."""
        Vm_prev = self._Vm_prev

        # 2nd-order extrapolation of the explicit coupling term: 2*phi_e^n - phi_e^{n-1}.
        # Capture BEFORE _solve_elliptic overwrites state.phi_e.
        phi_e_extrap = 2.0 * state.phi_e - self._phi_e_prev

        # Save current Vm, phi_e as the ^{n-1} history for the next step
        self._Vm_prev = state.Vm.clone()
        self._phi_e_prev = state.phi_e.clone()

        # RHS: (2/dt) * Vm^n - (1/(2dt)) * Vm^{n-1} + L_i * (2*phi_e^n - phi_e^{n-1})
        rhs_para = (2.0 / dt) * state.Vm \
                   - (1.0 / (2.0 * dt)) * Vm_prev \
                   + self._spatial.apply_L_i(phi_e_extrap)
        Vm_new = self.parabolic_solver.solve(self.A_bdf2, rhs_para)

        # Elliptic solve
        self._solve_elliptic(state, Vm_new)

    def _solve_elliptic(self, state, Vm_new):
        """Shared elliptic solve and state update."""
        rhs_ellip = self._spatial.apply_L_i(Vm_new)
        self._zero_dirichlet_rhs(rhs_ellip)
        if self._needs_pinning:
            rhs_ellip[self._pin_node] = 0.0
        phi_e_new = self.elliptic_solver.solve(self.A_ellip, rhs_ellip)

        if self._needs_pinning:
            phi_e_new = phi_e_new - phi_e_new[self._pin_node]

        state.Vm.copy_(Vm_new)
        state.phi_e.copy_(phi_e_new)

    def rebuild_operators(self, spatial, dt):
        """Rebuild operators when dt changes."""
        self._spatial = spatial
        self._dt = dt
        self._build_operators(spatial, dt)
        # Reset history since dt changed
        self._Vm_prev = None
        self._phi_e_prev = None
        self._step_count = 0
