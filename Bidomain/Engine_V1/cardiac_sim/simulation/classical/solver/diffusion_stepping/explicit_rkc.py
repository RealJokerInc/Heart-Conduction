"""
Explicit Runge-Kutta-Chebyshev (RKC) Bidomain Diffusion Solver

s-stage RKC extends the Forward Euler stability region by O(s^2) using
the Chebyshev polynomial 3-term recurrence. Eliminates the parabolic
linear solve entirely.

Algorithm (direct Chebyshev 3-term recurrence):
    Given F(Vm) = L_i * (Vm + phi_e^n), with phi_e frozen:

    W_0 = Vm^n
    W_1 = w0 * Vm^n + w1 * dt * F(Vm^n)
    For j = 2, ..., s:
        W_j = 2*w0*W_{j-1} - W_{j-2} + 2*w1*dt*F(W_{j-1})
    Vm^{n+1} = W_s / T_s(w0)

    Then: -(L_i + L_e) * phi_e^{n+1} = L_i * Vm^{n+1}

    w0 = 1 + eps/s^2 (damping shift, eps ~ 0.05)
    w1 = T_s'(w0) / T_s''(w0)

The stability polynomial is P_s(z) = T_s(w0 + w1*z) / T_s(w0),
which maps the stability region to [-2*s^2, 0] along the real axis.

Properties:
    - No parabolic linear solve (only SpMV per stage)
    - 1 elliptic solve per step (same as semi-implicit)
    - Stability: dt_max ~ 0.65 * s^2 * dx^2 / (2*D_i) for s stages
    - Cost: s SpMV + 1 elliptic solve per step

Ref: Sommeijer, Shampine, Verwer (1998) "RKC: An Explicit Method for
     Parabolic PDEs", SIAM J Sci Comput.
Ref: DIFFUSION_SPLITTING_DESIGN.md § Strategy 6
"""

import math
from typing import TYPE_CHECKING

from .base import BidomainDiffusionSolver

if TYPE_CHECKING:
    from ...discretization.base import BidomainSpatialDiscretization
    from ...state import BidomainState
    from ..linear_solver.base import LinearSolver


def _chebyshev_T(n, x):
    """Chebyshev polynomial of the first kind T_n(x) via 3-term recurrence."""
    if n == 0:
        return 1.0
    if n == 1:
        return x
    T_prev2 = 1.0
    T_prev1 = x
    for _ in range(2, n + 1):
        T_curr = 2.0 * x * T_prev1 - T_prev2
        T_prev2, T_prev1 = T_prev1, T_curr
    return T_prev1


def _chebyshev_Tp(n, x):
    """First derivative T'_n(x) = n * U_{n-1}(x)."""
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    U_prev2 = 1.0
    U_prev1 = 2.0 * x
    for _ in range(2, n):
        U_curr = 2.0 * x * U_prev1 - U_prev2
        U_prev2, U_prev1 = U_prev1, U_curr
    return n * U_prev1


def _chebyshev_Tpp(n, x):
    """Second derivative T''_n(x) from the Chebyshev ODE:
    (1-x^2)*T'' - x*T' + n^2*T = 0  =>  T'' = (n^2*T - x*T')/(x^2-1).
    """
    if n <= 1:
        return 0.0
    T_n = _chebyshev_T(n, x)
    Tp_n = _chebyshev_Tp(n, x)
    denom = x * x - 1.0
    if abs(denom) < 1e-15:
        # Limit as x → 1: T''_n(1) = n^2*(n^2-1)/3
        return n * n * (n * n - 1.0) / 3.0
    return (n * n * T_n - x * Tp_n) / denom


class ExplicitRKCSolver(BidomainDiffusionSolver):
    """
    Explicit Runge-Kutta-Chebyshev (RKC) bidomain diffusion solver.

    Uses the direct Chebyshev 3-term recurrence for the parabolic step.
    Each stage applies L_i (one SpMV) with phi_e frozen from the
    previous step. After all stages, one elliptic solve updates phi_e.

    Parameters
    ----------
    spatial : BidomainSpatialDiscretization
        Provides L_i, L_e stencils and elliptic operator.
    dt : float
        Time step (ms)
    elliptic_solver : LinearSolver
        Solver for phi_e sub-problem (one solve per step)
    n_stages : int
        Number of RKC stages (>= 2). More stages = larger stability region.
    damping : float
        RKC damping parameter epsilon (default 0.05)
    pin_node : int
        Node for null space pinning (Neumann only)
    """

    def __init__(self, spatial, dt, elliptic_solver, n_stages=20,
                 damping=0.05, pin_node=0):
        super().__init__(spatial, dt)
        self.elliptic_solver = elliptic_solver

        if n_stages < 2:
            raise ValueError(f"RKC requires n_stages >= 2, got {n_stages}")
        self.n_stages = n_stages
        self.damping = damping

        self._needs_pinning = spatial.grid.boundary_spec.phi_e_has_null_space
        self._pin_node = pin_node

        # Precompute RKC parameters
        # w0 > 1 shifts the Chebyshev polynomial for damping
        # w1 = T_s(w0)/T_s'(w0) ensures first-order consistency (dP/dz = 1)
        s = n_stages
        eps = damping
        self._w0 = 1.0 + eps / (s * s)
        self._Ts_w0 = _chebyshev_T(s, self._w0)
        Tps = _chebyshev_Tp(s, self._w0)
        self._w1 = self._Ts_w0 / Tps

        # Check stability
        D_i = spatial.conductivity.D_i
        dx = spatial.grid.dx
        dt_fe = dx ** 2 / (4.0 * D_i)  # Forward Euler CFL limit
        dt_rkc = 0.65 * s ** 2 * dt_fe  # RKC stability limit
        if dt > dt_rkc:
            raise ValueError(
                f"RKC stability violated: dt={dt:.4f} > dt_max={dt_rkc:.4f} ms "
                f"(s={n_stages}). Increase n_stages or reduce dt.")
        self._stability_ratio = dt / dt_rkc

        self._build_operators(spatial)

    def _build_operators(self, spatial):
        """Build elliptic operator."""
        self.A_ellip = spatial.get_elliptic_operator()
        if self._needs_pinning:
            self.A_ellip = self.apply_elliptic_pinning(
                self.A_ellip, self._pin_node)

    def step(self, state, dt):
        """
        Advance Vm and phi_e by one diffusion time step.

        RKC 3-term Chebyshev recurrence for Vm, then one elliptic solve.
        """
        s = self.n_stages
        w0 = self._w0
        w1 = self._w1
        Ts_w0 = self._Ts_w0
        phi_e_frozen = state.phi_e

        # F(Vm) = L_i * (Vm + phi_e^n)
        def F(Vm):
            return self._spatial.apply_L_i(Vm + phi_e_frozen)

        # Stage 0
        W_prev2 = state.Vm.clone()  # W_0

        # Stage 1: W_1 = w0 * Vm + w1 * dt * F(Vm)
        F0 = F(W_prev2)
        W_prev1 = w0 * W_prev2 + w1 * dt * F0

        # Stages 2..s: W_j = 2*w0*W_{j-1} - W_{j-2} + 2*w1*dt*F(W_{j-1})
        for _ in range(2, s + 1):
            Fj = F(W_prev1)
            W_curr = 2.0 * w0 * W_prev1 - W_prev2 + 2.0 * w1 * dt * Fj
            W_prev2 = W_prev1
            W_prev1 = W_curr

        # Normalize: Vm^{n+1} = W_s / T_s(w0)
        Vm_new = W_prev1 / Ts_w0

        # Elliptic solve for phi_e
        rhs_ellip = self._spatial.apply_L_i(Vm_new)
        if self._needs_pinning:
            rhs_ellip[self._pin_node] = 0.0
        phi_e_new = self.elliptic_solver.solve(self.A_ellip, rhs_ellip)

        if self._needs_pinning:
            phi_e_new = phi_e_new - phi_e_new[self._pin_node]

        state.Vm.copy_(Vm_new)
        state.phi_e.copy_(phi_e_new)

    def rebuild_operators(self, spatial, dt):
        """Rebuild operators when dt changes."""
        D_i = spatial.conductivity.D_i
        dx = spatial.grid.dx
        dt_fe = dx ** 2 / (4.0 * D_i)
        dt_rkc = 0.65 * self.n_stages ** 2 * dt_fe
        if dt > dt_rkc:
            raise ValueError(
                f"RKC stability violated: dt={dt:.4f} > dt_max={dt_rkc:.4f}")
        self._stability_ratio = dt / dt_rkc
        self._spatial = spatial
        self._dt = dt
        self._build_operators(spatial)
