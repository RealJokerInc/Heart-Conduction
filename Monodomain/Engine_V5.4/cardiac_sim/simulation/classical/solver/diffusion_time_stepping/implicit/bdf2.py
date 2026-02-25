"""
BDF2 (Second-Order Backward Differentiation) Diffusion Solver

Second-order implicit time integrator for diffusion.
Two-step method requiring BDF1 for first step.

BDF2 formula:
    3V^{n+1} - 4V^n + V^{n-1} = 2*dt*L*V^{n+1}

Rearranged:
    (3M - 2*dt*L)*V^{n+1} = 4M*V^n - M*V^{n-1}

Where M is the mass matrix (identity for FDM, χ·Cm·Vol for FVM, assembled mass for FEM).

Uses LinearSolver for the implicit system.

Ref: V5.3/IMPLEMENTATION.md:L780-825
Ref: IMPLEMENTATION.md § Phase 6 (6-V10, 6-V11)
"""

from typing import TYPE_CHECKING, Optional
import torch

from ..base import DiffusionSolver

if TYPE_CHECKING:
    from ....discretization_scheme.base import SpatialDiscretization, DiffusionOperators
    from .....state import SimulationState
    from ..linear_solver.base import LinearSolver


class BDF2Solver(DiffusionSolver):
    """
    BDF2 (Second-Order Backward Differentiation) implicit diffusion solver.

    - Second-order accurate in time O(dt²)
    - A-stable (not L-stable like BDF1)
    - Two-step method: requires V^{n-1} from previous step
    - Automatically uses BDF1 for first step

    The solver tracks V_prev internally to compute the two-step RHS.

    Parameters
    ----------
    spatial : SpatialDiscretization
        Provides get_diffusion_operators()
    dt : float
        Time step (ms)
    linear_solver : LinearSolver
        Solver for the implicit linear system
    """

    def __init__(
        self,
        spatial: 'SpatialDiscretization',
        dt: float,
        linear_solver: 'LinearSolver'
    ):
        self.linear_solver = linear_solver
        self._V_prev: Optional[torch.Tensor] = None  # V^{n-1}
        self._first_step = True
        super().__init__(spatial, dt)

        # Also build BDF1 operators for first step
        self._bdf1_ops = spatial.get_diffusion_operators(dt, "BDF1")

    def _build_operators(
        self,
        spatial: 'SpatialDiscretization',
        dt: float
    ) -> 'DiffusionOperators':
        """Build BDF2 operators from spatial discretization."""
        return spatial.get_diffusion_operators(dt, "BDF2")

    def step(self, state: 'SimulationState', dt: float) -> None:
        """
        Advance diffusion by dt using BDF2.

        First step uses BDF1 (no V^{n-1} available).
        Subsequent steps use BDF2:
            A_lhs * V^{n+1} = B_rhs * V^n - M * V^{n-1}

        Parameters
        ----------
        state : SimulationState
            Simulation state (modified in-place)
        dt : float
            Time step (ms)
        """
        V = state.V.clone()

        if self._first_step:
            # Use BDF1 for first step
            # A_lhs * V^{n+1} = B_rhs * V^n
            rhs = torch.sparse.mm(
                self._bdf1_ops.B_rhs, V.unsqueeze(1)
            ).squeeze(1)

            V_new = self.linear_solver.solve(self._bdf1_ops.A_lhs, rhs)

            # Store V^n for next step
            self._V_prev = V
            self._first_step = False
        else:
            # BDF2: A_lhs * V^{n+1} = B_rhs * V^n - M * V^{n-1}
            # B_rhs encodes 4*M, so RHS = 4*M*V^n - M*V^{n-1}
            # We use apply_mass to compute M*V^{n-1}

            # Compute 4*M*V^n (B_rhs * V)
            rhs_4M_V = torch.sparse.mm(
                self.ops.B_rhs, V.unsqueeze(1)
            ).squeeze(1)

            # Compute M*V^{n-1}
            M_V_prev = self.ops.apply_mass(self._V_prev)

            # Full RHS
            rhs = rhs_4M_V - M_V_prev

            # Solve: A_lhs * V^{n+1} = rhs
            V_new = self.linear_solver.solve(self.ops.A_lhs, rhs)

            # Shift history: V^{n-1} <- V^n
            self._V_prev = V

        # Update state in-place
        state.V = V_new

    def reset(self) -> None:
        """
        Reset solver state for new simulation.

        Clears the previous voltage history, requiring BDF1 for next first step.
        """
        self._V_prev = None
        self._first_step = True

    def rebuild_operators(
        self,
        spatial: 'SpatialDiscretization',
        dt: float
    ) -> None:
        """
        Rebuild operators when dt changes.

        Also rebuilds BDF1 operators and resets history (dt change
        invalidates previous V^{n-1} relationship).
        """
        super().rebuild_operators(spatial, dt)
        self._bdf1_ops = spatial.get_diffusion_operators(dt, "BDF1")
        self.reset()
