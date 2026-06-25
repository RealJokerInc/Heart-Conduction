"""
BDF1 (Backward Euler) Diffusion Solver

First-order implicit time integrator for diffusion.

(A_lhs) * V^{n+1} = (B_rhs) * V^n

Uses LinearSolver for the implicit system.

Ref: V5.3/IMPLEMENTATION.md:L780-825
"""

from typing import TYPE_CHECKING
import torch

from ..base import DiffusionSolver

if TYPE_CHECKING:
    from ....discretization_scheme.base import SpatialDiscretization, DiffusionOperators
    from .....state import SimulationState
    from ..linear_solver.base import LinearSolver


class BDF1Solver(DiffusionSolver):
    """
    BDF1 (Backward Euler) implicit diffusion solver.

    - First-order accurate in time O(dt)
    - L-stable (strongly damping)

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
        super().__init__(spatial, dt)

    def _build_operators(
        self,
        spatial: 'SpatialDiscretization',
        dt: float
    ) -> 'DiffusionOperators':
        """Build BDF1 operators from spatial discretization."""
        return spatial.get_diffusion_operators(dt, "BDF1")

    def step(self, state: 'SimulationState', dt: float) -> None:
        """
        Advance diffusion by dt using BDF1.

        Solves: A_lhs * V^{n+1} = B_rhs * V^n

        Parameters
        ----------
        state : SimulationState
            Simulation state (modified in-place)
        dt : float
            Time step (ms)
        """
        V = state.V

        # Compute RHS: b = B_rhs * V
        rhs = torch.sparse.mm(
            self.ops.B_rhs, V.unsqueeze(1)
        ).squeeze(1)

        # Solve: A_lhs * V_new = rhs
        V_new = self.linear_solver.solve(self.ops.A_lhs, rhs)

        # Update state in-place
        state.V = V_new
