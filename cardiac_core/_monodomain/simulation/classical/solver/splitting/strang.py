"""
Strang Operator Splitting

Second-order splitting: half-ionic, full-diffusion, half-ionic.

Ref: improvement.md:L999-1004
"""

from typing import TYPE_CHECKING

from .base import SplittingStrategy

if TYPE_CHECKING:
    from ..ionic_time_stepping.base import IonicSolver
    from ..diffusion_time_stepping.base import DiffusionSolver
    from ....state import SimulationState


class StrangSplitting(SplittingStrategy):
    """
    Strang second-order operator splitting.

    step(state, dt):
        1. ionic_solver.step(state, dt/2)
        2. diffusion_solver.step(state, dt)
        3. ionic_solver.step(state, dt/2)

    Second-order accurate in time O(dt^2) for the splitting error.
    """

    def step(self, state: 'SimulationState', dt: float) -> None:
        """
        Advance state by dt: half-ionic, full-diffusion, half-ionic.

        Parameters
        ----------
        state : SimulationState
            Simulation state (modified in-place)
        dt : float
            Time step (ms)
        """
        half_dt = dt / 2.0
        self.ionic_solver.step(state, half_dt)
        self.diffusion_solver.step(state, dt)
        self.ionic_solver.step(state, half_dt)
