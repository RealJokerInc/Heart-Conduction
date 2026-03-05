"""
Godunov (Lie) Operator Splitting

First-order splitting: ionic step then diffusion step.

Ref: improvement.md:L993-997
"""

from typing import TYPE_CHECKING

from .base import SplittingStrategy

if TYPE_CHECKING:
    from ..ionic_stepping.base import IonicSolver
    from ..diffusion_stepping.base import BidomainDiffusionSolver
    from ...state import BidomainState


class GodunovSplitting(SplittingStrategy):
    """
    Godunov (Lie) first-order operator splitting.

    step(state, dt):
        1. ionic_solver.step(state, dt)
        2. diffusion_solver.step(state, dt)

    First-order accurate in time O(dt) for the splitting error.
    """

    def step(self, state: 'BidomainState', dt: float) -> None:
        """
        Advance state by dt: ionic then diffusion.

        Parameters
        ----------
        state : SimulationState
            Simulation state (modified in-place)
        dt : float
            Time step (ms)
        """
        self.ionic_solver.step(state, dt)
        self.diffusion_solver.step(state, dt)
