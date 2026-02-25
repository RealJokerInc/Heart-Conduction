"""
RK4 (Classical Runge-Kutta) Diffusion Solver

Fourth-order explicit time integrator for diffusion.
Four Laplacian evaluations per step. High accuracy, useful as reference
solution generator.

k1 = L * V^n
k2 = L * (V^n + dt/2 * k1)
k3 = L * (V^n + dt/2 * k2)
k4 = L * (V^n + dt * k3)
V^{n+1} = V^n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

Ref: Research/00_Research_Summary:L68
"""

from typing import TYPE_CHECKING

from ..base import DiffusionSolver

if TYPE_CHECKING:
    from ....discretization_scheme.base import SpatialDiscretization, DiffusionOperators
    from .....state import SimulationState


class RK4Solver(DiffusionSolver):
    """
    RK4 (Classical Runge-Kutta) explicit diffusion solver.

    Fourth-order accurate in time O(dt⁴). Uses four evaluations of the
    diffusion operator per step.

    Overkill for most cardiac simulations, but useful for:
    - Generating reference solutions
    - Verifying convergence order of other methods
    - High-accuracy requirements

    Parameters
    ----------
    spatial : SpatialDiscretization
        Provides apply_diffusion() method
    dt : float
        Time step (ms) - stored for API consistency
    """

    def __init__(
        self,
        spatial: 'SpatialDiscretization',
        dt: float
    ):
        self._spatial = spatial
        self._dt = dt
        self.ops = None  # Not needed for explicit method

    def _build_operators(
        self,
        spatial: 'SpatialDiscretization',
        dt: float
    ) -> 'DiffusionOperators':
        """No operators needed for explicit method."""
        return None

    def step(self, state: 'SimulationState', dt: float) -> None:
        """
        Advance diffusion by dt using classical RK4.

        k1 = L * V^n
        k2 = L * (V^n + dt/2 * k1)
        k3 = L * (V^n + dt/2 * k2)
        k4 = L * (V^n + dt * k3)
        V^{n+1} = V^n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        Parameters
        ----------
        state : SimulationState
            Simulation state (modified in-place)
        dt : float
            Time step (ms)
        """
        V = state.V

        # Stage 1: k1 = L * V
        k1 = self._spatial.apply_diffusion(V)

        # Stage 2: k2 = L * (V + dt/2 * k1)
        V_stage2 = V + 0.5 * dt * k1
        k2 = self._spatial.apply_diffusion(V_stage2)

        # Stage 3: k3 = L * (V + dt/2 * k2)
        V_stage3 = V + 0.5 * dt * k2
        k3 = self._spatial.apply_diffusion(V_stage3)

        # Stage 4: k4 = L * (V + dt * k3)
        V_stage4 = V + dt * k3
        k4 = self._spatial.apply_diffusion(V_stage4)

        # Update: V_new = V + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        state.V = V + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
