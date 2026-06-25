"""
RK2 (Heun's Method) Diffusion Solver

Second-order explicit time integrator for diffusion.
Two Laplacian evaluations per step, allows ~2x larger dt than Forward Euler
for same accuracy.

k1 = L * V^n
k2 = L * (V^n + dt * k1)
V^{n+1} = V^n + dt/2 * (k1 + k2)

CFL-limited: dt <= Cm * h^2 / (2 * D_max)

Ref: Research/02_openCARP:L280-290
"""

from typing import TYPE_CHECKING

from ..base import DiffusionSolver

if TYPE_CHECKING:
    from ....discretization_scheme.base import SpatialDiscretization, DiffusionOperators
    from .....state import SimulationState


class RK2Solver(DiffusionSolver):
    """
    RK2 (Heun's method) explicit diffusion solver.

    Second-order accurate in time O(dt²). Uses two evaluations of the
    diffusion operator per step.

    k1 = L * V^n
    k2 = L * (V^n + dt * k1)
    V^{n+1} = V^n + dt/2 * (k1 + k2)

    CFL-limited, but allows approximately 2x larger dt than Forward Euler
    for the same accuracy level.

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
        Advance diffusion by dt using RK2 (Heun's method).

        k1 = L * V^n
        k2 = L * (V^n + dt * k1)
        V^{n+1} = V^n + dt/2 * (k1 + k2)

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

        # Stage 2: k2 = L * (V + dt * k1)
        V_stage = V + dt * k1
        k2 = self._spatial.apply_diffusion(V_stage)

        # Update: V_new = V + dt/2 * (k1 + k2)
        state.V = V + 0.5 * dt * (k1 + k2)
