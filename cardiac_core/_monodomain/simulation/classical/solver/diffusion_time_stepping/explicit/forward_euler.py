"""
Forward Euler Diffusion Solver

Explicit time integrator for diffusion. No linear solve required.
CFL-limited: dt <= Cm * h^2 / (4 * D_max)

V^{n+1} = V^n + dt * L * V^n

Ref: Research/01_FDM:L67 (CFL constraint)
"""

import warnings
from typing import TYPE_CHECKING

from ..base import DiffusionSolver

if TYPE_CHECKING:
    from ....discretization_scheme.base import SpatialDiscretization, DiffusionOperators
    from .....state import SimulationState


class ForwardEulerDiffusionSolver(DiffusionSolver):
    """
    Forward Euler explicit diffusion solver.

    V^{n+1} = V^n + dt * apply_diffusion(V^n)

    No linear solve required, but CFL-limited:
    dt <= Cm * h^2 / (4 * D_max)

    For typical cardiac parameters (h=0.025cm, D=0.001cm^2/ms, Cm=1uF/cm^2):
    dt_max ~ 1 * 0.000625 / (4 * 0.001) = 0.156 ms

    Parameters
    ----------
    spatial : SpatialDiscretization
        Provides apply_diffusion() method
    dt : float
        Time step (ms) - ignored for explicit, but stored for API consistency
    """

    def __init__(
        self,
        spatial: 'SpatialDiscretization',
        dt: float
    ):
        self._spatial = spatial
        self._dt = dt
        self.ops = None  # Not needed for explicit
        self._cfl_checked = False

    def _check_cfl(self, dt: float) -> None:
        """Warn (once) if dt exceeds the explicit CFL stability limit (B6).

        Limit: dt <= chi*Cm*min(dx,dy)^2/(4*D_max). The spatial operator applies
        L*V/(chi*Cm), so the raw D_max and chi*Cm both enter. Skips silently if the
        discretization doesn't expose the needed attributes (e.g. a non-FDM scheme).
        """
        sp = self._spatial
        dx = getattr(sp, '_dx', None)
        dy = getattr(sp, '_dy', None)
        D_max = getattr(sp, '_D_max', None)
        chi = getattr(sp, '_chi', 1.0)
        Cm = getattr(sp, '_Cm', 1.0)
        if dx is None or dy is None or not D_max or D_max <= 0:
            return
        h2 = min(dx, dy) ** 2
        dt_max = chi * Cm * h2 / (4.0 * D_max)
        if dt > dt_max:
            warnings.warn(
                f"forward_euler dt={dt:g} ms exceeds the CFL stability limit "
                f"dt_max={dt_max:g} ms (= chi*Cm*min(dx,dy)^2/(4*D_max)); the explicit "
                f"diffusion update will oscillate/blow up. Reduce dt, refine the grid, "
                f"or use an implicit solver (diffusion_solver='crank_nicolson').",
                stacklevel=3,
            )

    def _build_operators(
        self,
        spatial: 'SpatialDiscretization',
        dt: float
    ) -> 'DiffusionOperators':
        """No operators needed for explicit method."""
        return None

    def step(self, state: 'SimulationState', dt: float) -> None:
        """
        Advance diffusion by dt using Forward Euler.

        V^{n+1} = V^n + dt * L * V^n

        WARNING: Will be unstable if dt exceeds CFL limit!

        Parameters
        ----------
        state : SimulationState
            Simulation state (modified in-place)
        dt : float
            Time step (ms)
        """
        if not self._cfl_checked:
            self._check_cfl(dt)
            self._cfl_checked = True

        V = state.V

        # Compute diffusion term: L * V
        # apply_diffusion returns div(D*grad(V))
        LV = self._spatial.apply_diffusion(V)

        # Forward Euler update: V_new = V + dt * L * V
        state.V = V + dt * LV
