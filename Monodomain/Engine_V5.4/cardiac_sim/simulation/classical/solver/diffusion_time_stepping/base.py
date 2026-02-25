"""
DiffusionSolver Abstract Base Class

Owns DiffusionOperators (built at init from spatial discretization).
Implicit solvers also own a LinearSolver.
Advances the diffusion PDE in-place on state.

Ref: improvement.md:L1033-1074
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...discretization_scheme.base import SpatialDiscretization, DiffusionOperators
    from ....state import SimulationState


class DiffusionSolver(ABC):
    """
    Abstract base class for diffusion time integrators.

    Owns DiffusionOperators built from a SpatialDiscretization.
    Implicit solvers also own a LinearSolver.

    The solver is responsible for:
    - Building operators from the spatial discretization
    - Advancing the voltage via diffusion
    - Optionally supporting adaptive time stepping (rebuild_operators)
    """

    def __init__(self, spatial: 'SpatialDiscretization', dt: float):
        """
        Initialize diffusion solver.

        Parameters
        ----------
        spatial : SpatialDiscretization
            Provides get_diffusion_operators()
        dt : float
            Time step (ms) used to build operators
        """
        self._spatial = spatial
        self._dt = dt
        self.ops = self._build_operators(spatial, dt)

    @abstractmethod
    def _build_operators(
        self,
        spatial: 'SpatialDiscretization',
        dt: float
    ) -> 'DiffusionOperators':
        """
        Build scheme-specific operators from spatial discretization.

        Parameters
        ----------
        spatial : SpatialDiscretization
            The spatial discretization
        dt : float
            Time step (ms)

        Returns
        -------
        ops : DiffusionOperators
            A_lhs, B_rhs, apply_mass for this scheme
        """
        pass

    @abstractmethod
    def step(self, state: 'SimulationState', dt: float) -> None:
        """
        Advance diffusion by dt.

        Modifies state.V in-place.

        Parameters
        ----------
        state : SimulationState
            Simulation state (modified in-place)
        dt : float
            Time step (ms)
        """
        pass

    def rebuild_operators(
        self,
        spatial: 'SpatialDiscretization',
        dt: float
    ) -> None:
        """
        Rebuild operators when dt changes (adaptive time stepping).

        Parameters
        ----------
        spatial : SpatialDiscretization
            The spatial discretization
        dt : float
            New time step (ms)
        """
        self._dt = dt
        self.ops = self._build_operators(spatial, dt)
