"""
SplittingStrategy Abstract Base Class

Owns ionic and diffusion solvers. Determines the order they are called.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ionic_time_stepping.base import IonicSolver
    from ..diffusion_time_stepping.base import DiffusionSolver
    from ....state import SimulationState


class SplittingStrategy(ABC):
    """
    Abstract base class for operator splitting strategies.

    Owns ionic and diffusion solvers and determines the order
    they are called to advance the simulation by one time step.

    Ref: improvement.md:L975-991
    """

    def __init__(
        self,
        ionic_solver: 'IonicSolver',
        diffusion_solver: 'DiffusionSolver'
    ):
        """
        Initialize splitting strategy.

        Parameters
        ----------
        ionic_solver : IonicSolver
            Solver for ionic ODEs
        diffusion_solver : DiffusionSolver
            Solver for diffusion PDE
        """
        self.ionic_solver = ionic_solver
        self.diffusion_solver = diffusion_solver

    @abstractmethod
    def step(self, state: 'SimulationState', dt: float) -> None:
        """
        Advance state by dt using operator splitting.

        Modifies state in-place.

        Parameters
        ----------
        state : SimulationState
            Current simulation state (modified in-place)
        dt : float
            Time step (ms)
        """
        pass
