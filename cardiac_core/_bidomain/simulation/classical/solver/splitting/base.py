"""
SplittingStrategy Abstract Base Class

Owns ionic and diffusion solvers. Determines the order they are called.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ionic_stepping.base import IonicSolver
    from ..diffusion_stepping.base import BidomainDiffusionSolver
    from ...state import BidomainState


class SplittingStrategy(ABC):
    """
    Abstract base class for operator splitting strategies.

    Owns ionic and diffusion solvers and determines the order
    they are called to advance the simulation by one time step.
    """

    def __init__(
        self,
        ionic_solver: 'IonicSolver',
        diffusion_solver: 'BidomainDiffusionSolver'
    ):
        """
        Initialize splitting strategy.

        Parameters
        ----------
        ionic_solver : IonicSolver
            Solver for ionic ODEs
        diffusion_solver : BidomainDiffusionSolver
            Solver for bidomain diffusion PDE
        """
        self.ionic_solver = ionic_solver
        self.diffusion_solver = diffusion_solver

    @abstractmethod
    def step(self, state: 'BidomainState', dt: float) -> None:
        """
        Advance state by dt using operator splitting.

        Modifies state in-place.

        Parameters
        ----------
        state : BidomainState
            Current simulation state (modified in-place)
        dt : float
            Time step (ms)
        """
        pass
