"""
BidomainDiffusionSolver Abstract Base Class

Solves the bidomain diffusion step. Updates both Vm and phi_e in-place.

Ref: improvement.md L916-948
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...discretization.base import BidomainSpatialDiscretization
    from ...state import BidomainState


class BidomainDiffusionSolver(ABC):
    """
    Abstract base class for bidomain diffusion solvers.

    Solves the bidomain diffusion step, updating both Vm and phi_e.
    The decoupled approach splits this into two sequential N x N SPD solves.

    Owns:
    - BidomainSpatialDiscretization (provides L_i, L_e stencils)
    - LinearSolver for parabolic sub-problem (Vm)
    - LinearSolver for elliptic sub-problem (phi_e)
    """

    def __init__(self, spatial: 'BidomainSpatialDiscretization', dt: float):
        self._spatial = spatial
        self._dt = dt

    @abstractmethod
    def step(self, state: 'BidomainState', dt: float) -> None:
        """
        Advance diffusion by dt.
        Modifies state.Vm AND state.phi_e in-place.
        """
        pass

    def rebuild_operators(self, spatial: 'BidomainSpatialDiscretization', dt: float) -> None:
        """Rebuild operators when dt changes (adaptive time stepping)."""
        self._spatial = spatial
        self._dt = dt
