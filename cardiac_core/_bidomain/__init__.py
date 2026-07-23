"""cardiac_core._bidomain — the bidomain solver implementation.

Holds the full bidomain solver stack: spatial discretization, splitting strategies,
ionic/diffusion steppers, and the linear-solver tiers. Ionic models and mesh types are
shared with the rest of cardiac_core (``cardiac_core.ionic`` / ``cardiac_core.mesh``);
only ``BidomainConductivity`` is bidomain-specific and lives here.

This package is private (leading underscore) so it does not shadow the public
``bidomain()`` factory; import the names re-exported below rather than reaching into
submodules.
"""

from .simulation.classical import BidomainSimulation
from .simulation.classical.discretization import (
    BidomainFDMDiscretization,
    BidomainSpatialDiscretization,
)
from .tissue.conductivity import BidomainConductivity
from .tissue.isotropic import IsotropicTissue

__all__ = [
    "BidomainSimulation",
    "BidomainFDMDiscretization",
    "BidomainSpatialDiscretization",
    "BidomainConductivity",
    "IsotropicTissue",
]
