"""cardiac_core._bidomain — the vendored Bidomain V1 solver (engine_consolidation).

Copied verbatim from Bidomain/Engine_V1/cardiac_sim (solver subtree only; dead simulation/lbm/
dropped); the only edits are the 9 cross-imports to the shared cardiac_core.{ionic,mesh} packages
(BidomainConductivity stays per-engine under ._bidomain.tissue). Private package (underscore) so it
does not shadow the public bidomain() factory.
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
