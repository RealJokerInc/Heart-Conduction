"""cardiac_core._monodomain — the monodomain solver internals.

Contains the solver subtree only (`simulation/`, `utils/`, `tissue/`); ionic models, meshes
and stimuli come from the shared `cardiac_core.{ionic,mesh,stimulus}` packages. The leading
underscore is intentional (private plumbing): the supported public surface is the
`cardiac_core.monodomain(...)` factory, not `from cardiac_core._monodomain import
MonodomainSimulation`.
"""

from .simulation.classical import MonodomainSimulation, SimulationState
from .simulation.classical.discretization_scheme import (
    FDMDiscretization,
    FVMDiscretization,
)

__all__ = [
    "MonodomainSimulation",
    "SimulationState",
    "FDMDiscretization",
    "FVMDiscretization",
]
