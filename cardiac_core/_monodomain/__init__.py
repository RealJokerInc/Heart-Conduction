"""cardiac_core._monodomain — the vendored Monodomain V5.5 solver (engine_consolidation).

Copied verbatim from Monodomain/Engine_V5.5/cardiac_sim (solver subtree only — `simulation/`,
`utils/`, `tissue/`); the only edits are the 8 cross-imports to the shared `cardiac_core.{ionic,
mesh,stimulus}` packages. Solver-internal relative imports are untouched. The underscore is
intentional (private plumbing): `from cardiac_core._monodomain import MonodomainSimulation`; the
public surface is the `cardiac_core.monodomain(...)` factory.
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
