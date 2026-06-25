"""cardiac_core.monodomain — the vendored Monodomain V5.5 solver (engine_consolidation).

Copied verbatim from Monodomain/Engine_V5.5/cardiac_sim (solver subtree only — `simulation/`,
`utils/`, `tissue/`); the only edits are the 8 cross-imports to the shared `cardiac_core.{ionic,
mesh,stimulus}` packages. Solver-internal relative imports are untouched. This facade gives the
flat import surface (`from cardiac_core.monodomain import MonodomainSimulation`).
"""

from .simulation.classical import MonodomainSimulation, SimulationState
from .simulation.classical.discretization_scheme import (
    FDMDiscretization,
    FVMDiscretization,
    FEMDiscretization,
)

__all__ = [
    "MonodomainSimulation",
    "SimulationState",
    "FDMDiscretization",
    "FVMDiscretization",
    "FEMDiscretization",
]
