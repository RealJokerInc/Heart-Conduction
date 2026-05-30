"""
Classical Simulation Path (FEM / FDM / FVM)

Operator splitting with pluggable spatial discretization and time stepping.

Subpackages:
- discretization_scheme/: Spatial operators (M, K, L)
- solver/: Time stepping algorithms (splitting, ionic, diffusion)
- state.py: Runtime state container
- monodomain.py: Top-level orchestrator
"""

from .state import SimulationState
from .monodomain import MonodomainSimulation

__all__ = ['SimulationState', 'MonodomainSimulation']
