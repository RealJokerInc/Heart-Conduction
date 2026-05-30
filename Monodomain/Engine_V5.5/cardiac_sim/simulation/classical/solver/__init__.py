"""
Solver Subpackage

Hierarchical solver architecture for operator splitting:
- splitting/: Operator splitting strategies (Godunov, Strang)
- ionic_time_stepping/: Ionic ODE integrators (Rush-Larsen, Forward Euler)
- diffusion_time_stepping/: Diffusion PDE integrators (explicit/implicit)
  - linear_solver/: Linear system solvers (PCG)

Ownership: MonodomainSimulation -> SplittingStrategy -> IonicSolver + DiffusionSolver
All solvers operate on SimulationState in-place. No allocation per step.

Ref: improvement.md:L939-1109
"""

from .splitting import SplittingStrategy, GodunovSplitting, StrangSplitting
from .ionic_time_stepping import IonicSolver, RushLarsenSolver, ForwardEulerIonicSolver
from .diffusion_time_stepping import DiffusionSolver
from .diffusion_time_stepping.implicit import CrankNicolsonSolver, BDF1Solver
from .diffusion_time_stepping.explicit import ForwardEulerDiffusionSolver
from .diffusion_time_stepping.linear_solver import LinearSolver, PCGSolver

__all__ = [
    # Splitting
    'SplittingStrategy', 'GodunovSplitting', 'StrangSplitting',
    # Ionic
    'IonicSolver', 'RushLarsenSolver', 'ForwardEulerIonicSolver',
    # Diffusion
    'DiffusionSolver', 'CrankNicolsonSolver', 'BDF1Solver', 'ForwardEulerDiffusionSolver',
    # Linear
    'LinearSolver', 'PCGSolver',
]
