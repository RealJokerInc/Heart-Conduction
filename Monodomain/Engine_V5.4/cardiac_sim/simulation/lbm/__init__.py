"""
Lattice-Boltzmann Simulation Path

Self-contained LBM-EP implementation. Shares only ionic/ with classical path.

Key classes:
- LBMSimulation: Top-level orchestrator
- LBMState: State container (distributions, voltage, ionic states)
- BGKCollision: Single relaxation time (isotropic)
- MRTCollision: Multiple relaxation time (anisotropic)
- d2q5: 2D lattice constants
- d3q7: 3D lattice constants
"""

from .d2q5 import D2Q5, d2q5
from .d3q7 import D3Q7, d3q7
from .collision import (
    CollisionOperator,
    BGKCollision,
    MRTCollision,
    create_isotropic_bgk,
    create_anisotropic_mrt,
)
from .state import LBMState, create_lbm_state
from .monodomain import LBMSimulation

__all__ = [
    'D2Q5', 'd2q5',
    'D3Q7', 'd3q7',
    'CollisionOperator', 'BGKCollision', 'MRTCollision',
    'create_isotropic_bgk', 'create_anisotropic_mrt',
    'LBMState', 'create_lbm_state',
    'LBMSimulation',
]
