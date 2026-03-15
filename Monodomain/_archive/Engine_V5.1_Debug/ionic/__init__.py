"""
Ionic Model Components for ORd (O'Hara-Rudy 2011) Model

This module provides a complete PyTorch GPU implementation of the ORd
cardiac action potential model.

Main Classes:
- ORdModel: Complete ionic model with step() and run() methods
- CellType: Enum for cell types (ENDO, EPI, M_CELL)
- StateIndex: Enum for state variable indices

Submodules:
- parameters: Model parameters and state indexing
- gating: Gate kinetics (steady-state and time constants)
- currents: All 15 ionic currents
- calcium: SR handling, buffering, concentration updates
- camkii: CaMKII signaling pathway
"""

from .parameters import (
    StateIndex,
    CellType,
    ORdParameters,
    get_celltype_parameters,
    get_initial_state,
)

from .model import ORdModel

from . import gating
from . import currents
from . import calcium
from . import camkii

__all__ = [
    # Main model
    'ORdModel',
    # Enums and parameters
    'StateIndex',
    'CellType',
    'ORdParameters',
    'get_celltype_parameters',
    'get_initial_state',
    # Submodules
    'gating',
    'currents',
    'calcium',
    'camkii',
]
