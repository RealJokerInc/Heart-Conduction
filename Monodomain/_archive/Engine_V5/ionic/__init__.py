"""
O'Hara-Rudy (ORd 2011) Ionic Model Components

This package contains the ORd human ventricular action potential model.

Reference: O'Hara T, Virag L, Varro A, Rudy Y. PLoS Comput Biol. 2011;7(5):e1002061.

Main components:
- ORdModel: Main model class
- ORdParameters: Parameter container
- StateIndex: State variable indices
- CellType: Cell type enumeration (ENDO, EPI, M_CELL)

Submodules:
- gating: Voltage-dependent gating kinetics
- currents: Ionic current calculations
- calcium: SR and Ca2+ handling
- camkii: CaMKII signaling

Legacy LRd07 model available in legacy/ subpackage.
"""

from .parameters import ORdParameters, StateIndex, CellType, DEFAULT_PARAMS, STATE_NAMES
from .ord_model import ORdModel

__all__ = [
    'ORdModel',
    'ORdParameters',
    'StateIndex',
    'CellType',
    'DEFAULT_PARAMS',
    'STATE_NAMES',
]
