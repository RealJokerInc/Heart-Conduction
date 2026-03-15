"""
Ionic Models Package

Contains cardiac ionic models that implement the IonicModel interface:
- ORdModel: O'Hara-Rudy 2011 (41 states)
- TTP06Model: ten Tusscher-Panfilov 2006 (19 states)
- PHAS13Model: Paci-Hyttinen-Aalto-Setala-Severi 2013 hiPSC-CM (17 states)

Also provides LUT (Lookup Table) acceleration for gating functions.
"""

from .base import IonicModel, CellType
from .ord import ORdModel
from .ttp06 import TTP06Model
from .phas13 import PHAS13Model
from .lut import LookupTable, TTP06LUT, get_ttp06_lut, clear_lut_cache

# Backward compatibility
from .phas13 import PHAS13Model as PaciModel

__all__ = [
    'IonicModel',
    'CellType',
    'ORdModel',
    'TTP06Model',
    'PHAS13Model',
    'PaciModel',  # backward compat
    'LookupTable',
    'TTP06LUT',
    'get_ttp06_lut',
    'clear_lut_cache',
]
