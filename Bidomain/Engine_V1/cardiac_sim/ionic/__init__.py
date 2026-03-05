"""
Ionic Models Package

Contains cardiac ionic models that implement the IonicModel interface:
- ORdModel: O'Hara-Rudy 2011 (41 states)
- TTP06Model: ten Tusscher-Panfilov 2006 (19 states)

Also provides LUT (Lookup Table) acceleration for gating functions.
"""

from .base import IonicModel, CellType
from .ord import ORdModel
from .ttp06 import TTP06Model
from .lut import LookupTable, TTP06LUT, get_ttp06_lut, clear_lut_cache

__all__ = [
    'IonicModel',
    'CellType',
    'ORdModel',
    'TTP06Model',
    'LookupTable',
    'TTP06LUT',
    'get_ttp06_lut',
    'clear_lut_cache',
]
