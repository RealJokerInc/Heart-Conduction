"""
Ionic Models Package

Contains cardiac ionic models that implement the IonicModel interface:
- ORdModel: O'Hara-Rudy 2011 (41 states)
- TTP06Model: ten Tusscher-Panfilov 2006 (19 states)

Also provides LUT (Lookup Table) acceleration for gating functions.
"""

from ionic.base import IonicModel, CellType
from ionic.ord import ORdModel
from ionic.ttp06 import TTP06Model
from ionic.lut import LookupTable, TTP06LUT, get_ttp06_lut, clear_lut_cache

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
