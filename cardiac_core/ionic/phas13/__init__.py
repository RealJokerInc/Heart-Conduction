"""
PHAS13 hiPSC-CM Ionic Model Package

Paci-Hyttinen-Aalto-Setala-Severi 2013 human induced pluripotent stem
cell-derived cardiomyocyte model with 17 state variables and 12 ionic
currents. Features spontaneous beating via If (funny current).

Reference:
Paci M, Hyttinen J, Aalto-Setala K, Severi S (2013).
Ann Biomed Eng 41(11):2334-2348.
"""

from .model import PHAS13Model
from .parameters import StateIndex, PHAS13Parameters, STATE_NAMES

__all__ = [
    'PHAS13Model',
    'StateIndex',
    'PHAS13Parameters',
    'STATE_NAMES',
]
