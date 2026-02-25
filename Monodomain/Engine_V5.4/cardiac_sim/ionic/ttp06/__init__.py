"""
TTP06 (ten Tusscher-Panfilov 2006) Ionic Model Package

Human ventricular myocyte model with 19 state variables and 12 ionic currents.

Reference:
ten Tusscher KHWJ, Panfilov AV (2006). "Alternans and spiral breakup in a
human ventricular tissue model." Am J Physiol Heart Circ Physiol.
"""

from .model import TTP06Model
from .parameters import StateIndex, TTP06Parameters, STATE_NAMES

__all__ = [
    'TTP06Model',
    'StateIndex',
    'TTP06Parameters',
    'STATE_NAMES',
]
