"""Operator splitting strategies."""

from .base import SplittingStrategy
from .strang import StrangSplitting
from .godunov import GodunovSplitting

__all__ = ['SplittingStrategy', 'StrangSplitting', 'GodunovSplitting']
