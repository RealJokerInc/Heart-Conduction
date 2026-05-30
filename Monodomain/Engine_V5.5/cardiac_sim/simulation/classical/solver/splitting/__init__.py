"""
Operator Splitting Strategies

Splitting strategies determine the order of ionic and diffusion sub-steps.
- GodunovSplitting: First-order, ionic then diffusion
- StrangSplitting: Second-order, half-ionic, full-diffusion, half-ionic

Ref: improvement.md:L975-1004
"""

from .base import SplittingStrategy
from .godunov import GodunovSplitting
from .strang import StrangSplitting

__all__ = ['SplittingStrategy', 'GodunovSplitting', 'StrangSplitting']
