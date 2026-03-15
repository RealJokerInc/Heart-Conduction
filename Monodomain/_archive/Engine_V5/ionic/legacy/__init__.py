"""
Legacy LRd07 Model Components

This package contains the archived Livshitz-Rudy 2007 cardiac cell model.
Preserved for reference during ORd implementation.

Reference: Livshitz LM, Rudy Y. Am J Physiol Heart Circ Physiol. 2007;292(6):H2854-66.
"""

from .lrd07_parameters import LRd07Parameters, StateIndex as LRd07StateIndex
from .lrd07_model import LRd07Model

__all__ = ['LRd07Parameters', 'LRd07StateIndex', 'LRd07Model']
