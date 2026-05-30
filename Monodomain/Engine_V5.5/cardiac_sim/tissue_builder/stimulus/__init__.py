"""
Stimulus Subpackage

Provides stimulus definition and pacing protocols:
- Stimulus: Single stimulus event
- StimulusProtocol: Collection of stimuli
- Region functions: rectangular, circular, etc.
"""

from .protocol import Stimulus, StimulusProtocol
from .regions import rectangular_region, circular_region, left_edge_region, point_stimulus

__all__ = [
    'Stimulus',
    'StimulusProtocol',
    'rectangular_region',
    'circular_region',
    'left_edge_region',
    'point_stimulus',
]
