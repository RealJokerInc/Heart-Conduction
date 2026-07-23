"""cardiac_core.stimulus — shared stimulus protocol + region helpers.

`protocol.py` accumulates overlapping stimuli (`+=`) rather than overwriting; the two conventions
differ only where two stimulus regions overlap in both space and time.
"""

from .protocol import Stimulus, StimulusProtocol
from .regions import rectangular_region, circular_region, left_edge_region, point_stimulus
from .stim import Stim

__all__ = [
    'Stim',
    'Stimulus',
    'StimulusProtocol',
    'rectangular_region',
    'circular_region',
    'left_edge_region',
    'point_stimulus',
]
