"""cardiac_core.stimulus — shared stimulus protocol + region helpers (engine_consolidation).

`protocol.py` is the canonical accumulate (`+=`) form (bidomain/LBM convention; the census decided
accumulate is canonical — V5.5's protocol overwrote (`=`), which differs only for OVERLAPPING stimuli).
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
