"""
Backward compatibility — use ionic.phas13 directly.

Paci 2013 model renamed to PHAS13 (author initials + year convention).
"""

from ..phas13 import PHAS13Model as PaciModel
from ..phas13 import PHAS13Parameters as PaciParameters
from ..phas13.parameters import StateIndex, STATE_NAMES

__all__ = [
    'PaciModel',
    'StateIndex',
    'PaciParameters',
    'STATE_NAMES',
]
