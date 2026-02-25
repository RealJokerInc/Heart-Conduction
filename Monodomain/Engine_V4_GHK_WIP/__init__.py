"""
Engine V4: Luo-Rudy 1994 (LRd) Cardiac Electrophysiology Model
==============================================================

A biophysical ionic model with 13 state variables, 10 ionic currents,
and Goldman-Hodgkin-Katz calcium dynamics.

Reference:
    Luo CH, Rudy Y. Circ Res. 1994;74(6):1071-1096.
"""

from .parameters_lrd import (
    LRdParams,
    LRdInitialConditions,
    default_params,
    default_initial_conditions,
    STATE_INDICES,
    N_STATES,
)

__version__ = "0.1.0"
__all__ = [
    "LRdParams",
    "LRdInitialConditions",
    "default_params",
    "default_initial_conditions",
    "STATE_INDICES",
    "N_STATES",
]
