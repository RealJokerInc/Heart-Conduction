"""
MHAS13 Model Parameters

Extends PHAS13Parameters with maturation modifications:
- GK1_ttp06: TTP06 IK1 conductance (replaces native Paci IK1)
- g_f = 0: If (funny current) suppressed

Fabbri et al. 2019 tested GK1_critical = 3170 S/F for the TTP06 IK1
formulation on the Paci 2013 model. In TTP06 nS/pF units, this is 3.170.
The original TTP06 value is 5.405. We use GK1_critical as default.
"""

from dataclasses import dataclass
from ..phas13.parameters import PHAS13Parameters


@dataclass
class MHAS13Parameters(PHAS13Parameters):
    """
    Matured PHAS13 parameters.

    Inherits all PHAS13 parameters, overrides:
    - g_f = 0.0 (suppress If / funny current)
    - GK1_ttp06 = 3.170 (TTP06 IK1 at GK1_critical from Fabbri 2019)

    The native Paci IK1 (g_K1) is retained but unused — the TTP06
    IK1 formulation replaces it in the model step.
    """

    # Maturation: suppress funny current
    g_f: float = 0.0

    # TTP06 IK1 conductance (nS/pF, same as TTP06 convention)
    # Fabbri 2019 GK1_critical = 3170 S/F = 3.170 nS/pF
    # Original TTP06 value = 5.405 nS/pF
    GK1_ttp06: float = 3.170
