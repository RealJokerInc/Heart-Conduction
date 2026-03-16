"""
MHAS13 Current Modifications

Only IK1 changes: uses the ten Tusscher (TTP06) formulation instead of the
native Paci formulation. All other currents are inherited from PHAS13.

The TTP06 IK1 has stronger rectification and more physiological behavior
at hyperpolarized potentials, enabling stable quiescent resting potential.

Unit note: TTP06 IK1 uses Ki as a dynamic state (mM), but PHAS13 fixes
Ki = 150 mM. We pass the fixed value.

Reference:
ten Tusscher KHWJ et al. (2004). Am J Physiol Heart Circ Physiol 286:H1573-89.
Verkerk AO et al. (2019). Biophys J 117:2303-2315.
"""

import torch
from ..ttp06.gating import safe_exp
from ..phas13.currents import E_K


def I_K1_ttp06(V: torch.Tensor,
               Ki: float = 150.0,
               GK1: float = 3.170,
               Ko: float = 5.4) -> torch.Tensor:
    """
    Inward rectifier K+ current — TTP06 formulation.

    This replaces the native Paci IK1 in the matured model. The TTP06
    formulation has stronger rectification, producing a more negative
    and stable resting potential.

    Parameters
    ----------
    V : Membrane potential (mV)
    Ki : Intracellular K+ (mM) — fixed at 150 in PHAS13
    GK1 : Maximum conductance (nS/pF). Default 3.170 = GK1_critical
          from Verkerk 2019 for the Paci model.
    Ko : Extracellular K+ (mM)

    Returns
    -------
    IK1 : Current density (A/F = pA/pF)
    """
    # Ki needs to be a tensor for E_K
    Ki_t = torch.tensor(Ki, dtype=V.dtype, device=V.device) if not isinstance(Ki, torch.Tensor) else Ki
    EK = E_K(Ki, Ko)  # scalar

    # TTP06 rectification (identical to ttp06/currents.py I_K1)
    VmEK = V - EK
    alpha = 0.1 / (1.0 + safe_exp(0.06 * (VmEK - 200.0)))
    beta = (3.0 * safe_exp(0.0002 * (VmEK + 100.0)) +
            safe_exp(0.1 * (VmEK - 10.0))) / \
           (1.0 + safe_exp(-0.5 * VmEK))
    xK1 = alpha / (alpha + beta)

    return GK1 * (Ko / 5.4) ** 0.5 * xK1 * (V - EK)
