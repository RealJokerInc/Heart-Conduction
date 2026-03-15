"""
O'Hara-Rudy CaMKII Signaling Module

CaMKII (Ca2+/calmodulin-dependent protein kinase II) modulates multiple
ion channels in the ORd model, affecting gating kinetics and conductances.

Reference: O'Hara T, Virag L, Varro A, Rudy Y. PLoS Comput Biol. 2011;7(5):e1002061.
"""

import numpy as np
from numba import njit


# =============================================================================
# CaMKII Activation Dynamics
# =============================================================================

@njit(cache=True)
def compute_CaMKa(CaMKt: float, cass: float,
                  CaMKo: float = 0.05, KmCaM: float = 0.0015) -> tuple:
    """
    Compute active CaMKII from trapped CaMKII and subspace Ca2+.

    CaMKII binds Ca2+/CaM complex, becomes active, and can be "trapped"
    in an active state even after Ca2+ returns to baseline.

    From C++:
        CaMKb = CaMKo * (1.0 - CaMKt) / (1.0 + KmCaM/cass)
        CaMKa = CaMKb + CaMKt

    Args:
        CaMKt: Trapped (autophosphorylated) CaMKII fraction
        cass: Subspace Ca2+ concentration (mM)
        CaMKo: Total CaMKII concentration (default 0.05)
        KmCaM: Ca2+/CaM binding Kd (default 0.0015 mM)

    Returns:
        Tuple of (CaMKb, CaMKa):
            CaMKb: Bound (Ca2+/CaM-activated) CaMKII
            CaMKa: Total active CaMKII (bound + trapped)
    """
    CaMKb = CaMKo * (1.0 - CaMKt) / (1.0 + KmCaM / cass)
    CaMKa = CaMKb + CaMKt
    return CaMKb, CaMKa


@njit(cache=True)
def dCaMKt_dt(CaMKb: float, CaMKt: float,
              aCaMK: float = 0.05, bCaMK: float = 0.00068) -> float:
    """
    Derivative of trapped CaMKII.

    Trapping occurs when active CaMKII autophosphorylates. This creates
    a "memory" effect where CaMKII remains active after Ca2+ returns
    to baseline.

    From C++:
        dCaMKt = aCaMK * CaMKb * (CaMKb + CaMKt) - bCaMK * CaMKt

    Args:
        CaMKb: Bound (Ca2+/CaM-activated) CaMKII
        CaMKt: Trapped (autophosphorylated) CaMKII
        aCaMK: Trapping rate constant (default 0.05)
        bCaMK: Detrapping rate constant (default 0.00068)

    Returns:
        dCaMKt/dt
    """
    return aCaMK * CaMKb * (CaMKb + CaMKt) - bCaMK * CaMKt


# =============================================================================
# Phosphorylation Fractions
# =============================================================================

@njit(cache=True)
def fCaMKp(CaMKa: float, KmCaMK: float = 0.15) -> float:
    """
    Fraction of channels in CaMKII-phosphorylated state.

    This function determines the weighting between non-phosphorylated
    and phosphorylated channel populations for:
    - INa (hsp, jp gates)
    - INaL (hLp gate)
    - Ito (ap, iFp, iSp gates)
    - ICaL (ffp, fcafp gates)
    - Jrel (Jrelp)
    - Jup (Jupp)

    From C++:
        fINap = 1.0 / (1.0 + KmCaMK/CaMKa)

    Args:
        CaMKa: Total active CaMKII
        KmCaMK: Half-saturation for CaMKII effect (default 0.15)

    Returns:
        Phosphorylation fraction [0, 1]
    """
    return 1.0 / (1.0 + KmCaMK / CaMKa)


# =============================================================================
# CaMKII Step Function (for integration)
# =============================================================================

@njit(cache=True)
def camkii_step(CaMKt: float, cass: float, dt: float,
                CaMKo: float = 0.05, KmCaM: float = 0.0015,
                aCaMK: float = 0.05, bCaMK: float = 0.00068) -> tuple:
    """
    Update CaMKII state for one time step.

    Uses forward Euler integration for CaMKt.

    Args:
        CaMKt: Current trapped CaMKII
        cass: Current subspace Ca2+
        dt: Time step (ms)
        CaMKo: Total CaMKII
        KmCaM: CaM binding Kd
        aCaMK: Trapping rate
        bCaMK: Detrapping rate

    Returns:
        Tuple of (CaMKt_new, CaMKa):
            CaMKt_new: Updated trapped CaMKII
            CaMKa: Current active CaMKII (for use in current calculations)
    """
    CaMKb, CaMKa = compute_CaMKa(CaMKt, cass, CaMKo, KmCaM)
    dCaMKt = dCaMKt_dt(CaMKb, CaMKt, aCaMK, bCaMK)
    CaMKt_new = CaMKt + dt * dCaMKt

    # Ensure bounds
    if CaMKt_new < 0.0:
        CaMKt_new = 0.0
    elif CaMKt_new > 1.0:
        CaMKt_new = 1.0

    return CaMKt_new, CaMKa
