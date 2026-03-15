"""
CaMKII (Calcium/Calmodulin-Dependent Protein Kinase II) Signaling

CaMKII modulates multiple ion channels and transporters in cardiac myocytes:
- INa: Faster recovery from inactivation
- INaL: Increased late current
- Ito: Altered inactivation
- ICaL: Increased current, altered inactivation
- RyR: Increased SR release
- SERCA: Increased uptake

The model tracks:
- CaMKb: Calmodulin-bound (activated) CaMKII
- CaMKa: Total active CaMKII (bound + trapped)
- CaMKt: Trapped (autophosphorylated) CaMKII fraction
"""

import torch
from typing import Tuple


def compute_CaMKb(CaMKt: torch.Tensor, cass: torch.Tensor,
                  CaMKo: float = 0.05, KmCaM: float = 0.0015) -> torch.Tensor:
    """
    Compute calmodulin-bound CaMKII fraction.

    CaMKII is activated when Ca2+/calmodulin binds to it.
    Activation depends on subspace Ca2+ concentration.

    Parameters
    ----------
    CaMKt : Trapped (autophosphorylated) CaMKII fraction
    cass : Subspace Ca2+ concentration (mM)
    CaMKo : Total CaMKII concentration (fraction of max)
    KmCaM : Ca2+/CaM binding affinity (mM)

    Returns
    -------
    CaMKb : Calmodulin-bound CaMKII fraction
    """
    # Available CaMKII (not trapped)
    CaMKb = CaMKo * (1.0 - CaMKt) / (1.0 + KmCaM / cass)
    return CaMKb


def compute_CaMKa(CaMKt: torch.Tensor, cass: torch.Tensor,
                  CaMKo: float = 0.05, KmCaM: float = 0.0015) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute total active CaMKII.

    Active CaMKII = bound + trapped fractions.

    Parameters
    ----------
    CaMKt : Trapped CaMKII fraction
    cass : Subspace Ca2+ (mM)
    CaMKo : Total CaMKII (fraction)
    KmCaM : Ca2+/CaM affinity (mM)

    Returns
    -------
    CaMKb : Calmodulin-bound fraction
    CaMKa : Total active fraction (bound + trapped)
    """
    CaMKb = compute_CaMKb(CaMKt, cass, CaMKo, KmCaM)
    CaMKa = CaMKb + CaMKt
    return CaMKb, CaMKa


def fCaMKp(CaMKa: torch.Tensor, KmCaMK: float = 0.15) -> torch.Tensor:
    """
    Compute phosphorylation factor for CaMKII-dependent effects.

    This factor (0 to 1) determines the blend between non-phosphorylated
    and phosphorylated channel/transporter kinetics.

    Parameters
    ----------
    CaMKa : Total active CaMKII fraction
    KmCaMK : Half-activation constant

    Returns
    -------
    fCaMKp : Phosphorylation factor (0 = non-phosphorylated, 1 = fully phosphorylated)
    """
    return 1.0 / (1.0 + KmCaMK / CaMKa)


def update_CaMKt(CaMKt: torch.Tensor, CaMKb: torch.Tensor, CaMKa: torch.Tensor,
                 dt: float, aCaMK: float = 0.05, bCaMK: float = 0.00068) -> torch.Tensor:
    """
    Update trapped CaMKII fraction.

    Trapping (autophosphorylation) occurs when active CaMKII phosphorylates
    itself, allowing it to remain active even after Ca2+ decreases.

    Parameters
    ----------
    CaMKt : Current trapped CaMKII fraction
    CaMKb : Calmodulin-bound CaMKII fraction
    CaMKa : Total active CaMKII fraction
    dt : Time step (ms)
    aCaMK : Trapping rate constant (/ms)
    bCaMK : Detrapping rate constant (/ms)

    Returns
    -------
    CaMKt_new : Updated trapped fraction
    """
    # Trapping rate depends on both bound (CaMKb) and total active (CaMKa)
    # Detrapping is first-order
    dCaMKt = aCaMK * CaMKb * CaMKa - bCaMK * CaMKt

    CaMKt_new = CaMKt + dt * dCaMKt

    # Ensure physical bounds
    CaMKt_new = torch.clamp(CaMKt_new, min=0.0, max=1.0)

    return CaMKt_new


# =============================================================================
# Utility Functions
# =============================================================================

def get_CaMK_steady_state(cass: torch.Tensor,
                          CaMKo: float = 0.05, KmCaM: float = 0.0015,
                          KmCaMK: float = 0.15,
                          aCaMK: float = 0.05, bCaMK: float = 0.00068
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute steady-state CaMKII activation for given Ca2+ level.

    At steady state: aCaMK * CaMKb * CaMKa = bCaMK * CaMKt

    Parameters
    ----------
    cass : Subspace Ca2+ (mM)

    Returns
    -------
    CaMKb_ss : Steady-state bound fraction
    CaMKa_ss : Steady-state active fraction
    CaMKt_ss : Steady-state trapped fraction
    """
    # This requires solving a quadratic equation
    # For simplicity, use iterative approach

    # Initial guess
    CaMKt = torch.zeros_like(cass)

    # Iterate to steady state
    for _ in range(100):
        CaMKb = CaMKo * (1.0 - CaMKt) / (1.0 + KmCaM / cass)
        CaMKa = CaMKb + CaMKt

        # At steady state: dCaMKt/dt = 0
        # aCaMK * CaMKb * CaMKa = bCaMK * CaMKt
        CaMKt_new = aCaMK * CaMKb * CaMKa / bCaMK

        # Damped update
        CaMKt = 0.9 * CaMKt + 0.1 * CaMKt_new

    CaMKb_ss = CaMKo * (1.0 - CaMKt) / (1.0 + KmCaM / cass)
    CaMKa_ss = CaMKb_ss + CaMKt
    CaMKt_ss = CaMKt

    return CaMKb_ss, CaMKa_ss, CaMKt_ss
