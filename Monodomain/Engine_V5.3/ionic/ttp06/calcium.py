"""
Calcium Handling for TTP06 Model

Contains:
- SERCA pump (SR uptake)
- RyR release (SR Ca release)
- SR leak
- Buffering calculations
- Concentration update functions

Reference: ten Tusscher-Panfilov 2006, Am J Physiol Heart Circ Physiol.
"""

import torch
from typing import Tuple


# =============================================================================
# SERCA Pump (SR Ca Uptake)
# =============================================================================

def I_up(Cai: torch.Tensor, Vmax_up: float = 0.006375,
         Kup: float = 0.00025) -> torch.Tensor:
    """
    SERCA pump rate (mM/ms).

    Parameters
    ----------
    Cai : Intracellular Ca2+ (mM)
    Vmax_up : Maximum uptake rate (mM/ms)
    Kup : Half-saturation (mM)

    Returns
    -------
    I_up : Uptake rate (mM/ms)
    """
    return Vmax_up / (1.0 + (Kup / Cai) ** 2)


# =============================================================================
# RyR Release (SR Ca Release)
# =============================================================================

def I_rel(CaSR: torch.Tensor, CaSS: torch.Tensor, RR: torch.Tensor,
          Vrel: float = 0.102,
          k1_prime: float = 0.15, k2_prime: float = 0.045,
          k3: float = 0.060,
          EC: float = 1.5, maxsr: float = 2.5, minsr: float = 1.0
          ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Ryanodine receptor release and RR state dynamics.

    Parameters
    ----------
    CaSR : SR Ca2+ (mM)
    CaSS : Subspace Ca2+ (mM)
    RR : Recovery fraction (dimensionless)

    Returns
    -------
    I_rel : Release rate (mM/ms)
    dRR_dt : Time derivative of RR
    """
    # SR-dependent scaling
    kcasr = maxsr - (maxsr - minsr) / (1.0 + (EC / CaSR) ** 2)
    k1 = k1_prime / kcasr
    k2 = k2_prime * kcasr

    # Open probability - uses quasi-steady-state approximation
    # OO = k1 * CaSS^2 * RR / (k3 + k1 * CaSS^2)
    OO = k1 * (CaSS ** 2) * RR / (k3 + k1 * (CaSS ** 2))

    # Release current
    Irel = Vrel * OO * (CaSR - CaSS)

    # RR dynamics (recovery from inactivation)
    # dRR/dt = k4 * (1 - RR) - k2 * CaSS * RR
    k4 = 0.005  # Recovery rate constant
    dRR = k4 * (1.0 - RR) - k2 * CaSS * RR

    return Irel, dRR


# =============================================================================
# SR Leak
# =============================================================================

def I_leak(CaSR: torch.Tensor, Cai: torch.Tensor,
           Vleak: float = 0.00036) -> torch.Tensor:
    """
    SR passive leak (mM/ms).

    Parameters
    ----------
    CaSR : SR Ca2+ (mM)
    Cai : Intracellular Ca2+ (mM)
    Vleak : Leak rate constant (1/ms)

    Returns
    -------
    I_leak : Leak rate (mM/ms)
    """
    return Vleak * (CaSR - Cai)


# =============================================================================
# Subspace Transfer
# =============================================================================

def I_xfer(CaSS: torch.Tensor, Cai: torch.Tensor,
           Vxfer: float = 0.0038) -> torch.Tensor:
    """
    Ca transfer from subspace to cytoplasm (mM/ms).

    Parameters
    ----------
    CaSS : Subspace Ca2+ (mM)
    Cai : Intracellular Ca2+ (mM)
    Vxfer : Transfer rate constant (1/ms)

    Returns
    -------
    I_xfer : Transfer rate (mM/ms)
    """
    return Vxfer * (CaSS - Cai)


# =============================================================================
# Calcium Buffering
# =============================================================================

def buffering_factor_cyt(Cai: torch.Tensor,
                         Bufc: float = 0.2, Kbufc: float = 0.001) -> torch.Tensor:
    """
    Cytoplasmic calcium buffering factor.

    Returns multiplier for dCai/dt to account for buffering.
    """
    return 1.0 / (1.0 + Bufc * Kbufc / ((Cai + Kbufc) ** 2))


def buffering_factor_sr(CaSR: torch.Tensor,
                        Bufsr: float = 10.0, Kbufsr: float = 0.3) -> torch.Tensor:
    """
    SR calcium buffering factor.

    Returns multiplier for dCaSR/dt to account for buffering.
    """
    return 1.0 / (1.0 + Bufsr * Kbufsr / ((CaSR + Kbufsr) ** 2))


def buffering_factor_ss(CaSS: torch.Tensor,
                        Bufss: float = 0.4, Kbufss: float = 0.00025) -> torch.Tensor:
    """
    Subspace calcium buffering factor.

    Returns multiplier for dCaSS/dt to account for buffering.
    """
    return 1.0 / (1.0 + Bufss * Kbufss / ((CaSS + Kbufss) ** 2))


# =============================================================================
# Concentration Updates
# =============================================================================

def update_concentrations(
    V: torch.Tensor,
    Ki: torch.Tensor, Nai: torch.Tensor,
    Cai: torch.Tensor, CaSR: torch.Tensor, CaSS: torch.Tensor,
    RR: torch.Tensor,
    INa: torch.Tensor, ICaL: torch.Tensor,
    Ito: torch.Tensor, IKr: torch.Tensor, IKs: torch.Tensor, IK1: torch.Tensor,
    INaCa: torch.Tensor, INaK: torch.Tensor,
    IpCa: torch.Tensor, IpK: torch.Tensor, IbNa: torch.Tensor, IbCa: torch.Tensor,
    dt: float,
    Cm: float = 0.185,
    Vc: float = 16.404,
    Vsr: float = 1.094,
    Vss: float = 0.05468
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Update ionic concentrations using Forward Euler.

    Parameters
    ----------
    V : Membrane potential (mV)
    Ki, Nai : Intracellular K+, Na+ (mM)
    Cai, CaSR, CaSS : Calcium concentrations (mM)
    RR : RyR recovery state
    I* : All ionic currents (pA/pF)
    dt : Time step (ms)
    Cm : Membrane capacitance (uF)
    Vc, Vsr, Vss : Compartment volumes (pL)

    Returns
    -------
    Ki_new, Nai_new, Cai_new, CaSR_new, CaSS_new : Updated concentrations (mM)
    RR_new : Updated RyR state
    """
    # Physical constants
    F = 96485.3415  # C/mol

    # Volume ratios
    Vsr_Vc = Vsr / Vc
    Vss_Vc = Vss / Vc

    # Capacitive surface area normalization
    # Factor converts pA/pF to flux: Cm * 1e-6 / (Vc * 1e-9 * F) = Cm / (Vc * F * 1e3)
    # Simplified: Cm / (Vc * F) when currents are in pA/pF
    inv_VcF = Cm / (Vc * F) * 1000.0  # mM/(ms * pA/pF)
    inv_VssF = Cm / (Vss * F) * 1000.0

    # Calculate Ca handling fluxes
    Iup = I_up(Cai)
    Irel, dRR = I_rel(CaSR, CaSS, RR)
    Ileak = I_leak(CaSR, Cai)
    Ixfer = I_xfer(CaSS, Cai)

    # dKi/dt = -(IK1 + Ito + IKr + IKs + IpK - 2*INaK) * inv_VcF
    IK_total = IK1 + Ito + IKr + IKs + IpK - 2.0 * INaK
    dKi = -IK_total * inv_VcF

    # dNai/dt = -(INa + IbNa + 3*INaK + 3*INaCa) * inv_VcF
    INa_total = INa + IbNa + 3.0 * INaK + 3.0 * INaCa
    dNai = -INa_total * inv_VcF

    # dCai/dt = buffering * (Ileak - Iup + Ixfer - (IbCa + IpCa - 2*INaCa) * inv_VcF / 2)
    ICa_sarcolemma = IbCa + IpCa - 2.0 * INaCa  # Net Ca flux through membrane
    dCai_unbuffered = (Ileak - Iup) * Vsr_Vc + Ixfer - ICa_sarcolemma * inv_VcF / 2.0
    dCai = dCai_unbuffered * buffering_factor_cyt(Cai)

    # dCaSR/dt = buffering * (Iup - Irel - Ileak)
    dCaSR_unbuffered = Iup - Irel * Vss / Vsr - Ileak
    dCaSR = dCaSR_unbuffered * buffering_factor_sr(CaSR)

    # dCaSS/dt = buffering * (Irel - Ixfer * Vc/Vss - ICaL * inv_VssF / 2)
    dCaSS_unbuffered = Irel - Ixfer * Vc / Vss - ICaL * inv_VssF / 2.0
    dCaSS = dCaSS_unbuffered * buffering_factor_ss(CaSS)

    # Forward Euler update
    Ki_new = Ki + dKi * dt
    Nai_new = Nai + dNai * dt
    Cai_new = Cai + dCai * dt
    CaSR_new = CaSR + dCaSR * dt
    CaSS_new = CaSS + dCaSS * dt
    RR_new = RR + dRR * dt

    # Ensure concentrations stay positive
    Ki_new = torch.clamp(Ki_new, min=1.0)
    Nai_new = torch.clamp(Nai_new, min=1.0)
    Cai_new = torch.clamp(Cai_new, min=1e-8)
    CaSR_new = torch.clamp(CaSR_new, min=1e-4)
    CaSS_new = torch.clamp(CaSS_new, min=1e-8)
    RR_new = torch.clamp(RR_new, min=0.0, max=1.0)

    return Ki_new, Nai_new, Cai_new, CaSR_new, CaSS_new, RR_new
