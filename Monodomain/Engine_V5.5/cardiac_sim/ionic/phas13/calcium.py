"""
Calcium Handling for PHAS13 Model

Two-compartment model (cytoplasm + SR, no subspace).
- SERCA pump (SR uptake)
- RyR release (d-gate triggered, NOT CaSS-dependent)
- SR leak
- Cytoplasmic and SR buffering

Rates converted from /s to /ms relative to .mmt source.

Reference:
Paci M, Hyttinen J, Aalto-Setala K, Severi S (2013).
Ann Biomed Eng 41(11):2334-2348.
"""

import torch
from typing import Tuple


# =============================================================================
# SERCA Pump (SR Ca Uptake)
# =============================================================================

def i_up(Cai: torch.Tensor, VmaxUp: float = 5.6064e-4,
         Kup: float = 0.00025) -> torch.Tensor:
    """SERCA pump rate (mM/ms)."""
    return VmaxUp / (1.0 + (Kup / Cai) ** 2)


# =============================================================================
# RyR Release (d-gate triggered)
# =============================================================================

def i_rel(CaSR: torch.Tensor, d: torch.Tensor, g_rel: torch.Tensor,
          a_rel: float = 0.016464, b_rel: float = 0.25,
          c_rel: float = 0.008232) -> torch.Tensor:
    """
    RyR release rate (mM/ms).

    Triggered by ICaL d-gate, NOT CaSS-dependent (no subspace in PHAS13).
    """
    return ((c_rel + a_rel * CaSR ** 2 / (b_rel ** 2 + CaSR ** 2)) *
            d * g_rel * 0.0411)


# =============================================================================
# SR Leak
# =============================================================================

def i_leak(CaSR: torch.Tensor, Cai: torch.Tensor,
           V_leak: float = 4.4444e-7) -> torch.Tensor:
    """SR passive leak (mM/ms)."""
    return (CaSR - Cai) * V_leak


# =============================================================================
# Calcium Buffering
# =============================================================================

def buffering_factor_cyt(Cai: torch.Tensor,
                         Buf_C: float = 0.25,
                         Kbuf_C: float = 0.001) -> torch.Tensor:
    """Cytoplasmic Ca buffering factor (dimensionless, in (0,1))."""
    return 1.0 / (1.0 + Buf_C * Kbuf_C / (Cai + Kbuf_C) ** 2)


def buffering_factor_sr(CaSR: torch.Tensor,
                        Buf_SR: float = 10.0,
                        Kbuf_SR: float = 0.3) -> torch.Tensor:
    """SR Ca buffering factor (dimensionless, in (0,1))."""
    return 1.0 / (1.0 + Buf_SR * Kbuf_SR / (CaSR + Kbuf_SR) ** 2)


# =============================================================================
# g_rel (RyR inactivation gate) update
# =============================================================================

def update_g_rel(V: torch.Tensor, g_rel: torch.Tensor,
                 Cai: torch.Tensor, dt: float,
                 tau_g: float = 2.0) -> torch.Tensor:
    """
    Update RyR inactivation gate with conditional Forward Euler.

    Freezes (const2=0) when g_inf > g AND V > -60 mV.
    """
    from .gating import g_rel_inf

    g_inf = g_rel_inf(Cai)

    # Conditional: freeze when recovering AND depolarized
    const2 = torch.where(
        (g_inf > g_rel) & (V > -60.0),
        torch.zeros_like(g_rel),
        torch.ones_like(g_rel)
    )

    # Forward Euler: dg/dt = const2 * (g_inf - g) / tau_g
    g_new = g_rel + const2 * (g_inf - g_rel) / tau_g * dt
    return torch.clamp(g_new, 0.0, 1.0)


# =============================================================================
# Concentration Updates
# =============================================================================

def update_concentrations(
    V: torch.Tensor,
    Nai: torch.Tensor, Cai: torch.Tensor, CaSR: torch.Tensor,
    g_rel_state: torch.Tensor,
    d_gate: torch.Tensor,
    INa: torch.Tensor, ICaL: torch.Tensor,
    INaCa: torch.Tensor, INaK: torch.Tensor,
    IpCa: torch.Tensor, IbNa: torch.Tensor, IbCa: torch.Tensor,
    dt: float,
    Cm: float = 9.87109e-11,
    Vc: float = 8800.0e-18,
    V_SR: float = 583.73e-18,
    F_const: float = 96485.3415,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Update ionic concentrations using Forward Euler.

    Parameters
    ----------
    All currents in A/F, dt in ms.

    Returns
    -------
    Nai_new, Cai_new, CaSR_new, g_rel_new
    """
    # Current-to-concentration conversion factor (A/F -> mM/ms)
    # From .mmt: dNai/dt = -Cm * I / (F * Vc * 1e-18) [mM/s]
    # For mM/ms: divide by 1000
    inv_VcF = Cm / (F_const * Vc) / 1000.0

    # Volume ratio
    Vc_Vsr = Vc / V_SR

    # Calcium handling fluxes (all in mM/ms)
    Iup = i_up(Cai)
    Irel = i_rel(CaSR, d_gate, g_rel_state)
    Ileak = i_leak(CaSR, Cai)

    # dNai/dt = -Cm/(F*Vc) * (INa + IbNa + 3*INaK + 3*INaCa) / 1000
    INa_total = INa + IbNa + 3.0 * INaK + 3.0 * INaCa
    dNai = -INa_total * inv_VcF

    # dCai/dt = bufc * (leak - up + rel - I_Ca_total * Cm/(2*F*Vc) / 1000)
    ICa_total = ICaL + IbCa + IpCa - 2.0 * INaCa
    dCai_unbuffered = (Ileak - Iup + Irel -
                       ICa_total * Cm / (2.0 * Vc * F_const) / 1000.0)
    dCai = dCai_unbuffered * buffering_factor_cyt(Cai)

    # dCaSR/dt = bufSR * Vc/V_SR * (up - rel - leak)
    dCaSR_unbuffered = Vc_Vsr * (Iup - Irel - Ileak)
    dCaSR = dCaSR_unbuffered * buffering_factor_sr(CaSR)

    # g_rel update (conditional Forward Euler)
    g_rel_new = update_g_rel(V, g_rel_state, Cai, dt)

    # Forward Euler update
    Nai_new = Nai + dNai * dt
    Cai_new = Cai + dCai * dt
    CaSR_new = CaSR + dCaSR * dt

    # Clamp to physical bounds
    Nai_new = torch.clamp(Nai_new, min=1.0)
    Cai_new = torch.clamp(Cai_new, min=1e-8)
    CaSR_new = torch.clamp(CaSR_new, min=1e-4)

    return Nai_new, Cai_new, CaSR_new, g_rel_new
