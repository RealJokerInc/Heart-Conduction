"""
Ion Current Calculations for PHAS13 Model

Contains all 12 ionic currents:
- INa: Fast sodium
- ICaL: L-type calcium (GHK)
- IKr: Rapid delayed rectifier potassium
- IKs: Slow delayed rectifier potassium (Ca-dependent)
- IK1: Inward rectifier potassium
- Ito: Transient outward potassium
- If: Funny current (hyperpolarization-activated)
- INaCa: Sodium-calcium exchanger
- INaK: Sodium-potassium pump
- IpCa: Sarcolemmal calcium pump
- IbNa: Background sodium
- IbCa: Background calcium

All functions use mV convention. Ohmic conductances are pre-divided by 1000
relative to .mmt values (which use V driving force).

Reference:
Paci M, Hyttinen J, Aalto-Setala K, Severi S (2013).
Ann Biomed Eng 41(11):2334-2348.
"""

import torch
from ..ttp06.gating import safe_exp


# Physical constants (same convention as TTP06)
R = 8314.472        # mJ/(mol*K), works with mV
T = 310.0           # K
F = 96485.3415      # C/mol
RTONF = R * T / F   # ~26.713 mV


# =============================================================================
# Reversal Potentials
# =============================================================================

def E_Na(Nai: torch.Tensor, Nao: float = 151.0) -> torch.Tensor:
    """Sodium reversal potential (mV)."""
    return RTONF * torch.log(Nao / Nai)

def E_K(Ki: float = 150.0, Ko: float = 5.4) -> float:
    """Potassium reversal potential (mV). Ki is fixed."""
    import math
    return RTONF * math.log(Ko / Ki)


def E_Ca(Cai: torch.Tensor, Cao: float = 1.8) -> torch.Tensor:
    """Calcium reversal potential (mV)."""
    return 0.5 * RTONF * torch.log(Cao / Cai)


def E_Ks(Ki: float = 150.0, Nai: torch.Tensor = None,
         Ko: float = 5.4, Nao: float = 151.0,
         PkNa: float = 0.03) -> torch.Tensor:
    """IKs reversal potential with Na permeability (mV)."""
    return RTONF * torch.log((Ko + PkNa * Nao) / (Ki + PkNa * Nai))


# =============================================================================
# INa (Fast Sodium Current)
# =============================================================================

def I_Na(V: torch.Tensor, m: torch.Tensor, h: torch.Tensor, j: torch.Tensor,
         Nai: torch.Tensor, g_Na: float = 3.6712302,
         Nao: float = 151.0) -> torch.Tensor:
    """Fast sodium current (A/F)."""
    ENa = E_Na(Nai, Nao)
    return g_Na * (m ** 3) * h * j * (V - ENa)


# =============================================================================
# ICaL (L-type Calcium Current) — GHK formulation
# =============================================================================

def I_CaL(V: torch.Tensor, d: torch.Tensor, f1: torch.Tensor,
          f2: torch.Tensor, fCa: torch.Tensor,
          Cai: torch.Tensor, g_CaL: float = 8.635702e-5,
          Cao: float = 1.8) -> torch.Tensor:
    """
    L-type calcium current using GHK driving force (A/F).

    Uses Cai (not CaSS — PHAS13 model has no subspace compartment).
    """
    # GHK argument: 2*V*F/(R*T) — dimensionless in mV convention
    zfrt = 2.0 * V * F / (R * T)

    # Normal GHK case
    ghk_normal = (4.0 * V * F * F / (R * T) *
                  (Cai * safe_exp(zfrt) - 0.341 * Cao) /
                  (safe_exp(zfrt) - 1.0))

    # L'Hopital limit at V=0
    ghk_limit = 2.0 * F * (Cai - 0.341 * Cao)

    ghk = torch.where(torch.abs(V) > 0.01, ghk_normal, ghk_limit)

    return g_CaL * d * f1 * f2 * fCa * ghk


# =============================================================================
# IKr (Rapid Delayed Rectifier K+ Current)
# =============================================================================

def I_Kr(V: torch.Tensor, Xr1: torch.Tensor, Xr2: torch.Tensor,
         g_Kr: float = 0.0298667,
         Ki: float = 150.0, Ko: float = 5.4) -> torch.Tensor:
    """Rapid delayed rectifier potassium current (A/F)."""
    EK = E_K(Ki, Ko)
    return g_Kr * Xr1 * Xr2 * torch.sqrt(torch.tensor(Ko / 5.4)) * (V - EK)


# =============================================================================
# IKs (Slow Delayed Rectifier K+ Current) — Ca-dependent
# =============================================================================

def I_Ks(V: torch.Tensor, Xs: torch.Tensor,
         Nai: torch.Tensor, Cai: torch.Tensor,
         g_Ks: float = 0.002041,
         Ki: float = 150.0, Ko: float = 5.4,
         Nao: float = 151.0, PkNa: float = 0.03) -> torch.Tensor:
    """Slow delayed rectifier potassium current with Ca-dependent scaling (A/F)."""
    EKs = E_Ks(Ki, Nai, Ko, Nao, PkNa)
    k = 1.0 + 0.6 / (1.0 + (3.8e-5 / Cai) ** 1.4)
    return g_Ks * (Xs ** 2) * (V - EKs) * k


# =============================================================================
# IK1 (Inward Rectifier K+ Current)
# =============================================================================

def I_K1(V: torch.Tensor, g_K1: float = 0.0281492,
         Ki: float = 150.0, Ko: float = 5.4) -> torch.Tensor:
    """Inward rectifier potassium current (A/F)."""
    EK = E_K(Ki, Ko)
    VmEK = V - EK

    alpha = 3.91 / (1.0 + safe_exp(0.5942 * (VmEK - 200.0)))
    beta = ((-1.509 * safe_exp(0.0002 * (VmEK + 100.0)) +
             safe_exp(0.5886 * (VmEK - 10.0))) /
            (1.0 + safe_exp(0.4547 * VmEK)))

    inf = alpha / (alpha + beta)
    return g_K1 * inf * torch.sqrt(torch.tensor(Ko / 5.4)) * (V - EK)


# =============================================================================
# Ito (Transient Outward K+ Current)
# =============================================================================

def I_to(V: torch.Tensor, q: torch.Tensor, r_gate: torch.Tensor,
         g_to: float = 0.0299038,
         Ki: float = 150.0, Ko: float = 5.4) -> torch.Tensor:
    """Transient outward potassium current (A/F)."""
    EK = E_K(Ki, Ko)
    return g_to * q * r_gate * (V - EK)


# =============================================================================
# If (Funny Current)
# =============================================================================

def I_f(V: torch.Tensor, Xf: torch.Tensor,
        g_f: float = 0.03010312, E_f: float = -17.0) -> torch.Tensor:
    """Funny (hyperpolarization-activated) current (A/F)."""
    return g_f * Xf * (V - E_f)


# =============================================================================
# INaCa (Sodium-Calcium Exchanger)
# =============================================================================

def I_NaCa(V: torch.Tensor, Nai: torch.Tensor, Cai: torch.Tensor,
           kNaCa: float = 4900.0, Cao: float = 1.8, Nao: float = 151.0,
           KmNai: float = 87.5, KmCa: float = 1.38,
           Ksat: float = 0.1, alpha: float = 2.8571432,
           gamma: float = 0.35) -> torch.Tensor:
    """Sodium-calcium exchanger current (A/F)."""
    vfrt = V * F / (R * T)  # dimensionless

    return kNaCa * (
        safe_exp(gamma * vfrt) * (Nai ** 3) * Cao -
        safe_exp((gamma - 1.0) * vfrt) * (Nao ** 3) * Cai * alpha
    ) / (
        ((KmNai ** 3) + (Nao ** 3)) *
        (KmCa + Cao) *
        (1.0 + Ksat * safe_exp((gamma - 1.0) * vfrt))
    )


# =============================================================================
# INaK (Sodium-Potassium Pump)
# =============================================================================

def I_NaK(V: torch.Tensor, Nai: torch.Tensor,
          PNaK: float = 1.841424,
          Ki: float = 150.0, Ko: float = 5.4,
          Km_K: float = 1.0, Km_Na: float = 40.0) -> torch.Tensor:
    """Sodium-potassium pump current (A/F)."""
    vfrt = V * F / (R * T)

    return (PNaK * Ko / (Ko + Km_K) * Nai / (Nai + Km_Na) /
            (1.0 + 0.1245 * safe_exp(-0.1 * vfrt) +
             0.0353 * safe_exp(-vfrt)))


# =============================================================================
# Background and Pump Currents
# =============================================================================

def I_pCa(Cai: torch.Tensor, g_pCa: float = 0.4125,
          KpCa: float = 0.0005) -> torch.Tensor:
    """Sarcolemmal calcium pump current (A/F)."""
    return g_pCa * Cai / (KpCa + Cai)


def I_bNa(V: torch.Tensor, Nai: torch.Tensor,
          g_bNa: float = 0.0009, Nao: float = 151.0) -> torch.Tensor:
    """Background sodium current (A/F)."""
    ENa = E_Na(Nai, Nao)
    return g_bNa * (V - ENa)


def I_bCa(V: torch.Tensor, Cai: torch.Tensor,
          g_bCa: float = 0.00069264, Cao: float = 1.8) -> torch.Tensor:
    """Background calcium current (A/F)."""
    ECa = E_Ca(Cai, Cao)
    return g_bCa * (V - ECa)
