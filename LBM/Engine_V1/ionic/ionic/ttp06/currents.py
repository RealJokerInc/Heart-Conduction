"""
Ion Current Calculations for TTP06 Model

Contains all 12 ionic currents:
- INa: Fast sodium
- Ito: Transient outward potassium
- ICaL: L-type calcium
- IKr: Rapid delayed rectifier potassium
- IKs: Slow delayed rectifier potassium
- IK1: Inward rectifier potassium
- INaCa: Sodium-calcium exchanger
- INaK: Sodium-potassium pump
- IpCa: Sarcolemmal calcium pump
- IpK: Plateau potassium
- INab: Background sodium
- ICab: Background calcium

All functions are vectorized for batch operation over tissue tensors.
"""

import torch
from typing import Tuple
from ionic.ttp06.gating import safe_exp


# =============================================================================
# Physical Constants and Reversal Potentials
# =============================================================================

R = 8314.472    # Gas constant (J/(mol·K))
T = 310.0       # Temperature (K)
F = 96485.3415  # Faraday constant (C/mol)
RTONF = R * T / F  # ~26.71 mV


def E_K(Ki: torch.Tensor, Ko: float = 5.4) -> torch.Tensor:
    """Potassium reversal potential (mV)."""
    return RTONF * torch.log(Ko / Ki)


def E_Na(Nai: torch.Tensor, Nao: float = 140.0) -> torch.Tensor:
    """Sodium reversal potential (mV)."""
    return RTONF * torch.log(Nao / Nai)


def E_Ca(Cai: torch.Tensor, Cao: float = 2.0) -> torch.Tensor:
    """Calcium reversal potential (mV)."""
    return 0.5 * RTONF * torch.log(Cao / Cai)


def E_Ks(Ki: torch.Tensor, Nai: torch.Tensor,
         Ko: float = 5.4, Nao: float = 140.0, PRNaK: float = 0.03) -> torch.Tensor:
    """IKs reversal potential with Na permeability."""
    return RTONF * torch.log((Ko + PRNaK * Nao) / (Ki + PRNaK * Nai))


# =============================================================================
# INa (Fast Sodium Current)
# =============================================================================

def I_Na(V: torch.Tensor, m: torch.Tensor, h: torch.Tensor, j: torch.Tensor,
         Nai: torch.Tensor, GNa: float = 14.838, Nao: float = 140.0) -> torch.Tensor:
    """
    Fast sodium current.

    Parameters
    ----------
    V : Membrane potential (mV)
    m, h, j : Gating variables
    Nai : Intracellular Na+ (mM)
    GNa : Maximum conductance (nS/pF)

    Returns
    -------
    INa : Current density (pA/pF)
    """
    ENa = E_Na(Nai, Nao)
    return GNa * (m ** 3) * h * j * (V - ENa)


# =============================================================================
# Ito (Transient Outward Potassium Current)
# =============================================================================

def I_to(V: torch.Tensor, r: torch.Tensor, s: torch.Tensor,
         Ki: torch.Tensor, Gto: float = 0.294, Ko: float = 5.4) -> torch.Tensor:
    """
    Transient outward potassium current.

    Parameters
    ----------
    V : Membrane potential (mV)
    r, s : Gating variables
    Ki : Intracellular K+ (mM)
    Gto : Maximum conductance (nS/pF)

    Returns
    -------
    Ito : Current density (pA/pF)
    """
    EK = E_K(Ki, Ko)
    return Gto * r * s * (V - EK)


# =============================================================================
# ICaL (L-type Calcium Current)
# =============================================================================

def I_CaL(V: torch.Tensor, d: torch.Tensor, f: torch.Tensor,
          f2: torch.Tensor, fCass: torch.Tensor,
          CaSS: torch.Tensor, PCa: float = 3.98e-5,
          Cao: float = 2.0) -> torch.Tensor:
    """
    L-type calcium current (GHK formulation).

    Parameters
    ----------
    V : Membrane potential (mV)
    d, f, f2, fCass : Gating variables
    CaSS : Subspace Ca2+ (mM)
    PCa : Permeability (cm/s)

    Returns
    -------
    ICaL : Current density (pA/pF)
    """
    # Shifted voltage for ICaL (15 mV shift)
    Veff = V - 15.0

    # GHK driving force argument
    zfrt = 2.0 * Veff * F / (R * T)

    # Precompute for L'Hopital limit case (V = 15 mV)
    # At V = 15 mV, zfrt = 0, and the limit of (x*exp(x) - C)/(exp(x) - 1) as x->0
    # is evaluated using L'Hopital's rule
    import math
    exp_30F_RT = math.exp(2.0 * 15.0 * F / (R * T))  # constant ~3.07

    # Avoid division by zero near V=15
    ICaL = torch.where(
        torch.abs(Veff) < 0.01,
        # Limit case (L'Hopital)
        d * f * f2 * fCass * PCa * 4.0 * F * F / (R * T) *
        (0.25 * CaSS - Cao / exp_30F_RT),
        # Normal case
        d * f * f2 * fCass * PCa * 4.0 * Veff * F * F / (R * T) *
        (0.25 * CaSS * safe_exp(zfrt) - Cao) /
        (safe_exp(zfrt) - 1.0)
    )

    return ICaL


# =============================================================================
# IKr (Rapid Delayed Rectifier Potassium Current)
# =============================================================================

def I_Kr(V: torch.Tensor, Xr1: torch.Tensor, Xr2: torch.Tensor,
         Ki: torch.Tensor, GKr: float = 0.153, Ko: float = 5.4) -> torch.Tensor:
    """
    Rapid delayed rectifier potassium current.

    Parameters
    ----------
    V : Membrane potential (mV)
    Xr1, Xr2 : Gating variables
    Ki : Intracellular K+ (mM)
    GKr : Maximum conductance (nS/pF)

    Returns
    -------
    IKr : Current density (pA/pF)
    """
    EK = E_K(Ki, Ko)
    return GKr * (Ko / 5.4) ** 0.5 * Xr1 * Xr2 * (V - EK)


# =============================================================================
# IKs (Slow Delayed Rectifier Potassium Current)
# =============================================================================

def I_Ks(V: torch.Tensor, Xs: torch.Tensor,
         Ki: torch.Tensor, Nai: torch.Tensor,
         GKs: float = 0.392, Ko: float = 5.4, Nao: float = 140.0) -> torch.Tensor:
    """
    Slow delayed rectifier potassium current.

    Parameters
    ----------
    V : Membrane potential (mV)
    Xs : Gating variable
    Ki : Intracellular K+ (mM)
    Nai : Intracellular Na+ (mM)
    GKs : Maximum conductance (nS/pF)

    Returns
    -------
    IKs : Current density (pA/pF)
    """
    EKs = E_Ks(Ki, Nai, Ko, Nao)
    return GKs * (Xs ** 2) * (V - EKs)


# =============================================================================
# IK1 (Inward Rectifier Potassium Current)
# =============================================================================

def I_K1(V: torch.Tensor, Ki: torch.Tensor,
         GK1: float = 5.405, Ko: float = 5.4) -> torch.Tensor:
    """
    Inward rectifier potassium current.

    Parameters
    ----------
    V : Membrane potential (mV)
    Ki : Intracellular K+ (mM)
    GK1 : Maximum conductance (nS/pF)

    Returns
    -------
    IK1 : Current density (pA/pF)
    """
    EK = E_K(Ki, Ko)

    # Rectification factor
    alpha = 0.1 / (1.0 + safe_exp(0.06 * (V - EK - 200.0)))
    beta = (3.0 * safe_exp(0.0002 * (V - EK + 100.0)) +
            safe_exp(0.1 * (V - EK - 10.0))) / \
           (1.0 + safe_exp(-0.5 * (V - EK)))
    xK1 = alpha / (alpha + beta)

    return GK1 * (Ko / 5.4) ** 0.5 * xK1 * (V - EK)


# =============================================================================
# INaCa (Sodium-Calcium Exchanger)
# =============================================================================

def I_NaCa(V: torch.Tensor, Nai: torch.Tensor, Cai: torch.Tensor,
           KNaCa: float = 1000.0, Cao: float = 2.0, Nao: float = 140.0,
           KmNai: float = 87.5, KmCa: float = 1.38,
           ksat: float = 0.1, alpha: float = 2.5, gamma: float = 0.35) -> torch.Tensor:
    """
    Sodium-calcium exchanger current.

    Parameters
    ----------
    V : Membrane potential (mV)
    Nai : Intracellular Na+ (mM)
    Cai : Intracellular Ca2+ (mM)
    KNaCa : Maximum exchange rate (pA/pF)

    Returns
    -------
    INaCa : Current density (pA/pF)
    """
    vfrt = V * F / (R * T)

    INaCa = KNaCa * (
        safe_exp(gamma * vfrt) * (Nai ** 3) * Cao -
        safe_exp((gamma - 1.0) * vfrt) * (Nao ** 3) * Cai * alpha
    ) / (
        ((KmNai ** 3) + (Nao ** 3)) *
        (KmCa + Cao) *
        (1.0 + ksat * safe_exp((gamma - 1.0) * vfrt))
    )

    return INaCa


# =============================================================================
# INaK (Sodium-Potassium Pump)
# =============================================================================

def I_NaK(V: torch.Tensor, Nai: torch.Tensor, Ki: torch.Tensor,
          PNaK: float = 2.724, Ko: float = 5.4, Nao: float = 140.0,
          KmK: float = 1.0, KmNa: float = 40.0) -> torch.Tensor:
    """
    Sodium-potassium pump current.

    Parameters
    ----------
    V : Membrane potential (mV)
    Nai : Intracellular Na+ (mM)
    Ki : Intracellular K+ (mM)
    PNaK : Maximum pump rate (pA/pF)

    Returns
    -------
    INaK : Current density (pA/pF)
    """
    vfrt = V * F / (R * T)

    INaK = PNaK * Ko / (Ko + KmK) * Nai / (Nai + KmNa) / \
           (1.0 + 0.1245 * safe_exp(-0.1 * vfrt) +
            0.0353 * safe_exp(-vfrt))

    return INaK


# =============================================================================
# Background and Pump Currents
# =============================================================================

def I_pCa(Cai: torch.Tensor, GpCa: float = 0.1238,
          KpCa: float = 0.0005) -> torch.Tensor:
    """Sarcolemmal calcium pump current (pA/pF)."""
    return GpCa * Cai / (KpCa + Cai)


def I_pK(V: torch.Tensor, Ki: torch.Tensor,
         GpK: float = 0.0146, Ko: float = 5.4) -> torch.Tensor:
    """Plateau potassium current (pA/pF)."""
    EK = E_K(Ki, Ko)
    return GpK * (V - EK) / (1.0 + safe_exp((25.0 - V) / 5.98))


def I_bNa(V: torch.Tensor, Nai: torch.Tensor,
          GbNa: float = 0.00029, Nao: float = 140.0) -> torch.Tensor:
    """Background sodium current (pA/pF)."""
    ENa = E_Na(Nai, Nao)
    return GbNa * (V - ENa)


def I_bCa(V: torch.Tensor, Cai: torch.Tensor,
          GbCa: float = 0.000592, Cao: float = 2.0) -> torch.Tensor:
    """Background calcium current (pA/pF)."""
    ECa = E_Ca(Cai, Cao)
    return GbCa * (V - ECa)
