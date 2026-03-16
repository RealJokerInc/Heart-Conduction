"""
Voltage-Dependent Gating Kinetics for PHAS13 Model

Contains steady-state (x_inf) and time constant (tau_x) functions
for all gating variables. All functions are vectorized for batch
operation over tissue tensors.

All expressions converted from .mmt (V in Volts, t in seconds) to
our convention (V in mV, t in ms). Conversion:
  - V*1000 in .mmt -> V in ours
  - tau / 1000 in .mmt -> tau in ours (drop the /1000)

Reference:
Paci M, Hyttinen J, Aalto-Setala K, Severi S (2013).
Ann Biomed Eng 41(11):2334-2348.
"""

import torch

# Import shared utilities from TTP06
from ..ttp06.gating import safe_exp, rush_larsen


# =============================================================================
# INa (Fast Sodium Current) Gates
# =============================================================================

def INa_m_inf(V: torch.Tensor) -> torch.Tensor:
    """INa activation steady-state. Cube root form."""
    return (1.0 / (1.0 + safe_exp((-V - 34.1) / 5.9))) ** (1.0 / 3.0)


def INa_m_tau(V: torch.Tensor) -> torch.Tensor:
    """INa activation time constant (ms)."""
    alpha = 1.0 / (1.0 + safe_exp((-V - 60.0) / 5.0))
    beta = (0.1 / (1.0 + safe_exp((V + 35.0) / 5.0)) +
            0.1 / (1.0 + safe_exp((V - 50.0) / 200.0)))
    return alpha * beta


def INa_h_inf(V: torch.Tensor) -> torch.Tensor:
    """INa fast inactivation steady-state. Square root form."""
    return 1.0 / torch.sqrt(1.0 + safe_exp((V + 72.1) / 5.7))


def INa_h_tau(V: torch.Tensor) -> torch.Tensor:
    """INa fast inactivation time constant (ms). Biphasic at V=-40 mV."""
    alpha = torch.where(
        V < -40.0,
        0.057 * safe_exp(-(V + 80.0) / 6.8),
        torch.zeros_like(V)
    )
    beta = torch.where(
        V < -40.0,
        2.7 * safe_exp(0.079 * V) + 3.1e5 * safe_exp(0.3485 * V),
        0.77 / (0.13 * (1.0 + safe_exp(-(V + 10.66) / 11.1)))
    )
    # For V >= -40, alpha=0, so tau = 1/beta
    # For V < -40, tau = 1.5/(alpha+beta)
    return torch.where(
        V < -40.0,
        1.5 / (alpha + beta),
        2.542
    )


def INa_j_inf(V: torch.Tensor) -> torch.Tensor:
    """INa slow inactivation steady-state. Same as h_inf."""
    return INa_h_inf(V)


def INa_j_tau(V: torch.Tensor) -> torch.Tensor:
    """INa slow inactivation time constant (ms). Biphasic at V=-40 mV."""
    alpha = torch.where(
        V < -40.0,
        (-25428.0 * safe_exp(0.2444 * V) -
         6.948e-6 * safe_exp(-0.04391 * V)) *
        (V + 37.78) / (1.0 + safe_exp(0.311 * (V + 79.23))),
        torch.zeros_like(V)
    )
    beta = torch.where(
        V < -40.0,
        0.02424 * safe_exp(-0.01052 * V) /
        (1.0 + safe_exp(-0.1378 * (V + 40.14))),
        0.6 * safe_exp(0.057 * V) /
        (1.0 + safe_exp(-0.1 * (V + 32.0)))
    )
    return 7.0 / (alpha + beta)


# =============================================================================
# ICaL (L-type Calcium Current) Gates
# =============================================================================

def ICaL_d_inf(V: torch.Tensor) -> torch.Tensor:
    """ICaL activation steady-state."""
    return 1.0 / (1.0 + safe_exp(-(V + 9.1) / 7.0))


def ICaL_d_tau(V: torch.Tensor) -> torch.Tensor:
    """ICaL activation time constant (ms)."""
    alpha_d = 0.25 + 1.4 / (1.0 + safe_exp((-V - 35.0) / 13.0))
    beta_d = 1.4 / (1.0 + safe_exp((V + 5.0) / 5.0))
    gamma_d = 1.0 / (1.0 + safe_exp((-V + 50.0) / 20.0))
    return alpha_d * beta_d + gamma_d


def ICaL_f1_inf(V: torch.Tensor) -> torch.Tensor:
    """ICaL voltage inactivation 1 steady-state."""
    return 1.0 / (1.0 + safe_exp((V + 26.0) / 3.0))


def ICaL_f1_tau(V: torch.Tensor) -> torch.Tensor:
    """ICaL voltage inactivation 1 base time constant (ms).

    Note: The full tau includes Ca-dependent scaling (constf1)
    which is applied in model.py during step().
    """
    return (20.0 +
            1102.5 * safe_exp(-((V + 27.0) ** 2 / 15.0) ** 2) +
            200.0 / (1.0 + safe_exp((13.0 - V) / 10.0)) +
            180.0 / (1.0 + safe_exp((30.0 + V) / 10.0)))


def ICaL_f2_inf(V: torch.Tensor) -> torch.Tensor:
    """ICaL voltage inactivation 2 steady-state."""
    return 0.33 + 0.67 / (1.0 + safe_exp((V + 35.0) / 4.0))


def ICaL_f2_tau(V: torch.Tensor) -> torch.Tensor:
    """ICaL voltage inactivation 2 time constant (ms)."""
    return (600.0 * safe_exp(-(V + 25.0) ** 2 / 170.0) +
            31.0 / (1.0 + safe_exp((25.0 - V) / 10.0)) +
            16.0 / (1.0 + safe_exp((30.0 + V) / 10.0)))


def ICaL_fCa_inf(Cai: torch.Tensor) -> torch.Tensor:
    """ICaL Ca-dependent inactivation steady-state.

    Depends on Cai, not V.
    """
    alpha = 1.0 / (1.0 + (Cai / 0.0006) ** 8)
    beta = 0.1 / (1.0 + safe_exp((Cai - 0.0009) / 0.0001))
    gamma = 0.3 / (1.0 + safe_exp((Cai - 0.00075) / 0.0008))
    return (alpha + beta + gamma) / 1.3156


# fCa tau is constant: 2.0 ms (from 0.002 s)
FCAL_TAU = 2.0


# =============================================================================
# IKr (Rapid Delayed Rectifier K+ Current) Gates
# =============================================================================

def IKr_Xr1_inf(V: torch.Tensor, Cao: float = 1.8) -> torch.Tensor:
    """IKr activation steady-state. Ca-dependent V_half (constant for fixed Cao)."""
    # V_half from .mmt: V_half = 1000 * (-RTF_V/Q * ln(...) - 0.019)
    # RTF must be in Volts (0.02671), NOT mV
    import math
    L0 = 0.025
    Q = 2.3
    RTF_V = 8.314472 * 310.0 / 96485.3415  # ~0.02671 V (R in J/(mol*K))
    V_half = 1000.0 * (
        -RTF_V / Q * math.log(
            (1.0 + Cao / 2.6) ** 4 / (L0 * (1.0 + Cao / 0.58) ** 4)
        ) - 0.019
    )  # Result in mV (~-20.7 mV)
    return 1.0 / (1.0 + safe_exp((V_half - V) / 4.9))


def IKr_Xr1_tau(V: torch.Tensor) -> torch.Tensor:
    """IKr activation time constant (ms)."""
    alpha = 450.0 / (1.0 + safe_exp((-45.0 - V) / 10.0))
    beta = 6.0 / (1.0 + safe_exp((30.0 + V) / 11.5))
    return alpha * beta


def IKr_Xr2_inf(V: torch.Tensor) -> torch.Tensor:
    """IKr inactivation steady-state."""
    return 1.0 / (1.0 + safe_exp((V + 88.0) / 50.0))


def IKr_Xr2_tau(V: torch.Tensor) -> torch.Tensor:
    """IKr inactivation time constant (ms)."""
    alpha = 3.0 / (1.0 + safe_exp((-60.0 - V) / 20.0))
    beta = 1.12 / (1.0 + safe_exp((-60.0 + V) / 20.0))
    return alpha * beta


# =============================================================================
# IKs (Slow Delayed Rectifier K+ Current) Gate
# =============================================================================

def IKs_Xs_inf(V: torch.Tensor) -> torch.Tensor:
    """IKs activation steady-state."""
    return 1.0 / (1.0 + safe_exp((-V - 20.0) / 16.0))


def IKs_Xs_tau(V: torch.Tensor) -> torch.Tensor:
    """IKs activation time constant (ms)."""
    alpha = 1100.0 / torch.sqrt(1.0 + safe_exp((-10.0 - V) / 6.0))
    beta = 1.0 / (1.0 + safe_exp((-60.0 + V) / 20.0))
    return alpha * beta


# =============================================================================
# Ito (Transient Outward K+ Current) Gates
# =============================================================================

def Ito_q_inf(V: torch.Tensor) -> torch.Tensor:
    """Ito inactivation steady-state."""
    return 1.0 / (1.0 + safe_exp((V + 53.0) / 13.0))


def Ito_q_tau(V: torch.Tensor) -> torch.Tensor:
    """Ito inactivation time constant (ms)."""
    return (6.06 + 39.102 /
            (0.57 * safe_exp(-0.08 * (V + 44.0)) +
             0.065 * safe_exp(0.1 * (V + 45.93))))


def Ito_r_inf(V: torch.Tensor) -> torch.Tensor:
    """Ito activation steady-state."""
    return 1.0 / (1.0 + safe_exp(-(V - 22.3) / 18.75))


def Ito_r_tau(V: torch.Tensor) -> torch.Tensor:
    """Ito activation time constant (ms)."""
    return (2.75352 + 14.40516 /
            (1.037 * safe_exp(0.09 * (V + 30.61)) +
             0.369 * safe_exp(-0.12 * (V + 23.84))))


# =============================================================================
# If (Funny Current) Gate
# =============================================================================

def If_Xf_inf(V: torch.Tensor) -> torch.Tensor:
    """If activation steady-state."""
    return 1.0 / (1.0 + safe_exp((V + 77.85) / 5.0))


def If_Xf_tau(V: torch.Tensor) -> torch.Tensor:
    """If activation time constant (ms)."""
    return 1900.0 / (1.0 + safe_exp((V + 15.0) / 10.0))


# =============================================================================
# g_rel (RyR inactivation) — Ca-dependent, not voltage-dependent
# =============================================================================

def g_rel_inf(Cai: torch.Tensor) -> torch.Tensor:
    """RyR inactivation gate steady-state. Depends on Cai."""
    return torch.where(
        Cai <= 0.00035,
        1.0 / (1.0 + (Cai / 0.00035) ** 6),
        1.0 / (1.0 + (Cai / 0.00035) ** 16)
    )
