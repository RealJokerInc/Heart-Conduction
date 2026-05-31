"""
Voltage-Dependent Gating Kinetics for TTP06 Model

Contains steady-state (x_inf) and time constant (tau_x) functions
for all gating variables. All functions are vectorized for batch
operation over tissue tensors.

Reference: ten Tusscher-Panfilov 2006, Am J Physiol Heart Circ Physiol.
"""

import torch


def safe_exp(x: torch.Tensor, limit: float = 80.0) -> torch.Tensor:
    """Clamped exponential to prevent overflow."""
    return torch.exp(torch.clamp(x, -limit, limit))


def rush_larsen(x: torch.Tensor, x_inf: torch.Tensor,
                tau: torch.Tensor, dt: float) -> torch.Tensor:
    """Rush-Larsen exponential integration."""
    return x_inf - (x_inf - x) * torch.exp(-dt / tau)


# =============================================================================
# INa (Fast Sodium Current) Gates
# =============================================================================

def INa_m_inf(V: torch.Tensor) -> torch.Tensor:
    """INa activation steady-state."""
    return 1.0 / (1.0 + safe_exp((-56.86 - V) / 9.03)) ** 2


def INa_m_tau(V: torch.Tensor) -> torch.Tensor:
    """INa activation time constant (ms)."""
    alpha = 1.0 / (1.0 + safe_exp((-60.0 - V) / 5.0))
    beta = 0.1 / (1.0 + safe_exp((V + 35.0) / 5.0)) + \
           0.1 / (1.0 + safe_exp((V - 50.0) / 200.0))
    return alpha * beta


def INa_h_inf(V: torch.Tensor) -> torch.Tensor:
    """INa fast inactivation steady-state."""
    return 1.0 / (1.0 + safe_exp((V + 71.55) / 7.43)) ** 2


def INa_h_tau(V: torch.Tensor) -> torch.Tensor:
    """INa fast inactivation time constant (ms)."""
    # Use conditional for V < -40 mV
    alpha_h = torch.where(
        V < -40.0,
        0.057 * safe_exp(-(V + 80.0) / 6.8),
        torch.zeros_like(V)
    )
    beta_h = torch.where(
        V < -40.0,
        2.7 * safe_exp(0.079 * V) + 310000.0 * safe_exp(0.3485 * V),
        0.77 / (0.13 * (1.0 + safe_exp(-(V + 10.66) / 11.1)))
    )
    return 1.0 / (alpha_h + beta_h)


def INa_j_inf(V: torch.Tensor) -> torch.Tensor:
    """INa slow inactivation steady-state."""
    return 1.0 / (1.0 + safe_exp((V + 71.55) / 7.43)) ** 2


def INa_j_tau(V: torch.Tensor) -> torch.Tensor:
    """INa slow inactivation time constant (ms)."""
    alpha_j = torch.where(
        V < -40.0,
        (-25428.0 * safe_exp(0.2444 * V) - 6.948e-6 * safe_exp(-0.04391 * V)) *
        (V + 37.78) / (1.0 + safe_exp(0.311 * (V + 79.23))),
        torch.zeros_like(V)
    )
    beta_j = torch.where(
        V < -40.0,
        0.02424 * safe_exp(-0.01052 * V) / (1.0 + safe_exp(-0.1378 * (V + 40.14))),
        0.6 * safe_exp(0.057 * V) / (1.0 + safe_exp(-0.1 * (V + 32.0)))
    )
    return 1.0 / (alpha_j + beta_j)


# =============================================================================
# Ito (Transient Outward K+ Current) Gates
# =============================================================================

def Ito_r_inf(V: torch.Tensor) -> torch.Tensor:
    """Ito activation steady-state."""
    return 1.0 / (1.0 + safe_exp((20.0 - V) / 6.0))


def Ito_r_tau(V: torch.Tensor) -> torch.Tensor:
    """Ito activation time constant (ms)."""
    return 9.5 * safe_exp(-((V + 40.0) ** 2) / 1800.0) + 0.8


def Ito_s_inf_endo(V: torch.Tensor) -> torch.Tensor:
    """Ito inactivation steady-state (ENDO)."""
    return 1.0 / (1.0 + safe_exp((V + 28.0) / 5.0))


def Ito_s_inf_epi(V: torch.Tensor) -> torch.Tensor:
    """Ito inactivation steady-state (EPI/M-cell)."""
    return 1.0 / (1.0 + safe_exp((V + 20.0) / 5.0))


def Ito_s_tau_endo(V: torch.Tensor) -> torch.Tensor:
    """Ito inactivation time constant (ENDO) (ms)."""
    return 1000.0 * safe_exp(-((V + 67.0) ** 2) / 1000.0) + 8.0


def Ito_s_tau_epi(V: torch.Tensor) -> torch.Tensor:
    """Ito inactivation time constant (EPI/M-cell) (ms)."""
    return 85.0 * safe_exp(-((V + 45.0) ** 2) / 320.0) + \
           5.0 / (1.0 + safe_exp((V - 20.0) / 5.0)) + 3.0


# =============================================================================
# ICaL (L-type Calcium Current) Gates
# =============================================================================

def ICaL_d_inf(V: torch.Tensor) -> torch.Tensor:
    """ICaL activation steady-state."""
    return 1.0 / (1.0 + safe_exp((-8.0 - V) / 7.5))


def ICaL_d_tau(V: torch.Tensor) -> torch.Tensor:
    """ICaL activation time constant (ms)."""
    alpha_d = 1.4 / (1.0 + safe_exp((-35.0 - V) / 13.0)) + 0.25
    beta_d = 1.4 / (1.0 + safe_exp((V + 5.0) / 5.0))
    gamma_d = 1.0 / (1.0 + safe_exp((50.0 - V) / 20.0))
    return alpha_d * beta_d + gamma_d


def ICaL_f_inf(V: torch.Tensor) -> torch.Tensor:
    """ICaL voltage inactivation steady-state."""
    return 1.0 / (1.0 + safe_exp((V + 20.0) / 7.0))


def ICaL_f_tau(V: torch.Tensor) -> torch.Tensor:
    """ICaL voltage inactivation time constant (ms)."""
    return 1102.5 * safe_exp(-((V + 27.0) ** 2) / 225.0) + \
           200.0 / (1.0 + safe_exp((13.0 - V) / 10.0)) + \
           180.0 / (1.0 + safe_exp((V + 30.0) / 10.0)) + 20.0


def ICaL_f2_inf(V: torch.Tensor) -> torch.Tensor:
    """ICaL voltage inactivation 2 steady-state."""
    return 0.67 / (1.0 + safe_exp((V + 35.0) / 7.0)) + 0.33


def ICaL_f2_tau(V: torch.Tensor) -> torch.Tensor:
    """ICaL voltage inactivation 2 time constant (ms)."""
    return 562.0 * safe_exp(-((V + 27.0) ** 2) / 240.0) + \
           31.0 / (1.0 + safe_exp((25.0 - V) / 10.0)) + \
           80.0 / (1.0 + safe_exp((V + 30.0) / 10.0))


def ICaL_fCass_inf(CaSS: torch.Tensor) -> torch.Tensor:
    """ICaL Ca-dependent inactivation steady-state."""
    return 0.6 / (1.0 + (CaSS / 0.00005) ** 2) + 0.4


def ICaL_fCass_tau(CaSS: torch.Tensor) -> torch.Tensor:
    """ICaL Ca-dependent inactivation time constant (ms)."""
    return 80.0 / (1.0 + (CaSS / 0.00005) ** 2) + 2.0


# =============================================================================
# IKr (Rapid Delayed Rectifier K+ Current) Gates
# =============================================================================

def IKr_Xr1_inf(V: torch.Tensor) -> torch.Tensor:
    """IKr activation steady-state."""
    return 1.0 / (1.0 + safe_exp((-26.0 - V) / 7.0))


def IKr_Xr1_tau(V: torch.Tensor) -> torch.Tensor:
    """IKr activation time constant (ms)."""
    alpha = 450.0 / (1.0 + safe_exp((-45.0 - V) / 10.0))
    beta = 6.0 / (1.0 + safe_exp((V + 30.0) / 11.5))
    return alpha * beta


def IKr_Xr2_inf(V: torch.Tensor) -> torch.Tensor:
    """IKr inactivation steady-state."""
    return 1.0 / (1.0 + safe_exp((V + 88.0) / 24.0))


def IKr_Xr2_tau(V: torch.Tensor) -> torch.Tensor:
    """IKr inactivation time constant (ms)."""
    alpha = 3.0 / (1.0 + safe_exp((-60.0 - V) / 20.0))
    beta = 1.12 / (1.0 + safe_exp((V - 60.0) / 20.0))
    return alpha * beta


# =============================================================================
# IKs (Slow Delayed Rectifier K+ Current) Gate
# =============================================================================

def IKs_Xs_inf(V: torch.Tensor) -> torch.Tensor:
    """IKs activation steady-state."""
    return 1.0 / (1.0 + safe_exp((-5.0 - V) / 14.0))


def IKs_Xs_tau(V: torch.Tensor) -> torch.Tensor:
    """IKs activation time constant (ms)."""
    alpha = 1400.0 / torch.sqrt(1.0 + safe_exp((5.0 - V) / 6.0))
    beta = 1.0 / (1.0 + safe_exp((V - 35.0) / 15.0))
    return alpha * beta + 80.0


# =============================================================================
# IK1 Rectification
# =============================================================================

def IK1_xK1_inf(V: torch.Tensor, EK: torch.Tensor) -> torch.Tensor:
    """IK1 rectification factor."""
    alpha = 0.1 / (1.0 + safe_exp(0.06 * (V - EK - 200.0)))
    beta = (3.0 * safe_exp(0.0002 * (V - EK + 100.0)) +
            safe_exp(0.1 * (V - EK - 10.0))) / \
           (1.0 + safe_exp(-0.5 * (V - EK)))
    return alpha / (alpha + beta)
