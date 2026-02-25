"""
Voltage-Dependent Gating Kinetics for ORd Model

Contains steady-state (x_inf) and time constant (tau_x) functions
for all gating variables. All functions are vectorized for batch
operation over tissue tensors.

Gate naming convention:
- INa: m, hf, hs, j (+ phosphorylated: hsp, jp)
- INaL: mL, hL (+ phosphorylated: hLp)
- Ito: a, iF, iS (+ phosphorylated: ap, iFp, iSp)
- ICaL: d, ff, fs, fcaf, fcas, jca (+ phosphorylated: ffp, fcafp)
- IKr: xrf, xrs
- IKs: xs1, xs2
- IK1: xk1
"""

import torch


def safe_exp(x: torch.Tensor, limit: float = 80.0) -> torch.Tensor:
    """
    Clamped exponential to prevent overflow.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor
    limit : float
        Clamp range [-limit, limit]

    Returns
    -------
    torch.Tensor
        exp(clamp(x, -limit, limit))
    """
    return torch.exp(torch.clamp(x, -limit, limit))


# =============================================================================
# INa (Fast Sodium Current) Gates
# =============================================================================

def INa_m_inf(V: torch.Tensor) -> torch.Tensor:
    """INa activation steady-state."""
    return 1.0 / (1.0 + safe_exp(-(V + 39.57) / 9.871))


def INa_m_tau(V: torch.Tensor) -> torch.Tensor:
    """INa activation time constant (ms)."""
    return 1.0 / (6.765 * safe_exp((V + 11.64) / 34.77) +
                  8.552 * safe_exp(-(V + 77.42) / 5.955))


def INa_h_inf(V: torch.Tensor) -> torch.Tensor:
    """INa inactivation steady-state (shared by hf, hs)."""
    return 1.0 / (1.0 + safe_exp((V + 82.90) / 6.086))


def INa_hf_tau(V: torch.Tensor) -> torch.Tensor:
    """INa fast inactivation time constant (ms)."""
    return 1.0 / (1.432e-5 * safe_exp(-(V + 1.196) / 6.285) +
                  6.149 * safe_exp((V + 0.5096) / 20.27))


def INa_hs_tau(V: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """INa slow inactivation time constant (ms)."""
    return scale / (0.009794 * safe_exp(-(V + 17.95) / 28.05) +
                    0.3343 * safe_exp((V + 5.730) / 56.66))


def INa_j_inf(V: torch.Tensor) -> torch.Tensor:
    """INa recovery steady-state."""
    return INa_h_inf(V)  # Same as h_inf


def INa_j_tau(V: torch.Tensor) -> torch.Tensor:
    """INa recovery time constant (ms)."""
    return 2.038 + 1.0 / (0.02136 * safe_exp(-(V + 100.6) / 8.281) +
                          0.3052 * safe_exp((V + 0.9941) / 38.45))


# Phosphorylated variants
def INa_hsp_inf(V: torch.Tensor) -> torch.Tensor:
    """Phosphorylated INa fast inactivation steady-state."""
    return 1.0 / (1.0 + safe_exp((V + 89.1) / 6.086))


def INa_hsp_tau(V: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Phosphorylated INa slow inactivation time constant (ms)."""
    return 3.0 * INa_hs_tau(V, scale)


def INa_jp_inf(V: torch.Tensor) -> torch.Tensor:
    """Phosphorylated INa recovery steady-state."""
    return INa_hsp_inf(V)


def INa_jp_tau(V: torch.Tensor) -> torch.Tensor:
    """Phosphorylated INa recovery time constant (ms)."""
    return 1.46 * INa_j_tau(V)


# =============================================================================
# INaL (Late Sodium Current) Gates
# =============================================================================

def INaL_mL_inf(V: torch.Tensor) -> torch.Tensor:
    """INaL activation steady-state."""
    return 1.0 / (1.0 + safe_exp(-(V + 42.85) / 5.264))


def INaL_mL_tau(V: torch.Tensor) -> torch.Tensor:
    """INaL activation time constant (ms)."""
    return INa_m_tau(V)  # Same as INa


def INaL_hL_inf(V: torch.Tensor) -> torch.Tensor:
    """INaL inactivation steady-state."""
    return 1.0 / (1.0 + safe_exp((V + 87.61) / 7.488))


def INaL_hL_tau(V: torch.Tensor) -> torch.Tensor:
    """INaL inactivation time constant (ms)."""
    return 200.0 * torch.ones_like(V)  # Constant 200 ms


def INaL_hLp_inf(V: torch.Tensor) -> torch.Tensor:
    """Phosphorylated INaL inactivation steady-state."""
    return 1.0 / (1.0 + safe_exp((V + 93.81) / 7.488))


def INaL_hLp_tau(V: torch.Tensor) -> torch.Tensor:
    """Phosphorylated INaL inactivation time constant (ms)."""
    return 3.0 * INaL_hL_tau(V)


# =============================================================================
# Ito (Transient Outward K+ Current) Gates
# =============================================================================

def Ito_a_inf(V: torch.Tensor) -> torch.Tensor:
    """Ito activation steady-state."""
    return 1.0 / (1.0 + safe_exp(-(V - 14.34) / 14.82))


def Ito_a_tau(V: torch.Tensor) -> torch.Tensor:
    """Ito activation time constant (ms)."""
    return 1.0515 / (1.0 / (1.2089 * (1.0 + safe_exp(-(V - 18.4099) / 29.3814))) +
                     3.5 / (1.0 + safe_exp((V + 100.0) / 29.3814)))


def Ito_i_inf(V: torch.Tensor) -> torch.Tensor:
    """Ito inactivation steady-state (shared by iF, iS)."""
    return 1.0 / (1.0 + safe_exp((V + 43.94) / 5.711))


def Ito_iF_tau(V: torch.Tensor) -> torch.Tensor:
    """Ito fast inactivation time constant (ms)."""
    return 4.562 + 1.0 / (0.3933 * safe_exp(-(V + 100.0) / 100.0) +
                          0.08004 * safe_exp((V + 50.0) / 16.59))


def Ito_iS_tau(V: torch.Tensor) -> torch.Tensor:
    """Ito slow inactivation time constant (ms)."""
    return 23.62 + 1.0 / (0.001416 * safe_exp(-(V + 96.52) / 59.05) +
                          1.780e-8 * safe_exp((V + 114.1) / 8.079))


# Phosphorylated variants
def Ito_ap_inf(V: torch.Tensor) -> torch.Tensor:
    """Phosphorylated Ito activation steady-state."""
    return 1.0 / (1.0 + safe_exp(-(V - 24.34) / 14.82))


def Ito_ap_tau(V: torch.Tensor) -> torch.Tensor:
    """Phosphorylated Ito activation time constant (ms)."""
    return Ito_a_tau(V)


def Ito_iFp_tau(V: torch.Tensor) -> torch.Tensor:
    """Phosphorylated Ito fast inactivation time constant (ms)."""
    return Ito_iF_tau(V)


def Ito_iSp_tau(V: torch.Tensor) -> torch.Tensor:
    """Phosphorylated Ito slow inactivation time constant (ms)."""
    return Ito_iS_tau(V)


# Ito delta (for EPI cells)
def Ito_delta_epi(V: torch.Tensor) -> torch.Tensor:
    """
    Epicardial Ito inactivation time constant scaling factor.

    From ORd C++: delta_epi=1.0-(0.95/(1.0+exp((v+70.0)/5.0)))
    Multiplies tiF and tiS for EPI cells only.
    """
    return 1.0 - 0.95 / (1.0 + safe_exp((V + 70.0) / 5.0))


# =============================================================================
# ICaL (L-type Calcium Current) Gates
# =============================================================================

def ICaL_d_inf(V: torch.Tensor) -> torch.Tensor:
    """ICaL activation steady-state."""
    return 1.0 / (1.0 + safe_exp(-(V + 3.940) / 4.230))


def ICaL_d_tau(V: torch.Tensor) -> torch.Tensor:
    """ICaL activation time constant (ms)."""
    return 0.6 + 1.0 / (safe_exp(-0.05 * (V + 6.0)) +
                        safe_exp(0.09 * (V + 14.0)))


def ICaL_f_inf(V: torch.Tensor) -> torch.Tensor:
    """ICaL voltage inactivation steady-state (shared by ff, fs)."""
    return 1.0 / (1.0 + safe_exp((V + 19.58) / 3.696))


def ICaL_ff_tau(V: torch.Tensor) -> torch.Tensor:
    """ICaL fast voltage inactivation time constant (ms)."""
    return 7.0 + 1.0 / (0.0045 * safe_exp(-(V + 20.0) / 10.0) +
                        0.0045 * safe_exp((V + 20.0) / 10.0))


def ICaL_fs_tau(V: torch.Tensor) -> torch.Tensor:
    """ICaL slow voltage inactivation time constant (ms)."""
    return 1000.0 + 1.0 / (0.000035 * safe_exp(-(V + 5.0) / 4.0) +
                           0.000035 * safe_exp((V + 5.0) / 6.0))


def ICaL_fcaf_tau(V: torch.Tensor) -> torch.Tensor:
    """ICaL fast Ca-dependent inactivation time constant (ms)."""
    return 7.0 + 1.0 / (0.04 * safe_exp(-(V - 4.0) / 7.0) +
                        0.04 * safe_exp((V - 4.0) / 7.0))


def ICaL_fcas_tau(V: torch.Tensor) -> torch.Tensor:
    """ICaL slow Ca-dependent inactivation time constant (ms)."""
    return 100.0 + 1.0 / (0.00012 * safe_exp(-V / 3.0) +
                          0.00012 * safe_exp(V / 7.0))


def ICaL_jca_tau(V: torch.Tensor) -> torch.Tensor:
    """ICaL Ca-dependent recovery time constant (ms)."""
    return 75.0 * torch.ones_like(V)  # Constant 75 ms


# Phosphorylated variants
def ICaL_ffp_tau(V: torch.Tensor) -> torch.Tensor:
    """Phosphorylated ICaL fast voltage inactivation time constant (ms)."""
    return 2.5 * ICaL_ff_tau(V)


def ICaL_fcafp_tau(V: torch.Tensor) -> torch.Tensor:
    """Phosphorylated ICaL fast Ca-dependent inactivation time constant (ms)."""
    return 2.5 * ICaL_fcaf_tau(V)


# =============================================================================
# IKr (Rapid Delayed Rectifier K+ Current) Gates
# =============================================================================

def IKr_xr_inf(V: torch.Tensor) -> torch.Tensor:
    """IKr activation steady-state (shared by xrf, xrs)."""
    return 1.0 / (1.0 + safe_exp(-(V + 8.337) / 6.789))


def IKr_xrf_tau(V: torch.Tensor) -> torch.Tensor:
    """IKr fast activation time constant (ms)."""
    return 12.98 + 1.0 / (0.3652 * safe_exp((V - 31.66) / 3.869) +
                          4.123e-5 * safe_exp(-(V - 47.78) / 20.38))


def IKr_xrs_tau(V: torch.Tensor) -> torch.Tensor:
    """IKr slow activation time constant (ms)."""
    return 1.865 + 1.0 / (0.06629 * safe_exp((V - 34.70) / 7.355) +
                          1.128e-5 * safe_exp(-(V - 29.74) / 25.94))


def IKr_Axrf(V: torch.Tensor) -> torch.Tensor:
    """IKr fast/slow activation fraction."""
    return 1.0 / (1.0 + safe_exp((V + 54.81) / 38.21))


def IKr_RKr(V: torch.Tensor) -> torch.Tensor:
    """IKr rectification factor."""
    return 1.0 / (1.0 + safe_exp((V + 55.0) / 75.0)) * \
           1.0 / (1.0 + safe_exp((V - 10.0) / 30.0))


# =============================================================================
# IKs (Slow Delayed Rectifier K+ Current) Gates
# =============================================================================

def IKs_xs1_inf(V: torch.Tensor) -> torch.Tensor:
    """IKs first activation gate steady-state."""
    return 1.0 / (1.0 + safe_exp(-(V + 11.60) / 8.932))


def IKs_xs1_tau(V: torch.Tensor) -> torch.Tensor:
    """IKs first activation gate time constant (ms)."""
    return 817.3 + 1.0 / (2.326e-4 * safe_exp((V + 48.28) / 17.80) +
                          0.001292 * safe_exp(-(V + 210.0) / 230.0))


def IKs_xs2_inf(V: torch.Tensor) -> torch.Tensor:
    """IKs second activation gate steady-state."""
    return IKs_xs1_inf(V)  # Same as xs1


def IKs_xs2_tau(V: torch.Tensor) -> torch.Tensor:
    """IKs second activation gate time constant (ms)."""
    return 1.0 / (0.01 * safe_exp((V - 50.0) / 20.0) +
                  0.0193 * safe_exp(-(V + 66.54) / 31.0))


# =============================================================================
# IK1 (Inward Rectifier K+ Current) Gates
# =============================================================================

def IK1_xk1_inf(V: torch.Tensor, ko: float) -> torch.Tensor:
    """IK1 activation steady-state (K-dependent)."""
    return 1.0 / (1.0 + safe_exp(-(V + 2.5538 * ko + 144.59) /
                                  (1.5692 * ko + 3.8115)))


def IK1_xk1_tau(V: torch.Tensor) -> torch.Tensor:
    """IK1 activation time constant (ms)."""
    return 122.2 / (safe_exp(-(V + 127.2) / 20.36) +
                    safe_exp((V + 236.8) / 69.33))


def IK1_rk1(V: torch.Tensor, ko: float) -> torch.Tensor:
    """IK1 rectification factor."""
    return 1.0 / (1.0 + safe_exp((V + 105.8 - 2.6 * ko) / 9.493))


# =============================================================================
# Utility Functions
# =============================================================================

def rush_larsen(x: torch.Tensor, x_inf: torch.Tensor,
                tau: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Rush-Larsen exponential integration.

    Exact solution for dx/dt = (x_inf - x) / tau over timestep dt.

    Parameters
    ----------
    x : torch.Tensor
        Current gate value
    x_inf : torch.Tensor
        Steady-state value
    tau : torch.Tensor
        Time constant (ms)
    dt : float
        Time step (ms)

    Returns
    -------
    torch.Tensor
        Updated gate value
    """
    return x_inf - (x_inf - x) * torch.exp(-dt / tau)
