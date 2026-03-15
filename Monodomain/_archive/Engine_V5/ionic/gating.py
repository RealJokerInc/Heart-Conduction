"""
O'Hara-Rudy (ORd 2011) Gating Kinetics

Voltage-dependent gating for all ion channels in the ORd model.
Each current has fast/slow components and CaMKII-phosphorylated variants.

Reference: O'Hara T, Virag L, Varro A, Rudy Y. PLoS Comput Biol. 2011;7(5):e1002061.

Currents with gating variables:
- INa:  m, hf, hs, j, hsp, jp
- INaL: mL, hL, hLp
- Ito:  a, iF, iS, ap, iFp, iSp
- ICaL: d, ff, fs, fcaf, fcas, jca, nca, ffp, fcafp
- IKr:  xrf, xrs
- IKs:  xs1, xs2
- IK1:  xk1
"""

import numpy as np
from numba import njit


# =============================================================================
# INa - Fast Sodium Current Gates
# =============================================================================

@njit(cache=True)
def INa_m_inf(V: float) -> float:
    """INa activation steady-state."""
    return 1.0 / (1.0 + np.exp(-(V + 39.57) / 9.871))


@njit(cache=True)
def INa_m_tau(V: float) -> float:
    """INa activation time constant (ms)."""
    return 1.0 / (6.765 * np.exp((V + 11.64) / 34.77) +
                  8.552 * np.exp(-(V + 77.42) / 5.955))


@njit(cache=True)
def INa_h_inf(V: float) -> float:
    """INa inactivation steady-state (for hf, hs, j)."""
    return 1.0 / (1.0 + np.exp((V + 82.90) / 6.086))


@njit(cache=True)
def INa_hf_tau(V: float) -> float:
    """INa fast inactivation time constant (ms)."""
    return 1.0 / (1.432e-5 * np.exp(-(V + 1.196) / 6.285) +
                  6.149 * np.exp((V + 0.5096) / 20.27))


@njit(cache=True)
def INa_hs_tau(V: float) -> float:
    """INa slow inactivation time constant (ms)."""
    return 1.0 / (0.009794 * np.exp(-(V + 17.95) / 28.05) +
                  0.3343 * np.exp((V + 5.730) / 56.66))


@njit(cache=True)
def INa_j_tau(V: float) -> float:
    """INa recovery time constant (ms)."""
    return 2.038 + 1.0 / (0.02136 * np.exp(-(V + 100.6) / 8.281) +
                          0.3052 * np.exp((V + 0.9941) / 38.45))


@njit(cache=True)
def INa_hsp_inf(V: float) -> float:
    """INa slow inactivation (phosphorylated) steady-state."""
    return 1.0 / (1.0 + np.exp((V + 89.1) / 6.086))


# =============================================================================
# INaL - Late Sodium Current Gates
# =============================================================================

@njit(cache=True)
def INaL_mL_inf(V: float) -> float:
    """INaL activation steady-state."""
    return 1.0 / (1.0 + np.exp(-(V + 42.85) / 5.264))


@njit(cache=True)
def INaL_hL_inf(V: float) -> float:
    """INaL inactivation steady-state."""
    return 1.0 / (1.0 + np.exp((V + 87.61) / 7.488))


@njit(cache=True)
def INaL_hLp_inf(V: float) -> float:
    """INaL inactivation (phosphorylated) steady-state."""
    return 1.0 / (1.0 + np.exp((V + 93.81) / 7.488))


# =============================================================================
# Ito - Transient Outward K+ Current Gates
# =============================================================================

@njit(cache=True)
def Ito_a_inf(V: float) -> float:
    """Ito activation steady-state."""
    return 1.0 / (1.0 + np.exp(-(V - 14.34) / 14.82))


@njit(cache=True)
def Ito_a_tau(V: float) -> float:
    """Ito activation time constant (ms)."""
    return 1.0515 / (1.0 / (1.2089 * (1.0 + np.exp(-(V - 18.4099) / 29.3814))) +
                     3.5 / (1.0 + np.exp((V + 100.0) / 29.3814)))


@njit(cache=True)
def Ito_i_inf(V: float) -> float:
    """Ito inactivation steady-state (for iF, iS)."""
    return 1.0 / (1.0 + np.exp((V + 43.94) / 5.711))


@njit(cache=True)
def Ito_iF_tau(V: float, celltype: int = 0) -> float:
    """Ito fast inactivation time constant (ms)."""
    tiF = 4.562 + 1.0 / (0.3933 * np.exp(-(V + 100.0) / 100.0) +
                         0.08004 * np.exp((V + 50.0) / 16.59))
    # Epicardial scaling
    if celltype == 1:
        delta_epi = 1.0 - 0.95 / (1.0 + np.exp((V + 70.0) / 5.0))
        tiF *= delta_epi
    return tiF


@njit(cache=True)
def Ito_iS_tau(V: float, celltype: int = 0) -> float:
    """Ito slow inactivation time constant (ms)."""
    tiS = 23.62 + 1.0 / (0.001416 * np.exp(-(V + 96.52) / 59.05) +
                         1.780e-8 * np.exp((V + 114.1) / 8.079))
    # Epicardial scaling
    if celltype == 1:
        delta_epi = 1.0 - 0.95 / (1.0 + np.exp((V + 70.0) / 5.0))
        tiS *= delta_epi
    return tiS


@njit(cache=True)
def Ito_ap_inf(V: float) -> float:
    """Ito activation (phosphorylated) steady-state."""
    return 1.0 / (1.0 + np.exp(-(V - 24.34) / 14.82))


# =============================================================================
# ICaL - L-type Calcium Current Gates
# =============================================================================

@njit(cache=True)
def ICaL_d_inf(V: float) -> float:
    """ICaL activation steady-state."""
    return 1.0 / (1.0 + np.exp(-(V + 3.940) / 4.230))


@njit(cache=True)
def ICaL_d_tau(V: float) -> float:
    """ICaL activation time constant (ms)."""
    return 0.6 + 1.0 / (np.exp(-0.05 * (V + 6.0)) + np.exp(0.09 * (V + 14.0)))


@njit(cache=True)
def ICaL_f_inf(V: float) -> float:
    """ICaL voltage inactivation steady-state (for ff, fs)."""
    return 1.0 / (1.0 + np.exp((V + 19.58) / 3.696))


@njit(cache=True)
def ICaL_ff_tau(V: float) -> float:
    """ICaL fast voltage inactivation time constant (ms)."""
    return 7.0 + 1.0 / (0.0045 * np.exp(-(V + 20.0) / 10.0) +
                        0.0045 * np.exp((V + 20.0) / 10.0))


@njit(cache=True)
def ICaL_fs_tau(V: float) -> float:
    """ICaL slow voltage inactivation time constant (ms)."""
    return 1000.0 + 1.0 / (0.000035 * np.exp(-(V + 5.0) / 4.0) +
                           0.000035 * np.exp((V + 5.0) / 6.0))


@njit(cache=True)
def ICaL_fcaf_tau(V: float) -> float:
    """ICaL fast Ca inactivation time constant (ms)."""
    return 7.0 + 1.0 / (0.04 * np.exp(-(V - 4.0) / 7.0) +
                        0.04 * np.exp((V - 4.0) / 7.0))


@njit(cache=True)
def ICaL_fcas_tau(V: float) -> float:
    """ICaL slow Ca inactivation time constant (ms)."""
    return 100.0 + 1.0 / (0.00012 * np.exp(-V / 3.0) +
                          0.00012 * np.exp(V / 7.0))


@njit(cache=True)
def ICaL_Afcaf(V: float) -> float:
    """Fraction of fast Ca inactivation."""
    return 0.3 + 0.6 / (1.0 + np.exp((V - 10.0) / 10.0))


# =============================================================================
# IKr - Rapid Delayed Rectifier K+ Current Gates
# =============================================================================

@njit(cache=True)
def IKr_xr_inf(V: float) -> float:
    """IKr activation steady-state."""
    return 1.0 / (1.0 + np.exp(-(V + 8.337) / 6.789))


@njit(cache=True)
def IKr_xrf_tau(V: float) -> float:
    """IKr fast activation time constant (ms)."""
    return 12.98 + 1.0 / (0.3652 * np.exp((V - 31.66) / 3.869) +
                          4.123e-5 * np.exp(-(V - 47.78) / 20.38))


@njit(cache=True)
def IKr_xrs_tau(V: float) -> float:
    """IKr slow activation time constant (ms)."""
    return 1.865 + 1.0 / (0.06629 * np.exp((V - 34.70) / 7.355) +
                          1.128e-5 * np.exp(-(V - 29.74) / 25.94))


@njit(cache=True)
def IKr_Axrf(V: float) -> float:
    """Fraction of fast IKr activation."""
    return 1.0 / (1.0 + np.exp((V + 54.81) / 38.21))


@njit(cache=True)
def IKr_rkr(V: float) -> float:
    """IKr rectification factor."""
    return 1.0 / (1.0 + np.exp((V + 55.0) / 75.0)) * \
           1.0 / (1.0 + np.exp((V - 10.0) / 30.0))


# =============================================================================
# IKs - Slow Delayed Rectifier K+ Current Gates
# =============================================================================

@njit(cache=True)
def IKs_xs1_inf(V: float) -> float:
    """IKs activation steady-state."""
    return 1.0 / (1.0 + np.exp(-(V + 11.60) / 8.932))


@njit(cache=True)
def IKs_xs1_tau(V: float) -> float:
    """IKs xs1 time constant (ms)."""
    return 817.3 + 1.0 / (2.326e-4 * np.exp((V + 48.28) / 17.80) +
                          0.001292 * np.exp(-(V + 210.0) / 230.0))


@njit(cache=True)
def IKs_xs2_tau(V: float) -> float:
    """IKs xs2 time constant (ms)."""
    return 1.0 / (0.01 * np.exp((V - 50.0) / 20.0) +
                  0.0193 * np.exp(-(V + 66.54) / 31.0))


# =============================================================================
# IK1 - Inward Rectifier K+ Current Gate
# =============================================================================

@njit(cache=True)
def IK1_xk1_inf(V: float, ko: float) -> float:
    """IK1 activation steady-state (K-dependent)."""
    return 1.0 / (1.0 + np.exp(-(V + 2.5538 * ko + 144.59) /
                               (1.5692 * ko + 3.8115)))


@njit(cache=True)
def IK1_xk1_tau(V: float) -> float:
    """IK1 activation time constant (ms)."""
    return 122.2 / (np.exp(-(V + 127.2) / 20.36) +
                    np.exp((V + 236.8) / 69.33))


@njit(cache=True)
def IK1_rk1(V: float, ko: float) -> float:
    """IK1 rectification factor."""
    return 1.0 / (1.0 + np.exp((V + 105.8 - 2.6 * ko) / 9.493))


# =============================================================================
# Rush-Larsen Integration
# =============================================================================

@njit(cache=True)
def rush_larsen(gate: float, gate_inf: float, tau: float, dt: float) -> float:
    """
    Rush-Larsen exponential integration for gating variables.

    More stable than forward Euler for fast gates.

    Args:
        gate: Current gate value
        gate_inf: Steady-state value
        tau: Time constant (ms)
        dt: Time step (ms)

    Returns:
        Updated gate value
    """
    return gate_inf - (gate_inf - gate) * np.exp(-dt / tau)
