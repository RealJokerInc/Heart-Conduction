"""
LRd07 Gating Kinetics

All gating equations are translated directly from cell_web2007.m.
Follows Hodgkin-Huxley formalism for voltage-dependent gates.

Reference: Livshitz LM, Rudy Y. Am J Physiol Heart Circ Physiol. 2007;292(6):H2854-66.
"""

import numpy as np
from numba import njit


# =============================================================================
# INa - Fast Sodium Current Gates (m, H, J)
# =============================================================================

@njit(cache=True)
def alpha_m(V: float) -> float:
    """
    INa activation rate constant.

    From MATLAB: am = 0.32*(V+47.13)/(1-exp(-0.1*(V+47.13)))
    """
    dv = V + 47.13
    if abs(dv) < 1e-6:
        # L'Hopital limit to avoid division by zero
        return 0.32 / 0.1
    return 0.32 * dv / (1.0 - np.exp(-0.1 * dv))


@njit(cache=True)
def beta_m(V: float) -> float:
    """
    INa activation rate constant.

    From MATLAB: bm = 0.08*exp(-V/11)
    """
    return 0.08 * np.exp(-V / 11.0)


@njit(cache=True)
def _h_j_factor(V: float) -> float:
    """
    Common factor 'a' used in h and j gate kinetics.

    From MATLAB: a = 1-1./(1+exp(-(V+40)/0.024))

    This is essentially a step function at V=-40:
    - V < -40: a ≈ 1 (uses one set of rate constants)
    - V >= -40: a ≈ 0 (uses another set)
    """
    return 1.0 - 1.0 / (1.0 + np.exp(-(V + 40.0) / 0.024))


@njit(cache=True)
def alpha_h(V: float) -> float:
    """
    INa fast inactivation rate constant.

    From MATLAB: ah = a.*0.135.*exp((80+V)./(-6.8))
    Only applies when V < -40 (a ≈ 1)
    """
    a = _h_j_factor(V)
    return a * 0.135 * np.exp((80.0 + V) / (-6.8))


@njit(cache=True)
def beta_h(V: float) -> float:
    """
    INa fast inactivation rate constant.

    From MATLAB:
    bh = (1-a)./(0.13*(1+exp((V+10.66)/(-11.1)))) +
         (a).*(3.56*exp(0.079*V)+3.1*1e5*exp(0.35*V))
    """
    a = _h_j_factor(V)
    bh1 = (1.0 - a) / (0.13 * (1.0 + np.exp((V + 10.66) / (-11.1))))
    bh2 = a * (3.56 * np.exp(0.079 * V) + 3.1e5 * np.exp(0.35 * V))
    return bh1 + bh2


@njit(cache=True)
def alpha_j(V: float) -> float:
    """
    INa slow inactivation rate constant.

    From MATLAB:
    aj = a.*(-1.2714e5*exp(0.2444*V)-3.474e-5*exp(-0.04391*V)).*
         (V+37.78)./(1+exp(0.311*(V+79.23)))
    Only applies when V < -40 (a ≈ 1)
    """
    a = _h_j_factor(V)
    term1 = -1.2714e5 * np.exp(0.2444 * V) - 3.474e-5 * np.exp(-0.04391 * V)
    term2 = (V + 37.78) / (1.0 + np.exp(0.311 * (V + 79.23)))
    return a * term1 * term2


@njit(cache=True)
def beta_j(V: float) -> float:
    """
    INa slow inactivation rate constant.

    From MATLAB:
    bj = (1-a).*(0.3*exp(-2.535e-7*V)./(1+exp(-0.1*(V+32)))) +
         (a).*(0.1212*exp(-0.01052*V)./(1+exp(-0.1378*(V+40.14))))
    """
    a = _h_j_factor(V)
    bj1 = (1.0 - a) * (0.3 * np.exp(-2.535e-7 * V) / (1.0 + np.exp(-0.1 * (V + 32.0))))
    bj2 = a * (0.1212 * np.exp(-0.01052 * V) / (1.0 + np.exp(-0.1378 * (V + 40.14))))
    return bj1 + bj2


# =============================================================================
# ICaL - L-type Calcium Current Gates (d, f)
# Note: fCa is instantaneous, computed in currents.py
# =============================================================================

@njit(cache=True)
def d_inf(V: float) -> float:
    """
    ICaL activation steady-state.

    From MATLAB:
    dss = 1./(1+exp(-(v+10)/6.24))
    dss1 = 1./(1+exp(-(v+60)/0.024))
    dss = dss * dss1

    The dss1 term is essentially a step function ensuring d→0 for V < -60.
    """
    dss1 = 1.0 / (1.0 + np.exp(-(V + 10.0) / 6.24))
    dss2 = 1.0 / (1.0 + np.exp(-(V + 60.0) / 0.024))
    return dss1 * dss2


@njit(cache=True)
def tau_d(V: float) -> float:
    """
    ICaL activation time constant (ms).

    From MATLAB:
    dss = 1./(1+exp(-(v+10)/6.24))  (first part only)
    taud = dss.*(1-exp(-(v+10)/6.24))./(0.035*(v+10))
    """
    dv = V + 10.0
    dss1 = 1.0 / (1.0 + np.exp(-dv / 6.24))

    if abs(dv) < 1e-6:
        # L'Hopital limit
        return dss1 / (0.035 * 6.24)

    return dss1 * (1.0 - np.exp(-dv / 6.24)) / (0.035 * dv)


@njit(cache=True)
def f_inf(V: float) -> float:
    """
    ICaL voltage inactivation steady-state.

    From MATLAB:
    fss = 1./(1+exp((v+32)/8))+(0.6)./(1+exp((50-v)/20))

    Note: fss can exceed 1 at hyperpolarized potentials.
    """
    return 1.0 / (1.0 + np.exp((V + 32.0) / 8.0)) + 0.6 / (1.0 + np.exp((50.0 - V) / 20.0))


@njit(cache=True)
def tau_f(V: float) -> float:
    """
    ICaL voltage inactivation time constant (ms).

    From MATLAB:
    tauf = 1./(0.0197*exp(-(0.0337*(v+10))^2)+0.02)
    """
    return 1.0 / (0.0197 * np.exp(-(0.0337 * (V + 10.0))**2) + 0.02)


@njit(cache=True)
def f_Ca(Ca_i: float, Km_Ca: float = 6e-4) -> float:
    """
    ICaL Ca-dependent inactivation (instantaneous).

    From MATLAB:
    fca = 1./(1+(cai./data.kmca))

    Args:
        Ca_i: Free intracellular Ca2+ (mM)
        Km_Ca: Half-saturation constant (mM), default 0.6 uM

    Returns:
        fCa gating variable [0, 1]
    """
    return 1.0 / (1.0 + Ca_i / Km_Ca)


# =============================================================================
# IKr - Rapid Delayed Rectifier K+ Current Gate (xr)
# =============================================================================

@njit(cache=True)
def xr_inf(V: float) -> float:
    """
    IKr activation steady-state.

    From MATLAB:
    xrss = 1/(1+exp(-(v+21.5)/7.5))
    """
    return 1.0 / (1.0 + np.exp(-(V + 21.5) / 7.5))


@njit(cache=True)
def tau_xr(V: float) -> float:
    """
    IKr activation time constant (ms).

    From MATLAB:
    tauxr = 1/(0.00138*(v+14.2)/(1-exp(-0.123*(v+14.2)))+
               0.00061*(v+38.9)/(exp(0.145*(v+38.9))-1))
    """
    dv1 = V + 14.2
    dv2 = V + 38.9

    # Handle potential singularities
    if abs(dv1) < 1e-6:
        term1 = 0.00138 / 0.123
    else:
        term1 = 0.00138 * dv1 / (1.0 - np.exp(-0.123 * dv1))

    if abs(dv2) < 1e-6:
        term2 = 0.00061 / 0.145
    else:
        term2 = 0.00061 * dv2 / (np.exp(0.145 * dv2) - 1.0)

    return 1.0 / (term1 + term2)


@njit(cache=True)
def r_Kr(V: float) -> float:
    """
    IKr rectification factor (instantaneous).

    From MATLAB:
    r = 1/(1+exp((v+9)/22.4))

    This reduces IKr at depolarized potentials.
    """
    return 1.0 / (1.0 + np.exp((V + 9.0) / 22.4))


# =============================================================================
# IKs - Slow Delayed Rectifier K+ Current Gates (xs, xs2)
# =============================================================================

@njit(cache=True)
def xs_inf(V: float) -> float:
    """
    IKs activation steady-state (used for both xs and xs2).

    From MATLAB:
    xss = 1/(1+exp(-(v-1.5)/16.7))
    """
    return 1.0 / (1.0 + np.exp(-(V - 1.5) / 16.7))


@njit(cache=True)
def tau_xs(V: float) -> float:
    """
    IKs activation time constant (ms).

    From MATLAB:
    tauxs = 1/(0.0000719*(v+30)/(1-exp(-0.148*(v+30)))+
               0.000131*(v+30)/(exp(0.0687*(v+30))-1))

    Note: xs2 uses tau_xs2 = 4 * tau_xs
    """
    dv = V + 30.0

    if abs(dv) < 1e-6:
        term1 = 0.0000719 / 0.148
        term2 = 0.000131 / 0.0687
    else:
        term1 = 0.0000719 * dv / (1.0 - np.exp(-0.148 * dv))
        term2 = 0.000131 * dv / (np.exp(0.0687 * dv) - 1.0)

    return 1.0 / (term1 + term2)


# =============================================================================
# ICaT - T-type Calcium Current Gates (B, G)
# =============================================================================

@njit(cache=True)
def B_inf(V: float) -> float:
    """
    ICaT activation steady-state.

    From MATLAB:
    bss = 1/(1+exp(-(V+14.0)/10.8))
    """
    return 1.0 / (1.0 + np.exp(-(V + 14.0) / 10.8))


@njit(cache=True)
def tau_B(V: float) -> float:
    """
    ICaT activation time constant (ms).

    From MATLAB:
    taub = 3.7+6.1/(1+exp((V+25.0)/4.5))
    """
    return 3.7 + 6.1 / (1.0 + np.exp((V + 25.0) / 4.5))


@njit(cache=True)
def G_inf(V: float) -> float:
    """
    ICaT inactivation steady-state.

    From MATLAB:
    gss = 1/(1+exp((V+60.0)/5.6))
    """
    return 1.0 / (1.0 + np.exp((V + 60.0) / 5.6))


@njit(cache=True)
def tau_G(V: float) -> float:
    """
    ICaT inactivation time constant (ms).

    From MATLAB:
    a = 1-1./(1+exp(-V/0.0024))
    taug = a.*(-0.875*V+12.0)+12.0*(1-a)

    This is essentially a step function at V=0:
    - V < 0: taug = -0.875*V + 12
    - V >= 0: taug = 12
    """
    a = 1.0 - 1.0 / (1.0 + np.exp(-V / 0.0024))
    return a * (-0.875 * V + 12.0) + 12.0 * (1.0 - a)


# =============================================================================
# Utility Functions
# =============================================================================

@njit(cache=True)
def rush_larsen_step(gate: float, gate_inf: float, tau: float, dt: float) -> float:
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
    exp_term = np.exp(-dt / tau)
    return gate_inf + (gate - gate_inf) * exp_term


@njit(cache=True)
def hh_step(gate: float, alpha: float, beta: float, dt: float) -> float:
    """
    Forward Euler integration for Hodgkin-Huxley gates.

    dgate/dt = alpha*(1-gate) - beta*gate

    Args:
        gate: Current gate value
        alpha: Opening rate constant (1/ms)
        beta: Closing rate constant (1/ms)
        dt: Time step (ms)

    Returns:
        Updated gate value
    """
    dgdt = alpha * (1.0 - gate) - beta * gate
    return gate + dgdt * dt
