"""
Luo-Rudy 1994 (LRd) Gating Variable Kinetics
=============================================

Rate equations (alpha/beta) for all gating variables in the LRd model.

Gating Variables:
- m, h, j: Fast sodium current (INa)
- d, f, f_Ca: L-type calcium current (ICaL)
- X: Time-dependent potassium current (IK)
- K1_inf: Inward rectifier (IK1) steady-state
- Kp: Plateau current (IKp) activation

All functions are Numba-accelerated for performance.

Reference:
    Luo CH, Rudy Y. Circ Res. 1994;74(6):1071-1096.
    Note: f-gate uses corrected equation from Circ Res 74:1097-113, 1994.

Author: Generated with Claude Code
Date: 2024-12
"""

from __future__ import annotations
import numpy as np
import numba
from typing import Tuple


# =============================================================================
# Numerical Safety Functions
# =============================================================================

@numba.jit(nopython=True, cache=True)
def safe_exp(x: float, limit: float = 50.0) -> float:
    """Exponential with overflow protection."""
    if x > limit:
        return np.exp(limit)
    elif x < -limit:
        return np.exp(-limit)
    return np.exp(x)


@numba.jit(nopython=True, cache=True)
def safe_div(num: float, denom: float, eps: float = 1e-10) -> float:
    """Safe division to avoid division by zero."""
    if abs(denom) < eps:
        return num / (eps if denom >= 0 else -eps)
    return num / denom


# =============================================================================
# INa Gating: m-gate (activation)
# =============================================================================

@numba.jit(nopython=True, cache=True)
def alpha_m(V: float) -> float:
    """
    m-gate opening rate.

    alpha_m = 0.32 * (V + 47.13) / (1 - exp(-0.1 * (V + 47.13)))

    Uses L'Hopital's rule near V = -47.13 mV.
    """
    dV = V + 47.13
    if abs(dV) < 1e-6:
        # L'Hopital limit: alpha_m -> 0.32 / 0.1 = 3.2
        return 3.2
    return 0.32 * dV / (1.0 - safe_exp(-0.1 * dV))


@numba.jit(nopython=True, cache=True)
def beta_m(V: float) -> float:
    """
    m-gate closing rate.

    beta_m = 0.08 * exp(-V / 11)
    """
    return 0.08 * safe_exp(-V / 11.0)


@numba.jit(nopython=True, cache=True)
def m_inf(V: float) -> float:
    """m-gate steady-state."""
    am = alpha_m(V)
    bm = beta_m(V)
    return am / (am + bm)


@numba.jit(nopython=True, cache=True)
def tau_m(V: float) -> float:
    """m-gate time constant [ms]."""
    am = alpha_m(V)
    bm = beta_m(V)
    return 1.0 / (am + bm)


# =============================================================================
# INa Gating: h-gate (fast inactivation)
# =============================================================================

@numba.jit(nopython=True, cache=True)
def alpha_h(V: float) -> float:
    """
    h-gate opening rate.

    For V >= -40 mV: alpha_h = 0
    For V < -40 mV:  alpha_h = 0.135 * exp(-(V + 80) / 6.8)
    """
    if V >= -40.0:
        return 0.0
    return 0.135 * safe_exp(-(V + 80.0) / 6.8)


@numba.jit(nopython=True, cache=True)
def beta_h(V: float) -> float:
    """
    h-gate closing rate.

    For V >= -40 mV: beta_h = 1 / (0.13 * (1 + exp(-(V + 10.66) / 11.1)))
    For V < -40 mV:  beta_h = 3.56 * exp(0.079 * V) + 3.1e5 * exp(0.35 * V)
    """
    if V >= -40.0:
        return 1.0 / (0.13 * (1.0 + safe_exp(-(V + 10.66) / 11.1)))
    return 3.56 * safe_exp(0.079 * V) + 3.1e5 * safe_exp(0.35 * V)


@numba.jit(nopython=True, cache=True)
def h_inf(V: float) -> float:
    """h-gate steady-state."""
    ah = alpha_h(V)
    bh = beta_h(V)
    if ah + bh < 1e-10:
        # At high V, alpha_h = 0, so h_inf -> 0
        return 0.0
    return ah / (ah + bh)


@numba.jit(nopython=True, cache=True)
def tau_h(V: float) -> float:
    """h-gate time constant [ms]."""
    ah = alpha_h(V)
    bh = beta_h(V)
    if ah + bh < 1e-10:
        return 1.0  # Avoid division by zero, use finite value
    return 1.0 / (ah + bh)


# =============================================================================
# INa Gating: j-gate (slow inactivation)
# =============================================================================

@numba.jit(nopython=True, cache=True)
def alpha_j(V: float) -> float:
    """
    j-gate opening rate.

    For V >= -40 mV: alpha_j = 0
    For V < -40 mV:
        alpha_j = (-1.2714e5 * exp(0.2444*V) - 3.474e-5 * exp(-0.04391*V))
                  * (V + 37.78) / (1 + exp(0.311 * (V + 79.23)))
    """
    if V >= -40.0:
        return 0.0

    term1 = -1.2714e5 * safe_exp(0.2444 * V)
    term2 = -3.474e-5 * safe_exp(-0.04391 * V)
    numerator = (term1 + term2) * (V + 37.78)
    denominator = 1.0 + safe_exp(0.311 * (V + 79.23))

    return numerator / denominator


@numba.jit(nopython=True, cache=True)
def beta_j(V: float) -> float:
    """
    j-gate closing rate.

    For V >= -40 mV:
        beta_j = 0.3 * exp(-2.535e-7 * V) / (1 + exp(-0.1 * (V + 32)))
    For V < -40 mV:
        beta_j = 0.1212 * exp(-0.01052 * V) / (1 + exp(-0.1378 * (V + 40.14)))
    """
    if V >= -40.0:
        return 0.3 * safe_exp(-2.535e-7 * V) / (1.0 + safe_exp(-0.1 * (V + 32.0)))
    return 0.1212 * safe_exp(-0.01052 * V) / (1.0 + safe_exp(-0.1378 * (V + 40.14)))


@numba.jit(nopython=True, cache=True)
def j_inf(V: float) -> float:
    """j-gate steady-state."""
    aj = alpha_j(V)
    bj = beta_j(V)
    if aj + bj < 1e-10:
        return 0.0
    return aj / (aj + bj)


@numba.jit(nopython=True, cache=True)
def tau_j(V: float) -> float:
    """j-gate time constant [ms]."""
    aj = alpha_j(V)
    bj = beta_j(V)
    if aj + bj < 1e-10:
        return 1.0
    return 1.0 / (aj + bj)


# =============================================================================
# ICaL Gating: d-gate (activation)
# =============================================================================

@numba.jit(nopython=True, cache=True)
def d_inf(V: float) -> float:
    """
    d-gate steady-state (ICaL activation).

    d_inf = 1 / (1 + exp(-(V + 10) / 6.24))
    """
    return 1.0 / (1.0 + safe_exp(-(V + 10.0) / 6.24))


@numba.jit(nopython=True, cache=True)
def tau_d(V: float) -> float:
    """
    d-gate time constant [ms].

    tau_d = d_inf * (1 - exp(-(V + 10) / 6.24)) / (0.035 * (V + 10))

    Uses L'Hopital's rule near V = -10 mV.
    """
    dV = V + 10.0
    d_inf_val = d_inf(V)

    if abs(dV) < 1e-6:
        # L'Hopital limit at V = -10
        # tau_d -> d_inf / (0.035 * 6.24) = d_inf / 0.2184
        return d_inf_val / 0.2184

    exp_term = safe_exp(-dV / 6.24)
    return d_inf_val * (1.0 - exp_term) / (0.035 * dV)


# =============================================================================
# ICaL Gating: f-gate (voltage inactivation)
# =============================================================================

@numba.jit(nopython=True, cache=True)
def f_inf(V: float) -> float:
    """
    f-gate steady-state (ICaL voltage inactivation).

    CORRECTED equation from Circ Res 74:1097-113, 1994:
    f_inf = 1 / (1 + exp((V + 35.06) / 8.6)) + 0.6 / (1 + exp((50 - V) / 20))

    Note: Original paper had an error in this equation.
    """
    term1 = 1.0 / (1.0 + safe_exp((V + 35.06) / 8.6))
    term2 = 0.6 / (1.0 + safe_exp((50.0 - V) / 20.0))
    return term1 + term2


@numba.jit(nopython=True, cache=True)
def tau_f(V: float) -> float:
    """
    f-gate time constant [ms].

    tau_f = 1 / (0.0197 * exp(-(0.0337 * (V + 10))^2) + 0.02)
    """
    exp_term = safe_exp(-(0.0337 * (V + 10.0)) ** 2)
    return 1.0 / (0.0197 * exp_term + 0.02)


# =============================================================================
# ICaL Gating: f_Ca-gate (calcium inactivation)
# =============================================================================

@numba.jit(nopython=True, cache=True)
def f_Ca_inf(Ca_i: float, Km_fCa: float = 0.6e-3) -> float:
    """
    f_Ca steady-state (ICaL calcium-dependent inactivation).

    f_Ca_inf = 1 / (1 + (Ca_i / Km_fCa))

    This is an instantaneous gate - responds immediately to Ca_i.

    Parameters
    ----------
    Ca_i : float
        Intracellular calcium [mM]
    Km_fCa : float
        Half-saturation constant [mM], default 0.6e-3

    Note: Some formulations use Km_fCa = 0.35e-3 (0.35 µM)
    """
    return 1.0 / (1.0 + Ca_i / Km_fCa)


# =============================================================================
# IK Gating: X-gate (activation)
# =============================================================================

@numba.jit(nopython=True, cache=True)
def alpha_X(V: float) -> float:
    """
    X-gate opening rate.

    alpha_X = 0.0005 * (V + 50) / (1 - exp(-(V + 50) / 13))

    Uses L'Hopital's rule near V = -50 mV.
    """
    dV = V + 50.0
    if abs(dV) < 1e-6:
        # L'Hopital: alpha_X -> 0.0005 * 13 = 0.0065
        return 0.0065
    return 0.0005 * dV / (1.0 - safe_exp(-dV / 13.0))


@numba.jit(nopython=True, cache=True)
def beta_X(V: float) -> float:
    """
    X-gate closing rate.

    beta_X = 0.0013 * (V + 20) / (exp((V + 20) / 10) - 1)

    Uses L'Hopital's rule near V = -20 mV.
    """
    dV = V + 20.0
    if abs(dV) < 1e-6:
        # L'Hopital: beta_X -> 0.0013 * 10 = 0.013
        return 0.013
    return 0.0013 * dV / (safe_exp(dV / 10.0) - 1.0)


@numba.jit(nopython=True, cache=True)
def X_inf(V: float) -> float:
    """X-gate steady-state."""
    aX = alpha_X(V)
    bX = beta_X(V)
    return aX / (aX + bX)


@numba.jit(nopython=True, cache=True)
def tau_X(V: float) -> float:
    """X-gate time constant [ms]."""
    aX = alpha_X(V)
    bX = beta_X(V)
    return 1.0 / (aX + bX)


# =============================================================================
# IK Rectification Factor: Xi
# =============================================================================

@numba.jit(nopython=True, cache=True)
def Xi(V: float) -> float:
    """
    IK rectification factor.

    Xi = 2.837 * (exp(0.04*(V + 77)) - 1) / ((V + 77) * exp(0.04*(V + 35)))

    For V < -100 mV: Xi = 1 (to avoid numerical issues)
    """
    if V < -100.0:
        return 1.0

    dV = V + 77.0
    if abs(dV) < 1e-6:
        # Use finite difference approximation
        dV = 1e-6

    exp1 = safe_exp(0.04 * (V + 77.0))
    exp2 = safe_exp(0.04 * (V + 35.0))

    return 2.837 * (exp1 - 1.0) / (dV * exp2)


# =============================================================================
# IK1 Gating: K1_inf (inward rectifier steady-state)
# =============================================================================

@numba.jit(nopython=True, cache=True)
def alpha_K1(V: float, E_K: float) -> float:
    """
    IK1 alpha rate.

    alpha_K1 = 1.02 / (1 + exp(0.2385 * (V - E_K - 59.215)))
    """
    return 1.02 / (1.0 + safe_exp(0.2385 * (V - E_K - 59.215)))


@numba.jit(nopython=True, cache=True)
def beta_K1(V: float, E_K: float) -> float:
    """
    IK1 beta rate.

    beta_K1 = (0.49124 * exp(0.08032*(V - E_K + 5.476))
               + exp(0.06175*(V - E_K - 594.31)))
              / (1 + exp(-0.5143*(V - E_K + 4.753)))
    """
    dV = V - E_K
    numerator = (0.49124 * safe_exp(0.08032 * (dV + 5.476)) +
                 safe_exp(0.06175 * (dV - 594.31)))
    denominator = 1.0 + safe_exp(-0.5143 * (dV + 4.753))
    return numerator / denominator


@numba.jit(nopython=True, cache=True)
def K1_inf(V: float, E_K: float) -> float:
    """
    IK1 steady-state (instantaneous).

    K1_inf = alpha_K1 / (alpha_K1 + beta_K1)
    """
    aK1 = alpha_K1(V, E_K)
    bK1 = beta_K1(V, E_K)
    return aK1 / (aK1 + bK1)


# =============================================================================
# IKp Gating: Kp (plateau current activation)
# =============================================================================

@numba.jit(nopython=True, cache=True)
def Kp(V: float) -> float:
    """
    IKp activation (instantaneous).

    Kp = 1 / (1 + exp((7.488 - V) / 5.98))
    """
    return 1.0 / (1.0 + safe_exp((7.488 - V) / 5.98))


# =============================================================================
# Convenience: Get all INa gate values
# =============================================================================

@numba.jit(nopython=True, cache=True)
def get_INa_gates(V: float) -> Tuple[float, float, float, float, float, float]:
    """
    Get all INa gating parameters at voltage V.

    Returns
    -------
    m_inf, tau_m, h_inf, tau_h, j_inf, tau_j
    """
    return (
        m_inf(V), tau_m(V),
        h_inf(V), tau_h(V),
        j_inf(V), tau_j(V)
    )


# =============================================================================
# Convenience: Get all ICaL gate values
# =============================================================================

@numba.jit(nopython=True, cache=True)
def get_ICaL_gates(V: float, Ca_i: float) -> Tuple[float, float, float, float, float]:
    """
    Get all ICaL gating parameters.

    Returns
    -------
    d_inf, tau_d, f_inf, tau_f, f_Ca_inf
    """
    return (
        d_inf(V), tau_d(V),
        f_inf(V), tau_f(V),
        f_Ca_inf(Ca_i)
    )


# =============================================================================
# Rush-Larsen Update Functions
# =============================================================================

@numba.jit(nopython=True, cache=True)
def rush_larsen_update(y: float, y_inf: float, tau: float, dt: float) -> float:
    """
    Rush-Larsen exponential integrator for gating variables.

    y_new = y_inf - (y_inf - y) * exp(-dt / tau)

    More stable than forward Euler for stiff gating variables.
    """
    return y_inf - (y_inf - y) * np.exp(-dt / tau)


@numba.jit(nopython=True, cache=True)
def update_m(m: float, V: float, dt: float) -> float:
    """Update m-gate using Rush-Larsen."""
    return rush_larsen_update(m, m_inf(V), tau_m(V), dt)


@numba.jit(nopython=True, cache=True)
def update_h(h: float, V: float, dt: float) -> float:
    """Update h-gate using Rush-Larsen."""
    return rush_larsen_update(h, h_inf(V), tau_h(V), dt)


@numba.jit(nopython=True, cache=True)
def update_j(j: float, V: float, dt: float) -> float:
    """Update j-gate using Rush-Larsen."""
    return rush_larsen_update(j, j_inf(V), tau_j(V), dt)


@numba.jit(nopython=True, cache=True)
def update_d(d: float, V: float, dt: float) -> float:
    """Update d-gate using Rush-Larsen."""
    return rush_larsen_update(d, d_inf(V), tau_d(V), dt)


@numba.jit(nopython=True, cache=True)
def update_f(f: float, V: float, dt: float) -> float:
    """Update f-gate using Rush-Larsen."""
    return rush_larsen_update(f, f_inf(V), tau_f(V), dt)


@numba.jit(nopython=True, cache=True)
def update_X(X: float, V: float, dt: float) -> float:
    """Update X-gate using Rush-Larsen."""
    return rush_larsen_update(X, X_inf(V), tau_X(V), dt)


# =============================================================================
# Test Module
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Testing LRd94 Gating Kinetics")
    print("=" * 60)

    # Voltage range for testing
    V_range = np.linspace(-120, 60, 500)

    # Test INa gates
    print("\n--- INa Gating (m, h, j) ---")

    # Compute values
    m_inf_vals = np.array([m_inf(V) for V in V_range])
    h_inf_vals = np.array([h_inf(V) for V in V_range])
    j_inf_vals = np.array([j_inf(V) for V in V_range])
    tau_m_vals = np.array([tau_m(V) for V in V_range])
    tau_h_vals = np.array([tau_h(V) for V in V_range])
    tau_j_vals = np.array([tau_j(V) for V in V_range])

    print(f"m_inf at -84 mV: {m_inf(-84):.6f}")
    print(f"h_inf at -84 mV: {h_inf(-84):.6f}")
    print(f"j_inf at -84 mV: {j_inf(-84):.6f}")
    print(f"tau_m at -40 mV: {tau_m(-40):.4f} ms")
    print(f"tau_h at -40 mV: {tau_h(-40):.4f} ms")
    print(f"tau_j at -40 mV: {tau_j(-40):.4f} ms")

    # Test ICaL gates
    print("\n--- ICaL Gating (d, f, f_Ca) ---")

    d_inf_vals = np.array([d_inf(V) for V in V_range])
    f_inf_vals = np.array([f_inf(V) for V in V_range])
    tau_d_vals = np.array([tau_d(V) for V in V_range])
    tau_f_vals = np.array([tau_f(V) for V in V_range])

    print(f"d_inf at 0 mV: {d_inf(0):.6f}")
    print(f"f_inf at 0 mV: {f_inf(0):.6f}")
    print(f"tau_d at 0 mV: {tau_d(0):.4f} ms")
    print(f"tau_f at 0 mV: {tau_f(0):.4f} ms")

    # Test f_Ca at different Ca levels
    Ca_range = np.logspace(-5, -2, 100)  # 0.01 µM to 10 mM
    f_Ca_vals = np.array([f_Ca_inf(Ca) for Ca in Ca_range])

    print(f"f_Ca at 0.1 µM Ca: {f_Ca_inf(0.1e-3):.4f}")
    print(f"f_Ca at 1.0 µM Ca: {f_Ca_inf(1.0e-3):.4f}")

    # Test IK gate
    print("\n--- IK Gating (X) ---")

    X_inf_vals = np.array([X_inf(V) for V in V_range])
    tau_X_vals = np.array([tau_X(V) for V in V_range])

    print(f"X_inf at 0 mV: {X_inf(0):.6f}")
    print(f"tau_X at 0 mV: {tau_X(0):.4f} ms")

    # Test IK1 and IKp
    print("\n--- IK1 and IKp ---")
    E_K = -90.0  # Approximate reversal potential
    K1_vals = np.array([K1_inf(V, E_K) for V in V_range])
    Kp_vals = np.array([Kp(V) for V in V_range])

    print(f"K1_inf at -84 mV (E_K=-90): {K1_inf(-84, E_K):.6f}")
    print(f"Kp at 0 mV: {Kp(0):.6f}")

    # Test Xi rectification
    Xi_vals = np.array([Xi(V) for V in V_range])
    print(f"Xi at 0 mV: {Xi(0):.6f}")
    print(f"Xi at -50 mV: {Xi(-50):.6f}")

    # Plotting
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))

    # INa steady-state
    axes[0, 0].plot(V_range, m_inf_vals, 'b-', label='m_inf', linewidth=2)
    axes[0, 0].plot(V_range, h_inf_vals, 'r-', label='h_inf', linewidth=2)
    axes[0, 0].plot(V_range, j_inf_vals, 'g-', label='j_inf', linewidth=2)
    axes[0, 0].set_xlabel('V [mV]')
    axes[0, 0].set_ylabel('Steady-state')
    axes[0, 0].set_title('INa Gating: Steady-State')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(-120, 60)

    # INa time constants
    axes[0, 1].semilogy(V_range, tau_m_vals, 'b-', label='tau_m', linewidth=2)
    axes[0, 1].semilogy(V_range, tau_h_vals, 'r-', label='tau_h', linewidth=2)
    axes[0, 1].semilogy(V_range, tau_j_vals, 'g-', label='tau_j', linewidth=2)
    axes[0, 1].set_xlabel('V [mV]')
    axes[0, 1].set_ylabel('Time constant [ms]')
    axes[0, 1].set_title('INa Gating: Time Constants')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(-120, 60)

    # ICaL steady-state
    axes[0, 2].plot(V_range, d_inf_vals, 'b-', label='d_inf', linewidth=2)
    axes[0, 2].plot(V_range, f_inf_vals, 'r-', label='f_inf', linewidth=2)
    axes[0, 2].set_xlabel('V [mV]')
    axes[0, 2].set_ylabel('Steady-state')
    axes[0, 2].set_title('ICaL Gating: Steady-State')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_xlim(-120, 60)

    # ICaL time constants
    axes[1, 0].plot(V_range, tau_d_vals, 'b-', label='tau_d', linewidth=2)
    axes[1, 0].plot(V_range, tau_f_vals, 'r-', label='tau_f', linewidth=2)
    axes[1, 0].set_xlabel('V [mV]')
    axes[1, 0].set_ylabel('Time constant [ms]')
    axes[1, 0].set_title('ICaL Gating: Time Constants')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(-120, 60)

    # f_Ca vs Ca
    axes[1, 1].semilogx(Ca_range * 1000, f_Ca_vals, 'b-', linewidth=2)  # Convert to µM
    axes[1, 1].set_xlabel('[Ca2+]i [µM]')
    axes[1, 1].set_ylabel('f_Ca_inf')
    axes[1, 1].set_title('ICaL Ca-Dependent Inactivation')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvline(x=0.6, color='r', linestyle='--', label='Km = 0.6 µM')
    axes[1, 1].legend()

    # IK gating
    axes[1, 2].plot(V_range, X_inf_vals, 'b-', label='X_inf', linewidth=2)
    axes[1, 2].set_xlabel('V [mV]')
    axes[1, 2].set_ylabel('Steady-state')
    axes[1, 2].set_title('IK Gating: X Steady-State')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_xlim(-120, 60)

    # IK time constant
    axes[2, 0].plot(V_range, tau_X_vals, 'b-', linewidth=2)
    axes[2, 0].set_xlabel('V [mV]')
    axes[2, 0].set_ylabel('tau_X [ms]')
    axes[2, 0].set_title('IK Gating: X Time Constant')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_xlim(-120, 60)

    # IK1 and IKp
    axes[2, 1].plot(V_range, K1_vals, 'b-', label='K1_inf', linewidth=2)
    axes[2, 1].plot(V_range, Kp_vals, 'r-', label='Kp', linewidth=2)
    axes[2, 1].set_xlabel('V [mV]')
    axes[2, 1].set_ylabel('Steady-state')
    axes[2, 1].set_title('IK1 and IKp Gating')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_xlim(-120, 60)

    # Xi rectification
    axes[2, 2].plot(V_range, Xi_vals, 'b-', linewidth=2)
    axes[2, 2].set_xlabel('V [mV]')
    axes[2, 2].set_ylabel('Xi')
    axes[2, 2].set_title('IK Rectification Factor (Xi)')
    axes[2, 2].grid(True, alpha=0.3)
    axes[2, 2].set_xlim(-120, 60)
    axes[2, 2].set_ylim(0, 2)

    plt.tight_layout()
    plt.savefig('gating_lrd_test.png', dpi=150)
    print(f"\nPlot saved: gating_lrd_test.png")

    plt.show()

    print("\n" + "=" * 60)
    print("Gating kinetics test complete!")
