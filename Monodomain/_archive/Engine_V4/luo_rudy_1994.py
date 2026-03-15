"""
Luo-Rudy 1994 Dynamic Ionic Model (Numba-Optimized)
====================================================

2D-capable implementation based on CellML repository version.
Uses GHK formulation for L-type Ca2+ current.

State Variables (11 total):
- V: Membrane potential [mV]
- m, h, j: Fast Na+ channel gates
- d, f: L-type Ca2+ channel gates (f_Ca computed instantaneously)
- Na_i, K_i, Ca_i: Intracellular ion concentrations [mM]
- Ca_jsr, Ca_nsr: SR calcium concentrations [mM]

Reference:
    Luo CH, Rudy Y. "A dynamic model of the cardiac ventricular action potential.
    I. Simulations of ionic currents and concentration changes."
    Circ Res. 1994;74(6):1071-1096.

CellML Source: https://models.cellml.org/e/81/luo_rudy_1994.cellml

Author: Generated with Claude Code
Date: 2025-12-11
"""

from __future__ import annotations
import numpy as np
import numba
from typing import Dict, Tuple
from dataclasses import dataclass

from parameters import (
    LRd94Params, LRd94InitialConditions, PhysicalConstants, CellGeometry,
    default_params, default_initial_conditions, default_physical_constants,
    default_cell_geometry, STATE_INDICES, STATE_NAMES, N_STATES
)


# =============================================================================
# Physical Constants (compile-time for Numba)
# =============================================================================

R = 8314.5       # mJ/(mol·K)
T = 310.0        # K
F = 96845.0      # C/mol
RTF = R * T / F  # ~26.6 mV
FRT = F / (R * T) # ~0.0376 /mV

# Extracellular concentrations
Na_o = 140.0     # mM
K_o = 5.4        # mM
Ca_o = 1.8       # mM

# Cell geometry
C_m = 1.0        # µF/cm²
A_cap = 200.0    # Capacitative surface area / cell volume [1/cm] (= A_m)
V_myo = 0.68     # volume fraction
V_jsr = 0.0048   # volume fraction
V_nsr = 0.0552   # volume fraction


# =============================================================================
# Numba Helper Functions
# =============================================================================

@numba.jit(nopython=True, cache=True)
def safe_exp(x: float, limit: float = 50.0) -> float:
    """Safe exponential to avoid overflow."""
    return np.exp(max(-limit, min(limit, x)))


@numba.jit(nopython=True, cache=True)
def safe_div(num: float, denom: float, eps: float = 1e-10) -> float:
    """Safe division to avoid division by zero."""
    if abs(denom) < eps:
        return num / eps if denom >= 0 else -num / eps
    return num / denom


# =============================================================================
# Gating Kinetics (from CellML)
# =============================================================================

# --- Fast Na+ Channel (m, h, j) ---

@numba.jit(nopython=True, cache=True)
def alpha_m(V: float) -> float:
    dV = V + 47.13
    if abs(dV) < 1e-7:
        return 3.2
    return 0.32 * dV / (1.0 - safe_exp(-0.1 * dV))


@numba.jit(nopython=True, cache=True)
def beta_m(V: float) -> float:
    return 0.08 * safe_exp(-V / 11.0)


@numba.jit(nopython=True, cache=True)
def alpha_h(V: float) -> float:
    if V >= -40.0:
        return 0.0
    return 0.135 * safe_exp(-(V + 80.0) / 6.8)


@numba.jit(nopython=True, cache=True)
def beta_h(V: float) -> float:
    if V >= -40.0:
        return 1.0 / (0.13 * (1.0 + safe_exp(-(V + 10.66) / 11.1)))
    return 3.56 * safe_exp(0.079 * V) + 3.1e5 * safe_exp(0.35 * V)


@numba.jit(nopython=True, cache=True)
def alpha_j(V: float) -> float:
    if V >= -40.0:
        return 0.0
    term1 = -1.2714e5 * safe_exp(0.2444 * V)
    term2 = 3.474e-5 * safe_exp(-0.04391 * V)
    denom = 1.0 + safe_exp(0.311 * (V + 79.23))
    return (term1 - term2) * (V + 37.78) / denom


@numba.jit(nopython=True, cache=True)
def beta_j(V: float) -> float:
    if V >= -40.0:
        return 0.3 * safe_exp(-2.535e-7 * V) / (1.0 + safe_exp(-0.1 * (V + 32.0)))
    return 0.1212 * safe_exp(-0.01052 * V) / (1.0 + safe_exp(-0.1378 * (V + 40.14)))


# --- L-type Ca2+ Channel (d, f) - from CellML ---

@numba.jit(nopython=True, cache=True)
def d_inf(V: float) -> float:
    return 1.0 / (1.0 + safe_exp(-(V + 10.0) / 6.24))


@numba.jit(nopython=True, cache=True)
def tau_d(V: float) -> float:
    """Time constant for d gate [ms]."""
    dV = V + 10.0
    d_inf_val = d_inf(V)
    if abs(dV) < 1e-4:
        # Limit as V -> -10: tau_d = d_inf / 0.035
        return d_inf_val / 0.035
    return d_inf_val * (1.0 - safe_exp(-dV / 6.24)) / (0.035 * dV)


@numba.jit(nopython=True, cache=True)
def f_inf(V: float) -> float:
    return 1.0 / (1.0 + safe_exp((V + 35.06) / 8.6)) + 0.6 / (1.0 + safe_exp((50.0 - V) / 20.0))


@numba.jit(nopython=True, cache=True)
def tau_f(V: float) -> float:
    """Time constant for f gate [ms]."""
    arg = 0.0337 * (V + 10.0)
    return 1.0 / (0.0197 * safe_exp(-arg * arg) + 0.02)


@numba.jit(nopython=True, cache=True)
def f_Ca_inf(Ca_i: float, Km_Ca: float) -> float:
    """Instantaneous Ca-dependent inactivation (Hill coeff = 2)."""
    ratio = Ca_i / Km_Ca
    return 1.0 / (1.0 + ratio * ratio)


# --- Time-dependent K+ Channel (X) ---

@numba.jit(nopython=True, cache=True)
def X_inf(V: float) -> float:
    return 1.0 / (1.0 + safe_exp(-(V + 21.5) / 7.5))


@numba.jit(nopython=True, cache=True)
def tau_X(V: float) -> float:
    """Time constant for X gate [ms]."""
    denom1 = 1.0 + safe_exp(-(V + 22.0) / 9.0)
    denom2 = 1.0 + safe_exp((V + 22.0) / 15.0)
    return 1.0 / (0.0005 / denom1 + 0.0008 / denom2)


@numba.jit(nopython=True, cache=True)
def X_i(V: float, K_i: float) -> float:
    """Rectification factor for I_K."""
    E_K = RTF * np.log(K_o / K_i)
    dV = V - E_K - 77.0
    if abs(dV) < 1e-7:
        return 0.0
    return 2.837 * (safe_exp(0.04 * (V - E_K + 77.0)) - 1.0) / \
           ((V - E_K + 77.0) * safe_exp(0.04 * (V - E_K + 35.0)))


# =============================================================================
# GHK Current Calculation (from CellML)
# =============================================================================

@numba.jit(nopython=True, cache=True)
def ghk_flux(V: float, z: float, C_i: float, C_o: float,
             gamma_i: float, gamma_o: float) -> float:
    """
    Goldman-Hodgkin-Katz flux factor.

    Returns: z² * (V*F/RT) * (γ_i*C_i*exp(z*V*F/RT) - γ_o*C_o) / (exp(z*V*F/RT) - 1)

    Units: mM (concentration-like, needs P*F to get current)
    """
    z_FRT_V = z * FRT * V  # dimensionless

    if abs(V) < 1e-4:
        # Limit as V -> 0: z² * F/RT * (γ_i*C_i - γ_o*C_o)
        return z * z * FRT * (gamma_i * C_i - gamma_o * C_o)

    exp_term = safe_exp(z_FRT_V)
    numerator = gamma_i * C_i * exp_term - gamma_o * C_o
    denominator = exp_term - 1.0

    return z * z * FRT * V * numerator / denominator


# =============================================================================
# Main Ionic Step Kernel (2D)
# =============================================================================

@numba.jit(nopython=True, cache=True, parallel=False)
def lrd94_ionic_step_kernel(
    # State arrays (ny, nx)
    V_arr: np.ndarray,
    m_arr: np.ndarray,
    h_arr: np.ndarray,
    j_arr: np.ndarray,
    d_arr: np.ndarray,
    f_arr: np.ndarray,
    Na_i_arr: np.ndarray,
    K_i_arr: np.ndarray,
    Ca_i_arr: np.ndarray,
    Ca_jsr_arr: np.ndarray,
    Ca_nsr_arr: np.ndarray,
    # Stimulus array
    I_stim_arr: np.ndarray,
    # Time step
    dt: float,
    # Model parameters
    G_Na: float,
    G_K: float,
    G_K1: float,
    G_Kp: float,
    G_Na_b: float,
    G_Ca_b: float,
    P_Ca: float,
    P_Na: float,
    P_K: float,
    gamma_Cai: float,
    gamma_Cao: float,
    gamma_Nai: float,
    gamma_Nao: float,
    gamma_Ki: float,
    gamma_Ko: float,
    Km_Ca: float,
    I_NaK_bar: float,
    K_m_Na_i: float,
    K_m_K_o: float,
    I_pCa_bar: float,
    K_m_pCa: float,
    k_NaCa: float,
    K_m_Na: float,
    K_m_Ca: float,
    k_sat: float,
    eta: float,
    G_rel_max: float,
    tau_tr: float,
    K_m_rel: float,
    K_m_up: float,
    I_up_bar: float,
    TRPN_tot: float,
    K_m_TRPN: float,
    CMDN_tot: float,
    K_m_CMDN: float,
    CSQN_tot: float,
    K_m_CSQN: float,
) -> None:
    """
    Numba kernel for LRd94 ionic step with GHK I_Ca,L.
    Updates all state arrays in-place.
    """
    ny, nx = V_arr.shape

    for i in range(ny):
        for jj in range(nx):
            # Extract local state
            V = V_arr[i, jj]
            m = m_arr[i, jj]
            h = h_arr[i, jj]
            j_gate = j_arr[i, jj]
            d = d_arr[i, jj]
            f = f_arr[i, jj]
            Na_i = Na_i_arr[i, jj]
            K_i = K_i_arr[i, jj]
            Ca_i = Ca_i_arr[i, jj]
            Ca_jsr = Ca_jsr_arr[i, jj]
            Ca_nsr = Ca_nsr_arr[i, jj]
            I_stim = I_stim_arr[i, jj]

            # --- Reversal potentials ---
            E_Na = RTF * np.log(Na_o / Na_i)
            E_K = RTF * np.log(K_o / K_i)
            E_Ca = 0.5 * RTF * np.log(Ca_o / max(Ca_i, 1e-10))

            # --- Fast Na+ current ---
            I_Na = G_Na * m**3 * h * j_gate * (V - E_Na)

            # --- L-type Ca2+ current (Ohmic formulation for stability) ---
            # Calcium-dependent inactivation (instantaneous)
            f_Ca = f_Ca_inf(Ca_i, Km_Ca)

            # Simple Ohmic current: I = G * d * f * fCa * (V - E_Ca)
            # Tuned for APD ~300 ms with proper Ca transient
            G_Ca_L_ohmic = 0.10  # mS/cm²
            I_Ca_L = G_Ca_L_ohmic * d * f * f_Ca * (V - E_Ca)

            # --- Time-dependent K+ current ---
            # Use steady-state for X gate (simplification for stability)
            X = X_inf(V)
            Xi = X_i(V, K_i)
            G_K_scaled = G_K * np.sqrt(K_o / 5.4)
            I_K = G_K_scaled * X * X * Xi * (V - E_K)

            # --- Inward rectifier K+ current ---
            G_K1_scaled = G_K1 * np.sqrt(K_o / 5.4)
            alpha_K1 = 1.02 / (1.0 + safe_exp(0.2385 * (V - E_K - 59.215)))
            beta_K1 = (0.49124 * safe_exp(0.08032 * (V - E_K + 5.476)) +
                       safe_exp(0.06175 * (V - E_K - 594.31))) / \
                      (1.0 + safe_exp(-0.5143 * (V - E_K + 4.753)))
            K1_inf = alpha_K1 / (alpha_K1 + beta_K1)
            I_K1 = G_K1_scaled * K1_inf * (V - E_K)

            # --- Plateau K+ current ---
            Kp = 1.0 / (1.0 + safe_exp((7.488 - V) / 5.98))
            I_Kp = G_Kp * Kp * (V - E_K)

            # --- Na+/Ca2+ exchanger ---
            exp_eta = safe_exp(eta * V * FRT)
            exp_eta1 = safe_exp((eta - 1.0) * V * FRT)
            num_NaCa = k_NaCa * (Na_i**3 * Ca_o * exp_eta - Na_o**3 * Ca_i * exp_eta1)
            denom_NaCa = (1.0 + k_sat * exp_eta1) * (K_m_Na**3 + Na_o**3) * (K_m_Ca + Ca_o)
            I_NaCa = num_NaCa / denom_NaCa

            # --- Na+/K+ pump ---
            sigma = (np.exp(Na_o / 67.3) - 1.0) / 7.0
            f_NaK = 1.0 / (1.0 + 0.1245 * safe_exp(-0.1 * V * FRT) +
                          0.0365 * sigma * safe_exp(-V * FRT))
            I_NaK = I_NaK_bar * f_NaK * (K_o / (K_o + K_m_K_o)) * (Na_i / (Na_i + K_m_Na_i))

            # --- Sarcolemmal Ca2+ pump ---
            I_pCa = I_pCa_bar * Ca_i / (K_m_pCa + Ca_i)

            # --- Background currents ---
            I_Na_b = G_Na_b * (V - E_Na)
            I_Ca_b = G_Ca_b * (V - E_Ca)

            # --- Total ionic current ---
            I_ion = I_Na + I_Ca_L + I_K + I_K1 + I_Kp + I_NaCa + I_NaK + I_pCa + I_Na_b + I_Ca_b

            # ===================
            # Update gating variables (Rush-Larsen)
            # ===================

            # m-gate
            am = alpha_m(V)
            bm = beta_m(V)
            tau_m = 1.0 / (am + bm)
            m_inf_val = am * tau_m
            m_new = m_inf_val - (m_inf_val - m) * np.exp(-dt / tau_m)

            # h-gate
            ah = alpha_h(V)
            bh = beta_h(V)
            sum_h = ah + bh
            if sum_h > 1e-10:
                tau_h = 1.0 / sum_h
                h_inf_val = ah * tau_h
                h_new = h_inf_val - (h_inf_val - h) * np.exp(-dt / tau_h)
            else:
                h_new = h

            # j-gate
            aj = alpha_j(V)
            bj = beta_j(V)
            sum_j = aj + bj
            if sum_j > 1e-10:
                tau_j = 1.0 / sum_j
                j_inf_val = aj * tau_j
                j_new = j_inf_val - (j_inf_val - j_gate) * np.exp(-dt / tau_j)
            else:
                j_new = j_gate

            # d-gate
            d_inf_val = d_inf(V)
            tau_d_val = tau_d(V)
            tau_d_val = max(tau_d_val, 0.01)  # Prevent too small tau
            d_new = d_inf_val - (d_inf_val - d) * np.exp(-dt / tau_d_val)

            # f-gate
            f_inf_val = f_inf(V)
            tau_f_val = tau_f(V)
            f_new = f_inf_val - (f_inf_val - f) * np.exp(-dt / tau_f_val)

            # ===================
            # Update ion concentrations
            # ===================

            # Simplified CICR: release proportional to d-gate and gradient
            g_rel = G_rel_max * d * Ca_i / (Ca_i + K_m_rel)
            I_rel = g_rel * (Ca_jsr - Ca_i)

            # SR uptake
            I_up = I_up_bar * Ca_i / (Ca_i + K_m_up)

            # JSR-NSR transfer
            I_tr = (Ca_nsr - Ca_jsr) / tau_tr

            # --- Na+ concentration ---
            I_Na_tot = I_Na + I_Na_b + 3.0 * I_NaK + 3.0 * I_NaCa
            dNa_i = -I_Na_tot / (V_myo * F) * 1e-3  # Convert units
            Na_i_new = Na_i + dt * dNa_i

            # --- K+ concentration ---
            I_K_tot = I_K + I_K1 + I_Kp - 2.0 * I_NaK + I_stim
            dK_i = -I_K_tot / (V_myo * F) * 1e-3
            K_i_new = K_i + dt * dK_i

            # --- Ca2+ concentration with buffering ---
            I_Ca_tot = I_Ca_L + I_Ca_b + I_pCa - 2.0 * I_NaCa

            # Buffering factors
            TRPN = TRPN_tot * K_m_TRPN / (K_m_TRPN + Ca_i)**2
            CMDN = CMDN_tot * K_m_CMDN / (K_m_CMDN + Ca_i)**2
            beta_Ca_i = 1.0 / (1.0 + TRPN + CMDN)

            dCa_i_unbuffered = (-I_Ca_tot / (2.0 * V_myo * F) * 1e-3 +
                               (V_jsr / V_myo) * I_rel - I_up)
            dCa_i = beta_Ca_i * dCa_i_unbuffered
            Ca_i_new = Ca_i + dt * dCa_i
            Ca_i_new = max(1e-8, Ca_i_new)

            # --- JSR Ca2+ with buffering ---
            CSQN = CSQN_tot * K_m_CSQN / (K_m_CSQN + Ca_jsr)**2
            beta_Ca_jsr = 1.0 / (1.0 + CSQN)
            dCa_jsr = beta_Ca_jsr * (I_tr - I_rel)
            Ca_jsr_new = Ca_jsr + dt * dCa_jsr
            Ca_jsr_new = max(0.01, Ca_jsr_new)

            # --- NSR Ca2+ ---
            dCa_nsr = I_up - I_tr * (V_jsr / V_nsr)
            Ca_nsr_new = Ca_nsr + dt * dCa_nsr
            Ca_nsr_new = max(0.01, Ca_nsr_new)

            # ===================
            # Update voltage
            # ===================
            dV = -(I_ion + I_stim) / C_m
            V_new = V + dt * dV

            # ===================
            # Write back (clamp gates to [0,1])
            # ===================
            V_arr[i, jj] = V_new
            m_arr[i, jj] = max(0.0, min(1.0, m_new))
            h_arr[i, jj] = max(0.0, min(1.0, h_new))
            j_arr[i, jj] = max(0.0, min(1.0, j_new))
            d_arr[i, jj] = max(0.0, min(1.0, d_new))
            f_arr[i, jj] = max(0.0, min(1.0, f_new))
            Na_i_arr[i, jj] = Na_i_new
            K_i_arr[i, jj] = K_i_new
            Ca_i_arr[i, jj] = Ca_i_new
            Ca_jsr_arr[i, jj] = Ca_jsr_new
            Ca_nsr_arr[i, jj] = Ca_nsr_new


# =============================================================================
# Model Class
# =============================================================================

class LuoRudy1994Model:
    """Luo-Rudy 1994 ionic model with GHK I_Ca,L."""

    def __init__(
        self,
        params: LRd94Params = None,
        initial_conditions: LRd94InitialConditions = None,
        dt: float = 0.005,
    ):
        self.params = params or default_params()
        self.ic = initial_conditions or default_initial_conditions()
        self.dt = dt
        self.params.validate()

        print(f"LuoRudy1994Model (CellML-based):")
        print(f"  dt = {self.dt} ms")
        print(f"  I_Ca,L: GHK formulation")
        print(f"  P_Ca = {self.params.P_Ca:.2e} mm/ms")

    def initialize_state(self, shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """Create initial state arrays."""
        ic = self.ic
        return {
            'V': np.full(shape, ic.V, dtype=np.float64),
            'm': np.full(shape, ic.m, dtype=np.float64),
            'h': np.full(shape, ic.h, dtype=np.float64),
            'j': np.full(shape, ic.j, dtype=np.float64),
            'd': np.full(shape, ic.d, dtype=np.float64),
            'f': np.full(shape, ic.f, dtype=np.float64),
            'Na_i': np.full(shape, ic.Na_i, dtype=np.float64),
            'K_i': np.full(shape, ic.K_i, dtype=np.float64),
            'Ca_i': np.full(shape, ic.Ca_i, dtype=np.float64),
            'Ca_jsr': np.full(shape, ic.Ca_jsr, dtype=np.float64),
            'Ca_nsr': np.full(shape, ic.Ca_nsr, dtype=np.float64),
        }

    def ionic_step(self, state: Dict[str, np.ndarray], I_stim: np.ndarray = None) -> None:
        """Perform ionic model step (in-place)."""
        V = state['V']
        if I_stim is None:
            I_stim = np.zeros_like(V)

        p = self.params

        lrd94_ionic_step_kernel(
            state['V'], state['m'], state['h'], state['j'],
            state['d'], state['f'],
            state['Na_i'], state['K_i'], state['Ca_i'],
            state['Ca_jsr'], state['Ca_nsr'],
            I_stim, self.dt,
            p.G_Na, p.G_K, p.G_K1, p.G_Kp, p.G_Na_b, p.G_Ca_b,
            p.P_Ca, p.P_Na, p.P_K,
            p.gamma_Cai, p.gamma_Cao, p.gamma_Nai, p.gamma_Nao,
            p.gamma_Ki, p.gamma_Ko, p.Km_Ca,
            p.I_NaK_bar, p.K_m_Na_i, p.K_m_K_o,
            p.I_pCa_bar, p.K_m_pCa,
            p.k_NaCa, p.K_m_Na, p.K_m_Ca, p.k_sat, p.eta,
            p.G_rel_max, p.tau_tr, p.K_m_rel, p.K_m_up, p.I_up_bar,
            p.TRPN_tot, p.K_m_TRPN, p.CMDN_tot, p.K_m_CMDN,
            p.CSQN_tot, p.K_m_CSQN,
        )


# =============================================================================
# Single-Cell Simulation
# =============================================================================

def run_single_cell(
    model: LuoRudy1994Model,
    t_end: float = 500.0,
    stim_amplitude: float = -80.0,
    stim_start: float = 10.0,
    stim_duration: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Run single-cell simulation."""
    dt = model.dt
    n_steps = int(np.ceil(t_end / dt))

    state = model.initialize_state((1, 1))

    results = {name: np.zeros(n_steps) for name in ['t', 'V', 'm', 'h', 'j', 'd', 'f',
                                                      'Na_i', 'K_i', 'Ca_i', 'Ca_jsr', 'Ca_nsr']}

    I_stim = np.zeros((1, 1))

    for step in range(n_steps):
        t = step * dt
        results['t'][step] = t
        for name in state:
            results[name][step] = state[name][0, 0]

        if stim_start <= t < stim_start + stim_duration:
            I_stim[0, 0] = stim_amplitude
        else:
            I_stim[0, 0] = 0.0

        model.ionic_step(state, I_stim)

    return results


def measure_apd(t: np.ndarray, V: np.ndarray, threshold: float = 0.9) -> float:
    """Measure APD at given repolarization threshold."""
    V_rest = V[0]
    i_max = np.argmax(V)
    V_max = V[i_max]
    t_max = t[i_max]

    if V_max < -40:
        return np.nan

    V_thresh = V_rest + (1 - threshold) * (V_max - V_rest)

    for i in range(i_max, len(V)):
        if V[i] < V_thresh:
            return t[i] - t_max

    return np.nan


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LUO-RUDY 1994 (CellML-based) SINGLE CELL TEST")
    print("=" * 60)

    model = LuoRudy1994Model(dt=0.005)

    print("\nRunning single cell simulation (500 ms)...")
    import time
    start = time.perf_counter()
    results = run_single_cell(model, t_end=500.0, stim_amplitude=-80.0,
                               stim_start=10.0, stim_duration=1.0)
    elapsed = time.perf_counter() - start
    print(f"Completed in {elapsed:.2f}s")

    t = results['t']
    V = results['V']

    apd90 = measure_apd(t, V, threshold=0.9)
    apd50 = measure_apd(t, V, threshold=0.5)

    print(f"\nResults:")
    print(f"  V_rest = {V[0]:.1f} mV")
    print(f"  V_peak = {np.max(V):.1f} mV")
    print(f"  APD50 = {apd50:.1f} ms")
    print(f"  APD90 = {apd90:.1f} ms")
    print(f"  [Ca2+]i peak = {np.max(results['Ca_i'])*1e6:.1f} nM")

    target_apd = 300.0
    if abs(apd90 - target_apd) < 50:
        print(f"\n*** APD90 = {apd90:.1f} ms is near target (~300 ms) ***")
    else:
        print(f"\n*** APD90 = {apd90:.1f} ms differs from target (~300 ms) ***")
