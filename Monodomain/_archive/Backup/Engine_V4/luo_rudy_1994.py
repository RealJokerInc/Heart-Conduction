"""
Luo-Rudy 1994 Dynamic (LRd) Ionic Model
========================================

Complete implementation of the Luo-Rudy 1994 ventricular action potential model
with dynamic ion concentration tracking.

Reference:
    Luo CH, Rudy Y. "A dynamic model of the cardiac ventricular action potential.
    I. Simulations of ionic currents and concentration changes."
    Circ Res. 1994;74(6):1071-1096.

Units: mV, ms, mM, uA/cm^2, uF/cm^2
"""

import numpy as np
from typing import Dict

# Physical Constants
R = 8314.0       # Gas constant [mJ/(mol·K)]
T = 310.0        # Temperature [K] (37°C)
F = 96485.0      # Faraday constant [C/mol]
RTF = R * T / F  # ~26.7 mV

# Cell Geometry
C_m = 1.0        # Membrane capacitance [uF/cm^2]
A_cap = 1.534e-4  # Capacitive membrane area [cm^2]
V_myo = 25.84e-6  # Myoplasm volume [uL]
V_jsr = 0.16e-6   # Junctional SR volume [uL]
V_nsr = 2.1e-6    # Network SR volume [uL]

# Ion Concentrations (extracellular - fixed)
Na_o = 140.0     # [mM]
K_o = 5.4        # [mM]
Ca_o = 1.8       # [mM]

# Maximal Conductances
G_Na = 16.0       # Fast Na+ [mS/cm^2]
G_Ca_L = 0.09     # L-type Ca2+ [mS/cm^2]
G_K = 0.282       # Time-dependent K+ [mS/cm^2]
G_K1 = 0.6047     # Time-independent K+ [mS/cm^2]
G_Kp = 0.0183     # Plateau K+ [mS/cm^2]
G_Na_b = 0.001    # Background Na+ [mS/cm^2]
G_Ca_b = 0.003    # Background Ca2+ [mS/cm^2]

# Pump and exchanger parameters
I_NaK_bar = 1.5   # Na+/K+ pump max [uA/cm^2]
K_m_Na_i = 10.0   # [mM]
K_m_K_o = 1.5     # [mM]
I_pCa_bar = 1.15  # Sarcolemmal Ca2+ pump max [uA/cm^2]
K_m_pCa = 0.0005  # [mM]
k_NaCa = 2000.0   # Na+/Ca2+ exchanger scaling
K_m_Na = 87.5     # [mM]
K_m_Ca = 1.38     # [mM]
k_sat = 0.1       # Saturation factor
eta = 0.35        # Position of energy barrier

# SR parameters
K_rel = 30.0      # Ca2+ release rate from JSR [1/ms]
K_up = 0.92       # Half-saturation for SR uptake [mM]
I_up_bar = 0.00875  # Max SR uptake [mM/ms]
tau_tr = 180.0    # JSR-NSR transfer time constant [ms]

# Buffer parameters
TRPN_tot = 0.070   # Total troponin [mM]
K_m_TRPN = 0.0005  # Troponin Kd [mM]
CMDN_tot = 0.050   # Total calmodulin [mM]
K_m_CMDN = 0.00238 # Calmodulin Kd [mM]
CSQN_tot = 10.0    # Total calsequestrin [mM]
K_m_CSQN = 0.8     # Calsequestrin Kd [mM]

# Initial Conditions
V_init = -84.0
m_init = 0.0008
h_init = 0.993
j_init = 0.995
d_init = 0.0
f_init = 1.0
f_Ca_init = 1.0
X_init = 0.0
Na_i_init = 10.0
K_i_init = 145.0
Ca_i_init = 0.0001
Ca_jsr_init = 1.8
Ca_nsr_init = 1.8


def safe_exp(x, limit=50.0):
    return np.exp(np.clip(x, -limit, limit))


def alpha_m(V):
    dV = V + 47.13
    if abs(dV) < 1e-7:
        return 3.2
    return 0.32 * dV / (1.0 - safe_exp(-0.1 * dV))


def beta_m(V):
    return 0.08 * safe_exp(-V / 11.0)


def alpha_h(V):
    if V >= -40.0:
        return 0.0
    return 0.135 * safe_exp(-(V + 80.0) / 6.8)


def beta_h(V):
    if V >= -40.0:
        return 1.0 / (0.13 * (1.0 + safe_exp(-(V + 10.66) / 11.1)))
    return 3.56 * safe_exp(0.079 * V) + 3.1e5 * safe_exp(0.35 * V)


def alpha_j(V):
    if V >= -40.0:
        return 0.0
    term1 = -1.2714e5 * safe_exp(0.2444 * V)
    term2 = 3.474e-5 * safe_exp(-0.04391 * V)
    denom = 1.0 + safe_exp(0.311 * (V + 79.23))
    return (term1 - term2) * (V + 37.78) / denom


def beta_j(V):
    if V >= -40.0:
        return 0.3 * safe_exp(-2.535e-7 * V) / (1.0 + safe_exp(-0.1 * (V + 32.0)))
    return 0.1212 * safe_exp(-0.01052 * V) / (1.0 + safe_exp(-0.1378 * (V + 40.14)))


def d_inf(V):
    return 1.0 / (1.0 + safe_exp(-(V + 10.0) / 6.24))


def tau_d(V):
    dV = V + 10.0
    if abs(dV) < 1e-7:
        return 4.579 / (1.0 + safe_exp(-dV / 6.24))
    return (1.0 - safe_exp(-dV / 6.24)) / (0.035 * dV * (1.0 + safe_exp(-dV / 6.24)))


def f_inf(V):
    return 1.0 / (1.0 + safe_exp((V + 35.06) / 8.6)) + 0.6 / (1.0 + safe_exp((50.0 - V) / 20.0))


def tau_f(V):
    return 1.0 / (0.0197 * safe_exp(-(0.0337 * (V + 10.0))**2) + 0.02)


def f_Ca_inf(Ca_i):
    return 1.0 / (1.0 + (Ca_i / 0.00035)**1)


def alpha_X(V):
    dV = V + 30.0
    if abs(dV) < 1e-7:
        return 0.25
    return 0.0005 * dV / (1.0 - safe_exp(-dV / 5.0))


def beta_X(V):
    dV = V + 30.0
    if abs(dV) < 1e-7:
        return 0.35
    return 0.0007 * dV / (safe_exp(dV / 6.0) - 1.0)


def X_i(V, K_o):
    return 2.837 * (safe_exp(0.04 * (V + 77.0)) - 1.0) / ((V + 77.0 + 1e-10) * safe_exp(0.04 * (V + 35.0)))


# Reversal potentials
def calc_E_Na(Na_i):
    return RTF * np.log(Na_o / Na_i)


def calc_E_K(K_i):
    return RTF * np.log(K_o / K_i)


def calc_E_Ca(Ca_i):
    return 0.5 * RTF * np.log(Ca_o / Ca_i)


# Ionic currents
def I_Na(V, m, h, j, Na_i):
    E_Na = calc_E_Na(Na_i)
    return G_Na * m**3 * h * j * (V - E_Na)


def I_Ca_L(V, d, f, f_Ca, Ca_i):
    E_Ca = calc_E_Ca(Ca_i)
    return G_Ca_L * d * f * f_Ca * (V - E_Ca)


def I_K(V, X, K_i):
    E_K = calc_E_K(K_i)
    G_K_scaled = G_K * np.sqrt(K_o / 5.4)
    Xi = X_i(V, K_o)
    return G_K_scaled * X**2 * Xi * (V - E_K)


def I_K1(V, K_i):
    E_K = calc_E_K(K_i)
    G_K1_scaled = G_K1 * np.sqrt(K_o / 5.4)
    alpha_K1 = 1.02 / (1.0 + safe_exp(0.2385 * (V - E_K - 59.215)))
    beta_K1 = (0.49124 * safe_exp(0.08032 * (V - E_K + 5.476)) +
               safe_exp(0.06175 * (V - E_K - 594.31))) / (1.0 + safe_exp(-0.5143 * (V - E_K + 4.753)))
    K1_inf = alpha_K1 / (alpha_K1 + beta_K1)
    return G_K1_scaled * K1_inf * (V - E_K)


def I_Kp(V, K_i):
    E_K = calc_E_K(K_i)
    Kp = 1.0 / (1.0 + safe_exp((7.488 - V) / 5.98))
    return G_Kp * Kp * (V - E_K)


def I_NaCa(V, Na_i, Ca_i):
    num = k_NaCa * (Na_i**3 * Ca_o * safe_exp(eta * V * F / (R * T)) -
                    Na_o**3 * Ca_i * safe_exp((eta - 1.0) * V * F / (R * T)))
    denom = 1.0 + k_sat * safe_exp((eta - 1.0) * V * F / (R * T))
    return num / (denom * (K_m_Na**3 + Na_o**3) * (K_m_Ca + Ca_o))


def I_NaK(V, Na_i):
    sigma = (np.exp(Na_o / 67.3) - 1.0) / 7.0
    f_NaK = 1.0 / (1.0 + 0.1245 * safe_exp(-0.1 * V * F / (R * T)) +
                   0.0365 * sigma * safe_exp(-V * F / (R * T)))
    return I_NaK_bar * f_NaK * (K_o / (K_o + K_m_K_o)) * (Na_i / (Na_i + K_m_Na_i))


def I_pCa(Ca_i):
    return I_pCa_bar * Ca_i / (K_m_pCa + Ca_i)


def I_Na_b(V, Na_i):
    E_Na = calc_E_Na(Na_i)
    return G_Na_b * (V - E_Na)


def I_Ca_b(V, Ca_i):
    E_Ca = calc_E_Ca(Ca_i)
    return G_Ca_b * (V - E_Ca)


# SR fluxes
def I_rel(Ca_jsr, Ca_i, d):
    g_rel = K_rel * d
    return g_rel * (Ca_jsr - Ca_i)


def I_up(Ca_i):
    return I_up_bar * Ca_i / (Ca_i + K_up)


def I_tr(Ca_nsr, Ca_jsr):
    return (Ca_nsr - Ca_jsr) / tau_tr


# Buffering
def beta_Ca_i(Ca_i):
    TRPN = TRPN_tot * K_m_TRPN / (K_m_TRPN + Ca_i)**2
    CMDN = CMDN_tot * K_m_CMDN / (K_m_CMDN + Ca_i)**2
    return 1.0 / (1.0 + TRPN + CMDN)


def beta_Ca_jsr(Ca_jsr):
    CSQN = CSQN_tot * K_m_CSQN / (K_m_CSQN + Ca_jsr)**2
    return 1.0 / (1.0 + CSQN)


class LuoRudy1994:
    """Luo-Rudy 1994 Dynamic (LRd94) cardiac ventricular cell model."""

    def __init__(self, dt=0.005):
        self.dt = dt
        self.C_m = C_m

    def initialize_state(self) -> Dict[str, float]:
        return {
            'V': V_init, 'm': m_init, 'h': h_init, 'j': j_init,
            'd': d_init, 'f': f_init, 'f_Ca': f_Ca_init, 'X': X_init,
            'Na_i': Na_i_init, 'K_i': K_i_init, 'Ca_i': Ca_i_init,
            'Ca_jsr': Ca_jsr_init, 'Ca_nsr': Ca_nsr_init,
        }

    def compute_currents(self, state: Dict[str, float]) -> Dict[str, float]:
        V = state['V']
        m, h, j = state['m'], state['h'], state['j']
        d, f, f_Ca = state['d'], state['f'], state['f_Ca']
        X = state['X']
        Na_i, K_i, Ca_i = state['Na_i'], state['K_i'], state['Ca_i']

        currents = {
            'I_Na': I_Na(V, m, h, j, Na_i),
            'I_Ca_L': I_Ca_L(V, d, f, f_Ca, Ca_i),
            'I_K': I_K(V, X, K_i),
            'I_K1': I_K1(V, K_i),
            'I_Kp': I_Kp(V, K_i),
            'I_NaCa': I_NaCa(V, Na_i, Ca_i),
            'I_NaK': I_NaK(V, Na_i),
            'I_pCa': I_pCa(Ca_i),
            'I_Na_b': I_Na_b(V, Na_i),
            'I_Ca_b': I_Ca_b(V, Ca_i),
        }

        currents['I_ion'] = (currents['I_Na'] + currents['I_Ca_L'] +
                            currents['I_K'] + currents['I_K1'] + currents['I_Kp'] +
                            currents['I_NaCa'] + currents['I_NaK'] + currents['I_pCa'] +
                            currents['I_Na_b'] + currents['I_Ca_b'])

        # Reversal potentials
        currents['E_Na'] = calc_E_Na(Na_i)
        currents['E_K'] = calc_E_K(K_i)
        currents['E_Ca'] = calc_E_Ca(Ca_i)

        return currents

    def step(self, state: Dict[str, float], I_stim: float = 0.0) -> Dict[str, float]:
        V = state['V']
        m, h, j = state['m'], state['h'], state['j']
        d, f, f_Ca = state['d'], state['f'], state['f_Ca']
        X = state['X']
        Na_i, K_i, Ca_i = state['Na_i'], state['K_i'], state['Ca_i']
        Ca_jsr, Ca_nsr = state['Ca_jsr'], state['Ca_nsr']
        dt = self.dt

        currents = self.compute_currents(state)
        I_ion = currents['I_ion']

        # Update voltage
        dV = -(I_ion + I_stim) / C_m
        V_new = V + dt * dV

        # Update gating variables (Rush-Larsen)
        am, bm = alpha_m(V), beta_m(V)
        tau_m = 1.0 / (am + bm)
        m_inf = am * tau_m
        m_new = m_inf - (m_inf - m) * np.exp(-dt / tau_m)

        ah, bh = alpha_h(V), beta_h(V)
        if ah + bh > 1e-10:
            tau_h = 1.0 / (ah + bh)
            h_inf_val = ah * tau_h
            h_new = h_inf_val - (h_inf_val - h) * np.exp(-dt / tau_h)
        else:
            h_new = h

        aj, bj = alpha_j(V), beta_j(V)
        if aj + bj > 1e-10:
            tau_j = 1.0 / (aj + bj)
            j_inf_val = aj * tau_j
            j_new = j_inf_val - (j_inf_val - j) * np.exp(-dt / tau_j)
        else:
            j_new = j

        d_inf_val = d_inf(V)
        tau_d_val = tau_d(V)
        d_new = d_inf_val - (d_inf_val - d) * np.exp(-dt / tau_d_val)

        f_inf_val = f_inf(V)
        tau_f_val = tau_f(V)
        f_new = f_inf_val - (f_inf_val - f) * np.exp(-dt / tau_f_val)

        f_Ca_inf_val = f_Ca_inf(Ca_i)
        tau_f_Ca = 2.0
        f_Ca_new = f_Ca_inf_val - (f_Ca_inf_val - f_Ca) * np.exp(-dt / tau_f_Ca)

        aX, bX = alpha_X(V), beta_X(V)
        tau_X = 1.0 / (aX + bX)
        X_inf = aX * tau_X
        X_new = X_inf - (X_inf - X) * np.exp(-dt / tau_X)

        # Update concentrations
        I_Na_tot = currents['I_Na'] + currents['I_Na_b'] + 3.0 * currents['I_NaK'] + 3.0 * currents['I_NaCa']
        dNa_i = -I_Na_tot * A_cap / (V_myo * F)
        Na_i_new = Na_i + dt * dNa_i

        I_K_tot = currents['I_K'] + currents['I_K1'] + currents['I_Kp'] - 2.0 * currents['I_NaK']
        dK_i = -(I_K_tot + I_stim) * A_cap / (V_myo * F)
        K_i_new = K_i + dt * dK_i

        I_Ca_tot = currents['I_Ca_L'] + currents['I_Ca_b'] + currents['I_pCa'] - 2.0 * currents['I_NaCa']
        I_rel_val = I_rel(Ca_jsr, Ca_i, d)
        I_up_val = I_up(Ca_i)

        dCa_i_unbuffered = (-I_Ca_tot * A_cap / (2.0 * V_myo * F) +
                           (V_jsr / V_myo) * I_rel_val - I_up_val)
        dCa_i = beta_Ca_i(Ca_i) * dCa_i_unbuffered
        Ca_i_new = Ca_i + dt * dCa_i
        Ca_i_new = max(1e-8, Ca_i_new)

        I_tr_val = I_tr(Ca_nsr, Ca_jsr)
        dCa_jsr_unbuffered = I_tr_val - I_rel_val
        dCa_jsr = beta_Ca_jsr(Ca_jsr) * dCa_jsr_unbuffered
        Ca_jsr_new = Ca_jsr + dt * dCa_jsr
        Ca_jsr_new = max(0.01, Ca_jsr_new)

        dCa_nsr = I_up_val - I_tr_val * (V_jsr / V_nsr)
        Ca_nsr_new = Ca_nsr + dt * dCa_nsr
        Ca_nsr_new = max(0.01, Ca_nsr_new)

        return {
            'V': V_new,
            'm': np.clip(m_new, 0, 1), 'h': np.clip(h_new, 0, 1), 'j': np.clip(j_new, 0, 1),
            'd': np.clip(d_new, 0, 1), 'f': np.clip(f_new, 0, 1), 'f_Ca': np.clip(f_Ca_new, 0, 1),
            'X': np.clip(X_new, 0, 1),
            'Na_i': Na_i_new, 'K_i': K_i_new, 'Ca_i': Ca_i_new,
            'Ca_jsr': Ca_jsr_new, 'Ca_nsr': Ca_nsr_new,
        }

    def run_simulation(self, t_end: float, stim_times: list = None,
                       stim_duration: float = 1.0, stim_amplitude: float = -80.0):
        if stim_times is None:
            stim_times = [10.0]

        n_steps = int(t_end / self.dt)

        results = {
            't': np.zeros(n_steps), 'V': np.zeros(n_steps),
            'I_Na': np.zeros(n_steps), 'I_Ca_L': np.zeros(n_steps),
            'I_K': np.zeros(n_steps), 'I_K1': np.zeros(n_steps),
            'I_Kp': np.zeros(n_steps), 'I_NaCa': np.zeros(n_steps),
            'I_NaK': np.zeros(n_steps),
            'E_Na': np.zeros(n_steps), 'E_K': np.zeros(n_steps), 'E_Ca': np.zeros(n_steps),
            'Na_i': np.zeros(n_steps), 'K_i': np.zeros(n_steps), 'Ca_i': np.zeros(n_steps),
        }

        state = self.initialize_state()

        for i in range(n_steps):
            t = i * self.dt

            I_stim = 0.0
            for t_stim in stim_times:
                if t_stim <= t < t_stim + stim_duration:
                    I_stim = stim_amplitude
                    break

            results['t'][i] = t
            results['V'][i] = state['V']
            results['Na_i'][i] = state['Na_i']
            results['K_i'][i] = state['K_i']
            results['Ca_i'][i] = state['Ca_i']

            currents = self.compute_currents(state)
            results['I_Na'][i] = currents['I_Na']
            results['I_Ca_L'][i] = currents['I_Ca_L']
            results['I_K'][i] = currents['I_K']
            results['I_K1'][i] = currents['I_K1']
            results['I_Kp'][i] = currents['I_Kp']
            results['I_NaCa'][i] = currents['I_NaCa']
            results['I_NaK'][i] = currents['I_NaK']
            results['E_Na'][i] = currents['E_Na']
            results['E_K'][i] = currents['E_K']
            results['E_Ca'][i] = currents['E_Ca']

            state = self.step(state, I_stim)

        return results
