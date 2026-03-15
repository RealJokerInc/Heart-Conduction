"""
Debug LRd94 Action Potential
============================

Comprehensive diagnostic to identify issues with the AP morphology.
Plots all ionic currents, gating variables, and ion concentrations.

Author: Generated with Claude Code
Date: 2025-12-11
"""

import numpy as np
import matplotlib.pyplot as plt

from parameters import default_params, default_initial_conditions
from luo_rudy_1994 import (
    RTF, FRT, Na_o, K_o, Ca_o, C_m, A_cap, V_myo, V_jsr, V_nsr,
    safe_exp, alpha_m, beta_m, alpha_h, beta_h, alpha_j, beta_j,
    d_inf, tau_d, f_inf, tau_f, f_Ca_inf, alpha_X, beta_X, X_i
)


def compute_all_currents(V, m, h, j, d, f, f_Ca, X, Na_i, K_i, Ca_i, Ca_jsr, params):
    """Compute all ionic currents for given state."""

    # Reversal potentials
    E_Na = RTF * np.log(Na_o / Na_i)
    E_K = RTF * np.log(K_o / K_i)
    E_Ca = 0.5 * RTF * np.log(Ca_o / max(Ca_i, 1e-10))

    # Fast Na+ current
    I_Na = params.G_Na * m**3 * h * j * (V - E_Na)

    # L-type Ca2+ current
    I_Ca_L = params.G_Ca_L * d * f * f_Ca * (V - E_Ca)

    # Time-dependent K+ current
    G_K_scaled = params.G_K * np.sqrt(K_o / 5.4)
    Xi = X_i(V)
    I_K = G_K_scaled * X**2 * Xi * (V - E_K)

    # Inward rectifier K+ current
    G_K1_scaled = params.G_K1 * np.sqrt(K_o / 5.4)
    alpha_K1 = 1.02 / (1.0 + safe_exp(0.2385 * (V - E_K - 59.215)))
    beta_K1 = (0.49124 * safe_exp(0.08032 * (V - E_K + 5.476)) +
               safe_exp(0.06175 * (V - E_K - 594.31))) / \
              (1.0 + safe_exp(-0.5143 * (V - E_K + 4.753)))
    K1_inf = alpha_K1 / (alpha_K1 + beta_K1)
    I_K1 = G_K1_scaled * K1_inf * (V - E_K)

    # Plateau K+ current
    Kp = 1.0 / (1.0 + safe_exp((7.488 - V) / 5.98))
    I_Kp = params.G_Kp * Kp * (V - E_K)

    # Na+/Ca2+ exchanger
    exp_eta = safe_exp(params.eta * V * FRT)
    exp_eta1 = safe_exp((params.eta - 1.0) * V * FRT)
    num_NaCa = params.k_NaCa * (Na_i**3 * Ca_o * exp_eta - Na_o**3 * Ca_i * exp_eta1)
    denom_NaCa = (1.0 + params.k_sat * exp_eta1) * (params.K_m_Na**3 + Na_o**3) * (params.K_m_Ca + Ca_o)
    I_NaCa = num_NaCa / denom_NaCa

    # Na+/K+ pump
    sigma = (np.exp(Na_o / 67.3) - 1.0) / 7.0
    f_NaK = 1.0 / (1.0 + 0.1245 * safe_exp(-0.1 * V * FRT) +
                  0.0365 * sigma * safe_exp(-V * FRT))
    I_NaK = params.I_NaK_bar * f_NaK * (K_o / (K_o + params.K_m_K_o)) * (Na_i / (Na_i + params.K_m_Na_i))

    # Sarcolemmal Ca2+ pump
    I_pCa = params.I_pCa_bar * Ca_i / (params.K_m_pCa + Ca_i)

    # Background currents
    I_Na_b = params.G_Na_b * (V - E_Na)
    I_Ca_b = params.G_Ca_b * (V - E_Ca)

    # SR release
    g_rel = params.K_rel * d
    I_rel = g_rel * (Ca_jsr - Ca_i)

    # SR uptake
    I_up = params.I_up_bar * Ca_i / (Ca_i + params.K_up)

    return {
        'E_Na': E_Na, 'E_K': E_K, 'E_Ca': E_Ca,
        'I_Na': I_Na, 'I_Ca_L': I_Ca_L, 'I_K': I_K, 'I_K1': I_K1, 'I_Kp': I_Kp,
        'I_NaCa': I_NaCa, 'I_NaK': I_NaK, 'I_pCa': I_pCa,
        'I_Na_b': I_Na_b, 'I_Ca_b': I_Ca_b,
        'I_rel': I_rel, 'I_up': I_up,
    }


def run_debug_simulation(t_end=500.0, dt=0.005, stim_start=10.0, stim_duration=1.0, stim_amplitude=-80.0):
    """Run simulation with full current recording."""

    params = default_params()
    ic = default_initial_conditions()

    n_steps = int(t_end / dt)

    # Initialize state
    V = ic.V
    m, h, j = ic.m, ic.h, ic.j
    d, f, f_Ca = ic.d, ic.f, ic.f_Ca
    X = ic.X
    Na_i, K_i, Ca_i = ic.Na_i, ic.K_i, ic.Ca_i
    Ca_jsr, Ca_nsr = ic.Ca_jsr, ic.Ca_nsr

    # Storage
    results = {name: np.zeros(n_steps) for name in [
        't', 'V', 'm', 'h', 'j', 'd', 'f', 'f_Ca', 'X',
        'Na_i', 'K_i', 'Ca_i', 'Ca_jsr', 'Ca_nsr',
        'E_Na', 'E_K', 'E_Ca',
        'I_Na', 'I_Ca_L', 'I_K', 'I_K1', 'I_Kp',
        'I_NaCa', 'I_NaK', 'I_pCa', 'I_Na_b', 'I_Ca_b',
        'I_rel', 'I_up', 'I_ion', 'I_stim'
    ]}

    for step in range(n_steps):
        t = step * dt

        # Compute currents
        currents = compute_all_currents(V, m, h, j, d, f, f_Ca, X, Na_i, K_i, Ca_i, Ca_jsr, params)

        I_ion = (currents['I_Na'] + currents['I_Ca_L'] + currents['I_K'] +
                 currents['I_K1'] + currents['I_Kp'] + currents['I_NaCa'] +
                 currents['I_NaK'] + currents['I_pCa'] + currents['I_Na_b'] +
                 currents['I_Ca_b'])

        # Stimulus
        I_stim = stim_amplitude if stim_start <= t < stim_start + stim_duration else 0.0

        # Record
        results['t'][step] = t
        results['V'][step] = V
        results['m'][step] = m
        results['h'][step] = h
        results['j'][step] = j
        results['d'][step] = d
        results['f'][step] = f
        results['f_Ca'][step] = f_Ca
        results['X'][step] = X
        results['Na_i'][step] = Na_i
        results['K_i'][step] = K_i
        results['Ca_i'][step] = Ca_i
        results['Ca_jsr'][step] = Ca_jsr
        results['Ca_nsr'][step] = Ca_nsr
        results['I_ion'][step] = I_ion
        results['I_stim'][step] = I_stim

        for key in ['E_Na', 'E_K', 'E_Ca', 'I_Na', 'I_Ca_L', 'I_K', 'I_K1',
                    'I_Kp', 'I_NaCa', 'I_NaK', 'I_pCa', 'I_Na_b', 'I_Ca_b',
                    'I_rel', 'I_up']:
            results[key][step] = currents[key]

        # Update gating variables (Rush-Larsen)
        am, bm = alpha_m(V), beta_m(V)
        tau_m = 1.0 / (am + bm)
        m_inf = am * tau_m
        m = m_inf - (m_inf - m) * np.exp(-dt / tau_m)

        ah, bh = alpha_h(V), beta_h(V)
        sum_h = ah + bh
        if sum_h > 1e-10:
            tau_h = 1.0 / sum_h
            h_inf = ah * tau_h
            h = h_inf - (h_inf - h) * np.exp(-dt / tau_h)

        aj, bj = alpha_j(V), beta_j(V)
        sum_j = aj + bj
        if sum_j > 1e-10:
            tau_j = 1.0 / sum_j
            j_inf = aj * tau_j
            j = j_inf - (j_inf - j) * np.exp(-dt / tau_j)

        d_inf_val = d_inf(V)
        tau_d_val = tau_d(V)
        d = d_inf_val - (d_inf_val - d) * np.exp(-dt / tau_d_val)

        f_inf_val = f_inf(V)
        tau_f_val = tau_f(V)
        f = f_inf_val - (f_inf_val - f) * np.exp(-dt / tau_f_val)

        f_Ca_inf_val = f_Ca_inf(Ca_i)
        tau_f_Ca = 2.0
        f_Ca = f_Ca_inf_val - (f_Ca_inf_val - f_Ca) * np.exp(-dt / tau_f_Ca)

        aX, bX = alpha_X(V), beta_X(V)
        tau_X = 1.0 / (aX + bX)
        X_inf = aX * tau_X
        X = X_inf - (X_inf - X) * np.exp(-dt / tau_X)

        # Update concentrations
        F = 96485.0

        I_Na_tot = currents['I_Na'] + currents['I_Na_b'] + 3.0 * currents['I_NaK'] + 3.0 * currents['I_NaCa']
        dNa_i = -I_Na_tot * A_cap / (V_myo * F)
        Na_i = Na_i + dt * dNa_i

        I_K_tot = currents['I_K'] + currents['I_K1'] + currents['I_Kp'] - 2.0 * currents['I_NaK']
        dK_i = -(I_K_tot + I_stim) * A_cap / (V_myo * F)
        K_i = K_i + dt * dK_i

        # Ca2+ with buffering
        I_Ca_tot = currents['I_Ca_L'] + currents['I_Ca_b'] + currents['I_pCa'] - 2.0 * currents['I_NaCa']
        I_rel_val = currents['I_rel']
        I_up_val = currents['I_up']

        TRPN = params.TRPN_tot * params.K_m_TRPN / (params.K_m_TRPN + Ca_i)**2
        CMDN = params.CMDN_tot * params.K_m_CMDN / (params.K_m_CMDN + Ca_i)**2
        beta_Ca_i = 1.0 / (1.0 + TRPN + CMDN)

        dCa_i_unbuffered = (-I_Ca_tot * A_cap / (2.0 * V_myo * F) +
                           (V_jsr / V_myo) * I_rel_val - I_up_val)
        dCa_i = beta_Ca_i * dCa_i_unbuffered
        Ca_i = max(1e-8, Ca_i + dt * dCa_i)

        I_tr = (Ca_nsr - Ca_jsr) / params.tau_tr

        CSQN = params.CSQN_tot * params.K_m_CSQN / (params.K_m_CSQN + Ca_jsr)**2
        beta_Ca_jsr = 1.0 / (1.0 + CSQN)
        dCa_jsr = beta_Ca_jsr * (I_tr - I_rel_val)
        Ca_jsr = max(0.01, Ca_jsr + dt * dCa_jsr)

        dCa_nsr = I_up_val - I_tr * (V_jsr / V_nsr)
        Ca_nsr = max(0.01, Ca_nsr + dt * dCa_nsr)

        # Update voltage
        dV = -(I_ion + I_stim) / C_m
        V = V + dt * dV

        # Clamp gates
        m = max(0, min(1, m))
        h = max(0, min(1, h))
        j = max(0, min(1, j))
        d = max(0, min(1, d))
        f = max(0, min(1, f))
        f_Ca = max(0, min(1, f_Ca))
        X = max(0, min(1, X))

    return results


def plot_debug_results(results):
    """Create comprehensive debug plots."""

    t = results['t']

    fig = plt.figure(figsize=(16, 20))

    # 1. Voltage
    ax1 = fig.add_subplot(6, 2, 1)
    ax1.plot(t, results['V'], 'b-', linewidth=1.5)
    ax1.set_ylabel('V [mV]')
    ax1.set_title('Membrane Voltage')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 500)

    # 2. Na+ current
    ax2 = fig.add_subplot(6, 2, 2)
    ax2.plot(t, results['I_Na'], 'r-', linewidth=1)
    ax2.set_ylabel('I_Na [uA/cm2]')
    ax2.set_title('Fast Na+ Current')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 50)  # Zoom to upstroke

    # 3. Ca2+ current
    ax3 = fig.add_subplot(6, 2, 3)
    ax3.plot(t, results['I_Ca_L'], 'm-', linewidth=1)
    ax3.set_ylabel('I_Ca,L [uA/cm2]')
    ax3.set_title('L-type Ca2+ Current')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 500)

    # 4. K+ currents
    ax4 = fig.add_subplot(6, 2, 4)
    ax4.plot(t, results['I_K'], 'g-', linewidth=1, label='I_K')
    ax4.plot(t, results['I_K1'], 'c-', linewidth=1, label='I_K1')
    ax4.plot(t, results['I_Kp'], 'y-', linewidth=1, label='I_Kp')
    ax4.set_ylabel('I [uA/cm2]')
    ax4.set_title('K+ Currents')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 500)

    # 5. Exchanger and pump currents
    ax5 = fig.add_subplot(6, 2, 5)
    ax5.plot(t, results['I_NaCa'], 'r-', linewidth=1, label='I_NaCa')
    ax5.plot(t, results['I_NaK'], 'b-', linewidth=1, label='I_NaK')
    ax5.plot(t, results['I_pCa'], 'm-', linewidth=1, label='I_pCa')
    ax5.set_ylabel('I [uA/cm2]')
    ax5.set_title('Exchanger & Pump Currents')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, 500)

    # 6. Total ionic current
    ax6 = fig.add_subplot(6, 2, 6)
    ax6.plot(t, results['I_ion'], 'k-', linewidth=1)
    ax6.set_ylabel('I_ion [uA/cm2]')
    ax6.set_title('Total Ionic Current')
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0, 500)

    # 7. Na+ gates
    ax7 = fig.add_subplot(6, 2, 7)
    ax7.plot(t, results['m'], 'r-', linewidth=1, label='m')
    ax7.plot(t, results['h'], 'g-', linewidth=1, label='h')
    ax7.plot(t, results['j'], 'b-', linewidth=1, label='j')
    ax7.set_ylabel('Gate')
    ax7.set_title('Na+ Channel Gates')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim(0, 100)

    # 8. Ca2+ gates
    ax8 = fig.add_subplot(6, 2, 8)
    ax8.plot(t, results['d'], 'r-', linewidth=1, label='d')
    ax8.plot(t, results['f'], 'g-', linewidth=1, label='f')
    ax8.plot(t, results['f_Ca'], 'b-', linewidth=1, label='f_Ca')
    ax8.set_ylabel('Gate')
    ax8.set_title('Ca2+ Channel Gates')
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3)
    ax8.set_xlim(0, 500)

    # 9. K+ gate
    ax9 = fig.add_subplot(6, 2, 9)
    ax9.plot(t, results['X'], 'g-', linewidth=1)
    ax9.set_ylabel('X')
    ax9.set_title('K+ Channel Gate (X)')
    ax9.grid(True, alpha=0.3)
    ax9.set_xlim(0, 500)

    # 10. Reversal potentials
    ax10 = fig.add_subplot(6, 2, 10)
    ax10.plot(t, results['E_Na'], 'r-', linewidth=1, label='E_Na')
    ax10.plot(t, results['E_K'], 'g-', linewidth=1, label='E_K')
    ax10.plot(t, results['E_Ca'], 'm-', linewidth=1, label='E_Ca')
    ax10.plot(t, results['V'], 'b-', linewidth=1, alpha=0.5, label='V')
    ax10.set_ylabel('E [mV]')
    ax10.set_title('Reversal Potentials')
    ax10.legend(fontsize=8)
    ax10.grid(True, alpha=0.3)
    ax10.set_xlim(0, 500)

    # 11. Ion concentrations
    ax11 = fig.add_subplot(6, 2, 11)
    ax11.plot(t, results['Na_i'], 'r-', linewidth=1, label='[Na+]i')
    ax11.plot(t, results['K_i'], 'g-', linewidth=1, label='[K+]i')
    ax11.set_ylabel('[ion] [mM]')
    ax11.set_title('Na+ and K+ Concentrations')
    ax11.legend(fontsize=8)
    ax11.grid(True, alpha=0.3)
    ax11.set_xlim(0, 500)

    # 12. Ca2+ concentrations
    ax12 = fig.add_subplot(6, 2, 12)
    ax12.plot(t, results['Ca_i'] * 1e6, 'r-', linewidth=1, label='[Ca2+]i (nM)')
    ax12.set_ylabel('[Ca2+]i [nM]')
    ax12.set_title('Intracellular Ca2+')
    ax12.grid(True, alpha=0.3)
    ax12.set_xlim(0, 500)

    # Add secondary y-axis for SR calcium
    ax12b = ax12.twinx()
    ax12b.plot(t, results['Ca_jsr'], 'm-', linewidth=1, label='[Ca2+]jsr')
    ax12b.plot(t, results['Ca_nsr'], 'c-', linewidth=1, label='[Ca2+]nsr')
    ax12b.set_ylabel('[Ca2+]SR [mM]')
    ax12b.legend(loc='right', fontsize=8)

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]:
        ax.set_xlabel('Time [ms]')

    plt.tight_layout()
    plt.savefig('/Users/lemon/Documents/Python/Heart Conduction/Monodomain/Engine_V4/images/debug_ap.png',
                dpi=150, bbox_inches='tight')
    print("Debug plot saved: images/debug_ap.png")

    return fig


def analyze_results(results):
    """Print analysis of AP characteristics."""
    t = results['t']
    V = results['V']

    # Find peak
    i_peak = np.argmax(V)
    V_peak = V[i_peak]
    t_peak = t[i_peak]

    # Resting potential
    V_rest = V[0]

    # dV/dt max
    dV = np.diff(V)
    dt = np.diff(t)
    dV_dt = dV / dt
    dV_dt_max = np.max(dV_dt)

    # APD measurements
    V_90 = V_rest + 0.1 * (V_peak - V_rest)
    V_50 = V_rest + 0.5 * (V_peak - V_rest)

    apd90 = None
    apd50 = None
    for i in range(i_peak, len(V)):
        if apd50 is None and V[i] < V_50:
            apd50 = t[i] - t_peak
        if apd90 is None and V[i] < V_90:
            apd90 = t[i] - t_peak
            break

    # Peak currents
    I_Na_peak = np.min(results['I_Na'])  # Most negative (inward)
    I_Ca_peak = np.min(results['I_Ca_L'])
    I_K_peak = np.max(results['I_K'])
    I_K1_peak = np.max(results['I_K1'])

    # Ca2+ transient
    Ca_i_rest = results['Ca_i'][0] * 1e6  # nM
    Ca_i_peak = np.max(results['Ca_i']) * 1e6  # nM

    print("\n" + "="*60)
    print("ACTION POTENTIAL ANALYSIS")
    print("="*60)
    print(f"\nVoltage:")
    print(f"  V_rest      = {V_rest:.1f} mV")
    print(f"  V_peak      = {V_peak:.1f} mV")
    print(f"  Amplitude   = {V_peak - V_rest:.1f} mV")
    print(f"  dV/dt_max   = {dV_dt_max:.1f} mV/ms")

    print(f"\nAPD:")
    print(f"  APD50       = {apd50:.1f} ms" if apd50 else "  APD50       = N/A")
    print(f"  APD90       = {apd90:.1f} ms" if apd90 else "  APD90       = N/A")

    print(f"\nPeak Currents:")
    print(f"  I_Na (peak) = {I_Na_peak:.1f} uA/cm2")
    print(f"  I_Ca,L (peak) = {I_Ca_peak:.2f} uA/cm2")
    print(f"  I_K (peak)  = {I_K_peak:.2f} uA/cm2")
    print(f"  I_K1 (peak) = {I_K1_peak:.2f} uA/cm2")

    print(f"\nCa2+ Transient:")
    print(f"  [Ca2+]i rest = {Ca_i_rest:.1f} nM")
    print(f"  [Ca2+]i peak = {Ca_i_peak:.1f} nM")

    print(f"\nReversal Potentials (at rest):")
    print(f"  E_Na = {results['E_Na'][0]:.1f} mV")
    print(f"  E_K  = {results['E_K'][0]:.1f} mV")
    print(f"  E_Ca = {results['E_Ca'][0]:.1f} mV")

    # Check for issues
    print("\n" + "-"*60)
    print("DIAGNOSTIC CHECKS:")
    print("-"*60)

    issues = []

    # Expected values from literature
    if V_rest > -80 or V_rest < -90:
        issues.append(f"V_rest ({V_rest:.1f} mV) outside expected range [-90, -80]")

    if V_peak < 20 or V_peak > 50:
        issues.append(f"V_peak ({V_peak:.1f} mV) outside expected range [20, 50]")

    if dV_dt_max < 100 or dV_dt_max > 500:
        issues.append(f"dV/dt_max ({dV_dt_max:.1f}) outside expected range [100, 500]")

    if apd90 and (apd90 < 200 or apd90 > 350):
        issues.append(f"APD90 ({apd90:.1f} ms) outside expected range [200, 350]")

    if I_Na_peak > -100:
        issues.append(f"I_Na peak ({I_Na_peak:.1f}) seems too small (expected < -100)")

    if Ca_i_peak > 2000:
        issues.append(f"[Ca2+]i peak ({Ca_i_peak:.0f} nM) seems too high (expected < 1500)")

    if len(issues) == 0:
        print("  All parameters within expected ranges!")
    else:
        for issue in issues:
            print(f"  WARNING: {issue}")

    print("="*60)


if __name__ == "__main__":
    print("Running LRd94 debug simulation...")
    results = run_debug_simulation(t_end=500.0)
    analyze_results(results)
    fig = plot_debug_results(results)
    plt.show()
