"""
Single Cell LRd94 Test
======================

Tests the LRd94 model using the actual Numba kernel for a single cell.
"""

import numpy as np
import matplotlib.pyplot as plt
from parameters import default_params, default_initial_conditions


def run_single_cell(t_end=600.0, dt=0.005, stim_start=10.0, stim_duration=1.0, stim_amplitude=-80.0):
    """Run single cell simulation using the Numba kernel."""

    from luo_rudy_1994 import lrd94_step_2d

    # Create 1x1 arrays (single cell)
    ic = default_initial_conditions()

    V = np.array([[ic.V]], dtype=np.float64)
    m = np.array([[ic.m]], dtype=np.float64)
    h = np.array([[ic.h]], dtype=np.float64)
    j = np.array([[ic.j]], dtype=np.float64)
    d = np.array([[ic.d]], dtype=np.float64)
    f = np.array([[ic.f]], dtype=np.float64)
    Na_i = np.array([[ic.Na_i]], dtype=np.float64)
    K_i = np.array([[ic.K_i]], dtype=np.float64)
    Ca_i = np.array([[ic.Ca_i]], dtype=np.float64)
    Ca_jsr = np.array([[ic.Ca_jsr]], dtype=np.float64)
    Ca_nsr = np.array([[ic.Ca_nsr]], dtype=np.float64)

    I_stim = np.array([[0.0]], dtype=np.float64)

    n_steps = int(t_end / dt)

    # Storage for key variables
    t_arr = np.zeros(n_steps)
    V_arr = np.zeros(n_steps)
    Ca_i_arr = np.zeros(n_steps)
    Ca_jsr_arr = np.zeros(n_steps)
    d_arr = np.zeros(n_steps)
    f_arr = np.zeros(n_steps)
    Na_i_arr = np.zeros(n_steps)
    K_i_arr = np.zeros(n_steps)

    print(f"Running {n_steps} steps...")

    for step in range(n_steps):
        t = step * dt

        # Apply stimulus
        if stim_start <= t < stim_start + stim_duration:
            I_stim[0, 0] = stim_amplitude
        else:
            I_stim[0, 0] = 0.0

        # Store
        t_arr[step] = t
        V_arr[step] = V[0, 0]
        Ca_i_arr[step] = Ca_i[0, 0]
        Ca_jsr_arr[step] = Ca_jsr[0, 0]
        d_arr[step] = d[0, 0]
        f_arr[step] = f[0, 0]
        Na_i_arr[step] = Na_i[0, 0]
        K_i_arr[step] = K_i[0, 0]

        # Step
        lrd94_step_2d(V, m, h, j, d, f, Na_i, K_i, Ca_i, Ca_jsr, Ca_nsr, I_stim, dt)

        if step % 20000 == 0:
            print(f"  t={t:.0f}ms, V={V[0,0]:.1f}mV, Ca_i={Ca_i[0,0]*1e6:.0f}nM")

    return {
        't': t_arr, 'V': V_arr, 'Ca_i': Ca_i_arr, 'Ca_jsr': Ca_jsr_arr,
        'd': d_arr, 'f': f_arr, 'Na_i': Na_i_arr, 'K_i': K_i_arr
    }


def analyze_ap(results):
    """Analyze action potential characteristics."""
    t = results['t']
    V = results['V']
    Ca_i = results['Ca_i']

    # Find V_rest and V_peak
    V_rest = V[0]
    V_peak = np.max(V)

    # Find APD90 and APD50
    V_90 = V_rest + 0.1 * (V_peak - V_rest)  # 90% repolarization
    V_50 = V_rest + 0.5 * (V_peak - V_rest)  # 50% repolarization

    # Find when V first crosses threshold (upstroke)
    upstroke_idx = np.argmax(V > -40)

    # Find APD90 (time from upstroke to 90% repolarization)
    APD90 = np.nan
    APD50 = np.nan

    if upstroke_idx > 0:
        # Search after upstroke for repolarization
        for i in range(upstroke_idx + 100, len(V)):
            if V[i] < V_90 and APD90 != APD90:  # First crossing
                APD90 = t[i] - t[upstroke_idx]
                break

        for i in range(upstroke_idx + 100, len(V)):
            if V[i] < V_50 and APD50 != APD50:
                APD50 = t[i] - t[upstroke_idx]
                break

    # Ca transient peak
    Ca_peak = np.max(Ca_i)

    print("\n" + "=" * 50)
    print("Action Potential Analysis")
    print("=" * 50)
    print(f"V_rest  = {V_rest:.1f} mV")
    print(f"V_peak  = {V_peak:.1f} mV")
    print(f"APD50   = {APD50:.1f} ms" if not np.isnan(APD50) else "APD50   = DID NOT REPOLARIZE")
    print(f"APD90   = {APD90:.1f} ms" if not np.isnan(APD90) else "APD90   = DID NOT REPOLARIZE")
    print(f"[Ca]i peak = {Ca_peak * 1e6:.0f} nM")
    print("=" * 50)

    return {'V_rest': V_rest, 'V_peak': V_peak, 'APD50': APD50, 'APD90': APD90, 'Ca_peak': Ca_peak}


def plot_results(results, save_path='images/test_ap.png'):
    """Plot simulation results."""

    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    t = results['t']

    # Voltage
    axes[0, 0].plot(t, results['V'], 'b-', linewidth=1)
    axes[0, 0].set_ylabel('V (mV)')
    axes[0, 0].set_title('Membrane Potential')
    axes[0, 0].grid(True, alpha=0.3)

    # Ca_i
    axes[0, 1].plot(t, results['Ca_i'] * 1e6, 'r-', linewidth=1)
    axes[0, 1].set_ylabel('[Ca]i (nM)')
    axes[0, 1].set_title('Intracellular Calcium')
    axes[0, 1].grid(True, alpha=0.3)

    # d gate (L-type activation)
    axes[1, 0].plot(t, results['d'], 'g-', linewidth=1)
    axes[1, 0].set_ylabel('d')
    axes[1, 0].set_title('L-type Ca activation')
    axes[1, 0].grid(True, alpha=0.3)

    # f gate (L-type inactivation)
    axes[1, 1].plot(t, results['f'], 'm-', linewidth=1)
    axes[1, 1].set_ylabel('f')
    axes[1, 1].set_title('L-type Ca inactivation')
    axes[1, 1].grid(True, alpha=0.3)

    # Ca_jsr
    axes[2, 0].plot(t, results['Ca_jsr'], 'orange', linewidth=1)
    axes[2, 0].set_ylabel('[Ca]_JSR (mM)')
    axes[2, 0].set_title('JSR Calcium')
    axes[2, 0].grid(True, alpha=0.3)

    # Na_i
    axes[2, 1].plot(t, results['Na_i'], 'c-', linewidth=1)
    axes[2, 1].set_ylabel('[Na]i (mM)')
    axes[2, 1].set_title('Intracellular Sodium')
    axes[2, 1].grid(True, alpha=0.3)

    # K_i
    axes[3, 0].plot(t, results['K_i'], 'brown', linewidth=1)
    axes[3, 0].set_ylabel('[K]i (mM)')
    axes[3, 0].set_xlabel('Time (ms)')
    axes[3, 0].set_title('Intracellular Potassium')
    axes[3, 0].grid(True, alpha=0.3)

    # V zoom (first 100 ms)
    axes[3, 1].plot(t, results['V'], 'b-', linewidth=1)
    axes[3, 1].set_xlim(0, 100)
    axes[3, 1].set_ylabel('V (mV)')
    axes[3, 1].set_xlabel('Time (ms)')
    axes[3, 1].set_title('Upstroke Detail')
    axes[3, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    print("LRd94 Single Cell Test")
    print("=" * 50)

    results = run_single_cell(t_end=600.0, dt=0.005)
    analysis = analyze_ap(results)
    plot_results(results)
