"""
Single Cell Action Potential Plot (LRd94)
=========================================

Displays a single action potential from the LRd94 model with:
- Membrane voltage (V)
- Sodium reversal potential (E_Na)
- Potassium reversal potential (E_K)
- Calcium reversal potential (E_Ca)

No diffusion - pure ionic model.

Author: Generated with Claude Code
Date: 2025-12-11
"""

import numpy as np
import matplotlib.pyplot as plt

from luo_rudy_1994 import LuoRudy1994Model, RTF, Na_o, K_o, Ca_o


def run_single_cell_with_reversal_potentials(
    model: LuoRudy1994Model,
    t_end: float = 500.0,
    stim_amplitude: float = -80.0,
    stim_start: float = 10.0,
    stim_duration: float = 1.0,
):
    """
    Run single-cell simulation and record reversal potentials.

    Parameters
    ----------
    model : LuoRudy1994Model
        Configured model
    t_end : float
        Simulation duration [ms]
    stim_amplitude : float
        Stimulus amplitude [uA/cm^2]
    stim_start : float
        Stimulus start time [ms]
    stim_duration : float
        Stimulus duration [ms]

    Returns
    -------
    results : dict
        Time traces for V, E_Na, E_K, E_Ca
    """
    dt = model.dt
    n_steps = int(np.ceil(t_end / dt))

    # Single cell = 1x1 grid
    state = model.initialize_state((1, 1))

    # Storage
    results = {
        't': np.zeros(n_steps),
        'V': np.zeros(n_steps),
        'E_Na': np.zeros(n_steps),
        'E_K': np.zeros(n_steps),
        'E_Ca': np.zeros(n_steps),
        'Na_i': np.zeros(n_steps),
        'K_i': np.zeros(n_steps),
        'Ca_i': np.zeros(n_steps),
    }

    I_stim = np.zeros((1, 1))

    for step in range(n_steps):
        t = step * dt

        # Get current state
        V = state['V'][0, 0]
        Na_i = state['Na_i'][0, 0]
        K_i = state['K_i'][0, 0]
        Ca_i = state['Ca_i'][0, 0]

        # Calculate reversal potentials (Nernst equation)
        E_Na = RTF * np.log(Na_o / Na_i)
        E_K = RTF * np.log(K_o / K_i)
        E_Ca = 0.5 * RTF * np.log(Ca_o / max(Ca_i, 1e-10))

        # Record
        results['t'][step] = t
        results['V'][step] = V
        results['E_Na'][step] = E_Na
        results['E_K'][step] = E_K
        results['E_Ca'][step] = E_Ca
        results['Na_i'][step] = Na_i
        results['K_i'][step] = K_i
        results['Ca_i'][step] = Ca_i

        # Apply stimulus
        if stim_start <= t < stim_start + stim_duration:
            I_stim[0, 0] = stim_amplitude
        else:
            I_stim[0, 0] = 0.0

        # Ionic step
        model.ionic_step(state, I_stim)

    return results


def measure_apd(t, V, threshold=0.9, V_rest=-84.0):
    """Measure APD at given repolarization threshold."""
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


def plot_single_ap():
    """Run simulation and create plot."""
    print("=" * 60)
    print("SINGLE CELL ACTION POTENTIAL (LRd94)")
    print("=" * 60)

    # Create model
    model = LuoRudy1994Model(dt=0.005)

    # Run simulation
    print("\nRunning 500 ms single cell simulation...")
    results = run_single_cell_with_reversal_potentials(
        model,
        t_end=500.0,
        stim_amplitude=-80.0,
        stim_start=10.0,
        stim_duration=1.0,
    )

    t = results['t']
    V = results['V']
    E_Na = results['E_Na']
    E_K = results['E_K']
    E_Ca = results['E_Ca']

    # Measure APD
    apd90 = measure_apd(t, V, threshold=0.9)
    apd50 = measure_apd(t, V, threshold=0.5)

    print(f"\nResults:")
    print(f"  V_rest = {V[0]:.1f} mV")
    print(f"  V_peak = {np.max(V):.1f} mV")
    print(f"  APD50 = {apd50:.1f} ms")
    print(f"  APD90 = {apd90:.1f} ms")
    print(f"\nReversal Potentials (at rest):")
    print(f"  E_Na = {E_Na[0]:.1f} mV")
    print(f"  E_K = {E_K[0]:.1f} mV")
    print(f"  E_Ca = {E_Ca[0]:.1f} mV")

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Top panel: Membrane voltage ---
    ax1 = axes[0]
    ax1.plot(t, V, 'b-', linewidth=2, label='V (membrane)')
    ax1.axhline(y=V[0], color='gray', linestyle='--', alpha=0.5, label=f'V_rest = {V[0]:.0f} mV')
    ax1.set_ylabel('Voltage [mV]', fontsize=12)
    ax1.set_ylim(-100, 80)
    ax1.set_title(f'Luo-Rudy 1994 Single Cell Action Potential\n'
                  f'APD90 = {apd90:.0f} ms, APD50 = {apd50:.0f} ms', fontsize=14)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Mark APD90
    i_peak = np.argmax(V)
    t_peak = t[i_peak]
    V_thresh_90 = V[0] + 0.1 * (np.max(V) - V[0])
    ax1.axhline(y=V_thresh_90, color='red', linestyle=':', alpha=0.5, label='90% repol')
    ax1.annotate(f'APD90 = {apd90:.0f} ms',
                xy=(t_peak + apd90, V_thresh_90),
                xytext=(t_peak + apd90 + 50, V_thresh_90 + 30),
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))

    # --- Bottom panel: Reversal potentials ---
    ax2 = axes[1]
    ax2.plot(t, E_Na, 'r-', linewidth=1.5, label=f'E_Na (Na+ reversal)')
    ax2.plot(t, E_K, 'g-', linewidth=1.5, label=f'E_K (K+ reversal)')
    ax2.plot(t, E_Ca, 'm-', linewidth=1.5, label=f'E_Ca (Ca2+ reversal)')
    ax2.plot(t, V, 'b-', linewidth=1.5, alpha=0.5, label='V (membrane)')

    ax2.set_xlabel('Time [ms]', fontsize=12)
    ax2.set_ylabel('Potential [mV]', fontsize=12)
    ax2.set_ylim(-120, 180)
    ax2.set_title('Membrane Voltage and Reversal Potentials', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Add annotations for reversal potentials at rest
    ax2.annotate(f'E_Na = {E_Na[0]:.0f} mV', xy=(0, E_Na[0]), xytext=(50, E_Na[0]+10),
                fontsize=9, color='red')
    ax2.annotate(f'E_K = {E_K[0]:.0f} mV', xy=(0, E_K[0]), xytext=(50, E_K[0]-15),
                fontsize=9, color='green')
    ax2.annotate(f'E_Ca = {E_Ca[0]:.0f} mV', xy=(0, E_Ca[0]), xytext=(50, E_Ca[0]+10),
                fontsize=9, color='purple')

    plt.tight_layout()

    # Save figure
    output_path = '/Users/lemon/Documents/Python/Heart Conduction/Monodomain/Engine_V4/images/single_ap_lrd94.png'
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {output_path}")

    plt.show()

    print("\nDone!")


if __name__ == "__main__":
    plot_single_ap()
