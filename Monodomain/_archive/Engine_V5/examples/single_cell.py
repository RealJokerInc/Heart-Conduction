"""
ORd Single Cell Simulation

Basic example demonstrating the O'Hara-Rudy model.

Reference: O'Hara T, Virag L, Varro A, Rudy Y. PLoS Comput Biol. 2011;7(5):e1002061.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def run_simulation():
    """Run a basic ORd simulation."""
    from ionic import ORdModel, CellType, StateIndex

    print("=" * 60)
    print("O'Hara-Rudy (ORd 2011) Single Cell Simulation")
    print("=" * 60)

    # Create model for endocardial cell
    model = ORdModel(celltype=CellType.ENDO)

    # Simulation parameters
    bcl = 1000.0    # Basic cycle length (ms)
    n_beats = 5     # Number of beats
    dt = 0.01       # Time step (ms)
    t_end = n_beats * bcl

    print(f"\nSimulation parameters:")
    print(f"  Cell type: Endocardial")
    print(f"  BCL: {bcl} ms")
    print(f"  Beats: {n_beats}")
    print(f"  dt: {dt} ms")
    print(f"  Total time: {t_end} ms")

    # Run simulation
    print(f"\nSimulating...")
    t, y = model.simulate(
        t_span=(0, t_end),
        dt=dt,
        bcl=bcl,
        stim_duration=0.5,
        stim_amplitude=80.0
    )
    print("Done!")

    # Extract results
    V = y[:, StateIndex.V]
    cai = y[:, StateIndex.cai]

    # Basic measurements from last beat
    last_beat_start = (n_beats - 1) * bcl
    last_beat_mask = t >= last_beat_start

    V_last = V[last_beat_mask]
    V_rest = np.mean(V_last[-100:])  # Last 1 ms
    V_peak = np.max(V_last)

    print(f"\nResults (last beat):")
    print(f"  Resting potential: {V_rest:.1f} mV")
    print(f"  Peak potential: {V_peak:.1f} mV")
    print(f"  AP amplitude: {V_peak - V_rest:.1f} mV")

    # Try to plot if matplotlib available
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        axes[0].plot(t, V, 'b-', linewidth=0.5)
        axes[0].set_ylabel('V (mV)')
        axes[0].set_title(f'ORd Action Potentials (Endo, BCL={bcl} ms)')

        axes[1].plot(t, cai * 1e6, 'g-', linewidth=0.5)
        axes[1].set_xlabel('Time (ms)')
        axes[1].set_ylabel('[Ca]i (nM)')
        axes[1].set_title('Calcium Transient')

        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), 'ord_single_cell.png'), dpi=150)
        print(f"\nPlot saved to examples/ord_single_cell.png")
        plt.show()

    except ImportError:
        print("\nNote: matplotlib not available for plotting")

    return t, y


if __name__ == '__main__':
    run_simulation()
