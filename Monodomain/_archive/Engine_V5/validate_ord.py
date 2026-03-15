"""
ORd Model Validation Script

Compares implementation against reference values from:
O'Hara T, Virag L, Varro A, Rudy Y. PLoS Comput Biol. 2011;7(5):e1002061.

Expected values for endocardial cells at 1 Hz (BCL=1000ms):
- APD90: ~270 ms
- Resting potential: ~-87 mV
- Peak potential: ~40 mV
- dV/dt max: ~260 V/s
- [Ca]i peak: ~0.35-0.4 µM (350-400 nM)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from ionic import ORdModel, CellType, StateIndex


def compute_apd90(t, V, threshold=-40):
    """Compute APD90 from voltage trace."""
    # Find upstroke (crossing threshold going up)
    above = V > threshold
    upstroke_idx = None
    for i in range(1, len(V)):
        if above[i] and not above[i-1]:
            upstroke_idx = i
            break

    if upstroke_idx is None:
        return None

    # Find resting potential (use end of beat when cell is at rest)
    V_rest = np.mean(V[-100:])  # Last 1 ms at dt=0.01
    V_peak = np.max(V[upstroke_idx:])

    # APD90 threshold
    V_90 = V_rest + 0.1 * (V_peak - V_rest)

    # Find repolarization to 90%
    for i in range(upstroke_idx, len(V)):
        if V[i] < V_90:
            return t[i] - t[upstroke_idx]

    return None


def compute_dvdt_max(t, V, dt):
    """Compute maximum upstroke velocity (dV/dt max)."""
    dVdt = np.diff(V) / dt
    return np.max(dVdt)


def run_validation():
    """Run validation tests."""
    print("=" * 70)
    print("O'Hara-Rudy (ORd 2011) Model Validation")
    print("=" * 70)

    # Reference values from ORd 2011 paper (Table 1, endocardial)
    ref_values = {
        'APD90': 271,      # ms
        'V_rest': -87.5,   # mV
        'V_peak': 40,      # mV (approximate)
        'dVdt_max': 260,   # V/s
        'Ca_peak': 0.35,   # µM
    }

    print("\nReference values (ORd 2011 paper, Endo, BCL=1000ms):")
    for key, val in ref_values.items():
        unit = 'ms' if 'APD' in key else 'mV' if 'V_' in key else 'V/s' if 'dVdt' in key else 'µM'
        print(f"  {key}: {val} {unit}")

    # Run simulation
    print("\n" + "-" * 70)
    print("Running simulation (10 beats at BCL=1000ms)...")

    model = ORdModel(celltype=CellType.ENDO)

    bcl = 1000.0
    n_beats = 10  # Need enough beats to reach steady state
    dt = 0.01
    t_end = n_beats * bcl

    t, y = model.simulate(
        t_span=(0, t_end),
        dt=dt,
        bcl=bcl,
        stim_duration=0.5,
        stim_amplitude=80.0
    )

    # Extract last beat
    last_beat_start = (n_beats - 1) * bcl
    last_beat_mask = (t >= last_beat_start) & (t < last_beat_start + bcl)
    t_last = t[last_beat_mask] - last_beat_start
    V_last = y[last_beat_mask, StateIndex.V]
    cai_last = y[last_beat_mask, StateIndex.cai]

    # Compute biomarkers
    V_rest = np.mean(V_last[-100:])  # Last 1 ms
    V_peak = np.max(V_last)
    APD90 = compute_apd90(t_last, V_last)
    dVdt_max = compute_dvdt_max(t_last, V_last, dt)
    Ca_peak = np.max(cai_last) * 1000  # Convert mM to µM

    print("\n" + "-" * 70)
    print("Computed biomarkers:")
    print("-" * 70)

    results = {
        'APD90': APD90,
        'V_rest': V_rest,
        'V_peak': V_peak,
        'dVdt_max': dVdt_max,
        'Ca_peak': Ca_peak,
    }

    # Compare with reference
    print(f"\n{'Biomarker':<15} {'Computed':>12} {'Reference':>12} {'Error':>12} {'Status':>10}")
    print("-" * 70)

    all_pass = True
    tolerances = {
        'APD90': 30,       # ±30 ms
        'V_rest': 3,       # ±3 mV
        'V_peak': 15,      # ±15 mV
        'dVdt_max': 100,   # ±100 V/s
        'Ca_peak': 0.2,    # ±0.2 µM
    }

    units = {
        'APD90': 'ms',
        'V_rest': 'mV',
        'V_peak': 'mV',
        'dVdt_max': 'V/s',
        'Ca_peak': 'µM',
    }

    for key in ref_values:
        computed = results[key]
        reference = ref_values[key]
        tol = tolerances[key]
        unit = units[key]

        if computed is None:
            error = "N/A"
            status = "FAIL"
            all_pass = False
        else:
            error_val = computed - reference
            error = f"{error_val:+.1f} {unit}"
            if abs(error_val) <= tol:
                status = "PASS"
            else:
                status = "FAIL"
                all_pass = False

        computed_str = f"{computed:.1f} {unit}" if computed is not None else "N/A"
        ref_str = f"{reference} {unit}"
        print(f"{key:<15} {computed_str:>12} {ref_str:>12} {error:>12} {status:>10}")

    print("-" * 70)
    print(f"\nOverall validation: {'PASS' if all_pass else 'FAIL'}")

    # Additional diagnostics
    print("\n" + "=" * 70)
    print("Additional Diagnostics")
    print("=" * 70)

    # Check concentrations at end
    print("\nFinal concentrations (should be near initial):")
    print(f"  nai: {y[-1, StateIndex.nai]:.2f} mM (initial: 7.0)")
    print(f"  ki:  {y[-1, StateIndex.ki]:.2f} mM (initial: 145.0)")
    print(f"  cai: {y[-1, StateIndex.cai]*1e6:.1f} nM (initial: 100)")

    # Check if model is at steady state
    beat_8_start = 7 * bcl
    beat_8_mask = (t >= beat_8_start) & (t < beat_8_start + bcl)
    V_beat8 = y[beat_8_mask, StateIndex.V]

    apd_beat8 = compute_apd90(t[beat_8_mask] - beat_8_start, V_beat8)
    apd_beat10 = APD90

    if apd_beat8 is not None and apd_beat10 is not None:
        apd_change = abs(apd_beat10 - apd_beat8)
        print(f"\nSteady state check:")
        print(f"  APD90 beat 8: {apd_beat8:.1f} ms")
        print(f"  APD90 beat 10: {apd_beat10:.1f} ms")
        print(f"  Change: {apd_change:.1f} ms ({'Steady state' if apd_change < 5 else 'NOT at steady state'})")

    return all_pass, results


if __name__ == '__main__':
    success, results = run_validation()
    sys.exit(0 if success else 1)
