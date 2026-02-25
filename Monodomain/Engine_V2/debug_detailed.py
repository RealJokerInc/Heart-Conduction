"""
Detailed V1 vs V2 Debug
=======================

Run both simulations side-by-side with identical parameters.

Author: Generated with Claude Code
Date: 2025-12-09
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/lemon/Documents/Python/Heart Conduction/Engine_V1')

# Import both versions
from simulate_infarct import InfarctSimulation as SimV1
from simulate_infarct_v2 import InfarctSimulationV2 as SimV2

print("=" * 70)
print("DETAILED V1 vs V2 SIDE-BY-SIDE")
print("=" * 70)

# Create both simulations with IDENTICAL parameters
params = dict(
    domain_size=80.0,
    resolution=0.5,
    infarct_radius=10.0,
    D_parallel=1.0,
    D_perp=0.5,
    T_scale=10.0
)

print("\nCreating V1 simulation...")
sim_v1 = SimV1(**params)

print("\nCreating V2 simulation...")
sim_v2 = SimV2(**params)

# Create IDENTICAL stimulus
stim_params = dict(
    amplitude=30.0,
    duration=2.0,
    start_time=5.0
)

print(f"\nStimulus parameters:")
print(f"  Amplitude: {stim_params['amplitude']} mV")
print(f"  Duration: {stim_params['duration']} ms")
print(f"  Start: {stim_params['start_time']} ms")

stim_v1 = sim_v1.create_stimulus(**stim_params)
stim_v2 = sim_v2.create_stimulus(**stim_params)

# Verify stimulus functions are identical
print("\nVerifying stimulus functions match...")
for t in [0.0, 4.9, 5.0, 5.5, 7.0]:
    I1 = stim_v1(t)
    I2 = stim_v2(t)
    diff = np.max(np.abs(I1 - I2))
    print(f"  t={t:.1f}ms: max|I_v1 - I_v2| = {diff:.6e}")

if np.allclose(stim_v1(5.5), stim_v2(5.5)):
    print("  ✓ Stimulus functions match!")
else:
    print("  ✗ WARNING: Stimulus functions differ!")

# Run simulations
print("\n" + "=" * 70)
print("RUNNING SIMULATIONS")
print("=" * 70)

test_duration = 50.0  # ms
dt = 0.01
save_every_ms = 5.0

print(f"\nV1 simulation (50ms)...")
times_v1, V_hist_v1 = sim_v1.simulate(
    t_end=test_duration,
    dt=dt,
    stim_func=stim_v1,
    save_every_ms=save_every_ms,
    verbose=False
)

print(f"V2 simulation (50ms)...")
times_v2, V_hist_v2 = sim_v2.simulate(
    t_end=test_duration,
    dt=dt,
    stim_func=stim_v2,
    save_every_ms=save_every_ms,
    verbose=False
)

# Compare results
print("\n" + "=" * 70)
print("RESULTS COMPARISON")
print("=" * 70)

print(f"\n{'Time (ms)':>10} | {'V1 V_max':>10} | {'V2 V_max':>10} | {'Difference':>12} | Status")
print("-" * 70)

for i, (t1, t2) in enumerate(zip(times_v1, times_v2)):
    V1_max = np.max(V_hist_v1[i])
    V2_max = np.max(V_hist_v2[i])
    diff = abs(V1_max - V2_max)

    if diff < 1e-6:
        status = "✓"
    elif diff < 0.01:
        status = "~"
    else:
        status = "✗"

    print(f"{t1:>10.1f} | {V1_max:>10.6f} | {V2_max:>10.6f} | {diff:>12.6e} | {status}")

# Final diagnosis
print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)

V1_final = np.max(V_hist_v1[-1])
V2_final = np.max(V_hist_v2[-1])

print(f"\nFinal V_max at t={test_duration}ms:")
print(f"  V1: {V1_final:.6f}")
print(f"  V2: {V2_final:.6f}")
print(f"  Difference: {abs(V1_final - V2_final):.6e}")

if V1_final > 0.5 and V2_final > 0.5:
    print("\n✓ Both simulations propagate successfully!")
elif V1_final > 0.5 and V2_final < 0.1:
    print("\n✗ V2 FAILS while V1 works!")
    print("  → BUG IS IN V2 IMPLEMENTATION")
elif V2_final > 0.5 and V1_final < 0.1:
    print("\n✗ V1 FAILS while V2 works!")
    print("  → BUG IS IN V1 IMPLEMENTATION")
else:
    print("\n✗ Both simulations fail!")
    print("  → Check stimulus parameters")

# Check where they first diverge
print("\n" + "=" * 70)
print("DIVERGENCE ANALYSIS")
print("=" * 70)

divergence_threshold = 0.01  # 1% difference
first_divergence_idx = None

for i in range(len(times_v1)):
    diff = abs(np.max(V_hist_v1[i]) - np.max(V_hist_v2[i]))
    if diff > divergence_threshold:
        first_divergence_idx = i
        break

if first_divergence_idx is not None:
    print(f"\nFirst significant divergence at t={times_v1[first_divergence_idx]:.1f}ms")
    print(f"  V1: {np.max(V_hist_v1[first_divergence_idx]):.6f}")
    print(f"  V2: {np.max(V_hist_v2[first_divergence_idx]):.6f}")
    print(f"  Difference: {abs(np.max(V_hist_v1[first_divergence_idx]) - np.max(V_hist_v2[first_divergence_idx])):.6f}")
else:
    print("\nNo significant divergence detected!")
    print("Both simulations evolve identically.")
