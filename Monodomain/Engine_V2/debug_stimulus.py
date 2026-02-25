"""
Debug Stimulus Application
===========================

Check if stimulus is strong/long enough to trigger AP.

Author: Generated with Claude Code
Date: 2025-12-09
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/lemon/Documents/Python/Heart Conduction/Engine_V1')

from simulate_infarct import InfarctSimulation as SimV1

print("=" * 70)
print("DEBUGGING STIMULUS")
print("=" * 70)

# Create V1 simulation (known to work from earlier)
sim = SimV1(
    domain_size=80.0,
    resolution=0.5,
    infarct_radius=10.0,
    D_parallel=1.0,
    D_perp=0.5,
    T_scale=10.0
)

# Create stimulus
stim_func = sim.create_stimulus(
    amplitude=30.0,
    duration=2.0,
    start_time=5.0
)

print("\nChecking stimulus function...")
for t in [0.0, 4.9, 5.0, 5.5, 6.0, 6.9, 7.0, 7.1]:
    I_stim = stim_func(t)
    I_max = np.max(I_stim)
    I_sum = np.sum(I_stim)
    print(f"  t={t:4.1f}ms: I_max={I_max:.6f}, I_sum={I_sum:.6f}, points={np.sum(I_stim > 0)}")

# Run short simulation
print("\nRunning 50ms V1 simulation...")
times, V_hist = sim.simulate(
    t_end=50.0,
    dt=0.01,
    stim_func=stim_func,
    save_every_ms=1.0,
    verbose=False
)

print(f"\nV1 Results:")
for i, t in enumerate(times[::5]):
    V_max = np.max(V_hist[i*5])
    V_left = np.max(V_hist[i*5, :, 0])
    print(f"  t={t:5.1f}ms: V_max={V_max:.6f}, V_left_edge={V_left:.6f}")

if np.max(V_hist) > 0.5:
    print("\n✓ V1 wave propagates!")
else:
    print("\n✗ V1 wave also dies!")
    print(f"  Max V reached: {np.max(V_hist):.6f}")

# Check parameters
print("\n" + "=" * 70)
print("PARAMETER CHECK")
print("=" * 70)
print(f"Ionic model parameters:")
print(f"  k={sim.ionic_model.ionic.k}")
print(f"  a={sim.ionic_model.ionic.a}")
print(f"  epsilon0={sim.ionic_model.ionic.epsilon0}")
print(f"  mu1={sim.ionic_model.ionic.mu1}")
print(f"  mu2={sim.ionic_model.ionic.mu2}")
print(f"  epsilon_rest={sim.ionic_model.ionic.epsilon_rest}")
print(f"  T_scale={sim.ionic_model.T_scale}")

print(f"\nVoltage normalization:")
print(f"  V_rest={sim.ionic_model.V_rest} mV")
print(f"  V_peak={sim.ionic_model.V_peak} mV")
print(f"  V_range={sim.ionic_model.V_range} mV")

print(f"\nStimulus:")
print(f"  Amplitude: 30 mV")
print(f"  Normalized: {30.0 / sim.ionic_model.V_range:.6f}")
print(f"  Duration: 2.0 ms")
print(f"  Steps with stimulus: {int(2.0 / 0.01)}")

print(f"\nTime stepping:")
print(f"  dt (physical): 0.01 ms")
print(f"  dtau (dimensionless): {sim.ionic_model.time_to_tau(0.01):.6f}")
print(f"  Steps in 2ms stimulus: {int(2.0 / 0.01)}")

# Calculate expected V change per step
dtau = sim.ionic_model.time_to_tau(0.01)
I_stim_norm = 30.0 / sim.ionic_model.V_range
dV_per_step = dtau * I_stim_norm
print(f"\nExpected ΔV per step:")
print(f"  dV = dtau * I_stim = {dtau:.6f} * {I_stim_norm:.6f} = {dV_per_step:.6f}")
print(f"  Total ΔV over 2ms = {dV_per_step * 200:.6f}")
