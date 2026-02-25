#!/usr/bin/env python3
"""Run LRd94 tests from Engine_V1"""
import sys
import os

# Add Engine_V1 to path
sys.path.insert(0, '/Users/lemon/Documents/Python/Heart Conduction/Bidomain/Engine_V1')

print("Starting LRd94 test...")
print("=" * 60)

try:
    import numpy as np
    print("NumPy imported successfully")
    print(f"NumPy version: {np.__version__}")
except ImportError as e:
    print(f"ERROR: Could not import NumPy: {e}")
    sys.exit(1)

try:
    from luo_rudy_1994 import LuoRudy1994
    print("LuoRudy1994 model imported successfully")
except ImportError as e:
    print(f"ERROR: Could not import LuoRudy1994: {e}")
    sys.exit(1)

# Create model
model = LuoRudy1994(dt=0.005)
print(f"Model created with dt = {model.dt} ms")

# Initialize state
state = model.initialize_state()
print(f"\nInitial state:")
print(f"  V = {state['V']:.1f} mV")
print(f"  [Na]_i = {state['Na_i']:.2f} mM")
print(f"  [K]_i = {state['K_i']:.2f} mM")
print(f"  [Ca]_i = {state['Ca_i']*1e6:.1f} nM")

# Run a simulation (100 ms)
t_end = 100.0
n_steps = int(t_end / model.dt)
print(f"\nRunning {n_steps} steps ({t_end} ms)...")

stim_start = 10.0
stim_duration = 1.0
stim_amplitude = -80.0

V_max = state['V']
V_min = state['V']
V_data = []
t_data = []

for i in range(n_steps):
    t = i * model.dt

    # Stimulus
    if stim_start <= t < stim_start + stim_duration:
        I_stim = stim_amplitude
    else:
        I_stim = 0.0

    # Step
    state = model.step(state, I_stim)

    # Track extremes
    V_max = max(V_max, state['V'])
    V_min = min(V_min, state['V'])

    # Store data
    V_data.append(state['V'])
    t_data.append(t)

    # Print progress
    if i % 2000 == 0:
        print(f"  t = {t:.0f} ms, V = {state['V']:.1f} mV")

# Calculate APD90
V_rest = V_data[0]
V_peak = max(V_data)
V_90 = V_rest + 0.1 * (V_peak - V_rest)

APD90 = None
above_threshold = False
t_start = None
for i, V in enumerate(V_data):
    if not above_threshold and V > V_90:
        above_threshold = True
        t_start = t_data[i]
    elif above_threshold and V <= V_90:
        APD90 = t_data[i] - t_start
        break

print(f"\nResults:")
print(f"  V_rest = {V_rest:.1f} mV")
print(f"  V_peak = {V_peak:.1f} mV")
print(f"  V_min = {V_min:.1f} mV")
print(f"  Final V = {state['V']:.1f} mV")
print(f"  [Ca]_i final = {state['Ca_i']*1e6:.1f} nM")
if APD90:
    print(f"  APD90 = {APD90:.1f} ms")
else:
    print(f"  APD90 = Could not calculate (AP may not have repolarized)")

print("\n" + "=" * 60)
print("Physiological expectations:")
print("  V_rest: -84 to -90 mV")
print("  V_peak: +30 to +50 mV")
print("  APD90: 200-300 ms")
print("=" * 60)

# Check values
success = True
if not (-90 <= V_rest <= -84):
    print(f"WARNING: V_rest ({V_rest:.1f} mV) outside expected range")
    success = False
if not (30 <= V_peak <= 50):
    print(f"WARNING: V_peak ({V_peak:.1f} mV) outside expected range")
    success = False
if APD90 and not (200 <= APD90 <= 300):
    print(f"WARNING: APD90 ({APD90:.1f} ms) outside expected range (but this is a short simulation)")

if success:
    print("\nTest completed successfully!")
else:
    print("\nTest completed with warnings!")
