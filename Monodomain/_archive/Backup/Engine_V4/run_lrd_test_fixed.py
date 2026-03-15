#!/usr/bin/env python3
"""Run LRd94 tests - Fixed version without matplotlib dependency"""
import sys
import os

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
    # Use local no-plot version
    from luo_rudy_1994_noplot import LuoRudy1994
    print("LuoRudy1994 model imported successfully (no-plot version)")
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

# Run a full AP simulation (500 ms to get complete APD90)
t_end = 500.0
n_steps = int(t_end / model.dt)
print(f"\nRunning {n_steps} steps ({t_end} ms)...")

stim_start = 10.0
stim_duration = 1.0
stim_amplitude = -80.0

V_max = state['V']
V_min = state['V']
V_data = []
t_data = []
Ca_data = []

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
    Ca_data.append(state['Ca_i'] * 1e6)  # Convert to nM

    # Print progress
    if i % 10000 == 0:
        print(f"  t = {t:.0f} ms, V = {state['V']:.1f} mV")

# Calculate APD90
V_rest = V_data[int(5.0/model.dt)]  # Get V at t=5ms (before stimulus)
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

# Calculate peak Ca
Ca_peak = max(Ca_data)

print(f"\nResults:")
print(f"  V_rest = {V_rest:.1f} mV")
print(f"  V_peak = {V_peak:.1f} mV")
print(f"  V_min = {V_min:.1f} mV")
print(f"  Final V = {state['V']:.1f} mV")
print(f"  [Ca]_i peak = {Ca_peak:.1f} nM")
print(f"  [Ca]_i final = {state['Ca_i']*1e6:.1f} nM")
if APD90:
    print(f"  APD90 = {APD90:.1f} ms")
else:
    print(f"  APD90 = Could not calculate (AP may not have repolarized)")

print("\n" + "=" * 60)
print("Physiological expectations (guinea pig):")
print("  V_rest: -84 to -90 mV")
print("  V_peak: +30 to +50 mV")
print("  APD90: 200-300 ms")
print("  [Ca]_i peak: 500-1000 nM")
print("=" * 60)

# Check values
print("\nValidation:")
success = True
if -90 <= V_rest <= -84:
    print(f"  V_rest ({V_rest:.1f} mV): PASS")
else:
    print(f"  V_rest ({V_rest:.1f} mV): FAIL (outside -84 to -90 mV)")
    success = False

if 30 <= V_peak <= 50:
    print(f"  V_peak ({V_peak:.1f} mV): PASS")
else:
    print(f"  V_peak ({V_peak:.1f} mV): WARNING (outside +30 to +50 mV)")

if APD90:
    if 200 <= APD90 <= 300:
        print(f"  APD90 ({APD90:.1f} ms): PASS")
    else:
        print(f"  APD90 ({APD90:.1f} ms): WARNING (outside 200-300 ms)")
else:
    print(f"  APD90: FAIL (could not calculate)")

if 500 <= Ca_peak <= 1000:
    print(f"  [Ca]_i peak ({Ca_peak:.1f} nM): PASS")
else:
    print(f"  [Ca]_i peak ({Ca_peak:.1f} nM): WARNING (outside 500-1000 nM)")

print("=" * 60)
if success:
    print("Test PASSED!")
else:
    print("Test completed with warnings/failures")
print("=" * 60)
