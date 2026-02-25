"""Simple test for LRd94 model - no plotting."""
import sys
sys.stdout.flush()

print("Starting LRd94 test...")
print("=" * 60)

import numpy as np
print("NumPy imported successfully")

# Import the model
from luo_rudy_1994 import LuoRudy1994
print("LuoRudy1994 model imported successfully")

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

# Run a short simulation (50 ms)
t_end = 100.0
n_steps = int(t_end / model.dt)
print(f"\nRunning {n_steps} steps ({t_end} ms)...")

stim_start = 10.0
stim_duration = 1.0
stim_amplitude = -80.0

V_max = state['V']
V_min = state['V']

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

    # Print progress
    if i % 2000 == 0:
        print(f"  t = {t:.0f} ms, V = {state['V']:.1f} mV")

print(f"\nResults:")
print(f"  V_max = {V_max:.1f} mV")
print(f"  V_min = {V_min:.1f} mV")
print(f"  Final V = {state['V']:.1f} mV")
print(f"  [Ca]_i final = {state['Ca_i']*1e6:.1f} nM")

print("\n" + "=" * 60)
print("Test completed successfully!")
