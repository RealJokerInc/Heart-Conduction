"""
Test Interactive Simulation Setup
==================================

Verify interactive simulation components work without GUI.

Author: Generated with Claude Code
Date: 2025-12-09
"""

import numpy as np
from interactive_simulation import InteractiveSimulation

print("=" * 70)
print("TESTING INTERACTIVE SIMULATION COMPONENTS")
print("=" * 70)

# Create simulation
print("\nTest 1: Creating simulation...")
sim = InteractiveSimulation(
    domain_size=80.0,
    resolution=0.5,
    initial_stim_amplitude=30.0,
    initial_stim_radius=3.0
)
print("✓ Simulation created")

# Test stimulus addition
print("\nTest 2: Adding stimuli...")
sim.add_stimulus(20.0, 40.0)  # Center-left
sim.add_stimulus(60.0, 40.0)  # Center-right
print(f"  Active stimuli: {len(sim.active_stimuli)}")
assert len(sim.active_stimuli) == 2, "Should have 2 active stimuli"
print("✓ Stimulus addition works")

# Test stimulus computation
print("\nTest 3: Computing stimulus field...")
I_stim = sim.get_current_stimulus()
print(f"  Stimulus shape: {I_stim.shape}")
print(f"  Stimulus max: {np.max(I_stim):.6f}")
print(f"  Stimulus points: {np.sum(I_stim > 0)}")
assert np.max(I_stim) > 0, "Stimulus should be non-zero"
print("✓ Stimulus computation works")

# Test simulation step
print("\nTest 4: Running simulation steps...")
initial_V_max = np.max(sim.V)
print(f"  Initial V_max: {initial_V_max:.6f}")

for step in range(200):  # 2ms of stimulus
    I_stim = sim.get_current_stimulus()
    sim.step(sim.dt, I_stim)

after_stim_V_max = np.max(sim.V)
print(f"  After 2ms: V_max = {after_stim_V_max:.6f}")
assert after_stim_V_max > 0.02, "Voltage should increase from stimulus"
print("✓ Simulation steps work")

# Continue simulation
print("\nTest 5: Continued simulation...")
for step in range(500):  # 5ms more
    I_stim = sim.get_current_stimulus()
    sim.step(sim.dt, I_stim)

final_t = sim.t
print(f"  Simulation time: {final_t:.1f} ms")
print(f"  Total steps: {sim.frame_count}")
assert final_t > 5.0, "Simulation should advance in time"
print("✓ Simulation advances correctly")

# Test stimulus expiration
print("\nTest 6: Stimulus expiration...")
print(f"  Active stimuli: {len(sim.active_stimuli)}")
assert len(sim.active_stimuli) == 0, "Stimuli should have expired"
print("✓ Stimulus expiration works")

# Test reset
print("\nTest 7: Reset simulation...")
sim.reset()
print(f"  V_max after reset: {np.max(sim.V):.6f}")
print(f"  Time after reset: {sim.t:.1f} ms")
assert np.max(sim.V) < 1e-10, "Voltage should be zero"
assert sim.t == 0.0, "Time should be zero"
assert len(sim.active_stimuli) == 0, "No active stimuli"
print("✓ Reset works")

# Test parameter changes
print("\nTest 8: Parameter adjustments...")
original_amp = sim.stim_amplitude
original_rad = sim.stim_radius

sim.stim_amplitude = 50.0
sim.stim_radius = 5.0
print(f"  Changed amplitude: {original_amp:.1f} → {sim.stim_amplitude:.1f} mV")
print(f"  Changed radius: {original_rad:.1f} → {sim.stim_radius:.1f} mm")

sim.add_stimulus(40.0, 40.0)
I_stim_new = sim.get_current_stimulus()
stim_points_new = np.sum(I_stim_new > 0)
print(f"  New stimulus points: {stim_points_new}")
print("✓ Parameter changes work")

print("\n" + "=" * 70)
print("ALL TESTS PASSED!")
print("=" * 70)
print("\n✓ Interactive simulation is ready to use!")
print("\nTo run interactively:")
print("  python3.11 interactive_simulation.py")
print("\nThen click anywhere to add stimulation!")
