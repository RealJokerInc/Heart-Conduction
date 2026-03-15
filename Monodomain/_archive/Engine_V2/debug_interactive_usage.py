"""
Debug Interactive Usage Pattern
================================

Test with multiple stimuli (like user clicking multiple times)
to see if we can reproduce the -121.2 mV floor behavior.

Author: Generated with Claude Code
Date: 2025-12-09
"""

import numpy as np
from interactive_simulation import InteractiveSimulation
import matplotlib.pyplot as plt

print("=" * 70)
print("TESTING INTERACTIVE USAGE PATTERN")
print("=" * 70)

# Create simulation with user's likely settings
sim = InteractiveSimulation(
    domain_size=80.0,
    resolution=0.5,
    initial_stim_amplitude=100.0,  # Moderate stimulus
    initial_stim_radius=5.0
)

print(f"\nSimulating multiple clicks across the tissue...")
print(f"(Like user exploring the simulation)")

# Add multiple stimuli at different locations
stimulus_points = [
    (20.0, 40.0, 0.0),    # Left
    (40.0, 40.0, 50.0),   # Center after 50ms
    (60.0, 40.0, 100.0),  # Right after 100ms
    (40.0, 20.0, 150.0),  # Bottom after 150ms
    (40.0, 60.0, 200.0),  # Top after 200ms
]

V_min_history = []
V_max_history = []
t_history = []
at_floor_history = []

# Run for 300ms
duration_ms = 300.0
n_steps = int(duration_ms / sim.dt)

print(f"Running for {duration_ms} ms with 5 stimuli...")
print(f"Safeguard floor = {-sim.mu2 + 0.01:.4f} (dimensionless) = -121.2 mV\n")

for step in range(n_steps):
    t = sim.t

    # Add stimuli at scheduled times
    for x_stim, y_stim, t_stim in stimulus_points:
        if abs(t - t_stim) < sim.dt / 2:
            sim.add_stimulus(x_stim, y_stim)
            print(f"  ⚡ Stimulus added at ({x_stim:.0f}, {y_stim:.0f}) mm, t={t:.1f}ms")

    # Get stimulus and step
    I_stim = sim.get_current_stimulus()
    sim.step(sim.dt, I_stim)

    # Record
    V_min = np.min(sim.V)
    V_max = np.max(sim.V)
    V_min_history.append(V_min)
    V_max_history.append(V_max)
    t_history.append(sim.t)

    # Check if at floor
    at_floor = np.sum(np.abs(sim.V - (-sim.mu2 + 0.01)) < 1e-6)
    at_floor_history.append(at_floor)

    # Progress
    if step % 2000 == 0:
        V_min_phys = sim.ionic_model.voltage_to_physical(np.array([[V_min]]))[0, 0]
        V_max_phys = sim.ionic_model.voltage_to_physical(np.array([[V_max]]))[0, 0]

        at_floor_pct = 100.0 * at_floor / sim.V.size
        floor_marker = f" ⚠️ {at_floor_pct:.1f}% at floor!" if at_floor > 10 else ""

        print(f"  t={sim.t:6.1f}ms: V∈[{V_min:7.4f}, {V_max:7.4f}] = [{V_min_phys:7.1f}, {V_max_phys:6.1f}]mV{floor_marker}")

print(f"\n" + "=" * 70)
print(f"ANALYSIS")
print(f"=" * 70)

V_min_final = np.min(sim.V)
V_max_final = np.max(sim.V)
V_min_phys_final = sim.ionic_model.voltage_to_physical(np.array([[V_min_final]]))[0, 0]
V_max_phys_final = sim.ionic_model.voltage_to_physical(np.array([[V_max_final]]))[0, 0]

print(f"\nFinal voltage bounds:")
print(f"  V_min = {V_min_final:.4f} → {V_min_phys_final:.1f} mV")
print(f"  V_max = {V_max_final:.4f} → {V_max_phys_final:.1f} mV")

print(f"\nSafeguard floor = {-sim.mu2 + 0.01:.4f} → -121.2 mV")

# Check if we hit the floor
V_min_array = np.array(V_min_history)
at_floor_mask = np.abs(V_min_array - (-sim.mu2 + 0.01)) < 1e-4

if np.any(at_floor_mask):
    n_at_floor = np.sum(at_floor_mask)
    pct_at_floor = 100.0 * n_at_floor / len(V_min_array)
    print(f"\n⚠️  VOLTAGE HIT THE SAFEGUARD FLOOR!")
    print(f"    Time steps at floor: {n_at_floor}/{len(V_min_array)} ({pct_at_floor:.1f}%)")

    # Find when it first hit
    first_hit_idx = np.where(at_floor_mask)[0][0]
    print(f"    First hit at t = {t_history[first_hit_idx]:.1f} ms")
else:
    print(f"\n✓ Voltage never hit safeguard floor")
    print(f"  Min voltage reached: {np.min(V_min_array):.4f} ({sim.ionic_model.voltage_to_physical(np.array([[np.min(V_min_array)]]))[0, 0]:.1f} mV)")
    print(f"  Distance from floor: {np.min(V_min_array) - (-sim.mu2 + 0.01):.4f}")

# Check V_max
V_max_array = np.array(V_max_history)
max_V_max = np.max(V_max_array)
max_V_max_phys = sim.ionic_model.voltage_to_physical(np.array([[max_V_max]]))[0, 0]

print(f"\nV_max analysis:")
print(f"  Maximum V_max reached: {max_V_max:.4f} → {max_V_max_phys:.1f} mV")
if max_V_max > 1.0:
    print(f"  ⚠️  V_max > 1.0 detected!")
    print(f"      Overshoot: {max_V_max_phys - 40.0:.1f} mV above expected peak")

    # Find when
    overshoot_mask = V_max_array > 1.0
    if np.any(overshoot_mask):
        first_overshoot = np.where(overshoot_mask)[0][0]
        print(f"      First overshoot at t = {t_history[first_overshoot]:.1f} ms")
else:
    print(f"  ✓ V_max stayed within [0, 1] range")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# V_min time series
V_min_phys_array = sim.ionic_model.voltage_to_physical(np.array([V_min_array]))[0]
axes[0].plot(t_history, V_min_phys_array, 'b-', linewidth=1.5, label='V_min')
axes[0].axhline(y=-121.2, color='orange', linestyle='--', linewidth=2, label='Safeguard floor (-121.2 mV)')
axes[0].axhline(y=-85.0, color='gray', linestyle=':', label='V_rest (-85.0 mV)')
axes[0].set_xlabel('Time [ms]', fontsize=12)
axes[0].set_ylabel('V_min [mV]', fontsize=12)
axes[0].set_title('Minimum Voltage vs Time', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Mark stimulus times
for x_stim, y_stim, t_stim in stimulus_points:
    axes[0].axvline(x=t_stim, color='red', linestyle=':', alpha=0.5)

# V_max time series
V_max_phys_array = sim.ionic_model.voltage_to_physical(np.array([V_max_array]))[0]
axes[1].plot(t_history, V_max_phys_array, 'r-', linewidth=1.5, label='V_max')
axes[1].axhline(y=40.0, color='gray', linestyle=':', label='V_peak (40.0 mV)')
axes[1].set_xlabel('Time [ms]', fontsize=12)
axes[1].set_ylabel('V_max [mV]', fontsize=12)
axes[1].set_title('Maximum Voltage vs Time', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

# Mark stimulus times
for x_stim, y_stim, t_stim in stimulus_points:
    axes[1].axvline(x=t_stim, color='red', linestyle=':', alpha=0.5)

# Points at floor
axes[2].plot(t_history, at_floor_history, 'g-', linewidth=1.5)
axes[2].set_xlabel('Time [ms]', fontsize=12)
axes[2].set_ylabel('Number of grid points at floor', fontsize=12)
axes[2].set_title('Safeguard Activation', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('interactive_usage_analysis.png', dpi=150)
print(f"\n📊 Plot saved: interactive_usage_analysis.png")

print(f"\n" + "=" * 70)
print(f"CONCLUSION")
print(f"=" * 70)

if np.any(at_floor_mask):
    print(f"\n❌ PROBLEM CONFIRMED:")
    print(f"   Voltage IS hitting the safeguard floor (-121.2 mV)")
    print(f"   This means the physics is trying to push V below this level")
    print(f"   The safeguard is preventing a crash, but physics may be wrong")
elif np.min(V_min_array) < -0.2:
    print(f"\n⚠️  CLOSE TO FLOOR:")
    print(f"   V_min = {np.min(V_min_array):.4f} is approaching floor (-0.29)")
    print(f"   May hit floor with longer simulation or stronger stimuli")
else:
    print(f"\n✓ NO PROBLEM:")
    print(f"   Voltage stays well above safeguard floor")
    print(f"   -121.2 mV observation may be from different parameters")

print(f"=" * 70)
