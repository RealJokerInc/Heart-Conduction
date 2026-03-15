"""
Debug Voltage Bounds Issue
===========================

Investigating:
1. Why V_min = -121.2 mV (not -85 mV as expected)
2. Why V_max > 1.0 (around 1.137 dimensionless)
3. Is -121.2 mV the safeguard floor or natural behavior?

Author: Generated with Claude Code
Date: 2025-12-09
"""

import numpy as np
from interactive_simulation import InteractiveSimulation
import matplotlib.pyplot as plt

print("=" * 70)
print("VOLTAGE BOUNDS DIAGNOSTIC")
print("=" * 70)

# Create simulation
sim = InteractiveSimulation(
    domain_size=80.0,
    resolution=0.5,
    initial_stim_amplitude=200.0,
    initial_stim_radius=10.0
)

print(f"\n📊 PARAMETER ANALYSIS:")
print(f"=" * 70)
print(f"\nPhysical voltage range:")
print(f"  V_rest = {sim.ionic_model.V_rest} mV")
print(f"  V_peak = {sim.ionic_model.V_peak} mV")
voltage_range = sim.ionic_model.V_peak - sim.ionic_model.V_rest
print(f"  Voltage range = {voltage_range} mV")

print(f"\nIonic model parameters:")
print(f"  mu2 = {sim.mu2}")

print(f"\nSafeguard floor (from Fix #2):")
print(f"  V_floor (dimensionless) = -mu2 + 0.01 = {-sim.mu2 + 0.01}")
print(f"  V_floor (physical) = ?")

# Convert safeguard floor to physical
V_floor_norm = -sim.mu2 + 0.01  # -0.29
V_floor_phys = sim.ionic_model.voltage_to_physical(np.array([[V_floor_norm]]))[0, 0]
print(f"  V_floor (physical) = {V_floor_phys:.1f} mV")
print(f"  ⚠️  This is -121.2 mV, matching your observation!")

print(f"\nDimensionless voltage mapping:")
print(f"  V = 0.0 (dimensionless) → {sim.ionic_model.voltage_to_physical(np.array([[0.0]]))[0, 0]:.1f} mV")
print(f"  V = 1.0 (dimensionless) → {sim.ionic_model.voltage_to_physical(np.array([[1.0]]))[0, 0]:.1f} mV")
print(f"  V = 1.137 (dimensionless) → {sim.ionic_model.voltage_to_physical(np.array([[1.137]]))[0, 0]:.1f} mV")

print(f"\n" + "=" * 70)
print(f"🔍 HYPOTHESIS:")
print(f"=" * 70)
print(f"\n1. V_min = -121.2 mV:")
print(f"   - This is the SAFEGUARD floor from Fix #2")
print(f"   - We clamped V ≥ -mu2 + 0.01 = -0.29 (dimensionless)")
print(f"   - In physical units: -0.29 → -121.2 mV")
print(f"   - Question: Is voltage HITTING this floor, or is this natural?")

print(f"\n2. V_max = 1.137:")
print(f"   - This is ABOVE the expected range [0, 1]")
print(f"   - In physical units: 1.137 → {sim.ionic_model.voltage_to_physical(np.array([[1.137]]))[0, 0]:.1f} mV")
print(f"   - Expected V_peak = {sim.ionic_model.V_peak} mV")
print(f"   - Overshoot by ≈ {sim.ionic_model.voltage_to_physical(np.array([[1.137]]))[0, 0] - sim.ionic_model.V_peak:.1f} mV")

print(f"\n" + "=" * 70)
print(f"🧪 RUNNING TEST SIMULATION")
print(f"=" * 70)

# Add stimulus
sim.add_stimulus(40.0, 40.0)

# Run and track when floor is hit
duration_ms = 100.0
n_steps = int(duration_ms / sim.dt)

V_min_history = []
V_max_history = []
t_history = []
hit_floor_count = 0
total_points = sim.V.size

print(f"\nRunning for {duration_ms} ms...")
print(f"Checking if V hits the safeguard floor (-0.29 dimensionless)...\n")

for step in range(n_steps):
    I_stim = sim.get_current_stimulus()
    sim.step(sim.dt, I_stim)

    V_min = np.min(sim.V)
    V_max = np.max(sim.V)
    V_min_history.append(V_min)
    V_max_history.append(V_max)
    t_history.append(sim.t)

    # Count how many points are at the floor
    at_floor = np.sum(np.abs(sim.V - (-sim.mu2 + 0.01)) < 1e-6)
    if at_floor > 0:
        hit_floor_count += 1

    if step % 1000 == 0:
        V_min_phys = sim.ionic_model.voltage_to_physical(np.array([[V_min]]))[0, 0]
        V_max_phys = sim.ionic_model.voltage_to_physical(np.array([[V_max]]))[0, 0]

        # Check if at floor
        at_floor_pct = 100.0 * at_floor / total_points
        floor_marker = f" ⚠️ {at_floor_pct:.1f}% at floor!" if at_floor > 0 else ""

        print(f"  t={sim.t:6.1f}ms: V∈[{V_min:7.4f}, {V_max:7.4f}] = [{V_min_phys:7.1f}, {V_max_phys:6.1f}]mV{floor_marker}")

print(f"\n" + "=" * 70)
print(f"📊 ANALYSIS RESULTS")
print(f"=" * 70)

V_min_final = np.min(sim.V)
V_max_final = np.max(sim.V)
V_min_phys_final = sim.ionic_model.voltage_to_physical(np.array([[V_min_final]]))[0, 0]
V_max_phys_final = sim.ionic_model.voltage_to_physical(np.array([[V_max_final]]))[0, 0]

print(f"\nFinal voltage bounds:")
print(f"  V_min = {V_min_final:.4f} (dimensionless) = {V_min_phys_final:.1f} mV")
print(f"  V_max = {V_max_final:.4f} (dimensionless) = {V_max_phys_final:.1f} mV")

print(f"\nSafeguard floor:")
print(f"  Floor = {-sim.mu2 + 0.01:.4f} (dimensionless) = {V_floor_phys:.1f} mV")

if np.abs(V_min_final - (-sim.mu2 + 0.01)) < 1e-6:
    print(f"  ⚠️  V_min IS AT THE SAFEGUARD FLOOR!")
    print(f"      This means voltage tried to go below -0.29 but was clamped")
else:
    print(f"  ✓ V_min is NOT at safeguard floor (natural value)")

print(f"\nV_max analysis:")
if V_max_final > 1.0:
    overshoot = sim.ionic_model.voltage_to_physical(np.array([[V_max_final]]))[0, 0] - sim.ionic_model.V_peak
    print(f"  ⚠️  V_max > 1.0 (overshooting expected range)")
    print(f"      Overshoot: {overshoot:.1f} mV above V_peak")
else:
    print(f"  ✓ V_max within expected range [0, 1]")

print(f"\nTemporal analysis:")
print(f"  Time steps where points hit floor: {hit_floor_count}/{n_steps}")
if hit_floor_count > 0:
    print(f"  ⚠️  Safeguard was triggered during simulation!")
else:
    print(f"  ✓ Safeguard never triggered (V naturally stayed above floor)")

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Time series (dimensionless)
axes[0, 0].plot(t_history, V_min_history, 'b-', label='V_min', linewidth=2)
axes[0, 0].plot(t_history, V_max_history, 'r-', label='V_max', linewidth=2)
axes[0, 0].axhline(y=-sim.mu2 + 0.01, color='orange', linestyle='--',
                   label=f'Safeguard floor = {-sim.mu2 + 0.01:.2f}', linewidth=2)
axes[0, 0].axhline(y=0.0, color='gray', linestyle=':', label='V=0 (rest)')
axes[0, 0].axhline(y=1.0, color='gray', linestyle=':', label='V=1 (peak)')
axes[0, 0].set_xlabel('Time [ms]', fontsize=12)
axes[0, 0].set_ylabel('Voltage (dimensionless)', fontsize=12)
axes[0, 0].set_title('Voltage Bounds (Dimensionless)', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# Time series (physical)
V_min_phys = sim.ionic_model.voltage_to_physical(np.array([V_min_history]))[0]
V_max_phys = sim.ionic_model.voltage_to_physical(np.array([V_max_history]))[0]

axes[0, 1].plot(t_history, V_min_phys, 'b-', label='V_min', linewidth=2)
axes[0, 1].plot(t_history, V_max_phys, 'r-', label='V_max', linewidth=2)
axes[0, 1].axhline(y=V_floor_phys, color='orange', linestyle='--',
                   label=f'Safeguard floor = {V_floor_phys:.1f} mV', linewidth=2)
axes[0, 1].axhline(y=sim.ionic_model.V_rest, color='gray', linestyle=':', label=f'V_rest = {sim.ionic_model.V_rest} mV')
axes[0, 1].axhline(y=sim.ionic_model.V_peak, color='gray', linestyle=':', label=f'V_peak = {sim.ionic_model.V_peak} mV')
axes[0, 1].set_xlabel('Time [ms]', fontsize=12)
axes[0, 1].set_ylabel('Voltage [mV]', fontsize=12)
axes[0, 1].set_title('Voltage Bounds (Physical)', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# Zoom on V_min
axes[1, 0].plot(t_history, V_min_history, 'b-', linewidth=2)
axes[1, 0].axhline(y=-sim.mu2 + 0.01, color='orange', linestyle='--',
                   label=f'Floor = {-sim.mu2 + 0.01:.4f}', linewidth=2)
axes[1, 0].set_xlabel('Time [ms]', fontsize=12)
axes[1, 0].set_ylabel('V_min (dimensionless)', fontsize=12)
axes[1, 0].set_title('V_min Detail (Checking if at floor)', fontsize=14, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# Zoom on V_max
axes[1, 1].plot(t_history, V_max_history, 'r-', linewidth=2)
axes[1, 1].axhline(y=1.0, color='gray', linestyle=':', label='Expected max = 1.0', linewidth=2)
axes[1, 1].set_xlabel('Time [ms]', fontsize=12)
axes[1, 1].set_ylabel('V_max (dimensionless)', fontsize=12)
axes[1, 1].set_title('V_max Detail (Checking overshoot)', fontsize=14, fontweight='bold')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('voltage_bounds_analysis.png', dpi=150)
print(f"\n📊 Diagnostic plot saved: voltage_bounds_analysis.png")

# Summary
print(f"\n" + "=" * 70)
print(f"🎯 DIAGNOSIS SUMMARY")
print(f"=" * 70)

print(f"\n**Issue 1: V_min = -121.2 mV**")
if np.abs(V_min_final - (-sim.mu2 + 0.01)) < 1e-6:
    print(f"  CAUSE: Safeguard is actively clamping voltage")
    print(f"  MEANING: Without safeguard, voltage would drop further (unstable)")
    print(f"  STATUS: ⚠️  Safeguard preventing crash, but physics may be wrong")
else:
    print(f"  CAUSE: Natural equilibrium of the model")
    print(f"  MEANING: This is the expected resting/repolarized state")
    print(f"  STATUS: ✓ Normal behavior")

print(f"\n**Issue 2: V_max > 1.0**")
if V_max_final > 1.0:
    print(f"  CAUSE: Model allowing overshoot beyond expected range")
    print(f"  OVERSHOOT: {overshoot:.1f} mV above V_peak ({sim.ionic_model.V_peak} mV)")
    print(f"  STATUS: ⚠️  May indicate wrong parameters or normalization")
else:
    print(f"  STATUS: ✓ Normal behavior")

print(f"\n**Issue 3: Is -121.2 mV a crash threshold?**")
if hit_floor_count > 0:
    print(f"  ⚠️  YES - safeguard is preventing further drops")
    print(f"  Without safeguard, V would drop below -0.29 → crash likely")
else:
    print(f"  ✓ NO - voltage naturally stays above this level")

print(f"\n" + "=" * 70)
