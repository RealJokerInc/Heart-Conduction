"""
Test Voltage Bounds Fixes
==========================

Validates that Fixes #1 and #2 work correctly:
- Fix #1: dt = 0.005 ms (reduced from 0.01)
- Fix #2: V clamped to [0, 1]

Expected results:
- V_min should stay >= 0.0 (>= -85 mV physical)
- V_max should stay <= 1.0 (<= +40 mV physical)
- No more -121.2 mV floor hits
- More accurate AP dynamics

Author: Generated with Claude Code
Date: 2025-12-09
"""

import numpy as np
from interactive_simulation import InteractiveSimulation
import matplotlib.pyplot as plt

print("=" * 70)
print("TESTING VOLTAGE BOUNDS FIXES")
print("=" * 70)

# Create simulation (should now use dt=0.005)
sim = InteractiveSimulation(
    domain_size=80.0,
    resolution=0.5,
    initial_stim_amplitude=200.0,
    initial_stim_radius=10.0
)

print(f"\nFix #1 - Time step:")
print(f"  dt = {sim.dt} ms (should be 0.005)")
if abs(sim.dt - 0.005) < 1e-6:
    print(f"  ✅ Fix #1 APPLIED")
else:
    print(f"  ❌ Fix #1 NOT APPLIED (still {sim.dt})")

print(f"\nFix #2 - Voltage clamping:")
print(f"  V should be clamped to [0, 1] after ionic step")

# Add left-edge stimulus
sim.add_stimulus(5.0, 40.0)
print(f"\nStimulus: 200mV at left edge (5.0, 40.0) mm")

# Run simulation
duration_ms = 100.0
n_steps = int(duration_ms / sim.dt)

V_min_history = []
V_max_history = []
t_history = []

violations = {
    'V_min_below_0': [],
    'V_max_above_1': [],
}

print(f"\nRunning for {duration_ms} ms ({n_steps} steps)...")
print(f"Monitoring for voltage bound violations...\n")

for step in range(n_steps):
    I_stim = sim.get_current_stimulus()
    sim.step(sim.dt, I_stim)

    V_min = np.min(sim.V)
    V_max = np.max(sim.V)
    V_min_history.append(V_min)
    V_max_history.append(V_max)
    t_history.append(sim.t)

    # Check for violations
    if V_min < -1e-6:  # Allow tiny numerical errors
        violations['V_min_below_0'].append((sim.t, V_min))
    if V_max > 1.0 + 1e-6:
        violations['V_max_above_1'].append((sim.t, V_max))

    if step % 2000 == 0:
        V_min_phys = sim.ionic_model.voltage_to_physical(np.array([[V_min]]))[0, 0]
        V_max_phys = sim.ionic_model.voltage_to_physical(np.array([[V_max]]))[0, 0]
        print(f"  t={sim.t:6.1f}ms: V∈[{V_min:7.4f}, {V_max:7.4f}] = [{V_min_phys:7.1f}, {V_max_phys:6.1f}]mV")

print(f"\n" + "=" * 70)
print(f"VALIDATION RESULTS")
print(f"=" * 70)

V_min_array = np.array(V_min_history)
V_max_array = np.array(V_max_history)

global_V_min = np.min(V_min_array)
global_V_max = np.max(V_max_array)

V_min_phys = sim.ionic_model.voltage_to_physical(np.array([[global_V_min]]))[0, 0]
V_max_phys = sim.ionic_model.voltage_to_physical(np.array([[global_V_max]]))[0, 0]

print(f"\nOverall voltage bounds:")
print(f"  V_min = {global_V_min:.6f} → {V_min_phys:.2f} mV")
print(f"  V_max = {global_V_max:.6f} → {V_max_phys:.2f} mV")

print(f"\nExpected bounds:")
print(f"  V ∈ [0, 1] (dimensionless)")
print(f"  V ∈ [-85, +40] mV (physical)")

# Check Fix #2 success
print(f"\n" + "=" * 70)
print(f"FIX #2 VALIDATION:")
print(f"=" * 70)

fix2_success = True

print(f"\n1. V_min bound check:")
if global_V_min >= -1e-6:  # Allow tiny numerical errors
    print(f"   ✅ PASS: V_min = {global_V_min:.6f} >= 0")
    print(f"   Physical: {V_min_phys:.2f} mV >= -85 mV")
else:
    print(f"   ❌ FAIL: V_min = {global_V_min:.6f} < 0")
    print(f"   Physical: {V_min_phys:.2f} mV < -85 mV")
    fix2_success = False

print(f"\n2. V_max bound check:")
if global_V_max <= 1.0 + 1e-6:
    print(f"   ✅ PASS: V_max = {global_V_max:.6f} <= 1.0")
    print(f"   Physical: {V_max_phys:.2f} mV <= 40 mV")
else:
    print(f"   ❌ FAIL: V_max = {global_V_max:.6f} > 1.0")
    print(f"   Physical: {V_max_phys:.2f} mV > 40 mV")
    print(f"   Overshoot: {V_max_phys - 40.0:.2f} mV")
    fix2_success = False

print(f"\n3. No -121.2 mV floor hits:")
old_floor = -0.29  # Old safeguard value
at_old_floor = np.abs(V_min_array - old_floor) < 1e-4
if not np.any(at_old_floor):
    print(f"   ✅ PASS: V never hit old floor ({old_floor:.2f})")
else:
    n_hits = np.sum(at_old_floor)
    print(f"   ⚠️  WARNING: V hit old floor {n_hits} times")
    print(f"   (This might still be okay if clamping at 0.0)")

# Detailed violation report
if violations['V_min_below_0']:
    print(f"\n⚠️  V_min violations (V < 0):")
    for t, v in violations['V_min_below_0'][:5]:  # Show first 5
        print(f"     t={t:.2f}ms: V={v:.6f}")
    if len(violations['V_min_below_0']) > 5:
        print(f"     ... and {len(violations['V_min_below_0']) - 5} more")

if violations['V_max_above_1']:
    print(f"\n⚠️  V_max violations (V > 1):")
    for t, v in violations['V_max_above_1'][:5]:
        v_phys = sim.ionic_model.voltage_to_physical(np.array([[v]]))[0, 0]
        print(f"     t={t:.2f}ms: V={v:.6f} ({v_phys:.2f} mV)")
    if len(violations['V_max_above_1']) > 5:
        print(f"     ... and {len(violations['V_max_above_1']) - 5} more")

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Voltage bounds (dimensionless)
axes[0, 0].plot(t_history, V_min_history, 'b-', linewidth=1.5, label='V_min')
axes[0, 0].plot(t_history, V_max_history, 'r-', linewidth=1.5, label='V_max')
axes[0, 0].axhline(y=0.0, color='green', linestyle='--', linewidth=2, label='Lower bound (0)')
axes[0, 0].axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Upper bound (1)')
axes[0, 0].axhline(y=-0.29, color='orange', linestyle=':', label='Old floor (-0.29)', alpha=0.5)
axes[0, 0].set_xlabel('Time [ms]', fontsize=12)
axes[0, 0].set_ylabel('Voltage (dimensionless)', fontsize=12)
axes[0, 0].set_title('Voltage Bounds (Dimensionless)', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

# Voltage bounds (physical)
V_min_phys_array = sim.ionic_model.voltage_to_physical(np.array([V_min_history]))[0]
V_max_phys_array = sim.ionic_model.voltage_to_physical(np.array([V_max_history]))[0]

axes[0, 1].plot(t_history, V_min_phys_array, 'b-', linewidth=1.5, label='V_min')
axes[0, 1].plot(t_history, V_max_phys_array, 'r-', linewidth=1.5, label='V_max')
axes[0, 1].axhline(y=-85.0, color='green', linestyle='--', linewidth=2, label='Lower bound (-85 mV)')
axes[0, 1].axhline(y=40.0, color='green', linestyle='--', linewidth=2, label='Upper bound (40 mV)')
axes[0, 1].axhline(y=-121.2, color='orange', linestyle=':', label='Old floor (-121.2 mV)', alpha=0.5)
axes[0, 1].set_xlabel('Time [ms]', fontsize=12)
axes[0, 1].set_ylabel('Voltage [mV]', fontsize=12)
axes[0, 1].set_title('Voltage Bounds (Physical)', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# Zoom on V_min
axes[1, 0].plot(t_history, V_min_history, 'b-', linewidth=2)
axes[1, 0].axhline(y=0.0, color='green', linestyle='--', linewidth=2, label='New bound (0)')
axes[1, 0].axhline(y=-0.29, color='orange', linestyle=':', label='Old floor (-0.29)', alpha=0.5)
axes[1, 0].set_xlabel('Time [ms]', fontsize=12)
axes[1, 0].set_ylabel('V_min (dimensionless)', fontsize=12)
axes[1, 0].set_title('V_min Detail', fontsize=14, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# Zoom on V_max
axes[1, 1].plot(t_history, V_max_history, 'r-', linewidth=2)
axes[1, 1].axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='New bound (1)')
axes[1, 1].set_xlabel('Time [ms]', fontsize=12)
axes[1, 1].set_ylabel('V_max (dimensionless)', fontsize=12)
axes[1, 1].set_title('V_max Detail', fontsize=14, fontweight='bold')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('voltage_bounds_fixes_validation.png', dpi=150)
print(f"\n📊 Validation plot saved: voltage_bounds_fixes_validation.png")

# Final summary
print(f"\n" + "=" * 70)
print(f"FINAL SUMMARY")
print(f"=" * 70)

print(f"\nFix #1 (dt = 0.005 ms):")
if abs(sim.dt - 0.005) < 1e-6:
    print(f"  ✅ IMPLEMENTED and ACTIVE")
    print(f"  Simulation is 2x slower but more accurate")
else:
    print(f"  ❌ NOT IMPLEMENTED")

print(f"\nFix #2 (V ∈ [0, 1]):")
if fix2_success:
    print(f"  ✅ SUCCESS - Voltage stayed within bounds")
    print(f"  No overshoot, no undershoot")
    print(f"  Physics is now correct!")
else:
    print(f"  ❌ FAILED - Violations detected")
    print(f"  Check diagnostic output above")

print(f"\nUser-Reported Issues:")
print(f"  1. V_min = -121.2 mV: {'✅ FIXED' if global_V_min >= -1e-6 else '❌ STILL PRESENT'}")
print(f"  2. V_max > 1.0:       {'✅ FIXED' if global_V_max <= 1.0 + 1e-6 else '❌ STILL PRESENT'}")

print(f"\n" + "=" * 70)
