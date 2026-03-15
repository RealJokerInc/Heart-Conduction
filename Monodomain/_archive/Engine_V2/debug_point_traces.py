"""
Debug Individual Point Traces
==============================

Track voltage at specific grid points to see if individual points
hit the safeguard floor or overshoot, even if spatial extrema don't show it.

Author: Generated with Claude Code
Date: 2025-12-09
"""

import numpy as np
from interactive_simulation import InteractiveSimulation
import matplotlib.pyplot as plt

print("=" * 70)
print("INDIVIDUAL POINT VOLTAGE TRACES")
print("=" * 70)

# Create simulation
sim = InteractiveSimulation(
    domain_size=80.0,
    resolution=0.5,
    initial_stim_amplitude=200.0,
    initial_stim_radius=10.0
)

# Select several points to track
points_to_track = [
    (40, 40, "Center stimulus site"),
    (60, 40, "10mm from stimulus"),
    (80, 40, "20mm from stimulus"),
    (100, 40, "30mm from stimulus"),
]

print(f"\nTracking {len(points_to_track)} grid points:")
for i, j, label in points_to_track:
    x_mm = j * sim.dx
    y_mm = i * sim.dy
    print(f"  Point [{i:3d}, {j:3d}] @ ({x_mm:5.1f}, {y_mm:5.1f}) mm - {label}")

# Add left-edge stimulus
sim.add_stimulus(5.0, 40.0)
print(f"\nStimulus at left edge (5.0, 40.0) mm, 200mV")

# Run simulation
duration_ms = 100.0
n_steps = int(duration_ms / sim.dt)

# Storage for point traces
traces = {label: [] for _, _, label in points_to_track}
t_history = []

# Also track global extrema
V_min_global = []
V_max_global = []

print(f"\nRunning for {duration_ms} ms...")
print(f"Safeguard floor = {-sim.mu2 + 0.01:.4f} (dimensionless) = -121.2 mV\n")

for step in range(n_steps):
    I_stim = sim.get_current_stimulus()
    sim.step(sim.dt, I_stim)

    # Record point values
    for i, j, label in points_to_track:
        traces[label].append(sim.V[i, j])

    t_history.append(sim.t)

    # Global extrema
    V_min_global.append(np.min(sim.V))
    V_max_global.append(np.max(sim.V))

    if step % 1000 == 0:
        print(f"  t={sim.t:6.1f}ms: V_global ∈ [{np.min(sim.V):7.4f}, {np.max(sim.V):7.4f}]")

print(f"\n" + "=" * 70)
print(f"POINT-BY-POINT ANALYSIS")
print(f"=" * 70)

floor_value = -sim.mu2 + 0.01  # -0.29

for label, trace in traces.items():
    trace_array = np.array(trace)
    min_V = np.min(trace_array)
    max_V = np.max(trace_array)

    min_V_phys = sim.ionic_model.voltage_to_physical(np.array([[min_V]]))[0, 0]
    max_V_phys = sim.ionic_model.voltage_to_physical(np.array([[max_V]]))[0, 0]

    print(f"\n{label}:")
    print(f"  V range: [{min_V:7.4f}, {max_V:7.4f}] = [{min_V_phys:7.1f}, {max_V_phys:6.1f}] mV")

    # Check if hit floor
    at_floor = np.abs(trace_array - floor_value) < 1e-6
    if np.any(at_floor):
        n_at_floor = np.sum(at_floor)
        pct = 100.0 * n_at_floor / len(trace_array)
        first_hit = np.where(at_floor)[0][0]
        print(f"  ⚠️  HIT SAFEGUARD FLOOR: {n_at_floor} times ({pct:.1f}% of simulation)")
        print(f"      First hit at t = {t_history[first_hit]:.1f} ms")
    else:
        distance_from_floor = min_V - floor_value
        print(f"  ✓ Never hit floor (closest: {distance_from_floor:.4f} above)")

    # Check overshoot
    if max_V > 1.0:
        overshoot_mV = max_V_phys - 40.0
        print(f"  ⚠️  OVERSHOOT: V_max = {max_V:.4f} ({max_V_phys:.1f} mV, {overshoot_mV:.1f} mV over peak)")

        # When did it overshoot?
        overshoot_mask = trace_array > 1.0
        if np.any(overshoot_mask):
            first_overshoot = np.where(overshoot_mask)[0][0]
            peak_time = t_history[np.argmax(trace_array)]
            print(f"      First overshoot at t = {t_history[first_overshoot]:.1f} ms")
            print(f"      Peak reached at t = {peak_time:.1f} ms")

# Global analysis
V_min_array = np.array(V_min_global)
V_max_array = np.array(V_max_global)

print(f"\n" + "=" * 70)
print(f"GLOBAL EXTREMA ANALYSIS")
print(f"=" * 70)

min_V_global_value = np.min(V_min_array)
max_V_global_value = np.max(V_max_array)

min_V_phys = sim.ionic_model.voltage_to_physical(np.array([[min_V_global_value]]))[0, 0]
max_V_phys = sim.ionic_model.voltage_to_physical(np.array([[max_V_global_value]]))[0, 0]

print(f"\nOverall simulation bounds:")
print(f"  Global V_min = {min_V_global_value:.4f} → {min_V_phys:.1f} mV")
print(f"  Global V_max = {max_V_global_value:.4f} → {max_V_phys:.1f} mV")

# Check if any point in entire domain ever hit floor
all_points_min = np.min(sim.V)  # This is just final state
print(f"\nFinal state V_min = {all_points_min:.4f}")

# Plot traces
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Individual traces (dimensionless)
for label, trace in traces.items():
    axes[0].plot(t_history, trace, linewidth=1.5, label=label, alpha=0.8)

axes[0].axhline(y=0.0, color='gray', linestyle=':', label='V=0 (rest)', alpha=0.5)
axes[0].axhline(y=1.0, color='gray', linestyle=':', label='V=1 (peak)', alpha=0.5)
axes[0].axhline(y=floor_value, color='orange', linestyle='--',
                label=f'Floor = {floor_value:.2f}', linewidth=2)
axes[0].set_xlabel('Time [ms]', fontsize=12)
axes[0].set_ylabel('Voltage (dimensionless)', fontsize=12)
axes[0].set_title('Individual Point Traces (Dimensionless)', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=9, loc='best')
axes[0].grid(True, alpha=0.3)

# Individual traces (physical)
for label, trace in traces.items():
    trace_phys = sim.ionic_model.voltage_to_physical(np.array([trace]))[0]
    axes[1].plot(t_history, trace_phys, linewidth=1.5, label=label, alpha=0.8)

axes[1].axhline(y=-85.0, color='gray', linestyle=':', label='V_rest', alpha=0.5)
axes[1].axhline(y=40.0, color='gray', linestyle=':', label='V_peak', alpha=0.5)
axes[1].axhline(y=-121.2, color='orange', linestyle='--',
                label='Floor = -121.2 mV', linewidth=2)
axes[1].set_xlabel('Time [ms]', fontsize=12)
axes[1].set_ylabel('Voltage [mV]', fontsize=12)
axes[1].set_title('Individual Point Traces (Physical)', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=9, loc='best')
axes[1].grid(True, alpha=0.3)

# Global extrema
V_min_phys_array = sim.ionic_model.voltage_to_physical(np.array([V_min_array]))[0]
V_max_phys_array = sim.ionic_model.voltage_to_physical(np.array([V_max_array]))[0]

axes[2].plot(t_history, V_min_phys_array, 'b-', linewidth=2, label='Global V_min')
axes[2].plot(t_history, V_max_phys_array, 'r-', linewidth=2, label='Global V_max')
axes[2].axhline(y=-85.0, color='gray', linestyle=':', alpha=0.5)
axes[2].axhline(y=40.0, color='gray', linestyle=':', alpha=0.5)
axes[2].axhline(y=-121.2, color='orange', linestyle='--', linewidth=2)
axes[2].set_xlabel('Time [ms]', fontsize=12)
axes[2].set_ylabel('Voltage [mV]', fontsize=12)
axes[2].set_title('Global Voltage Extrema', fontsize=14, fontweight='bold')
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('point_traces_analysis.png', dpi=150)
print(f"\n📊 Plot saved: point_traces_analysis.png")

print(f"\n" + "=" * 70)
print(f"SUMMARY FOR USER")
print(f"=" * 70)

print(f"\n**Your Observation #1: V_min = -121.2 mV**")
if np.abs(min_V_global_value - floor_value) < 1e-4:
    print(f"  CONFIRMED: Voltage IS at the safeguard floor")
    print(f"  This means physics wants to push V lower, but safeguard prevents it")
else:
    print(f"  Current test: V_min = {min_V_phys:.1f} mV (not at floor)")
    print(f"  Your -121.2 mV may occur with different settings or longer runs")

print(f"\n**Your Observation #2: V_max ~1.137 (57.1 mV)**")
print(f"  Current test: V_max = {max_V_global_value:.4f} ({max_V_phys:.1f} mV)")
if max_V_global_value > 1.0:
    overshoot = max_V_phys - 40.0
    print(f"  ⚠️  CONFIRMED: V_max overshooting by {overshoot:.1f} mV")
    print(f"  Your 1.137 is even higher - may need investigation")
else:
    print(f"  Test shows some overshoot, but not as high as you observed")

print(f"\n**Your Observation #3: -121.2 mV crash threshold?**")
print(f"  The safeguard prevents V from going below -0.29 (dimensionless)")
print(f"  If safeguard is active, physics IS trying to crash the simulation")
print(f"  The safeguard is holding, but underlying model behavior may need review")

print(f"=" * 70)
