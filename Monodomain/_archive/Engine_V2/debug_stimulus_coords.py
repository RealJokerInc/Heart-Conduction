"""
Debug Stimulus Coordinate System
=================================

The stimulus isn't affecting the tissue at the right location!

Author: Generated with Claude Code
Date: 2025-12-09
"""

import numpy as np
import matplotlib.pyplot as plt
from interactive_simulation import InteractiveSimulation

print("=" * 70)
print("DEBUGGING STIMULUS COORDINATES")
print("=" * 70)

# Create simulation
sim = InteractiveSimulation(
    domain_size=80.0,
    resolution=0.5,
    initial_stim_amplitude=100.0,
    initial_stim_radius=5.0
)

print(f"\nDomain info:")
print(f"  Lx = {sim.Lx} mm, Ly = {sim.Ly} mm")
print(f"  nx = {sim.nx}, ny = {sim.ny}")
print(f"  dx = {sim.dx} mm, dy = {sim.dy} mm")

# Check meshgrid
print(f"\nMeshgrid X:")
print(f"  Shape: {sim.X.shape}")
print(f"  Range: [{np.min(sim.X):.2f}, {np.max(sim.X):.2f}] mm")
print(f"  X[0, 0] = {sim.X[0, 0]:.2f} (bottom-left)")
print(f"  X[0, -1] = {sim.X[0, -1]:.2f} (bottom-right)")
print(f"  X[-1, 0] = {sim.X[-1, 0]:.2f} (top-left)")

print(f"\nMeshgrid Y:")
print(f"  Shape: {sim.Y.shape}")
print(f"  Range: [{np.min(sim.Y):.2f}, {np.max(sim.Y):.2f}] mm")
print(f"  Y[0, 0] = {sim.Y[0, 0]:.2f} (bottom-left)")
print(f"  Y[0, -1] = {sim.Y[0, -1]:.2f} (bottom-right)")
print(f"  Y[-1, 0] = {sim.Y[-1, 0]:.2f} (top-left)")

# Add stimulus at center
x_stim, y_stim = 40.0, 40.0
print(f"\nAdding stimulus at ({x_stim}, {y_stim}) mm...")
sim.add_stimulus(x_stim, y_stim)

# Compute stimulus
I_stim = sim.get_current_stimulus()

print(f"\nStimulus array:")
print(f"  Shape: {I_stim.shape}")
print(f"  Max value: {np.max(I_stim):.6f}")
print(f"  Number of non-zero points: {np.sum(I_stim > 0)}")

# Find where stimulus is actually applied
stim_indices = np.where(I_stim > 0)
if len(stim_indices[0]) > 0:
    i_center = stim_indices[0][len(stim_indices[0])//2]
    j_center = stim_indices[1][len(stim_indices[1])//2]

    print(f"\nStimulus location in grid:")
    print(f"  Grid indices: i={i_center}, j={j_center}")
    print(f"  Corresponds to: x={sim.X[i_center, j_center]:.2f}, y={sim.Y[i_center, j_center]:.2f} mm")
    print(f"  Expected: x={x_stim:.2f}, y={y_stim:.2f} mm")

    # Check distance
    dist_computed = np.sqrt((sim.X - x_stim)**2 + (sim.Y - y_stim)**2)
    dist_at_center = dist_computed[i_center, j_center]
    print(f"  Distance from stimulus center: {dist_at_center:.3f} mm (should be ~0)")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: X coordinates
im1 = axes[0].imshow(sim.X, origin='lower', cmap='viridis', aspect='equal')
axes[0].plot(x_stim/sim.dx, y_stim/sim.dy, 'r*', markersize=20, label=f'Stimulus ({x_stim}, {y_stim})')
axes[0].set_title('X Coordinate Field', fontweight='bold')
axes[0].set_xlabel('j (column index)')
axes[0].set_ylabel('i (row index)')
plt.colorbar(im1, ax=axes[0], label='x (mm)')
axes[0].legend()

# Plot 2: Y coordinates
im2 = axes[1].imshow(sim.Y, origin='lower', cmap='plasma', aspect='equal')
axes[1].plot(x_stim/sim.dx, y_stim/sim.dy, 'r*', markersize=20, label=f'Stimulus ({x_stim}, {y_stim})')
axes[1].set_title('Y Coordinate Field', fontweight='bold')
axes[1].set_xlabel('j (column index)')
axes[1].set_ylabel('i (row index)')
plt.colorbar(im2, ax=axes[1], label='y (mm)')
axes[1].legend()

# Plot 3: Stimulus mask
im3 = axes[2].imshow(I_stim, origin='lower', cmap='hot', aspect='equal')
axes[2].plot(x_stim/sim.dx, y_stim/sim.dy, 'b*', markersize=20, label='Expected location')
axes[2].set_title('Stimulus Application', fontweight='bold')
axes[2].set_xlabel('j (column index)')
axes[2].set_ylabel('i (row index)')
plt.colorbar(im3, ax=axes[2], label='I_stim')
axes[2].legend()

plt.tight_layout()
plt.savefig('debug_stimulus_coords.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: debug_stimulus_coords.png")

# Now apply stimulus and check V
print(f"\n" + "=" * 70)
print("APPLYING STIMULUS FOR 2MS")
print("=" * 70)

V_before = sim.V.copy()
print(f"\nBefore stimulus:")
print(f"  V at [i={i_center}, j={j_center}]: {V_before[i_center, j_center]:.6f}")
print(f"  V_max: {np.max(V_before):.6f}")

# Apply for 2ms
for step in range(200):
    I_stim = sim.get_current_stimulus()
    sim.step(sim.dt, I_stim)

V_after = sim.V.copy()
print(f"\nAfter 2ms stimulus:")
print(f"  V at [i={i_center}, j={j_center}]: {V_after[i_center, j_center]:.6f}")
print(f"  V_max: {np.max(V_after):.6f}")
print(f"  Change at stimulus center: {V_after[i_center, j_center] - V_before[i_center, j_center]:.6f}")

# Check if ANY point changed significantly
V_change = V_after - V_before
max_change_idx = np.unravel_index(np.argmax(np.abs(V_change)), V_change.shape)
print(f"\nMaximum V change:")
print(f"  Location: [i={max_change_idx[0]}, j={max_change_idx[1]}]")
print(f"  Corresponds to: x={sim.X[max_change_idx]:.2f}, y={sim.Y[max_change_idx]:.2f} mm")
print(f"  Change: {V_change[max_change_idx]:.6f}")

# Visualize V change
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

im1 = axes2[0].imshow(I_stim, origin='lower', cmap='hot', aspect='equal')
axes2[0].set_title('Stimulus Applied (last step)', fontweight='bold')
plt.colorbar(im1, ax=axes2[0], label='I_stim')

im2 = axes2[1].imshow(V_change, origin='lower', cmap='RdBu_r', aspect='equal')
axes2[1].set_title('V Change After 2ms', fontweight='bold')
axes2[1].plot(j_center, i_center, 'r*', markersize=20, label='Stimulus center')
axes2[1].plot(max_change_idx[1], max_change_idx[0], 'g*', markersize=20, label='Max change')
plt.colorbar(im2, ax=axes2[1], label='ΔV')
axes2[1].legend()

plt.tight_layout()
plt.savefig('debug_stimulus_effect.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: debug_stimulus_effect.png")

print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)

if np.max(np.abs(V_change)) < 0.01:
    print("❌ CONFIRMED: Stimulus has almost no effect on tissue!")
    print("\nPossible causes:")
    print("  1. Stimulus in wrong location (coordinates mismatch)")
    print("  2. Ionic current immediately counteracts stimulus")
    print("  3. Time step integration issue")
else:
    print(f"✓ Stimulus is affecting tissue (max change: {np.max(np.abs(V_change)):.4f})")
