"""
Debug Diffusion Computation
============================

Test just the diffusion step in isolation.

Author: Generated with Claude Code
Date: 2025-12-09
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/lemon/Documents/Python/Heart Conduction/Engine_V1')

# Import both versions
from simulate_infarct import InfarctSimulation as SimV1
from simulate_infarct_v2 import InfarctSimulationV2 as SimV2, compute_diffusion_flux_based

print("=" * 70)
print("DIFFUSION DEBUG - Single Step Analysis")
print("=" * 70)

# Create both sims
sim_v1 = SimV1(
    domain_size=80.0,
    resolution=0.5,
    infarct_radius=10.0,
    D_parallel=1.0,
    D_perp=0.5,
    T_scale=10.0
)

sim_v2 = SimV2(
    domain_size=80.0,
    resolution=0.5,
    infarct_radius=10.0,
    D_parallel=1.0,
    D_perp=0.5,
    T_scale=10.0
)

# Set up identical initial condition with stimulus
V_init = np.zeros((sim_v1.ny, sim_v1.nx))
V_init[:, 0] = 0.24  # Left edge stimulus

sim_v1.V = V_init.copy()
sim_v2.V = V_init.copy()

print(f"\nInitial V:")
print(f"  V_max = {np.max(V_init):.6f}")
print(f"  V_left_edge_max = {np.max(V_init[:, 0]):.6f}")
print(f"  V_left_edge_min = {np.min(V_init[:, 0]):.6f}")

# Compute diffusion with V1
print("\n" + "-" * 70)
print("V1 DIFFUSION")
print("-" * 70)
lap_v1 = sim_v1.compute_laplacian(sim_v1.V)
print(f"Laplacian stats:")
print(f"  min = {np.min(lap_v1):.6e}")
print(f"  max = {np.max(lap_v1):.6e}")
print(f"  mean = {np.mean(lap_v1):.6e}")
print(f"  std = {np.std(lap_v1):.6e}")

# Where is laplacian largest?
max_idx = np.unravel_index(np.argmax(np.abs(lap_v1)), lap_v1.shape)
print(f"  Max |lap| at: [{max_idx[0]}, {max_idx[1]}] = {lap_v1[max_idx[0], max_idx[1]]:.6e}")

# Compute diffusion with V2
print("\n" + "-" * 70)
print("V2 DIFFUSION")
print("-" * 70)
lap_v2 = compute_diffusion_flux_based(sim_v2.V, sim_v2.Dxx, sim_v2.Dyy, sim_v2.Dxy, sim_v2.dx, sim_v2.dy)
print(f"Laplacian stats:")
print(f"  min = {np.min(lap_v2):.6e}")
print(f"  max = {np.max(lap_v2):.6e}")
print(f"  mean = {np.mean(lap_v2):.6e}")
print(f"  std = {np.std(lap_v2):.6e}")

max_idx = np.unravel_index(np.argmax(np.abs(lap_v2)), lap_v2.shape)
print(f"  Max |lap| at: [{max_idx[0]}, {max_idx[1]}] = {lap_v2[max_idx[0], max_idx[1]]:.6e}")

# Compare
print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)

diff = np.abs(lap_v1 - lap_v2)
print(f"\nAbsolute difference:")
print(f"  max = {np.max(diff):.6e}")
print(f"  mean = {np.mean(diff):.6e}")
print(f"  std = {np.std(diff):.6e}")

if np.max(diff) > 1e-10:
    # Find where they differ most
    max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
    print(f"\nMax difference at [{max_diff_idx[0]}, {max_diff_idx[1]}]:")
    print(f"  V1: {lap_v1[max_diff_idx[0], max_diff_idx[1]]:.6e}")
    print(f"  V2: {lap_v2[max_diff_idx[0], max_diff_idx[1]]:.6e}")
    print(f"  Diff: {diff[max_diff_idx[0], max_diff_idx[1]]:.6e}")

    # Show context around that point
    i, j = max_diff_idx
    print(f"\n  V field around [{i}, {j}]:")
    for di in range(-2, 3):
        row_str = "  "
        for dj in range(-2, 3):
            ni, nj = i + di, j + dj
            if 0 <= ni < V_init.shape[0] and 0 <= nj < V_init.shape[1]:
                row_str += f"{V_init[ni, nj]:7.4f} "
            else:
                row_str += "   -    "
        if di == 0 and j >= 0:
            row_str += " <-- center"
        print(row_str)

# Apply diffusion step
dt = 0.01
V1_after_diffusion = sim_v1.V + dt * lap_v1
V2_after_diffusion = sim_v2.V + dt * lap_v2

print("\n" + "=" * 70)
print("AFTER ONE DIFFUSION STEP (dt=0.01ms)")
print("=" * 70)

print(f"\nV1:")
print(f"  V_max = {np.max(V1_after_diffusion):.6f}")
print(f"  V_at_[80,1] = {V1_after_diffusion[80, 1]:.6f} (just right of stimulus)")

print(f"\nV2:")
print(f"  V_max = {np.max(V2_after_diffusion):.6f}")
print(f"  V_at_[80,1] = {V2_after_diffusion[80, 1]:.6f} (just right of stimulus)")

print(f"\nDifference in V after diffusion:")
print(f"  max = {np.max(np.abs(V1_after_diffusion - V2_after_diffusion)):.6e}")

if np.max(np.abs(V1_after_diffusion - V2_after_diffusion)) < 1e-6:
    print("\n✓ DIFFUSION COMPUTATIONS ARE IDENTICAL!")
else:
    print("\n✗ DIFFUSION COMPUTATIONS DIFFER!")
