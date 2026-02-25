"""
Debug V1 vs V2 Comparison
==========================

Compare diffusion and wave propagation between V1 and V2.

Author: Generated with Claude Code
Date: 2025-12-09
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/lemon/Documents/Python/Heart Conduction/Engine_V1')

# Import both versions
from simulate_infarct import InfarctSimulation as SimV1
from simulate_infarct_v2 import InfarctSimulationV2 as SimV2

print("=" * 70)
print("DEBUGGING V1 vs V2")
print("=" * 70)

# Create both simulations
print("\nCreating V1 simulation...")
sim_v1 = SimV1(
    domain_size=80.0,
    resolution=0.5,
    infarct_radius=10.0,
    D_parallel=1.0,
    D_perp=0.5,
    T_scale=10.0
)

print("\nCreating V2 simulation...")
sim_v2 = SimV2(
    domain_size=80.0,
    resolution=0.5,
    infarct_radius=10.0,
    D_parallel=1.0,
    D_perp=0.5,
    T_scale=10.0
)

# Test diffusion computation on same field
print("\n" + "=" * 70)
print("TEST 1: Diffusion Computation")
print("=" * 70)

# Create test voltage field
V_test = np.zeros((sim_v1.ny, sim_v1.nx))
# Add a Gaussian bump in the center
center_i, center_j = sim_v1.ny // 2, sim_v1.nx // 2
y_idx, x_idx = np.ogrid[:sim_v1.ny, :sim_v1.nx]
r_sq = (x_idx - center_j)**2 + (y_idx - center_i)**2
V_test = 0.5 * np.exp(-r_sq / (10.0**2))

print(f"Test field: V_max = {np.max(V_test):.4f}, V_min = {np.min(V_test):.4f}")

# Compute diffusion with V1
lap_v1 = sim_v1.compute_laplacian(V_test)
print(f"\nV1 laplacian: min={np.min(lap_v1):.6f}, max={np.max(lap_v1):.6f}, mean={np.mean(lap_v1):.6f}")

# Compute diffusion with V2
from simulate_infarct_v2 import compute_diffusion_flux_based
lap_v2 = compute_diffusion_flux_based(V_test, sim_v2.Dxx, sim_v2.Dyy, sim_v2.Dxy, sim_v2.dx, sim_v2.dy)
print(f"V2 laplacian: min={np.min(lap_v2):.6f}, max={np.max(lap_v2):.6f}, mean={np.mean(lap_v2):.6f}")

diff = np.abs(lap_v1 - lap_v2)
print(f"\nDifference: max={np.max(diff):.6e}, mean={np.mean(diff):.6e}")

if np.max(diff) > 1e-6:
    print("⚠️  WARNING: Large difference in diffusion computation!")
    print(f"   Max relative error: {np.max(diff) / (np.max(np.abs(lap_v1)) + 1e-10):.2e}")
else:
    print("✓ Diffusion computations match!")

# Test ionic model
print("\n" + "=" * 70)
print("TEST 2: Ionic Model")
print("=" * 70)

V_test_ion = 0.5
w_test_ion = 0.1
dt = 0.01
dtau = sim_v1.ionic_model.time_to_tau(dt)
I_stim_test = 0.5

# V1 ionic model
V_new_v1, w_new_v1 = sim_v1.ionic_model.step_explicit_euler(
    V_test_ion, w_test_ion, dtau, I_stim_test
)
print(f"V1: V={V_test_ion:.4f} -> {V_new_v1:.4f}, w={w_test_ion:.4f} -> {w_new_v1:.4f}")

# V2 ionic model (Numba)
from simulate_infarct_v2 import ionic_step_numba
V_arr = np.array([[V_test_ion]])
w_arr = np.array([[w_test_ion]])
I_stim_arr = np.array([[I_stim_test]])
tissue_mask_arr = np.array([[True]])

ionic_step_numba(
    V_arr, w_arr, dtau, I_stim_arr, tissue_mask_arr,
    sim_v2.k, sim_v2.a, sim_v2.epsilon0, sim_v2.mu1, sim_v2.mu2,
    sim_v2.epsilon_rest, sim_v2.V_threshold, sim_v2.k_sigmoid
)

V_new_v2 = V_arr[0, 0]
w_new_v2 = w_arr[0, 0]
print(f"V2: V={V_test_ion:.4f} -> {V_new_v2:.4f}, w={w_test_ion:.4f} -> {w_new_v2:.4f}")

V_diff = abs(V_new_v1 - V_new_v2)
w_diff = abs(w_new_v1 - w_new_v2)
print(f"\nDifference: ΔV={V_diff:.6e}, Δw={w_diff:.6e}")

if V_diff > 1e-6 or w_diff > 1e-6:
    print("⚠️  WARNING: Ionic models don't match!")
else:
    print("✓ Ionic models match!")

# Test full step
print("\n" + "=" * 70)
print("TEST 3: Full Time Step")
print("=" * 70)

# Reset both simulations
sim_v1.V = np.zeros((sim_v1.ny, sim_v1.nx))
sim_v1.w = np.zeros((sim_v1.ny, sim_v1.nx))

sim_v2.V = np.zeros((sim_v2.ny, sim_v2.nx))
sim_v2.w = np.zeros((sim_v2.ny, sim_v2.nx))

# Add stimulus at left edge
I_stim = np.zeros((sim_v1.ny, sim_v1.nx))
I_stim[:, 0] = 0.24  # Same as 30mV normalized

print("Taking one time step with stimulus...")

# V1 step
sim_v1.step(dt, I_stim)
print(f"V1 after step: V_max={np.max(sim_v1.V):.6f}, V_left_edge={np.max(sim_v1.V[:, 0]):.6f}")

# V2 step
sim_v2.step(dt, I_stim)
print(f"V2 after step: V_max={np.max(sim_v2.V):.6f}, V_left_edge={np.max(sim_v2.V[:, 0]):.6f}")

V_diff_step = np.abs(sim_v1.V - sim_v2.V)
print(f"\nV difference after step: max={np.max(V_diff_step):.6e}, mean={np.mean(V_diff_step):.6e}")

# Run 10 steps
print("\nRunning 10 steps...")
for i in range(9):
    sim_v1.step(dt, np.zeros((sim_v1.ny, sim_v1.nx)))
    sim_v2.step(dt, np.zeros((sim_v2.ny, sim_v2.nx)))

print(f"V1 after 10 steps: V_max={np.max(sim_v1.V):.6f}")
print(f"V2 after 10 steps: V_max={np.max(sim_v2.V):.6f}")

print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)

if np.max(sim_v1.V) > 0.3 and np.max(sim_v2.V) < 0.1:
    print("❌ V2 wave is dying while V1 propagates!")
    print("   Issue is in V2 implementation")
elif np.max(sim_v2.V) > 0.3 and np.max(sim_v1.V) < 0.1:
    print("❌ V1 wave is dying while V2 propagates!")
    print("   Issue is in V1 implementation")
elif np.max(sim_v1.V) < 0.1 and np.max(sim_v2.V) < 0.1:
    print("❌ Both waves dying - check stimulus amplitude")
else:
    print("✓ Both simulations working similarly")
