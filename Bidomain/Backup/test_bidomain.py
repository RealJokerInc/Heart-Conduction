"""
Test Bidomain Solver with LRd94 Ionic Model
============================================

Runs a 2D wave propagation simulation using the full bidomain model
coupled with the Luo-Rudy 1994 ionic model.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bidomain_solver import BidomainSolver, MonodomainSolver
from luo_rudy_1994 import LuoRudy1994

print("=" * 60)
print("BIDOMAIN SOLVER TEST WITH LRd94")
print("=" * 60)


# =============================================================================
# Test 1: Monodomain solver (faster) for basic validation
# =============================================================================

print("\n--- Test 1: Monodomain Wave Propagation ---")

# Small domain for testing (computational cost is high with LRd94)
params = {
    'Lx': 1.0,          # 1 cm
    'Ly': 0.5,          # 0.5 cm
    'dx': 0.025,        # 250 um spacing
    'dy': 0.025,
    'sigma_il': 3.0,    # mS/cm
    'sigma_it': 0.31,
    'sigma_el': 2.0,
    'sigma_et': 1.65,
    'chi': 2000.0,      # 1/cm
    'C_m': 1.0,         # uF/cm²
}

# Create ionic model
ionic = LuoRudy1994(dt=0.01)

# Create solver
solver = MonodomainSolver(params, ionic_model=ionic)
solver.initialize(V_rest=-84.0)

print(f"Domain: {solver.Lx} x {solver.Ly} cm")
print(f"Grid: {solver.Nx} x {solver.Ny} = {solver.N} nodes")
print(f"Effective conductivity: σ_l = {solver.sigma_l:.3f}, σ_t = {solver.sigma_t:.3f} mS/cm")

# Simulation parameters
dt = 0.01          # ms (same as ionic model)
t_end = 50.0       # ms (short test)
stim_duration = 1.0
stim_amplitude = -80.0

# Stimulus region: left edge
stim_region = (slice(None), slice(0, 3))  # All y, first 3 x columns

print(f"Time step: {dt} ms")
print(f"Simulation duration: {t_end} ms")
print(f"Stimulus: {stim_amplitude} uA/cm² for {stim_duration} ms at t=5 ms")

# Run simulation
print("\nRunning simulation...")
start_time = time.time()

n_steps = int(t_end / dt)
save_interval = 100  # Save every 1 ms
n_saves = n_steps // save_interval

Vm_history = []
t_history = []

solver.I_stim = np.zeros((solver.Ny, solver.Nx))

for step in range(n_steps):
    t = step * dt

    # Apply stimulus at t=5 ms
    if 5.0 <= t < 5.0 + stim_duration:
        solver.I_stim[stim_region] = stim_amplitude
    else:
        solver.I_stim[:] = 0.0

    # Step
    solver.step(dt)

    # Save
    if step % save_interval == 0:
        t_history.append(t)
        Vm_history.append(solver.Vm.copy())

        # Progress
        if step % 1000 == 0:
            V_max = np.max(solver.Vm)
            V_min = np.min(solver.Vm)
            print(f"  t = {t:.1f} ms, V = [{V_min:.1f}, {V_max:.1f}] mV")

elapsed = time.time() - start_time
print(f"\nSimulation completed in {elapsed:.1f} s")
print(f"Performance: {n_steps} steps, {n_steps/elapsed:.0f} steps/s")

# Final state
t_history.append(t_end)
Vm_history.append(solver.Vm.copy())

# Analysis
V_max_final = np.max(Vm_history[-1])
V_min_final = np.min(Vm_history[-1])
print(f"\nFinal state: V = [{V_min_final:.1f}, {V_max_final:.1f}] mV")

# Check if wave propagated
activated = np.any(np.array(Vm_history) > 0, axis=0)
pct_activated = 100 * np.sum(activated) / solver.N
print(f"Percentage of tissue activated (V > 0 mV): {pct_activated:.1f}%")

if pct_activated > 50:
    print("✓ Wave propagation successful!")
else:
    print("✗ Wave did not propagate - check parameters")


# =============================================================================
# Plot results
# =============================================================================

# Create figure with snapshots
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Time points to show
t_points = [0, 5, 10, 20, 30, 40, 50, len(t_history)-1]
t_points = [min(tp, len(t_history)-1) for tp in t_points]

for idx, (ax_row, tidx) in enumerate(zip(axes.flatten(), t_points)):
    t = t_history[tidx] if tidx < len(t_history) else t_end
    Vm = Vm_history[tidx]

    im = ax_row.imshow(Vm, extent=[0, solver.Lx*10, 0, solver.Ly*10],
                       origin='lower', cmap='jet',
                       vmin=-90, vmax=40, aspect='auto')
    ax_row.set_title(f't = {t:.0f} ms', fontsize=12)
    ax_row.set_xlabel('x [mm]')
    ax_row.set_ylabel('y [mm]')

plt.colorbar(im, ax=axes, label='Vm [mV]', shrink=0.8)
plt.suptitle('Monodomain Wave Propagation (LRd94)', fontsize=14)
plt.tight_layout()

output_path = os.path.join(os.path.dirname(__file__), 'monodomain_wave.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved: {output_path}")


# =============================================================================
# Plot single cell trace from center
# =============================================================================

fig2, ax2 = plt.subplots(figsize=(10, 6))

# Extract trace from center of domain
j_center = solver.Ny // 2
i_center = solver.Nx // 2

V_trace = [Vm[j_center, i_center] for Vm in Vm_history]
t_trace = t_history

ax2.plot(t_trace, V_trace, 'b-', linewidth=2)
ax2.set_xlabel('Time [ms]', fontsize=12)
ax2.set_ylabel('Vm [mV]', fontsize=12)
ax2.set_title(f'Membrane Voltage at Center ({i_center}, {j_center})', fontsize=14)
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-100, 60)

output_path2 = os.path.join(os.path.dirname(__file__), 'monodomain_trace.png')
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"Trace plot saved: {output_path2}")


# =============================================================================
# Conduction velocity measurement
# =============================================================================

print("\n--- Conduction Velocity Analysis ---")

# Find activation times at different x positions
threshold = -30.0  # mV activation threshold
j_row = solver.Ny // 2  # Middle row

activation_times = []
x_positions = []

for i in range(solver.Nx):
    # Find first time this node crosses threshold
    for tidx, Vm in enumerate(Vm_history):
        if Vm[j_row, i] > threshold:
            activation_times.append(t_history[tidx])
            x_positions.append(i * solver.dx * 10)  # mm
            break
    else:
        activation_times.append(np.nan)
        x_positions.append(i * solver.dx * 10)

# Plot activation times
fig3, ax3 = plt.subplots(figsize=(10, 6))

valid = ~np.isnan(activation_times)
ax3.plot(np.array(x_positions)[valid], np.array(activation_times)[valid], 'bo-', markersize=4)
ax3.set_xlabel('Position x [mm]', fontsize=12)
ax3.set_ylabel('Activation Time [ms]', fontsize=12)
ax3.set_title('Activation Wavefront', fontsize=14)
ax3.grid(True, alpha=0.3)

# Linear fit for CV
valid_idx = np.where(valid & (np.array(activation_times) > 5) & (np.array(activation_times) < t_end-10))[0]
if len(valid_idx) > 5:
    x_fit = np.array(x_positions)[valid_idx]
    t_fit = np.array(activation_times)[valid_idx]

    # Linear regression
    slope, intercept = np.polyfit(t_fit, x_fit, 1)
    CV = slope  # mm/ms = m/s

    ax3.plot(intercept + slope * t_fit, t_fit, 'r--', linewidth=2,
             label=f'CV = {CV:.2f} mm/ms = {CV*100:.1f} cm/s')
    ax3.legend(fontsize=11)

    print(f"Conduction velocity: {CV:.2f} mm/ms = {CV*100:.1f} cm/s")
    print(f"Expected range: 30-100 cm/s for ventricular tissue")

output_path3 = os.path.join(os.path.dirname(__file__), 'activation_wavefront.png')
plt.savefig(output_path3, dpi=150, bbox_inches='tight')
print(f"Activation plot saved: {output_path3}")


print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
