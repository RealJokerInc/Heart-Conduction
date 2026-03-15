"""
Test Wave Propagation with Strong Stimulus
===========================================

This validates the most critical fix: preventing crashes during propagation.
Uses 200mV stimulus which we know can trigger APs.

Author: Generated with Claude Code
Date: 2025-12-09
"""

import numpy as np
from interactive_simulation import InteractiveSimulation
import matplotlib.pyplot as plt

print("=" * 70)
print("PROPAGATION TEST WITH 200mV STIMULUS")
print("=" * 70)

sim = InteractiveSimulation(
    domain_size=80.0,
    resolution=0.5,
    initial_stim_amplitude=200.0,  # Strong enough to reliably trigger AP
    initial_stim_radius=10.0
)

# Left edge stimulus for half-circle wave
x_stim = 5.0
y_stim = 40.0
sim.add_stimulus(x_stim, y_stim)
print(f"\nAdding 200mV stimulus at left edge ({x_stim}, {y_stim}) mm...")
print(f"This should trigger a propagating wave without crashes.")
print(f"\nRunning for 100ms with detailed monitoring...")

# Run for 100ms
duration_ms = 100.0
n_steps = int(duration_ms / sim.dt)
crashed = False
V_min_history = []
V_max_history = []
t_history = []

try:
    for step in range(n_steps):
        I_stim = sim.get_current_stimulus()
        sim.step(sim.dt, I_stim)

        # Record history
        V_min_history.append(np.min(sim.V))
        V_max_history.append(np.max(sim.V))
        t_history.append(sim.t)

        # Check for crash
        if np.isnan(sim.V).any() or np.isinf(sim.V).any():
            crashed = True
            print(f"\n❌ CRASH at t={sim.t:.2f}ms: NaN/Inf detected")
            break

        # Progress report
        if step % 1000 == 0:
            V_min = np.min(sim.V)
            V_max = np.max(sim.V)
            v_plus_mu2_min = np.min(sim.V + sim.mu2)
            V_min_phys = sim.ionic_model.voltage_to_physical(np.array([[V_min]]))[0, 0]
            V_max_phys = sim.ionic_model.voltage_to_physical(np.array([[V_max]]))[0, 0]

            # Find wavefront
            activated = sim.V > 0.5
            if np.any(activated):
                y_indices, x_indices = np.where(activated)
                x_mm = x_indices * sim.dx
                x_front = np.max(x_mm)
                wave_info = f", wavefront @ x={x_front:.1f}mm"
            else:
                wave_info = ""

            print(f"  t={sim.t:6.1f}ms: V∈[{V_min_phys:7.1f}, {V_max_phys:6.1f}]mV, "
                  f"min(v+μ₂)={v_plus_mu2_min:.4f}{wave_info}")

    # Final analysis
    if not crashed:
        print(f"\n✅ SUCCESS: Simulation completed without crashes!")
        print(f"\nFinal statistics:")
        print(f"  V_min = {np.min(sim.V):.4f} ({sim.ionic_model.voltage_to_physical(np.array([[np.min(sim.V)]]))[0, 0]:.1f} mV)")
        print(f"  V_max = {np.max(sim.V):.4f} ({sim.ionic_model.voltage_to_physical(np.array([[np.max(sim.V)]]))[0, 0]:.1f} mV)")
        print(f"  min(v + μ₂) = {np.min(sim.V + sim.mu2):.4f}")

        # Check wave propagation
        activated_final = sim.V > 0.5
        if np.any(activated_final):
            y_indices, x_indices = np.where(activated_final)
            x_mm = x_indices * sim.dx
            x_front_final = np.max(x_mm)
            propagation_distance = x_front_final - x_stim

            print(f"\n✅ Wave propagated {propagation_distance:.1f}mm from stimulus point")

            # Estimate CV
            # Find time to 50% activation at x=20mm
            cv_mm_per_ms = propagation_distance / duration_ms
            cv_mm_per_sec = cv_mm_per_ms * 1000.0
            print(f"   Estimated CV ≈ {cv_mm_per_sec:.0f} mm/s")

            # Check activation coverage
            percent_activated = 100.0 * np.sum(activated_final) / activated_final.size
            print(f"   Tissue activated: {percent_activated:.1f}%")
        else:
            print(f"\n⚠️  No significant wave propagation (V_max < 0.5)")

        # Plot results
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Time series
        V_min_phys = sim.ionic_model.voltage_to_physical(np.array([V_min_history]))[0]
        V_max_phys = sim.ionic_model.voltage_to_physical(np.array([V_max_history]))[0]

        axes[0, 0].plot(t_history, V_min_phys, 'b-', label='V_min')
        axes[0, 0].plot(t_history, V_max_phys, 'r-', label='V_max')
        axes[0, 0].set_xlabel('Time [ms]')
        axes[0, 0].set_ylabel('Voltage [mV]')
        axes[0, 0].set_title('Voltage Extrema vs Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Critical value
        v_plus_mu2_history = [v + sim.mu2 for v in V_min_history]
        axes[0, 1].plot(t_history, v_plus_mu2_history, 'g-')
        axes[0, 1].axhline(y=0.0, color='r', linestyle='--', label='Division by zero')
        axes[0, 1].axhline(y=0.01, color='orange', linestyle='--', label='Safeguard threshold')
        axes[0, 1].set_xlabel('Time [ms]')
        axes[0, 1].set_ylabel('v + μ₂')
        axes[0, 1].set_title('Division by Zero Monitor')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Final voltage map
        V_phys = sim.ionic_model.voltage_to_physical(sim.V)
        im = axes[1, 0].imshow(V_phys, origin='lower', extent=[0, sim.Lx, 0, sim.Ly],
                               cmap='turbo', vmin=-85, vmax=40)
        axes[1, 0].plot(x_stim, y_stim, 'w*', markersize=15, label='Stimulus')
        axes[1, 0].set_xlabel('x [mm]')
        axes[1, 0].set_ylabel('y [mm]')
        axes[1, 0].set_title(f'Voltage at t={sim.t:.1f}ms')
        axes[1, 0].legend()
        plt.colorbar(im, ax=axes[1, 0], label='Voltage [mV]')

        # Activation map
        activation = (np.array(V_max_history).max(axis=0).reshape(sim.ny, sim.nx) > 0.5).astype(float)
        axes[1, 1].imshow(activation, origin='lower', extent=[0, sim.Lx, 0, sim.Ly],
                         cmap='gray', vmin=0, vmax=1)
        axes[1, 1].plot(x_stim, y_stim, 'r*', markersize=15, label='Stimulus')
        axes[1, 1].set_xlabel('x [mm]')
        axes[1, 1].set_ylabel('y [mm]')
        axes[1, 1].set_title('Tissue Ever Activated (V > 0.5)')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig('test_propagation_results.png', dpi=150)
        print(f"\n📊 Results saved to: test_propagation_results.png")

except Exception as e:
    crashed = True
    print(f"\n❌ EXCEPTION: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 70)
print("CRITICAL VALIDATION SUMMARY")
print("=" * 70)
print(f"Fix #2 (Prevent Crashes): {'✅ PASSED' if not crashed else '❌ FAILED'}")
if not crashed:
    safeguard_ok = np.min(sim.V + sim.mu2) >= 0.01
    print(f"Safeguard Active:         {'✅ YES' if safeguard_ok else '⚠️  MARGINAL'}")
    propagation_ok = np.any(sim.V > 0.5)
    print(f"Wave Propagation:         {'✅ YES' if propagation_ok else '❌ NO'}")
print("=" * 70)
