"""
Reproduce User-Reported Crash
==============================

Create conditions that trigger the instability:
- Strong stimulus to ensure AP forms
- Run until wave propagates and crashes

Author: Generated with Claude Code
Date: 2025-12-09
"""

import numpy as np
from debug_stability import StabilityMonitor, run_controlled_test
from interactive_simulation import InteractiveSimulation

print("=" * 70)
print("REPRODUCING USER-REPORTED CRASH")
print("=" * 70)
print("\nConditions:")
print("  - Strong stimulus (50mV) to ensure AP forms")
print("  - Run for 100ms to allow propagation")
print("  - Monitor for V_min crash")
print("=" * 70)

# Create simulation with stronger stimulus
sim = InteractiveSimulation(
    domain_size=80.0,
    resolution=0.5,
    initial_stim_amplitude=50.0,  # Stronger stimulus
    initial_stim_radius=5.0
)

# Create monitor
monitor = StabilityMonitor(sim)

# Add stimulus at center
print(f"\nAdding 50mV stimulus at center...")
sim.add_stimulus(40.0, 40.0)

# Run with detailed monitoring
duration_ms = 100.0
dt = sim.dt
n_steps = int(duration_ms / dt)
record_every = 5  # Record frequently

print(f"Running for {duration_ms} ms...")
print(f"Recording every {record_every} steps\n")

try:
    for step in range(n_steps):
        t = step * dt

        # Get stimulus
        I_stim = sim.get_current_stimulus()

        # Step forward
        sim.step(dt, I_stim)

        # Monitor
        if step % record_every == 0:
            monitor.record(t)

            if not monitor.check_stability(t):
                print(f"\n⚠️  CRASH DETECTED at t = {t:.2f} ms")
                break

            # Detailed progress
            if step % 500 == 0:
                V_min = np.min(sim.V)
                V_max = np.max(sim.V)
                V_min_phys = sim.ionic_model.voltage_to_physical(np.array([[V_min]]))[0,0]
                V_max_phys = sim.ionic_model.voltage_to_physical(np.array([[V_max]]))[0,0]

                print(f"  t = {t:6.1f} ms | V ∈ [{V_min:8.4f}, {V_max:8.4f}] | Physical: [{V_min_phys:7.1f}, {V_max_phys:6.1f}] mV", end='')

                if V_min < -0.5:
                    print(f"  ⚠️  V_min very negative!", end='')
                if V_max > 0.8:
                    print(f"  ✓ AP forming", end='')
                print()

    # Final record
    monitor.record(sim.t)

except Exception as e:
    print(f"\n❌ Exception: {e}")
    import traceback
    traceback.print_exc()
    monitor.crashed = True
    monitor.crash_time = sim.t
    monitor.crash_state = {'V': sim.V.copy(), 'w': sim.w.copy(), 'reason': f'Exception: {e}'}

# Generate report
monitor.generate_report(save_prefix='crash_reproduction')

print("\n" + "=" * 70)
print("CHECK DIAGNOSTIC PLOTS:")
print("=" * 70)
print("  crash_reproduction_timeseries.png")
print("  crash_reproduction_critical_values.png")
if monitor.crashed:
    print("  crash_reproduction_crash_state.png")
print("\nLook for:")
print("  1. When does v + μ₂ approach zero?")
print("  2. Does epsilon go negative or huge?")
print("  3. Rate of V_min decrease")
