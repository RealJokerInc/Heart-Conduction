"""
Phase 1: Stability Debug - Characterize Failure Mode
====================================================

Focus areas based on user observations:
1. V_min crashes during AP propagation (not at stimulus)
2. Hypothesis 1: Division by zero in epsilon when v < -mu2
3. Stimulus application issue: 100mV stim only gives 17mV change
4. Boundary condition review

Author: Generated with Claude Code
Date: 2025-12-09
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from collections import deque
import sys

from parameters import create_default_parameters
from aliev_panfilov_fixed import AlievPanfilovModel
from simulate_infarct_v2 import compute_diffusion_flux_based, ionic_step_numba


class StabilityMonitor:
    """Monitor simulation stability and detect instability."""

    def __init__(self, sim):
        self.sim = sim

        # Time series data
        self.times = []
        self.V_min = []
        self.V_max = []
        self.w_min = []
        self.w_max = []

        # Critical values for Hypothesis 1
        self.v_plus_mu2_min = []  # Monitor (v + mu2) for division by zero
        self.epsilon_min = []
        self.epsilon_max = []

        # Ionic currents
        self.I_ion_min = []
        self.I_ion_max = []

        # Spatial statistics
        self.V_std = []
        self.w_std = []

        # Crash detection
        self.crashed = False
        self.crash_time = None
        self.crash_state = None

    def record(self, t):
        """Record current state."""
        V = self.sim.V
        w = self.sim.w

        self.times.append(t)
        self.V_min.append(np.min(V))
        self.V_max.append(np.max(V))
        self.w_min.append(np.min(w))
        self.w_max.append(np.max(w))

        # CRITICAL: Monitor (v + mu2) for division by zero
        v_plus_mu2 = V + self.sim.mu2
        self.v_plus_mu2_min.append(np.min(v_plus_mu2))

        # Compute epsilon field
        # eps = epsilon0 + epsilon_rest * sigmoid + (mu1 * w) / (v + mu2)
        sigmoid = 1.0 / (1.0 + np.exp(self.sim.k_sigmoid * (V - self.sim.V_threshold)))

        # Safe epsilon computation with warning
        epsilon = np.zeros_like(V)
        safe_mask = v_plus_mu2 > 1e-10  # Avoid division by zero

        epsilon[safe_mask] = (self.sim.epsilon0 +
                              self.sim.epsilon_rest * sigmoid[safe_mask] +
                              (self.sim.mu1 * w[safe_mask]) / v_plus_mu2[safe_mask])
        epsilon[~safe_mask] = self.sim.epsilon0  # Fallback for problematic points

        self.epsilon_min.append(np.min(epsilon))
        self.epsilon_max.append(np.max(epsilon))

        # Compute ionic current
        I_ion = self.sim.k * V * (V - self.sim.a) * (V - 1.0) + V * w
        self.I_ion_min.append(np.min(I_ion))
        self.I_ion_max.append(np.max(I_ion))

        # Spatial variance (indicates wave activity)
        self.V_std.append(np.std(V))
        self.w_std.append(np.std(w))

    def check_stability(self, t):
        """Check if simulation is still stable."""
        V = self.sim.V
        w = self.sim.w

        # Check for NaN or Inf
        if np.any(np.isnan(V)) or np.any(np.isinf(V)):
            self.crashed = True
            self.crash_time = t
            self.crash_state = {'V': V.copy(), 'w': w.copy(), 'reason': 'NaN/Inf in V'}
            return False

        if np.any(np.isnan(w)) or np.any(np.isinf(w)):
            self.crashed = True
            self.crash_time = t
            self.crash_state = {'V': V.copy(), 'w': w.copy(), 'reason': 'NaN/Inf in w'}
            return False

        # Check for extreme values
        V_min = np.min(V)
        V_max = np.max(V)
        w_min = np.min(w)
        w_max = np.max(w)

        if V_min < -5.0 or V_max > 5.0:
            self.crashed = True
            self.crash_time = t
            self.crash_state = {'V': V.copy(), 'w': w.copy(), 'reason': f'V out of bounds: [{V_min:.2f}, {V_max:.2f}]'}
            return False

        if w_min < -5.0 or w_max > 5.0:
            self.crashed = True
            self.crash_time = t
            self.crash_state = {'V': V.copy(), 'w': w.copy(), 'reason': f'w out of bounds: [{w_min:.2f}, {w_max:.2f}]'}
            return False

        return True

    def generate_report(self, save_prefix='stability_debug'):
        """Generate comprehensive diagnostic report."""
        print("\n" + "=" * 70)
        print("STABILITY ANALYSIS REPORT")
        print("=" * 70)

        times = np.array(self.times)
        V_min_arr = np.array(self.V_min)
        V_max_arr = np.array(self.V_max)

        if self.crashed:
            print(f"\n⚠️  CRASH DETECTED at t = {self.crash_time:.2f} ms")
            print(f"    Reason: {self.crash_state['reason']}")
            print(f"    V range at crash: [{np.min(self.crash_state['V']):.6f}, {np.max(self.crash_state['V']):.6f}]")
            print(f"    w range at crash: [{np.min(self.crash_state['w']):.6f}, {np.max(self.crash_state['w']):.6f}]")
        else:
            print(f"\n✓ Simulation completed without crash")
            print(f"    Duration: {times[-1]:.2f} ms")

        # Hypothesis 1 check: Division by zero
        print("\n" + "-" * 70)
        print("HYPOTHESIS 1 CHECK: Division by Zero in Epsilon")
        print("-" * 70)

        v_plus_mu2_min_arr = np.array(self.v_plus_mu2_min)
        critical_threshold = 0.0  # mu2 = 0.3, so v + mu2 should stay > 0

        if np.any(v_plus_mu2_min_arr <= critical_threshold):
            print(f"⚠️  CRITICAL: (v + mu2) reached {np.min(v_plus_mu2_min_arr):.6f}")
            print(f"    This causes division by zero/sign flip in epsilon!")
            first_critical = np.where(v_plus_mu2_min_arr <= critical_threshold)[0][0]
            print(f"    First occurrence at t = {times[first_critical]:.2f} ms")
            print(f"    mu2 = {self.sim.mu2}")
            print(f"    Minimum v reached: {np.min(V_min_arr):.6f}")
            print(f"    v < -mu2 would give: {np.min(V_min_arr)} < {-self.sim.mu2:.3f}")
        else:
            print(f"✓ (v + mu2) stayed above zero")
            print(f"    Minimum value: {np.min(v_plus_mu2_min_arr):.6f}")

        # V_min trajectory
        print("\n" + "-" * 70)
        print("V_MIN TRAJECTORY (User-reported issue)")
        print("-" * 70)

        print(f"V_min evolution:")
        print(f"  Initial: {V_min_arr[0]:.6f}")
        print(f"  Minimum: {np.min(V_min_arr):.6f} at t={times[np.argmin(V_min_arr)]:.2f} ms")
        print(f"  Final: {V_min_arr[-1]:.6f}")

        # Check rate of V_min decrease
        if len(times) > 10:
            dVmin_dt = np.diff(V_min_arr) / np.diff(times)
            max_decrease_rate = np.min(dVmin_dt)
            print(f"  Max decrease rate: {max_decrease_rate:.3f} per ms")

            if max_decrease_rate < -10.0:
                print(f"  ⚠️  Rapid decrease detected! Runaway instability.")

        # Epsilon bounds
        print("\n" + "-" * 70)
        print("EPSILON FUNCTION ANALYSIS")
        print("-" * 70)

        epsilon_min_arr = np.array(self.epsilon_min)
        epsilon_max_arr = np.array(self.epsilon_max)

        print(f"Epsilon range:")
        print(f"  Min: {np.min(epsilon_min_arr):.6f}")
        print(f"  Max: {np.max(epsilon_max_arr):.6f}")

        if np.min(epsilon_min_arr) < 0:
            print(f"  ⚠️  Epsilon became NEGATIVE! This flips dw/dt sign!")

        if np.max(epsilon_max_arr) > 1.0:
            print(f"  ⚠️  Epsilon became very large! Recovery too fast!")

        # Create diagnostic plots
        self._plot_diagnostics(save_prefix)

        print("\n" + "=" * 70)
        print("DIAGNOSTIC PLOTS SAVED")
        print("=" * 70)
        print(f"  {save_prefix}_timeseries.png")
        print(f"  {save_prefix}_phase_plane.png")
        print(f"  {save_prefix}_critical_values.png")
        if self.crashed:
            print(f"  {save_prefix}_crash_state.png")

    def _plot_diagnostics(self, save_prefix):
        """Create diagnostic plots."""
        times = np.array(self.times)

        # Plot 1: Time series of all variables
        fig1, axes1 = plt.subplots(4, 2, figsize=(16, 16))

        # V bounds
        ax = axes1[0, 0]
        ax.plot(times, self.V_min, 'b-', label='V_min', linewidth=2)
        ax.plot(times, self.V_max, 'r-', label='V_max', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axhline(y=1, color='k', linestyle='--', alpha=0.3)
        ax.set_ylabel('V (dimensionless)', fontsize=11)
        ax.set_title('Voltage Bounds', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # w bounds
        ax = axes1[0, 1]
        ax.plot(times, self.w_min, 'b-', label='w_min', linewidth=2)
        ax.plot(times, self.w_max, 'r-', label='w_max', linewidth=2)
        ax.set_ylabel('w (dimensionless)', fontsize=11)
        ax.set_title('Recovery Variable Bounds', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # CRITICAL: v + mu2
        ax = axes1[1, 0]
        ax.plot(times, self.v_plus_mu2_min, 'r-', linewidth=2, label=f'min(v + μ₂)')
        ax.axhline(y=0, color='k', linestyle='--', linewidth=2, label='Critical: v + μ₂ = 0')
        ax.axhline(y=self.sim.mu2, color='g', linestyle='--', alpha=0.5, label=f'μ₂ = {self.sim.mu2}')
        ax.set_ylabel('v + μ₂', fontsize=11)
        ax.set_title('HYPOTHESIS 1: Division by Zero Check', fontweight='bold', color='red')
        ax.legend()
        ax.grid(alpha=0.3)

        # Epsilon bounds
        ax = axes1[1, 1]
        ax.plot(times, self.epsilon_min, 'b-', label='ε_min', linewidth=2)
        ax.plot(times, self.epsilon_max, 'r-', label='ε_max', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, label='Zero')
        ax.set_ylabel('ε (recovery rate)', fontsize=11)
        ax.set_title('Epsilon Function', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # Ionic current bounds
        ax = axes1[2, 0]
        ax.plot(times, self.I_ion_min, 'b-', label='I_ion_min', linewidth=2)
        ax.plot(times, self.I_ion_max, 'r-', label='I_ion_max', linewidth=2)
        ax.set_ylabel('I_ion', fontsize=11)
        ax.set_title('Ionic Current', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # Spatial variance (wave activity indicator)
        ax = axes1[2, 1]
        ax.plot(times, self.V_std, 'b-', label='σ(V)', linewidth=2)
        ax.plot(times, self.w_std, 'r-', label='σ(w)', linewidth=2)
        ax.set_ylabel('Standard deviation', fontsize=11)
        ax.set_title('Spatial Variance (Wave Activity)', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # V_min closeup (user-reported issue)
        ax = axes1[3, 0]
        ax.plot(times, self.V_min, 'r-', linewidth=2)
        ax.set_xlabel('Time (ms)', fontsize=11)
        ax.set_ylabel('V_min', fontsize=11)
        ax.set_title('V_min Trajectory (Crash Indicator)', fontweight='bold', color='red')
        ax.grid(alpha=0.3)

        # dV_min/dt (rate of change)
        if len(times) > 1:
            ax = axes1[3, 1]
            dVmin_dt = np.diff(self.V_min) / np.diff(times)
            ax.plot(times[1:], dVmin_dt, 'r-', linewidth=2)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax.set_xlabel('Time (ms)', fontsize=11)
            ax.set_ylabel('dV_min/dt', fontsize=11)
            ax.set_title('Rate of V_min Decrease', fontweight='bold')
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{save_prefix}_timeseries.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Plot 2: Phase plane
        fig2, ax2 = plt.subplots(figsize=(10, 8))

        # Sample points throughout simulation
        sample_indices = np.linspace(0, len(times)-1, min(1000, len(times)), dtype=int)

        for idx in sample_indices:
            V_sample = self.sim.V.flatten()[::10]  # Subsample spatially
            w_sample = self.sim.w.flatten()[::10]

            ax2.scatter(V_sample, w_sample, c=np.full(len(V_sample), times[idx]),
                       cmap='viridis', s=1, alpha=0.3)

        ax2.set_xlabel('V', fontsize=12)
        ax2.set_ylabel('w', fontsize=12)
        ax2.set_title('Phase Plane (V vs w)', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(ax2.collections[-1], ax=ax2, label='Time (ms)')
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{save_prefix}_phase_plane.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Plot 3: Critical values summary
        fig3, axes3 = plt.subplots(2, 1, figsize=(12, 10))

        ax = axes3[0]
        ax.plot(times, self.V_min, 'b-', linewidth=2, label='V_min')
        ax_twin = ax.twinx()
        ax_twin.plot(times, self.v_plus_mu2_min, 'r-', linewidth=2, label='min(v+μ₂)')
        ax_twin.axhline(y=0, color='k', linestyle='--', linewidth=2)

        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('V_min', fontsize=12, color='b')
        ax_twin.set_ylabel('v + μ₂', fontsize=12, color='r')
        ax.tick_params(axis='y', labelcolor='b')
        ax_twin.tick_params(axis='y', labelcolor='r')
        ax.set_title('Hypothesis 1: V_min vs Division by Zero Risk', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(loc='upper left')
        ax_twin.legend(loc='upper right')

        ax = axes3[1]
        ax.plot(times, self.epsilon_min, 'b-', linewidth=2, label='ε_min')
        ax.plot(times, self.epsilon_max, 'r-', linewidth=2, label='ε_max')
        ax.axhline(y=0, color='k', linestyle='--', linewidth=2, alpha=0.5)
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Epsilon', fontsize=12)
        ax.set_title('Recovery Rate (Epsilon) Bounds', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{save_prefix}_critical_values.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Plot 4: Crash state (if crashed)
        if self.crashed:
            fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6))

            V_crash = self.crash_state['V']
            w_crash = self.crash_state['w']

            # Convert to physical voltage
            V_crash_mV = self.sim.ionic_model.voltage_to_physical(V_crash)

            im1 = axes4[0].imshow(V_crash_mV, origin='lower', cmap='turbo', aspect='equal')
            axes4[0].set_title(f'V at crash (t={self.crash_time:.2f} ms)', fontweight='bold')
            axes4[0].set_xlabel('x')
            axes4[0].set_ylabel('y')
            plt.colorbar(im1, ax=axes4[0], label='V (mV)')

            im2 = axes4[1].imshow(w_crash, origin='lower', cmap='viridis', aspect='equal')
            axes4[1].set_title(f'w at crash (t={self.crash_time:.2f} ms)', fontweight='bold')
            axes4[1].set_xlabel('x')
            axes4[1].set_ylabel('y')
            plt.colorbar(im2, ax=axes4[1], label='w')

            plt.tight_layout()
            plt.savefig(f'{save_prefix}_crash_state.png', dpi=150, bbox_inches='tight')
            plt.close()


def test_stimulus_application():
    """
    Test user-reported issue: 100mV stimulus only gives 17mV change.
    """
    print("\n" + "=" * 70)
    print("STIMULUS APPLICATION TEST")
    print("=" * 70)

    from interactive_simulation import InteractiveSimulation

    sim = InteractiveSimulation(
        domain_size=80.0,
        resolution=0.5,
        initial_stim_amplitude=100.0,  # 100 mV
        initial_stim_radius=5.0
    )

    # Check initial state
    V_initial_phys = sim.ionic_model.voltage_to_physical(sim.V)
    print(f"\nInitial state:")
    print(f"  V (dimensionless): {np.mean(sim.V):.6f}")
    print(f"  V (physical): {np.mean(V_initial_phys):.2f} mV (should be ≈ {sim.ionic_model.V_rest:.1f} mV)")

    # Add stimulus at center
    sim.add_stimulus(40.0, 40.0)

    # Get stimulus array
    I_stim = sim.get_current_stimulus()

    print(f"\nStimulus applied:")
    print(f"  Amplitude setting: {sim.stim_amplitude} mV")
    print(f"  Radius: {sim.stim_radius} mm")
    print(f"  Normalized amplitude: {np.max(I_stim):.6f}")
    print(f"  Expected normalized: {sim.stim_amplitude / sim.ionic_model.V_range:.6f}")
    print(f"  Number of stimulated points: {np.sum(I_stim > 0)}")

    # Apply stimulus for 2ms (200 steps)
    print(f"\nApplying stimulus for 2ms...")
    for step in range(200):
        I_stim = sim.get_current_stimulus()
        sim.step(sim.dt, I_stim)

    # Check final state
    V_final_phys = sim.ionic_model.voltage_to_physical(sim.V)
    V_stim_region = V_final_phys[sim.Y**2 + (sim.X - 40)**2 <= sim.stim_radius**2]

    print(f"\nAfter 2ms stimulus:")
    print(f"  V_max (dimensionless): {np.max(sim.V):.6f}")
    print(f"  V_max (physical): {np.max(V_final_phys):.2f} mV")
    print(f"  V in stimulated region: {np.mean(V_stim_region):.2f} mV")
    print(f"  Expected if 100mV added: {sim.ionic_model.V_rest + 100:.2f} mV")
    print(f"  Actual change: {np.mean(V_stim_region) - sim.ionic_model.V_rest:.2f} mV")

    # Diagnosis
    actual_change = np.mean(V_stim_region) - sim.ionic_model.V_rest
    expected_change = sim.stim_amplitude

    print(f"\n" + "-" * 70)
    if abs(actual_change - expected_change) > 10.0:
        print(f"⚠️  ISSUE CONFIRMED: Expected {expected_change:.1f} mV, got {actual_change:.1f} mV")
        print(f"    Difference: {expected_change - actual_change:.1f} mV")
        print(f"\nPossible causes:")
        print(f"  1. Ionic current counteracting stimulus too fast")
        print(f"  2. Stimulus not accumulating over multiple time steps")
        print(f"  3. Normalization or conversion error")
    else:
        print(f"✓ Stimulus application seems reasonable")


def run_controlled_test(duration_ms=50.0):
    """
    Run controlled test with single stimulus.

    Focus on user observations:
    - Crash during AP propagation (not at stimulus)
    - V_min becomes increasingly negative
    """
    print("\n" + "=" * 70)
    print("CONTROLLED STABILITY TEST")
    print("=" * 70)

    from interactive_simulation import InteractiveSimulation

    # Create simulation
    sim = InteractiveSimulation(
        domain_size=80.0,
        resolution=0.5,
        initial_stim_amplitude=30.0,
        initial_stim_radius=3.0
    )

    # Create monitor
    monitor = StabilityMonitor(sim)

    # Add single stimulus
    print(f"\nAdding single stimulus at (40, 40) mm...")
    sim.add_stimulus(40.0, 40.0)

    # Run simulation with monitoring
    print(f"Running simulation for {duration_ms} ms...")
    print(f"Monitoring every step...\n")

    dt = sim.dt
    n_steps = int(duration_ms / dt)
    record_every = 10  # Record every 10 steps

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
                    print(f"\n⚠️  Instability detected at t = {t:.2f} ms")
                    break

                # Progress report
                if step % 1000 == 0:
                    V_min = np.min(sim.V)
                    V_max = np.max(sim.V)
                    print(f"  t = {t:6.1f} ms | V ∈ [{V_min:8.4f}, {V_max:8.4f}]", end='')

                    if V_min < -0.5:
                        print(f"  ⚠️ V_min becoming very negative!", end='')
                    print()

        # Final record
        monitor.record(sim.t)

    except Exception as e:
        print(f"\n❌ Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        monitor.crashed = True
        monitor.crash_time = sim.t
        monitor.crash_state = {'V': sim.V.copy(), 'w': sim.w.copy(), 'reason': f'Exception: {e}'}

    # Generate report
    monitor.generate_report(save_prefix='phase1_stability')

    return monitor


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 1: STABILITY DEBUG")
    print("=" * 70)
    print("\nFocus: User-reported issues")
    print("  1. V_min crashes during AP propagation")
    print("  2. Division by zero in epsilon (Hypothesis 1)")
    print("  3. Stimulus application (100mV → only 17mV?)")
    print("=" * 70)

    # Test 1: Stimulus application
    test_stimulus_application()

    # Test 2: Controlled stability test
    print("\n" + "=" * 70)
    print("Starting controlled stability test (50ms)...")
    monitor = run_controlled_test(duration_ms=50.0)

    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review diagnostic plots")
    print("  2. Analyze findings")
    print("  3. Implement targeted fix")
