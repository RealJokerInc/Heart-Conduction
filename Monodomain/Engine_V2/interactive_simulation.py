"""
Interactive Cardiac Simulation - Click to Stimulate
====================================================

Real-time simulation with interactive point stimulation.

Controls:
- LEFT CLICK: Add stimulus at cursor position
- SPACE: Pause/Resume simulation
- 'R': Reset simulation
- 'Q' or ESC: Quit
- '+/-': Increase/decrease stimulus amplitude
- '[/]': Increase/decrease stimulus radius

Author: Generated with Claude Code
Date: 2025-12-09
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import sys
from collections import deque

from parameters import create_default_parameters
from aliev_panfilov_fixed import AlievPanfilovModel
from simulate_infarct_v2 import compute_diffusion_flux_based, ionic_step_numba


class InteractiveSimulation:
    """
    Real-time interactive cardiac simulation.

    Click anywhere to add local stimulation.
    """

    def __init__(
        self,
        domain_size=80.0,
        resolution=0.5,
        D_parallel=1.0,
        D_perp=0.5,
        T_scale=10.0,
        fiber_angle=0.0,
        initial_stim_amplitude=30.0,
        initial_stim_radius=3.0
    ):
        """Initialize interactive simulation."""

        print("=" * 70)
        print("INTERACTIVE CARDIAC SIMULATION")
        print("=" * 70)

        # Domain setup
        self.Lx = domain_size
        self.Ly = domain_size
        self.dx = resolution
        self.dy = resolution
        self.nx = int(np.round(self.Lx / self.dx)) + 1
        self.ny = int(np.round(self.Ly / self.dy)) + 1

        print(f"\nDomain: {self.Lx} × {self.Ly} mm")
        print(f"Grid: {self.nx} × {self.ny} points")
        print(f"Resolution: {self.dx} mm")

        # Create coordinate arrays for stimulus application
        x = np.linspace(0, self.Lx, self.nx)
        y = np.linspace(0, self.Ly, self.ny)
        self.X, self.Y = np.meshgrid(x, y)

        # Uniform fiber field
        theta = np.radians(fiber_angle)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Diffusion tensor
        self.Dxx = np.full((self.ny, self.nx), D_parallel * cos_t**2 + D_perp * sin_t**2)
        self.Dyy = np.full((self.ny, self.nx), D_parallel * sin_t**2 + D_perp * cos_t**2)
        self.Dxy = np.full((self.ny, self.nx), (D_parallel - D_perp) * cos_t * sin_t)

        # Ionic model
        params = create_default_parameters()
        params.ionic.epsilon_rest = 0.05
        self.ionic_model = AlievPanfilovModel(params.ionic, params.physical, T_scale=T_scale)

        # Store ionic parameters for Numba
        self.k = params.ionic.k
        self.a = params.ionic.a
        self.epsilon0 = params.ionic.epsilon0
        self.mu1 = params.ionic.mu1
        self.mu2 = params.ionic.mu2
        self.epsilon_rest = params.ionic.epsilon_rest
        self.V_threshold = params.ionic.V_threshold
        self.k_sigmoid = params.ionic.k_sigmoid

        # State variables
        self.V = np.zeros((self.ny, self.nx))
        self.w = np.zeros((self.ny, self.nx))

        # Tissue mask (all True for flat domain - no infarct)
        self.tissue_mask = np.ones((self.ny, self.nx), dtype=bool)

        # Simulation parameters
        self.dt = 0.005  # ms (FIX: reduced from 0.01 for stability)
        self.t = 0.0
        self.frame_count = 0

        # Interactive stimulus parameters
        self.stim_amplitude = initial_stim_amplitude  # mV
        self.stim_radius = initial_stim_radius  # mm
        self.stim_duration = 2.0  # ms

        # Stimulus queue: list of (x, y, t_start)
        self.active_stimuli = deque()

        # Control flags
        self.running = True
        self.paused = False

        # Performance tracking
        self.fps_samples = deque(maxlen=30)
        self.last_frame_time = time.time()

        print(f"\n✓ Interactive simulation ready!")
        print(f"\nInitial stimulus parameters:")
        print(f"  Amplitude: {self.stim_amplitude} mV")
        print(f"  Radius: {self.stim_radius} mm")
        print(f"  Duration: {self.stim_duration} ms")

    def step(self, dt, I_stim):
        """Single time step."""
        # Diffusion
        div_J = compute_diffusion_flux_based(self.V, self.Dxx, self.Dyy, self.Dxy, self.dx, self.dy)
        self.V += dt * div_J

        # Reaction (Numba-accelerated)
        dtau = self.ionic_model.time_to_tau(dt)

        ionic_step_numba(
            self.V, self.w, dtau, I_stim, self.tissue_mask,
            self.k, self.a, self.epsilon0, self.mu1, self.mu2,
            self.epsilon_rest, self.V_threshold, self.k_sigmoid
        )

        # FIX #3: Runtime stability warnings
        V_min = np.min(self.V)
        V_max = np.max(self.V)
        v_plus_mu2_min = np.min(self.V + self.mu2)

        if V_min < -1.0:
            V_min_phys = self.ionic_model.voltage_to_physical(np.array([[V_min]]))[0, 0]
            print(f"\n⚠️  WARNING at t={self.t:.2f}ms: V becoming very negative!")
            print(f"    V_min = {V_min:.4f} (dimensionless) = {V_min_phys:.1f} mV")
            print(f"    min(v + μ₂) = {v_plus_mu2_min:.4f}")
            if v_plus_mu2_min < 0.1:
                print(f"    ⚠️  CRITICAL: Approaching division by zero!")

        if np.isnan(V_min) or np.isnan(V_max) or np.isinf(V_min) or np.isinf(V_max):
            print(f"\n❌ ERROR at t={self.t:.2f}ms: NaN or Inf detected!")
            print(f"    V_min = {V_min}, V_max = {V_max}")
            raise ValueError("Simulation became unstable (NaN/Inf detected)")

        self.t += dt
        self.frame_count += 1

    def add_stimulus(self, x_mm, y_mm):
        """
        Add stimulus at specified location.

        Parameters
        ----------
        x_mm, y_mm : float
            Coordinates in mm
        """
        # Add to active stimuli queue
        self.active_stimuli.append((x_mm, y_mm, self.t))

        print(f"  Stimulus added at ({x_mm:.1f}, {y_mm:.1f}) mm, t={self.t:.1f} ms")

    def get_current_stimulus(self):
        """
        Compute current stimulus based on active stimuli.

        Returns
        -------
        I_stim : ndarray
            Stimulus current array (ny, nx)
        """
        I_stim = np.zeros((self.ny, self.nx))

        # Remove expired stimuli
        while self.active_stimuli and (self.t - self.active_stimuli[0][2]) > self.stim_duration:
            self.active_stimuli.popleft()

        # Add contribution from each active stimulus
        amplitude_norm = self.stim_amplitude / self.ionic_model.V_range

        for x_stim, y_stim, t_start in self.active_stimuli:
            if self.t >= t_start and self.t < t_start + self.stim_duration:
                # Compute distance from stimulus center
                dist = np.sqrt((self.X - x_stim)**2 + (self.Y - y_stim)**2)

                # Apply stimulus to points within radius
                mask = dist <= self.stim_radius
                I_stim[mask] = amplitude_norm

        return I_stim

    def reset(self):
        """Reset simulation to initial state."""
        self.V = np.zeros((self.ny, self.nx))
        self.w = np.zeros((self.ny, self.nx))
        self.t = 0.0
        self.frame_count = 0
        self.active_stimuli.clear()
        print("\n  Simulation reset!")

    def update_fps(self):
        """Update FPS counter."""
        current_time = time.time()
        dt = current_time - self.last_frame_time
        if dt > 0:
            fps = 1.0 / dt
            self.fps_samples.append(fps)
        self.last_frame_time = current_time

    def get_avg_fps(self):
        """Get average FPS."""
        if len(self.fps_samples) > 0:
            return np.mean(self.fps_samples)
        return 0.0

    def run_interactive(self, steps_per_frame=10):
        """
        Run interactive simulation with matplotlib.

        Parameters
        ----------
        steps_per_frame : int
            Number of simulation steps per display update
        """
        print("\n" + "=" * 70)
        print("STARTING INTERACTIVE SIMULATION")
        print("=" * 70)
        print("\nControls:")
        print("  LEFT CLICK   : Add stimulus at cursor")
        print("  SPACE        : Pause/Resume")
        print("  R            : Reset simulation")
        print("  Q or ESC     : Quit")
        print("  +/-          : Increase/decrease amplitude")
        print("  [/]          : Increase/decrease radius")
        print("\n" + "=" * 70)

        # Setup figure
        fig, (ax_main, ax_info) = plt.subplots(1, 2, figsize=(16, 7),
                                                 gridspec_kw={'width_ratios': [3, 1]})

        # Main plot
        V_mV = self.ionic_model.voltage_to_physical(self.V)

        im = ax_main.imshow(
            V_mV,
            origin='lower',
            extent=[0, self.Lx, 0, self.Ly],
            cmap='turbo',
            vmin=self.ionic_model.V_rest,
            vmax=self.ionic_model.V_peak,
            aspect='equal',
            interpolation='bilinear'
        )

        plt.colorbar(im, ax=ax_main, label='Voltage (mV)', fraction=0.046)

        ax_main.set_xlabel('x (mm)', fontsize=12)
        ax_main.set_ylabel('y (mm)', fontsize=12)
        title = ax_main.set_title(f't = {self.t:.1f} ms | RUNNING',
                                   fontsize=14, fontweight='bold')

        # Info panel
        ax_info.axis('off')
        info_text = ax_info.text(0.05, 0.95, '', transform=ax_info.transAxes,
                                  verticalalignment='top', fontsize=10,
                                  family='monospace')

        # Stimulus indicator (circle showing next stimulus location/size)
        self.stim_indicator = None

        # Event handlers
        def on_click(event):
            if event.inaxes == ax_main and event.button == 1:  # Left click
                x_mm = event.xdata
                y_mm = event.ydata
                if x_mm is not None and y_mm is not None:
                    self.add_stimulus(x_mm, y_mm)

        def on_key(event):
            if event.key == ' ':  # Space - pause/resume
                self.paused = not self.paused
                status = "PAUSED" if self.paused else "RUNNING"
                print(f"\n  Simulation {status}")

            elif event.key == 'r':  # Reset
                self.reset()

            elif event.key in ['q', 'escape']:  # Quit
                print("\n  Quitting...")
                self.running = False
                plt.close(fig)

            elif event.key == '+' or event.key == '=':  # Increase amplitude
                self.stim_amplitude = min(self.stim_amplitude + 5.0, 100.0)
                print(f"\n  Amplitude: {self.stim_amplitude:.1f} mV")

            elif event.key == '-':  # Decrease amplitude
                self.stim_amplitude = max(self.stim_amplitude - 5.0, 5.0)
                print(f"\n  Amplitude: {self.stim_amplitude:.1f} mV")

            elif event.key == ']':  # Increase radius
                self.stim_radius = min(self.stim_radius + 1.0, 20.0)
                print(f"\n  Radius: {self.stim_radius:.1f} mm")

            elif event.key == '[':  # Decrease radius
                self.stim_radius = max(self.stim_radius - 1.0, 1.0)
                print(f"\n  Radius: {self.stim_radius:.1f} mm")

        def on_motion(event):
            # Show stimulus indicator at cursor
            if event.inaxes == ax_main:
                if self.stim_indicator:
                    self.stim_indicator.remove()

                x_mm = event.xdata
                y_mm = event.ydata
                if x_mm is not None and y_mm is not None:
                    self.stim_indicator = patches.Circle(
                        (x_mm, y_mm), self.stim_radius,
                        fill=False, edgecolor='yellow', linewidth=2,
                        linestyle='--', alpha=0.7
                    )
                    ax_main.add_patch(self.stim_indicator)

        # Connect events
        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('key_press_event', on_key)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)

        # Animation loop
        plt.ion()
        plt.show()

        try:
            while self.running:
                if not self.paused:
                    # Simulate multiple steps per frame for efficiency
                    for _ in range(steps_per_frame):
                        I_stim = self.get_current_stimulus()
                        self.step(self.dt, I_stim)

                # Update display
                V_mV = self.ionic_model.voltage_to_physical(self.V)
                im.set_data(V_mV)

                # Update title
                status = "PAUSED" if self.paused else "RUNNING"
                V_max = np.max(self.V)
                title.set_text(f't = {self.t:.1f} ms | {status} | V_max = {V_max:.3f}')

                # Update info panel
                self.update_fps()
                avg_fps = self.get_avg_fps()
                sim_speed = avg_fps * steps_per_frame * self.dt  # ms/sec

                info_str = f"""
STATUS: {status}

TIME
  Simulation: {self.t:.1f} ms
  Frame: {self.frame_count}

PERFORMANCE
  FPS: {avg_fps:.1f}
  Speed: {sim_speed:.1f} ms/sec

STIMULUS
  Amplitude: {self.stim_amplitude:.1f} mV
  Radius: {self.stim_radius:.1f} mm
  Duration: {self.stim_duration:.1f} ms
  Active: {len(self.active_stimuli)}

VOLTAGE
  Max: {np.max(V_mV):.1f} mV
  Min: {np.min(V_mV):.1f} mV

CONTROLS
  Click    : Stimulate
  Space    : Pause/Resume
  R        : Reset
  +/-      : Amplitude
  [/]      : Radius
  Q or ESC : Quit
                """.strip()

                info_text.set_text(info_str)

                # Redraw
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

                # Small delay to prevent busy-waiting when paused
                if self.paused:
                    plt.pause(0.05)
                else:
                    plt.pause(0.001)

        except KeyboardInterrupt:
            print("\n\n  Interrupted by user")

        finally:
            plt.ioff()
            print("\n" + "=" * 70)
            print("SIMULATION ENDED")
            print("=" * 70)
            print(f"\nFinal statistics:")
            print(f"  Total time: {self.t:.1f} ms")
            print(f"  Total frames: {self.frame_count}")
            print(f"  Average FPS: {self.get_avg_fps():.1f}")
            print(f"  Stimuli applied: {self.frame_count}")


# =============================================================================
# Main Script
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Interactive cardiac simulation')
    parser.add_argument('--domain-size', type=float, default=80.0,
                        help='Domain size in mm (default: 80.0)')
    parser.add_argument('--resolution', type=float, default=0.5,
                        help='Grid resolution in mm (default: 0.5)')
    parser.add_argument('--amplitude', type=float, default=30.0,
                        help='Initial stimulus amplitude in mV (default: 30.0)')
    parser.add_argument('--radius', type=float, default=3.0,
                        help='Initial stimulus radius in mm (default: 3.0)')
    parser.add_argument('--steps-per-frame', type=int, default=10,
                        help='Simulation steps per display frame (default: 10)')

    args = parser.parse_args()

    # Create interactive simulation
    sim = InteractiveSimulation(
        domain_size=args.domain_size,
        resolution=args.resolution,
        initial_stim_amplitude=args.amplitude,
        initial_stim_radius=args.radius
    )

    # Run
    sim.run_interactive(steps_per_frame=args.steps_per_frame)
