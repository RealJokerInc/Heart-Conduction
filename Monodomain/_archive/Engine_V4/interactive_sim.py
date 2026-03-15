"""
Interactive Cardiac Simulation (LRd94)
======================================

Simple 2D simulation with:
- Rightward-aligned fibers (horizontal)
- Point stimulus with BCL control
- No-flux boundary conditions
- Interactive visualization

Uses LRd94 ionic model with physical voltage units (mV).

Expected behavior:
- Elliptical wave spreading faster horizontally (2:1 anisotropy)
- APD90 ~ 250 ms

Author: Generated with Claude Code
Date: 2025-12-11
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation

from simulation import CardiacSimulation2D, Stimulus
from parameters import (
    LRd94Params, SpatialParams,
    default_params, default_spatial_params
)


class InteractiveSimulation:
    """
    Interactive cardiac simulation with real-time visualization.
    Uses LRd94 ionic model with physical mV voltage.
    """

    def __init__(
        self,
        domain_size: float = 30.0,
        resolution: float = 0.5,
        D_parallel: float = 0.1,
        D_perp: float = 0.05,
    ):
        """
        Initialize interactive simulation.

        Parameters
        ----------
        domain_size : float
            Square domain side [mm]
        resolution : float
            Grid spacing [mm]
        D_parallel : float
            Diffusion along fibers [mm^2/ms]
        D_perp : float
            Diffusion perpendicular to fibers [mm^2/ms]
        """
        # Spatial parameters (fibers aligned horizontally)
        self.spatial = SpatialParams(
            Lx=domain_size,
            Ly=domain_size,
            dx=resolution,
            dy=resolution,
            D_parallel=D_parallel,
            D_perp=D_perp,
            fiber_angle=0.0,  # Horizontal fibers
        )

        # LRd94 parameters
        self.lrd_params = default_params()

        # Time step
        self.dt = 0.005  # ms - LRd94 requires small step

        # Create simulation
        self.sim = CardiacSimulation2D(
            lrd_params=self.lrd_params,
            spatial_params=self.spatial,
            dt=self.dt,
        )

        # Stimulus parameters
        self.stim_x = 4.0  # Left side
        self.stim_y = domain_size / 2  # Center vertically
        self.stim_radius = 2.0
        self.stim_amplitude = -100.0  # uA/cm^2
        self.stim_duration = 1.0

        # BCL control
        self.bcl = 400.0  # Default BCL [ms]
        self.last_stim_time = -1000.0

        # Animation state
        self.running = False
        self.animation = None
        self.frame_skip = 20  # Steps per frame (dt=0.005, so 20 steps = 0.1 ms)

        # Setup figure
        self._setup_figure()

    def _setup_figure(self):
        """Setup matplotlib figure with controls."""
        self.fig = plt.figure(figsize=(14, 8))

        # Main voltage display
        self.ax_main = self.fig.add_axes([0.05, 0.25, 0.55, 0.70])

        # Colorbar axes
        self.ax_cbar = self.fig.add_axes([0.62, 0.25, 0.02, 0.70])

        # Voltage trace axes
        self.ax_trace = self.fig.add_axes([0.72, 0.55, 0.25, 0.40])

        # Info text axes
        self.ax_info = self.fig.add_axes([0.72, 0.25, 0.25, 0.25])
        self.ax_info.axis('off')

        # Initialize voltage image
        V_mV = self.sim.get_voltage_mV()
        self.im = self.ax_main.imshow(
            V_mV,
            origin='lower',
            extent=[0, self.spatial.Lx, 0, self.spatial.Ly],
            cmap='turbo',
            vmin=-90,
            vmax=50,
            aspect='equal',
        )

        # Stimulus marker
        self.stim_marker, = self.ax_main.plot(
            self.stim_x, self.stim_y, 'w*', markersize=15,
            markeredgecolor='black', markeredgewidth=1
        )

        # Colorbar
        self.cbar = self.fig.colorbar(self.im, cax=self.ax_cbar)
        self.cbar.set_label('Voltage [mV]', fontsize=11)

        # Main plot labels
        self.ax_main.set_xlabel('x [mm]', fontsize=12)
        self.ax_main.set_ylabel('y [mm]', fontsize=12)
        self.ax_main.set_title('LRd94 Wave Propagation (Fibers -> Rightward)', fontsize=14)

        # Add fiber direction arrow
        arrow_y = self.spatial.Ly * 0.9
        self.ax_main.annotate(
            '', xy=(self.spatial.Lx * 0.3, arrow_y),
            xytext=(self.spatial.Lx * 0.1, arrow_y),
            arrowprops=dict(arrowstyle='->', color='white', lw=2)
        )
        self.ax_main.text(
            self.spatial.Lx * 0.2, arrow_y + 1.0, 'Fiber Direction',
            color='white', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5)
        )

        # Voltage trace setup
        self.trace_times = []
        self.trace_voltages = []
        self.trace_line, = self.ax_trace.plot([], [], 'b-', linewidth=1.5)
        self.ax_trace.set_xlim(0, 500)
        self.ax_trace.set_ylim(-100, 60)
        self.ax_trace.set_xlabel('Time [ms]', fontsize=10)
        self.ax_trace.set_ylabel('V [mV]', fontsize=10)
        self.ax_trace.set_title('Voltage at Stimulus Point', fontsize=11)
        self.ax_trace.axhline(y=-84, color='gray', linestyle='--', alpha=0.3, label='V_rest')
        self.ax_trace.axhline(y=40, color='gray', linestyle='--', alpha=0.3, label='V_peak')
        self.ax_trace.grid(True, alpha=0.3)

        # Info text
        self.info_text = self.ax_info.text(
            0.05, 0.95, '', transform=self.ax_info.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

        # === CONTROLS ===

        # BCL Slider
        self.ax_bcl = self.fig.add_axes([0.15, 0.10, 0.35, 0.03])
        self.slider_bcl = Slider(
            self.ax_bcl, 'BCL [ms]',
            valmin=300, valmax=1000, valinit=self.bcl,
            valstep=50, color='steelblue'
        )
        self.slider_bcl.on_changed(self._on_bcl_change)

        # Start/Stop button
        self.ax_btn_start = self.fig.add_axes([0.60, 0.10, 0.10, 0.04])
        self.btn_start = Button(self.ax_btn_start, 'Start', color='lightgreen')
        self.btn_start.on_clicked(self._on_start)

        # Reset button
        self.ax_btn_reset = self.fig.add_axes([0.72, 0.10, 0.10, 0.04])
        self.btn_reset = Button(self.ax_btn_reset, 'Reset', color='lightyellow')
        self.btn_reset.on_clicked(self._on_reset)

        # Single stimulus button
        self.ax_btn_stim = self.fig.add_axes([0.84, 0.10, 0.12, 0.04])
        self.btn_stim = Button(self.ax_btn_stim, 'Single Stim', color='lightcoral')
        self.btn_stim.on_clicked(self._on_single_stim)

        # Status text
        self.ax_status = self.fig.add_axes([0.15, 0.02, 0.70, 0.05])
        self.ax_status.axis('off')
        self.status_text = self.ax_status.text(
            0.5, 0.5, 'Ready - Click "Start" or "Single Stim"',
            transform=self.ax_status.transAxes,
            fontsize=11, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8)
        )

    def _on_bcl_change(self, val):
        """Handle BCL slider change."""
        self.bcl = val
        self._update_info()

    def _on_start(self, event):
        """Start/stop animation."""
        if self.running:
            self.running = False
            self.btn_start.label.set_text('Start')
            self.btn_start.color = 'lightgreen'
            self.status_text.set_text('Paused')
        else:
            self.running = True
            self.btn_start.label.set_text('Pause')
            self.btn_start.color = 'salmon'
            self.status_text.set_text(f'Running (BCL = {self.bcl:.0f} ms)')

            if self.animation is None:
                self.animation = FuncAnimation(
                    self.fig, self._animate_frame,
                    interval=30,  # ~33 fps display
                    blit=False,
                    cache_frame_data=False,
                )
        self.fig.canvas.draw_idle()

    def _on_reset(self, event):
        """Reset simulation."""
        self.running = False
        self.btn_start.label.set_text('Start')
        self.btn_start.color = 'lightgreen'

        self.sim.reset()
        self.sim.stimuli.clear()
        self.last_stim_time = -1000.0

        self.trace_times = []
        self.trace_voltages = []
        self.trace_line.set_data([], [])
        self.ax_trace.set_xlim(0, 500)

        self._update_display()
        self.status_text.set_text('Reset - Ready')
        self.fig.canvas.draw_idle()

    def _on_single_stim(self, event):
        """Apply single stimulus."""
        self._apply_stimulus()
        self.status_text.set_text(f'Stimulus applied at t = {self.sim.time:.1f} ms')
        self.fig.canvas.draw_idle()

    def _apply_stimulus(self):
        """Apply stimulus at current time."""
        stim = Stimulus(
            x_center=self.stim_x,
            y_center=self.stim_y,
            radius=self.stim_radius,
            amplitude=self.stim_amplitude,
            t_start=self.sim.time,
            duration=self.stim_duration,
        )
        self.sim.add_stimulus(stim)
        self.last_stim_time = self.sim.time

    def _animate_frame(self, frame):
        """Animation update function."""
        if not self.running:
            return

        # Check if we need periodic stimulus
        if self.sim.time - self.last_stim_time >= self.bcl:
            self._apply_stimulus()

        # Run simulation steps
        for _ in range(self.frame_skip):
            self.sim.step()

        # Record trace at stimulus point
        i_stim = int(self.stim_y / self.spatial.dy)
        j_stim = int(self.stim_x / self.spatial.dx)
        V_at_stim = self.sim.state['V'][i_stim, j_stim]

        self.trace_times.append(self.sim.time)
        self.trace_voltages.append(V_at_stim)

        # Limit trace length
        max_trace = 5000
        if len(self.trace_times) > max_trace:
            self.trace_times = self.trace_times[-max_trace:]
            self.trace_voltages = self.trace_voltages[-max_trace:]

        self._update_display()

        self.status_text.set_text(
            f'Running: t = {self.sim.time:.1f} ms | BCL = {self.bcl:.0f} ms | '
            f'Next stim in {self.bcl - (self.sim.time - self.last_stim_time):.0f} ms'
        )

    def _update_display(self):
        """Update voltage image and trace."""
        V_mV = self.sim.get_voltage_mV()
        self.im.set_data(V_mV)

        if self.trace_times:
            self.trace_line.set_data(self.trace_times, self.trace_voltages)

            t_max = max(self.trace_times)
            t_window = 500
            if t_max > t_window:
                self.ax_trace.set_xlim(t_max - t_window, t_max)
            else:
                self.ax_trace.set_xlim(0, max(t_window, t_max + 50))

        self._update_info()

    def _update_info(self):
        """Update info text."""
        V = self.sim.state['V']
        Ca_i = self.sim.state['Ca_i']

        activated = np.sum(V > -30.0) / V.size * 100

        info = (
            f"Time: {self.sim.time:.1f} ms\n"
            f"BCL: {self.bcl:.0f} ms\n"
            f"\n"
            f"Voltage [mV]:\n"
            f"  Min: {np.min(V):.1f}\n"
            f"  Max: {np.max(V):.1f}\n"
            f"\n"
            f"[Ca2+]i [nM]:\n"
            f"  Max: {np.max(Ca_i)*1e6:.0f}\n"
            f"\n"
            f"Activated: {activated:.1f}%\n"
            f"\n"
            f"Parameters:\n"
            f"  D|| = {self.spatial.D_parallel} mm^2/ms\n"
            f"  D_|_ = {self.spatial.D_perp} mm^2/ms\n"
            f"  APD90 ~ 250 ms"
        )
        self.info_text.set_text(info)

    def run(self):
        """Start the interactive simulation."""
        print("=" * 50)
        print("INTERACTIVE CARDIAC SIMULATION (LRd94)")
        print("=" * 50)
        print("\nControls:")
        print("  - BCL Slider: Adjust pacing rate")
        print("  - Start/Pause: Toggle continuous pacing")
        print("  - Reset: Clear and restart")
        print("  - Single Stim: Apply one stimulus")
        print("\nModel: Luo-Rudy 1994")
        print("  - dt = 0.005 ms")
        print("  - APD90 ~ 250 ms")
        print("  - Physical voltage (mV)")
        print("=" * 50)

        self._update_info()
        plt.show()


def main():
    """Run interactive simulation."""
    sim = InteractiveSimulation(
        domain_size=30.0,
        resolution=0.5,
        D_parallel=0.1,
        D_perp=0.05,
    )
    sim.run()


if __name__ == "__main__":
    main()
