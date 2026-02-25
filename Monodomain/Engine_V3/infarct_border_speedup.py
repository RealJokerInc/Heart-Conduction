"""
Infarct Border Speedup Demonstration
=====================================

Tests whether waves speed up along no-flux infarct borders.

Setup:
- Long vertical rectangular infarct in center
- Planar wave from left edge
- Wave propagates along both sides of infarct
- Compare wave speed: center (free tissue) vs. near infarct border

Author: Generated with Claude Code
Date: 2025-12-10
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

from spiral_wave_sim import SpiralWaveSimulation


class InfarctBorderDemo:
    """
    Interactive demo showing wave propagation along infarct border.
    """

    def __init__(self):
        # Create simulation
        self.sim = SpiralWaveSimulation(
            domain_size=60.0,
            resolution=0.5,
            D_parallel=0.1,
            D_perp=0.05,
        )

        # Create long horizontal rectangle infarct in center
        # Long side aligned with fiber direction (horizontal)
        infarct_width = 40.0   # mm (long, aligned with fibers)
        infarct_height = 8.0   # mm (short, perpendicular to fibers)
        infarct_x_center = self.sim.domain_size / 2  # 30 mm
        infarct_y_center = self.sim.domain_size / 2  # 30 mm

        self.create_rectangular_infarct(
            x_center=infarct_x_center,
            y_center=infarct_y_center,
            width=infarct_width,
            height=infarct_height,
        )

        # Measurement points - measure HORIZONTAL propagation along borders
        # All at y = border position, different x positions
        # Compare propagation speed: free tissue (far from infarct) vs. near border
        x_start = infarct_x_center - 5.0  # 5mm left of infarct center
        x_end = infarct_x_center + 10.0   # 10mm right of infarct center
        y_border = infarct_y_center + infarct_height/2 + 1.0  # 1mm above top border
        y_free = infarct_y_center + infarct_height/2 + 5.0    # 5mm above top border (free tissue)

        self.measure_points = {
            'free_start': (x_start, y_free),     # Free tissue, left
            'free_end': (x_end, y_free),         # Free tissue, right
            'border_start': (x_start, y_border), # Near border, left
            'border_end': (x_end, y_border),     # Near border, right
        }

        # Convert to indices
        self.measure_indices = {}
        for name, (x, y) in self.measure_points.items():
            i = int(y / self.sim.resolution)
            j = int(x / self.sim.resolution)
            self.measure_indices[name] = (i, j)

        # Activation times
        self.activation_times = {name: None for name in self.measure_points.keys()}
        self.activation_threshold = 0.5  # u threshold for activation

        # Animation state
        self.running = False
        self.animation = None
        self.frame_skip = 10

        # Stimulus state
        self.stimulus_applied = False
        self.stim_start = 10.0
        self.stim_duration = 3.0

        # Setup figure
        self._setup_figure()

    def create_rectangular_infarct(self, x_center, y_center, width, height):
        """Create rectangular infarct."""
        x_min = x_center - width / 2
        x_max = x_center + width / 2
        y_min = y_center - height / 2
        y_max = y_center + height / 2

        infarct_region = (
            (self.sim.X >= x_min) & (self.sim.X <= x_max) &
            (self.sim.Y >= y_min) & (self.sim.Y <= y_max)
        )
        self.sim.mask[infarct_region] = 0.0
        self.sim.state['u'][infarct_region] = 0.0
        self.sim.state['v'][infarct_region] = 1.0
        self.sim.state['w'][infarct_region] = 1.0

        self.infarct_bounds = (x_min, x_max, y_min, y_max)

        n_infarct = np.sum(infarct_region)
        print(f"Created rectangular infarct:")
        print(f"  Size: {width}×{height} mm at ({x_center}, {y_center})")
        print(f"  Infarct cells: {n_infarct} ({n_infarct/self.sim.mask.size*100:.1f}%)")

    def _setup_figure(self):
        """Setup matplotlib figure."""
        self.fig = plt.figure(figsize=(16, 8))

        # Main voltage display
        self.ax_main = self.fig.add_axes([0.05, 0.20, 0.50, 0.75])

        # Colorbar
        self.ax_cbar = self.fig.add_axes([0.56, 0.20, 0.02, 0.75])

        # Info panel
        self.ax_info = self.fig.add_axes([0.64, 0.50, 0.33, 0.45])
        self.ax_info.axis('off')

        # Activation times panel
        self.ax_times = self.fig.add_axes([0.64, 0.20, 0.33, 0.25])
        self.ax_times.axis('off')

        # Initialize voltage display
        V_mV = self.sim.voltage_to_mV()
        self.im = self.ax_main.imshow(
            V_mV,
            origin='lower',
            extent=[0, self.sim.domain_size, 0, self.sim.domain_size],
            cmap='turbo',
            vmin=-85,
            vmax=40,
            aspect='equal',
        )

        # Overlay infarct
        infarct_overlay = np.ma.masked_where(
            self.sim.mask > 0.5,
            np.ones_like(self.sim.mask)
        )
        self.infarct_im = self.ax_main.imshow(
            infarct_overlay,
            origin='lower',
            extent=[0, self.sim.domain_size, 0, self.sim.domain_size],
            cmap='gray',
            vmin=0, vmax=1,
            alpha=0.8,
            aspect='equal',
        )

        # Draw infarct outline
        x_min, x_max, y_min, y_max = self.infarct_bounds
        rect = Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            fill=False,
            edgecolor='white',
            linewidth=2,
            linestyle='--',
        )
        self.ax_main.add_patch(rect)

        # Mark measurement points
        colors = {'free_start': 'cyan', 'free_end': 'cyan',
                  'border_start': 'yellow', 'border_end': 'yellow'}
        for name, (x, y) in self.measure_points.items():
            self.ax_main.plot(x, y, 'o', color=colors[name],
                            markersize=8, markeredgecolor='black', markeredgewidth=1)
            self.ax_main.text(x, y + 2, name.replace('_', '\n'),
                            color='white', fontsize=7, ha='center', va='bottom',
                            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

        # Add wave direction arrow
        arrow_y = self.sim.domain_size * 0.9
        self.ax_main.annotate(
            '', xy=(12, arrow_y), xytext=(3, arrow_y),
            arrowprops=dict(arrowstyle='->', color='yellow', lw=3)
        )
        self.ax_main.text(
            7.5, arrow_y + 2, 'Wave',
            color='yellow', fontsize=11, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5)
        )

        # Colorbar
        self.cbar = self.fig.colorbar(self.im, cax=self.ax_cbar)
        self.cbar.set_label('Voltage [mV]', fontsize=11)

        # Labels
        self.ax_main.set_xlabel('x [mm]', fontsize=12)
        self.ax_main.set_ylabel('y [mm]', fontsize=12)
        self.ax_main.set_title('Infarct Border Speedup Test', fontsize=14)

        # Info text
        self.info_text = self.ax_info.text(
            0.05, 0.95, '', transform=self.ax_info.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

        # Activation times text
        self.times_text = self.ax_times.text(
            0.05, 0.95, '', transform=self.ax_times.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        )

        # === CONTROLS ===

        # Start button
        self.ax_btn_start = self.fig.add_axes([0.25, 0.05, 0.10, 0.05])
        self.btn_start = Button(self.ax_btn_start, 'Start', color='lightgreen')
        self.btn_start.on_clicked(self._on_start)

        # Pause button
        self.ax_btn_pause = self.fig.add_axes([0.37, 0.05, 0.10, 0.05])
        self.btn_pause = Button(self.ax_btn_pause, 'Pause', color='lightyellow')
        self.btn_pause.on_clicked(self._on_pause)

        # Reset button
        self.ax_btn_reset = self.fig.add_axes([0.49, 0.05, 0.10, 0.05])
        self.btn_reset = Button(self.ax_btn_reset, 'Reset', color='lightcoral')
        self.btn_reset.on_clicked(self._on_reset)

        # Status text
        self.ax_status = self.fig.add_axes([0.05, 0.12, 0.55, 0.04])
        self.ax_status.axis('off')
        self.status_text = self.ax_status.text(
            0.5, 0.5,
            'Ready - Click "Start" to launch planar wave',
            transform=self.ax_status.transAxes,
            fontsize=11, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8)
        )

    def _on_start(self, event):
        """Start simulation."""
        if self.running:
            return

        self.running = True
        self.btn_start.label.set_text('Running...')
        self.btn_start.color = 'gray'

        # Start animation
        if self.animation is None:
            self.animation = FuncAnimation(
                self.fig, self._animate_frame,
                interval=20,
                blit=False,
                cache_frame_data=False,
            )

        self.status_text.set_text('Simulation running - watch for border speedup...')
        self.fig.canvas.draw_idle()

    def _on_pause(self, event):
        """Toggle pause."""
        self.running = not self.running
        if self.running:
            self.btn_pause.label.set_text('Pause')
            self.status_text.set_text(f't = {self.sim.time:.0f} ms - Running')
        else:
            self.btn_pause.label.set_text('Resume')
            self.status_text.set_text(f't = {self.sim.time:.0f} ms - Paused')
        self.fig.canvas.draw_idle()

    def _on_reset(self, event):
        """Reset simulation."""
        self.running = False
        self.stimulus_applied = False
        self.activation_times = {name: None for name in self.measure_points.keys()}

        # Reset simulation
        self.sim.reset()
        x_center = self.sim.domain_size / 2
        y_center = self.sim.domain_size / 2
        self.create_rectangular_infarct(x_center, y_center, 40.0, 8.0)

        # Update display
        self._update_display()
        self.btn_start.label.set_text('Start')
        self.btn_start.color = 'lightgreen'
        self.btn_pause.label.set_text('Pause')
        self.status_text.set_text('Reset - Ready')
        self.fig.canvas.draw_idle()

    def _animate_frame(self, frame):
        """Animation update."""
        if not self.running:
            return

        # Apply stimulus at t=10 ms
        if self.sim.time >= self.stim_start and not self.stimulus_applied:
            self.sim.apply_planar_stimulus('left', width=3.0, amplitude=1.0)
            self.stimulus_applied = True
            self.status_text.set_text('Planar wave launched - measuring activation times')

        # Turn off stimulus after duration
        if self.stimulus_applied and self.sim.time >= self.stim_start + self.stim_duration:
            self.sim.clear_stimulus()

        # Run simulation steps
        for _ in range(self.frame_skip):
            self.sim.step()

        # Check for activations
        for name, (i, j) in self.measure_indices.items():
            if self.activation_times[name] is None:
                if self.sim.state['u'][i, j] > self.activation_threshold:
                    self.activation_times[name] = self.sim.time
                    print(f"Activation at {name}: t = {self.sim.time:.1f} ms")

        # Update display
        self._update_display()

        # Calculate conduction velocities if we have activation times
        all_activated = all(t is not None for t in self.activation_times.values())
        if all_activated:
            self._display_results()

    def _display_results(self):
        """Display conduction velocity analysis."""
        t_free_start = self.activation_times['free_start']
        t_free_end = self.activation_times['free_end']
        t_border_start = self.activation_times['border_start']
        t_border_end = self.activation_times['border_end']

        # Calculate time differences (horizontal propagation)
        dt_free = t_free_end - t_free_start      # Time to travel in free tissue
        dt_border = t_border_end - t_border_start  # Time to travel near border

        # Distance (horizontal, along fibers)
        x_free_start = self.measure_points['free_start'][0]
        x_free_end = self.measure_points['free_end'][0]
        x_border_start = self.measure_points['border_start'][0]
        x_border_end = self.measure_points['border_end'][0]

        dist_free = x_free_end - x_free_start      # Should be 15mm
        dist_border = x_border_end - x_border_start  # Should be 15mm

        # Velocities (along fibers - fast direction)
        cv_free = dist_free / dt_free if dt_free > 0 else 0  # mm/ms
        cv_border = dist_border / dt_border if dt_border > 0 else 0  # mm/ms

        speedup = ((cv_border - cv_free) / cv_free * 100) if cv_free > 0 else 0

        result_text = (
            f"\n{'='*40}\n"
            f"BORDER SPEEDUP ANALYSIS\n"
            f"{'='*40}\n"
            f"Free tissue CV:   {cv_free:.3f} mm/ms ({cv_free*1000:.1f} mm/s)\n"
            f"Border CV:        {cv_border:.3f} mm/ms ({cv_border*1000:.1f} mm/s)\n"
            f"Speedup:          {speedup:+.1f}%\n"
            f"{'='*40}\n"
        )
        print(result_text)

        if speedup > 5:
            self.status_text.set_text(f'Border speedup detected: {speedup:+.1f}% faster!')
        elif speedup < -5:
            self.status_text.set_text(f'Border slowdown: {speedup:.1f}% slower')
        else:
            self.status_text.set_text(f'No significant speed change: {speedup:+.1f}%')

    def _update_display(self):
        """Update voltage image and info."""
        V_mV = self.sim.voltage_to_mV()
        self.im.set_data(V_mV)

        # Update info
        u = self.sim.state['u']
        activated = np.sum((u > 0.5) & (self.sim.mask > 0.5))
        total_cells = np.sum(self.sim.mask > 0.5)

        info = (
            f"Time: {self.sim.time:.1f} ms\n"
            f"\n"
            f"Voltage (u):\n"
            f"  Min: {np.min(u):.3f}\n"
            f"  Max: {np.max(u):.3f}\n"
            f"\n"
            f"Activated: {activated}/{total_cells}\n"
            f"           ({activated/total_cells*100:.1f}%)\n"
            f"\n"
            f"Measurement Points:\n"
        )

        for name in ['free_start', 'free_end', 'border_start', 'border_end']:
            i, j = self.measure_indices[name]
            u_val = self.sim.state['u'][i, j]
            t_act = self.activation_times[name]
            status = f"t={t_act:.1f}ms" if t_act else "waiting..."
            info += f"  {name}: {status}\n"

        self.info_text.set_text(info)

        # Activation times summary
        times_info = (
            "ACTIVATION TIMES\n"
            "━━━━━━━━━━━━━━━━━━━━━━━\n"
        )
        for name, t in self.activation_times.items():
            if t is not None:
                times_info += f"{name}:\n  {t:.1f} ms\n"
            else:
                times_info += f"{name}:\n  (not yet)\n"

        self.times_text.set_text(times_info)

    def run(self):
        """Start interactive demo."""
        print("=" * 60)
        print("INFARCT BORDER SPEEDUP TEST")
        print("=" * 60)
        print("\nSetup:")
        print("  - Long horizontal rectangular infarct (40×8 mm)")
        print("  - Aligned with fiber direction (horizontal)")
        print("  - Planar wave from left")
        print("  - 4 measurement points (horizontal propagation):")
        print("    • free_start: x=25mm, y=39mm (5mm above infarct)")
        print("    • free_end: x=40mm, y=39mm (5mm above infarct)")
        print("    • border_start: x=25mm, y=35mm (1mm above infarct border)")
        print("    • border_end: x=40mm, y=35mm (1mm above infarct border)")
        print("\nWatch for:")
        print("  - Does wave speed up near infarct border?")
        print("  - Compare CV: free tissue (far from border) vs. near border")
        print("  - Measuring horizontal propagation (along fibers)")
        print("=" * 60)

        self._update_display()
        plt.show()


def main():
    """Run infarct border speedup demo."""
    demo = InfarctBorderDemo()
    demo.run()


if __name__ == "__main__":
    main()
