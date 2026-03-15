"""
Infarct Border Speedup Demonstration (LRd94)
============================================

Tests whether waves speed up along no-flux infarct borders.

Setup:
- Long horizontal rectangular infarct in center
- Planar wave from left edge
- Wave propagates along both sides of infarct
- Compare wave speed: free tissue vs. near infarct border

Uses LRd94 ionic model with physical mV voltage.

Author: Generated with Claude Code
Date: 2025-12-11
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
    Uses LRd94 ionic model.
    """

    def __init__(self):
        # Create simulation
        self.sim = SpiralWaveSimulation(
            domain_size=40.0,  # Smaller domain for faster simulation
            resolution=0.5,
            D_parallel=0.1,
            D_perp=0.05,
        )

        # Create long horizontal rectangle infarct in center
        infarct_width = 25.0   # mm (long, aligned with fibers)
        infarct_height = 5.0   # mm (short, perpendicular to fibers)
        infarct_x_center = self.sim.domain_size / 2  # 20 mm
        infarct_y_center = self.sim.domain_size / 2  # 20 mm

        self.create_rectangular_infarct(
            x_center=infarct_x_center,
            y_center=infarct_y_center,
            width=infarct_width,
            height=infarct_height,
        )

        # Measurement points - measure HORIZONTAL propagation along borders
        x_start = infarct_x_center - 5.0  # 5mm left of infarct center
        x_end = infarct_x_center + 8.0    # 8mm right of infarct center
        y_border = infarct_y_center + infarct_height/2 + 1.0  # 1mm above top border
        y_free = infarct_y_center + infarct_height/2 + 4.0    # 4mm above top border

        self.measure_points = {
            'free_start': (x_start, y_free),
            'free_end': (x_end, y_free),
            'border_start': (x_start, y_border),
            'border_end': (x_end, y_border),
        }

        # Convert to indices
        self.measure_indices = {}
        for name, (x, y) in self.measure_points.items():
            i = int(y / self.sim.resolution)
            j = int(x / self.sim.resolution)
            self.measure_indices[name] = (i, j)

        # Activation times
        self.activation_times = {name: None for name in self.measure_points.keys()}
        self.activation_threshold = -30.0  # mV threshold for activation

        # Animation state
        self.running = False
        self.animation = None
        self.frame_skip = 20  # Steps per frame

        # Stimulus state
        self.stimulus_applied = False
        self.stim_start = 5.0
        self.stim_duration = 1.0

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

        # Reset infarct cells to resting state
        ic = self.sim.ic
        for name in ['V', 'm', 'h', 'j', 'd', 'f', 'f_Ca', 'X',
                     'Na_i', 'K_i', 'Ca_i', 'Ca_jsr', 'Ca_nsr']:
            self.sim.state[name][infarct_region] = getattr(ic, name)

        self.infarct_bounds = (x_min, x_max, y_min, y_max)

        n_infarct = np.sum(infarct_region)
        print(f"Created rectangular infarct:")
        print(f"  Size: {width}x{height} mm at ({x_center}, {y_center})")
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
        V_mV = self.sim.get_voltage_mV()
        self.im = self.ax_main.imshow(
            V_mV,
            origin='lower',
            extent=[0, self.sim.domain_size, 0, self.sim.domain_size],
            cmap='turbo',
            vmin=-90,
            vmax=50,
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
            self.ax_main.text(x, y + 1.5, name.replace('_', '\n'),
                            color='white', fontsize=7, ha='center', va='bottom',
                            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

        # Wave direction arrow
        arrow_y = self.sim.domain_size * 0.9
        self.ax_main.annotate(
            '', xy=(10, arrow_y), xytext=(2, arrow_y),
            arrowprops=dict(arrowstyle='->', color='yellow', lw=3)
        )
        self.ax_main.text(
            6, arrow_y + 1.5, 'Wave',
            color='yellow', fontsize=11, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5)
        )

        # Colorbar
        self.cbar = self.fig.colorbar(self.im, cax=self.ax_cbar)
        self.cbar.set_label('Voltage [mV]', fontsize=11)

        # Labels
        self.ax_main.set_xlabel('x [mm]', fontsize=12)
        self.ax_main.set_ylabel('y [mm]', fontsize=12)
        self.ax_main.set_title('Infarct Border Speedup Test (LRd94)', fontsize=14)

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

        self.ax_btn_start = self.fig.add_axes([0.25, 0.05, 0.10, 0.05])
        self.btn_start = Button(self.ax_btn_start, 'Start', color='lightgreen')
        self.btn_start.on_clicked(self._on_start)

        self.ax_btn_pause = self.fig.add_axes([0.37, 0.05, 0.10, 0.05])
        self.btn_pause = Button(self.ax_btn_pause, 'Pause', color='lightyellow')
        self.btn_pause.on_clicked(self._on_pause)

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

        if self.animation is None:
            self.animation = FuncAnimation(
                self.fig, self._animate_frame,
                interval=30,
                blit=False,
                cache_frame_data=False,
            )

        self.status_text.set_text('Simulation running - measuring activation times')
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

        self.sim.reset()
        x_center = self.sim.domain_size / 2
        y_center = self.sim.domain_size / 2
        self.create_rectangular_infarct(x_center, y_center, 25.0, 5.0)

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

        # Apply stimulus
        if self.sim.time >= self.stim_start and not self.stimulus_applied:
            self.sim.apply_planar_stimulus('left', width=3.0, amplitude=-100.0)
            self.stimulus_applied = True
            self.status_text.set_text('Planar wave launched')

        if self.stimulus_applied and self.sim.time >= self.stim_start + self.stim_duration:
            self.sim.clear_stimulus()

        # Run steps
        for _ in range(self.frame_skip):
            self.sim.step()

        # Check for activations
        for name, (i, j) in self.measure_indices.items():
            if self.activation_times[name] is None:
                if self.sim.state['V'][i, j] > self.activation_threshold:
                    self.activation_times[name] = self.sim.time
                    print(f"Activation at {name}: t = {self.sim.time:.1f} ms")

        self._update_display()

        # Calculate CV if all activated
        all_activated = all(t is not None for t in self.activation_times.values())
        if all_activated:
            self._display_results()

    def _display_results(self):
        """Display conduction velocity analysis."""
        t_free_start = self.activation_times['free_start']
        t_free_end = self.activation_times['free_end']
        t_border_start = self.activation_times['border_start']
        t_border_end = self.activation_times['border_end']

        dt_free = t_free_end - t_free_start
        dt_border = t_border_end - t_border_start

        x_free_start = self.measure_points['free_start'][0]
        x_free_end = self.measure_points['free_end'][0]
        x_border_start = self.measure_points['border_start'][0]
        x_border_end = self.measure_points['border_end'][0]

        dist_free = x_free_end - x_free_start
        dist_border = x_border_end - x_border_start

        cv_free = dist_free / dt_free if dt_free > 0 else 0
        cv_border = dist_border / dt_border if dt_border > 0 else 0

        speedup = ((cv_border - cv_free) / cv_free * 100) if cv_free > 0 else 0

        result_text = (
            f"\n{'='*40}\n"
            f"BORDER SPEEDUP ANALYSIS (LRd94)\n"
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
        """Update display."""
        V_mV = self.sim.get_voltage_mV()
        self.im.set_data(V_mV)

        V = self.sim.state['V']
        activated = np.sum((V > -30.0) & (self.sim.mask > 0.5))
        total_cells = np.sum(self.sim.mask > 0.5)

        info = (
            f"Time: {self.sim.time:.1f} ms\n"
            f"\n"
            f"Voltage [mV]:\n"
            f"  Min: {np.min(V):.1f}\n"
            f"  Max: {np.max(V):.1f}\n"
            f"\n"
            f"Activated: {activated}/{total_cells}\n"
            f"           ({activated/total_cells*100:.1f}%)\n"
            f"\n"
            f"Measurement Points:\n"
        )

        for name in ['free_start', 'free_end', 'border_start', 'border_end']:
            t_act = self.activation_times[name]
            status = f"t={t_act:.1f}ms" if t_act else "waiting..."
            info += f"  {name}: {status}\n"

        self.info_text.set_text(info)

        times_info = (
            "ACTIVATION TIMES\n"
            "===============\n"
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
        print("INFARCT BORDER SPEEDUP TEST (LRd94)")
        print("=" * 60)
        print("\nSetup:")
        print("  - Rectangular infarct (25x5 mm)")
        print("  - Planar wave from left")
        print("  - 4 measurement points")
        print("\nWatch for:")
        print("  - Does wave speed up near border?")
        print("=" * 60)

        self._update_display()
        plt.show()


def main():
    """Run infarct border speedup demo."""
    demo = InfarctBorderDemo()
    demo.run()


if __name__ == "__main__":
    main()
