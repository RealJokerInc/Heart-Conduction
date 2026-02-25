"""
Fast Real-Time Animation using Matplotlib

Uses smaller grid (200x200) for improved FPS (~5-10 FPS vs 0.6 FPS).
For high-resolution use tissue_animation.py.

Controls:
- Click STIMULATE button or press SPACE to apply stimulus
- Close window to quit
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation
from time import perf_counter

from ionic import CellType
from tissue import MonodomainSimulation


class FastMatplotlibSimulation:
    """Real-time tissue simulation with matplotlib at reduced resolution."""

    def __init__(self, grid_size=200):
        # Smaller grid for real-time performance
        self.ny, self.nx = grid_size, grid_size
        self.dx = 0.025  # 250um resolution

        domain_cm = self.nx * self.dx

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print("=" * 60)
        print("FAST Matplotlib Cardiac Simulation")
        print("=" * 60)
        print(f"Grid: {self.ny} x {self.nx} = {self.ny * self.nx:,} cells")
        print(f"Domain: {domain_cm:.1f} x {domain_cm:.1f} cm @ {self.dx*10:.0f}mm resolution")
        print(f"Device: {self.device}", end="")
        if self.device == 'cuda':
            print(f" ({torch.cuda.get_device_name(0)})")
        else:
            print()

        # Create simulation
        self.sim = MonodomainSimulation(
            ny=self.ny, nx=self.nx,
            dx=self.dx, dy=self.dx,
            cv_long=0.06,
            cv_trans=0.02,
            celltype=CellType.ENDO,
            device=self.device
        )

        # Time stepping
        dt_stability = self.sim.diffusion.get_stability_limit()
        self.dt = dt_stability * 0.8
        self.steps_per_frame = max(1, int(0.5 / self.dt))
        self.speed_multiplier = 1.0

        print(f"dt = {self.dt:.5f} ms")
        print(f"Steps per frame: {self.steps_per_frame} (~{self.steps_per_frame * self.dt:.2f} ms)")

        # Stimulus state
        self.stimulus_active = False
        self.stimulus_start_time = None
        self.stimulus_duration = 2.0

        # Timing
        self.start_time = None
        self.frame_count = 0

    def apply_stimulus(self):
        """Apply stimulus if active."""
        if self.stimulus_active:
            elapsed = self.sim.time - self.stimulus_start_time
            if elapsed < self.stimulus_duration:
                V = self.sim.get_voltage()
                stim_width = max(3, int(0.1 / self.dx))
                V[:, :stim_width] = 20.0
                self.sim.set_voltage(V)
            else:
                self.stimulus_active = False
                print(f"  Stimulus ended at t = {self.sim.time:.1f} ms")

    def on_stimulate_click(self, event):
        """Handle stimulus button click."""
        if not self.stimulus_active:
            self.stimulus_active = True
            self.stimulus_start_time = self.sim.time
            print(f"Stimulus applied at t = {self.sim.time:.1f} ms")

    def on_key_press(self, event):
        """Handle keyboard input."""
        if event.key == ' ':
            self.on_stimulate_click(None)
        elif event.key == '+' or event.key == '=':
            self.speed_multiplier = min(4.0, self.speed_multiplier * 1.5)
            self.speed_text.set_text(f'Speed: {self.speed_multiplier:.1f}x')
            print(f"Speed: {self.speed_multiplier:.1f}x")
        elif event.key == '-':
            self.speed_multiplier = max(0.25, self.speed_multiplier / 1.5)
            self.speed_text.set_text(f'Speed: {self.speed_multiplier:.1f}x')
            print(f"Speed: {self.speed_multiplier:.1f}x")

    def setup_figure(self):
        """Setup matplotlib figure."""
        self.fig = plt.figure(figsize=(10, 10))

        # Main axis
        self.ax = self.fig.add_axes([0.1, 0.18, 0.75, 0.72])

        domain = self.nx * self.dx
        V = self.sim.get_voltage().cpu().numpy()
        self.im = self.ax.imshow(
            V, cmap='RdBu_r', vmin=-90, vmax=50,
            origin='lower', aspect='equal',
            extent=[0, domain, 0, domain]
        )

        self.ax.set_xlabel('x (cm)', fontsize=12)
        self.ax.set_ylabel('y (cm)', fontsize=12)
        self.ax.set_title('Cardiac Wave Propagation (Fast Mode)', fontsize=14)

        # Colorbar
        cbar_ax = self.fig.add_axes([0.87, 0.18, 0.02, 0.72])
        self.cbar = self.fig.colorbar(self.im, cax=cbar_ax, label='V (mV)')

        # Time/FPS display
        self.time_text = self.ax.text(
            0.02, 0.98, 't = 0.0 ms | 0.0 fps',
            transform=self.ax.transAxes, fontsize=11,
            verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
        )

        # Speed display
        self.speed_text = self.ax.text(
            0.98, 0.02, f'Speed: {self.speed_multiplier:.1f}x',
            transform=self.ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9)
        )

        # Stimulus indicator
        self.stim_text = self.ax.text(
            0.98, 0.98, '',
            transform=self.ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            fontweight='bold', color='red',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9)
        )

        # Stimulate button
        button_ax = self.fig.add_axes([0.35, 0.02, 0.15, 0.05])
        self.stim_button = Button(button_ax, 'STIMULATE', color='lightgreen', hovercolor='green')
        self.stim_button.on_clicked(self.on_stimulate_click)
        self.stim_button.label.set_fontsize(12)
        self.stim_button.label.set_fontweight('bold')

        # Connect keyboard
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        print("\nControls:")
        print("  SPACE - Apply stimulus")
        print("  +/-   - Adjust speed")
        print("  Close window to quit\n")

    def update(self, frame):
        """Animation update."""
        if self.start_time is None:
            self.start_time = perf_counter()

        # Apply stimulus
        self.apply_stimulus()

        # Run simulation
        actual_steps = max(1, int(self.steps_per_frame * self.speed_multiplier))
        for _ in range(actual_steps):
            self.sim.step(self.dt)
            if self.stimulus_active:
                self.apply_stimulus()

        self.frame_count += 1

        # Update plot
        V = self.sim.get_voltage().cpu().numpy()
        self.im.set_array(V)

        # FPS
        elapsed = perf_counter() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        speed_ratio = self.sim.time / (elapsed * 1000) if elapsed > 0 else 0

        self.time_text.set_text(
            f't = {self.sim.time:.1f} ms | {fps:.1f} fps | {speed_ratio:.2f}x realtime'
        )

        # Stimulus indicator
        self.stim_text.set_text('STIMULATING' if self.stimulus_active else '')

        return [self.im, self.time_text, self.stim_text, self.speed_text]

    def run(self):
        """Run animation."""
        self.setup_figure()

        self.anim = FuncAnimation(
            self.fig, self.update,
            frames=None,
            interval=33,  # ~30 FPS target
            blit=True,
            cache_frame_data=False
        )

        plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Fast matplotlib cardiac simulation')
    parser.add_argument('--grid', type=int, default=200,
                       help='Grid size (default: 200)')
    args = parser.parse_args()

    sim = FastMatplotlibSimulation(grid_size=args.grid)
    sim.run()


if __name__ == '__main__':
    main()
