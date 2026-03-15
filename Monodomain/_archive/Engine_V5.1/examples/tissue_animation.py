"""
Live Animation of Tissue Wave Propagation (V5.1 PyTorch GPU)

Shows real-time wave propagation across 2D tissue.
Interactive stimulus button to inject current at left edge.

This version uses PyTorch GPU acceleration for high performance.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')  # Interactive window
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation
from time import perf_counter

from ionic import CellType
from tissue import MonodomainSimulation


class InteractiveSimulation:
    """Interactive tissue simulation with stimulus button."""

    def __init__(self):
        # Domain: 5cm x 5cm at 100um resolution = 500x500 cells
        self.ny, self.nx = 500, 500
        self.dx = 0.01  # 100 um = 0.01 cm

        # Determine device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print("Setting up 5cm x 5cm tissue simulation...")
        print(f"Grid: {self.ny} x {self.nx} = {self.ny * self.nx:,} cells")
        print(f"Device: {self.device}", end="")
        if self.device == 'cuda':
            print(f" ({torch.cuda.get_device_name(0)})")
        else:
            print()

        # Create simulation with CV-based parameters
        # Human ventricular conduction velocities:
        #   Longitudinal: 0.6 m/s (0.06 cm/ms)
        #   Transverse: 0.2 m/s (0.02 cm/ms)
        self.sim = MonodomainSimulation(
            ny=self.ny, nx=self.nx,
            dx=self.dx, dy=self.dx,
            cv_long=0.06,   # Target CV: 0.6 m/s longitudinal
            cv_trans=0.02,  # Target CV: 0.2 m/s transverse
            celltype=CellType.ENDO,
            device=self.device
        )

        # Use stable time step (prioritize stability)
        dt_stability = self.sim.diffusion.get_stability_limit()
        self.dt = dt_stability * 0.8  # 80% of stability limit for safety
        self.steps_per_frame = max(1, int(0.5 / self.dt))  # Update display every ~0.5 ms

        # Note: Animation FPS is limited by matplotlib rendering (~1-2 fps for 500x500)
        # The actual GPU simulation is ~1.5x faster than CPU, but display is the bottleneck

        print(f"Domain: {self.ny*self.dx:.1f} x {self.nx*self.dx:.1f} cm")
        print(f"Target CV: {self.sim.cv_long*10:.1f} m/s (long), {self.sim.cv_trans*10:.1f} m/s (trans)")
        print(f"D_L = {self.sim.D_L:.6f}, D_T = {self.sim.D_T:.6f} cm^2/ms")
        print(f"Stability limit: {dt_stability:.5f} ms")
        print(f"Using dt = {self.dt:.5f} ms ({self.dt/dt_stability*100:.0f}% of limit)")
        print(f"Display update: every {self.steps_per_frame} steps ({self.steps_per_frame * self.dt:.2f} ms)")

        # Stimulus state
        self.stimulus_active = False
        self.stimulus_start_time = None
        self.stimulus_duration = 2.0  # ms
        self.stimulus_amplitude = 80.0  # uA/uF

        # Stimulus region: left edge, 3 cells wide
        self.stim_i_slice = slice(None)  # All rows
        self.stim_j_slice = slice(0, 3)  # First 3 columns

        # Track timing
        self.start_time = None
        self.frame_count = 0

    def apply_stimulus(self):
        """Apply stimulus current to left edge."""
        if self.stimulus_active:
            elapsed = self.sim.time - self.stimulus_start_time
            if elapsed < self.stimulus_duration:
                # Strong depolarization (+20mV) to reliably trigger AP
                V = self.sim.get_voltage()
                V[self.stim_i_slice, self.stim_j_slice] = 20.0
                self.sim.set_voltage(V)
            else:
                self.stimulus_active = False
                print(f"  Stimulus ended at t = {self.sim.time:.1f} ms")

    def on_stimulate_click(self, event):
        """Handle stimulus button click."""
        if not self.stimulus_active:
            self.stimulus_active = True
            self.stimulus_start_time = self.sim.time
            print(f"Stimulus applied at t = {self.sim.time:.1f} ms (left edge)")

    def setup_figure(self):
        """Setup the matplotlib figure with button."""
        # Create figure with space for button
        self.fig = plt.figure(figsize=(10, 10))

        # Main axis for the simulation
        self.ax = self.fig.add_axes([0.1, 0.15, 0.8, 0.75])

        # Initial voltage display (transfer from GPU to CPU for plotting)
        V = self.sim.get_voltage().cpu().numpy()
        self.im = self.ax.imshow(
            V, cmap='RdBu_r', vmin=-90, vmax=50,
            origin='lower', aspect='equal',
            extent=[0, self.nx*self.dx, 0, self.ny*self.dx]
        )

        self.ax.set_xlabel('x (cm)', fontsize=12)
        self.ax.set_ylabel('y (cm)', fontsize=12)
        self.ax.set_title('Cardiac Wave Propagation (5cm x 5cm) [V5.1 GPU]', fontsize=14)

        # Colorbar
        cbar_ax = self.fig.add_axes([0.92, 0.15, 0.02, 0.75])
        self.cbar = self.fig.colorbar(self.im, cax=cbar_ax, label='V (mV)')

        # Time display text
        self.time_text = self.ax.text(
            0.02, 0.98, 't = 0.0 ms | elapsed: 0.0 s | 0 fps',
            transform=self.ax.transAxes, fontsize=11,
            verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
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
        button_ax = self.fig.add_axes([0.4, 0.02, 0.2, 0.06])
        self.stim_button = Button(button_ax, 'STIMULATE', color='lightgreen', hovercolor='green')
        self.stim_button.on_clicked(self.on_stimulate_click)
        self.stim_button.label.set_fontsize(14)
        self.stim_button.label.set_fontweight('bold')

        print("\nStarting animation...")
        print("Click 'STIMULATE' button to inject current at left edge")
        print("Close window to stop\n")

    def update(self, frame):
        """Update function for animation."""
        # Track FPS
        if self.start_time is None:
            self.start_time = perf_counter()

        # Apply stimulus if active
        self.apply_stimulus()

        # Run simulation steps
        for _ in range(self.steps_per_frame):
            self.sim.step(self.dt)
            # Check stimulus during each step
            if self.stimulus_active:
                self.apply_stimulus()

        self.frame_count += 1

        # Update plot (transfer from GPU to CPU for display)
        V = self.sim.get_voltage().cpu().numpy()
        self.im.set_array(V)

        # Compute FPS
        elapsed = perf_counter() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0

        # Compute simulation speed ratio
        sim_time = self.sim.time
        speed_ratio = sim_time / (elapsed * 1000) if elapsed > 0 else 0  # sim_ms per real_ms

        # Update time display
        self.time_text.set_text(
            f't = {sim_time:.1f} ms | elapsed: {elapsed:.1f} s | {fps:.1f} fps | {speed_ratio:.2f}x'
        )

        # Update stimulus indicator
        if self.stimulus_active:
            self.stim_text.set_text('STIMULATING')
        else:
            self.stim_text.set_text('')

        return [self.im, self.time_text, self.stim_text]

    def run(self):
        """Run the interactive animation."""
        self.setup_figure()

        # Create animation (blit=False for button compatibility)
        self.anim = FuncAnimation(
            self.fig, self.update,
            frames=None,  # Run indefinitely
            interval=50,  # 50 ms between frames (20 fps target)
            blit=True,
            cache_frame_data=False
        )

        plt.show()


def run_animation():
    """Run the interactive tissue animation."""
    sim = InteractiveSimulation()
    sim.run()


if __name__ == '__main__':
    run_animation()
