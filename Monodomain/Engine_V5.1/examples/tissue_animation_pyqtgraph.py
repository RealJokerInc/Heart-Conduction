"""
Live Animation using PyQtGraph (Fast Scientific Visualization)

PyQtGraph is optimized for real-time scientific plotting and can achieve
30-60 FPS with proper configuration. Offers more features than OpenCV
while being much faster than matplotlib.

Requires: pip install pyqtgraph PyQt5
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from time import perf_counter

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets
except ImportError:
    print("PyQtGraph not installed. Install with: pip install pyqtgraph PyQt5")
    sys.exit(1)

from ionic import CellType
from tissue import MonodomainSimulation


class PyQtGraphSimulation:
    """Fast tissue simulation visualization using PyQtGraph."""

    def __init__(self):
        # Domain: 5cm x 5cm at 100um resolution
        self.ny, self.nx = 500, 500
        self.dx = 0.01

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print("=" * 60)
        print("Tissue Simulation with PyQtGraph Rendering")
        print("=" * 60)
        print(f"Grid: {self.ny} x {self.nx} = {self.ny * self.nx:,} cells")
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
        self.steps_per_frame = max(1, int(1.0 / self.dt))

        print(f"dt = {self.dt:.5f} ms")
        print(f"Steps per frame: {self.steps_per_frame}")

        # Stimulus state
        self.stimulus_active = False
        self.stimulus_start_time = None
        self.stimulus_duration = 2.0

        # FPS tracking
        self.last_time = perf_counter()
        self.frame_count = 0
        self.fps = 0.0

    def setup_ui(self):
        """Setup PyQtGraph UI."""
        # Create application
        self.app = pg.mkQApp("Cardiac Simulation")

        # Create main window
        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle('Cardiac Wave Propagation (PyQtGraph)')
        self.win.resize(900, 900)

        # Central widget
        central = QtWidgets.QWidget()
        self.win.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # Create ImageView (fast image display widget)
        self.imv = pg.ImageView()
        layout.addWidget(self.imv)

        # Configure colormap (RdBu_r like)
        colors = [
            (0, 0, 255),    # Blue at -90mV
            (255, 255, 255), # White at ~-20mV
            (255, 0, 0)     # Red at +50mV
        ]
        positions = [0.0, 0.5, 1.0]
        cmap = pg.ColorMap(positions, colors)
        self.imv.setColorMap(cmap)

        # Hide histogram and ROI for speed
        self.imv.ui.histogram.hide()
        self.imv.ui.roiBtn.hide()
        self.imv.ui.menuBtn.hide()

        # Initial image
        V = self.sim.get_voltage().cpu().numpy()
        self.imv.setImage(V.T, levels=(-90, 50))

        # Button panel
        btn_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(btn_layout)

        # Stimulate button
        self.stim_btn = QtWidgets.QPushButton('STIMULATE (Space)')
        self.stim_btn.setStyleSheet("background-color: lightgreen; font-weight: bold; font-size: 14px; padding: 10px;")
        self.stim_btn.clicked.connect(self.on_stimulate)
        btn_layout.addWidget(self.stim_btn)

        # Status label
        self.status_label = QtWidgets.QLabel('t = 0.0 ms | 0.0 FPS')
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        btn_layout.addWidget(self.status_label)

        # Keyboard shortcut
        self.win.keyPressEvent = self.on_key_press

        # Timer for animation
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(16)  # ~60 FPS target

        self.win.show()

    def on_key_press(self, event):
        """Handle keyboard input."""
        if event.key() == QtCore.Qt.Key.Key_Space:
            self.on_stimulate()
        elif event.key() == QtCore.Qt.Key.Key_Q or event.key() == QtCore.Qt.Key.Key_Escape:
            self.app.quit()

    def on_stimulate(self):
        """Apply stimulus."""
        if not self.stimulus_active:
            self.stimulus_active = True
            self.stimulus_start_time = self.sim.time
            self.stim_btn.setStyleSheet("background-color: red; font-weight: bold; font-size: 14px; padding: 10px;")
            print(f"Stimulus applied at t = {self.sim.time:.1f} ms")

    def apply_stimulus(self):
        """Apply stimulus if active."""
        if self.stimulus_active:
            elapsed = self.sim.time - self.stimulus_start_time
            if elapsed < self.stimulus_duration:
                V = self.sim.get_voltage()
                # Strong depolarization (+20mV) to reliably trigger AP
                V[:, :3] = 20.0
                self.sim.set_voltage(V)
            else:
                self.stimulus_active = False
                self.stim_btn.setStyleSheet("background-color: lightgreen; font-weight: bold; font-size: 14px; padding: 10px;")
                print(f"Stimulus ended at t = {self.sim.time:.1f} ms")

    def update(self):
        """Animation update."""
        # Apply stimulus
        self.apply_stimulus()

        # Run simulation steps
        for _ in range(self.steps_per_frame):
            self.sim.step(self.dt)
            if self.stimulus_active:
                self.apply_stimulus()

        # Update image
        V = self.sim.get_voltage().cpu().numpy()
        self.imv.setImage(V.T, levels=(-90, 50), autoRange=False, autoLevels=False)

        # Update FPS
        self.frame_count += 1
        current_time = perf_counter()
        elapsed = current_time - self.last_time
        if elapsed > 0.5:  # Update FPS every 0.5s
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_time = current_time

        # Update status
        sim_speed = self.sim.time / (current_time * 1000) if current_time > 0 else 0
        self.status_label.setText(
            f't = {self.sim.time:.1f} ms | {self.fps:.1f} FPS | {sim_speed:.3f}x realtime'
        )

    def run(self):
        """Run the application."""
        self.setup_ui()
        print("\nControls:")
        print("  SPACE - Apply stimulus")
        print("  Q/ESC - Quit")
        print()
        self.app.exec()


def run_pyqtgraph_animation():
    """Run the PyQtGraph-based animation."""
    sim = PyQtGraphSimulation()
    sim.run()


if __name__ == '__main__':
    run_pyqtgraph_animation()
