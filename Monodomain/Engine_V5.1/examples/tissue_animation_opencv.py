"""
Live Animation using OpenCV (Fast Rendering)

OpenCV's imshow is optimized for real-time video display and can achieve
60+ FPS, much faster than matplotlib's ~1-2 FPS for 500x500 images.

Controls:
- SPACE: Apply stimulus
- Q/ESC: Quit
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import cv2
from time import perf_counter

from ionic import CellType
from tissue import MonodomainSimulation


class OpenCVSimulation:
    """Fast tissue simulation visualization using OpenCV."""

    def __init__(self):
        # Domain: 5cm x 5cm at 100um resolution = 500x500 cells
        self.ny, self.nx = 500, 500
        self.dx = 0.01

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print("=" * 60)
        print("Tissue Simulation with OpenCV Rendering")
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
        # More steps per frame for faster simulation progression
        self.steps_per_frame = max(1, int(1.0 / self.dt))  # ~1ms per frame

        print(f"dt = {self.dt:.5f} ms")
        print(f"Steps per frame: {self.steps_per_frame}")
        print()
        print("Controls:")
        print("  SPACE - Apply stimulus at left edge")
        print("  Q/ESC - Quit")
        print()

        # Stimulus state
        self.stimulus_active = False
        self.stimulus_start_time = None
        self.stimulus_duration = 2.0

        # Colormap (blue-white-red like RdBu_r)
        self.colormap = self._create_colormap()

        # FPS tracking
        self.frame_times = []
        self.max_frame_times = 30

    def _create_colormap(self):
        """Create RdBu_r-like colormap for voltage display."""
        # Create 256-level colormap: blue (-90mV) -> white (0mV) -> red (+50mV)
        colormap = np.zeros((256, 1, 3), dtype=np.uint8)

        for i in range(256):
            # Map 0-127 to blue->white, 128-255 to white->red
            if i < 128:
                # Blue to white
                t = i / 127.0
                colormap[i, 0] = [
                    int(255 * t),      # B: 0->255
                    int(255 * t),      # G: 0->255
                    255                 # R: 255
                ]
            else:
                # White to red
                t = (i - 128) / 127.0
                colormap[i, 0] = [
                    int(255 * (1 - t)),  # B: 255->0
                    int(255 * (1 - t)),  # G: 255->0
                    255                   # R: 255
                ]

        return colormap

    def voltage_to_image(self, V):
        """Convert voltage array to BGR image for OpenCV."""
        # Normalize voltage: -90mV -> 0, +50mV -> 255
        V_norm = np.clip((V + 90) / 140 * 255, 0, 255).astype(np.uint8)

        # Apply colormap
        img = cv2.applyColorMap(V_norm, cv2.COLORMAP_JET)

        # Flip for correct orientation (origin='lower' equivalent)
        img = cv2.flip(img, 0)

        return img

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
                print(f"Stimulus ended at t = {self.sim.time:.1f} ms")

    def run(self):
        """Main loop."""
        cv2.namedWindow('Cardiac Wave Propagation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Cardiac Wave Propagation', 800, 800)

        print("Starting simulation...")

        while True:
            frame_start = perf_counter()

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord(' '):  # Space
                if not self.stimulus_active:
                    self.stimulus_active = True
                    self.stimulus_start_time = self.sim.time
                    print(f"Stimulus applied at t = {self.sim.time:.1f} ms")

            # Apply stimulus
            self.apply_stimulus()

            # Run simulation steps
            for _ in range(self.steps_per_frame):
                self.sim.step(self.dt)
                if self.stimulus_active:
                    self.apply_stimulus()

            # Get voltage and convert to image
            V = self.sim.get_voltage().cpu().numpy()
            img = self.voltage_to_image(V)

            # Calculate FPS
            frame_time = perf_counter() - frame_start
            self.frame_times.append(frame_time)
            if len(self.frame_times) > self.max_frame_times:
                self.frame_times.pop(0)
            fps = len(self.frame_times) / sum(self.frame_times)

            # Add text overlay
            sim_speed = self.sim.time / (sum(self.frame_times) * 1000) if self.frame_times else 0
            text = f"t={self.sim.time:.1f}ms | {fps:.1f} FPS | {sim_speed:.2f}x realtime"
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (255, 255, 255), 2)

            if self.stimulus_active:
                cv2.putText(img, "STIMULATING", (img.shape[1] - 150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display
            cv2.imshow('Cardiac Wave Propagation', img)

        cv2.destroyAllWindows()
        print("Simulation ended.")


def run_opencv_animation():
    """Run the OpenCV-based animation."""
    sim = OpenCVSimulation()
    sim.run()


if __name__ == '__main__':
    run_opencv_animation()
