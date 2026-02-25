"""
Fast Real-Time Animation using OpenCV

Uses smaller grid (200x200) for real-time visualization at 20-30 FPS.
For high-resolution (500x500) use tissue_animation_opencv.py.

Controls:
- SPACE: Apply stimulus
- Q/ESC: Quit
- +/-: Adjust simulation speed
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


class FastSimulation:
    """Real-time tissue simulation at reduced resolution."""

    def __init__(self, grid_size=200):
        # Smaller grid for real-time performance
        self.ny, self.nx = grid_size, grid_size
        self.dx = 0.025  # 250um resolution (coarser for speed)

        # Domain size: grid_size * 0.025 = 5cm for 200x200
        domain_cm = self.nx * self.dx

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print("=" * 60)
        print("FAST Real-Time Cardiac Simulation")
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
            cv_long=0.06,   # 0.6 m/s
            cv_trans=0.02,  # 0.2 m/s
            celltype=CellType.ENDO,
            device=self.device
        )

        # Time stepping - fewer steps per frame for responsiveness
        dt_stability = self.sim.diffusion.get_stability_limit()
        self.dt = dt_stability * 0.8
        self.steps_per_frame = max(1, int(0.5 / self.dt))  # ~0.5ms per frame
        self.speed_multiplier = 1.0  # Adjustable

        print(f"dt = {self.dt:.5f} ms")
        print(f"Steps per frame: {self.steps_per_frame} (~{self.steps_per_frame * self.dt:.2f} ms)")
        print()
        print("Controls:")
        print("  SPACE   - Apply stimulus")
        print("  +/=     - Increase speed")
        print("  -       - Decrease speed")
        print("  Q/ESC   - Quit")
        print()

        # Stimulus state
        self.stimulus_active = False
        self.stimulus_start_time = None
        self.stimulus_duration = 2.0

        # FPS tracking
        self.frame_times = []
        self.max_frame_times = 30

        # Pre-compute static display elements for performance
        self._setup_display()

    def _setup_display(self):
        """Pre-compute static display elements (colorbar, axes, labels) once."""
        # Image scaling
        target_size = 700
        self.scale = target_size // max(self.ny, self.nx)
        self.img_w = self.nx * self.scale if self.scale > 1 else self.nx
        self.img_h = self.ny * self.scale if self.scale > 1 else self.ny

        # Margins
        self.margin_left = 60
        self.margin_bottom = 50
        self.margin_right = 80
        self.margin_top = 40

        # Canvas dimensions
        self.canvas_h = self.img_h + self.margin_top + self.margin_bottom
        self.canvas_w = self.img_w + self.margin_left + self.margin_right

        # Pre-create the static background canvas with colorbar and axes
        self.static_canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
        self.static_canvas[:] = (40, 40, 40)  # Dark gray background

        # Colorbar position
        cbar_x = self.margin_left + self.img_w + 15
        cbar_w = 20
        cbar_h = self.img_h
        cbar_y = self.margin_top

        # Pre-compute colorbar gradient ONCE using vectorized operations
        # Create gradient values: 255 at top (row 0), 0 at bottom (row cbar_h-1)
        gradient = np.linspace(255, 0, cbar_h).astype(np.uint8).reshape(-1, 1)
        gradient_2d = np.tile(gradient, (1, cbar_w))
        colorbar = cv2.applyColorMap(gradient_2d, cv2.COLORMAP_JET)
        self.static_canvas[cbar_y:cbar_y+cbar_h, cbar_x:cbar_x+cbar_w] = colorbar

        # Colorbar border
        cv2.rectangle(self.static_canvas, (cbar_x, cbar_y),
                     (cbar_x + cbar_w, cbar_y + cbar_h), (200, 200, 200), 1)

        # Colorbar labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.static_canvas, '+50', (cbar_x + cbar_w + 5, cbar_y + 15),
                   font, 0.4, (255, 255, 255), 1)
        cv2.putText(self.static_canvas, '0', (cbar_x + cbar_w + 5, cbar_y + cbar_h // 3 + 5),
                   font, 0.4, (255, 255, 255), 1)
        cv2.putText(self.static_canvas, '-40', (cbar_x + cbar_w + 5, cbar_y + 2 * cbar_h // 3 + 5),
                   font, 0.4, (255, 255, 255), 1)
        cv2.putText(self.static_canvas, '-90', (cbar_x + cbar_w + 5, cbar_y + cbar_h),
                   font, 0.4, (255, 255, 255), 1)
        cv2.putText(self.static_canvas, 'mV', (cbar_x + 2, cbar_y - 8),
                   font, 0.4, (255, 255, 255), 1)

        # Axis labels
        domain_x = self.nx * self.dx
        domain_y = self.ny * self.dx

        # X-axis tick marks and labels
        for i in range(int(domain_x) + 1):
            x_pos = self.margin_left + int(i / domain_x * self.img_w)
            cv2.line(self.static_canvas, (x_pos, self.margin_top + self.img_h),
                    (x_pos, self.margin_top + self.img_h + 5), (200, 200, 200), 1)
            cv2.putText(self.static_canvas, f'{i}', (x_pos - 5, self.margin_top + self.img_h + 20),
                       font, 0.4, (255, 255, 255), 1)
        cv2.putText(self.static_canvas, 'x (cm)',
                   (self.margin_left + self.img_w // 2 - 20, self.canvas_h - 5),
                   font, 0.45, (255, 255, 255), 1)

        # Y-axis tick marks and labels
        for i in range(int(domain_y) + 1):
            y_pos = self.margin_top + self.img_h - int(i / domain_y * self.img_h)
            cv2.line(self.static_canvas, (self.margin_left - 5, y_pos),
                    (self.margin_left, y_pos), (200, 200, 200), 1)
            cv2.putText(self.static_canvas, f'{i}', (self.margin_left - 25, y_pos + 5),
                       font, 0.4, (255, 255, 255), 1)

        # Y-axis label
        cv2.putText(self.static_canvas, 'y', (5, self.margin_top + self.img_h // 2 - 10),
                   font, 0.45, (255, 255, 255), 1)
        cv2.putText(self.static_canvas, '(cm)', (5, self.margin_top + self.img_h // 2 + 10),
                   font, 0.35, (255, 255, 255), 1)

        # Title
        domain = self.nx * self.dx
        title = f"Cardiac Wave Propagation ({domain:.1f}x{domain:.1f} cm)"
        cv2.putText(self.static_canvas, title, (60, 25), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 255), 1)

    def apply_stimulus(self):
        """Apply stimulus if active."""
        if self.stimulus_active:
            elapsed = self.sim.time - self.stimulus_start_time
            if elapsed < self.stimulus_duration:
                V = self.sim.get_voltage()
                # Strong stimulus on left edge (5 cells wide for coarser grid)
                stim_width = max(3, int(0.1 / self.dx))  # ~1mm
                V[:, :stim_width] = 20.0
                self.sim.set_voltage(V)
            else:
                self.stimulus_active = False
                print(f"Stimulus ended at t = {self.sim.time:.1f} ms")

    def voltage_to_image(self, V):
        """Convert voltage array to BGR image - fast version using pre-computed elements."""
        # Normalize: -90mV -> 0, +50mV -> 255
        V_norm = np.clip((V + 90) / 140 * 255, 0, 255).astype(np.uint8)

        # Use JET colormap
        img = cv2.applyColorMap(V_norm, cv2.COLORMAP_JET)

        # Flip for correct orientation
        img = cv2.flip(img, 0)

        # Upscale if needed
        if self.scale > 1:
            img = cv2.resize(img, (self.img_w, self.img_h),
                           interpolation=cv2.INTER_NEAREST)

        # Copy pre-computed static canvas and place voltage image
        canvas = self.static_canvas.copy()
        canvas[self.margin_top:self.margin_top+self.img_h,
               self.margin_left:self.margin_left+self.img_w] = img

        return canvas

    def run(self):
        """Main loop."""
        cv2.namedWindow('Cardiac Wave Propagation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Cardiac Wave Propagation', 900, 900)

        print("Starting simulation...")

        while True:
            frame_start = perf_counter()

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
                if not self.stimulus_active:
                    self.stimulus_active = True
                    self.stimulus_start_time = self.sim.time
                    print(f"Stimulus applied at t = {self.sim.time:.1f} ms")
            elif key == ord('+') or key == ord('='):
                self.speed_multiplier = min(4.0, self.speed_multiplier * 1.5)
                print(f"Speed: {self.speed_multiplier:.1f}x")
            elif key == ord('-'):
                self.speed_multiplier = max(0.25, self.speed_multiplier / 1.5)
                print(f"Speed: {self.speed_multiplier:.1f}x")

            # Stimulus
            self.apply_stimulus()

            # Run simulation
            actual_steps = max(1, int(self.steps_per_frame * self.speed_multiplier))
            for _ in range(actual_steps):
                self.sim.step(self.dt)
                if self.stimulus_active:
                    self.apply_stimulus()

            # Get voltage and convert to image (includes colorbar and axes)
            V = self.sim.get_voltage().cpu().numpy()
            img = self.voltage_to_image(V)

            # Calculate FPS
            frame_time = perf_counter() - frame_start
            self.frame_times.append(frame_time)
            if len(self.frame_times) > self.max_frame_times:
                self.frame_times.pop(0)
            fps = len(self.frame_times) / sum(self.frame_times)

            # Compute simulation speed
            sim_ms_per_real_s = self.sim.time / sum(self.frame_times) if self.frame_times else 0

            # Status bar at bottom (dynamic text - only thing that changes)
            V_min, V_max = V.min(), V.max()
            status = f"t={self.sim.time:.1f}ms | {fps:.1f}FPS | {sim_ms_per_real_s/1000:.2f}x realtime | V:[{V_min:.0f},{V_max:.0f}]mV | speed:{self.speed_multiplier:.1f}x"
            cv2.putText(img, status, (60, img.shape[0] - 35), cv2.FONT_HERSHEY_SIMPLEX,
                       0.45, (200, 200, 200), 1)

            if self.stimulus_active:
                cv2.putText(img, "STIMULATING", (img.shape[1] - 150, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Display
            cv2.imshow('Cardiac Wave Propagation', img)

        cv2.destroyAllWindows()
        print("Simulation ended.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Fast real-time cardiac simulation')
    parser.add_argument('--grid', type=int, default=200,
                       help='Grid size (default: 200 for 200x200)')
    args = parser.parse_args()

    sim = FastSimulation(grid_size=args.grid)
    sim.run()


if __name__ == '__main__':
    main()
