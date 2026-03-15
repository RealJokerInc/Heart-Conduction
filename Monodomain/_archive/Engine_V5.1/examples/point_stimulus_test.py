"""
Point Stimulus Test Animation

Debug script to verify wave propagation symmetry.
Press SPACE to apply a point stimulus at the center.
The wave should propagate as a circle (isotropic) or ellipse (anisotropic).

Controls:
- SPACE: Apply point stimulus at center
- R: Reset simulation
- A: Toggle anisotropic/isotropic diffusion
- +/-: Adjust simulation speed
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
from tissue.diffusion import compute_D_min, validate_D_for_mesh, compute_D_from_cv


class PointStimulusTest:
    """Test wave propagation symmetry with point stimulus."""

    # APD-shortening parameters (produces APD ~150ms)
    APD_SHORTENING_PARAMS = {
        'GKr_scale': 2.5,
        'PCa_scale': 0.4,
    }
    APD_SHORTENED_MS = 150  # Approximate APD with above parameters
    APD_NORMAL_MS = 280     # Normal ORd APD

    def __init__(self, grid_size=300, domain_cm=6.0, isotropic=True):
        """
        Initialize simulation.

        Args:
            grid_size: Number of cells in each dimension
            domain_cm: Physical size of domain in cm
            isotropic: If True, use same CV in x and y
        """
        self.ny, self.nx = grid_size, grid_size
        self.domain_cm = domain_cm
        self.dx = domain_cm / grid_size
        self.isotropic = isotropic

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print("=" * 70)
        print("POINT STIMULUS TEST - Wave Propagation Symmetry")
        print("=" * 70)
        print(f"Grid: {self.ny} x {self.nx} = {self.ny * self.nx:,} cells")
        print(f"Domain: {domain_cm:.1f} x {domain_cm:.1f} cm")
        print(f"Resolution: {self.dx*10:.2f} mm/cell")
        print(f"Device: {self.device}")

        self._create_simulation()

        # Point stimulus parameters
        self.stim_radius_cm = 0.2  # 2mm radius
        self.stim_radius_cells = max(2, int(self.stim_radius_cm / self.dx))

        # Display setup
        self._setup_display()

        # FPS tracking
        self.frame_times = []
        self.max_frame_times = 30
        self.speed_multiplier = 1.0

        print()
        print("Controls:")
        print("  SPACE   - Apply point stimulus at center")
        print("  R       - Reset simulation")
        print("  A       - Toggle anisotropic/isotropic")
        print("  +/-     - Adjust speed")
        print("  Q/ESC   - Quit")
        print()

    def _create_simulation(self):
        """Create or recreate the simulation with current settings."""
        # Determine target CV values
        if self.isotropic:
            cv_long = 0.06  # 0.6 m/s
            cv_trans = 0.06  # Same as longitudinal
            print(f"Diffusion: ISOTROPIC (cv = 0.06 cm/ms = 0.6 m/s)")
        else:
            cv_long = 0.06  # 0.6 m/s
            cv_trans = 0.02  # 0.2 m/s (3:1 ratio)
            print(f"Diffusion: ANISOTROPIC (cv_x = 0.06, cv_y = 0.02 cm/ms)")

        # Compute D values and validate against D_min for this mesh/APD
        D_L = compute_D_from_cv(cv_long, self.dx, enforce_minimum=False, warn=False)
        D_T = compute_D_from_cv(cv_trans, self.dx, enforce_minimum=False, warn=False)

        # D_min safeguard: validate against mesh-dependent minimum
        # Using shortened APD since we use APD_SHORTENING_PARAMS
        D_min = compute_D_min(self.dx, self.APD_SHORTENED_MS)

        print(f"D_min for dx={self.dx*10:.2f}mm, APD={self.APD_SHORTENED_MS}ms: {D_min:.6f} cm²/ms")

        # Check and warn about D values
        status_L, msg_L = validate_D_for_mesh(
            D_L, self.dx, self.APD_SHORTENED_MS, direction="longitudinal"
        )
        status_T, msg_T = validate_D_for_mesh(
            D_T, self.dx, self.APD_SHORTENED_MS, direction="transverse"
        )

        # If transverse D is below minimum, clamp it to D_min with warning
        if status_T == "CRITICAL":
            print(f"\n*** WARNING: D_T={D_T:.6f} < D_min={D_min:.6f} ***")
            print(f"    Clamping D_T to D_min to prevent propagation failure.")
            print(f"    Actual CV_trans will be higher than requested.\n")
            D_T = D_min
        elif status_T == "WARNING":
            print(f"Note: D_T={D_T:.6f} is close to D_min={D_min:.6f} (marginal stability)")

        # Create simulation with validated D values
        self.sim = MonodomainSimulation(
            ny=self.ny, nx=self.nx,
            dx=self.dx, dy=self.dx,
            D_L=D_L, D_T=D_T,  # Use explicit D values instead of CV
            celltype=CellType.ENDO,
            device=self.device,
            params_override=self.APD_SHORTENING_PARAMS
        )

        print(f"Using D_L={D_L:.6f}, D_T={D_T:.6f} cm²/ms (ratio {D_L/D_T:.1f}:1)")

        # Time stepping
        dt_stability = self.sim.diffusion.get_stability_limit()
        self.dt = dt_stability * 0.8
        self.steps_per_frame = max(1, int(0.5 / self.dt))

        print(f"dt = {self.dt:.5f} ms, stability limit = {dt_stability:.5f} ms")

    def _setup_display(self):
        """Pre-compute static display elements."""
        target_size = 700
        self.scale = max(1, target_size // max(self.ny, self.nx))
        self.img_w = self.nx * self.scale
        self.img_h = self.ny * self.scale

        # Margins
        self.margin_left = 60
        self.margin_bottom = 50
        self.margin_right = 80
        self.margin_top = 50

        # Canvas dimensions
        self.canvas_h = self.img_h + self.margin_top + self.margin_bottom
        self.canvas_w = self.img_w + self.margin_left + self.margin_right

        # Pre-create static canvas
        self.static_canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
        self.static_canvas[:] = (30, 30, 30)

        # Colorbar
        cbar_x = self.margin_left + self.img_w + 15
        cbar_w = 20
        cbar_h = self.img_h
        cbar_y = self.margin_top

        gradient = np.linspace(255, 0, cbar_h).astype(np.uint8).reshape(-1, 1)
        gradient_2d = np.tile(gradient, (1, cbar_w))
        colorbar = cv2.applyColorMap(gradient_2d, cv2.COLORMAP_JET)
        self.static_canvas[cbar_y:cbar_y+cbar_h, cbar_x:cbar_x+cbar_w] = colorbar

        cv2.rectangle(self.static_canvas, (cbar_x, cbar_y),
                     (cbar_x + cbar_w, cbar_y + cbar_h), (200, 200, 200), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.static_canvas, '+50', (cbar_x + cbar_w + 5, cbar_y + 15),
                   font, 0.4, (255, 255, 255), 1)
        cv2.putText(self.static_canvas, '-90', (cbar_x + cbar_w + 5, cbar_y + cbar_h),
                   font, 0.4, (255, 255, 255), 1)
        cv2.putText(self.static_canvas, 'mV', (cbar_x + 2, cbar_y - 8),
                   font, 0.4, (255, 255, 255), 1)

        # Axis labels
        for i in range(int(self.domain_cm) + 1):
            x_pos = self.margin_left + int(i / self.domain_cm * self.img_w)
            cv2.line(self.static_canvas, (x_pos, self.margin_top + self.img_h),
                    (x_pos, self.margin_top + self.img_h + 5), (200, 200, 200), 1)
            cv2.putText(self.static_canvas, f'{i}', (x_pos - 5, self.margin_top + self.img_h + 20),
                       font, 0.4, (255, 255, 255), 1)

        for i in range(int(self.domain_cm) + 1):
            y_pos = self.margin_top + self.img_h - int(i / self.domain_cm * self.img_h)
            cv2.line(self.static_canvas, (self.margin_left - 5, y_pos),
                    (self.margin_left, y_pos), (200, 200, 200), 1)
            cv2.putText(self.static_canvas, f'{i}', (self.margin_left - 25, y_pos + 5),
                       font, 0.4, (255, 255, 255), 1)

    def apply_point_stimulus(self):
        """Apply point stimulus at center of domain."""
        V = self.sim.get_voltage()

        # Center coordinates
        cy, cx = self.ny // 2, self.nx // 2
        r = self.stim_radius_cells

        # Create circular stimulus region
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy*dy + dx*dx <= r*r:
                    y, x = cy + dy, cx + dx
                    if 0 <= y < self.ny and 0 <= x < self.nx:
                        V[y, x] = 20.0  # Strong depolarization

        self.sim.set_voltage(V)
        self.stim_time = self.sim.time
        print(f"Point stimulus at center (t = {self.sim.time:.1f} ms)")

    def reset(self):
        """Reset simulation."""
        self.sim.reset()
        self.stim_time = None
        print("Simulation reset")

    def toggle_anisotropy(self):
        """Toggle between isotropic and anisotropic diffusion."""
        self.isotropic = not self.isotropic
        self._create_simulation()
        self.stim_time = None

    def voltage_to_image(self, V):
        """Convert voltage to display image."""
        V_norm = np.clip((V + 90) / 140 * 255, 0, 255).astype(np.uint8)
        img = cv2.applyColorMap(V_norm, cv2.COLORMAP_JET)
        img = cv2.flip(img, 0)

        if self.scale > 1:
            img = cv2.resize(img, (self.img_w, self.img_h),
                           interpolation=cv2.INTER_NEAREST)

        canvas = self.static_canvas.copy()
        canvas[self.margin_top:self.margin_top+self.img_h,
               self.margin_left:self.margin_left+self.img_w] = img

        # Draw crosshairs at center
        cx = self.margin_left + self.img_w // 2
        cy = self.margin_top + self.img_h // 2
        cv2.line(canvas, (cx - 10, cy), (cx + 10, cy), (255, 255, 255), 1)
        cv2.line(canvas, (cx, cy - 10), (cx, cy + 10), (255, 255, 255), 1)

        return canvas

    def run(self):
        """Main simulation loop."""
        cv2.namedWindow('Point Stimulus Test', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Point Stimulus Test', 900, 900)

        self.stim_time = None
        print("Press SPACE to apply point stimulus at center")

        while True:
            frame_start = perf_counter()

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
                self.apply_point_stimulus()
            elif key == ord('r'):
                self.reset()
            elif key == ord('a'):
                self.toggle_anisotropy()
            elif key == ord('+') or key == ord('='):
                self.speed_multiplier = min(4.0, self.speed_multiplier * 1.5)
                print(f"Speed: {self.speed_multiplier:.1f}x")
            elif key == ord('-'):
                self.speed_multiplier = max(0.25, self.speed_multiplier / 1.5)
                print(f"Speed: {self.speed_multiplier:.1f}x")

            # Run simulation steps
            actual_steps = max(1, int(self.steps_per_frame * self.speed_multiplier))
            for _ in range(actual_steps):
                self.sim.step(self.dt)

            # Get voltage and create image
            V = self.sim.get_voltage().cpu().numpy()
            img = self.voltage_to_image(V)

            # FPS calculation
            frame_time = perf_counter() - frame_start
            self.frame_times.append(frame_time)
            if len(self.frame_times) > self.max_frame_times:
                self.frame_times.pop(0)
            fps = len(self.frame_times) / sum(self.frame_times)

            # Title
            mode = "ISOTROPIC" if self.isotropic else "ANISOTROPIC"
            title = f"Point Stimulus Test - {mode}"
            cv2.putText(img, title, (60, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 255, 255), 1)

            # Status
            V_min, V_max = V.min(), V.max()
            status = f"t={self.sim.time:.1f}ms | {fps:.1f}FPS | V:[{V_min:.0f},{V_max:.0f}]mV"
            cv2.putText(img, status, (60, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.45, (200, 200, 200), 1)

            # Wave radius measurement (if stimulus was applied)
            if self.stim_time is not None:
                elapsed = self.sim.time - self.stim_time
                if elapsed > 0:
                    # Measure wavefront radius in x and y directions
                    cy, cx = self.ny // 2, self.nx // 2

                    # Find wavefront in +x direction
                    rx = 0
                    for i in range(cx, self.nx):
                        if V[cy, i] > -40:
                            rx = i - cx
                        else:
                            break

                    # Find wavefront in +y direction
                    ry = 0
                    for j in range(cy, self.ny):
                        if V[j, cx] > -40:
                            ry = j - cy
                        else:
                            break

                    rx_cm = rx * self.dx
                    ry_cm = ry * self.dx

                    if rx > 0 or ry > 0:
                        cv_x = rx_cm / elapsed if elapsed > 0 else 0
                        cv_y = ry_cm / elapsed if elapsed > 0 else 0
                        ratio = ry_cm / rx_cm if rx_cm > 0 else 0

                        info = f"r_x={rx_cm:.2f}cm r_y={ry_cm:.2f}cm ratio={ratio:.2f}"
                        cv2.putText(img, info, (img.shape[1] - 350, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

                        cv_info = f"CV_x={cv_x*10:.2f}m/s CV_y={cv_y*10:.2f}m/s"
                        cv2.putText(img, cv_info, (img.shape[1] - 350, 55),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            cv2.imshow('Point Stimulus Test', img)

        cv2.destroyAllWindows()
        print("Test ended.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Point stimulus wave propagation test')
    parser.add_argument('--grid', type=int, default=300,
                       help='Grid size (default: 300)')
    parser.add_argument('--domain', type=float, default=6.0,
                       help='Domain size in cm (default: 6.0)')
    parser.add_argument('--anisotropic', action='store_true',
                       help='Start with anisotropic diffusion')
    args = parser.parse_args()

    sim = PointStimulusTest(
        grid_size=args.grid,
        domain_cm=args.domain,
        isotropic=not args.anisotropic
    )
    sim.run()


if __name__ == '__main__':
    main()
