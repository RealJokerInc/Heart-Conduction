"""
Debug Diffusion Operator - Pure Diffusion Test

This script tests the diffusion operator in isolation (no ionic currents).
A Gaussian initial condition is diffused and the shape is measured.

Expected results:
- Isotropic: Circular Gaussian spreading
- Anisotropic: Elliptical Gaussian spreading

If anisotropic shows rectangular spreading, the FVM implementation has a bug.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import cv2
from time import perf_counter

from tissue.diffusion import DiffusionOperator, get_diffusion_params


class PureDiffusionTest:
    """Test diffusion operator in isolation."""

    def __init__(self, grid_size=200, domain_cm=4.0, isotropic=True):
        self.ny, self.nx = grid_size, grid_size
        self.domain_cm = domain_cm
        self.dx = domain_cm / grid_size
        self.isotropic = isotropic

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print("=" * 70)
        print("PURE DIFFUSION TEST - No Ionic Currents")
        print("=" * 70)
        print(f"Grid: {self.ny} x {self.nx}")
        print(f"Domain: {domain_cm:.1f} x {domain_cm:.1f} cm")
        print(f"dx = {self.dx*10:.2f} mm")

        self._create_operator()
        self._setup_display()

        self.speed_multiplier = 1.0
        self.frame_times = []

    def _create_operator(self):
        """Create diffusion operator."""
        if self.isotropic:
            # Same D in both directions
            D_L, D_T = get_diffusion_params(self.dx, cv_long=0.06, cv_trans=0.06)
            print(f"ISOTROPIC: D = {D_L:.6f} cm²/ms")
        else:
            # Different D in x and y
            D_L, D_T = get_diffusion_params(self.dx, cv_long=0.06, cv_trans=0.02)
            print(f"ANISOTROPIC: D_L = {D_L:.6f}, D_T = {D_T:.6f} cm²/ms")
            print(f"  Ratio D_L/D_T = {D_L/D_T:.2f}")

        self.D_L = D_L
        self.D_T = D_T

        self.diffusion = DiffusionOperator(
            ny=self.ny, nx=self.nx,
            dx=self.dx, dy=self.dx,
            D_L=D_L, D_T=D_T,
            fiber_angle=0.0,
            device=self.device
        )

        # Get stable timestep
        self.dt = self.diffusion.get_stability_limit() * 0.8
        print(f"dt = {self.dt:.5f} ms")

        # Initialize voltage field
        self.V = torch.zeros(self.ny, self.nx, dtype=torch.float64, device=self.device)
        self.time = 0.0

    def _setup_display(self):
        """Setup display canvas."""
        target_size = 600
        self.scale = max(1, target_size // max(self.ny, self.nx))
        self.img_w = self.nx * self.scale
        self.img_h = self.ny * self.scale

    def reset(self):
        """Reset to Gaussian initial condition at center."""
        self.V.zero_()
        self.time = 0.0

        # Create Gaussian blob at center
        cy, cx = self.ny // 2, self.nx // 2
        sigma_cells = 5  # Standard deviation in cells

        y_idx = torch.arange(self.ny, device=self.device, dtype=torch.float64)
        x_idx = torch.arange(self.nx, device=self.device, dtype=torch.float64)
        yy, xx = torch.meshgrid(y_idx, x_idx, indexing='ij')

        r2 = (xx - cx)**2 + (yy - cy)**2
        self.V = 100.0 * torch.exp(-r2 / (2 * sigma_cells**2))

        print(f"Reset: Gaussian at center, sigma={sigma_cells} cells")

    def toggle_anisotropy(self):
        """Toggle between isotropic and anisotropic."""
        self.isotropic = not self.isotropic
        self._create_operator()
        self.reset()

    def step(self):
        """Take one diffusion step (no ionic currents)."""
        # Pure diffusion: dV/dt = ∇·(D∇V)
        # Forward Euler: V_new = V + dt * ∇·(D∇V)
        diff = self.diffusion.apply(self.V)
        self.V = self.V + self.dt * diff
        self.time += self.dt

    def voltage_to_image(self, V):
        """Convert voltage to display image."""
        V_np = V.cpu().numpy()

        # Normalize to 0-255
        V_max = V_np.max()
        if V_max > 0:
            V_norm = (V_np / V_max * 255).astype(np.uint8)
        else:
            V_norm = np.zeros_like(V_np, dtype=np.uint8)

        img = cv2.applyColorMap(V_norm, cv2.COLORMAP_JET)
        img = cv2.flip(img, 0)

        if self.scale > 1:
            img = cv2.resize(img, (self.img_w, self.img_h),
                           interpolation=cv2.INTER_NEAREST)

        return img

    def measure_shape(self):
        """Measure the shape of the diffusing blob."""
        V_np = self.V.cpu().numpy()
        V_max = V_np.max()

        if V_max < 1.0:
            return 0, 0, 0

        # Find contour at 50% of max
        threshold = V_max * 0.5
        cy, cx = self.ny // 2, self.nx // 2

        # Measure radius in +x direction
        rx = 0
        for i in range(cx, self.nx):
            if V_np[cy, i] > threshold:
                rx = i - cx
            else:
                break

        # Measure radius in +y direction
        ry = 0
        for j in range(cy, self.ny):
            if V_np[j, cx] > threshold:
                ry = j - cy
            else:
                break

        ratio = ry / rx if rx > 0 else 0

        return rx * self.dx, ry * self.dx, ratio

    def run(self):
        """Main loop."""
        cv2.namedWindow('Pure Diffusion Test', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Pure Diffusion Test', 800, 800)

        self.reset()
        print("\nControls:")
        print("  SPACE - Reset with Gaussian")
        print("  A     - Toggle anisotropic/isotropic")
        print("  +/-   - Adjust speed")
        print("  Q/ESC - Quit")
        print()

        while True:
            frame_start = perf_counter()

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
                self.reset()
            elif key == ord('a'):
                self.toggle_anisotropy()
            elif key == ord('+') or key == ord('='):
                self.speed_multiplier = min(10.0, self.speed_multiplier * 1.5)
                print(f"Speed: {self.speed_multiplier:.1f}x")
            elif key == ord('-'):
                self.speed_multiplier = max(0.1, self.speed_multiplier / 1.5)
                print(f"Speed: {self.speed_multiplier:.1f}x")

            # Run simulation steps
            steps = max(1, int(10 * self.speed_multiplier))
            for _ in range(steps):
                self.step()

            # Create image
            img = self.voltage_to_image(self.V)

            # Draw crosshairs
            cx = self.img_w // 2
            cy = self.img_h // 2
            cv2.line(img, (cx - 20, cy), (cx + 20, cy), (255, 255, 255), 1)
            cv2.line(img, (cx, cy - 20), (cx, cy + 20), (255, 255, 255), 1)

            # Measure and display shape
            rx, ry, ratio = self.measure_shape()

            # Add info overlay
            mode = "ISOTROPIC" if self.isotropic else "ANISOTROPIC"
            cv2.putText(img, f"PURE DIFFUSION - {mode}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.putText(img, f"t = {self.time:.1f} ms", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.putText(img, f"r_x = {rx:.2f} cm, r_y = {ry:.2f} cm", (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.putText(img, f"ratio r_y/r_x = {ratio:.2f}", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            if not self.isotropic:
                expected_ratio = np.sqrt(self.D_T / self.D_L)
                cv2.putText(img, f"expected ratio = {expected_ratio:.2f}", (10, 125),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # FPS
            frame_time = perf_counter() - frame_start
            self.frame_times.append(frame_time)
            if len(self.frame_times) > 30:
                self.frame_times.pop(0)
            fps = len(self.frame_times) / sum(self.frame_times)
            cv2.putText(img, f"{fps:.0f} FPS", (img.shape[1] - 80, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow('Pure Diffusion Test', img)

        cv2.destroyAllWindows()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Pure diffusion test')
    parser.add_argument('--grid', type=int, default=200)
    parser.add_argument('--domain', type=float, default=4.0)
    parser.add_argument('--anisotropic', action='store_true')
    args = parser.parse_args()

    test = PureDiffusionTest(
        grid_size=args.grid,
        domain_cm=args.domain,
        isotropic=not args.anisotropic
    )
    test.run()


if __name__ == '__main__':
    main()
