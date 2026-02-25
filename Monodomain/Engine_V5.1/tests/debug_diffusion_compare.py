"""
Diffusion Operator Comparison Test

Compare three implementations:
1. FVM Isotropic (convolution-based) - KNOWN WORKING
2. FVM Anisotropic (flux-based) - SUSPECTED BUG
3. FDM Reference (simple finite difference) - VALIDATION

Run side by side to identify where the bug is.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from time import perf_counter

from tissue.diffusion import DiffusionOperator, get_diffusion_params


class FDMDiffusionReference:
    """
    Simple Finite Difference Method reference implementation.

    For anisotropic diffusion with fiber_angle=0 (D_xy=0):
        ∇·(D∇V) = D_L * ∂²V/∂x² + D_T * ∂²V/∂y²

    Using standard 5-point stencil:
        ∂²V/∂x² ≈ (V[i,j+1] - 2*V[i,j] + V[i,j-1]) / dx²
        ∂²V/∂y² ≈ (V[i+1,j] - 2*V[i,j] + V[i-1,j]) / dy²
    """

    def __init__(self, ny, nx, dx, dy, D_L, D_T, device='cuda'):
        self.ny = ny
        self.nx = nx
        self.dx = dx
        self.dy = dy
        self.D_L = D_L
        self.D_T = D_T
        self.device = device
        self.dtype = torch.float64

        # Build convolution kernel for anisotropic Laplacian
        # This is the CORRECT way to do it with convolution
        dx2 = dx ** 2
        dy2 = dy ** 2

        kernel = torch.zeros(3, 3, dtype=self.dtype, device=device)
        kernel[0, 1] = D_T / dy2      # North (y+1)
        kernel[2, 1] = D_T / dy2      # South (y-1)
        kernel[1, 0] = D_L / dx2      # West (x-1)
        kernel[1, 2] = D_L / dx2      # East (x+1)
        kernel[1, 1] = -2*D_L/dx2 - 2*D_T/dy2  # Center

        self.kernel = kernel.unsqueeze(0).unsqueeze(0)

    def apply(self, V):
        """Apply anisotropic Laplacian using convolution."""
        V_4d = V.unsqueeze(0).unsqueeze(0)
        V_padded = F.pad(V_4d, (1, 1, 1, 1), mode='replicate')
        diff = F.conv2d(V_padded, self.kernel)
        return diff[0, 0]

    def get_stability_limit(self):
        """Maximum stable timestep."""
        # dt <= 1 / (2 * (D_L/dx² + D_T/dy²))
        return 0.5 / (self.D_L / self.dx**2 + self.D_T / self.dy**2) * 0.9


class DiffusionComparisonTest:
    """Compare diffusion implementations side by side."""

    def __init__(self, grid_size=150, domain_cm=3.0):
        self.ny, self.nx = grid_size, grid_size
        self.domain_cm = domain_cm
        self.dx = domain_cm / grid_size

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print("=" * 70)
        print("DIFFUSION COMPARISON TEST")
        print("=" * 70)
        print(f"Grid: {self.ny} x {self.nx}, dx = {self.dx*10:.2f} mm")

        # Get diffusion coefficients for anisotropic case
        self.D_L, self.D_T = get_diffusion_params(self.dx, cv_long=0.06, cv_trans=0.02)
        print(f"D_L = {self.D_L:.6f}, D_T = {self.D_T:.6f} cm²/ms")
        print(f"D_L/D_T = {self.D_L/self.D_T:.2f}")

        # Create three operators
        print("\nCreating operators:")

        # 1. FVM Isotropic (for reference shape comparison)
        self.op_iso = DiffusionOperator(
            self.ny, self.nx, self.dx, self.dx,
            D_L=self.D_L, D_T=self.D_L,  # Same D in both directions
            fiber_angle=0.0, device=self.device
        )
        print("  1. FVM Isotropic (convolution) - baseline")

        # 2. FVM Anisotropic (suspected bug)
        self.op_fvm = DiffusionOperator(
            self.ny, self.nx, self.dx, self.dx,
            D_L=self.D_L, D_T=self.D_T,
            fiber_angle=0.0, device=self.device
        )
        print("  2. FVM Anisotropic (flux method) - TESTING")

        # 3. FDM Reference (simple, known correct)
        self.op_fdm = FDMDiffusionReference(
            self.ny, self.nx, self.dx, self.dx,
            self.D_L, self.D_T, device=self.device
        )
        print("  3. FDM Reference (convolution) - VALIDATION")

        # Get stable timestep (use smallest)
        dt1 = self.op_iso.get_stability_limit()
        dt2 = self.op_fvm.get_stability_limit()
        dt3 = self.op_fdm.get_stability_limit()
        self.dt = min(dt1, dt2, dt3) * 0.8
        print(f"\ndt = {self.dt:.5f} ms")

        # Initialize voltage fields
        self.V_iso = torch.zeros(self.ny, self.nx, dtype=torch.float64, device=self.device)
        self.V_fvm = torch.zeros(self.ny, self.nx, dtype=torch.float64, device=self.device)
        self.V_fdm = torch.zeros(self.ny, self.nx, dtype=torch.float64, device=self.device)

        self.time = 0.0
        self.speed = 1.0

        self._setup_display()

    def _setup_display(self):
        """Setup display for 3-panel comparison."""
        self.panel_size = 250
        self.margin = 10
        self.canvas_w = 3 * self.panel_size + 4 * self.margin
        self.canvas_h = self.panel_size + 2 * self.margin + 80

    def reset(self):
        """Reset all fields to same Gaussian initial condition."""
        self.time = 0.0

        cy, cx = self.ny // 2, self.nx // 2
        sigma = 5  # cells

        y_idx = torch.arange(self.ny, device=self.device, dtype=torch.float64)
        x_idx = torch.arange(self.nx, device=self.device, dtype=torch.float64)
        yy, xx = torch.meshgrid(y_idx, x_idx, indexing='ij')

        r2 = (xx - cx)**2 + (yy - cy)**2
        gaussian = 100.0 * torch.exp(-r2 / (2 * sigma**2))

        self.V_iso = gaussian.clone()
        self.V_fvm = gaussian.clone()
        self.V_fdm = gaussian.clone()

        print(f"Reset: Gaussian at center, t = 0")

    def step(self):
        """Take one timestep for all three methods."""
        # Forward Euler: V_new = V + dt * Laplacian(V)
        self.V_iso = self.V_iso + self.dt * self.op_iso.apply(self.V_iso)
        self.V_fvm = self.V_fvm + self.dt * self.op_fvm.apply(self.V_fvm)
        self.V_fdm = self.V_fdm + self.dt * self.op_fdm.apply(self.V_fdm)
        self.time += self.dt

    def voltage_to_image(self, V, size):
        """Convert voltage to RGB image."""
        V_np = V.cpu().numpy()
        V_max = max(V_np.max(), 1.0)
        V_norm = np.clip(V_np / V_max * 255, 0, 255).astype(np.uint8)
        img = cv2.applyColorMap(V_norm, cv2.COLORMAP_JET)
        img = cv2.flip(img, 0)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
        return img

    def measure_radii(self, V):
        """Measure radii at 50% contour."""
        V_np = V.cpu().numpy()
        V_max = V_np.max()
        if V_max < 1.0:
            return 0, 0

        threshold = V_max * 0.5
        cy, cx = self.ny // 2, self.nx // 2

        # Radius in x
        rx = 0
        for i in range(cx, self.nx):
            if V_np[cy, i] > threshold:
                rx = i - cx
            else:
                break

        # Radius in y
        ry = 0
        for j in range(cy, self.ny):
            if V_np[j, cx] > threshold:
                ry = j - cy
            else:
                break

        return rx * self.dx, ry * self.dx

    def compute_error(self, V1, V2):
        """Compute max absolute error between two fields."""
        return (V1 - V2).abs().max().item()

    def run(self):
        """Main loop."""
        cv2.namedWindow('Diffusion Comparison', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Diffusion Comparison', self.canvas_w, self.canvas_h)

        self.reset()

        print("\nControls:")
        print("  SPACE - Reset")
        print("  +/-   - Speed")
        print("  Q     - Quit")
        print()

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
                self.reset()
            elif key == ord('+') or key == ord('='):
                self.speed = min(10.0, self.speed * 1.5)
            elif key == ord('-'):
                self.speed = max(0.1, self.speed / 1.5)

            # Step simulation
            steps = max(1, int(5 * self.speed))
            for _ in range(steps):
                self.step()

            # Create canvas
            canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
            canvas[:] = (40, 40, 40)

            # Render three panels
            panels = [
                (self.V_iso, "1. FVM Isotropic", (200, 200, 200)),
                (self.V_fvm, "2. FVM Anisotropic", (0, 255, 255)),
                (self.V_fdm, "3. FDM Reference", (0, 255, 0)),
            ]

            for i, (V, title, color) in enumerate(panels):
                x0 = self.margin + i * (self.panel_size + self.margin)
                y0 = self.margin

                img = self.voltage_to_image(V, self.panel_size)
                canvas[y0:y0+self.panel_size, x0:x0+self.panel_size] = img

                # Title
                cv2.putText(canvas, title, (x0, y0 + self.panel_size + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                # Measure radii
                rx, ry = self.measure_radii(V)
                ratio = ry / rx if rx > 0 else 0
                info = f"rx={rx:.2f} ry={ry:.2f} r={ratio:.2f}"
                cv2.putText(canvas, info, (x0, y0 + self.panel_size + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

            # Compute errors
            err_fvm_fdm = self.compute_error(self.V_fvm, self.V_fdm)
            err_fvm_iso = self.compute_error(self.V_fvm, self.V_iso)

            # Bottom info
            info_y = self.canvas_h - 25
            cv2.putText(canvas, f"t = {self.time:.1f} ms", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.putText(canvas, f"FVM vs FDM error: {err_fvm_fdm:.4f}", (150, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

            expected_ratio = np.sqrt(self.D_T / self.D_L)
            cv2.putText(canvas, f"Expected ratio: {expected_ratio:.2f}", (400, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            cv2.imshow('Diffusion Comparison', canvas)

        cv2.destroyAllWindows()


def main():
    test = DiffusionComparisonTest(grid_size=150, domain_cm=3.0)
    test.run()


if __name__ == '__main__':
    main()
