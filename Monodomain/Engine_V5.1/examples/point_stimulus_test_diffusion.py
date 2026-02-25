"""
Point Stimulus Test - Pure Diffusion Only

Tests the diffusion operator in isolation (no ionic currents).
A Gaussian or step stimulus is applied and diffuses according to the
anisotropic diffusion equation:

    dV/dt = D_L * d²V/dx² + D_T * d²V/dy²

Expected behavior:
- Isotropic (D_L = D_T): Circular spreading
- Anisotropic (D_L > D_T): Elliptical spreading with ratio sqrt(D_T/D_L)

Controls:
- SPACE: Apply point stimulus at center
- R: Reset to flat state
- A: Toggle anisotropic/isotropic
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

from tissue.diffusion import DiffusionOperator, get_diffusion_params, compute_D_min


class PureDiffusionTest:
    """Test diffusion operator with point stimulus - no ionic model."""

    def __init__(self, grid_size=300, domain_cm=6.0, isotropic=True):
        """
        Initialize pure diffusion test.

        Args:
            grid_size: Number of cells in each dimension
            domain_cm: Physical size of domain in cm
            isotropic: If True, use same D in x and y
        """
        self.ny, self.nx = grid_size, grid_size
        self.domain_cm = domain_cm
        self.dx = domain_cm / grid_size
        self.isotropic = isotropic

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float64

        print("=" * 70)
        print("PURE DIFFUSION TEST - No Ionic Model")
        print("=" * 70)
        print(f"Grid: {self.ny} x {self.nx} = {self.ny * self.nx:,} cells")
        print(f"Domain: {domain_cm:.1f} x {domain_cm:.1f} cm")
        print(f"Resolution: {self.dx*10:.2f} mm/cell")
        print(f"Device: {self.device}")

        self._create_diffusion_operator()
        self._initialize_voltage()

        # Point stimulus parameters
        self.stim_radius_cm = 0.2  # 2mm radius
        self.stim_radius_cells = max(2, int(self.stim_radius_cm / self.dx))
        self.stim_amplitude = 100.0  # mV (arbitrary for pure diffusion)

        # Display setup
        self._setup_display()

        # Timing
        self.time = 0.0
        self.stim_time = None
        self.frame_times = []
        self.max_frame_times = 30
        self.speed_multiplier = 1.0

        print()
        print("Controls:")
        print("  SPACE   - Apply point stimulus at center")
        print("  R       - Reset to flat state")
        print("  A       - Toggle anisotropic/isotropic")
        print("  +/-     - Adjust speed")
        print("  Q/ESC   - Quit")
        print()

    def _create_diffusion_operator(self):
        """Create diffusion operator with current settings."""
        # Target CV values (used to compute D)
        cv_long = 0.06  # cm/ms

        if self.isotropic:
            cv_trans = 0.06
            print(f"Diffusion: ISOTROPIC (D_L = D_T)")
        else:
            cv_trans = 0.02  # 3:1 CV ratio -> 9:1 D ratio
            print(f"Diffusion: ANISOTROPIC (CV ratio 3:1 -> D ratio 9:1)")

        # Get D values from CV
        self.D_L, self.D_T = get_diffusion_params(
            self.dx, cv_long=cv_long, cv_trans=cv_trans,
            validate=False, warn=False
        )

        # Show D_min for reference (using normal APD since no ionic model)
        D_min = compute_D_min(self.dx, apd_ms=280)
        print(f"D_min (reference, APD=280ms): {D_min:.6f} cm²/ms")
        print(f"D_L = {self.D_L:.6f} cm²/ms")
        print(f"D_T = {self.D_T:.6f} cm²/ms")
        print(f"D_L/D_T ratio = {self.D_L/self.D_T:.1f}:1")
        print(f"Expected shape ratio (ry/rx) = sqrt(D_T/D_L) = {np.sqrt(self.D_T/self.D_L):.3f}")

        # Create operator
        self.diffusion = DiffusionOperator(
            ny=self.ny, nx=self.nx,
            dx=self.dx, dy=self.dx,
            D_L=self.D_L, D_T=self.D_T,
            fiber_angle=0.0,
            device=self.device
        )

        # Time stepping
        dt_stability = self.diffusion.get_stability_limit()
        self.dt = dt_stability * 0.8
        self.steps_per_frame = max(1, int(0.5 / self.dt))

        print(f"dt = {self.dt:.5f} ms, stability limit = {dt_stability:.5f} ms")

    def _initialize_voltage(self):
        """Initialize voltage field to zero."""
        self.V = torch.zeros(
            (self.ny, self.nx),
            dtype=self.dtype,
            device=self.device
        )

    def _setup_display(self):
        """Set up display elements."""
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

        # Static canvas background
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
        cv2.putText(self.static_canvas, '100', (cbar_x + cbar_w + 5, cbar_y + 15),
                   font, 0.4, (255, 255, 255), 1)
        cv2.putText(self.static_canvas, '0', (cbar_x + cbar_w + 5, cbar_y + cbar_h),
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
        """Apply point stimulus at center."""
        cy, cx = self.ny // 2, self.nx // 2
        r = self.stim_radius_cells

        # Circular stimulus
        for dy in range(-r, r + 1):
            for ddx in range(-r, r + 1):
                if dy*dy + ddx*ddx <= r*r:
                    y, x = cy + dy, cx + ddx
                    if 0 <= y < self.ny and 0 <= x < self.nx:
                        self.V[y, x] = self.stim_amplitude

        self.stim_time = self.time
        print(f"Point stimulus applied at center (t = {self.time:.1f} ms)")

    def reset(self):
        """Reset to flat state."""
        self.V.zero_()
        self.time = 0.0
        self.stim_time = None
        print("Reset to flat state")

    def toggle_anisotropy(self):
        """Toggle between isotropic and anisotropic."""
        self.isotropic = not self.isotropic
        self._create_diffusion_operator()
        self.reset()

    def step(self, dt):
        """Advance one time step using forward Euler."""
        # Pure diffusion: dV/dt = D·∇²V
        dV_dt = self.diffusion.apply(self.V)
        self.V = self.V + dt * dV_dt
        self.time += dt

    def voltage_to_image(self, V):
        """Convert voltage field to display image."""
        # Normalize to 0-255 (assuming V in [0, 100])
        V_norm = np.clip(V / self.stim_amplitude * 255, 0, 255).astype(np.uint8)
        img = cv2.applyColorMap(V_norm, cv2.COLORMAP_JET)
        img = cv2.flip(img, 0)

        if self.scale > 1:
            img = cv2.resize(img, (self.img_w, self.img_h),
                           interpolation=cv2.INTER_NEAREST)

        canvas = self.static_canvas.copy()
        canvas[self.margin_top:self.margin_top+self.img_h,
               self.margin_left:self.margin_left+self.img_w] = img

        # Crosshairs at center
        cx = self.margin_left + self.img_w // 2
        cy = self.margin_top + self.img_h // 2
        cv2.line(canvas, (cx - 10, cy), (cx + 10, cy), (255, 255, 255), 1)
        cv2.line(canvas, (cx, cy - 10), (cx, cy + 10), (255, 255, 255), 1)

        return canvas

    def measure_shape(self, V, threshold_frac=0.5):
        """Measure wavefront shape at given threshold."""
        V_max = V.max()
        if V_max < 1.0:
            return 0, 0, 0

        threshold = V_max * threshold_frac
        cy, cx = self.ny // 2, self.nx // 2

        # Radius in +x direction
        rx = 0
        for i in range(cx, self.nx):
            if V[cy, i] > threshold:
                rx = i - cx
            else:
                break

        # Radius in +y direction
        ry = 0
        for j in range(cy, self.ny):
            if V[j, cx] > threshold:
                ry = j - cy
            else:
                break

        ratio = ry / rx if rx > 0 else 0
        return rx * self.dx, ry * self.dx, ratio

    def run(self):
        """Main simulation loop."""
        cv2.namedWindow('Pure Diffusion Test', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Pure Diffusion Test', 900, 900)

        print("Press SPACE to apply point stimulus")

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
                self.step(self.dt)

            # Get voltage and create image
            V_np = self.V.cpu().numpy()
            img = self.voltage_to_image(V_np)

            # FPS calculation
            frame_time = perf_counter() - frame_start
            self.frame_times.append(frame_time)
            if len(self.frame_times) > self.max_frame_times:
                self.frame_times.pop(0)
            fps = len(self.frame_times) / sum(self.frame_times)

            # Title
            mode = "ISOTROPIC" if self.isotropic else f"ANISOTROPIC (D ratio {self.D_L/self.D_T:.1f}:1)"
            title = f"Pure Diffusion - {mode}"
            cv2.putText(img, title, (60, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 255, 255), 1)

            # Status
            V_max = V_np.max()
            status = f"t={self.time:.1f}ms | {fps:.1f}FPS | V_max={V_max:.1f}"
            cv2.putText(img, status, (60, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.45, (200, 200, 200), 1)

            # Shape measurement
            if self.stim_time is not None and V_max > 1.0:
                rx_cm, ry_cm, ratio = self.measure_shape(V_np)
                expected_ratio = np.sqrt(self.D_T / self.D_L)

                if rx_cm > 0:
                    info = f"r_x={rx_cm:.2f}cm r_y={ry_cm:.2f}cm ratio={ratio:.3f}"
                    cv2.putText(img, info, (img.shape[1] - 350, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

                    expected_info = f"Expected ratio: {expected_ratio:.3f}"
                    cv2.putText(img, expected_info, (img.shape[1] - 350, 55),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

                    # Error from expected
                    if ratio > 0:
                        error_pct = abs(ratio - expected_ratio) / expected_ratio * 100
                        error_info = f"Shape error: {error_pct:.1f}%"
                        color = (0, 255, 0) if error_pct < 5 else (0, 165, 255) if error_pct < 15 else (0, 0, 255)
                        cv2.putText(img, error_info, (img.shape[1] - 350, 80),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            cv2.imshow('Pure Diffusion Test', img)

        cv2.destroyAllWindows()
        print("Test ended.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Pure diffusion point stimulus test')
    parser.add_argument('--grid', type=int, default=300,
                       help='Grid size (default: 300)')
    parser.add_argument('--domain', type=float, default=6.0,
                       help='Domain size in cm (default: 6.0)')
    parser.add_argument('--anisotropic', action='store_true',
                       help='Start with anisotropic diffusion')
    args = parser.parse_args()

    test = PureDiffusionTest(
        grid_size=args.grid,
        domain_cm=args.domain,
        isotropic=not args.anisotropic
    )
    test.run()


if __name__ == '__main__':
    main()
