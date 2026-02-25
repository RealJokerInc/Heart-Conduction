"""
Spiral Wave Induction using S1-S2 Cross-Field Stimulation Protocol

This script implements the standard S1-S2 protocol for inducing spiral waves:
1. S1: Plane wave stimulus on left edge (travels right)
2. Wait for S1 wave to propagate partially across domain
3. S2: Rectangular stimulus in LOWER-LEFT QUADRANT (not a strip!)
4. S2 wavefront meets S1 refractory tail -> unidirectional block -> spiral forms

The key insight (from Bhattacharya et al., Biophysical Journal 1998):
"A second electrode (E2) was placed on the lower left quadrant.
 The size of E2 was 1 cm × 0.8 cm."

This creates asymmetric wave propagation where:
- The wave can propagate upward (tissue is recovered)
- The wave is blocked rightward (tissue still refractory from S1)
- This asymmetry creates the spiral rotation

Uses the MeshBuilder system with pre-tuned diffusion coefficients:
- Anisotropic mode: CV_long = 0.6 m/s, CV_trans = 0.3 m/s
- Pre-tuned D_L = 0.002161, D_T = 0.000819 cm²/ms (calibrated via 1D cable simulations)
- Default domain: 16x16 cm at dx = 0.02 cm (800x800 cells)

Controls:
- S: Start S1 (plane wave from left edge) / Reset if already running
- SPACE: Manually apply S2 (lower-left quadrant) - use during vulnerable window!
- R: Reset simulation
- +/-: Adjust simulation speed
- Q/ESC: Quit

References:
- Bhattacharya et al. (1998) - Cross-field stimulation, Biophys J.
- Panfilov & Holden (1990) - Cross-field stimulation protocol
- Winfree (1989) - Pinwheel experiment
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import cv2
from time import perf_counter

from ionic import CellType
from tissue import MeshBuilder, compute_D_min, D_L_DEFAULT


class SpiralWaveSimulation:
    """S1-S2 Cross-Field Stimulation for Spiral Wave Induction."""

    # Default APD for spiral wave simulations (can be overridden)
    DEFAULT_APD_MS = 250.0

    def __init__(self, domain_cm=16.0, apd_ms=None):
        """
        Initialize simulation.

        Args:
            domain_cm: Physical size of domain in cm (default: 16x16 cm)
            apd_ms: Action potential duration in ms. Used for:
                    - D validation (mesh-dependent minimum)
                    - S1-S2 timing window calculation
                    Default: 250ms
        """
        self.domain_cm = domain_cm
        self.apd_ms = apd_ms if apd_ms is not None else self.DEFAULT_APD_MS
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print("=" * 70)
        print("SPIRAL WAVE INDUCTION - S1-S2 Cross-Field Protocol")
        print("=" * 70)

        # Create mesh using MeshBuilder with default anisotropic diffusion
        # Default: 16x16 cm, dx=0.02 cm, CV_long=0.6 m/s, CV_trans=0.3 m/s
        mesh = (MeshBuilder.create_default(anisotropic=True)
                .set_domain(domain_cm, domain_cm)
                .set_apd(self.apd_ms))

        # Print mesh configuration
        mesh.print_summary()

        # Create simulation from mesh
        self.sim = mesh.create_simulation(
            celltype=CellType.ENDO,
            device=self.device
        )

        # Get actual grid dimensions from mesh
        cfg = mesh.get_config()
        self.ny, self.nx = cfg.ny, cfg.nx
        self.dx = cfg.dx

        # Display D validation info
        d_min = compute_D_min(self.dx, self.apd_ms)
        print(f"\nD Validation:")
        print(f"  D_L = {cfg.D_L:.6f} cm²/ms (CV = 0.6 m/s)")
        print(f"  D_T = {cfg.D_T:.6f} cm²/ms (CV = 0.3 m/s)")
        print(f"  D_min = {d_min:.6f} cm²/ms")
        print(f"  D_L/D_min = {cfg.D_L/d_min:.2f}x")

        # Time stepping
        dt_stability = self.sim.diffusion.get_stability_limit()
        self.dt = dt_stability * 0.8
        self.steps_per_frame = max(1, int(0.5 / self.dt))
        self.speed_multiplier = 1.0

        print(f"dt = {self.dt:.5f} ms")
        print(f"CV = 0.6 m/s -> wave crosses {domain_cm:.1f}cm in {domain_cm/0.06:.0f}ms")
        print()

        # S1-S2 Protocol parameters
        self.cv = 0.06  # cm/ms
        self.s1_width_cm = 0.3  # Width of S1 stimulus region (3mm)

        # S2 quadrant size: 3cm x 2cm (width x height)
        self.s2_width_cm = 8.0   # 2 cm in x direction
        self.s2_height_cm = 8.0  # 2 cm in y direction

        # Calculate optimal S1-S2 timing window
        #
        # CRITICAL: Tissue ERP (propagation) >> APD due to source-sink mismatch!
        #
        # Measured relationship (from tests/debug_tissue_erp_spiral.py):
        # - Single-cell APD90 = 300ms, Single-cell ERP = 10ms
        # - Tissue ERP (S2 capture only) = 280ms (0.93x APD)
        # - Tissue ERP (S2 propagation) = 350ms (1.17x APD)
        #
        # For spiral initiation, S2 must PROPAGATE (not just capture).
        # The 2cm × 2cm S2 electrode has tissue_erp_factor ≈ 1.17
        self.tissue_erp_factor = 1.17  # Measured factor for 2x2cm S2 electrode

        self.apd_estimate = self.apd_ms
        self.tissue_erp = self.apd_estimate * self.tissue_erp_factor

        # Time for wave to reach right edge of S2
        self.time_to_s2_right = self.s2_width_cm / self.cv  # ~33ms for 2cm

        # Optimal window for ASYMMETRIC spiral initiation:
        # - S2 left edge (x=0) was activated at t=0 by S1
        # - S2 right edge (x=s2_width) was activated at t=time_to_s2_right by S1
        #
        # For spiral: S2 must propagate UPWARD (left recovered) but be BLOCKED
        # rightward (right still refractory). Both edges need tissue_erp to propagate.
        #
        # - Start: tissue_erp (left edge can now propagate)
        # - End: time_to_s2_right + tissue_erp (right edge also propagates - no more asymmetry)
        self.s2_window_start = self.tissue_erp  # Left edge can propagate
        self.s2_window_end = self.time_to_s2_right + self.tissue_erp  # Right edge can propagate
        self.s2_optimal = (self.s2_window_start + self.s2_window_end) / 2  # Middle of window

        # Protocol state
        self.s1_applied = False
        self.s2_applied = False
        self.s1_time = None  # Time when S1 was applied

        # Display setup
        self._setup_display()

        # FPS tracking
        self.frame_times = []
        self.max_frame_times = 30

        print("S1-S2 Protocol Parameters:")
        print(f"  S1 region: left edge, {self.s1_width_cm*10:.0f}mm wide")
        print(f"  S2 region: lower-left quadrant, {self.s2_width_cm:.0f}cm x {self.s2_height_cm:.0f}cm")
        print()
        print("S2 Timing Window (calculated):")
        print(f"  Wave reaches S2 right edge (x={self.s2_width_cm:.0f}cm) at t = {self.time_to_s2_right:.0f}ms")
        print(f"  APD = {self.apd_estimate:.0f}ms")
        print(f"  Tissue ERP = {self.tissue_erp:.0f}ms (APD × {self.tissue_erp_factor:.2f}, source-sink factor)")
        print(f"  Left edge (x=0) recovered after tissue ERP = {self.tissue_erp:.0f}ms")
        print(f"  Right edge refractory until t = {self.time_to_s2_right:.0f} + {self.apd_estimate:.0f} = {self.s2_window_end:.0f}ms")
        print(f"  ==> VULNERABLE WINDOW: {self.s2_window_start:.0f}ms - {self.s2_window_end:.0f}ms after S1")
        print(f"  ==> SUGGESTED S2 TIME: ~{self.s2_optimal:.0f}ms after S1")
        print()
        print("Controls:")
        print("  S       - Start S1 / Reset (plane wave from left)")
        print("  SPACE   - Apply S2 manually (lower-left quadrant)")
        print("  R       - Reset simulation")
        print("  +/-     - Adjust speed")
        print("  Q/ESC   - Quit")
        print()

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
        cv2.putText(self.static_canvas, '0', (cbar_x + cbar_w + 5, cbar_y + cbar_h // 3 + 5),
                   font, 0.4, (255, 255, 255), 1)
        cv2.putText(self.static_canvas, '-40', (cbar_x + cbar_w + 5, cbar_y + 2 * cbar_h // 3 + 5),
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
        cv2.putText(self.static_canvas, 'x (cm)',
                   (self.margin_left + self.img_w // 2 - 20, self.canvas_h - 5),
                   font, 0.45, (255, 255, 255), 1)

        for i in range(int(self.domain_cm) + 1):
            y_pos = self.margin_top + self.img_h - int(i / self.domain_cm * self.img_h)
            cv2.line(self.static_canvas, (self.margin_left - 5, y_pos),
                    (self.margin_left, y_pos), (200, 200, 200), 1)
            cv2.putText(self.static_canvas, f'{i}', (self.margin_left - 25, y_pos + 5),
                       font, 0.4, (255, 255, 255), 1)

        cv2.putText(self.static_canvas, 'y', (5, self.margin_top + self.img_h // 2 - 10),
                   font, 0.45, (255, 255, 255), 1)
        cv2.putText(self.static_canvas, '(cm)', (5, self.margin_top + self.img_h // 2 + 10),
                   font, 0.35, (255, 255, 255), 1)

    def apply_s1_stimulus(self):
        """Apply S1 stimulus on left edge (creates rightward-propagating plane wave)."""
        V = self.sim.get_voltage()
        s1_cells = max(3, int(self.s1_width_cm / self.dx))
        V[:, :s1_cells] = 20.0  # Strong depolarization
        self.sim.set_voltage(V)
        self.s1_applied = True
        self.s1_time = self.sim.time
        print(f"  S1 applied at t = {self.sim.time:.1f} ms (left edge, {s1_cells} cells wide)")
        print(f"     Press SPACE when wave is ~halfway across to apply S2!")

    def apply_s2_stimulus(self):
        """Apply S2 stimulus in lower-left quadrant (rectangular box).

        This is the correct cross-field geometry from Bhattacharya et al. (1998):
        "A second electrode (E2) was placed on the lower left quadrant."

        The wave can only propagate upward (recovered tissue) and is blocked
        rightward (refractory tail of S1), creating the spiral rotation.
        """
        V = self.sim.get_voltage()

        # Lower-left quadrant: from (0,0) to (s2_width, s2_height)
        s2_x_cells = max(3, int(self.s2_width_cm / self.dx))
        s2_y_cells = max(3, int(self.s2_height_cm / self.dx))

        # Apply strong depolarization to lower-left quadrant
        # Note: In numpy array, row 0 is bottom when displayed with origin='lower'
        V[:s2_y_cells, :s2_x_cells] = 20.0

        self.sim.set_voltage(V)
        self.s2_applied = True

        elapsed = self.sim.time - self.s1_time if self.s1_time else 0
        print(f"  S2 applied at t = {self.sim.time:.1f} ms (S1-S2 interval: {elapsed:.0f} ms)")
        print(f"     Lower-left quadrant: {s2_x_cells}x{s2_y_cells} cells = {self.s2_width_cm:.1f}x{self.s2_height_cm:.1f} cm")

    def start_s1(self):
        """Start/Reset with S1 stimulus."""
        # Reset simulation first
        self.sim.reset()
        self.s1_applied = False
        self.s2_applied = False
        self.s1_time = None
        print(f"\n--- Starting S1 ---")
        # Apply S1 immediately
        self.apply_s1_stimulus()

    def reset(self):
        """Reset simulation to initial conditions."""
        self.sim.reset()
        self.s1_applied = False
        self.s2_applied = False
        self.s1_time = None
        print("Simulation reset")

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

        return canvas

    def run(self):
        """Main simulation loop."""
        cv2.namedWindow('Spiral Wave - S1-S2 Protocol', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Spiral Wave - S1-S2 Protocol', 900, 900)

        print("Starting simulation... Press 'S' to apply S1, then SPACE for S2")

        while True:
            frame_start = perf_counter()

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                self.start_s1()  # S key starts S1 / resets
            elif key == ord(' '):  # SPACE for S2
                if self.s1_applied and not self.s2_applied:
                    print(f"\n--- Applying S2 ---")
                    self.apply_s2_stimulus()
                elif not self.s1_applied:
                    print("  (Apply S1 first by pressing S)")
                else:
                    print("  (S2 already applied)")
            elif key == ord('r'):
                self.reset()
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
            title = f"Spiral Wave Induction ({self.domain_cm:.0f}x{self.domain_cm:.0f} cm)"
            cv2.putText(img, title, (60, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 255, 255), 1)

            # Status
            V_min, V_max = V.min(), V.max()
            status = f"t={self.sim.time:.1f}ms | {fps:.1f}FPS | V:[{V_min:.0f},{V_max:.0f}]mV"
            cv2.putText(img, status, (60, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.45, (200, 200, 200), 1)

            # Protocol status
            if self.s2_applied:
                cv2.putText(img, "SPIRAL FORMING", (img.shape[1] - 200, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            elif self.s1_applied:
                elapsed = self.sim.time - self.s1_time
                # Check if in optimal window
                in_window = self.s2_window_start <= elapsed <= self.s2_window_end
                if in_window:
                    # GREEN - optimal window, press SPACE now!
                    pstatus = f"{elapsed:.0f}ms | PRESS SPACE NOW!"
                    color = (0, 255, 0)  # Green
                    thickness = 2
                elif elapsed < self.s2_window_start:
                    # Yellow - waiting for left edge to recover
                    time_to_window = self.s2_window_start - elapsed
                    pstatus = f"{elapsed:.0f}ms | wait {time_to_window:.0f}ms"
                    color = (0, 255, 255)  # Yellow
                    thickness = 1
                else:
                    # Red - window passed
                    pstatus = f"{elapsed:.0f}ms | WINDOW PASSED"
                    color = (0, 0, 255)  # Red
                    thickness = 1
                cv2.putText(img, pstatus, (img.shape[1] - 280, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
            else:
                cv2.putText(img, "Press S to start S1", (img.shape[1] - 200, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)

            cv2.imshow('Spiral Wave - S1-S2 Protocol', img)

        cv2.destroyAllWindows()
        print("Simulation ended.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Spiral wave induction via S1-S2 protocol')
    parser.add_argument('--domain', type=float, default=16.0,
                       help='Domain size in cm (default: 16.0)')
    parser.add_argument('--apd', type=float, default=250.0,
                       help='Action potential duration in ms (default: 250)')
    args = parser.parse_args()

    # Create simulation with default MeshBuilder (16x16 cm, anisotropic)
    sim = SpiralWaveSimulation(
        domain_cm=args.domain,
        apd_ms=args.apd
    )

    sim.run()


if __name__ == '__main__':
    main()
