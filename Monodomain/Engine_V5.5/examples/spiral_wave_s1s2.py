#!/usr/bin/env python3
"""
Spiral Wave Induction using S1-S2 Cross-Field Stimulation Protocol (V5.4 Engine)

Uses V5.4 engine: FDM discretization + Godunov splitting + Rush-Larsen ionic solver
with TTP06 EPI ionic model.

This script implements the standard S1-S2 protocol for inducing spiral waves:
1. S1: Plane wave stimulus on left edge (travels right)
2. Wait for S1 wave to propagate partially across domain
3. S2: Rectangular stimulus in LOWER-LEFT QUADRANT
4. S2 wavefront meets S1 refractory tail -> unidirectional block -> spiral forms

Uses voltage clamping for S1/S2 (same approach as V5.3 interactive demo).

Controls:
- S: Start S1 (plane wave from left edge) / Reset if already running
- SPACE: Manually apply S2 (lower-left quadrant) - use during vulnerable window!
- R: Reset simulation
- +/-: Adjust simulation speed
- Q/ESC: Quit
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import numpy as np
import cv2
from time import perf_counter

from cardiac_sim.simulation.classical.monodomain import MonodomainSimulation
from cardiac_sim.simulation.classical.discretization_scheme.fdm import FDMDiscretization
from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
from cardiac_sim.tissue_builder.stimulus.protocol import StimulusProtocol
from cardiac_sim.ionic import TTP06Model, CellType


class SpiralWaveSimulation:
    """S1-S2 Cross-Field Stimulation for Spiral Wave Induction (V5.4 Engine)."""

    def __init__(self, domain_cm: float = 16.0, dx: float = 0.04, device: str = 'auto'):
        """
        Initialize simulation.

        Args:
            domain_cm: Physical size of square domain in cm
            dx: Grid spacing in cm (default 0.04 = 400 um, good interactive speed)
                Use 0.02 for V5.3-matching resolution (slower)
            device: 'auto', 'cuda', 'mps', or 'cpu'
        """
        # Device selection
        if device == 'auto':
            if torch.cuda.is_available():
                device_str = 'cuda'
            elif torch.backends.mps.is_available():
                device_str = 'mps'
            else:
                device_str = 'cpu'
        else:
            device_str = device

        # MPS requires float32; CUDA/CPU can use float64
        if device_str == 'mps':
            dtype = torch.float32
        else:
            dtype = torch.float64

        print("=" * 70)
        print("SPIRAL WAVE INDUCTION - S1-S2 Protocol (V5.4 Engine, TTP06)")
        print("=" * 70)
        print(f"Device: {device_str}, dtype: {dtype}")

        # Grid parameters (node-centered: Nx nodes, dx = Lx/(Nx-1))
        Nx = int(domain_cm / dx) + 1
        Ny = Nx
        self.domain_cm = domain_cm
        self.dx = dx
        self.Nx = Nx
        self.Ny = Ny
        self.device_str = device_str
        self.dtype = dtype

        print(f"\nGrid: {Nx} x {Ny} nodes = {Nx * Ny:,} DOFs")
        print(f"Domain: {domain_cm} x {domain_cm} cm, dx = {dx * 10:.1f} mm")

        # Create structured grid
        print("Creating structured grid...")
        self.grid = StructuredGrid.create_rectangle(
            Lx=domain_cm, Ly=domain_cm, Nx=Nx, Ny=Ny,
            device=device_str, dtype=dtype
        )

        # FDM discretization
        # chi=1.0, Cm=1.0 matches V5.3 formulation: dV/dt = D * laplacian(V)
        D = 0.00154  # cm^2/ms, tuned for CV ~60 cm/s
        print(f"Assembling FDM Laplacian (D={D} cm^2/ms)...")
        print("  (This may take a minute for large grids...)")
        t0 = perf_counter()
        self.spatial = FDMDiscretization(
            self.grid, D=D, chi=1.0, Cm=1.0
        )
        t_assembly = perf_counter() - t0
        print(f"  Assembly complete in {t_assembly:.1f}s")

        # CFL stability check for explicit FE diffusion
        dt_max = dx * dx / (4.0 * D)
        self.dt = min(0.02, dt_max * 0.8)
        print(f"  CFL dt_max = {dt_max:.4f} ms, using dt = {self.dt:.4f} ms")

        # Build ionic model with matching dtype (needed for MPS float32)
        ionic_model = TTP06Model(
            cell_type=CellType.EPI,
            device=torch.device(device_str),
            dtype=dtype,
        )

        # Build simulation: Godunov + Rush-Larsen + explicit FE diffusion
        # Empty stimulus protocol (using interactive voltage clamping instead)
        print("Building simulation...")
        self.sim = MonodomainSimulation(
            spatial=self.spatial,
            ionic_model=ionic_model,
            stimulus=StimulusProtocol(),
            dt=self.dt,
            splitting='godunov',
            ionic_solver='rush_larsen',
            diffusion_solver='forward_euler',
            linear_solver='none',
        )

        # Time stepping
        self.steps_per_frame = max(1, int(0.5 / self.dt))
        self.speed_multiplier = 1.0

        # S1-S2 Protocol parameters
        self.cv = 0.06  # cm/ms (estimated CV)
        self.s1_width_cm = 0.2  # S1 strip width (2mm)
        self.s2_width_cm = domain_cm / 2  # Half domain
        self.s2_height_cm = domain_cm / 2

        # APD/ERP estimates for TTP06 EPI
        self.apd_estimate = 280.0  # ms
        self.tissue_erp_factor = 1.1
        self.tissue_erp = self.apd_estimate * self.tissue_erp_factor
        self.time_to_s2_right = self.s2_width_cm / self.cv
        self.s2_window_start = self.tissue_erp
        self.s2_window_end = self.time_to_s2_right + self.tissue_erp
        self.s2_optimal = (self.s2_window_start + self.s2_window_end) / 2

        # Precompute S1/S2 cell counts
        self.s1_cells = max(3, int(self.s1_width_cm / self.dx))
        self.s2_x_cells = max(3, int(self.s2_width_cm / self.dx))
        self.s2_y_cells = max(3, int(self.s2_height_cm / self.dx))

        # Protocol state
        self.s1_applied = False
        self.s2_applied = False
        self.s1_time = None

        # Display setup
        self._setup_display()

        # FPS tracking
        self.frame_times = []
        self.max_frame_times = 30

        print(f"\nTime stepping:")
        print(f"  dt = {self.dt:.4f} ms")
        print(f"  steps/frame = {self.steps_per_frame}")
        print(f"\nS1-S2 Protocol:")
        print(f"  S1: left edge, {self.s1_width_cm * 10:.0f}mm wide ({self.s1_cells} cells)")
        print(f"  S2: lower-left quadrant, {self.s2_width_cm:.1f}x{self.s2_height_cm:.1f} cm")
        print(f"  APD estimate: {self.apd_estimate:.0f} ms")
        print(f"  Vulnerable window: {self.s2_window_start:.0f} - {self.s2_window_end:.0f} ms after S1")
        print(f"  Optimal S2 time: ~{self.s2_optimal:.0f} ms after S1")
        print()
        print("Controls:")
        print("  S       - Start S1 / Reset")
        print("  SPACE   - Apply S2 (in vulnerable window)")
        print("  R       - Reset simulation")
        print("  +/-     - Adjust speed")
        print("  Q/ESC   - Quit")
        print()

    def _setup_display(self):
        """Setup display parameters and static canvas with colorbar."""
        target_size = 600
        display_dim = min(self.Nx, self.Ny)
        self.scale = max(1, target_size // display_dim)
        self.img_w = display_dim * self.scale
        self.img_h = display_dim * self.scale

        # Margins
        self.margin_left = 60
        self.margin_bottom = 50
        self.margin_right = 80
        self.margin_top = 50

        # Canvas
        self.canvas_h = self.img_h + self.margin_top + self.margin_bottom
        self.canvas_w = self.img_w + self.margin_left + self.margin_right

        # Static canvas with colorbar
        self.static_canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
        self.static_canvas[:] = (30, 30, 30)

        # Colorbar
        cbar_x = self.margin_left + self.img_w + 15
        cbar_w = 20
        cbar_h = self.img_h
        cbar_y = self.margin_top

        gradient = np.linspace(255, 0, cbar_h).astype(np.uint8).reshape(-1, 1)
        colorbar = cv2.applyColorMap(np.tile(gradient, (1, cbar_w)), cv2.COLORMAP_JET)
        self.static_canvas[cbar_y:cbar_y + cbar_h, cbar_x:cbar_x + cbar_w] = colorbar

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.static_canvas, '+40', (cbar_x + cbar_w + 5, cbar_y + 15),
                    font, 0.4, (255, 255, 255), 1)
        cv2.putText(self.static_canvas, '-90', (cbar_x + cbar_w + 5, cbar_y + cbar_h),
                    font, 0.4, (255, 255, 255), 1)
        cv2.putText(self.static_canvas, 'mV', (cbar_x + 2, cbar_y - 8),
                    font, 0.4, (255, 255, 255), 1)

        # Axis labels
        for i in range(int(self.domain_cm) + 1):
            x_pos = self.margin_left + int(i / self.domain_cm * self.img_w)
            cv2.putText(self.static_canvas, f'{i}',
                        (x_pos - 5, self.margin_top + self.img_h + 20),
                        font, 0.4, (255, 255, 255), 1)
        cv2.putText(self.static_canvas, 'x (cm)',
                    (self.margin_left + self.img_w // 2 - 20, self.canvas_h - 5),
                    font, 0.45, (255, 255, 255), 1)

    def apply_s1_stimulus(self):
        """Apply S1 stimulus: voltage clamp left edge to +20 mV."""
        V = self.sim.get_voltage()
        V_grid = self.grid.flat_to_grid(V)
        # Left edge: small x values (first axis in 'ij' indexing)
        V_grid[:self.s1_cells, :] = 20.0
        self.sim.set_voltage(self.grid.grid_to_flat(V_grid))
        self.s1_applied = True
        self.s1_time = self.sim.state.t
        print(f"  S1 applied at t = {self.sim.state.t:.1f} ms")

    def apply_s2_stimulus(self):
        """Apply S2 stimulus: voltage clamp lower-left quadrant to +20 mV."""
        V = self.sim.get_voltage()
        V_grid = self.grid.flat_to_grid(V)
        # Lower-left quadrant: small x AND small y
        V_grid[:self.s2_x_cells, :self.s2_y_cells] = 20.0
        self.sim.set_voltage(self.grid.grid_to_flat(V_grid))
        self.s2_applied = True
        elapsed = self.sim.state.t - self.s1_time if self.s1_time is not None else 0
        print(f"  S2 applied at t = {self.sim.state.t:.1f} ms (interval: {elapsed:.0f} ms)")

    def start_s1(self):
        """Start/restart with S1 stimulus."""
        self.sim.reset()
        self.s1_applied = False
        self.s2_applied = False
        self.s1_time = None
        print(f"\n--- Starting S1 ---")
        self.apply_s1_stimulus()

    def reset(self):
        """Reset simulation to resting state."""
        self.sim.reset()
        self.s1_applied = False
        self.s2_applied = False
        self.s1_time = None
        print("Simulation reset")

    def voltage_to_image(self, V_grid: np.ndarray) -> np.ndarray:
        """
        Convert voltage grid to display image.

        V_grid has shape (Nx, Ny) with 'ij' indexing (first axis = x).
        For cv2 display we need (height, width) = (rows, cols) with x horizontal.
        """
        # Transpose: (Nx, Ny) -> (Ny, Nx) so rows=y, cols=x
        V_display = V_grid.T
        # Flip vertically so y=0 is at bottom of image
        V_display = np.flipud(V_display)

        # Normalize to 0-255: -90 mV -> 0, +40 mV -> 255
        V_norm = np.clip((V_display + 90) / 130 * 255, 0, 255).astype(np.uint8)
        img = cv2.applyColorMap(V_norm, cv2.COLORMAP_JET)

        if self.scale > 1:
            img = cv2.resize(img, (self.img_w, self.img_h),
                             interpolation=cv2.INTER_NEAREST)

        canvas = self.static_canvas.copy()
        canvas[self.margin_top:self.margin_top + self.img_h,
               self.margin_left:self.margin_left + self.img_w] = img

        return canvas

    def run(self):
        """Main interactive simulation loop."""
        cv2.namedWindow('Spiral Wave - V5.4 TTP06', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Spiral Wave - V5.4 TTP06', 800, 800)

        print("Starting... Press 'S' to apply S1, then SPACE for S2")

        while True:
            frame_start = perf_counter()

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                self.start_s1()
            elif key == ord(' '):
                if self.s1_applied and not self.s2_applied:
                    print(f"\n--- Applying S2 ---")
                    self.apply_s2_stimulus()
                elif not self.s1_applied:
                    print("  (Apply S1 first by pressing S)")
            elif key == ord('r'):
                self.reset()
            elif key == ord('+') or key == ord('='):
                self.speed_multiplier = min(4.0, self.speed_multiplier * 1.5)
                print(f"Speed: {self.speed_multiplier:.1f}x")
            elif key == ord('-'):
                self.speed_multiplier = max(0.25, self.speed_multiplier / 1.5)
                print(f"Speed: {self.speed_multiplier:.1f}x")

            # Simulation steps
            actual_steps = max(1, int(self.steps_per_frame * self.speed_multiplier))
            for _ in range(actual_steps):
                self.sim.step()

            # Get voltage and create image
            V = self.sim.get_voltage()
            V_grid = self.grid.flat_to_grid(V).cpu().numpy()
            img = self.voltage_to_image(V_grid)

            # FPS
            frame_time = perf_counter() - frame_start
            self.frame_times.append(frame_time)
            if len(self.frame_times) > self.max_frame_times:
                self.frame_times.pop(0)
            fps = len(self.frame_times) / sum(self.frame_times) if self.frame_times else 0

            # Title
            title = f"Spiral Wave - V5.4 TTP06 ({self.domain_cm:.0f}x{self.domain_cm:.0f} cm)"
            cv2.putText(img, title, (60, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1)

            # Status bar
            V_min, V_max = V_grid.min(), V_grid.max()
            status = f"t={self.sim.state.t:.1f}ms | {fps:.1f}FPS | V:[{V_min:.0f},{V_max:.0f}]mV"
            cv2.putText(img, status, (60, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (200, 200, 200), 1)

            # Protocol status overlay
            if self.s2_applied:
                cv2.putText(img, "SPIRAL FORMING", (img.shape[1] - 200, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            elif self.s1_applied:
                elapsed = self.sim.state.t - self.s1_time
                in_window = self.s2_window_start <= elapsed <= self.s2_window_end

                if in_window:
                    pstatus = f"{elapsed:.0f}ms | PRESS SPACE!"
                    color = (0, 255, 0)
                elif elapsed < self.s2_window_start:
                    pstatus = f"{elapsed:.0f}ms | wait..."
                    color = (0, 255, 255)
                else:
                    pstatus = f"{elapsed:.0f}ms | WINDOW PASSED"
                    color = (0, 0, 255)

                cv2.putText(img, pstatus, (img.shape[1] - 220, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                cv2.putText(img, "Press S to start", (img.shape[1] - 180, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)

            cv2.imshow('Spiral Wave - V5.4 TTP06', img)

        cv2.destroyAllWindows()
        print("Simulation ended.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Spiral wave S1-S2 protocol (V5.4 Engine)')
    parser.add_argument('--domain', type=float, default=16.0,
                        help='Domain size in cm (default: 16.0)')
    parser.add_argument('--dx', type=float, default=0.04,
                        help='Grid spacing in cm (default: 0.04, use 0.02 for full resolution)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cuda, or cpu (default: auto)')
    args = parser.parse_args()

    sim = SpiralWaveSimulation(domain_cm=args.domain, dx=args.dx, device=args.device)
    sim.run()


if __name__ == '__main__':
    main()
