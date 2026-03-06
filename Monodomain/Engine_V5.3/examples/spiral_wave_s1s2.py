#!/usr/bin/env python3
"""
Spiral Wave Induction using S1-S2 Cross-Field Stimulation Protocol

TTP06 model with GPU acceleration and LUT optimization.

This script implements the standard S1-S2 protocol for inducing spiral waves:
1. S1: Plane wave stimulus on left edge (travels right)
2. Wait for S1 wave to propagate partially across domain
3. S2: Rectangular stimulus in LOWER-LEFT QUADRANT
4. S2 wavefront meets S1 refractory tail -> unidirectional block -> spiral forms

Uses simple finite difference method (FDM) for diffusion with operator splitting.

Controls:
- S: Start S1 (plane wave from left edge) / Reset if already running
- SPACE: Manually apply S2 (lower-left quadrant) - use during vulnerable window!
- R: Reset simulation
- +/-: Adjust simulation speed
- Q/ESC: Quit
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import cv2
from time import perf_counter
from dataclasses import dataclass
from typing import Optional

from ionic import TTP06Model, CellType
from utils import Backend


@dataclass
class TissueConfig:
    """Configuration for 2D tissue simulation."""
    # Domain
    Lx: float = 16.0          # Domain width (cm)
    Ly: float = 16.0          # Domain height (cm)
    dx: float = 0.02         # Grid spacing (cm) - 200 µm

    # Diffusion
    D: float = 0.00154       # Diffusion coefficient (cm²/ms) for CV ~60 cm/s

    # Time stepping
    dt: float = 0.02         # Time step (ms)

    @property
    def nx(self) -> int:
        return int(self.Lx / self.dx)

    @property
    def ny(self) -> int:
        return int(self.Ly / self.dx)


class FDMDiffusion:
    """
    Finite Difference Method diffusion operator (5-point stencil).

    Solves: dV/dt = D * nabla²V

    Uses explicit Euler with Neumann (no-flux) boundary conditions.
    """

    def __init__(self, ny: int, nx: int, dx: float, D: float, device: torch.device):
        self.ny = ny
        self.nx = nx
        self.dx = dx
        self.D = D
        self.device = device

        # Precompute coefficient
        self.alpha = D / (dx * dx)

        # Stability limit for explicit scheme: dt < dx²/(4D)
        self.dt_max = dx * dx / (4 * D)

    def apply(self, V: torch.Tensor) -> torch.Tensor:
        """
        Apply Laplacian operator.

        Parameters
        ----------
        V : torch.Tensor
            Voltage field, shape (ny, nx)

        Returns
        -------
        torch.Tensor
            D * nabla²V, shape (ny, nx)
        """
        # Pad with replicate (Neumann BC)
        V_pad = torch.nn.functional.pad(
            V.unsqueeze(0).unsqueeze(0),
            (1, 1, 1, 1),
            mode='replicate'
        ).squeeze()

        # 5-point stencil Laplacian
        laplacian = (
            V_pad[1:-1, 2:] +   # right
            V_pad[1:-1, :-2] +  # left
            V_pad[2:, 1:-1] +   # top
            V_pad[:-2, 1:-1] -  # bottom
            4 * V_pad[1:-1, 1:-1]
        )

        return self.alpha * laplacian

    def step(self, V: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Advance diffusion by one time step (explicit Euler).
        """
        return V + dt * self.apply(V)


class TTP06TissueSimulation:
    """
    2D Tissue simulation with TTP06 model.

    Uses operator splitting:
    1. Ionic step: TTP06 model (Rush-Larsen)
    2. Diffusion step: FDM (explicit Euler)
    """

    def __init__(self, config: TissueConfig, device: str = 'auto'):
        self.config = config

        # Setup device
        self.backend = Backend(device=device, verbose=True)
        self.device = self.backend.device

        print(f"\nTissue Configuration:")
        print(f"  Domain: {config.Lx} x {config.Ly} cm")
        print(f"  Grid: {config.nx} x {config.ny} cells ({config.nx * config.ny:,} total)")
        print(f"  Resolution: {config.dx * 10:.1f} mm")
        print(f"  D = {config.D:.5f} cm²/ms")

        # Create ionic model with LUT acceleration
        self.ionic = TTP06Model(
            celltype=CellType.EPI,
            device=self.device,
            use_lut=False
        )
        print(f"  Ionic model: {self.ionic.name} (LUT enabled)")

        # Create diffusion operator
        self.diffusion = FDMDiffusion(
            config.ny, config.nx, config.dx, config.D, self.device
        )
        print(f"  Diffusion dt_max: {self.diffusion.dt_max:.4f} ms")

        # Initialize state array: (ny, nx, n_states)
        self.n_states = self.ionic.n_states
        self._init_states()

        # Time tracking
        self.time = 0.0

    def _init_states(self):
        """Initialize all cells to resting state."""
        initial = self.ionic.get_initial_state(1)  # Single cell state
        ny, nx = self.config.ny, self.config.nx

        # Expand to tissue: (ny, nx, n_states)
        self.states = initial.unsqueeze(0).unsqueeze(0).expand(ny, nx, -1).clone()

    def get_voltage(self) -> torch.Tensor:
        """Get voltage field (ny, nx)."""
        return self.states[:, :, 0]  # V is state index 0

    def set_voltage(self, V: torch.Tensor):
        """Set voltage field."""
        self.states[:, :, 0] = V

    def step(self, dt: float, I_stim: Optional[torch.Tensor] = None):
        """
        Advance simulation by one time step.

        Uses Godunov (first-order) operator splitting:
        1. Ionic step with stimulus
        2. Diffusion step
        """
        ny, nx = self.config.ny, self.config.nx

        # Reshape states for ionic model: (ny*nx, n_states)
        states_flat = self.states.view(-1, self.n_states)

        # Flatten stimulus if provided
        if I_stim is not None:
            I_stim_flat = I_stim.view(-1)
        else:
            I_stim_flat = None

        # === Ionic step ===
        states_flat = self.ionic.step(states_flat, dt, I_stim_flat)

        # Reshape back to (ny, nx, n_states)
        self.states = states_flat.view(ny, nx, self.n_states)

        # === Diffusion step ===
        V = self.get_voltage()
        V_new = self.diffusion.step(V, dt)
        self.set_voltage(V_new)

        self.time += dt

    def reset(self):
        """Reset to initial conditions."""
        self._init_states()
        self.time = 0.0


class SpiralWaveSimulation:
    """S1-S2 Cross-Field Stimulation for Spiral Wave Induction."""

    def __init__(self, domain_cm: float = 16.0, dx: float = 0.02):
        """
        Initialize simulation.

        Args:
            domain_cm: Physical size of domain in cm
            dx: Grid spacing in cm
        """
        # Create tissue configuration
        config = TissueConfig(
            Lx=domain_cm,
            Ly=domain_cm,
            dx=dx,
            D=0.00154,  # Tuned for CV ~60 cm/s
        )

        print("=" * 70)
        print("SPIRAL WAVE INDUCTION - S1-S2 Protocol (TTP06 + LUT)")
        print("=" * 70)

        # Create tissue simulation
        self.sim = TTP06TissueSimulation(config, device='auto')

        self.domain_cm = domain_cm
        self.ny, self.nx = config.ny, config.nx
        self.dx = config.dx

        # Time stepping
        self.dt = min(config.dt, self.sim.diffusion.dt_max * 0.8)
        self.steps_per_frame = max(1, int(0.5 / self.dt))
        self.speed_multiplier = 1.0

        print(f"\nTime stepping:")
        print(f"  dt = {self.dt:.4f} ms")
        print(f"  steps/frame = {self.steps_per_frame}")

        # S1-S2 Protocol parameters
        self.cv = 0.06  # cm/ms (estimated CV)
        self.s1_width_cm = 0.2  # Width of S1 stimulus (2mm)

        # S2 quadrant size
        self.s2_width_cm = domain_cm / 2   # Half domain width
        self.s2_height_cm = domain_cm / 2  # Half domain height

        # APD estimate for TTP06 EPI
        self.apd_estimate = 280.0  # ms

        # S2 timing window calculation
        self.tissue_erp_factor = 1.1  # Tissue ERP factor
        self.tissue_erp = self.apd_estimate * self.tissue_erp_factor
        self.time_to_s2_right = self.s2_width_cm / self.cv

        self.s2_window_start = self.tissue_erp
        self.s2_window_end = self.time_to_s2_right + self.tissue_erp
        self.s2_optimal = (self.s2_window_start + self.s2_window_end) / 2

        # Protocol state
        self.s1_applied = False
        self.s2_applied = False
        self.s1_time = None

        # Display setup
        self._setup_display()

        # FPS tracking
        self.frame_times = []
        self.max_frame_times = 30

        print(f"\nS1-S2 Protocol:")
        print(f"  S1: left edge, {self.s1_width_cm*10:.0f}mm wide")
        print(f"  S2: lower-left quadrant, {self.s2_width_cm:.1f}x{self.s2_height_cm:.1f} cm")
        print(f"  APD estimate: {self.apd_estimate:.0f} ms")
        print(f"  Vulnerable window: {self.s2_window_start:.0f} - {self.s2_window_end:.0f} ms")
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
        """Setup display parameters."""
        target_size = 600
        self.scale = max(1, target_size // max(self.ny, self.nx))
        self.img_w = self.nx * self.scale
        self.img_h = self.ny * self.scale

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
        self.static_canvas[cbar_y:cbar_y+cbar_h, cbar_x:cbar_x+cbar_w] = colorbar

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
            cv2.putText(self.static_canvas, f'{i}', (x_pos - 5, self.margin_top + self.img_h + 20),
                       font, 0.4, (255, 255, 255), 1)
        cv2.putText(self.static_canvas, 'x (cm)',
                   (self.margin_left + self.img_w // 2 - 20, self.canvas_h - 5),
                   font, 0.45, (255, 255, 255), 1)

    def apply_s1_stimulus(self):
        """Apply S1 stimulus on left edge."""
        V = self.sim.get_voltage()
        s1_cells = max(3, int(self.s1_width_cm / self.dx))
        V[:, :s1_cells] = 20.0  # Depolarize
        self.sim.set_voltage(V)
        self.s1_applied = True
        self.s1_time = self.sim.time
        print(f"  S1 applied at t = {self.sim.time:.1f} ms")

    def apply_s2_stimulus(self):
        """Apply S2 stimulus in lower-left quadrant."""
        V = self.sim.get_voltage()

        s2_x_cells = max(3, int(self.s2_width_cm / self.dx))
        s2_y_cells = max(3, int(self.s2_height_cm / self.dx))

        # Lower-left quadrant
        V[:s2_y_cells, :s2_x_cells] = 20.0

        self.sim.set_voltage(V)
        self.s2_applied = True

        elapsed = self.sim.time - self.s1_time if self.s1_time else 0
        print(f"  S2 applied at t = {self.sim.time:.1f} ms (interval: {elapsed:.0f} ms)")

    def start_s1(self):
        """Start with S1 stimulus."""
        self.sim.reset()
        self.s1_applied = False
        self.s2_applied = False
        self.s1_time = None
        print(f"\n--- Starting S1 ---")
        self.apply_s1_stimulus()

    def reset(self):
        """Reset simulation."""
        self.sim.reset()
        self.s1_applied = False
        self.s2_applied = False
        self.s1_time = None
        print("Simulation reset")

    def voltage_to_image(self, V: np.ndarray) -> np.ndarray:
        """Convert voltage to display image."""
        V_norm = np.clip((V + 90) / 130 * 255, 0, 255).astype(np.uint8)
        img = cv2.applyColorMap(V_norm, cv2.COLORMAP_JET)
        img = cv2.flip(img, 0)  # Flip Y axis

        if self.scale > 1:
            img = cv2.resize(img, (self.img_w, self.img_h),
                           interpolation=cv2.INTER_NEAREST)

        canvas = self.static_canvas.copy()
        canvas[self.margin_top:self.margin_top+self.img_h,
               self.margin_left:self.margin_left+self.img_w] = img

        return canvas

    def run(self):
        """Main simulation loop."""
        cv2.namedWindow('Spiral Wave - TTP06', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Spiral Wave - TTP06', 800, 800)

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
                self.sim.step(self.dt)

            # Get voltage and create image
            V = self.sim.get_voltage().cpu().numpy()
            img = self.voltage_to_image(V)

            # FPS
            frame_time = perf_counter() - frame_start
            self.frame_times.append(frame_time)
            if len(self.frame_times) > self.max_frame_times:
                self.frame_times.pop(0)
            fps = len(self.frame_times) / sum(self.frame_times)

            # Title
            title = f"Spiral Wave - TTP06 ({self.domain_cm:.0f}x{self.domain_cm:.0f} cm)"
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

            cv2.imshow('Spiral Wave - TTP06', img)

        cv2.destroyAllWindows()
        print("Simulation ended.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Spiral wave with TTP06 model')
    parser.add_argument('--domain', type=float, default=16.0,
                       help='Domain size in cm (default: 16.0)')
    parser.add_argument('--dx', type=float, default=0.02,
                       help='Grid spacing in cm (default: 0.02)')
    args = parser.parse_args()

    sim = SpiralWaveSimulation(domain_cm=args.domain, dx=args.dx)
    sim.run()


if __name__ == '__main__':
    main()
