#!/usr/bin/env python3
"""
Spiral Wave Induction using S1-S2 Cross-Field Stimulation Protocol

O'Hara-Rudy 2011 (ORd) model with GPU acceleration.

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

from ionic import ORdModel, CellType
from utils import Backend


@dataclass
class TissueConfig:
    """Configuration for 2D tissue simulation."""
    # Domain
    Lx: float = 16.0         # Domain width (cm)
    Ly: float = 16.0         # Domain height (cm)
    dx: float = 0.02         # Grid spacing (cm) - 200 um

    # Diffusion
    D: float = 0.00154       # Diffusion coefficient (cm^2/ms) for CV ~60 cm/s

    # Time stepping
    dt: float = 0.02         # Time step (ms) - same as V5.1

    @property
    def nx(self) -> int:
        return int(self.Lx / self.dx)

    @property
    def ny(self) -> int:
        return int(self.Ly / self.dx)


class FDMDiffusion:
    """
    Finite Difference Method diffusion operator (5-point stencil).

    Solves: dV/dt = D * nabla^2 V

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

        # Stability limit for explicit scheme: dt < dx^2/(4D)
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
            D * nabla^2 V, shape (ny, nx)
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


class ORdTissueSimulation:
    """
    2D Tissue simulation with ORd model.

    Uses operator splitting:
    1. Ionic step: ORd model (Rush-Larsen)
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
        print(f"  D = {config.D:.5f} cm^2/ms")

        # Create ionic model (no LUT for ORd - direct computation)
        self.ionic = ORdModel(
            celltype=CellType.ENDO,
            device=self.device
        )
        print(f"  Ionic model: {self.ionic.name} ({self.ionic.n_states} states)")

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

        # Step 1: Ionic step
        states_flat = self.ionic.step(states_flat, dt, I_stim_flat)

        # Reshape back to tissue
        self.states = states_flat.view(ny, nx, self.n_states)

        # Step 2: Diffusion step
        V = self.get_voltage()
        V_new = self.diffusion.step(V, dt)
        self.set_voltage(V_new)

        self.time += dt


class SpiralWaveSimulation:
    """S1-S2 Cross-Field Stimulation for Spiral Wave Induction."""

    def __init__(self, domain_cm: float = 16.0, dx: float = 0.02):
        """
        Initialize spiral wave simulation.

        Parameters
        ----------
        domain_cm : float
            Physical size of domain in cm
        dx : float
            Grid spacing in cm
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("=" * 60)
        print("SPIRAL WAVE INDUCTION - S1-S2 Protocol (ORd + GPU)")
        print("=" * 60)

        # Create tissue configuration
        config = TissueConfig(
            Lx=domain_cm,
            Ly=domain_cm,
            dx=dx,
            dt=0.02  # Same as V5.1
        )

        # Calculate stability limit
        dt_stability = dx * dx / (4 * config.D)
        print(f"\nStability Analysis:")
        print(f"  dt_stability = {dt_stability:.4f} ms")
        print(f"  dt_used = {config.dt:.4f} ms")
        print(f"  Safety factor = {dt_stability / config.dt:.2f}x")

        # Create tissue simulation
        self.sim = ORdTissueSimulation(config, device=self.device)
        self.config = config
        self.ny, self.nx = config.ny, config.nx
        self.dt = config.dt

        # Simulation parameters
        self.steps_per_frame = max(1, int(0.5 / self.dt))  # ~0.5ms per frame
        self.speed_multiplier = 1.0

        print(f"\nSimulation Parameters:")
        print(f"  dt = {self.dt:.4f} ms")
        print(f"  steps/frame = {self.steps_per_frame}")

        # S1-S2 Protocol parameters
        self.domain_cm = domain_cm
        self.cv = 0.06  # cm/ms (estimated CV)
        self.s1_width_cm = 0.2  # Width of S1 stimulus (2mm)

        # S2 quadrant size
        self.s2_width_cm = domain_cm / 2   # Half domain width
        self.s2_height_cm = domain_cm / 2  # Half domain height

        # APD estimate for ORd ENDO
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

        print("\nControls:")
        print("  S       - Start S1 / Reset")
        print("  SPACE   - Apply S2 (in vulnerable window)")
        print("  R       - Reset simulation")
        print("  +/-     - Adjust speed")
        print("  Q/ESC   - Quit")

    def _setup_display(self):
        """Setup OpenCV display."""
        # Display size
        self.img_w = 600
        self.img_h = 600

        # Margins for colorbar
        self.margin_left = 50
        self.margin_right = 80
        self.margin_top = 40
        self.margin_bottom = 50

        self.canvas_w = self.margin_left + self.img_w + self.margin_right
        self.canvas_h = self.margin_top + self.img_h + self.margin_bottom

        # Create colormap (jet)
        self.colormap = cv2.COLORMAP_JET

        # Static canvas elements
        self._create_static_canvas()

    def _create_static_canvas(self):
        """Create static parts of the display."""
        self.static_canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)

        # Draw colorbar
        cb_x = self.margin_left + self.img_w + 20
        cb_w = 20
        cb_h = self.img_h

        for i in range(cb_h):
            val = int(255 * (cb_h - 1 - i) / (cb_h - 1))
            color = cv2.applyColorMap(np.array([[val]], dtype=np.uint8), self.colormap)[0, 0]
            cv2.line(self.static_canvas, (cb_x, self.margin_top + i),
                    (cb_x + cb_w, self.margin_top + i), color.tolist(), 1)

        # Colorbar labels
        cv2.putText(self.static_canvas, "+40 mV", (cb_x, self.margin_top - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(self.static_canvas, "-85 mV", (cb_x, self.margin_top + cb_h + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Draw scale markers
        for i in range(int(self.domain_cm) + 1):
            x_pos = self.margin_left + int(i / self.domain_cm * self.img_w)
            y_pos = self.margin_top + self.img_h

            # X-axis ticks (every 2 cm)
            if i % 2 == 0:
                cv2.line(self.static_canvas, (x_pos, y_pos), (x_pos, y_pos + 5), (150, 150, 150), 1)
                cv2.putText(self.static_canvas, f"{i}", (x_pos - 5, y_pos + 18),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

        # Y-axis ticks
        for i in range(int(self.domain_cm) + 1):
            y_pos = self.margin_top + self.img_h - int(i / self.domain_cm * self.img_h)
            if i % 2 == 0:
                cv2.line(self.static_canvas, (self.margin_left - 5, y_pos),
                        (self.margin_left, y_pos), (150, 150, 150), 1)
                cv2.putText(self.static_canvas, f"{i}", (self.margin_left - 25, y_pos + 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

        # Axis labels
        cv2.putText(self.static_canvas, "x (cm)", (self.margin_left + self.img_w // 2 - 20,
                   self.canvas_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    def apply_s1_stimulus(self):
        """Apply S1 stimulus on left edge (direct voltage setting)."""
        V = self.sim.get_voltage()
        s1_cols = max(3, int(self.s1_width_cm / self.config.dx))
        V[:, :s1_cols] = 20.0  # Depolarize to +20 mV
        self.sim.set_voltage(V)
        self.s1_applied = True
        self.s1_time = self.sim.time
        print(f"  S1 applied at t = {self.sim.time:.1f} ms (cols 0-{s1_cols})")

    def apply_s2_stimulus(self):
        """Apply S2 stimulus in lower-left quadrant (direct voltage setting)."""
        V = self.sim.get_voltage()
        s2_cols = max(3, int(self.s2_width_cm / self.config.dx))
        s2_rows = max(3, int(self.s2_height_cm / self.config.dx))
        # Lower-left quadrant: rows [0, s2_rows), cols [0, s2_cols)
        V[:s2_rows, :s2_cols] = 20.0  # Depolarize to +20 mV
        self.sim.set_voltage(V)
        self.s2_applied = True
        elapsed = self.sim.time - self.s1_time if self.s1_time else 0
        print(f"  S2 applied at t = {self.sim.time:.1f} ms (interval: {elapsed:.0f} ms)")

    def start_s1_protocol(self):
        """Start with S1 stimulus."""
        self.sim._init_states()
        self.sim.time = 0.0
        self.s1_applied = False
        self.s2_applied = False
        print(f"\n--- Starting S1 ---")
        self.apply_s1_stimulus()

    def reset(self):
        """Reset simulation to initial state."""
        self.sim._init_states()
        self.sim.time = 0.0
        self.s1_applied = False
        self.s2_applied = False
        self.s1_time = None
        print("\n--- Simulation Reset ---")

    def render_frame(self) -> np.ndarray:
        """Render current state to image."""
        V = self.sim.get_voltage().cpu().numpy()

        # Normalize to [0, 255]
        V_norm = np.clip((V + 85) / 125, 0, 1)  # -85 to +40 mV
        V_uint8 = (V_norm * 255).astype(np.uint8)

        # Resize to display size
        V_resized = cv2.resize(V_uint8, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)

        # Flip Y axis so row 0 is at bottom (matches physical coordinates)
        V_resized = cv2.flip(V_resized, 0)

        # Apply colormap
        V_color = cv2.applyColorMap(V_resized, self.colormap)

        # Create canvas
        canvas = self.static_canvas.copy()

        # Place voltage image
        canvas[self.margin_top:self.margin_top + self.img_h,
               self.margin_left:self.margin_left + self.img_w] = V_color

        # Add dynamic info
        self._add_info_overlay(canvas, V)

        return canvas

    def _add_info_overlay(self, canvas: np.ndarray, V: np.ndarray):
        """Add dynamic information to canvas."""
        # FPS calculation
        if len(self.frame_times) > 1:
            avg_dt = np.mean(np.diff(self.frame_times[-30:]))
            fps = 1.0 / avg_dt if avg_dt > 0 else 0
        else:
            fps = 0

        # Title with time
        title = f"Spiral Wave - ORd ({self.domain_cm:.0f}x{self.domain_cm:.0f} cm)"
        cv2.putText(canvas, title, (self.margin_left, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Time and FPS
        time_str = f"t = {self.sim.time:.1f} ms | FPS: {fps:.0f} | Speed: {self.speed_multiplier:.1f}x"
        cv2.putText(canvas, time_str, (self.margin_left + 300, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Voltage range
        V_min, V_max = V.min(), V.max()
        range_str = f"V: [{V_min:.0f}, {V_max:.0f}] mV"
        cv2.putText(canvas, range_str, (self.margin_left, self.canvas_h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # Protocol status
        if self.s1_time:
            elapsed = self.sim.time - self.s1_time
            status = f"S1 @ {self.s1_time:.0f}ms | Elapsed: {elapsed:.0f}ms"
            if self.s2_applied:
                status += " | S2 applied"
            elif elapsed >= self.s2_window_start and elapsed <= self.s2_window_end:
                status += " | VULNERABLE WINDOW"
            cv2.putText(canvas, status, (self.margin_left + 200, self.canvas_h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)

    def run(self):
        """Run the interactive simulation."""
        cv2.namedWindow("Spiral Wave - ORd", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Spiral Wave - ORd", self.canvas_w, self.canvas_h)

        print("\nStarting... Press 'S' to apply S1, then SPACE for S2")

        running = True
        while running:
            t_start = perf_counter()

            # Run simulation steps
            steps = int(self.steps_per_frame * self.speed_multiplier)
            for _ in range(steps):
                self.sim.step(self.dt)

            # Render and display
            frame = self.render_frame()
            cv2.imshow("Spiral Wave - ORd", frame)

            # Track frame time
            self.frame_times.append(perf_counter())
            if len(self.frame_times) > self.max_frame_times:
                self.frame_times = self.frame_times[-self.max_frame_times:]

            # Handle input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # Q or ESC
                running = False
            elif key == ord('s'):
                self.start_s1_protocol()
            elif key == ord(' '):
                if self.s1_applied and not self.s2_applied:
                    print(f"\n--- Applying S2 ---")
                    self.apply_s2_stimulus()
                else:
                    print("  (Apply S1 first by pressing S)")
            elif key == ord('r'):
                self.reset()
            elif key == ord('+') or key == ord('='):
                self.speed_multiplier = min(10.0, self.speed_multiplier * 1.5)
                print(f"  Speed: {self.speed_multiplier:.1f}x")
            elif key == ord('-'):
                self.speed_multiplier = max(0.1, self.speed_multiplier / 1.5)
                print(f"  Speed: {self.speed_multiplier:.1f}x")

        cv2.destroyAllWindows()
        print("\nSimulation ended.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Spiral wave with ORd model')
    parser.add_argument('--domain', type=float, default=16.0,
                       help='Domain size in cm (default: 16.0)')
    parser.add_argument('--dx', type=float, default=0.02,
                       help='Grid spacing in cm (default: 0.02)')
    args = parser.parse_args()

    sim = SpiralWaveSimulation(domain_cm=args.domain, dx=args.dx)
    sim.run()


if __name__ == '__main__':
    main()
