#!/usr/bin/env python3
"""
Bidomain S1-S2 Spiral Wave Simulation

S1-S2 cross-field stimulation protocol on full bidomain equations.
GPU-accelerated via PyTorch + mixed spectral solver (DCT/DST).

Boundary modes (--bc flag):
  bath_all    - Dirichlet on all 4 edges (bath_coupled)
  bath_tb     - Dirichlet top/bottom, Neumann left/right
  bath_lr     - Dirichlet left/right, Neumann top/bottom
  insulated   - Neumann on all 4 edges (fully insulated)

Controls:
  S       - Start S1 (plane wave from left edge)
  SPACE   - Apply S2 (lower-left quadrant) during vulnerable window
  B       - Cycle through boundary conditions (rebuilds simulation)
  R       - Reset simulation
  +/-     - Adjust speed
  Q/ESC   - Quit
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
torch.set_default_dtype(torch.float64)

import numpy as np
import cv2
from time import perf_counter

from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
from cardiac_sim.tissue_builder.mesh.boundary import BoundarySpec, Edge
from cardiac_sim.tissue_builder.tissue.conductivity import BidomainConductivity
from cardiac_sim.simulation.classical.discretization.fdm import BidomainFDMDiscretization
from cardiac_sim.simulation.classical.bidomain import BidomainSimulation
from cardiac_sim.tissue_builder.stimulus import StimulusProtocol


# === Boundary condition modes ===

BC_MODES = ['bath_all', 'bath_tb', 'bath_lr', 'insulated']

BC_LABELS = {
    'bath_all':  'Dirichlet (all edges)',
    'bath_tb':   'Dirichlet TB / Neumann LR',
    'bath_lr':   'Dirichlet LR / Neumann TB',
    'insulated': 'Neumann (all edges)',
}


def _make_boundary_spec(bc_mode):
    if bc_mode == 'bath_all':
        return BoundarySpec.bath_coupled()
    elif bc_mode == 'bath_tb':
        return BoundarySpec.bath_coupled_edges([Edge.TOP, Edge.BOTTOM])
    elif bc_mode == 'bath_lr':
        return BoundarySpec.bath_coupled_edges([Edge.LEFT, Edge.RIGHT])
    elif bc_mode == 'insulated':
        return BoundarySpec.insulated()
    raise ValueError(f"Unknown BC mode: {bc_mode}")


# === Device selection ===

def select_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class BidomainSpiralWave:
    """S1-S2 protocol on bidomain equations with switchable BCs.

    Follows V5.4 spiral_wave_s1s2.py pattern: voltage clamping for S1/S2,
    start_s1() = reset + apply, no cached state references in run loop.
    """

    def __init__(self, domain_cm=8.0, dx=0.025, dt=0.01, bc='bath_tb'):
        self.domain_cm = domain_cm
        self.dx = dx
        self.dt = dt
        self.device = select_device()
        self.bc_mode = bc

        self.Nx = int(domain_cm / dx) + 1
        self.Ny = self.Nx

        print("=" * 70)
        print("BIDOMAIN S1-S2 SPIRAL WAVE")
        print("=" * 70)
        print(f"  Device: {self.device}")
        print(f"  Domain: {domain_cm} x {domain_cm} cm")
        print(f"  Grid: {self.Nx} x {self.Ny} ({self.Nx * self.Ny:,} nodes)")
        print(f"  dx = {dx} cm, dt = {dt} ms")
        print(f"  BCs: {BC_LABELS[bc]}")

        self._build_sim(bc)

        # S1-S2 protocol parameters
        self.cv = 0.054  # cm/ms estimated CV
        self.s1_width_cm = 0.2
        self.s2_width_cm = domain_cm / 2
        self.s2_height_cm = domain_cm / 2
        self.apd_estimate = 280.0

        # Precompute cell counts
        self.s1_cells = max(3, int(self.s1_width_cm / dx))
        self.s2_x_cells = max(3, int(self.s2_width_cm / dx))
        self.s2_y_cells = max(3, int(self.s2_height_cm / dx))

        # Vulnerable window
        erp = self.apd_estimate * 1.1
        travel = self.s2_width_cm / self.cv
        self.s2_window_start = erp
        self.s2_window_end = travel + erp
        self.s2_optimal = (self.s2_window_start + self.s2_window_end) / 2

        # Protocol state
        self.s1_applied = False
        self.s2_applied = False
        self.s1_time = None

        # Display
        self.speed_multiplier = 1.0
        self.steps_per_frame = max(1, int(0.5 / dt))
        self._setup_display()
        self.frame_times = []

        print(f"\n  Elliptic solver: {self.sim._elliptic_solver_name}")
        print(f"  S1: left edge, {self.s1_width_cm*10:.0f}mm wide ({self.s1_cells} cells)")
        print(f"  S2: lower-left quadrant, {self.s2_width_cm:.1f}x{self.s2_height_cm:.1f} cm")
        print(f"  Vulnerable window: {self.s2_window_start:.0f} - {self.s2_window_end:.0f} ms")
        print(f"  Optimal S2: ~{self.s2_optimal:.0f} ms after S1")
        print(f"\nControls: S=start S1, SPACE=S2, B=cycle BCs, "
              f"R=reset, +/-=speed, Q=quit")

    def _build_sim(self, bc_mode):
        """Build (or rebuild) the simulation with given BC mode."""
        self.bc_mode = bc_mode

        grid = StructuredGrid.create_rectangle(
            Lx=self.domain_cm, Ly=self.domain_cm,
            Nx=self.Nx, Ny=self.Ny,
            device=str(self.device))
        grid.boundary_spec = _make_boundary_spec(bc_mode)
        self.grid = grid

        cond = BidomainConductivity()
        spatial = BidomainFDMDiscretization(grid, cond, chi=1.0, Cm=1.0)

        # Empty stimulus protocol — using voltage clamping like V5.4
        self.sim = BidomainSimulation(
            spatial=spatial,
            ionic_model='ttp06',
            stimulus=StimulusProtocol(),
            dt=self.dt,
            splitting='strang',
            elliptic_solver='auto',
            theta=0.5,
            device=self.device)

    def _setup_display(self):
        """Setup OpenCV display with colorbar."""
        target_size = 600
        display_dim = min(self.Nx, self.Ny)
        self.scale = max(1, target_size // display_dim)
        self.img_w = display_dim * self.scale
        self.img_h = display_dim * self.scale

        self.margin_left = 60
        self.margin_bottom = 50
        self.margin_right = 80
        self.margin_top = 50
        self.canvas_h = self.img_h + self.margin_top + self.margin_bottom
        self.canvas_w = self.img_w + self.margin_left + self.margin_right

        self.static_canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
        self.static_canvas[:] = (30, 30, 30)

        cbar_x = self.margin_left + self.img_w + 15
        cbar_w, cbar_h = 20, self.img_h
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
            cv2.putText(self.static_canvas, f'{i}',
                        (x_pos - 5, self.margin_top + self.img_h + 20),
                        font, 0.4, (255, 255, 255), 1)
        cv2.putText(self.static_canvas, 'x (cm)',
                    (self.margin_left + self.img_w // 2 - 20, self.canvas_h - 5),
                    font, 0.45, (255, 255, 255), 1)

    # === S1-S2 Protocol (voltage clamping, matching V5.4) ===

    def apply_s1(self):
        """Apply S1 stimulus: voltage clamp left edge to +20 mV."""
        V_grid = self.grid.flat_to_grid(self.sim.state.Vm)
        V_grid[:self.s1_cells, :] = 20.0
        self.sim.state.Vm = self.grid.grid_to_flat(V_grid)
        self.s1_applied = True
        self.s1_time = self.sim.state.t
        print(f"  S1 applied at t = {self.sim.state.t:.1f} ms")

    def apply_s2(self):
        """Apply S2 stimulus: voltage clamp lower-left quadrant to +20 mV."""
        V_grid = self.grid.flat_to_grid(self.sim.state.Vm)
        V_grid[:self.s2_x_cells, :self.s2_y_cells] = 20.0
        self.sim.state.Vm = self.grid.grid_to_flat(V_grid)
        self.s2_applied = True
        elapsed = self.sim.state.t - self.s1_time if self.s1_time is not None else 0
        print(f"  S2 applied at t={self.sim.state.t:.1f}ms (interval: {elapsed:.0f}ms)")

    def start_s1(self):
        """Reset simulation and apply S1 (matches V5.4 pattern)."""
        self.reset()
        print(f"\n--- Starting S1 ---")
        self.apply_s1()

    def reset(self):
        """Reset simulation to resting state."""
        state = self.sim.state
        ionic_model = self.sim.splitting.ionic_solver.ionic_model
        n_dof = state.n_dof
        state.Vm = torch.full((n_dof,), ionic_model.V_rest,
                              device=self.device, dtype=torch.float64)
        state.phi_e = torch.zeros(n_dof, device=self.device, dtype=torch.float64)
        state.ionic_states = ionic_model.get_initial_state(n_dof)
        state.t = 0.0
        self.s1_applied = False
        self.s2_applied = False
        self.s1_time = None
        print("Simulation reset")

    def cycle_bc(self):
        """Cycle to next BC mode and rebuild simulation from scratch."""
        idx = BC_MODES.index(self.bc_mode)
        new_mode = BC_MODES[(idx + 1) % len(BC_MODES)]
        print(f"\n  Switching BCs: {BC_LABELS[self.bc_mode]} -> {BC_LABELS[new_mode]}")
        self._build_sim(new_mode)
        self.s1_applied = False
        self.s2_applied = False
        self.s1_time = None
        print(f"  Elliptic solver: {self.sim._elliptic_solver_name}")
        print(f"  Press S to start S1 with new BCs")

    # === Display ===

    def voltage_to_image(self, V_grid_np):
        """Convert voltage grid (Nx, Ny) to display image.

        Transpose so x=horizontal, flip so y increases upward.
        Matches V5.4 display convention.
        """
        V_display = V_grid_np.T   # (Ny, Nx): rows=y, cols=x
        V_display = np.flipud(V_display)  # y=0 at bottom

        V_norm = np.clip((V_display + 90) / 130 * 255, 0, 255).astype(np.uint8)
        img = cv2.applyColorMap(V_norm, cv2.COLORMAP_JET)

        if self.scale > 1:
            img = cv2.resize(img, (self.img_w, self.img_h),
                             interpolation=cv2.INTER_NEAREST)

        canvas = self.static_canvas.copy()
        canvas[self.margin_top:self.margin_top+self.img_h,
               self.margin_left:self.margin_left+self.img_w] = img
        return canvas

    # === Main loop ===

    def run(self):
        """Main interactive simulation loop.

        IMPORTANT: Never cache self.sim.state or self.sim.splitting as local
        vars — they change when cycle_bc() rebuilds the simulation.
        """
        win_name = 'Bidomain S1-S2'
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 800, 800)

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
                    self.apply_s2()
                elif not self.s1_applied:
                    print("  (Apply S1 first by pressing S)")
            elif key == ord('b'):
                self.cycle_bc()
            elif key == ord('r'):
                self.reset()
            elif key in (ord('+'), ord('=')):
                self.speed_multiplier = min(8.0, self.speed_multiplier * 1.5)
                print(f"Speed: {self.speed_multiplier:.1f}x")
            elif key == ord('-'):
                self.speed_multiplier = max(0.25, self.speed_multiplier / 1.5)
                print(f"Speed: {self.speed_multiplier:.1f}x")

            # Step simulation — always use self.sim (not cached refs)
            actual_steps = max(1, int(self.steps_per_frame * self.speed_multiplier))
            state = self.sim.state
            splitting = self.sim.splitting
            dt = self.dt
            for _ in range(actual_steps):
                splitting.step(state, dt)
                state.t += dt

            # Display
            V_grid = self.grid.flat_to_grid(state.Vm)
            V_np = V_grid.detach().cpu().numpy()
            img = self.voltage_to_image(V_np)

            # FPS
            frame_time = perf_counter() - frame_start
            self.frame_times.append(frame_time)
            if len(self.frame_times) > 30:
                self.frame_times.pop(0)
            fps = len(self.frame_times) / sum(self.frame_times)

            # Overlay text
            font = cv2.FONT_HERSHEY_SIMPLEX
            title = f"Bidomain {self.domain_cm:.0f}x{self.domain_cm:.0f}cm | {BC_LABELS[self.bc_mode]}"
            cv2.putText(img, title, (60, 30), font, 0.6, (255, 255, 255), 1)

            V_min, V_max = V_np.min(), V_np.max()
            status = (f"t={state.t:.1f}ms | {fps:.1f}FPS | "
                      f"V:[{V_min:.0f},{V_max:.0f}]mV | {self.device}")
            cv2.putText(img, status, (60, img.shape[0] - 30), font,
                        0.4, (200, 200, 200), 1)

            # Protocol status
            if self.s2_applied:
                cv2.putText(img, "SPIRAL FORMING", (img.shape[1] - 200, 30),
                            font, 0.6, (0, 255, 0), 2)
            elif self.s1_applied:
                elapsed = state.t - self.s1_time
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
                            font, 0.5, color, 1)
            else:
                cv2.putText(img, "Press S to start", (img.shape[1] - 180, 30),
                            font, 0.5, (100, 100, 255), 1)

            cv2.imshow(win_name, img)

        cv2.destroyAllWindows()
        print("Simulation ended.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Bidomain S1-S2 spiral wave')
    parser.add_argument('--domain', type=float, default=8.0,
                        help='Domain size in cm (default: 8.0)')
    parser.add_argument('--dx', type=float, default=0.025,
                        help='Grid spacing in cm (default: 0.025)')
    parser.add_argument('--dt', type=float, default=0.01,
                        help='Time step in ms (default: 0.01)')
    parser.add_argument('--bc', type=str, default='bath_tb',
                        choices=BC_MODES,
                        help='Boundary condition mode (default: bath_tb)')
    args = parser.parse_args()

    sim = BidomainSpiralWave(domain_cm=args.domain, dx=args.dx, dt=args.dt,
                             bc=args.bc)
    sim.run()


if __name__ == '__main__':
    main()
