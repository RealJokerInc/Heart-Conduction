#!/usr/bin/env python3
"""
Bidomain S1-S2 Spiral Wave Simulation

S1-S2 cross-field stimulation protocol on full bidomain equations.
Uses Dirichlet (bath-coupled) BCs: phi_e = 0 on all boundaries.
GPU-accelerated via PyTorch + torch_dct spectral solver.

Controls:
  S       - Start S1 (plane wave from left edge) / Reset
  SPACE   - Apply S2 (lower-left quadrant) during vulnerable window
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
from cardiac_sim.tissue_builder.mesh.boundary import BoundarySpec
from cardiac_sim.tissue_builder.tissue.conductivity import BidomainConductivity
from cardiac_sim.simulation.classical.discretization.fdm import BidomainFDMDiscretization
from cardiac_sim.simulation.classical.bidomain import BidomainSimulation
from cardiac_sim.tissue_builder.stimulus import StimulusProtocol, Stimulus


def select_device():
    """Auto-select best device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class BidomainSpiralWave:
    """S1-S2 protocol on bidomain equations with bath-coupled BCs."""

    def __init__(self, domain_cm=8.0, dx=0.025, dt=0.01):
        self.domain_cm = domain_cm
        self.dx = dx
        self.dt = dt
        self.device = select_device()

        nx = int(domain_cm / dx) + 1
        ny = nx
        self.nx, self.ny = nx, ny

        print("=" * 70)
        print("BIDOMAIN S1-S2 SPIRAL WAVE (Bath-Coupled Dirichlet BCs)")
        print("=" * 70)
        print(f"  Device: {self.device}")
        print(f"  Domain: {domain_cm} x {domain_cm} cm")
        print(f"  Grid: {nx} x {ny} ({nx * ny:,} nodes)")
        print(f"  dx = {dx} cm, dt = {dt} ms")

        # Build grid on device
        grid = StructuredGrid.create_rectangle(
            Lx=domain_cm, Ly=domain_cm, Nx=nx, Ny=ny,
            device=str(self.device))
        grid.boundary_spec = BoundarySpec.bath_coupled()

        # Conductivity: D_i and D_e already scaled by chi*Cm
        # sigma_i=1.74, sigma_e=6.25, chi=1400, Cm=1.0
        cond = BidomainConductivity()  # defaults
        print(f"  D_i = {cond.D_i:.6f}, D_e = {cond.D_e:.6f}")
        print(f"  D_eff = {cond.get_effective_monodomain_D():.6f}")

        # Build spatial discretization (chi=1, Cm=1 since D already scaled)
        spatial = BidomainFDMDiscretization(grid, cond, chi=1.0, Cm=1.0)

        # S1 stimulus: left edge strip
        s1_width = 0.2  # cm
        s1 = Stimulus(
            region=lambda x, y: (x <= s1_width),
            start_time=0.0, duration=1.0,
            amplitude=-52.0)
        stim = StimulusProtocol([s1])

        # Build simulation
        self.sim = BidomainSimulation(
            spatial=spatial,
            ionic_model='ttp06',
            stimulus=stim,
            dt=dt,
            splitting='strang',
            elliptic_solver='auto',  # will pick spectral (DST for Dirichlet)
            theta=0.5,
            device=self.device)
        self.grid = grid

        # S1-S2 protocol state
        self.s1_applied = True  # S1 is in the stimulus protocol at t=0
        self.s2_applied = False
        self.s1_time = 0.0

        # S2 parameters
        self.s2_width_cm = domain_cm / 2
        self.s2_height_cm = domain_cm / 2
        self.apd_estimate = 280.0
        self.cv_estimate = 0.054  # cm/ms

        # Vulnerable window
        erp = self.apd_estimate * 1.1
        travel = self.s2_width_cm / self.cv_estimate
        self.s2_window_start = erp
        self.s2_window_end = travel + erp
        self.s2_optimal = (self.s2_window_start + self.s2_window_end) / 2

        # Display
        self.speed_multiplier = 1.0
        self.steps_per_frame = max(1, int(0.5 / dt))  # ~0.5ms sim per frame
        self._setup_display()
        self.frame_times = []

        print(f"\n  Elliptic solver: {self.sim._elliptic_solver_name}")
        print(f"  S2 window: {self.s2_window_start:.0f} - {self.s2_window_end:.0f} ms")
        print(f"  Optimal S2: ~{self.s2_optimal:.0f} ms")
        print(f"\nControls: S=start/reset, SPACE=S2, R=reset, +/-=speed, Q=quit")

    def _setup_display(self):
        """Setup OpenCV display."""
        target_size = 600
        self.scale = max(1, target_size // max(self.ny, self.nx))
        self.img_w = self.nx * self.scale
        self.img_h = self.ny * self.scale

        self.margin_left = 60
        self.margin_bottom = 50
        self.margin_right = 80
        self.margin_top = 50
        self.canvas_h = self.img_h + self.margin_top + self.margin_bottom
        self.canvas_w = self.img_w + self.margin_left + self.margin_right

        # Static background with colorbar
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

    def apply_s2(self):
        """Apply S2 stimulus: depolarize lower-left quadrant."""
        state = self.sim.state
        Vm = state.Vm.reshape(self.nx, self.ny)
        s2_x = max(3, int(self.s2_width_cm / self.dx))
        s2_y = max(3, int(self.s2_height_cm / self.dx))
        Vm[:s2_x, :s2_y] = 20.0
        state.Vm = Vm.flatten()
        self.s2_applied = True
        elapsed = self.sim.state.t - self.s1_time
        print(f"  S2 applied at t={state.t:.1f}ms (interval: {elapsed:.0f}ms)")

    def reset(self):
        """Reset simulation by rebuilding state."""
        # Re-run the S1 stimulus by resetting time
        state = self.sim.state
        ionic_model = self.sim.splitting.ionic_solver.ionic_model
        n_dof = state.n_dof
        state.Vm = torch.full((n_dof,), ionic_model.V_rest,
                              device=self.device, dtype=torch.float64)
        state.phi_e = torch.zeros(n_dof, device=self.device, dtype=torch.float64)
        state.ionic_states = ionic_model.get_initial_state(n_dof)
        state.t = 0.0
        self.s1_applied = True
        self.s2_applied = False
        self.s1_time = 0.0
        print("Simulation reset")

    def voltage_to_image(self, V_np):
        """Convert voltage array to colored image."""
        V_norm = np.clip((V_np + 90) / 130 * 255, 0, 255).astype(np.uint8)
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
        win_name = 'Bidomain S1-S2'
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 800, 800)

        state = self.sim.state
        splitting = self.sim.splitting
        dt = self.dt

        print("Running... Press S to start S1, then SPACE for S2")

        while True:
            frame_start = perf_counter()

            # Keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                self.reset()
            elif key == ord(' '):
                if self.s1_applied and not self.s2_applied:
                    self.apply_s2()
            elif key == ord('r'):
                self.reset()
            elif key in (ord('+'), ord('=')):
                self.speed_multiplier = min(8.0, self.speed_multiplier * 1.5)
                print(f"Speed: {self.speed_multiplier:.1f}x")
            elif key == ord('-'):
                self.speed_multiplier = max(0.25, self.speed_multiplier / 1.5)
                print(f"Speed: {self.speed_multiplier:.1f}x")

            # Step simulation
            actual_steps = max(1, int(self.steps_per_frame * self.speed_multiplier))
            for _ in range(actual_steps):
                splitting.step(state, dt)
                state.t += dt

            # Get voltage for display
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
            title = f"Bidomain {self.domain_cm:.0f}x{self.domain_cm:.0f}cm (Dirichlet)"
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
                if self.s2_window_start <= elapsed <= self.s2_window_end:
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
    args = parser.parse_args()

    sim = BidomainSpiralWave(domain_cm=args.domain, dx=args.dx, dt=args.dt)
    sim.run()


if __name__ == '__main__':
    main()
