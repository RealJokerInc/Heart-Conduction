"""
Spiral Wave Formation via Triangle Infarct
===========================================

Demonstrates spiral wave (pinwheel) formation using:
- Single uniform planar wave from left edge
- Triangle infarct pointing LEFT (tip faces incoming wave)
- Wave break at triangle tip induces spiral rotation

Mechanism:
  1. Planar wave propagates left → right
  2. Wave hits left-pointing triangle tip first
  3. Wave splits around triangle
  4. Asymmetric recovery creates rotation at triangle apex
  5. Counter-rotating spirals form at top/bottom corners

Author: Generated with Claude Code
Date: 2025-12-10
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
import numba

from parameters import (
    FKParams, SpatialParams, PhysicalConstants,
    get_paramset_apd250
)
from fenton_karma import FentonKarmaModel


# =============================================================================
# Diffusion with Infarct Mask (Numba-accelerated)
# =============================================================================

@numba.jit(nopython=True, cache=True)
def diffusion_with_infarct_kernel(
    u: np.ndarray,
    mask: np.ndarray,
    Dxx_dt_dx2: float,
    Dyy_dt_dy2: float,
) -> np.ndarray:
    """
    Isotropic diffusion with binary infarct mask.

    Parameters
    ----------
    u : np.ndarray (ny, nx)
        Voltage field
    mask : np.ndarray (ny, nx)
        Conductivity mask: 1.0 = normal tissue, 0.0 = infarct (no conduction)
    Dxx_dt_dx2, Dyy_dt_dy2 : float
        Pre-scaled diffusion coefficients

    Returns
    -------
    u_new : np.ndarray
        Updated voltage field
    """
    ny, nx = u.shape
    u_new = np.copy(u)

    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            # Skip infarct cells
            if mask[i, j] < 0.5:
                continue

            # Check neighbor masks - treat infarct as no-flux boundary
            m_left = mask[i, j-1]
            m_right = mask[i, j+1]
            m_down = mask[i-1, j]
            m_up = mask[i+1, j]

            # Use current value if neighbor is infarct (no-flux)
            u_left = u[i, j-1] if m_left > 0.5 else u[i, j]
            u_right = u[i, j+1] if m_right > 0.5 else u[i, j]
            u_down = u[i-1, j] if m_down > 0.5 else u[i, j]
            u_up = u[i+1, j] if m_up > 0.5 else u[i, j]

            # Laplacian
            lap_x = u_right - 2.0 * u[i, j] + u_left
            lap_y = u_up - 2.0 * u[i, j] + u_down

            u_new[i, j] = u[i, j] + Dxx_dt_dx2 * lap_x + Dyy_dt_dy2 * lap_y

    # Domain boundary conditions (no-flux / Neumann)
    # Top
    for j in range(nx):
        if mask[ny-1, j] > 0.5:
            u_new[ny-1, j] = u_new[ny-2, j] if mask[ny-2, j] > 0.5 else u[ny-1, j]
    # Bottom
    for j in range(nx):
        if mask[0, j] > 0.5:
            u_new[0, j] = u_new[1, j] if mask[1, j] > 0.5 else u[0, j]
    # Left
    for i in range(ny):
        if mask[i, 0] > 0.5:
            u_new[i, 0] = u_new[i, 1] if mask[i, 1] > 0.5 else u[i, 0]
    # Right
    for i in range(ny):
        if mask[i, nx-1] > 0.5:
            u_new[i, nx-1] = u_new[i, nx-2] if mask[i, nx-2] > 0.5 else u[i, nx-1]

    return u_new


# =============================================================================
# Spiral Wave Simulation Class
# =============================================================================

class SpiralWaveSimulation:
    """
    Simulation demonstrating spiral wave formation via wave break at obstacle.
    """

    def __init__(
        self,
        domain_size: float = 60.0,
        resolution: float = 0.5,
        D_parallel: float = 0.1,
        D_perp: float = 0.05,
    ):
        """
        Initialize spiral wave simulation.

        Parameters
        ----------
        domain_size : float
            Square domain side [mm]
        resolution : float
            Grid spacing [mm]
        D_parallel, D_perp : float
            Diffusion coefficients [mm²/ms]
        """
        self.domain_size = domain_size
        self.resolution = resolution

        # Spatial parameters
        self.spatial = SpatialParams(
            Lx=domain_size,
            Ly=domain_size,
            dx=resolution,
            dy=resolution,
            D_parallel=D_parallel,
            D_perp=D_perp,
            fiber_angle=0.0,  # Horizontal fibers
        )

        # Grid
        self.ny = self.spatial.ny
        self.nx = self.spatial.nx
        self.x = np.linspace(0, domain_size, self.nx)
        self.y = np.linspace(0, domain_size, self.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Fenton-Karma model
        self.fk_params = get_paramset_apd250()
        self.physical = PhysicalConstants()
        self.dt = 0.02  # ms

        self.ionic = FentonKarmaModel(
            params=self.fk_params,
            physical=self.physical,
            dt=self.dt,
        )

        # Pre-scaled diffusion coefficients
        Dxx, Dyy, _ = self.spatial.diffusion_tensor()
        self.Dxx_dt_dx2 = Dxx * self.dt / (self.spatial.dx ** 2)
        self.Dyy_dt_dy2 = Dyy * self.dt / (self.spatial.dy ** 2)

        # State
        self.state = None
        self.time = 0.0

        # Infarct mask (1 = normal, 0 = infarct)
        self.mask = np.ones((self.ny, self.nx), dtype=np.float64)

        # Stimulus current
        self.I_stim = np.zeros((self.ny, self.nx), dtype=np.float64)

        # Triangle vertices for visualization
        self.triangle_vertices = None

        # Initialize
        self.reset()

        print(f"SpiralWaveSimulation initialized:")
        print(f"  Domain: {domain_size} × {domain_size} mm")
        print(f"  Grid: {self.ny} × {self.nx}")
        print(f"  dt = {self.dt} ms")

    def reset(self):
        """Reset to initial resting state."""
        self.state = self.ionic.initialize_state((self.ny, self.nx))
        self.time = 0.0
        self.I_stim.fill(0.0)

    def create_triangle_infarct(
        self,
        x_center: float,
        y_center: float,
        size: float,
        pointing: str = 'left',
    ):
        """
        Create triangular infarct region.

        Parameters
        ----------
        x_center, y_center : float
            Center of triangle [mm]
        size : float
            Triangle size (height and base width) [mm]
        pointing : str
            Direction the tip points: 'left', 'right', 'up', 'down'
        """
        half = size / 2

        # Define triangle vertices based on direction
        if pointing == 'left':
            # Tip points left (into incoming wave from left)
            # Vertex 1: left tip
            # Vertex 2: top-right corner
            # Vertex 3: bottom-right corner
            v1 = (x_center - half, y_center)           # Left tip
            v2 = (x_center + half, y_center + half)    # Top-right
            v3 = (x_center + half, y_center - half)    # Bottom-right
        elif pointing == 'right':
            v1 = (x_center + half, y_center)           # Right tip
            v2 = (x_center - half, y_center + half)    # Top-left
            v3 = (x_center - half, y_center - half)    # Bottom-left
        elif pointing == 'up':
            v1 = (x_center, y_center + half)           # Top tip
            v2 = (x_center - half, y_center - half)    # Bottom-left
            v3 = (x_center + half, y_center - half)    # Bottom-right
        elif pointing == 'down':
            v1 = (x_center, y_center - half)           # Bottom tip
            v2 = (x_center - half, y_center + half)    # Top-left
            v3 = (x_center + half, y_center + half)    # Top-right
        else:
            raise ValueError(f"Unknown direction: {pointing}")

        self.triangle_vertices = [v1, v2, v3]

        # Check if each grid point is inside the triangle
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        def point_in_triangle(pt, v1, v2, v3):
            d1 = sign(pt, v1, v2)
            d2 = sign(pt, v2, v3)
            d3 = sign(pt, v3, v1)
            has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
            has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
            return not (has_neg and has_pos)

        # Create mask
        n_infarct = 0
        for i in range(self.ny):
            for j in range(self.nx):
                pt = (self.x[j], self.y[i])
                if point_in_triangle(pt, v1, v2, v3):
                    self.mask[i, j] = 0.0
                    self.state['u'][i, j] = 0.0
                    self.state['v'][i, j] = 1.0
                    self.state['w'][i, j] = 1.0
                    n_infarct += 1

        print(f"Created triangle infarct pointing {pointing}:")
        print(f"  Center: ({x_center}, {y_center}) mm")
        print(f"  Size: {size} mm")
        print(f"  Vertices: {v1}, {v2}, {v3}")
        print(f"  Infarct cells: {n_infarct} ({n_infarct/self.mask.size*100:.1f}%)")

    def apply_planar_stimulus(
        self,
        edge: str = 'left',
        width: float = 3.0,
        amplitude: float = 1.0,
    ):
        """
        Apply planar wave stimulus at domain edge.

        Parameters
        ----------
        edge : str
            'left', 'right', 'bottom', 'top'
        width : float
            Stimulus strip width [mm]
        amplitude : float
            Stimulus amplitude (dimensionless)
        """
        if edge == 'left':
            region = self.X < width
        elif edge == 'right':
            region = self.X > (self.domain_size - width)
        elif edge == 'bottom':
            region = self.Y < width
        elif edge == 'top':
            region = self.Y > (self.domain_size - width)
        else:
            raise ValueError(f"Unknown edge: {edge}")

        # Only stimulate normal tissue
        region = region & (self.mask > 0.5)
        self.I_stim[region] = amplitude

        print(f"Planar stimulus applied: {edge} edge, width={width} mm")

    def clear_stimulus(self):
        """Turn off stimulus."""
        self.I_stim.fill(0.0)

    def step(self):
        """Perform one simulation step."""
        # Diffusion with infarct mask
        u_new = diffusion_with_infarct_kernel(
            self.state['u'],
            self.mask,
            self.Dxx_dt_dx2,
            self.Dyy_dt_dy2,
        )
        self.state['u'] = u_new

        # Ionic model step
        self.ionic.ionic_step(self.state, self.I_stim)

        # Enforce infarct remains at rest
        infarct = self.mask < 0.5
        self.state['u'][infarct] = 0.0
        self.state['v'][infarct] = 1.0
        self.state['w'][infarct] = 1.0

        self.time += self.dt

    def run(self, duration: float, verbose: bool = True):
        """Run simulation for specified duration."""
        n_steps = int(duration / self.dt)
        if verbose:
            print(f"Running {duration} ms ({n_steps} steps)...")

        for _ in range(n_steps):
            self.step()

        if verbose:
            print(f"  Time: {self.time:.1f} ms")
            u = self.state['u']
            print(f"  u range: [{np.min(u):.3f}, {np.max(u):.3f}]")

    def voltage_to_mV(self, u=None):
        """Convert dimensionless voltage to mV."""
        if u is None:
            u = self.state['u']
        return self.physical.to_physical(u)


# =============================================================================
# Interactive Demo
# =============================================================================

class SpiralWaveDemo:
    """
    Interactive demonstration of spiral wave formation.

    Single planar wave from left hits a left-pointing triangle,
    causing wave break and spiral formation.
    """

    def __init__(self):
        # Create simulation
        self.sim = SpiralWaveSimulation(
            domain_size=60.0,
            resolution=0.5,
            D_parallel=0.1,
            D_perp=0.05,
        )

        # Create left-pointing triangle infarct, centered, 20mm size
        self.sim.create_triangle_infarct(
            x_center=self.sim.domain_size / 2,  # 30 mm
            y_center=self.sim.domain_size / 2,  # 30 mm
            size=20.0,                          # 20 mm
            pointing='left',
        )

        # Animation state
        self.running = False
        self.animation = None
        self.frame_skip = 10  # Steps per frame

        # Stimulus state
        self.stimulus_applied = False
        self.stim_start = 10.0    # ms
        self.stim_duration = 3.0  # ms

        # Setup figure
        self._setup_figure()

    def _setup_figure(self):
        """Setup matplotlib figure with controls."""
        self.fig = plt.figure(figsize=(14, 8))

        # Main voltage display
        self.ax_main = self.fig.add_axes([0.05, 0.20, 0.55, 0.75])

        # Colorbar
        self.ax_cbar = self.fig.add_axes([0.62, 0.20, 0.02, 0.75])

        # Info panel
        self.ax_info = self.fig.add_axes([0.70, 0.45, 0.27, 0.50])
        self.ax_info.axis('off')

        # Description panel
        self.ax_desc = self.fig.add_axes([0.70, 0.20, 0.27, 0.22])
        self.ax_desc.axis('off')

        # Initialize voltage display
        V_mV = self.sim.voltage_to_mV()
        self.im = self.ax_main.imshow(
            V_mV,
            origin='lower',
            extent=[0, self.sim.domain_size, 0, self.sim.domain_size],
            cmap='turbo',
            vmin=-85,
            vmax=40,
            aspect='equal',
        )

        # Overlay infarct as gray region
        infarct_overlay = np.ma.masked_where(
            self.sim.mask > 0.5,
            np.ones_like(self.sim.mask)
        )
        self.infarct_im = self.ax_main.imshow(
            infarct_overlay,
            origin='lower',
            extent=[0, self.sim.domain_size, 0, self.sim.domain_size],
            cmap='gray',
            vmin=0, vmax=1,
            alpha=0.8,
            aspect='equal',
        )

        # Draw triangle outline
        if self.sim.triangle_vertices:
            tri_patch = Polygon(
                self.sim.triangle_vertices,
                fill=False,
                edgecolor='white',
                linewidth=2,
                linestyle='--',
            )
            self.ax_main.add_patch(tri_patch)

        # Add wave direction arrow
        arrow_y = self.sim.domain_size * 0.9
        self.ax_main.annotate(
            '', xy=(12, arrow_y), xytext=(3, arrow_y),
            arrowprops=dict(arrowstyle='->', color='yellow', lw=3)
        )
        self.ax_main.text(
            7.5, arrow_y + 2, 'Wave',
            color='yellow', fontsize=11, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5)
        )

        # Colorbar
        self.cbar = self.fig.colorbar(self.im, cax=self.ax_cbar)
        self.cbar.set_label('Voltage [mV]', fontsize=11)

        # Labels
        self.ax_main.set_xlabel('x [mm]', fontsize=12)
        self.ax_main.set_ylabel('y [mm]', fontsize=12)
        self.ax_main.set_title(
            'Spiral Wave Formation: Wave Break at Triangle Obstacle',
            fontsize=14
        )

        # Info text
        self.info_text = self.ax_info.text(
            0.05, 0.95, '', transform=self.ax_info.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

        # Description
        desc = (
            "SPIRAL FORMATION MECHANISM\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "1. Planar wave from left\n"
            "2. Wave hits triangle tip\n"
            "3. Wave splits around obstacle\n"
            "4. Wave fronts wrap around\n"
            "5. Spirals form at corners\n"
            "\n"
            "Triangle: 20mm, pointing LEFT\n"
            "(tip faces incoming wave)"
        )
        self.ax_desc.text(
            0.05, 0.95, desc, transform=self.ax_desc.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        )

        # === CONTROLS ===

        # Start button
        self.ax_btn_start = self.fig.add_axes([0.20, 0.05, 0.12, 0.05])
        self.btn_start = Button(self.ax_btn_start, 'Start', color='lightgreen')
        self.btn_start.on_clicked(self._on_start)

        # Pause button
        self.ax_btn_pause = self.fig.add_axes([0.35, 0.05, 0.12, 0.05])
        self.btn_pause = Button(self.ax_btn_pause, 'Pause', color='lightyellow')
        self.btn_pause.on_clicked(self._on_pause)

        # Reset button
        self.ax_btn_reset = self.fig.add_axes([0.50, 0.05, 0.12, 0.05])
        self.btn_reset = Button(self.ax_btn_reset, 'Reset', color='lightcoral')
        self.btn_reset.on_clicked(self._on_reset)

        # Status text
        self.ax_status = self.fig.add_axes([0.05, 0.12, 0.60, 0.04])
        self.ax_status.axis('off')
        self.status_text = self.ax_status.text(
            0.5, 0.5,
            'Ready - Click "Start" to launch planar wave from left edge',
            transform=self.ax_status.transAxes,
            fontsize=11, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8)
        )

    def _on_start(self, event):
        """Start simulation."""
        if self.running:
            return

        self.running = True
        self.btn_start.label.set_text('Running...')
        self.btn_start.color = 'gray'

        # Start animation
        if self.animation is None:
            self.animation = FuncAnimation(
                self.fig, self._animate_frame,
                interval=20,  # ~50 fps display
                blit=False,
                cache_frame_data=False,
            )

        self.status_text.set_text('Simulation running...')
        self.fig.canvas.draw_idle()

    def _on_pause(self, event):
        """Toggle pause."""
        self.running = not self.running
        if self.running:
            self.btn_pause.label.set_text('Pause')
            self.status_text.set_text(f't = {self.sim.time:.0f} ms - Running')
        else:
            self.btn_pause.label.set_text('Resume')
            self.status_text.set_text(f't = {self.sim.time:.0f} ms - Paused')
        self.fig.canvas.draw_idle()

    def _on_reset(self, event):
        """Reset simulation."""
        self.running = False
        self.stimulus_applied = False

        # Reset simulation
        self.sim.reset()
        self.sim.create_triangle_infarct(
            x_center=self.sim.domain_size / 2,
            y_center=self.sim.domain_size / 2,
            size=20.0,
            pointing='left',
        )

        # Update display
        self._update_display()
        self.btn_start.label.set_text('Start')
        self.btn_start.color = 'lightgreen'
        self.btn_pause.label.set_text('Pause')
        self.status_text.set_text('Reset - Ready')
        self.fig.canvas.draw_idle()

    def _animate_frame(self, frame):
        """Animation update."""
        if not self.running:
            return

        # Apply stimulus at t=10 ms
        if self.sim.time >= self.stim_start and not self.stimulus_applied:
            self.sim.apply_planar_stimulus('left', width=3.0, amplitude=1.0)
            self.stimulus_applied = True
            self.status_text.set_text('Planar wave launched from left edge')

        # Turn off stimulus after duration
        if self.stimulus_applied and self.sim.time >= self.stim_start + self.stim_duration:
            self.sim.clear_stimulus()

        # Run simulation steps
        for _ in range(self.frame_skip):
            self.sim.step()

        # Update display
        self._update_display()

        # Update status
        u = self.sim.state['u']
        activated = np.sum((u > 0.5) & (self.sim.mask > 0.5))
        total = np.sum(self.sim.mask > 0.5)
        pct = activated / total * 100 if total > 0 else 0

        if self.sim.time > 100:
            self.status_text.set_text(
                f't = {self.sim.time:.0f} ms | '
                f'Activated: {pct:.1f}% | '
                f'Watch for spiral formation at triangle corners'
            )

    def _update_display(self):
        """Update voltage image and info."""
        V_mV = self.sim.voltage_to_mV()
        self.im.set_data(V_mV)

        # Update info
        u = self.sim.state['u']
        v = self.sim.state['v']
        w = self.sim.state['w']

        activated = np.sum((u > 0.5) & (self.sim.mask > 0.5))
        total_cells = np.sum(self.sim.mask > 0.5)

        info = (
            f"Time: {self.sim.time:.1f} ms\n"
            f"\n"
            f"Voltage (u):\n"
            f"  Min: {np.min(u):.3f}\n"
            f"  Max: {np.max(u):.3f}\n"
            f"\n"
            f"Activated: {activated}/{total_cells}\n"
            f"           ({activated/total_cells*100:.1f}%)\n"
            f"\n"
            f"Gating Variables:\n"
            f"  v: [{np.min(v):.3f}, {np.max(v):.3f}]\n"
            f"  w: [{np.min(w):.3f}, {np.max(w):.3f}]\n"
            f"\n"
            f"Infarct: Triangle 20mm\n"
            f"         Pointing LEFT"
        )
        self.info_text.set_text(info)

    def run(self):
        """Start interactive demo."""
        print("=" * 60)
        print("SPIRAL WAVE FORMATION DEMO")
        print("=" * 60)
        print("\nMechanism:")
        print("  1. Uniform planar wave from left edge")
        print("  2. Wave hits left-pointing triangle tip")
        print("  3. Wave breaks around obstacle")
        print("  4. Spiral waves form at triangle corners")
        print("\nTriangle Configuration:")
        print("  - Position: Centered (30mm, 30mm)")
        print("  - Size: 20mm (large)")
        print("  - Orientation: Pointing LEFT (tip faces wave)")
        print("\nControls:")
        print("  - Start: Launch simulation")
        print("  - Pause/Resume: Toggle running")
        print("  - Reset: Return to initial state")
        print("=" * 60)

        self._update_display()
        plt.show()


# =============================================================================
# Main
# =============================================================================

def main():
    """Run spiral wave demo."""
    demo = SpiralWaveDemo()
    demo.run()


if __name__ == "__main__":
    main()
