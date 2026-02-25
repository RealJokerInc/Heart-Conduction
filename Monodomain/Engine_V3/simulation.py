"""
2D Cardiac Simulation Module
============================

Combines Fenton-Karma ionic model with diffusion for 2D wave propagation.

Uses operator splitting:
1. Diffusion step: u_temp = u + dt * ∇·(D∇u)
2. Ionic step: u_new = u_temp + dt * (-J_fi - J_so - J_si + I_stim)

References:
- Fenton & Karma 1998, Chaos 8:20-47
- Sundnes et al., Computing the Electrical Activity in the Heart

Author: Generated with Claude Code
Date: 2025-12-10
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Optional, Callable, List
from dataclasses import dataclass
import time

from parameters import (
    FKParams, SpatialParams, PhysicalConstants,
    default_fk_params, default_spatial_params, default_physical_constants
)
from fenton_karma import FentonKarmaModel
from diffusion import DiffusionOperator


# =============================================================================
# Stimulus Protocol
# =============================================================================

@dataclass
class Stimulus:
    """
    Stimulus pulse specification.

    Attributes
    ----------
    x_center : float
        Center x coordinate [mm]
    y_center : float
        Center y coordinate [mm]
    radius : float
        Stimulus radius [mm]
    amplitude : float
        Stimulus amplitude (dimensionless, ~0.5 typical)
    t_start : float
        Stimulus start time [ms]
    duration : float
        Stimulus duration [ms]
    """
    x_center: float
    y_center: float
    radius: float = 3.0
    amplitude: float = 0.5
    t_start: float = 0.0
    duration: float = 2.0


# =============================================================================
# 2D Simulation Class
# =============================================================================

class CardiacSimulation2D:
    """
    2D cardiac tissue simulation combining FK ionic model and diffusion.

    Operator splitting approach:
    1. Apply diffusion to voltage
    2. Apply ionic model to all state variables
    """

    def __init__(
        self,
        fk_params: FKParams = None,
        spatial_params: SpatialParams = None,
        physical: PhysicalConstants = None,
        dt: float = 0.02,
    ):
        """
        Initialize 2D simulation.

        Parameters
        ----------
        fk_params : FKParams
            Fenton-Karma model parameters
        spatial_params : SpatialParams
            Spatial domain configuration
        physical : PhysicalConstants
            Voltage conversion constants
        dt : float
            Time step [ms]
        """
        self.fk_params = fk_params or default_fk_params()
        self.spatial = spatial_params or default_spatial_params()
        self.physical = physical or default_physical_constants()
        self.dt = dt

        # Create components
        self.ionic_model = FentonKarmaModel(
            params=self.fk_params,
            physical=self.physical,
            dt=dt
        )

        self.diffusion = DiffusionOperator(
            spatial=self.spatial,
            dt=dt
        )

        # Grid dimensions
        self.ny = self.spatial.ny
        self.nx = self.spatial.nx

        # Physical coordinates
        self.x = np.linspace(0, self.spatial.Lx, self.nx)
        self.y = np.linspace(0, self.spatial.Ly, self.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # State variables
        self.state = None
        self.I_stim = None
        self.time = 0.0
        self.step_count = 0

        # Stimulus protocol
        self.stimuli: List[Stimulus] = []

        # Initialize
        self.reset()

        print(f"CardiacSimulation2D initialized:")
        print(f"  Grid: {self.ny} × {self.nx} ({self.ny * self.nx:,} points)")
        print(f"  Domain: {self.spatial.Lx} × {self.spatial.Ly} mm")
        print(f"  dt = {self.dt} ms")
        print(f"  D_parallel = {self.spatial.D_parallel} mm²/ms")
        print(f"  D_perp = {self.spatial.D_perp} mm²/ms")

    def reset(self) -> None:
        """Reset simulation to initial state (rest)."""
        self.state = self.ionic_model.initialize_state((self.ny, self.nx))
        self.I_stim = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.time = 0.0
        self.step_count = 0

    def add_stimulus(self, stim: Stimulus) -> None:
        """Add a stimulus to the protocol."""
        self.stimuli.append(stim)

    def add_point_stimulus(
        self,
        x: float,
        y: float,
        t_start: float,
        amplitude: float = 0.5,
        radius: float = 3.0,
        duration: float = 2.0,
    ) -> None:
        """
        Convenience method to add a point stimulus.

        Parameters
        ----------
        x, y : float
            Stimulus center [mm]
        t_start : float
            Start time [ms]
        amplitude : float
            Stimulus amplitude
        radius : float
            Stimulus radius [mm]
        duration : float
            Stimulus duration [ms]
        """
        self.add_stimulus(Stimulus(
            x_center=x,
            y_center=y,
            radius=radius,
            amplitude=amplitude,
            t_start=t_start,
            duration=duration
        ))

    def add_edge_stimulus(
        self,
        edge: str = 'left',
        t_start: float = 0.0,
        amplitude: float = 0.5,
        width: float = 3.0,
        duration: float = 2.0,
    ) -> None:
        """
        Add stimulus along an edge.

        Parameters
        ----------
        edge : str
            'left', 'right', 'bottom', or 'top'
        t_start : float
            Start time [ms]
        amplitude : float
            Stimulus amplitude
        width : float
            Width of stimulus region [mm]
        duration : float
            Stimulus duration [ms]
        """
        Lx, Ly = self.spatial.Lx, self.spatial.Ly

        if edge == 'left':
            x, y = width / 2, Ly / 2
            radius = Ly  # Cover full height
        elif edge == 'right':
            x, y = Lx - width / 2, Ly / 2
            radius = Ly
        elif edge == 'bottom':
            x, y = Lx / 2, width / 2
            radius = Lx
        elif edge == 'top':
            x, y = Lx / 2, Ly - width / 2
            radius = Lx
        else:
            raise ValueError(f"Unknown edge: {edge}")

        # Use rectangular stimulus instead
        self.stimuli.append(Stimulus(
            x_center=x,
            y_center=y,
            radius=radius,
            amplitude=amplitude,
            t_start=t_start,
            duration=duration
        ))

    def _compute_stimulus(self) -> None:
        """Compute stimulus current for current time."""
        self.I_stim.fill(0.0)

        for stim in self.stimuli:
            if stim.t_start <= self.time < stim.t_start + stim.duration:
                # Gaussian stimulus
                dist2 = (self.X - stim.x_center)**2 + (self.Y - stim.y_center)**2
                self.I_stim += stim.amplitude * np.exp(-dist2 / (2 * stim.radius**2))

    def step(self) -> None:
        """
        Perform one simulation step using operator splitting.

        1. Compute stimulus
        2. Apply diffusion to voltage
        3. Apply ionic model
        """
        # Stimulus
        self._compute_stimulus()

        # Operator splitting:
        # Step 1: Diffusion (only affects voltage u)
        u_after_diffusion = self.diffusion.step(
            self.state['u'], anisotropic=True
        )
        self.state['u'] = u_after_diffusion

        # Step 2: Ionic model (affects u, v, w)
        self.ionic_model.ionic_step(self.state, self.I_stim)

        # Update time
        self.time += self.dt
        self.step_count += 1

    def run(
        self,
        t_end: float,
        callback: Optional[Callable] = None,
        callback_interval: float = 1.0,
        verbose: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Run simulation to specified end time.

        Parameters
        ----------
        t_end : float
            End time [ms]
        callback : callable, optional
            Function called periodically: callback(sim, t)
        callback_interval : float
            Callback interval [ms]
        verbose : bool
            Print progress

        Returns
        -------
        result : dict
            {'t': time_array, 'u': final_voltage, ...}
        """
        n_steps = int(np.ceil((t_end - self.time) / self.dt))
        callback_steps = int(callback_interval / self.dt)

        if verbose:
            print(f"Running simulation: {self.time:.1f} → {t_end:.1f} ms ({n_steps} steps)")

        t_start = time.perf_counter()

        for i in range(n_steps):
            self.step()

            if callback and (i % callback_steps == 0):
                callback(self, self.time)

        elapsed = time.perf_counter() - t_start

        if verbose:
            print(f"Completed in {elapsed:.2f}s ({n_steps/elapsed:.0f} steps/s)")
            print(f"  Final u range: [{np.min(self.state['u']):.4f}, {np.max(self.state['u']):.4f}]")

        return {
            't': self.time,
            'u': self.state['u'].copy(),
            'v': self.state['v'].copy(),
            'w': self.state['w'].copy(),
        }

    def voltage_to_mV(self, u: np.ndarray = None) -> np.ndarray:
        """Convert voltage to physical mV."""
        if u is None:
            u = self.state['u']
        return self.physical.to_physical(u)

    def measure_activation_times(
        self,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Measure activation time at each grid point.

        Requires running simulation with recording enabled.

        Parameters
        ----------
        threshold : float
            Activation threshold (dimensionless)

        Returns
        -------
        t_act : np.ndarray
            Activation times [ms], NaN where not activated
        """
        # This is a simplified version - full implementation would
        # record u over time
        raise NotImplementedError("Use run_with_recording for activation times")

    def summary(self) -> str:
        """Generate simulation summary."""
        lines = [
            "=" * 60,
            "CARDIAC SIMULATION 2D",
            "=" * 60,
            "",
            "Grid:",
            f"  {self.ny} × {self.nx} points ({self.ny * self.nx:,} total)",
            f"  Domain: {self.spatial.Lx} × {self.spatial.Ly} mm",
            f"  Resolution: {self.spatial.dx} mm",
            "",
            "Time:",
            f"  dt = {self.dt} ms",
            f"  Current time = {self.time:.2f} ms",
            f"  Steps = {self.step_count}",
            "",
            "Diffusion:",
            f"  D_parallel = {self.spatial.D_parallel} mm²/ms",
            f"  D_perp = {self.spatial.D_perp} mm²/ms",
            f"  Anisotropy = {self.spatial.anisotropy_ratio:.1f}:1",
            f"  CFL = {self.diffusion.cfl_number:.4f}",
            "",
            "State:",
            f"  u range: [{np.min(self.state['u']):.4f}, {np.max(self.state['u']):.4f}]",
            f"  v range: [{np.min(self.state['v']):.4f}, {np.max(self.state['v']):.4f}]",
            f"  w range: [{np.min(self.state['w']):.4f}, {np.max(self.state['w']):.4f}]",
            "",
            f"Stimuli: {len(self.stimuli)}",
            "=" * 60,
        ]
        return "\n".join(lines)


# =============================================================================
# Factory Function
# =============================================================================

def create_simulation(
    domain_size: float = 80.0,
    resolution: float = 0.5,
    D_parallel: float = 0.1,   # Tuned for CV ≈ 500 mm/s
    D_perp: float = 0.05,      # 2:1 anisotropy
    fiber_angle: float = 0.0,
    dt: float = 0.02,
    param_preset: str = 'apd250',
) -> CardiacSimulation2D:
    """
    Create simulation with specified parameters.

    Parameters
    ----------
    domain_size : float
        Square domain side [mm]
    resolution : float
        Grid spacing [mm]
    D_parallel : float
        Diffusion along fibers [mm²/ms]
    D_perp : float
        Diffusion perpendicular [mm²/ms]
    fiber_angle : float
        Fiber orientation [degrees]
    dt : float
        Time step [ms]
    param_preset : str
        FK parameter preset name

    Returns
    -------
    sim : CardiacSimulation2D
        Configured simulation
    """
    from parameters import get_preset

    spatial = SpatialParams(
        Lx=domain_size,
        Ly=domain_size,
        dx=resolution,
        dy=resolution,
        D_parallel=D_parallel,
        D_perp=D_perp,
        fiber_angle=fiber_angle,
    )

    fk_params = get_preset(param_preset)

    return CardiacSimulation2D(
        fk_params=fk_params,
        spatial_params=spatial,
        dt=dt,
    )


# =============================================================================
# Test: Wave Propagation
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    print("=" * 60)
    print("2D WAVE PROPAGATION TEST")
    print("=" * 60)

    # Create simulation
    sim = create_simulation(
        domain_size=40.0,
        resolution=0.5,
        D_parallel=1.0,
        D_perp=0.5,
        dt=0.02,
        param_preset='apd250',
    )

    print(sim.summary())

    # Add left-edge stimulus
    # Using a strip along the left edge
    for y in np.linspace(5, 35, 7):
        sim.add_point_stimulus(
            x=3.0,
            y=y,
            t_start=10.0,
            amplitude=1.0,
            radius=3.0,
            duration=3.0,
        )

    print(f"\nAdded {len(sim.stimuli)} stimulus points along left edge")

    # Record snapshots
    snapshots = []
    snapshot_times = [0, 10, 15, 20, 30, 50, 75, 100]

    def record_snapshot(sim, t):
        """Callback to record snapshots."""
        for snap_t in snapshot_times:
            if abs(t - snap_t) < sim.dt:
                snapshots.append({
                    't': t,
                    'u': sim.state['u'].copy(),
                })
                print(f"  Snapshot at t = {t:.1f} ms, u_max = {np.max(sim.state['u']):.3f}")
                break

    # Run simulation
    print("\nRunning simulation...")
    result = sim.run(
        t_end=120.0,
        callback=record_snapshot,
        callback_interval=1.0,
        verbose=True,
    )

    # Plot snapshots
    n_snaps = len(snapshots)
    if n_snaps > 0:
        cols = min(4, n_snaps)
        rows = (n_snaps + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()

        for idx, snap in enumerate(snapshots):
            ax = axes[idx]
            V_mV = sim.voltage_to_mV(snap['u'])
            im = ax.imshow(
                V_mV,
                origin='lower',
                extent=[0, sim.spatial.Lx, 0, sim.spatial.Ly],
                cmap='turbo',
                vmin=-85,
                vmax=40,
            )
            ax.set_title(f't = {snap["t"]:.0f} ms', fontsize=12)
            ax.set_xlabel('x [mm]')
            ax.set_ylabel('y [mm]')
            plt.colorbar(im, ax=ax, label='V [mV]')

        # Hide unused axes
        for idx in range(len(snapshots), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig('images/fk_2d_wave_propagation.png', dpi=150)
        print(f"\nPlot saved: images/fk_2d_wave_propagation.png")

    # Measure conduction velocity
    print("\n" + "-" * 40)
    print("Conduction Velocity Estimate")
    print("-" * 40)

    # Find wave front position at different times
    if len(snapshots) >= 2:
        # Take two snapshots where wave is propagating
        early_snap = next((s for s in snapshots if s['t'] > 15), None)
        late_snap = next((s for s in snapshots if s['t'] > 50), None)

        if early_snap and late_snap:
            threshold = 0.5  # Dimensionless threshold

            # Find x-position of wave front (max activated x along centerline)
            j_center = sim.ny // 2

            u_early = early_snap['u'][j_center, :]
            u_late = late_snap['u'][j_center, :]

            # Find first point above threshold from left
            x_early = None
            for i, u_val in enumerate(u_early):
                if u_val > threshold:
                    x_early = sim.x[i]
                    break

            x_late = None
            for i in range(len(u_late) - 1, -1, -1):
                if u_late[i] > threshold:
                    x_late = sim.x[i]
                    break

            if x_early is not None and x_late is not None:
                dx = x_late - x_early
                dt_snap = late_snap['t'] - early_snap['t']
                cv = dx / dt_snap

                print(f"Wave front at t={early_snap['t']:.0f}ms: x ≈ {x_early:.1f} mm")
                print(f"Wave front at t={late_snap['t']:.0f}ms: x ≈ {x_late:.1f} mm")
                print(f"Δx = {dx:.1f} mm, Δt = {dt_snap:.1f} ms")
                print(f"CV ≈ {cv:.3f} mm/ms = {cv*1000:.1f} mm/s")
                print(f"\nTarget CV: 0.5 mm/ms = 500 mm/s")

    plt.show()

    print("\n2D wave propagation test complete!")
