"""
2D Cardiac Simulation Module (LRd94)
====================================

Combines Luo-Rudy 1994 ionic model with diffusion for 2D wave propagation.

Uses operator splitting:
1. Diffusion step: V_temp = V + dt * div(D*grad(V))
2. Ionic step: Update V, gates, and concentrations

The monodomain equation:
    Cm * dV/dt = div(D*grad(V)) - I_ion

Where:
- Cm: membrane capacitance [uF/cm^2]
- D: diffusion tensor [mm^2/ms]
- I_ion: total ionic current [uA/cm^2]

References:
- Luo CH, Rudy Y. Circ Res. 1994;74(6):1071-1096.
- Sundnes et al., Computing the Electrical Activity in the Heart

Author: Generated with Claude Code
Date: 2025-12-11
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Optional, Callable, List
from dataclasses import dataclass
import time

from parameters import (
    LRd94Params, SpatialParams, PhysicalConstants, CellGeometry,
    LRd94InitialConditions,
    default_params, default_spatial_params, default_physical_constants,
    default_cell_geometry, default_initial_conditions,
    STATE_NAMES, N_STATES
)
from luo_rudy_1994 import LuoRudy1994Model
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
        Stimulus amplitude [uA/cm^2] - negative for depolarizing!
    t_start : float
        Stimulus start time [ms]
    duration : float
        Stimulus duration [ms]
    """
    x_center: float
    y_center: float
    radius: float = 3.0
    amplitude: float = -80.0  # Negative for depolarizing stimulus
    t_start: float = 0.0
    duration: float = 1.0


# =============================================================================
# 2D Simulation Class
# =============================================================================

class CardiacSimulation2D:
    """
    2D cardiac tissue simulation with LRd94 ionic model.

    Operator splitting approach:
    1. Apply diffusion to voltage
    2. Apply ionic model to all state variables
    """

    def __init__(
        self,
        lrd_params: LRd94Params = None,
        spatial_params: SpatialParams = None,
        initial_conditions: LRd94InitialConditions = None,
        dt: float = 0.005,
    ):
        """
        Initialize 2D simulation.

        Parameters
        ----------
        lrd_params : LRd94Params
            Luo-Rudy 1994 model parameters
        spatial_params : SpatialParams
            Spatial domain configuration
        initial_conditions : LRd94InitialConditions
            Initial state values
        dt : float
            Time step [ms] - default 0.005 for LRd94 stability
        """
        self.lrd_params = lrd_params or default_params()
        self.spatial = spatial_params or default_spatial_params()
        self.ic = initial_conditions or default_initial_conditions()
        self.dt = dt

        # Create components
        self.ionic_model = LuoRudy1994Model(
            params=self.lrd_params,
            initial_conditions=self.ic,
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

        print(f"CardiacSimulation2D (LRd94) initialized:")
        print(f"  Grid: {self.ny} x {self.nx} ({self.ny * self.nx:,} points)")
        print(f"  Domain: {self.spatial.Lx} x {self.spatial.Ly} mm")
        print(f"  dt = {self.dt} ms")
        print(f"  D_parallel = {self.spatial.D_parallel} mm^2/ms")
        print(f"  D_perp = {self.spatial.D_perp} mm^2/ms")

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
        amplitude: float = -80.0,
        radius: float = 3.0,
        duration: float = 1.0,
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
            Stimulus amplitude [uA/cm^2] - negative for depolarizing
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
        amplitude: float = -80.0,
        width: float = 3.0,
        duration: float = 1.0,
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
            Stimulus amplitude [uA/cm^2]
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
                # Gaussian stimulus (smoother than sharp edges)
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
        # Step 1: Diffusion (only affects voltage V)
        V_after_diffusion = self.diffusion.step(
            self.state['V'], anisotropic=True
        )
        self.state['V'] = V_after_diffusion

        # Step 2: Ionic model (affects V, gates, concentrations)
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
            {'t': time, 'V': final_voltage, ...}
        """
        n_steps = int(np.ceil((t_end - self.time) / self.dt))
        callback_steps = int(callback_interval / self.dt)

        if verbose:
            print(f"Running simulation: {self.time:.1f} -> {t_end:.1f} ms ({n_steps:,} steps)")

        t_start = time.perf_counter()

        for i in range(n_steps):
            self.step()

            if callback and (i % callback_steps == 0):
                callback(self, self.time)

        elapsed = time.perf_counter() - t_start

        if verbose:
            print(f"Completed in {elapsed:.2f}s ({n_steps/elapsed:.0f} steps/s)")
            print(f"  V range: [{np.min(self.state['V']):.1f}, {np.max(self.state['V']):.1f}] mV")

        # Return final state
        result = {'t': self.time}
        for name in STATE_NAMES:
            result[name] = self.state[name].copy()

        return result

    def run_with_recording(
        self,
        t_end: float,
        record_interval: float = 1.0,
        verbose: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Run simulation and record voltage history.

        Parameters
        ----------
        t_end : float
            End time [ms]
        record_interval : float
            Recording interval [ms]
        verbose : bool
            Print progress

        Returns
        -------
        result : dict
            {'times': array, 'V_history': (n_times, ny, nx), ...}
        """
        n_steps = int(np.ceil((t_end - self.time) / self.dt))
        record_steps = int(record_interval / self.dt)
        n_records = n_steps // record_steps + 1

        # Allocate recording arrays
        times = np.zeros(n_records)
        V_history = np.zeros((n_records, self.ny, self.nx))

        if verbose:
            print(f"Running simulation with recording: {self.time:.1f} -> {t_end:.1f} ms")
            print(f"  Recording every {record_interval} ms ({n_records} frames)")

        t_start = time.perf_counter()
        record_idx = 0

        for i in range(n_steps):
            # Record
            if i % record_steps == 0:
                times[record_idx] = self.time
                V_history[record_idx] = self.state['V'].copy()
                record_idx += 1

            # Step
            self.step()

        elapsed = time.perf_counter() - t_start

        if verbose:
            print(f"Completed in {elapsed:.2f}s ({n_steps/elapsed:.0f} steps/s)")

        return {
            'times': times[:record_idx],
            'V_history': V_history[:record_idx],
        }

    def get_voltage_mV(self) -> np.ndarray:
        """Get current voltage field [mV]."""
        return self.state['V'].copy()

    def measure_activation_times(
        self,
        V_history: np.ndarray,
        times: np.ndarray,
        threshold: float = -30.0,
    ) -> np.ndarray:
        """
        Measure activation time at each grid point.

        Parameters
        ----------
        V_history : np.ndarray (n_times, ny, nx)
            Recorded voltage history
        times : np.ndarray (n_times,)
            Recording times
        threshold : float
            Activation threshold [mV]

        Returns
        -------
        t_act : np.ndarray (ny, nx)
            Activation times [ms], NaN where not activated
        """
        t_act = np.full((self.ny, self.nx), np.nan)

        for i in range(self.ny):
            for j in range(self.nx):
                V_trace = V_history[:, i, j]
                # Find first crossing above threshold
                for k in range(len(V_trace)):
                    if V_trace[k] > threshold:
                        t_act[i, j] = times[k]
                        break

        return t_act

    def summary(self) -> str:
        """Generate simulation summary."""
        lines = [
            "=" * 60,
            "CARDIAC SIMULATION 2D (LRd94)",
            "=" * 60,
            "",
            "Grid:",
            f"  {self.ny} x {self.nx} points ({self.ny * self.nx:,} total)",
            f"  Domain: {self.spatial.Lx} x {self.spatial.Ly} mm",
            f"  Resolution: {self.spatial.dx} mm",
            "",
            "Time:",
            f"  dt = {self.dt} ms",
            f"  Current time = {self.time:.2f} ms",
            f"  Steps = {self.step_count:,}",
            "",
            "Diffusion:",
            f"  D_parallel = {self.spatial.D_parallel} mm^2/ms",
            f"  D_perp = {self.spatial.D_perp} mm^2/ms",
            f"  Anisotropy = {self.spatial.anisotropy_ratio:.1f}:1",
            f"  CFL = {self.diffusion.cfl_number:.4f}",
            "",
            "Ionic Model:",
            f"  G_K = {self.lrd_params.G_K} mS/cm^2",
            f"  Target APD90 ~ 250 ms",
            "",
            "State:",
            f"  V range: [{np.min(self.state['V']):.1f}, {np.max(self.state['V']):.1f}] mV",
            f"  [Ca2+]i range: [{np.min(self.state['Ca_i'])*1e6:.0f}, {np.max(self.state['Ca_i'])*1e6:.0f}] nM",
            "",
            f"Stimuli: {len(self.stimuli)}",
            "=" * 60,
        ]
        return "\n".join(lines)


# =============================================================================
# Factory Function
# =============================================================================

def create_simulation(
    domain_size: float = 40.0,
    resolution: float = 0.5,
    D_parallel: float = 0.1,
    D_perp: float = 0.05,
    fiber_angle: float = 0.0,
    dt: float = 0.005,
    param_preset: str = 'default',
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
        Diffusion along fibers [mm^2/ms]
    D_perp : float
        Diffusion perpendicular [mm^2/ms]
    fiber_angle : float
        Fiber orientation [degrees]
    dt : float
        Time step [ms]
    param_preset : str
        LRd94 parameter preset name ('default', 'original', 'short', 'long')

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

    lrd_params = get_preset(param_preset)

    return CardiacSimulation2D(
        lrd_params=lrd_params,
        spatial_params=spatial,
        dt=dt,
    )


# =============================================================================
# Test: Wave Propagation
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("2D WAVE PROPAGATION TEST (LRd94)")
    print("=" * 60)

    # Create simulation with smaller domain for quick test
    sim = create_simulation(
        domain_size=20.0,   # Smaller domain for faster test
        resolution=0.5,
        D_parallel=0.1,
        D_perp=0.05,
        dt=0.005,
        param_preset='default',
    )

    print(sim.summary())

    # Add left-edge stimulus
    # Using multiple point stimuli along the left edge
    for y_pos in np.linspace(2, 18, 5):
        sim.add_point_stimulus(
            x=2.0,
            y=y_pos,
            t_start=5.0,
            amplitude=-100.0,  # Strong stimulus
            radius=2.0,
            duration=1.0,
        )

    print(f"\nAdded {len(sim.stimuli)} stimulus points along left edge")

    # Run short simulation
    print("\nRunning 50 ms simulation...")
    result = sim.run_with_recording(
        t_end=50.0,
        record_interval=5.0,
        verbose=True,
    )

    # Analyze results
    times = result['times']
    V_history = result['V_history']

    print(f"\nRecorded {len(times)} frames")
    for i, t in enumerate(times):
        V = V_history[i]
        activated = np.sum(V > -30) / V.size * 100
        print(f"  t={t:5.1f} ms: V=[{np.min(V):6.1f}, {np.max(V):5.1f}] mV, activated={activated:5.1f}%")

    # Measure conduction velocity
    print("\n" + "-" * 40)
    print("Conduction Velocity Estimate")
    print("-" * 40)

    # Find wave front position at different times
    threshold = -30.0  # mV
    j_center = sim.ny // 2

    # Get activation times along centerline
    x_activated = []
    t_activated = []

    for idx, t in enumerate(times):
        V_centerline = V_history[idx, j_center, :]
        # Find rightmost activated point
        for i in range(len(V_centerline) - 1, -1, -1):
            if V_centerline[i] > threshold:
                x_activated.append(sim.x[i])
                t_activated.append(t)
                break

    if len(x_activated) >= 2:
        # Simple linear fit for CV
        dx = x_activated[-1] - x_activated[0]
        dt_wave = t_activated[-1] - t_activated[0]
        if dt_wave > 0:
            cv = dx / dt_wave
            print(f"Wave front traveled {dx:.1f} mm in {dt_wave:.1f} ms")
            print(f"CV ~ {cv:.3f} mm/ms = {cv*1000:.1f} mm/s")
        else:
            print("Not enough propagation to measure CV")
    else:
        print("Wave did not propagate - check stimulus parameters")

    print("\n2D wave propagation test complete!")
