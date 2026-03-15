"""
Monodomain Tissue Simulation

Implements the monodomain equation with operator splitting:
    chi * Cm * dV/dt = -chi * Iion(V, u) + div(D * grad(V)) + Istim

Operator splitting separates:
1. Ionic step: dV/dt = -Iion/Cm (Rush-Larsen for gates)
2. Diffusion step: dV/dt = div(D*grad(V)) / (chi * Cm)

Supports Godunov (1st order) and Strang (2nd order) splitting.
"""

import numpy as np
from numba import njit, prange
from typing import Optional, Tuple, Callable, Union
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ionic import ORdModel, StateIndex, CellType
from tissue.diffusion import (
    DiffusionOperator, get_diffusion_params,
    CHI, CM, CV_LONGITUDINAL_DEFAULT, CV_TRANSVERSE_DEFAULT
)
from solvers.cpu_kernel import ionic_step_tissue, get_kernel_params


@njit(parallel=True, cache=True)
def ionic_step_kernel(states, dt, stim_mask, stim_amplitude, ionic_func):
    """
    Advance ionic model at all grid points (parallelized).

    Note: This is a placeholder - actual implementation calls model.step()
    which cannot be fully JIT-compiled due to class methods.
    """
    ny, nx, n_states = states.shape

    for i in prange(ny):
        for j in range(nx):
            Istim = stim_amplitude if stim_mask[i, j] else 0.0
            # This would call the ionic step function
            # states[i, j, :] = ionic_func(states[i, j, :], dt, Istim)

    return states


class MonodomainSimulation:
    """
    2D Monodomain tissue simulation.

    Supports two parameter modes:
    1. CV-based (recommended): Specify target conduction velocities
    2. Direct D: Specify diffusion coefficients directly

    Parameters
    ----------
    ny, nx : int
        Grid dimensions (number of cells)
    dx, dy : float
        Grid spacing in cm (default 0.01 cm = 100 um)
    cv_long : float, optional
        Target longitudinal CV (cm/ms). Default 0.06 (0.6 m/s).
        If specified, D_L is computed automatically.
    cv_trans : float, optional
        Target transverse CV (cm/ms). Default 0.02 (0.2 m/s).
        If specified, D_T is computed automatically.
    D_L : float, optional
        Longitudinal diffusion coefficient (cm^2/ms).
        Only used if cv_long is None.
    D_T : float, optional
        Transverse diffusion coefficient (cm^2/ms).
        Only used if cv_trans is None.
    fiber_angle : float or ndarray
        Fiber orientation angle in radians
    celltype : CellType
        Cell type (ENDO, EPI, or M_CELL)
    chi : float
        Surface-to-volume ratio (cm^-1), default 1400
    Cm : float
        Membrane capacitance (uF/cm^2), default 1.0
    splitting : str
        Operator splitting method: 'godunov' (1st order) or 'strang' (2nd order)
    """

    def __init__(
        self,
        ny: int,
        nx: int,
        dx: float = 0.01,
        dy: float = 0.01,
        cv_long: Optional[float] = None,
        cv_trans: Optional[float] = None,
        D_L: Optional[float] = None,
        D_T: Optional[float] = None,
        fiber_angle: float = 0.0,
        celltype: CellType = CellType.ENDO,
        chi: float = CHI,
        Cm: float = CM,
        splitting: str = 'godunov'
    ):
        self.ny = ny
        self.nx = nx
        self.dx = dx
        self.dy = dy
        self.chi = chi
        self.Cm = Cm
        self.splitting = splitting.lower()

        # Determine diffusion coefficients
        # Priority: explicit D values > CV-based computation > defaults
        if D_L is not None and D_T is not None:
            # Direct D specification
            self.D_L = D_L
            self.D_T = D_T
            self.cv_long = None
            self.cv_trans = None
        else:
            # CV-based computation (use defaults if not specified)
            self.cv_long = cv_long if cv_long is not None else CV_LONGITUDINAL_DEFAULT
            self.cv_trans = cv_trans if cv_trans is not None else CV_TRANSVERSE_DEFAULT
            self.D_L, self.D_T = get_diffusion_params(
                dx, self.cv_long, self.cv_trans
            )

        # Create ionic model
        self.model = ORdModel(celltype=celltype)
        self.n_states = 41  # ORd has 41 state variables

        # Create diffusion operator
        self.diffusion = DiffusionOperator(
            ny, nx, dx, dy, self.D_L, self.D_T, fiber_angle
        )

        # Initialize state array (ny, nx, n_states)
        self._init_states()

        # Stimulus configuration
        self.stim_sites = []  # List of (i_slice, j_slice, start_time, duration)
        self.stim_amplitude = 80.0  # uA/uF

        # Time tracking
        self.time = 0.0

        # Cache kernel parameters for parallel execution
        self._kernel_params = get_kernel_params(self.model)
        self._use_parallel = True  # Use parallel kernel by default

    def _init_states(self):
        """Initialize all cells to resting state."""
        initial_state = self.model.get_initial_state()
        self.states = np.zeros((self.ny, self.nx, self.n_states))
        for i in range(self.ny):
            for j in range(self.nx):
                self.states[i, j, :] = initial_state.copy()

    def get_voltage(self) -> np.ndarray:
        """Get current voltage field (ny, nx)."""
        return self.states[:, :, StateIndex.V]

    def set_voltage(self, V: np.ndarray):
        """Set voltage field directly (for initial conditions)."""
        self.states[:, :, StateIndex.V] = V

    def add_stimulus(
        self,
        region: Tuple[slice, slice],
        start_time: float,
        duration: float = 0.5,
        bcl: Optional[float] = None,
        n_beats: int = 1
    ):
        """
        Add a stimulus site.

        Parameters
        ----------
        region : tuple of slices
            (i_slice, j_slice) defining the stimulated region
        start_time : float
            Time of first stimulus (ms)
        duration : float
            Stimulus duration (ms)
        bcl : float, optional
            Basic cycle length for periodic stimulation (ms)
        n_beats : int
            Number of beats (only used if bcl is set)
        """
        if bcl is not None:
            for beat in range(n_beats):
                t_start = start_time + beat * bcl
                self.stim_sites.append((region[0], region[1], t_start, duration))
        else:
            self.stim_sites.append((region[0], region[1], start_time, duration))

    def _get_stimulus_mask(self, t: float) -> np.ndarray:
        """Get current stimulus mask based on time."""
        mask = np.zeros((self.ny, self.nx), dtype=np.bool_)

        for i_slice, j_slice, t_start, duration in self.stim_sites:
            if t_start <= t < t_start + duration:
                mask[i_slice, j_slice] = True

        return mask

    def _ionic_step(self, dt: float, stim_mask: np.ndarray):
        """
        Advance ionic model at all grid points.

        Uses parallel Numba kernel for performance.
        Updates states in-place including voltage update from Iion.
        """
        if self._use_parallel:
            # Use parallel kernel
            self.states = ionic_step_tissue(
                self.states, dt, stim_mask, self.stim_amplitude,
                *self._kernel_params
            )
        else:
            # Fallback to serial Python loop
            for i in range(self.ny):
                for j in range(self.nx):
                    Istim = -self.stim_amplitude if stim_mask[i, j] else 0.0
                    self.states[i, j, :] = self.model.step(
                        self.states[i, j, :], dt, Istim
                    )

    def _diffusion_step(self, dt: float):
        """
        Advance diffusion equation.

        The monodomain diffusion coefficient D already incorporates
        the conductivity scaling: D = sigma / (chi * Cm)

        So we just compute: dV/dt = div(D * grad(V))
        """
        V = self.get_voltage()

        # Compute diffusion term: div(D * grad(V))
        diff = self.diffusion.apply(V)

        # Update voltage (explicit Euler)
        # D already includes sigma/(chi*Cm) scaling
        dV = dt * diff
        self.states[:, :, StateIndex.V] += dV

    def step(self, dt: float):
        """
        Advance simulation by one time step using operator splitting.

        Parameters
        ----------
        dt : float
            Time step (ms)
        """
        stim_mask = self._get_stimulus_mask(self.time)

        if self.splitting == 'godunov':
            # First-order Godunov splitting: I -> D
            self._ionic_step(dt, stim_mask)
            self._diffusion_step(dt)

        elif self.splitting == 'strang':
            # Second-order Strang splitting: D/2 -> I -> D/2
            self._diffusion_step(dt / 2)
            self._ionic_step(dt, stim_mask)
            self._diffusion_step(dt / 2)

        else:
            raise ValueError(f"Unknown splitting method: {self.splitting}")

        self.time += dt

    def run(
        self,
        t_end: float,
        dt: float = 0.01,
        save_interval: Optional[float] = None,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run simulation.

        Parameters
        ----------
        t_end : float
            End time (ms)
        dt : float
            Time step (ms)
        save_interval : float, optional
            Interval for saving snapshots (ms). If None, saves every step.
        progress_callback : callable, optional
            Function called with (current_time, total_time) for progress updates

        Returns
        -------
        t : ndarray
            Time points
        V_history : ndarray (n_times, ny, nx)
            Voltage snapshots
        """
        # Check stability
        dt_max = self.diffusion.get_stability_limit()
        if dt > dt_max:
            print(f"Warning: dt={dt} exceeds stability limit {dt_max:.4f} ms")

        # Setup time stepping
        n_steps = int(np.ceil(t_end / dt))
        if save_interval is None:
            save_interval = dt

        save_every = max(1, int(save_interval / dt))
        n_saves = n_steps // save_every + 1

        # Allocate history
        t_history = []
        V_history = []

        # Save initial state
        t_history.append(self.time)
        V_history.append(self.get_voltage().copy())

        # Main loop
        for step_idx in range(n_steps):
            self.step(dt)

            # Save snapshot
            if (step_idx + 1) % save_every == 0:
                t_history.append(self.time)
                V_history.append(self.get_voltage().copy())

            # Progress callback
            if progress_callback is not None and step_idx % 100 == 0:
                progress_callback(self.time, t_end)

        return np.array(t_history), np.array(V_history)

    def compute_activation_time(self, V_history: np.ndarray, threshold: float = -40.0) -> np.ndarray:
        """
        Compute activation time at each point.

        Parameters
        ----------
        V_history : ndarray (n_times, ny, nx)
            Voltage history
        threshold : float
            Voltage threshold for activation (mV)

        Returns
        -------
        act_time : ndarray (ny, nx)
            Activation time at each point (NaN if never activated)
        """
        act_time = np.full((self.ny, self.nx), np.nan)

        for i in range(self.ny):
            for j in range(self.nx):
                trace = V_history[:, i, j]
                for t_idx in range(1, len(trace)):
                    if trace[t_idx] > threshold and trace[t_idx-1] <= threshold:
                        # Linear interpolation for more precise timing
                        frac = (threshold - trace[t_idx-1]) / (trace[t_idx] - trace[t_idx-1])
                        act_time[i, j] = t_idx - 1 + frac
                        break

        return act_time

    def compute_conduction_velocity(
        self,
        act_time: np.ndarray,
        direction: str = 'x'
    ) -> float:
        """
        Compute conduction velocity from activation times.

        Parameters
        ----------
        act_time : ndarray (ny, nx)
            Activation time map
        direction : str
            'x' for horizontal, 'y' for vertical

        Returns
        -------
        cv : float
            Conduction velocity (cm/ms = m/s * 10)
        """
        if direction == 'x':
            # Use middle row
            mid_i = self.ny // 2
            times = act_time[mid_i, :]
            distances = np.arange(self.nx) * self.dx
        else:
            # Use middle column
            mid_j = self.nx // 2
            times = act_time[:, mid_j]
            distances = np.arange(self.ny) * self.dy

        # Find valid points
        valid = ~np.isnan(times)
        if np.sum(valid) < 2:
            return np.nan

        # Linear fit
        t_valid = times[valid]
        d_valid = distances[valid]

        # CV = distance / time (slope of d vs t)
        # Use linear regression
        coeffs = np.polyfit(t_valid, d_valid, 1)
        cv = coeffs[0]  # slope = cm/ms

        return cv


def create_spiral_wave_ic(
    sim: MonodomainSimulation,
    center: Tuple[int, int] = None,
    arm_width: int = None
):
    """
    Create initial conditions for spiral wave using S1-S2 protocol.

    This sets up an S1-S2 cross-field stimulation protocol:
    - S1: Planar wave from left edge
    - S2: Half-plane stimulus applied during vulnerable window

    Parameters
    ----------
    sim : MonodomainSimulation
        Simulation object (will be modified)
    center : tuple, optional
        (i, j) center of the spiral. Default is grid center.
    arm_width : int, optional
        Width of S2 stimulus region. Default is nx//2.
    """
    if center is None:
        center = (sim.ny // 2, sim.nx // 2)
    if arm_width is None:
        arm_width = sim.nx // 2

    # S1: Full left edge stimulus
    sim.add_stimulus(
        region=(slice(None), slice(0, 5)),
        start_time=0.0,
        duration=1.0
    )

    # S2: Half-plane stimulus during vulnerable window
    # Timing depends on APD - typically 200-300ms after S1 for ORd
    # S2 applied to lower half of domain
    sim.add_stimulus(
        region=(slice(0, center[0]), slice(0, arm_width)),
        start_time=280.0,  # During repolarization
        duration=1.0
    )


def estimate_cv_from_params(D_L: float, D_T: float, fiber_angle: float = 0.0) -> Tuple[float, float]:
    """
    Estimate conduction velocity from diffusion parameters.

    Uses linearized cable equation approximation:
    CV ~ 2 * sqrt(D / (chi * Cm * tau_foot))

    where tau_foot ~ 0.5 ms for sodium current upstroke.

    Parameters
    ----------
    D_L, D_T : float
        Diffusion coefficients (cm^2/ms)
    fiber_angle : float
        Fiber angle (radians)

    Returns
    -------
    cv_long, cv_trans : float
        Estimated CV along and across fibers (cm/ms)
    """
    tau_foot = 0.5  # Approximate foot time constant (ms)

    cv_long = 2.0 * np.sqrt(D_L / (CHI * CM * tau_foot))
    cv_trans = 2.0 * np.sqrt(D_T / (CHI * CM * tau_foot))

    return cv_long, cv_trans
