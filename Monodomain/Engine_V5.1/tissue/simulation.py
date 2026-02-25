"""
Monodomain Tissue Simulation (PyTorch GPU Version)

Implements the monodomain equation with operator splitting:
    chi * Cm * dV/dt = -chi * Iion(V, u) + div(D * grad(V)) + Istim

Operator splitting separates:
1. Ionic step: dV/dt = -Iion/Cm (Rush-Larsen for gates)
2. Diffusion step: dV/dt = div(D*grad(V)) / (chi * Cm)

Supports Godunov (1st order) and Strang (2nd order) splitting.

All computations run on GPU via PyTorch for high performance.
"""

import torch
import numpy as np
from typing import Optional, Tuple, Callable, Union, List, Dict

from ionic import ORdModel, StateIndex, CellType
from tissue.diffusion import (
    DiffusionOperator, get_diffusion_params,
    CHI, CM, CV_LONGITUDINAL_DEFAULT, CV_TRANSVERSE_DEFAULT,
    APD_REF
)


class MonodomainSimulation:
    """
    2D Monodomain tissue simulation with GPU acceleration.

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
    fiber_angle : float or Tensor
        Fiber orientation angle in radians
    celltype : CellType
        Cell type (ENDO, EPI, or M_CELL)
    chi : float
        Surface-to-volume ratio (cm^-1), default 1400
    Cm : float
        Membrane capacitance (uF/cm^2), default 1.0
    splitting : str
        Operator splitting method: 'godunov' (1st order) or 'strang' (2nd order)
    device : str
        Device for computation ('cuda' or 'cpu')
    dtype : torch.dtype
        Data type for tensors (default torch.float64)
    params_override : dict, optional
        Dictionary of ORd model parameters to override (e.g., {'GKr_scale': 2.0}).
        Useful for APD shortening to enable spirals in smaller domains.
    apd_ms : float, optional
        Expected action potential duration in ms. Used for D validation.
        Default is 280 (normal ORd). Use ~150 for APD-shortening scenarios.
        If not specified and params_override contains APD-shortening parameters,
        the system will use 150ms automatically.
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
        fiber_angle: Union[float, torch.Tensor] = 0.0,
        celltype: CellType = CellType.ENDO,
        chi: float = CHI,
        Cm: float = CM,
        splitting: str = 'godunov',
        device: str = 'cuda',
        dtype: torch.dtype = torch.float64,
        params_override: Optional[Dict[str, float]] = None,
        apd_ms: Optional[float] = None
    ):
        self.ny = ny
        self.nx = nx
        self.dx = dx
        self.dy = dy
        self.chi = chi
        self.Cm = Cm
        self.splitting = splitting.lower()
        self.device = device if torch.cuda.is_available() or device == 'cpu' else 'cpu'
        self.dtype = dtype

        # Determine APD for validation
        # If apd_ms not specified, infer from params_override
        if apd_ms is not None:
            self.apd_ms = apd_ms
        elif params_override is not None:
            # Check for APD-shortening parameters
            gkr_scale = params_override.get('GKr_scale', 1.0)
            pca_scale = params_override.get('PCa_scale', 1.0)
            if gkr_scale > 1.5 or pca_scale < 0.7:
                # APD-shortening detected, use ~150ms
                self.apd_ms = 150.0
            else:
                self.apd_ms = APD_REF
        else:
            self.apd_ms = APD_REF

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
                dx, self.cv_long, self.cv_trans, apd_ms=self.apd_ms
            )

        # Create ionic model (uses float64 internally)
        self.model = ORdModel(
            celltype=celltype,
            device=self.device,
            params_override=params_override
        )
        self.n_states = StateIndex.N_STATES  # 41 state variables
        self.params_override = params_override

        # Create diffusion operator
        self.diffusion = DiffusionOperator(
            ny, nx, dx, dy, self.D_L, self.D_T, fiber_angle,
            device=self.device, dtype=dtype
        )

        # Initialize state array (ny, nx, n_states)
        self._init_states()

        # Stimulus configuration
        self.stim_sites: List[Tuple[slice, slice, float, float]] = []
        self.stim_amplitude = 80.0  # uA/uF (applied as -80 for depolarizing current)

        # Time tracking
        self.time = 0.0

    def _init_states(self):
        """Initialize all cells to resting state."""
        initial_state = self.model.get_initial_state()  # Shape: (n_states,)

        # Expand to tissue dimensions (ny, nx, n_states)
        self.states = initial_state.unsqueeze(0).unsqueeze(0).expand(
            self.ny, self.nx, -1
        ).clone()

    def get_voltage(self) -> torch.Tensor:
        """Get current voltage field (ny, nx)."""
        return self.states[:, :, StateIndex.V]

    def set_voltage(self, V: torch.Tensor):
        """Set voltage field directly (for initial conditions)."""
        if not isinstance(V, torch.Tensor):
            V = torch.tensor(V, dtype=self.dtype, device=self.device)
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

    def _get_stimulus_mask(self, t: float) -> torch.Tensor:
        """Get current stimulus mask based on time."""
        mask = torch.zeros((self.ny, self.nx), dtype=torch.bool, device=self.device)

        for i_slice, j_slice, t_start, duration in self.stim_sites:
            if t_start <= t < t_start + duration:
                mask[i_slice, j_slice] = True

        return mask

    def _ionic_step(self, dt: float, stim_mask: torch.Tensor):
        """
        Advance ionic model at all grid points.

        GPU-accelerated - processes all cells in parallel using PyTorch.
        Updates states in-place including voltage update from Iion.
        """
        # Create stimulus tensor: -amplitude where mask is True, 0 elsewhere
        Istim = torch.where(
            stim_mask,
            torch.tensor(-self.stim_amplitude, dtype=self.dtype, device=self.device),
            torch.tensor(0.0, dtype=self.dtype, device=self.device)
        )

        # Call model.step which handles the entire tissue at once
        # The model accepts (ny, nx, n_states) tensor and Istim (ny, nx) tensor
        self.states = self.model.step(self.states, dt, Istim)

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
        self.states[:, :, StateIndex.V] = self.states[:, :, StateIndex.V] + dV

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
        dt: float = 0.02,
        save_interval: Optional[float] = None,
        progress_callback: Optional[Callable[[float, float], None]] = None,
        return_numpy: bool = True
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
            Interval for saving snapshots (ms). If None, saves at dt intervals.
        progress_callback : callable, optional
            Function called with (current_time, total_time) for progress updates
        return_numpy : bool
            If True, return numpy arrays; otherwise return torch tensors

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

        # Allocate history on CPU to avoid GPU memory issues for long simulations
        t_history = []
        V_history = []

        # Save initial state
        t_history.append(self.time)
        V_history.append(self.get_voltage().cpu().numpy().copy())

        # Main loop
        for step_idx in range(n_steps):
            self.step(dt)

            # Save snapshot
            if (step_idx + 1) % save_every == 0:
                t_history.append(self.time)
                V_history.append(self.get_voltage().cpu().numpy().copy())

            # Progress callback
            if progress_callback is not None and step_idx % 100 == 0:
                progress_callback(self.time, t_end)

        t_arr = np.array(t_history)
        V_arr = np.array(V_history)

        return t_arr, V_arr

    def compute_activation_time(
        self,
        V_history: np.ndarray,
        threshold: float = -40.0,
        t_array: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute activation time at each point.

        Parameters
        ----------
        V_history : ndarray (n_times, ny, nx)
            Voltage history
        threshold : float
            Voltage threshold for activation (mV)
        t_array : ndarray, optional
            Time array corresponding to V_history. If None, uses indices.

        Returns
        -------
        act_time : ndarray (ny, nx)
            Activation time at each point (NaN if never activated)
        """
        n_times, ny, nx = V_history.shape
        act_time = np.full((ny, nx), np.nan)

        if t_array is None:
            t_array = np.arange(n_times, dtype=np.float64)

        for i in range(ny):
            for j in range(nx):
                trace = V_history[:, i, j]
                for t_idx in range(1, len(trace)):
                    if trace[t_idx] > threshold and trace[t_idx-1] <= threshold:
                        # Linear interpolation for more precise timing
                        frac = (threshold - trace[t_idx-1]) / (trace[t_idx] - trace[t_idx-1])
                        act_time[i, j] = t_array[t_idx-1] + frac * (t_array[t_idx] - t_array[t_idx-1])
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

    def reset(self):
        """Reset simulation to initial state."""
        self._init_states()
        self.time = 0.0
        self.stim_sites = []


def create_spiral_wave_ic(
    sim: MonodomainSimulation,
    center: Optional[Tuple[int, int]] = None,
    arm_width: Optional[int] = None,
    s2_time: float = 280.0
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
    s2_time : float
        Time for S2 stimulus (should be during repolarization, ~280ms for ORd)
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
    # S2 applied to lower half of domain
    sim.add_stimulus(
        region=(slice(0, center[0]), slice(0, arm_width)),
        start_time=s2_time,
        duration=1.0
    )


def estimate_cv_from_params(
    D_L: float,
    D_T: float,
    fiber_angle: float = 0.0
) -> Tuple[float, float]:
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
