"""
MonodomainSimulation — Top-Level Orchestrator

Builds state and solvers from config strings. Owns the time loop,
output buffering, and progress reporting.

Ref: improvement.md:L1113-1187
Ref: improvement.md:L1329-1393 (User API)
"""

from typing import Optional, Callable, Iterator, Tuple, Union
import torch
import numpy as np

from .state import SimulationState
from .discretization_scheme.base import SpatialDiscretization

# Discretization schemes
from .discretization_scheme.fem import FEMDiscretization
from .discretization_scheme.fdm import FDMDiscretization
from .discretization_scheme.fvm import FVMDiscretization

# Splitting strategies
from .solver.splitting.godunov import GodunovSplitting
from .solver.splitting.strang import StrangSplitting

# Ionic solvers
from .solver.ionic_time_stepping.rush_larsen import RushLarsenSolver
from .solver.ionic_time_stepping.forward_euler import ForwardEulerIonicSolver

# Diffusion solvers
from .solver.diffusion_time_stepping.implicit.crank_nicolson import CrankNicolsonSolver
from .solver.diffusion_time_stepping.implicit.bdf1 import BDF1Solver
from .solver.diffusion_time_stepping.implicit.bdf2 import BDF2Solver
from .solver.diffusion_time_stepping.explicit.forward_euler import ForwardEulerDiffusionSolver
from .solver.diffusion_time_stepping.explicit.rk2 import RK2Solver
from .solver.diffusion_time_stepping.explicit.rk4 import RK4Solver

# Linear solvers
from .solver.diffusion_time_stepping.linear_solver.pcg import PCGSolver
from .solver.diffusion_time_stepping.linear_solver.chebyshev import ChebyshevSolver
from .solver.diffusion_time_stepping.linear_solver.fft import DCTSolver, FFTSolver

# Ionic models
from ...ionic import IonicModel, TTP06Model, ORdModel, CellType

# Stimulus
from ...tissue_builder.stimulus.protocol import StimulusProtocol


# =============================================================================
# Factory Functions
# =============================================================================

def _build_ionic_model(
    name: Union[str, IonicModel],
    cell_type: str = 'EPI',
    device: str = 'cuda'
) -> IonicModel:
    """
    Build ionic model from string name or return existing model.

    Parameters
    ----------
    name : str or IonicModel
        Model name ('ttp06', 'ord') or existing IonicModel instance
    cell_type : str
        Cell type ('ENDO', 'EPI', 'M_CELL')
    device : str
        Computation device

    Returns
    -------
    model : IonicModel
    """
    if isinstance(name, IonicModel):
        return name

    name_lower = name.lower()
    cell_type_enum = getattr(CellType, cell_type.upper())
    device_obj = torch.device(device)

    if name_lower == 'ttp06':
        return TTP06Model(cell_type=cell_type_enum, device=device_obj)
    elif name_lower == 'ord':
        return ORdModel(cell_type=cell_type_enum, device=device_obj)
    else:
        raise ValueError(f"Unknown ionic model: {name}")


def _build_ionic_solver(name: str, ionic_model: IonicModel):
    """
    Build ionic solver from string name.

    Parameters
    ----------
    name : str
        Solver name ('rush_larsen', 'forward_euler')
    ionic_model : IonicModel
        The ionic model

    Returns
    -------
    solver : IonicSolver
    """
    name_lower = name.lower().replace('-', '_')

    if name_lower in ('rush_larsen', 'rl'):
        return RushLarsenSolver(ionic_model)
    elif name_lower in ('forward_euler', 'fe'):
        return ForwardEulerIonicSolver(ionic_model)
    else:
        raise ValueError(f"Unknown ionic solver: {name}")


def _build_linear_solver(name: str, tol: float = 1e-8, max_iters: int = 500, **kwargs):
    """
    Build linear solver from string name.

    Parameters
    ----------
    name : str
        Solver name ('pcg', 'chebyshev', 'dct', 'fft', 'none')
    tol : float
        Convergence tolerance (PCG only)
    max_iters : int
        Maximum iterations
    **kwargs : dict
        Additional parameters for specialized solvers (DCT/FFT require grid params)

    Returns
    -------
    solver : LinearSolver or None
    """
    name_lower = name.lower()

    if name_lower == 'pcg':
        return PCGSolver(tol=tol, max_iters=max_iters)
    elif name_lower == 'chebyshev':
        return ChebyshevSolver(max_iters=max_iters, tol=tol)
    elif name_lower == 'dct':
        # DCT requires grid parameters: nx, ny, dx, dy, dt, D, chi, Cm, scheme
        return DCTSolver(**kwargs)
    elif name_lower == 'fft':
        # FFT requires grid parameters
        return FFTSolver(**kwargs)
    elif name_lower == 'none':
        return None
    else:
        raise ValueError(f"Unknown linear solver: {name}")


def _build_diffusion_solver(
    name: str,
    spatial: SpatialDiscretization,
    dt: float,
    linear_solver=None
):
    """
    Build diffusion solver from string name.

    Parameters
    ----------
    name : str
        Solver name: 'crank_nicolson'/'cn', 'bdf1', 'bdf2',
                     'forward_euler'/'fe', 'rk2', 'rk4'
    spatial : SpatialDiscretization
        Spatial discretization
    dt : float
        Time step
    linear_solver : LinearSolver, optional
        Linear solver (required for implicit methods)

    Returns
    -------
    solver : DiffusionSolver
    """
    name_lower = name.lower().replace('-', '_')

    if name_lower in ('crank_nicolson', 'cn'):
        if linear_solver is None:
            linear_solver = PCGSolver()
        return CrankNicolsonSolver(spatial, dt, linear_solver)
    elif name_lower == 'bdf1':
        if linear_solver is None:
            linear_solver = PCGSolver()
        return BDF1Solver(spatial, dt, linear_solver)
    elif name_lower == 'bdf2':
        if linear_solver is None:
            linear_solver = PCGSolver()
        return BDF2Solver(spatial, dt, linear_solver)
    elif name_lower in ('forward_euler', 'fe'):
        return ForwardEulerDiffusionSolver(spatial, dt)
    elif name_lower == 'rk2':
        return RK2Solver(spatial, dt)
    elif name_lower == 'rk4':
        return RK4Solver(spatial, dt)
    else:
        raise ValueError(f"Unknown diffusion solver: {name}")


def _build_splitting(name: str, ionic_solver, diffusion_solver):
    """
    Build splitting strategy from string name.

    Parameters
    ----------
    name : str
        Splitting name ('godunov', 'strang')
    ionic_solver : IonicSolver
    diffusion_solver : DiffusionSolver

    Returns
    -------
    splitting : SplittingStrategy
    """
    name_lower = name.lower()

    if name_lower in ('godunov', 'lie'):
        return GodunovSplitting(ionic_solver, diffusion_solver)
    elif name_lower == 'strang':
        return StrangSplitting(ionic_solver, diffusion_solver)
    else:
        raise ValueError(f"Unknown splitting strategy: {name}")


# =============================================================================
# MonodomainSimulation
# =============================================================================

class MonodomainSimulation:
    """
    Top-level orchestrator for monodomain cardiac simulations.

    Builds state and solvers from config strings. Owns the time loop,
    output buffering, and progress reporting.

    Example
    -------
    >>> sim = MonodomainSimulation(
    ...     spatial=fem_discretization,
    ...     ionic_model='ttp06',
    ...     stimulus=stimulus_protocol,
    ...     dt=0.02,
    ...     splitting='strang',
    ...     ionic_solver='rush_larsen',
    ...     diffusion_solver='crank_nicolson',
    ...     linear_solver='pcg',
    ... )
    >>> for state in sim.run(t_end=500.0, save_every=1.0):
    ...     plot(state.V)
    """

    def __init__(
        self,
        spatial: SpatialDiscretization,
        ionic_model: Union[str, IonicModel],
        stimulus: Optional[StimulusProtocol] = None,
        dt: float = 0.02,
        splitting: str = 'strang',
        ionic_solver: str = 'rush_larsen',
        diffusion_solver: str = 'crank_nicolson',
        linear_solver: str = 'pcg',
        cell_type: str = 'EPI',
        pcg_tol: float = 1e-8,
        pcg_max_iter: int = 500,
    ):
        """
        Initialize simulation.

        Parameters
        ----------
        spatial : SpatialDiscretization
            Spatial discretization (FEM, FDM, or FVM)
        ionic_model : str or IonicModel
            Ionic model name ('ttp06', 'ord') or instance
        stimulus : StimulusProtocol, optional
            Stimulus protocol
        dt : float
            Time step (ms)
        splitting : str
            Splitting strategy ('godunov', 'strang')
        ionic_solver : str
            Ionic solver ('rush_larsen', 'forward_euler')
        diffusion_solver : str
            Diffusion solver ('crank_nicolson', 'bdf1', 'forward_euler')
        linear_solver : str
            Linear solver ('pcg', 'none')
        cell_type : str
            Cell type for ionic model ('ENDO', 'EPI', 'M_CELL')
        pcg_tol : float
            PCG convergence tolerance
        pcg_max_iter : int
            PCG maximum iterations
        """
        self.dt = dt
        self._spatial = spatial

        # Get coordinates and n_dof, then derive device/dtype from coordinates
        x, y = spatial.coordinates
        n_dof = spatial.n_dof
        device = x.device
        dtype = x.dtype

        # Build ionic model
        self._ionic_model = _build_ionic_model(
            ionic_model, cell_type=cell_type, device=str(device)
        )

        # Initialize V and ionic states separately
        V_init = torch.full((n_dof,), self._ionic_model.V_rest, device=device, dtype=dtype)
        ionic_states = self._ionic_model.get_initial_state(n_cells=n_dof)
        if ionic_states.device != device:
            ionic_states = ionic_states.to(device)

        # Precompute stimulus data
        if stimulus is None:
            stimulus = StimulusProtocol()

        stim_masks, stim_starts, stim_durations, stim_amplitudes = \
            self._precompute_stimulus(stimulus, x, y, device, dtype)

        # Build SimulationState
        self.state = SimulationState(
            spatial=spatial,
            n_dof=n_dof,
            x=x,
            y=y,
            V=V_init,
            ionic_states=ionic_states,
            gate_indices=list(self._ionic_model.gate_indices),
            concentration_indices=list(self._ionic_model.concentration_indices),
            t=0.0,
            stim_masks=stim_masks,
            stim_starts=stim_starts,
            stim_durations=stim_durations,
            stim_amplitudes=stim_amplitudes,
            output_buffer=None,
            buffer_idx=0,
        )

        # Build solver chain
        linear = _build_linear_solver(linear_solver, tol=pcg_tol, max_iters=pcg_max_iter)
        diffusion = _build_diffusion_solver(diffusion_solver, spatial, dt, linear)
        ionic = _build_ionic_solver(ionic_solver, self._ionic_model)
        self.splitting = _build_splitting(splitting, ionic, diffusion)

    def _precompute_stimulus(
        self,
        stimulus: StimulusProtocol,
        x: torch.Tensor,
        y: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype
    ) -> Tuple[torch.Tensor, list, list, list]:
        """
        Precompute stimulus masks and parameters.

        Returns
        -------
        stim_masks : torch.Tensor
            (n_stimuli, n_dof) float masks
        stim_starts : list
            Start times
        stim_durations : list
            Durations
        stim_amplitudes : list
            Amplitudes
        """
        n_stimuli = len(stimulus.stimuli)
        n_dof = len(x)

        if n_stimuli == 0:
            return (
                torch.zeros(0, n_dof, device=device, dtype=dtype),
                [],
                [],
                [],
            )

        stim_masks = torch.zeros(n_stimuli, n_dof, device=device, dtype=dtype)
        stim_starts = []
        stim_durations = []
        stim_amplitudes = []

        for i, stim in enumerate(stimulus.stimuli):
            mask = stim.get_mask(x, y)
            stim_masks[i] = mask.to(dtype)
            stim_starts.append(stim.start_time)
            stim_durations.append(stim.duration)
            stim_amplitudes.append(stim.amplitude)

        return stim_masks, stim_starts, stim_durations, stim_amplitudes

    def run(
        self,
        t_end: float,
        save_every: float = 1.0,
        callback: Optional[Callable[['SimulationState'], bool]] = None,
    ) -> Iterator['SimulationState']:
        """
        Run simulation as a generator.

        Yields state at each save point. Supports callback for early stopping.

        Parameters
        ----------
        t_end : float
            End time (ms)
        save_every : float
            Save interval (ms)
        callback : callable, optional
            Function(state) called at save points. Return True to stop early.

        Yields
        ------
        state : SimulationState
            State at each save point
        """
        state = self.state
        dt = self.dt
        next_save = save_every

        while state.t < t_end:
            # Core step: splitting.step advances ionic + diffusion
            self.splitting.step(state, dt)
            state.t += dt

            # Save at intervals
            if state.t >= next_save - 1e-9:  # Small epsilon for float comparison
                next_save += save_every

                if callback is not None:
                    if callback(state):
                        break

                yield state

    def run_to_array(
        self,
        t_end: float,
        save_every: float = 1.0,
        progress_callback: Optional[Callable[[float, float], None]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run simulation and return results as numpy arrays.

        Parameters
        ----------
        t_end : float
            End time (ms)
        save_every : float
            Save interval (ms)
        progress_callback : callable, optional
            Function(t, t_end) called each save point

        Returns
        -------
        times : np.ndarray
            Time points (n_saves,)
        voltages : np.ndarray
            Voltage at each save point (n_saves, n_dof)
        """
        times = []
        voltages = []

        for state in self.run(t_end, save_every):
            times.append(state.t)
            voltages.append(state.V.cpu().numpy().copy())

            if progress_callback is not None:
                progress_callback(state.t, t_end)

        return np.array(times), np.array(voltages)

    def step(self, dt: Optional[float] = None) -> None:
        """
        Advance simulation by one time step.

        Parameters
        ----------
        dt : float, optional
            Time step (uses self.dt if not specified)
        """
        if dt is None:
            dt = self.dt
        self.splitting.step(self.state, dt)
        self.state.t += dt

    def get_voltage(self) -> torch.Tensor:
        """Get current voltage field."""
        return self.state.V

    def set_voltage(self, V: torch.Tensor) -> None:
        """Set voltage field."""
        self.state.V = V

    def reset(self) -> None:
        """Reset simulation to initial state."""
        n_dof = self.state.n_dof
        device = self.state.V.device
        dtype = self.state.V.dtype
        self.state.V = torch.full((n_dof,), self._ionic_model.V_rest, device=device, dtype=dtype)
        ionic_states = self._ionic_model.get_initial_state(n_cells=n_dof)
        if ionic_states.device != device:
            ionic_states = ionic_states.to(device)
        self.state.ionic_states = ionic_states
        self.state.t = 0.0
        self.state.buffer_idx = 0

    def compute_activation_time(
        self,
        V_history: np.ndarray,
        times: np.ndarray,
        threshold: float = -20.0
    ) -> np.ndarray:
        """
        Compute local activation time (LAT) at each node.

        Parameters
        ----------
        V_history : np.ndarray
            Voltage history (n_times, n_nodes)
        times : np.ndarray
            Time points
        threshold : float
            Activation threshold (mV)

        Returns
        -------
        lat : np.ndarray
            Local activation time at each node (NaN if not activated)
        """
        n_nodes = V_history.shape[1]
        lat = np.full(n_nodes, np.nan)

        for j in range(n_nodes):
            V_trace = V_history[:, j]
            activated = np.where(V_trace > threshold)[0]
            if len(activated) > 0:
                lat[j] = times[activated[0]]

        return lat

    def compute_cv(
        self,
        V_history: np.ndarray,
        times: np.ndarray,
        x1: float,
        x2: float,
        threshold: float = -20.0
    ) -> float:
        """
        Compute conduction velocity between two x-positions.

        Parameters
        ----------
        V_history : np.ndarray
            Voltage history
        times : np.ndarray
            Time points
        x1, x2 : float
            X-coordinates for CV measurement
        threshold : float
            Activation threshold

        Returns
        -------
        cv : float
            Conduction velocity (cm/ms)
        """
        x = self.state.x.cpu().numpy()

        # Find nodes closest to x1 and x2
        idx1 = np.argmin(np.abs(x - x1))
        idx2 = np.argmin(np.abs(x - x2))

        # Get activation times
        lat = self.compute_activation_time(V_history, times, threshold)

        if np.isnan(lat[idx1]) or np.isnan(lat[idx2]):
            return np.nan

        # CV = distance / time
        distance = abs(x[idx2] - x[idx1])
        time_diff = lat[idx2] - lat[idx1]

        if time_diff <= 0:
            return np.nan

        return distance / time_diff

    @property
    def ionic_model(self) -> IonicModel:
        """Get the ionic model."""
        return self._ionic_model

    @property
    def spatial(self) -> SpatialDiscretization:
        """Get the spatial discretization."""
        return self._spatial

    def __repr__(self) -> str:
        return (
            f"MonodomainSimulation("
            f"spatial={type(self._spatial).__name__}, "
            f"ionic_model={self._ionic_model.name}, "
            f"t={self.state.t:.2f} ms)"
        )
