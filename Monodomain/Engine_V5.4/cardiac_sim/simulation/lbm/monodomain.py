"""
LBM Monodomain Simulation

Top-level orchestrator for LBM-EP cardiac simulation.
Integrates collision, streaming, boundary conditions, and ionic model.

Grid convention: (Nx, Ny) with indexing='ij', matching StructuredGrid.

Run loop:
    1. Collide (with source term from ionic current)
    2. Stream
    3. Apply bounce-back BC
    4. Recover voltage (V = Σf_i)
    5. Update ionic states (Rush-Larsen or Forward Euler)

Ref: Research/04_LBM_EP:L936-950
Ref: improvement.md:L1376-1392 (LBM user API)
"""

import torch
import numpy as np
from typing import Optional, Callable, Iterator, Tuple, Union

from .state import LBMState, create_lbm_state
from .collision import CollisionOperator, BGKCollision, MRTCollision, create_isotropic_bgk
from .d2q5 import d2q5, D2Q5

from ...ionic import IonicModel, TTP06Model, ORdModel, CellType
from ..classical.solver.ionic_time_stepping.rush_larsen import RushLarsenSolver
from ..classical.solver.ionic_time_stepping.forward_euler import ForwardEulerIonicSolver


class LBMSimulation:
    """
    LBM monodomain cardiac simulation.

    Alternative to classical FEM/FDM/FVM path. Uses Lattice-Boltzmann Method
    for diffusion, avoiding linear system solves entirely.

    Grid convention: (Nx, Ny) with indexing='ij'.

    Parameters
    ----------
    Nx, Ny : int
        Grid dimensions (Nx = axis 0 = x, Ny = axis 1 = y)
    dx : float
        Grid spacing (cm)
    dt : float
        Time step (ms)
    D : float
        Diffusion coefficient (cm²/ms) for isotropic BGK
    collision : CollisionOperator, optional
        Custom collision operator (overrides D if provided)
    ionic_model : str or IonicModel
        Ionic model ('ttp06', 'ord') or instance
    ionic_solver : str
        Ionic solver ('rush_larsen', 'forward_euler')
    chi : float
        Surface-to-volume ratio (cm⁻¹)
    Cm : float
        Membrane capacitance (µF/cm²)
    cell_type : str
        Cell type for ionic model
    mask : torch.Tensor, optional
        Domain mask (1 = active), shape (Nx, Ny)
    device : str or torch.device
        Computation device
    dtype : torch.dtype
        Data type

    Example
    -------
    >>> sim = LBMSimulation(
    ...     Nx=100, Ny=100, dx=0.025, dt=0.01,
    ...     D=0.001, ionic_model='ttp06'
    ... )
    >>> # Apply stimulus
    >>> sim.state.V[:5, :] = 0.0  # Depolarize left edge
    >>> sim.state.init_equilibrium()
    >>> # Run
    >>> for state in sim.run(t_end=100.0, save_every=10.0):
    ...     plot(state.V)
    """

    def __init__(
        self,
        Nx: int,
        Ny: int,
        dx: float,
        dt: float,
        D: float = 0.001,
        collision: Optional[CollisionOperator] = None,
        ionic_model: Union[str, IonicModel] = 'ttp06',
        ionic_solver: str = 'rush_larsen',
        chi: float = 1400.0,
        Cm: float = 1.0,
        cell_type: str = 'EPI',
        mask: Optional[torch.Tensor] = None,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float64
    ):
        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.dt = dt
        self.chi = chi
        self.Cm = Cm

        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.dtype = dtype

        # Build collision operator
        if collision is not None:
            self.collision = collision
        else:
            self.collision = create_isotropic_bgk(D, dx, dt, d2q5, device, dtype)

        # Build ionic model
        self._ionic_model = self._build_ionic_model(ionic_model, cell_type, device)

        # Build ionic solver
        self._ionic_solver = self._build_ionic_solver(ionic_solver, self._ionic_model)

        # Initialize state
        n_dof = Nx * Ny
        V_init = torch.full((Nx, Ny), self._ionic_model.V_rest, device=device, dtype=dtype)
        ionic_states = self._ionic_model.get_initial_state(n_cells=n_dof)
        if ionic_states.device != device:
            ionic_states = ionic_states.to(device)

        if mask is None:
            mask = torch.ones(Nx, Ny, device=device, dtype=torch.bool)

        self.state = create_lbm_state(
            Nx, Ny, V_init, ionic_states, mask, d2q5, device, dtype
        )

        # Stimulus storage (list of (mask, start, duration, amplitude))
        self._stimuli = []

    def _build_ionic_model(
        self,
        name: Union[str, IonicModel],
        cell_type: str,
        device: torch.device
    ) -> IonicModel:
        """Build ionic model from string or return existing."""
        if isinstance(name, IonicModel):
            return name

        name_lower = name.lower()
        cell_type_enum = getattr(CellType, cell_type.upper())

        if name_lower == 'ttp06':
            return TTP06Model(cell_type=cell_type_enum, device=device)
        elif name_lower == 'ord':
            return ORdModel(cell_type=cell_type_enum, device=device)
        else:
            raise ValueError(f"Unknown ionic model: {name}")

    def _build_ionic_solver(self, name: str, ionic_model: IonicModel):
        """Build ionic solver from string."""
        name_lower = name.lower().replace('-', '_')

        if name_lower in ('rush_larsen', 'rl'):
            return RushLarsenSolver(ionic_model)
        elif name_lower in ('forward_euler', 'fe'):
            return ForwardEulerIonicSolver(ionic_model)
        else:
            raise ValueError(f"Unknown ionic solver: {name}")

    def add_stimulus(
        self,
        mask: torch.Tensor,
        start: float,
        duration: float,
        amplitude: float
    ) -> None:
        """
        Add a stimulus to the simulation.

        Parameters
        ----------
        mask : torch.Tensor
            Spatial mask, shape (Nx, Ny), boolean or float
        start : float
            Start time (ms)
        duration : float
            Duration (ms)
        amplitude : float
            Amplitude (µA/µF), negative depolarizes
        """
        mask = mask.to(device=self.device, dtype=self.dtype)
        self._stimuli.append((mask, start, duration, amplitude))

    def _evaluate_stimulus(self, t: float) -> torch.Tensor:
        """
        Evaluate total stimulus current at time t.

        Returns
        -------
        Istim : torch.Tensor
            Stimulus current, shape (Nx*Ny,)
        """
        Istim = torch.zeros(self.Nx * self.Ny, device=self.device, dtype=self.dtype)

        for mask, start, duration, amplitude in self._stimuli:
            if start <= t < start + duration:
                Istim += amplitude * mask.flatten()

        return Istim

    def step(self, dt: Optional[float] = None) -> None:
        """
        Advance simulation by one time step.

        Order: ionic → collide → stream → bounce-back → recover
        """
        if dt is None:
            dt = self.dt

        state = self.state

        # Get V in flat format for ionic computations
        V_flat = state.V_flat

        # Evaluate stimulus
        Istim = self._evaluate_stimulus(state.t)

        # === Ionic step ===
        # Compute ionic current (V separate from ionic_states)
        Iion = self._ionic_model.compute_Iion(V_flat, state.ionic_states)

        # Update ionic states (gates and concentrations)
        self._ionic_solver.step_with_V(state.ionic_states, V_flat, Istim, dt)

        # === Compute source term for LBM ===
        # Source S = -(Iion + Istim) / (χ·Cm)
        source_flat = -(Iion + Istim) / (self.chi * self.Cm)
        source = source_flat.reshape(self.Nx, self.Ny)

        # === LBM steps ===
        # Collide
        state.f = self.collision.collide(state.f, state.V, source, dt)

        # Stream
        state.stream()

        # Bounce-back BC
        state.apply_bounce_back()

        # Recover voltage
        state.recover_voltage()

        # Advance time
        state.t += dt

    def run(
        self,
        t_end: float,
        save_every: float = 1.0,
        callback: Optional[Callable[[LBMState], bool]] = None
    ) -> Iterator[LBMState]:
        """
        Run simulation as a generator.

        Parameters
        ----------
        t_end : float
            End time (ms)
        save_every : float
            Save interval (ms)
        callback : callable, optional
            Called at each save point. Return True to stop.

        Yields
        ------
        state : LBMState
            State at each save point
        """
        state = self.state
        dt = self.dt
        next_save = save_every

        while state.t < t_end:
            self.step(dt)

            if state.t >= next_save - 1e-9:
                next_save += save_every

                if callback is not None:
                    if callback(state):
                        break

                yield state

    def run_to_array(
        self,
        t_end: float,
        save_every: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run simulation and return results as numpy arrays.

        Returns
        -------
        times : np.ndarray
            Time points
        voltages : np.ndarray
            Voltage at each save point, shape (n_saves, Nx, Ny)
        """
        times = []
        voltages = []

        for state in self.run(t_end, save_every):
            times.append(state.t)
            voltages.append(state.V.cpu().numpy().copy())

        return np.array(times), np.array(voltages)

    def reset(self) -> None:
        """Reset simulation to initial state."""
        n_dof = self.Nx * self.Ny
        V_init = torch.full(
            (self.Nx, self.Ny), self._ionic_model.V_rest,
            device=self.device, dtype=self.dtype
        )
        ionic_states = self._ionic_model.get_initial_state(n_cells=n_dof)
        if ionic_states.device != self.device:
            ionic_states = ionic_states.to(self.device)

        self.state.V = V_init
        self.state.ionic_states = ionic_states
        self.state.t = 0.0
        self.state.init_equilibrium()

    @property
    def ionic_model(self) -> IonicModel:
        """Get the ionic model."""
        return self._ionic_model

    def __repr__(self) -> str:
        return (
            f"LBMSimulation(Nx={self.Nx}, Ny={self.Ny}, "
            f"dx={self.dx}, dt={self.dt}, "
            f"ionic_model={self._ionic_model.name}, "
            f"t={self.state.t:.2f} ms)"
        )
