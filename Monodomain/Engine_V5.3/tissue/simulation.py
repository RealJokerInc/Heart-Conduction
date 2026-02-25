"""
Monodomain Simulation

Full tissue-level simulation combining:
- FEM spatial discretization
- Ionic models (ORd, TTP06)
- Implicit time stepping (CN, BDF)
- Stimulus protocols
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Callable, Union
import torch
import numpy as np

from fem import TriangularMesh, assemble_mass_matrix, assemble_stiffness_matrix
from solver import (
    CrankNicolsonStepper, BDFStepper, create_time_stepper,
    TimeStepperConfig, sparse_mv
)
from ionic import IonicModel
from .stimulus import Stimulus, StimulusProtocol


@dataclass
class SimulationConfig:
    """Configuration for monodomain simulation."""
    # Physical parameters
    D: float = 0.001              # Diffusion coefficient (cm²/ms)
    chi: float = 1400.0           # Surface-to-volume ratio (cm⁻¹)
    Cm: float = 1.0               # Membrane capacitance (µF/cm²)

    # Time stepping
    time_scheme: str = 'CN'       # 'CN', 'BDF1', or 'BDF2'
    dt: float = 0.02              # Time step (ms)

    # Solver
    pcg_tol: float = 1e-8
    pcg_max_iter: int = 500

    # Output
    save_interval: float = 1.0    # Save every N ms


class MonodomainSimulation:
    """
    FEM-based monodomain simulation with ionic models.

    Solves:
        χ·Cm·∂V/∂t = -χ·Iion(V, u) + ∇·(D·∇V) + Istim

    Using operator splitting:
        1. Ionic step: Advance ODEs for gating/concentrations (explicit)
        2. Diffusion step: Solve for voltage (implicit CN/BDF)
    """

    def __init__(
        self,
        mesh: TriangularMesh,
        ionic_model: IonicModel,
        config: Optional[SimulationConfig] = None,
        device: Optional[str] = None
    ):
        """
        Initialize monodomain simulation.

        Parameters
        ----------
        mesh : TriangularMesh
            Computational mesh
        ionic_model : IonicModel
            Ionic model (ORd, TTP06, etc.)
        config : SimulationConfig, optional
            Simulation configuration
        device : str, optional
            Device override ('cuda' or 'cpu')
        """
        self.mesh = mesh
        self.ionic_model = ionic_model
        self.config = config or SimulationConfig()

        # Device
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = mesh.device
        self.dtype = mesh.dtype

        # Coordinates for stimulus evaluation
        self.x = mesh.nodes[:, 0]
        self.y = mesh.nodes[:, 1]

        # Assemble FEM matrices
        self.M = assemble_mass_matrix(
            mesh, chi=self.config.chi, Cm=self.config.Cm
        )
        self.K = assemble_stiffness_matrix(mesh, D=self.config.D)

        # Initialize time stepper
        stepper_config = TimeStepperConfig(
            pcg_tol=self.config.pcg_tol,
            pcg_max_iter=self.config.pcg_max_iter,
            use_warm_start=True
        )
        self.time_stepper = create_time_stepper(
            self.M, self.K,
            scheme=self.config.time_scheme,
            config=stepper_config
        )

        # Initialize ionic states: (n_nodes, n_states)
        self.states = ionic_model.get_initial_state(mesh.n_nodes)
        if self.states.device != self.device:
            self.states = self.states.to(self.device)

        # Stimulus protocol
        self.stimulus_protocol = StimulusProtocol()

        # Simulation state
        self.t = 0.0

    def add_stimulus(
        self,
        region: Union[Callable, torch.Tensor],
        start_time: float,
        duration: float = 1.0,
        amplitude: float = -52.0
    ):
        """
        Add a stimulus event.

        Parameters
        ----------
        region : callable or tensor
            Function (x, y) -> bool mask, or precomputed mask
        start_time : float
            Start time (ms)
        duration : float
            Duration (ms)
        amplitude : float
            Amplitude (µA/µF), negative for depolarizing
        """
        self.stimulus_protocol.add_stimulus(
            region, start_time, duration, amplitude
        )

    def step(self, dt: Optional[float] = None) -> None:
        """
        Advance simulation by one time step.

        Uses operator splitting (Godunov/Lie):
        1. Ionic step: Advance ODE for gating/concentrations WITH stimulus
        2. Diffusion step: Solve PDE for voltage (no stimulus, handled in ionic)

        The ionic model handles: dV/dt = -(Iion - Istim)
        The diffusion step handles: dV/dt = (1/(chi*Cm)) * div(D*grad(V))

        Parameters
        ----------
        dt : float, optional
            Time step (uses config.dt if not specified)
        """
        if dt is None:
            dt = self.config.dt

        # Get current voltage
        V = self.ionic_model.get_voltage(self.states)

        # Compute stimulus current
        Istim = self.stimulus_protocol.get_current(
            self.t, self.x, self.y, self.device, self.dtype
        )

        # === Ionic step (explicit) ===
        # Advance gating variables, concentrations, AND voltage
        # The ionic model handles: dV/dt = -(Iion - Istim)
        self.states = self.ionic_model.step(self.states, dt, Istim)

        # Get updated voltage after ionic step
        V_ionic = self.ionic_model.get_voltage(self.states)

        # === Diffusion step (implicit) ===
        # Solve: M*dV/dt = -K*V (diffusion only, no source term)
        # The stimulus is already applied in the ionic step
        f = torch.zeros_like(V_ionic)

        # Get updated voltage from diffusion
        V_new = self.time_stepper.step(V_ionic, f, dt)

        # Update voltage in states
        self.states = self.ionic_model.set_voltage(self.states, V_new)

        # Advance time
        self.t += dt

    def run(
        self,
        t_end: float,
        dt: Optional[float] = None,
        save_interval: Optional[float] = None,
        progress_callback: Optional[Callable[[float, float], None]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run simulation until t_end.

        Parameters
        ----------
        t_end : float
            End time (ms)
        dt : float, optional
            Time step (ms)
        save_interval : float, optional
            Save voltage every N ms
        progress_callback : callable, optional
            Function(t, t_end) called each step

        Returns
        -------
        times : np.ndarray
            Time points (ms)
        voltages : np.ndarray
            Voltage at each save point, shape (n_times, n_nodes)
        """
        if dt is None:
            dt = self.config.dt
        if save_interval is None:
            save_interval = self.config.save_interval

        n_steps = int((t_end - self.t) / dt)
        save_every = max(1, int(save_interval / dt))

        times = []
        voltages = []

        for i in range(n_steps):
            self.step(dt)

            if i % save_every == 0:
                times.append(self.t)
                V = self.ionic_model.get_voltage(self.states)
                voltages.append(V.cpu().numpy().copy())

            if progress_callback is not None:
                progress_callback(self.t, t_end)

        return np.array(times), np.array(voltages)

    def get_voltage(self) -> torch.Tensor:
        """Get current voltage field."""
        return self.ionic_model.get_voltage(self.states)

    def set_voltage(self, V: torch.Tensor) -> None:
        """Set voltage field."""
        self.states = self.ionic_model.set_voltage(self.states, V)

    def get_state(self, name: str) -> torch.Tensor:
        """Get a specific state variable by name."""
        idx = self.ionic_model.state_names.index(name)
        return self.states[:, idx]

    def reset(self) -> None:
        """Reset simulation to initial state."""
        self.states = self.ionic_model.get_initial_state(self.mesh.n_nodes)
        if self.states.device != self.device:
            self.states = self.states.to(self.device)
        self.t = 0.0

        # Reset time stepper history if BDF
        if hasattr(self.time_stepper, 'reset_history'):
            self.time_stepper.reset_history()

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
            Voltage history, shape (n_times, n_nodes)
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
            Conduction velocity (cm/ms = m/s * 0.1)
        """
        x = self.x.cpu().numpy()

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

        return distance / time_diff  # cm/ms

    def __repr__(self) -> str:
        return (f"MonodomainSimulation(mesh={self.mesh}, "
                f"ionic_model={self.ionic_model.name}, "
                f"scheme={self.config.time_scheme}, t={self.t:.2f} ms)")
