"""
1D Cable Solver for Calibration.

Provides a minimal 1D monodomain solver for measuring:
- Conduction velocity (CV)
- Action potential duration (APD)
- Effective refractory period (ERP) via S1-S2 protocol

Uses the validated O'Hara-Rudy ionic model from ionic/ module.
"""

import torch
import numpy as np
from typing import Tuple, Optional, NamedTuple
from dataclasses import dataclass

# Import from parent ionic module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from ionic.model import ORdModel
from ionic.parameters import CellType


@dataclass
class CableConfig:
    """Configuration for 1D cable simulation."""

    # Spatial discretization
    length: float = 2.0        # Cable length (cm)
    dx: float = 0.01           # Mesh spacing (cm)

    # Temporal discretization
    dt: float = 0.02           # Time step (ms)

    # Tissue properties
    chi: float = 1400.0        # Surface-to-volume ratio (cm^-1)
    Cm: float = 1.0            # Membrane capacitance (uF/cm^2)

    # Stimulus
    stim_amplitude: float = -80.0   # Stimulus current (uA/uF)
    stim_duration: float = 1.0      # Stimulus duration (ms)
    stim_width: int = 5             # Number of cells to stimulate

    # Measurement
    cv_probe_start: float = 0.3     # Start position for CV measurement (cm)
    cv_probe_end: float = 1.7       # End position for CV measurement (cm)
    activation_threshold: float = -30.0  # Threshold for activation (mV)

    @property
    def n_cells(self) -> int:
        """Number of cells in the cable."""
        return int(self.length / self.dx) + 1

    @property
    def probe_indices(self) -> Tuple[int, int]:
        """Indices for CV measurement probes."""
        i_start = int(self.cv_probe_start / self.dx)
        i_end = int(self.cv_probe_end / self.dx)
        return i_start, i_end


class MeasurementResult(NamedTuple):
    """Results from a cable simulation."""
    cv: float                    # Conduction velocity (cm/ms)
    apd90: float                 # APD at 90% repolarization (ms)
    apd50: float                 # APD at 50% repolarization (ms)
    v_max: float                 # Maximum voltage (mV)
    v_min: float                 # Minimum voltage (mV)
    dv_dt_max: float            # Maximum dV/dt (mV/ms)
    activation_time: float       # Time to activation at center (ms)
    success: bool                # Whether simulation completed successfully


class Cable1D:
    """
    1D cable solver for calibration measurements.

    Uses operator splitting:
    1. Ionic step: Rush-Larsen update for gating, forward Euler for concentrations
    2. Diffusion step: Implicit solve for spatial coupling

    The diffusion equation in 1D:
        χ·Cm·∂V/∂t = D·∂²V/∂x² - χ·Iion

    Discretized with FEM (linear elements) or FDM:
        ∂²V/∂x² ≈ (V[i-1] - 2V[i] + V[i+1]) / dx²
    """

    def __init__(
        self,
        config: CableConfig,
        D: float,
        cell_type: int = 0,
        device: Optional[torch.device] = None
    ):
        """
        Initialize 1D cable solver.

        Args:
            config: Cable configuration
            D: Diffusion coefficient (cm²/ms)
            cell_type: Cell type (0=ENDO, 1=EPI, 2=M_CELL)
            device: Torch device (defaults to CPU)
        """
        self.config = config
        self.D = D
        self.cell_type = cell_type
        self.device = device or torch.device('cpu')

        # Map int cell type to CellType enum
        cell_type_map = {0: CellType.ENDO, 1: CellType.EPI, 2: CellType.M_CELL}
        self.cell_type_enum = cell_type_map.get(cell_type, CellType.ENDO)

        # Initialize ionic model
        device_str = 'cuda' if self.device.type == 'cuda' else 'cpu'
        self.ionic = ORdModel(celltype=self.cell_type_enum, device=device_str)

        # Initialize state (n_cells x n_states)
        self.n_cells = config.n_cells
        self.states = self._init_states()

        # Build diffusion matrix
        self._build_diffusion_matrix()

        # Tracking
        self.time = 0.0
        self.activation_times = torch.full(
            (self.n_cells,), float('inf'), device=self.device, dtype=torch.float64
        )
        self.activated = torch.zeros(self.n_cells, dtype=torch.bool, device=self.device)

    def _init_states(self) -> torch.Tensor:
        """Initialize states for all cells at resting potential."""
        # Get single-cell initial state from ORdModel
        single_state = self.ionic.get_initial_state()

        # Replicate for all cells
        states = single_state.unsqueeze(0).expand(self.n_cells, -1).clone()
        return states.to(self.device)

    def _build_diffusion_matrix(self):
        """
        Build the implicit diffusion operator.

        For 1D with Neumann BCs (no-flux):
            A·V_new = V_old

        Where A = I - α·L, with:
            α = D·dt / dx²
            L = tridiagonal Laplacian

        Note: We use the simpler diffusion formulation ∂V/∂t = D·∂²V/∂x²
        since the ionic model already handles Cm internally.
        D is the effective diffusion coefficient in cm²/ms.
        """
        n = self.n_cells
        dx = self.config.dx
        dt = self.config.dt

        # Diffusion number (simplified formulation)
        alpha = self.D * dt / (dx * dx)

        # Build tridiagonal matrix
        # Main diagonal: 1 + 2α (interior), 1 + α (boundaries)
        # Off diagonals: -α

        main_diag = torch.ones(n, device=self.device, dtype=torch.float64) * (1.0 + 2.0 * alpha)
        main_diag[0] = 1.0 + alpha      # Left boundary (no-flux)
        main_diag[-1] = 1.0 + alpha     # Right boundary (no-flux)

        off_diag = torch.ones(n - 1, device=self.device, dtype=torch.float64) * (-alpha)

        # Store for Thomas algorithm
        self.main_diag = main_diag
        self.off_diag = off_diag
        self.alpha = alpha

        # Pre-compute LU decomposition for Thomas algorithm
        self._factor_tridiagonal()

    def _factor_tridiagonal(self):
        """Pre-compute factors for Thomas algorithm."""
        n = self.n_cells

        # Modified coefficients
        c_prime = torch.zeros(n - 1, device=self.device, dtype=torch.float64)

        c_prime[0] = self.off_diag[0] / self.main_diag[0]
        for i in range(1, n - 1):
            c_prime[i] = self.off_diag[i] / (
                self.main_diag[i] - self.off_diag[i-1] * c_prime[i-1]
            )

        self.c_prime = c_prime

        # Store modified main diagonal
        d_mod = torch.zeros(n, device=self.device, dtype=torch.float64)
        d_mod[0] = self.main_diag[0]
        for i in range(1, n):
            d_mod[i] = self.main_diag[i] - self.off_diag[i-1] * c_prime[i-1]

        self.d_mod = d_mod

    def _solve_diffusion(self, V: torch.Tensor) -> torch.Tensor:
        """
        Solve the implicit diffusion step using Thomas algorithm.

        Args:
            V: Voltage vector (n_cells,)

        Returns:
            Updated voltage after diffusion
        """
        n = self.n_cells

        # Forward sweep (modified RHS)
        d_prime = torch.zeros(n, device=self.device, dtype=torch.float64)
        d_prime[0] = V[0] / self.d_mod[0]

        for i in range(1, n):
            d_prime[i] = (V[i] - self.off_diag[i-1] * d_prime[i-1]) / self.d_mod[i]

        # Back substitution
        V_new = torch.zeros(n, device=self.device, dtype=torch.float64)
        V_new[-1] = d_prime[-1]

        for i in range(n - 2, -1, -1):
            V_new[i] = d_prime[i] - self.c_prime[i] * V_new[i + 1]

        return V_new

    def _apply_stimulus(self, I_stim: torch.Tensor):
        """Apply stimulus current to first few cells."""
        if self.time < self.config.stim_duration:
            I_stim[:self.config.stim_width] = self.config.stim_amplitude

    def step(self):
        """
        Advance simulation by one time step.

        Uses Strang splitting:
        1. Half ionic step
        2. Full diffusion step
        3. Half ionic step
        """
        dt = self.config.dt

        # Stimulus current (same shape as V)
        I_stim = torch.zeros(self.n_cells, device=self.device, dtype=torch.float64)
        self._apply_stimulus(I_stim)

        # === Half ionic step ===
        # ORdModel.step(states, dt, Istim) - cell type already set in model
        self.states = self.ionic.step(self.states, dt / 2.0, I_stim)

        # === Full diffusion step ===
        V = self.states[:, 0].clone()
        V_new = self._solve_diffusion(V)
        self.states[:, 0] = V_new

        # === Half ionic step ===
        self.states = self.ionic.step(self.states, dt / 2.0, I_stim)

        # Track activation times
        V_current = self.states[:, 0]
        newly_activated = (
            (V_current > self.config.activation_threshold) &
            (~self.activated)
        )
        self.activation_times[newly_activated] = self.time
        self.activated |= newly_activated

        self.time += dt

    def run(
        self,
        duration: float,
        verbose: bool = False
    ) -> MeasurementResult:
        """
        Run simulation for specified duration.

        Args:
            duration: Simulation duration (ms)
            verbose: Print progress updates

        Returns:
            MeasurementResult with CV, APD, etc.
        """
        n_steps = int(duration / self.config.dt)

        # Track center cell for APD measurement
        center_idx = self.n_cells // 2
        v_trace = []
        time_trace = []

        # Store initial values
        v_max = float('-inf')
        v_min = float('inf')
        dv_dt_max = 0.0
        v_prev = self.states[center_idx, 0].item()

        for step_i in range(n_steps):
            self.step()

            v_current = self.states[center_idx, 0].item()
            v_trace.append(v_current)
            time_trace.append(self.time)

            v_max = max(v_max, v_current)
            v_min = min(v_min, v_current)

            dv_dt = (v_current - v_prev) / self.config.dt
            dv_dt_max = max(dv_dt_max, dv_dt)
            v_prev = v_current

            if verbose and step_i % 1000 == 0:
                print(f"  t = {self.time:.1f} ms, V_center = {v_current:.1f} mV")

        # Compute CV from activation times
        i_start, i_end = self.config.probe_indices
        t_start = self.activation_times[i_start].item()
        t_end = self.activation_times[i_end].item()

        if np.isinf(t_start) or np.isinf(t_end):
            # Propagation failed
            cv = 0.0
            success = False
        else:
            distance = (i_end - i_start) * self.config.dx
            cv = distance / (t_end - t_start)
            success = True

        # Compute APD90 and APD50
        v_trace = np.array(v_trace)
        time_trace = np.array(time_trace)

        apd90, apd50 = self._compute_apd(v_trace, time_trace)

        activation_time = self.activation_times[center_idx].item()

        return MeasurementResult(
            cv=cv,
            apd90=apd90,
            apd50=apd50,
            v_max=v_max,
            v_min=v_min,
            dv_dt_max=dv_dt_max,
            activation_time=activation_time,
            success=success
        )

    def _compute_apd(
        self,
        v_trace: np.ndarray,
        time_trace: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute APD90 and APD50 from voltage trace.

        Args:
            v_trace: Voltage trace at measurement point
            time_trace: Corresponding time points

        Returns:
            (APD90, APD50) in ms
        """
        # Find peak
        i_peak = np.argmax(v_trace)
        v_peak = v_trace[i_peak]
        t_peak = time_trace[i_peak]

        # Find resting potential (use last 10% of trace)
        v_rest = np.mean(v_trace[-len(v_trace)//10:])

        # Amplitude
        amplitude = v_peak - v_rest

        if amplitude < 50:  # No action potential
            return 0.0, 0.0

        # APD thresholds
        v_90 = v_rest + 0.1 * amplitude  # 90% repolarization
        v_50 = v_rest + 0.5 * amplitude  # 50% repolarization

        # Find crossing times after peak
        v_after_peak = v_trace[i_peak:]
        t_after_peak = time_trace[i_peak:]

        # APD90
        idx_90 = np.where(v_after_peak < v_90)[0]
        if len(idx_90) > 0:
            apd90 = t_after_peak[idx_90[0]] - t_peak
        else:
            apd90 = 0.0

        # APD50
        idx_50 = np.where(v_after_peak < v_50)[0]
        if len(idx_50) > 0:
            apd50 = t_after_peak[idx_50[0]] - t_peak
        else:
            apd50 = 0.0

        return apd90, apd50

    def reset(self):
        """Reset cable to initial state."""
        self.states = self._init_states()
        self.time = 0.0
        self.activation_times = torch.full(
            (self.n_cells,), float('inf'), device=self.device
        )
        self.activated = torch.zeros(
            self.n_cells, dtype=torch.bool, device=self.device
        )


def measure_cv_apd(
    D: float,
    dx: float = 0.01,
    dt: float = 0.02,
    cable_length: float = 2.0,
    duration: float = 500.0,
    cell_type: int = 0,
    device: Optional[torch.device] = None
) -> MeasurementResult:
    """
    Convenience function to measure CV and APD for given D.

    Args:
        D: Diffusion coefficient (cm²/ms)
        dx: Mesh spacing (cm)
        dt: Time step (ms)
        cable_length: Total cable length (cm)
        duration: Simulation duration (ms)
        cell_type: Cell type (0=ENDO, 1=EPI, 2=M_CELL)
        device: Torch device

    Returns:
        MeasurementResult with CV, APD, etc.
    """
    config = CableConfig(
        length=cable_length,
        dx=dx,
        dt=dt
    )

    cable = Cable1D(config, D=D, cell_type=cell_type, device=device)
    return cable.run(duration)


if __name__ == "__main__":
    # Quick test
    print("Testing 1D cable solver...")

    D = 0.001  # cm²/ms
    result = measure_cv_apd(D, dx=0.01, dt=0.02, duration=300.0)

    print(f"\nResults for D = {D} cm²/ms:")
    print(f"  CV = {result.cv:.4f} cm/ms = {result.cv * 10:.1f} m/s")
    print(f"  APD90 = {result.apd90:.1f} ms")
    print(f"  APD50 = {result.apd50:.1f} ms")
    print(f"  V_max = {result.v_max:.1f} mV")
    print(f"  V_min = {result.v_min:.1f} mV")
    print(f"  dV/dt_max = {result.dv_dt_max:.1f} mV/ms")
    print(f"  Success = {result.success}")
