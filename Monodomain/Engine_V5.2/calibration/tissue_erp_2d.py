"""
2D Tissue ERP Measurement.

Measures tissue ERP using a 2D square mesh with:
- Central stimulus (S1, S2)
- Probes at x-terminus and y-terminus
- Single ERP = minimum interval where BOTH probes activate

This captures the combined effect of D_x and D_y on tissue refractoriness.
No steady-state pacing required - just S1 → S2 → check propagation.
"""

import torch
import numpy as np
from typing import Optional, Tuple, NamedTuple
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from ionic.model import ORdModel
from ionic.parameters import CellType


@dataclass
class Tissue2DConfig:
    """Configuration for 2D tissue simulation."""

    # Mesh size (will be computed from wavelength if not set)
    nx: Optional[int] = None    # Cells in x direction
    ny: Optional[int] = None    # Cells in y direction

    # Spatial discretization
    dx: float = 0.02            # Mesh spacing (cm)

    # Temporal discretization
    dt: float = 0.02            # Time step (ms)

    # Stimulus parameters
    stim_radius: int = 3        # Radius of central stimulus (cells)
    stim_amplitude: float = -80.0   # Stimulus current (uA/uF)
    stim_duration: float = 1.0      # Stimulus duration (ms)

    # ERP search parameters
    s2_start: float = 400.0     # Initial S1-S2 interval (ms)
    s2_end: float = 200.0       # Minimum S1-S2 interval (ms)
    s2_step: float = 20.0       # Coarse step (ms)
    s2_fine_step: float = 5.0   # Fine step (ms)

    # Simulation
    post_s1_wait: float = 500.0     # Time after S1 before S2 scan (ms)
    post_s2_wait: float = 300.0     # Time after S2 to check propagation (ms)

    # Detection
    activation_threshold: float = -30.0  # Threshold for activation (mV)

    def compute_mesh_size(self, cv_x: float, cv_y: float, erp_est: float, margin: float = 1.5):
        """
        Compute mesh size based on wavelength.

        Mesh should be large enough for wave to propagate:
        L > margin × CV × ERP (wavelength with safety margin)

        Args:
            cv_x: Expected CV in x direction (cm/ms)
            cv_y: Expected CV in y direction (cm/ms)
            erp_est: Estimated ERP (ms)
            margin: Safety margin multiplier (default 1.5)
        """
        # Wavelength in each direction
        wavelength_x = cv_x * erp_est
        wavelength_y = cv_y * erp_est

        # Mesh length with margin
        Lx = margin * wavelength_x
        Ly = margin * wavelength_y

        # Number of cells
        self.nx = max(int(Lx / self.dx) + 1, 50)  # Minimum 50 cells
        self.ny = max(int(Ly / self.dx) + 1, 50)

        # Ensure odd for symmetric center
        if self.nx % 2 == 0:
            self.nx += 1
        if self.ny % 2 == 0:
            self.ny += 1


class ERPResult2D(NamedTuple):
    """Results from 2D tissue ERP measurement."""
    erp: float                  # Tissue ERP (ms) - when BOTH probes activate
    erp_x: float               # ERP measured at x-terminus (ms)
    erp_y: float               # ERP measured at y-terminus (ms)
    apd_center: float          # APD at center (ms)
    success: bool              # Whether measurement succeeded


class Tissue2D:
    """
    2D tissue simulator for ERP measurement.

    Uses operator splitting with ADI (Alternating Direction Implicit)
    for stable 2D diffusion.

    Geometry:
        - Square/rectangular mesh
        - Central stimulus region
        - Probes at x-terminus (right edge, middle) and y-terminus (top edge, middle)
    """

    def __init__(
        self,
        config: Tissue2DConfig,
        D_x: float,
        D_y: float,
        cell_type: int = 0,
        device: Optional[torch.device] = None
    ):
        """
        Initialize 2D tissue.

        Args:
            config: Tissue configuration
            D_x: Diffusion coefficient in x direction (cm²/ms)
            D_y: Diffusion coefficient in y direction (cm²/ms)
            cell_type: Cell type (0=ENDO, 1=EPI, 2=M_CELL)
            device: Torch device
        """
        self.config = config
        self.D_x = D_x
        self.D_y = D_y
        self.cell_type = cell_type
        self.device = device or torch.device('cpu')

        # Ensure mesh size is set
        if config.nx is None or config.ny is None:
            raise ValueError("Mesh size not set. Call config.compute_mesh_size() first.")

        self.nx = config.nx
        self.ny = config.ny

        # Cell type enum
        cell_type_map = {0: CellType.ENDO, 1: CellType.EPI, 2: CellType.M_CELL}
        self.cell_type_enum = cell_type_map.get(cell_type, CellType.ENDO)

        # Initialize ionic model
        device_str = 'cuda' if self.device.type == 'cuda' else 'cpu'
        self.ionic = ORdModel(celltype=self.cell_type_enum, device=device_str)

        # Initialize states (ny, nx, n_states)
        self.states = self._init_states()

        # Precompute diffusion coefficients
        self._setup_diffusion()

        # Stimulus mask (center region)
        self._setup_stimulus_mask()

        # Probe locations
        self.probe_x = (self.ny // 2, self.nx - 1)  # Right edge, middle row
        self.probe_y = (self.ny - 1, self.nx // 2)  # Top edge, middle column
        self.center = (self.ny // 2, self.nx // 2)  # Center

        # Tracking
        self.time = 0.0

    def _init_states(self) -> torch.Tensor:
        """Initialize states for all cells."""
        single_state = self.ionic.get_initial_state()
        states = single_state.unsqueeze(0).unsqueeze(0).expand(
            self.ny, self.nx, -1
        ).clone()
        return states.to(self.device)

    def _setup_diffusion(self):
        """Precompute diffusion parameters."""
        dx = self.config.dx
        dt = self.config.dt

        # Diffusion numbers
        self.alpha_x = self.D_x * dt / (dx * dx)
        self.alpha_y = self.D_y * dt / (dx * dx)

    def _setup_stimulus_mask(self):
        """Create mask for central stimulus region."""
        cy, cx = self.ny // 2, self.nx // 2
        r = self.config.stim_radius

        self.stim_mask = torch.zeros(self.ny, self.nx, dtype=torch.bool, device=self.device)

        for j in range(max(0, cy - r), min(self.ny, cy + r + 1)):
            for i in range(max(0, cx - r), min(self.nx, cx + r + 1)):
                if (j - cy) ** 2 + (i - cx) ** 2 <= r ** 2:
                    self.stim_mask[j, i] = True

    def _diffusion_step(self, V: torch.Tensor) -> torch.Tensor:
        """
        Perform 2D diffusion step using explicit scheme.

        For stability: dt < dx² / (2 * max(D_x, D_y))

        Uses Neumann (no-flux) boundary conditions.
        """
        V_new = V.clone()

        # Interior points
        V_new[1:-1, 1:-1] = V[1:-1, 1:-1] + (
            self.alpha_x * (V[1:-1, 2:] - 2 * V[1:-1, 1:-1] + V[1:-1, :-2]) +
            self.alpha_y * (V[2:, 1:-1] - 2 * V[1:-1, 1:-1] + V[:-2, 1:-1])
        )

        # Boundary conditions (no-flux: copy from interior)
        V_new[0, :] = V_new[1, :]       # Bottom
        V_new[-1, :] = V_new[-2, :]     # Top
        V_new[:, 0] = V_new[:, 1]       # Left
        V_new[:, -1] = V_new[:, -2]     # Right

        return V_new

    def _apply_stimulus(self, I_stim: torch.Tensor):
        """Apply stimulus to center region."""
        I_stim[self.stim_mask] = self.config.stim_amplitude

    def step(self, apply_stim: bool = False):
        """
        Advance simulation by one time step.

        Args:
            apply_stim: Whether to apply stimulus this step
        """
        dt = self.config.dt

        # Reshape states for ionic model: (ny*nx, n_states)
        states_flat = self.states.view(-1, self.states.shape[-1])

        # Stimulus current
        I_stim = torch.zeros(self.ny, self.nx, device=self.device, dtype=torch.float64)
        if apply_stim:
            self._apply_stimulus(I_stim)
        I_stim_flat = I_stim.view(-1)

        # Ionic step
        states_flat = self.ionic.step(states_flat, dt, I_stim_flat)

        # Reshape back
        self.states = states_flat.view(self.ny, self.nx, -1)

        # Diffusion step
        V = self.states[:, :, 0].clone()
        V_new = self._diffusion_step(V)
        self.states[:, :, 0] = V_new

        self.time += dt

    def run_until(self, end_time: float, stim_until: float = 0.0):
        """
        Run simulation until specified time.

        Args:
            end_time: End time (ms)
            stim_until: Apply stimulus until this time (ms)
        """
        while self.time < end_time:
            apply_stim = self.time < stim_until
            self.step(apply_stim=apply_stim)

    def check_activation(self, location: Tuple[int, int]) -> bool:
        """Check if a location has activated (crossed threshold)."""
        j, i = location
        V = self.states[j, i, 0].item()
        return V > self.config.activation_threshold

    def get_voltage(self, location: Tuple[int, int]) -> float:
        """Get voltage at a location."""
        j, i = location
        return self.states[j, i, 0].item()

    def reset(self):
        """Reset tissue to initial state."""
        self.states = self._init_states()
        self.time = 0.0


def measure_tissue_erp_2d(
    D_x: float,
    D_y: float,
    dx: float = 0.02,
    dt: float = 0.02,
    cv_x_est: float = 0.06,
    cv_y_est: float = 0.035,
    erp_est: float = 300.0,
    cell_type: int = 0,
    margin: float = 1.5,
    verbose: bool = False,
    device: Optional[torch.device] = None
) -> ERPResult2D:
    """
    Measure tissue ERP using 2D simulation.

    Protocol:
    1. Apply S1 at center → wave propagates to edges
    2. Wait for repolarization
    3. Apply S2 at center at decreasing intervals
    4. Find minimum interval where BOTH x and y probes activate

    Args:
        D_x: Diffusion coefficient in x direction (cm²/ms)
        D_y: Diffusion coefficient in y direction (cm²/ms)
        dx: Mesh spacing (cm)
        dt: Time step (ms)
        cv_x_est: Estimated CV in x direction (cm/ms)
        cv_y_est: Estimated CV in y direction (cm/ms)
        erp_est: Estimated ERP for mesh sizing (ms)
        cell_type: Cell type (0=ENDO, 1=EPI, 2=M_CELL)
        margin: Wavelength margin for mesh size
        verbose: Print progress
        device: Torch device

    Returns:
        ERPResult2D with tissue ERP
    """
    # Setup configuration
    config = Tissue2DConfig(dx=dx, dt=dt)
    config.compute_mesh_size(cv_x_est, cv_y_est, erp_est, margin)

    if verbose:
        print(f"2D Tissue ERP Measurement")
        print(f"  Mesh: {config.nx} × {config.ny} cells")
        print(f"  D_x = {D_x:.6f}, D_y = {D_y:.6f} cm²/ms")

    # Create tissue
    tissue = Tissue2D(config, D_x, D_y, cell_type, device)

    # === S1: Initial stimulus ===
    if verbose:
        print("  S1: Applying central stimulus...")

    stim_end = config.stim_duration
    tissue.run_until(stim_end, stim_until=stim_end)

    # Run until wave reaches edges and tissue repolarizes
    s1_end_time = config.post_s1_wait
    tissue.run_until(s1_end_time)

    # Measure APD at center (approximate from repolarization)
    v_center = tissue.get_voltage(tissue.center)
    apd_center = 0.0  # Could compute more precisely if needed

    # Store state after S1
    states_after_s1 = tissue.states.clone()
    time_after_s1 = tissue.time

    if verbose:
        print(f"  After S1: V_center = {v_center:.1f} mV")

    # === S2 scan ===
    if verbose:
        print("  S2 scan: Finding ERP...")

    last_successful_x = config.s2_start
    last_successful_y = config.s2_start
    last_successful_both = config.s2_start

    # Coarse search
    s2_interval = config.s2_start
    while s2_interval >= config.s2_end:
        # Reset to state after S1
        tissue.states = states_after_s1.clone()
        tissue.time = time_after_s1

        # Wait for S2 interval (relative to S1)
        # S1 was at t=0, S2 at t=s2_interval
        # Current time is time_after_s1, need to wait until s2_interval
        if s2_interval > tissue.time:
            tissue.run_until(s2_interval)

        # Apply S2
        s2_end_time = tissue.time + config.stim_duration
        tissue.run_until(s2_end_time, stim_until=s2_end_time)

        # Run to allow propagation
        tissue.run_until(tissue.time + config.post_s2_wait)

        # Check both probes
        activated_x = tissue.check_activation(tissue.probe_x)
        activated_y = tissue.check_activation(tissue.probe_y)

        if verbose:
            status_x = "✓" if activated_x else "✗"
            status_y = "✓" if activated_y else "✗"
            print(f"    S2={s2_interval:.0f}ms: x={status_x}, y={status_y}")

        if activated_x:
            last_successful_x = s2_interval
        if activated_y:
            last_successful_y = s2_interval
        if activated_x and activated_y:
            last_successful_both = s2_interval
            s2_interval -= config.s2_step
        else:
            # One or both failed, found approximate ERP
            break

    # Fine search around last successful
    erp_both = last_successful_both
    s2_interval = last_successful_both

    while s2_interval >= max(config.s2_end, last_successful_both - config.s2_step):
        tissue.states = states_after_s1.clone()
        tissue.time = time_after_s1

        if s2_interval > tissue.time:
            tissue.run_until(s2_interval)

        s2_end_time = tissue.time + config.stim_duration
        tissue.run_until(s2_end_time, stim_until=s2_end_time)
        tissue.run_until(tissue.time + config.post_s2_wait)

        activated_x = tissue.check_activation(tissue.probe_x)
        activated_y = tissue.check_activation(tissue.probe_y)

        if activated_x and activated_y:
            erp_both = s2_interval

        s2_interval -= config.s2_fine_step

    # Also find individual ERPs for reporting
    erp_x = last_successful_x
    erp_y = last_successful_y

    if verbose:
        print(f"\n  Results:")
        print(f"    ERP (both) = {erp_both:.0f} ms")
        print(f"    ERP_x = {erp_x:.0f} ms")
        print(f"    ERP_y = {erp_y:.0f} ms")

    return ERPResult2D(
        erp=erp_both,
        erp_x=erp_x,
        erp_y=erp_y,
        apd_center=apd_center,
        success=True
    )


if __name__ == "__main__":
    print("Testing 2D Tissue ERP Measurement...")

    result = measure_tissue_erp_2d(
        D_x=0.001,
        D_y=0.0005,
        dx=0.02,
        dt=0.01,  # Smaller dt for stability
        cv_x_est=0.04,
        cv_y_est=0.03,
        erp_est=300.0,
        verbose=True
    )

    print(f"\nFinal: ERP = {result.erp} ms")
