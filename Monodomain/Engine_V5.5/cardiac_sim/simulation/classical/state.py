"""
SimulationState — Runtime Data Container

Holds runtime simulation data. No algorithms — pure data.
Operators and workspace buffers live in their respective solvers.

State is scheme-agnostic: works with FEM, FDM, or FVM.

V (membrane potential) is stored as a separate field from ionic_states
(gates + concentrations). This matches the PDE structure where V is the
coupling variable and ionic_states are local ODE variables.

Ref: improvement.md:L521-594
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional
import torch

if TYPE_CHECKING:
    from .discretization_scheme.base import SpatialDiscretization


@dataclass
class SimulationState:
    """
    Runtime state for cardiac simulation.

    V is a direct field (n_dof,). ionic_states holds gates + concentrations
    only (n_dof, n_ionic_states). V is NOT embedded in ionic_states.

    Attributes
    ----------
    spatial : SpatialDiscretization
        Reference to FEM/FDM/FVM discretization
    n_dof : int
        Number of degrees of freedom
    x, y : torch.Tensor
        Coordinates (n_dof,)
    V : torch.Tensor
        Membrane potential (n_dof,)
    ionic_states : torch.Tensor
        Gates + concentrations (n_dof, n_ionic_states)
    gate_indices : List[int]
        Column indices for gating variables in ionic_states
    concentration_indices : List[int]
        Column indices for concentrations in ionic_states
    t : float
        Current simulation time (ms)
    stim_masks : torch.Tensor
        Stimulus masks (n_stimuli, n_dof)
    stim_starts : List[float]
        Stimulus start times (ms)
    stim_durations : List[float]
        Stimulus durations (ms)
    stim_amplitudes : List[float]
        Stimulus amplitudes (uA/uF)
    output_buffer : torch.Tensor, optional
        Buffer for saving output snapshots
    buffer_idx : int
        Current index in output buffer
    """
    # Discretization reference
    spatial: 'SpatialDiscretization'

    # Abstract geometry
    n_dof: int
    x: torch.Tensor
    y: torch.Tensor

    # Voltage (separate from ionic states)
    V: torch.Tensor

    # Ionic states (gates + concentrations only, no V)
    ionic_states: torch.Tensor

    # State layout (from ionic model)
    gate_indices: List[int]
    concentration_indices: List[int]

    # Time
    t: float = 0.0

    # Stimulus data
    stim_masks: torch.Tensor = None
    stim_starts: List[float] = field(default_factory=list)
    stim_durations: List[float] = field(default_factory=list)
    stim_amplitudes: List[float] = field(default_factory=list)

    # Output
    output_buffer: Optional[torch.Tensor] = None
    buffer_idx: int = 0

    def __post_init__(self):
        """Initialize empty stimulus data if not provided."""
        if self.stim_masks is None:
            device = self.V.device
            dtype = self.V.dtype
            self.stim_masks = torch.zeros(0, self.n_dof, device=device, dtype=dtype)
            self.stim_starts = []
            self.stim_durations = []
            self.stim_amplitudes = []

    @property
    def V_flat(self) -> torch.Tensor:
        """Return V as 1D (n_dof,) — no-op for classical (already 1D)."""
        return self.V
