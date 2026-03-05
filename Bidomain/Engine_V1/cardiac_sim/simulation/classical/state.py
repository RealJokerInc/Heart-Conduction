"""
BidomainState — Runtime Data Container

Holds runtime simulation data for bidomain. No algorithms — pure data.
Extends monodomain SimulationState with phi_e.

V (transmembrane potential Vm) is stored as a separate field from ionic_states
(gates + concentrations). phi_e (extracellular potential) is a second field.

Ref: improvement.md L599-656
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional
import torch

if TYPE_CHECKING:
    from .discretization.base import BidomainSpatialDiscretization


@dataclass
class BidomainState:
    """
    Runtime state for bidomain cardiac simulation.

    Vm and phi_e are separate 1D fields. ionic_states holds gates +
    concentrations only (no V).

    Attributes
    ----------
    spatial : BidomainSpatialDiscretization
        Reference to FDM discretization
    n_dof : int
        Number of degrees of freedom
    x, y : torch.Tensor
        Coordinates (n_dof,)
    Vm : torch.Tensor
        Transmembrane potential (n_dof,)
    phi_e : torch.Tensor
        Extracellular potential (n_dof,)
    ionic_states : torch.Tensor
        Gates + concentrations (n_dof, n_ionic_states)
    """
    # Discretization reference
    spatial: 'BidomainSpatialDiscretization'

    # Abstract geometry
    n_dof: int
    x: torch.Tensor
    y: torch.Tensor

    # Potentials (both separate 1D fields)
    Vm: torch.Tensor             # (n_dof,) — transmembrane
    phi_e: torch.Tensor          # (n_dof,) — extracellular

    # Ionic states (gates + concentrations only, no V)
    ionic_states: torch.Tensor

    # State layout (from ionic model)
    gate_indices: List[int]
    concentration_indices: List[int]

    # Time
    t: float = 0.0

    # Stimulus data (separate intracellular/extracellular amplitudes)
    stim_masks: torch.Tensor = None
    stim_starts: List[float] = field(default_factory=list)
    stim_durations: List[float] = field(default_factory=list)
    stim_amplitudes: List[float] = field(default_factory=list)       # Intracellular
    stim_amplitudes_e: List[float] = field(default_factory=list)     # Extracellular

    # Output buffers
    output_buffer_Vm: Optional[torch.Tensor] = None
    output_buffer_phi_e: Optional[torch.Tensor] = None
    buffer_idx: int = 0

    def __post_init__(self):
        """Initialize empty stimulus data if not provided."""
        if self.stim_masks is None:
            device = self.Vm.device
            dtype = self.Vm.dtype
            self.stim_masks = torch.zeros(0, self.n_dof, device=device, dtype=dtype)
            self.stim_starts = []
            self.stim_durations = []
            self.stim_amplitudes = []
            self.stim_amplitudes_e = []

    @property
    def V(self) -> torch.Tensor:
        """Alias for Vm — compatibility with IonicSolver which accesses state.V."""
        return self.Vm

    @V.setter
    def V(self, value: torch.Tensor):
        """Setter for V alias."""
        self.Vm = value

    @property
    def Vm_flat(self) -> torch.Tensor:
        """Return Vm as 1D (n_dof,) — no-op for classical."""
        return self.Vm

    @property
    def phi_e_flat(self) -> torch.Tensor:
        """Return phi_e as 1D (n_dof,) — no-op for classical."""
        return self.phi_e
