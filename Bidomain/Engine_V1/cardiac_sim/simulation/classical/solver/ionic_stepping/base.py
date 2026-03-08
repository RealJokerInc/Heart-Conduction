"""
IonicSolver Abstract Base Class

Owns an IonicModel. Advances ionic ODEs in-place on state.
Evaluates Istim internally from state's stimulus data.

Ref: improvement.md:L1006-1031
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from ....ionic.base import IonicModel
    from ...state import BidomainState


class IonicSolver(ABC):
    """
    Abstract base class for ionic time integrators.

    Owns an IonicModel (computation provider). Advances ionic variables
    in-place on BidomainState.

    The solver is responsible for:
    - Evaluating stimulus current from state's stimulus data
    - Integrating voltage, gating variables, and concentrations
    - Choosing the numerical method (Rush-Larsen, Forward Euler, etc.)
    """

    def __init__(self, ionic_model: 'IonicModel'):
        """
        Initialize ionic solver.

        Parameters
        ----------
        ionic_model : IonicModel
            The ionic model providing computation functions
        """
        self.ionic_model = ionic_model

    @abstractmethod
    def step(self, state: 'BidomainState', dt: float) -> None:
        """
        Advance ionic variables by dt.

        Modifies state.V and state.ionic_states in-place.

        Parameters
        ----------
        state : BidomainState
            Simulation state (modified in-place)
        dt : float
            Time step (ms)
        """
        pass

    def _evaluate_Istim(self, state: 'BidomainState') -> torch.Tensor:
        """
        Compute stimulus current at current time from state's stimulus data.

        Parameters
        ----------
        state : BidomainState
            Contains t, stim_masks, stim_starts, stim_durations, stim_amplitudes

        Returns
        -------
        Istim : torch.Tensor
            Stimulus current (n_dof,), in uA/uF
        """
        device = state.V.device
        dtype = state.V.dtype
        Istim = torch.zeros(state.n_dof, device=device, dtype=dtype)

        for i in range(len(state.stim_starts)):
            t_start = state.stim_starts[i]
            t_end = t_start + state.stim_durations[i]
            if t_start <= state.t < t_end:
                Istim = Istim + state.stim_amplitudes[i] * state.stim_masks[i]

        return Istim

    def step_with_V(
        self,
        ionic_states: torch.Tensor,
        V: torch.Tensor,
        Istim: torch.Tensor,
        dt: float
    ) -> None:
        """
        Advance ionic variables with externally provided V and Istim.

        For use with LBM where diffusion is handled separately and
        voltage comes from LBM distribution recovery.

        This method only updates gates and concentrations, NOT voltage.
        Voltage update is handled by the LBM collision step.

        Parameters
        ----------
        ionic_states : torch.Tensor
            Ionic state tensor (n_dof, n_states), modified in-place
        V : torch.Tensor
            Current voltage (n_dof,)
        Istim : torch.Tensor
            Stimulus current (n_dof,)
        dt : float
            Time step (ms)
        """
        model = self.ionic_model

        # Compute gate parameters from current state
        gate_inf = model.compute_gate_steady_states(V, ionic_states)
        gate_tau = model.compute_gate_time_constants(V, ionic_states)

        # Update gates (subclass-specific integration)
        self._update_gates(ionic_states, gate_inf, gate_tau, dt)

        # Update concentrations
        conc_rates = model.compute_concentration_rates(V, ionic_states)
        conc_indices = model.concentration_indices
        for i, idx in enumerate(conc_indices):
            ionic_states[:, idx] = ionic_states[:, idx] + dt * conc_rates[:, i]

    def _update_gates(
        self,
        ionic_states: torch.Tensor,
        gate_inf: torch.Tensor,
        gate_tau: torch.Tensor,
        dt: float
    ) -> None:
        """
        Update gating variables. Override in subclasses.

        Default: Forward Euler.
        """
        model = self.ionic_model
        gate_indices = model.gate_indices
        for i, idx in enumerate(gate_indices):
            x = ionic_states[:, idx]
            x_inf = gate_inf[:, i]
            tau = gate_tau[:, i]
            # Forward Euler: dx/dt = (x_inf - x) / tau
            ionic_states[:, idx] = x + dt * (x_inf - x) / tau
