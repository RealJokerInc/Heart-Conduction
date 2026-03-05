"""
IonicModel Abstract Base Class

Defines the interface that all ionic models (ORd, TTP06, etc.) must implement.
Based on openCARP's LIMPET pattern for model interoperability.

V is always separate from ionic_states. The ionic model is a pure data provider:
it computes Iion, gate steady-states, gate time constants, and concentration
rates given V and ionic_states as separate inputs.
"""

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import List, Tuple, Optional, Dict
import torch


class CellType(IntEnum):
    """Cell type variants with different ion channel expression."""
    ENDO = 0    # Endocardial
    EPI = 1     # Epicardial
    M_CELL = 2  # Mid-myocardial (M-cell)


class IonicModel(ABC):
    """
    Abstract base class for cardiac ionic models.

    All ionic models must implement this interface to be compatible with
    the tissue simulation framework.

    V (membrane potential) is always stored and passed separately from
    ionic_states (gates + concentrations).

    Attributes
    ----------
    device : torch.device
        Computation device (cuda or cpu)
    dtype : torch.dtype
        Data type (float64 for numerical stability)
    """

    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device)
        self.dtype = torch.float64

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name (e.g., 'ORd', 'TTP06')."""
        pass

    @property
    @abstractmethod
    def n_states(self) -> int:
        """Number of ionic state variables (excludes V)."""
        pass

    @property
    @abstractmethod
    def V_rest(self) -> float:
        """Resting membrane potential (mV)."""
        pass

    @property
    @abstractmethod
    def state_names(self) -> Tuple[str, ...]:
        """Names of all ionic state variables in order (excludes V)."""
        pass

    @property
    @abstractmethod
    def gate_indices(self) -> List[int]:
        """Indices of gating variables in ionic_states (Rush-Larsen targets)."""
        pass

    @property
    @abstractmethod
    def concentration_indices(self) -> List[int]:
        """Indices of concentration variables in ionic_states (Forward Euler targets)."""
        pass

    @abstractmethod
    def get_initial_state(self, n_cells: int = 1) -> torch.Tensor:
        """
        Get initial ionic state tensor (excludes V).

        Parameters
        ----------
        n_cells : int
            Number of cells (1 for single cell, >1 for tissue)

        Returns
        -------
        ionic_states : torch.Tensor
            Initial ionic state tensor of shape (n_cells, n_states) or (n_states,) if n_cells=1
        """
        pass

    @abstractmethod
    def step(self, V: torch.Tensor, ionic_states: torch.Tensor, dt: float,
             Istim: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Advance model by one time step.

        Parameters
        ----------
        V : torch.Tensor
            Membrane potential (mV), shape (...,)
        ionic_states : torch.Tensor
            Current ionic state (..., n_states)
        dt : float
            Time step (ms)
        Istim : torch.Tensor, optional
            Stimulus current (uA/uF), same shape as V or scalar

        Returns
        -------
        V_new : torch.Tensor
            Updated membrane potential
        ionic_states_new : torch.Tensor
            Updated ionic state tensor
        """
        pass

    @abstractmethod
    def compute_Iion(self, V: torch.Tensor, ionic_states: torch.Tensor) -> torch.Tensor:
        """
        Compute total ionic current.

        Parameters
        ----------
        V : torch.Tensor
            Membrane potential (...,)
        ionic_states : torch.Tensor
            Ionic state tensor (..., n_states)

        Returns
        -------
        Iion : torch.Tensor
            Total ionic current (uA/uF)
        """
        pass

    @abstractmethod
    def compute_gate_steady_states(self, V: torch.Tensor, ionic_states: torch.Tensor) -> torch.Tensor:
        """
        Compute steady-state values for all gating variables.

        Parameters
        ----------
        V : torch.Tensor
            Membrane potential (...,)
        ionic_states : torch.Tensor
            Ionic state tensor (..., n_states). Most gates depend only on V,
            but some depend on Ca concentrations or CaMKa.

        Returns
        -------
        gate_inf : torch.Tensor
            Steady-state values (n_cells, n_gates), ordered by gate_indices.
        """
        pass

    @abstractmethod
    def compute_gate_time_constants(self, V: torch.Tensor, ionic_states: torch.Tensor) -> torch.Tensor:
        """
        Compute time constants for all gating variables.

        Parameters
        ----------
        V : torch.Tensor
            Membrane potential (...,)
        ionic_states : torch.Tensor
            Ionic state tensor (..., n_states)

        Returns
        -------
        gate_tau : torch.Tensor
            Time constants in ms (n_cells, n_gates), ordered by gate_indices.
        """
        pass

    @abstractmethod
    def compute_concentration_rates(self, V: torch.Tensor, ionic_states: torch.Tensor) -> torch.Tensor:
        """
        Compute time derivatives for all concentration variables.

        Computes ionic currents internally and returns dc/dt for each
        concentration variable (Forward Euler targets).

        Parameters
        ----------
        V : torch.Tensor
            Membrane potential (...,)
        ionic_states : torch.Tensor
            Ionic state tensor (..., n_states)

        Returns
        -------
        rates : torch.Tensor
            Time derivatives (n_cells, n_concentrations), ordered by
            concentration_indices.
        """
        pass

    def run(self, t_end: float, dt: float = 0.01,
            stim_times: Optional[list] = None,
            stim_duration: float = 1.0,
            stim_amplitude: float = -80.0,
            save_interval: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run single-cell simulation.

        Parameters
        ----------
        t_end : float
            End time (ms)
        dt : float
            Time step (ms)
        stim_times : list, optional
            Times to apply stimulus (ms)
        stim_duration : float
            Stimulus duration (ms)
        stim_amplitude : float
            Stimulus amplitude (uA/uF)
        save_interval : float, optional
            Interval for saving states (ms)

        Returns
        -------
        t : torch.Tensor
            Time points
        V : torch.Tensor
            Voltage trace
        """
        if stim_times is None:
            stim_times = [10.0]

        V = torch.tensor(self.V_rest, dtype=self.dtype, device=self.device)
        ionic_states = self.get_initial_state(n_cells=1)

        if save_interval is None:
            save_interval = dt

        n_steps = int(t_end / dt)
        save_every = max(1, int(save_interval / dt))

        t_list = []
        V_list = []

        for i in range(n_steps):
            t = i * dt

            # Check stimulus
            Istim = 0.0
            for t_stim in stim_times:
                if t_stim <= t < t_stim + stim_duration:
                    Istim = stim_amplitude
                    break

            Istim_tensor = torch.tensor(Istim, dtype=self.dtype, device=self.device)
            V, ionic_states = self.step(V, ionic_states, dt, Istim_tensor)

            if i % save_every == 0:
                t_list.append(t)
                V_list.append(V.item())

        return (torch.tensor(t_list, device=self.device),
                torch.tensor(V_list, device=self.device))
