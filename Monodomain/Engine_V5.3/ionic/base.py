"""
IonicModel Abstract Base Class

Defines the interface that all ionic models (ORd, TTP06, etc.) must implement.
Based on openCARP's LIMPET pattern for model interoperability.
"""

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Tuple, Optional, Dict
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
        """Number of state variables."""
        pass

    @property
    @abstractmethod
    def state_names(self) -> Tuple[str, ...]:
        """Names of all state variables in order."""
        pass

    @property
    @abstractmethod
    def V_index(self) -> int:
        """Index of membrane potential in state vector."""
        pass

    @abstractmethod
    def get_initial_state(self, n_cells: int = 1) -> torch.Tensor:
        """
        Get initial state tensor.

        Parameters
        ----------
        n_cells : int
            Number of cells (1 for single cell, >1 for tissue)

        Returns
        -------
        state : torch.Tensor
            Initial state tensor of shape (n_cells, n_states) or (n_states,) if n_cells=1
        """
        pass

    @abstractmethod
    def step(self, states: torch.Tensor, dt: float,
             Istim: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Advance model by one time step.

        Parameters
        ----------
        states : torch.Tensor
            Current state (..., n_states)
        dt : float
            Time step (ms)
        Istim : torch.Tensor, optional
            Stimulus current (µA/µF), same shape as V or scalar

        Returns
        -------
        states : torch.Tensor
            Updated state tensor
        """
        pass

    @abstractmethod
    def compute_Iion(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute total ionic current.

        Parameters
        ----------
        states : torch.Tensor
            State tensor (..., n_states)

        Returns
        -------
        Iion : torch.Tensor
            Total ionic current (µA/µF)
        """
        pass

    def get_voltage(self, states: torch.Tensor) -> torch.Tensor:
        """
        Extract voltage from states.

        Parameters
        ----------
        states : torch.Tensor
            State tensor (..., n_states)

        Returns
        -------
        V : torch.Tensor
            Membrane potential (mV)
        """
        return states[..., self.V_index]

    def set_voltage(self, states: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Set voltage in states (for tissue coupling).

        Parameters
        ----------
        states : torch.Tensor
            State tensor (..., n_states)
        V : torch.Tensor
            New membrane potential (mV)

        Returns
        -------
        states : torch.Tensor
            Updated state tensor
        """
        states[..., self.V_index] = V
        return states

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
            Stimulus amplitude (µA/µF)
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

        state = self.get_initial_state(n_cells=1)

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
            state = self.step(state, dt, Istim_tensor)

            if i % save_every == 0:
                t_list.append(t)
                V_list.append(self.get_voltage(state).item())

        return (torch.tensor(t_list, device=self.device),
                torch.tensor(V_list, device=self.device))
