"""
Stimulus Protocol Definition

Provides stimulus event definition and pacing protocol management.

Migrated from V5.3 tissue/stimulus.py (Stimulus + StimulusProtocol classes).
"""

from dataclasses import dataclass, field
from typing import Callable, List, Union
import torch


@dataclass
class Stimulus:
    """
    Definition of a single stimulus event.

    Attributes
    ----------
    region : callable or torch.Tensor
        Either a function (x, y) -> bool mask, or a precomputed mask tensor
    start_time : float
        Start time of stimulus (ms)
    duration : float
        Duration of stimulus (ms)
    amplitude : float
        Stimulus amplitude (uA/uF), typically negative for depolarizing
    """
    region: Union[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], torch.Tensor]
    start_time: float
    duration: float
    amplitude: float = -52.0  # Default: standard stimulus amplitude

    def is_active(self, t: float) -> bool:
        """Check if stimulus is active at time t."""
        return self.start_time <= t < self.start_time + self.duration

    def get_mask(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Get boolean mask for stimulus region.

        Parameters
        ----------
        x, y : torch.Tensor
            Node coordinates

        Returns
        -------
        mask : torch.Tensor
            Boolean mask (True where stimulus is applied)
        """
        if callable(self.region):
            return self.region(x, y)
        else:
            return self.region


@dataclass
class StimulusProtocol:
    """
    Collection of stimuli forming a pacing protocol.

    Supports:
    - Single stimulus
    - S1-S2 protocol
    - Regular pacing at given BCL
    """
    stimuli: List[Stimulus] = field(default_factory=list)

    def add_stimulus(
        self,
        region: Union[Callable, torch.Tensor],
        start_time: float,
        duration: float = 1.0,
        amplitude: float = -52.0
    ):
        """Add a single stimulus event."""
        self.stimuli.append(Stimulus(
            region=region,
            start_time=start_time,
            duration=duration,
            amplitude=amplitude
        ))

    def add_s1s2_protocol(
        self,
        region: Union[Callable, torch.Tensor],
        n_s1: int = 10,
        bcl: float = 1000.0,
        s2_ci: float = 300.0,
        duration: float = 1.0,
        amplitude: float = -52.0
    ):
        """
        Add S1-S2 protocol.

        Parameters
        ----------
        region : callable or tensor
            Stimulus region
        n_s1 : int
            Number of S1 beats
        bcl : float
            Basic cycle length (ms)
        s2_ci : float
            S2 coupling interval from last S1 (ms)
        duration : float
            Stimulus duration (ms)
        amplitude : float
            Stimulus amplitude (uA/uF)
        """
        # S1 beats
        for i in range(n_s1):
            self.add_stimulus(region, i * bcl, duration, amplitude)

        # S2 beat
        s2_time = (n_s1 - 1) * bcl + s2_ci
        self.add_stimulus(region, s2_time, duration, amplitude)

    def add_regular_pacing(
        self,
        region: Union[Callable, torch.Tensor],
        bcl: float = 1000.0,
        n_beats: int = 10,
        start_time: float = 0.0,
        duration: float = 1.0,
        amplitude: float = -52.0
    ):
        """
        Add regular pacing protocol.

        Parameters
        ----------
        region : callable or tensor
            Stimulus region
        bcl : float
            Basic cycle length (ms)
        n_beats : int
            Number of beats
        start_time : float
            Time of first stimulus (ms)
        duration : float
            Stimulus duration (ms)
        amplitude : float
            Stimulus amplitude (uA/uF)
        """
        for i in range(n_beats):
            self.add_stimulus(region, start_time + i * bcl, duration, amplitude)

    def get_current(
        self,
        t: float,
        x: torch.Tensor,
        y: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Get total stimulus current at time t.

        Parameters
        ----------
        t : float
            Current time (ms)
        x, y : torch.Tensor
            Node coordinates
        device : torch.device
            Device for output tensor
        dtype : torch.dtype
            Data type for output tensor

        Returns
        -------
        Istim : torch.Tensor
            Stimulus current at each node (uA/uF)
        """
        Istim = torch.zeros(len(x), dtype=dtype, device=device)

        for stim in self.stimuli:
            if stim.is_active(t):
                mask = stim.get_mask(x, y)
                Istim[mask] += stim.amplitude

        return Istim

    def get_active_stimuli(self, t: float) -> List[Stimulus]:
        """Get list of stimuli active at time t."""
        return [s for s in self.stimuli if s.is_active(t)]
