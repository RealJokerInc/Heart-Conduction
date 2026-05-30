"""
StimBuilder data models.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
from enum import Enum


class StimType(Enum):
    """Stimulation type."""
    CURRENT_INJECTION = "current_injection"
    VOLTAGE_CLAMP = "voltage_clamp"


class StimTarget(Enum):
    """Stimulation target."""
    INTRACELLULAR = "intracellular"
    EXTRACELLULAR = "extracellular"


@dataclass
class StimProtocol:
    """
    Timing protocol for a stimulus region.

    Stores frequency internally; use set_bcl() for BCL input.
    """
    duration: float = 1.0       # ms - pulse duration
    start_time: float = 0.0     # ms - when first pulse starts
    frequency: float = 1.0      # Hz - pulse frequency (stored internally)
    num_pulses: Optional[int] = None  # None = continuous pacing

    @property
    def bcl(self) -> float:
        """Get basic cycle length in ms."""
        if self.frequency <= 0:
            return float('inf')
        return 1000.0 / self.frequency

    def set_bcl(self, bcl_ms: float) -> None:
        """Set frequency from BCL (ms)."""
        if bcl_ms <= 0:
            raise ValueError("BCL must be positive")
        self.frequency = 1000.0 / bcl_ms

    @property
    def period(self) -> float:
        """Alias for bcl (ms between pulses)."""
        return self.bcl

    def get_pulse_times(self, max_time: float = None) -> list:
        """
        Get list of pulse start times.

        Args:
            max_time: Maximum simulation time (for continuous pacing)

        Returns:
            List of pulse start times in ms
        """
        times = []
        t = self.start_time
        count = 0

        while True:
            if self.num_pulses is not None and count >= self.num_pulses:
                break
            if max_time is not None and t > max_time:
                break
            times.append(t)
            t += self.bcl
            count += 1

            # Safety limit for continuous pacing without max_time
            if self.num_pulses is None and max_time is None and count >= 1000:
                break

        return times


@dataclass
class StimRegion:
    """
    Represents a stimulation region identified by a unique color.

    Contains spatial info (from image) and stimulus parameters.
    """
    # Spatial (from image)
    color: Tuple[int, ...]
    pixel_count: int
    label: Optional[str] = None
    is_background: bool = False

    # Stimulus type
    stim_type: Optional[StimType] = None
    amplitude: Optional[float] = None  # mV (voltage) or μA/cm² (current)
    target: StimTarget = StimTarget.INTRACELLULAR

    # Timing protocol
    protocol: Optional[StimProtocol] = None

    @property
    def is_configured(self) -> bool:
        """Check if region is fully configured."""
        if self.is_background:
            return True
        return (
            self.label is not None and
            self.stim_type is not None and
            self.amplitude is not None and
            self.protocol is not None
        )

    @property
    def amplitude_unit(self) -> str:
        """Get appropriate unit for amplitude."""
        if self.stim_type == StimType.VOLTAGE_CLAMP:
            return "mV"
        elif self.stim_type == StimType.CURRENT_INJECTION:
            return "uA/cm2"
        return ""

    def configure(
        self,
        label: str,
        stim_type: StimType,
        amplitude: float,
        duration: float = 1.0,
        start_time: float = 0.0,
        bcl: float = 1000.0,
        num_pulses: Optional[int] = None,
        target: StimTarget = StimTarget.INTRACELLULAR
    ) -> None:
        """
        Fully configure the stimulus region.

        Args:
            label: Region name (e.g., "S1_pacing")
            stim_type: CURRENT_INJECTION or VOLTAGE_CLAMP
            amplitude: Stimulus strength
            duration: Pulse duration (ms)
            start_time: First pulse start time (ms)
            bcl: Basic cycle length (ms) - converted to frequency
            num_pulses: Number of pulses (None = continuous)
            target: INTRACELLULAR or EXTRACELLULAR
        """
        self.label = label
        self.stim_type = stim_type
        self.amplitude = amplitude
        self.target = target
        self.is_background = False

        self.protocol = StimProtocol(
            duration=duration,
            start_time=start_time,
            num_pulses=num_pulses
        )
        self.protocol.set_bcl(bcl)

    def summary_dict(self) -> dict:
        """Get summary as dict for simulation access."""
        return {
            'label': self.label,
            'color': self.color,
            'pixel_count': self.pixel_count,
            'stim_type': self.stim_type.value if self.stim_type else None,
            'amplitude': self.amplitude,
            'amplitude_unit': self.amplitude_unit,
            'target': self.target.value,
            'protocol': {
                'duration': self.protocol.duration,
                'start_time': self.protocol.start_time,
                'frequency': self.protocol.frequency,
                'bcl': self.protocol.bcl,
                'num_pulses': self.protocol.num_pulses,
            } if self.protocol else None
        }
