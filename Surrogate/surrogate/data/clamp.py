"""Voltage clamp protocols for Tier 6 training data.

All clamp protocols return is_clamped() = True. The SingleCellGenerator
handles clamped protocols by overriding Vm instead of using the model's
Vm update. Gates still evolve at the clamped voltage.
"""

import torch
from typing import Optional, List

from .protocols import Protocol


class StepClamp(Protocol):
    """Hold at v_hold, then step to v_test. Tier 6."""

    def __init__(self, v_hold: float = -80.0, v_test: float = -20.0,
                 hold_time: float = 500.0, test_time: float = 500.0,
                 dt_default: float = 0.01):
        duration_ms = hold_time + test_time
        super().__init__(name=f'step_clamp_{int(v_test)}', tier=6,
                         duration_ms=duration_ms, dt_default=dt_default)
        self.v_hold = v_hold
        self.v_test = v_test
        self.hold_time = hold_time

    def is_clamped(self, t: float) -> bool:
        return True

    def get_clamp_voltage(self, t: float) -> float:
        return self.v_hold if t < self.hold_time else self.v_test


class RampClamp(Protocol):
    """Linear voltage ramp. Tier 6."""

    def __init__(self, v_start: float = -80.0, v_end: float = 40.0,
                 ramp_duration: float = 300.0, dt_default: float = 0.01):
        super().__init__(name='ramp_clamp', tier=6,
                         duration_ms=ramp_duration, dt_default=dt_default)
        self.v_start = v_start
        self.v_end = v_end

    def is_clamped(self, t: float) -> bool:
        return True

    def get_clamp_voltage(self, t: float) -> float:
        frac = min(t / self.duration_ms, 1.0)
        return self.v_start + frac * (self.v_end - self.v_start)


class StaircaseClamp(Protocol):
    """Multi-step voltage staircase. Tier 6."""

    def __init__(self, voltages: Optional[List[float]] = None,
                 step_duration: float = 100.0, dt_default: float = 0.01):
        voltages = voltages or [-80, -60, -40, -20, 0, 20, 40]
        duration_ms = step_duration * len(voltages)
        super().__init__(name='staircase_clamp', tier=6,
                         duration_ms=duration_ms, dt_default=dt_default)
        self.voltages = voltages
        self.step_duration = step_duration

    def is_clamped(self, t: float) -> bool:
        return True

    def get_clamp_voltage(self, t: float) -> float:
        step_idx = min(int(t / self.step_duration), len(self.voltages) - 1)
        return self.voltages[step_idx]


class APClamp(Protocol):
    """Play back recorded AP waveform as Vm command. Tier 6."""

    def __init__(self, vm_waveform: torch.Tensor, dt_waveform: float = 0.01,
                 dt_default: float = 0.01):
        duration_ms = len(vm_waveform) * dt_waveform
        super().__init__(name='ap_clamp', tier=6,
                         duration_ms=duration_ms, dt_default=dt_default)
        self.vm_waveform = vm_waveform
        self.dt_waveform = dt_waveform

    def is_clamped(self, t: float) -> bool:
        return True

    def get_clamp_voltage(self, t: float) -> float:
        idx = min(int(t / self.dt_waveform), len(self.vm_waveform) - 1)
        return float(self.vm_waveform[idx])


class PartialClamp(Protocol):
    """Partially clamped: Vm = alpha*V_cmd + (1-alpha)*V_free. Tier 6.

    is_clamped() returns True. SingleCellGenerator detects `hasattr(protocol, 'alpha')`
    and blends V_cmd with V_free instead of full override.
    """

    def __init__(self, alpha: float = 0.5, command_protocol: 'Protocol' = None,
                 dt_default: float = 0.01):
        super().__init__(name=f'partial_clamp_a{alpha}', tier=6,
                         duration_ms=command_protocol.duration_ms,
                         dt_default=dt_default)
        self.alpha = alpha
        self.command_protocol = command_protocol

    def is_clamped(self, t: float) -> bool:
        return True

    def get_clamp_voltage(self, t: float) -> float:
        return self.command_protocol.get_clamp_voltage(t)
