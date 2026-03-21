"""Current injection profiles for tissue-mimicking training data (Tier 5).

All injection profiles follow TTP06 sign convention: negative = depolarizing.
The SingleCellGenerator sign-flips when recording.
"""

import numpy as np
from typing import Optional, List


class OUNoiseInjection:
    """Ornstein-Uhlenbeck noise current injection.

    dI/dt = -I/tau + sigma * sqrt(2/tau) * dW
    Pre-generates full trajectory for reproducibility.
    """

    def __init__(self, tau: float, sigma: float, duration_ms: float,
                 dt_ou: float = 0.01, seed: int = 0):
        rng = np.random.RandomState(seed)
        n = int(duration_ms / dt_ou) + 1
        self.trajectory = np.zeros(n)
        I = 0.0
        for i in range(1, n):
            I += -I / tau * dt_ou + sigma * np.sqrt(2 * dt_ou / tau) * rng.randn()
            self.trajectory[i] = I
        self.dt_ou = dt_ou
        self.tau = tau
        self.sigma = sigma

    def __call__(self, t: float) -> float:
        idx = int(t / self.dt_ou)
        idx = min(idx, len(self.trajectory) - 1)
        return float(self.trajectory[idx])


class RampInjection:
    """Smooth depolarizing ramp mimicking approaching wavefront."""

    def __init__(self, peak: float = -30.0, ramp_time: float = 3.0,
                 onset: float = 50.0, decay_time: float = 5.0):
        self.peak = peak
        self.ramp_time = ramp_time
        self.onset = onset
        self.decay_time = decay_time

    def __call__(self, t: float) -> float:
        if t < self.onset:
            return 0.0
        dt = t - self.onset
        if dt < self.ramp_time:
            return self.peak * dt / self.ramp_time
        dt_decay = dt - self.ramp_time
        return self.peak * np.exp(-dt_decay / self.decay_time)


class SubThresholdBlips:
    """Random sub-threshold current blips at Poisson times."""

    def __init__(self, amplitude: float = -15.0, duration: float = 2.0,
                 rate: float = 0.01, total_duration: float = 1000.0,
                 seed: int = 0):
        self.amplitude = amplitude
        self.duration = duration
        rng = np.random.RandomState(seed)
        n_blips = rng.poisson(rate * total_duration)
        self.onsets = sorted(rng.uniform(0, total_duration, n_blips))

    def __call__(self, t: float) -> float:
        for onset in self.onsets:
            if onset <= t < onset + self.duration:
                return self.amplitude
        return 0.0


class SustainedOffset:
    """Constant current offset."""

    def __init__(self, amplitude: float = -5.0, start: float = 0.0,
                 end: Optional[float] = None):
        self.amplitude = amplitude
        self.start = start
        self.end = end

    def __call__(self, t: float) -> float:
        if t < self.start:
            return 0.0
        if self.end is not None and t >= self.end:
            return 0.0
        return self.amplitude


class BiphasicPulse:
    """Depolarizing then hyperpolarizing pulse. Mimics wavefront passage."""

    def __init__(self, depol_amp: float = -20.0, hyperpol_amp: float = 10.0,
                 pulse_duration: float = 3.0, onset: float = 50.0):
        self.depol_amp = depol_amp
        self.hyperpol_amp = hyperpol_amp
        self.pulse_duration = pulse_duration
        self.onset = onset

    def __call__(self, t: float) -> float:
        if t < self.onset:
            return 0.0
        dt = t - self.onset
        if dt < self.pulse_duration:
            return self.depol_amp
        if dt < 2 * self.pulse_duration:
            return self.hyperpol_amp
        return 0.0


class RandomTelegraph:
    """Poisson-switching current between 0 and I_max."""

    def __init__(self, I_max: float = -20.0, rate: float = 5.0,
                 duration_ms: float = 1000.0, seed: int = 0):
        rng = np.random.RandomState(seed)
        n_switches = rng.poisson(rate * duration_ms / 1000.0)
        self.switch_times = sorted(rng.uniform(0, duration_ms, n_switches))
        self.I_max = I_max

    def __call__(self, t: float) -> float:
        # Count how many switches have occurred
        n = sum(1 for st in self.switch_times if st <= t)
        return self.I_max if n % 2 == 1 else 0.0


class CompositeInjection:
    """Combine multiple injection profiles additively."""

    def __init__(self, *injections):
        self.injections = injections

    def __call__(self, t: float) -> float:
        return sum(inj(t) for inj in self.injections)


class InjectedPacing:
    """Standard pacing with additional current injection. Tier 5.

    Wraps a base protocol and adds I_ext from an injection profile.
    This is a Protocol-like object that delegates to base_protocol.
    """

    def __init__(self, base_protocol, injection, name_suffix: str = ''):
        from .protocols import Protocol
        self.base_protocol = base_protocol
        self.injection = injection
        self.name = f'injected{name_suffix}_{base_protocol.name}'
        self.tier = 5
        self.duration_ms = base_protocol.duration_ms
        self.dt_default = base_protocol.dt_default
        self.initial_states = getattr(base_protocol, 'initial_states', None)

    def get_I_stim(self, t: float) -> float:
        return self.base_protocol.get_I_stim(t)

    def get_I_ext(self, t: float) -> float:
        return self.injection(t)

    def get_dt(self, t: float) -> float:
        return self.base_protocol.get_dt(t)

    def is_clamped(self, t: float) -> bool:
        return self.base_protocol.is_clamped(t)

    def get_clamp_voltage(self, t: float) -> float:
        return self.base_protocol.get_clamp_voltage(t)
