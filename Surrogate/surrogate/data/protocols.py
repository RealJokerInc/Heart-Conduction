"""Protocol definitions for single-cell data generation.

Protocols define pacing patterns, stimulus timing, voltage clamp schedules,
and other experimental conditions. Each protocol is passed to
SingleCellGenerator.run_protocol() for execution.

All protocols use regular classes (not dataclasses) to avoid inheritance
issues with required-after-default fields.
"""

import numpy as np
import torch
from typing import Optional, List


class Protocol:
    """Base class for pacing protocols.

    The run loop uses `while t < protocol.duration_ms`, not a fixed step count.
    Subclasses define their own __init__ and call super().__init__().
    """

    def __init__(self, name: str, tier: int, duration_ms: float,
                 dt_default: float = 0.01, initial_states=None):
        self.name = name
        self.tier = tier
        self.duration_ms = duration_ms
        self.dt_default = dt_default
        self.initial_states = initial_states

    def get_I_stim(self, t: float) -> float:
        """Return stimulus current at time t (TTP06 convention: negative = depolarizing)."""
        return 0.0

    def get_I_ext(self, t: float) -> float:
        """Return external injection current at time t (TTP06 convention)."""
        return 0.0

    def get_dt(self, t: float) -> float:
        """Return timestep at time t."""
        return self.dt_default

    def is_clamped(self, t: float) -> bool:
        """Whether voltage is clamped at time t."""
        return False

    def get_clamp_voltage(self, t: float) -> float:
        """Return clamp voltage at time t. Only called if is_clamped(t)."""
        raise NotImplementedError("Not a clamp protocol")


# ──────────────────────────────────────────────────────────────
# Tier 1: Steady-state pacing
# ──────────────────────────────────────────────────────────────

class SteadyStatePacing(Protocol):
    """BCL-paced protocol. Tier 1."""

    def __init__(self, bcl: float, n_beats: int = 20,
                 stim_amplitude: float = -80.0, stim_duration: float = 1.0,
                 dt_default: float = 0.01):
        duration_ms = bcl * n_beats
        super().__init__(name=f'steady_bcl{int(bcl)}', tier=1,
                         duration_ms=duration_ms, dt_default=dt_default)
        self.bcl = bcl
        self.n_beats = n_beats
        self.stim_amplitude = stim_amplitude
        self.stim_duration = stim_duration

    def get_I_stim(self, t: float) -> float:
        # Tolerance-based beat phase to avoid float modulo edge cases
        beat_phase = t % self.bcl
        if beat_phase < self.stim_duration or (self.bcl - beat_phase) < 1e-9:
            return self.stim_amplitude
        return 0.0


class QuiescentProtocol(Protocol):
    """No pacing — cell sits at rest. Used for rest periods in stitching."""

    def __init__(self, duration_ms: float, dt_default: float = 0.01):
        super().__init__(name='quiescent', tier=0,
                         duration_ms=duration_ms, dt_default=dt_default)


# ──────────────────────────────────────────────────────────────
# Tier 2: S1-S2 restitution
# ──────────────────────────────────────────────────────────────

class S1S2Restitution(Protocol):
    """S1 train + single S2 premature beat. Tier 2."""

    def __init__(self, s2_di: float, s1_bcl: float = 1000.0, s1_beats: int = 10,
                 stim_amplitude: float = -80.0, stim_duration: float = 1.0,
                 dt_default: float = 0.01):
        # S1 train + S2 beat + tail for repolarization
        tail_ms = 500.0
        s1_duration = s1_bcl * s1_beats
        duration_ms = s1_duration + s2_di + tail_ms
        super().__init__(name=f's1s2_di{int(s2_di)}', tier=2,
                         duration_ms=duration_ms, dt_default=dt_default)
        self.s1_bcl = s1_bcl
        self.s1_beats = s1_beats
        self.s2_di = s2_di
        self.stim_amplitude = stim_amplitude
        self.stim_duration = stim_duration
        self._s1_end = s1_duration
        self._s2_onset = s1_duration + s2_di

    def get_I_stim(self, t: float) -> float:
        if t < self._s1_end:
            # S1 pacing
            beat_phase = t % self.s1_bcl
            if beat_phase < self.stim_duration or (self.s1_bcl - beat_phase) < 1e-9:
                return self.stim_amplitude
        elif self._s2_onset <= t < self._s2_onset + self.stim_duration:
            return self.stim_amplitude
        return 0.0


# ──────────────────────────────────────────────────────────────
# Tier 3: Dynamic protocols
# ──────────────────────────────────────────────────────────────

class BCLRamp(Protocol):
    """Linear BCL decrease over N beats. Tier 3."""

    def __init__(self, bcl_start: float = 1000.0, bcl_end: float = 300.0,
                 n_beats: int = 30, stim_amplitude: float = -80.0,
                 stim_duration: float = 1.0, dt_default: float = 0.01):
        # Pre-compute beat onset times
        self.bcls = np.linspace(bcl_start, bcl_end, n_beats)
        self.onsets = np.concatenate([[0.0], np.cumsum(self.bcls[:-1])])
        duration_ms = float(np.sum(self.bcls))
        super().__init__(name=f'ramp_{int(bcl_start)}to{int(bcl_end)}', tier=3,
                         duration_ms=duration_ms, dt_default=dt_default)
        self.stim_amplitude = stim_amplitude
        self.stim_duration = stim_duration

    def get_I_stim(self, t: float) -> float:
        idx = np.searchsorted(self.onsets, t, side='right') - 1
        if idx < 0 or idx >= len(self.onsets):
            return 0.0
        beat_phase = t - self.onsets[idx]
        if beat_phase < self.stim_duration:
            return self.stim_amplitude
        return 0.0


class BurstPacing(Protocol):
    """Fast burst + pause, repeated. Tier 3."""

    def __init__(self, burst_bcl: float = 300.0, n_burst: int = 5,
                 pause_ms: float = 2000.0, n_cycles: int = 5,
                 stim_amplitude: float = -80.0, stim_duration: float = 1.0,
                 dt_default: float = 0.01):
        burst_duration = burst_bcl * n_burst
        cycle_duration = burst_duration + pause_ms
        duration_ms = cycle_duration * n_cycles
        super().__init__(name='burst_pacing', tier=3,
                         duration_ms=duration_ms, dt_default=dt_default)
        self.burst_bcl = burst_bcl
        self.n_burst = n_burst
        self.pause_ms = pause_ms
        self.n_cycles = n_cycles
        self.cycle_duration = cycle_duration
        self.burst_duration = burst_duration
        self.stim_amplitude = stim_amplitude
        self.stim_duration = stim_duration

    def get_I_stim(self, t: float) -> float:
        cycle_phase = t % self.cycle_duration
        if cycle_phase < self.burst_duration:
            beat_phase = cycle_phase % self.burst_bcl
            if beat_phase < self.stim_duration:
                return self.stim_amplitude
        return 0.0


class AlternansProtocol(Protocol):
    """Constant fast BCL to trigger alternans. Tier 3."""

    def __init__(self, bcl: float = 330.0, n_beats: int = 20,
                 stim_amplitude: float = -80.0, stim_duration: float = 1.0,
                 dt_default: float = 0.01):
        duration_ms = bcl * n_beats
        super().__init__(name=f'alternans_bcl{int(bcl)}', tier=3,
                         duration_ms=duration_ms, dt_default=dt_default)
        self.bcl = bcl
        self.stim_amplitude = stim_amplitude
        self.stim_duration = stim_duration

    def get_I_stim(self, t: float) -> float:
        beat_phase = t % self.bcl
        if beat_phase < self.stim_duration:
            return self.stim_amplitude
        return 0.0


# ──────────────────────────────────────────────────────────────
# Tier 4: Random intervals
# ──────────────────────────────────────────────────────────────

class RandomIntervalPacing(Protocol):
    """Random inter-beat intervals from LogUniform distribution. Tier 4."""

    def __init__(self, n_beats: int, interval_min: float = 200.0,
                 interval_max: float = 2000.0, seed: Optional[int] = None,
                 stim_amplitude: float = -80.0, stim_duration: float = 1.0,
                 dt_default: float = 0.01):
        rng = np.random.RandomState(seed)
        self.intervals = np.exp(rng.uniform(
            np.log(interval_min), np.log(interval_max), n_beats))
        self.cumulative = np.cumsum(self.intervals)
        duration_ms = float(self.cumulative[-1])
        super().__init__(name=f'random_seed{seed}', tier=4,
                         duration_ms=duration_ms, dt_default=dt_default)
        self.n_beats = n_beats
        self.stim_amplitude = stim_amplitude
        self.stim_duration = stim_duration

    def get_I_stim(self, t: float) -> float:
        # O(log n) lookup via binary search on cumulative sums
        beat_idx = int(np.searchsorted(self.cumulative, t, side='right'))
        if beat_idx >= self.n_beats:
            return 0.0
        beat_start = 0.0 if beat_idx == 0 else float(self.cumulative[beat_idx - 1])
        beat_phase = t - beat_start
        if beat_phase < self.stim_duration:
            return self.stim_amplitude
        return 0.0


# ──────────────────────────────────────────────────────────────
# Tier 7: Concentration perturbation
# ──────────────────────────────────────────────────────────────

class ConcentrationPerturbation(Protocol):
    """Modify extracellular concentrations. Wraps another protocol. Tier 7.

    K_o, Na_o, Ca_o are fields of TTP06Parameters, NOT CellTypeConfig.
    SingleCellGenerator detects this type and modifies TTP06Parameters
    via deepcopy before running.
    """

    def __init__(self, base_protocol: 'Protocol',
                 Ko: Optional[float] = None,
                 Nai_init: Optional[float] = None,
                 Cai_scale: float = 1.0):
        super().__init__(
            name=f'conc_Ko{Ko}_{base_protocol.name}', tier=7,
            duration_ms=base_protocol.duration_ms,
            dt_default=base_protocol.dt_default,
        )
        self.base_protocol = base_protocol
        self.Ko = Ko
        self.Nai_init = Nai_init
        self.Cai_scale = Cai_scale

    def get_I_stim(self, t): return self.base_protocol.get_I_stim(t)
    def get_I_ext(self, t): return self.base_protocol.get_I_ext(t)
    def get_dt(self, t): return self.base_protocol.get_dt(t)
    def is_clamped(self, t): return self.base_protocol.is_clamped(t)
    def get_clamp_voltage(self, t): return self.base_protocol.get_clamp_voltage(t)


# ──────────────────────────────────────────────────────────────
# Tier 9: Corruption recovery
# ──────────────────────────────────────────────────────────────

class CorruptionRecovery(Protocol):
    """Start from corrupted states, record recovery. Tier 9.

    SingleCellGenerator detects this type and corrupts the initial
    ionic states before running the base protocol.
    """

    def __init__(self, base_protocol: 'Protocol',
                 corruption_type: str = 'random_gates',
                 severity: float = 0.5):
        super().__init__(
            name=f'corrupt_{corruption_type}_{base_protocol.name}', tier=9,
            duration_ms=base_protocol.duration_ms,
            dt_default=base_protocol.dt_default,
        )
        self.base_protocol = base_protocol
        self.corruption_type = corruption_type
        self.severity = severity

    def get_I_stim(self, t): return self.base_protocol.get_I_stim(t)
    def get_I_ext(self, t): return self.base_protocol.get_I_ext(t)
    def get_dt(self, t): return self.base_protocol.get_dt(t)


# ──────────────────────────────────────────────────────────────
# Protocol Library
# ──────────────────────────────────────────────────────────────

class ProtocolLibrary:
    """Factory methods for generating protocol sets by tier."""

    @staticmethod
    def tier1() -> List[Protocol]:
        """9 BCLs × 20 beats each."""
        return [SteadyStatePacing(bcl=b, n_beats=20)
                for b in [300, 400, 500, 600, 700, 800, 1000, 1500, 2000]]

    @staticmethod
    def tier2() -> List[Protocol]:
        """S1-S2 at 8 DI values."""
        return [S1S2Restitution(s2_di=di)
                for di in [50, 75, 100, 150, 200, 300, 500, 800]]

    @staticmethod
    def tier3() -> List[Protocol]:
        """Dynamic protocols: ramp, burst, alternans."""
        return [
            BCLRamp(bcl_start=1000, bcl_end=300, n_beats=30),
            BCLRamp(bcl_start=300, bcl_end=1000, n_beats=30),
            BurstPacing(burst_bcl=300, n_burst=5, pause_ms=2000, n_cycles=5),
            AlternansProtocol(bcl=330, n_beats=20),
        ]

    @staticmethod
    def tier4(n_protocols: int = 200) -> List[Protocol]:
        """Random interval protocols with variable trace lengths."""
        rng = np.random.RandomState(42)
        return [RandomIntervalPacing(
            n_beats=int(rng.randint(5, 201)), seed=i)
            for i in range(n_protocols)]
