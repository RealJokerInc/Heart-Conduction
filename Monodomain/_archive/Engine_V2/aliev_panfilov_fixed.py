"""
Aliev-Panfilov Ionic Model with Time Scaling
==============================================

Clean implementation using standard Aliev-Panfilov parameters with time scaling.

Key Design Principles:
----------------------
1. Model runs in dimensionless time τ with standard parameters (mu1=0.2, epsilon0=0.002)
2. Time scaling factor T_scale converts between physical and dimensionless time:
   t_physical [ms] = τ [dimensionless] * T_scale [ms/τ]
3. No custom refractory gate - natural model dynamics provide proper refractoriness
4. User API in physical units (ms, mV), internal computation in dimensionless units

Governing Equations (dimensionless τ):
--------------------------------------
dV/dτ = -I_ion(V, w) + I_stim

I_ion(V, w) = k * V * (V - a) * (V - 1) + V * w

dw/dτ = ε(V, w) * [-k * V * (V - a - 1) - w]

ε(V, w) = ε₀ + ε_rest * sigmoid_rest(V) + (μ₁ * w) / (V + μ₂)

sigmoid_rest(V) = 1 / (1 + exp(k_sigmoid * (V - V_threshold)))

Where:
- V ∈ [0, 1] dimensionless voltage
- w dimensionless recovery variable
- τ dimensionless time
- All derivatives with respect to τ

Physical Conversion:
--------------------
- V_physical [mV] = V_rest + V * (V_peak - V_rest)
- t_physical [ms] = τ * T_scale

Standard Parameters:
--------------------
k = 8.0
a = 0.15
ε₀ = 0.002
μ₁ = 0.2
μ₂ = 0.3
T_scale = 10.0 ms/τ (gives APD90 ≈ 250 ms)

Expected Performance:
---------------------
- APD90 ≈ 250 ms (25τ * 10 ms/τ)
- ERP ≈ 200 ms (20τ * 10 ms/τ, ~80% of APD90)
- w returns to rest between beats at BCL ≥ 300 ms
- Natural rate-dependent APD shortening

Author: Generated with Claude Code
Date: 2025-12-05
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Callable

try:
    from .parameters import AlievPanfilovParameters, PhysicalConstants
except ImportError:
    from parameters import AlievPanfilovParameters, PhysicalConstants


class AlievPanfilovModel:
    """
    Time-scaled Aliev-Panfilov model with clean dynamics.

    This implementation runs the model in dimensionless time τ with standard
    parameters, then scales to physical time via T_scale.
    """

    def __init__(
        self,
        ionic_params: AlievPanfilovParameters,
        physical_params: PhysicalConstants,
        T_scale: float = 10.0  # ms per dimensionless time unit
    ):
        """
        Initialize time-scaled Aliev-Panfilov model.

        Parameters
        ----------
        ionic_params : AlievPanfilovParameters
            Ionic model parameters (k, a, epsilon0, mu1, mu2)
        physical_params : PhysicalConstants
            Physical constants (V_rest, V_peak for voltage conversion)
        T_scale : float
            Time scaling factor [ms / τ]
            Default 10.0 gives APD90 ≈ 250 ms with standard parameters
        """
        self.ionic = ionic_params
        self.physical = physical_params
        self.T_scale = T_scale

        self.V_rest = physical_params.V_rest
        self.V_peak = physical_params.V_peak
        self.V_range = physical_params.voltage_range()

        self.ionic.validate()

    # =========================================================================
    # Unit Conversion Methods
    # =========================================================================

    def time_to_tau(self, t_ms: float | np.ndarray) -> float | np.ndarray:
        """Convert physical time [ms] to dimensionless τ."""
        return t_ms / self.T_scale

    def tau_to_time(self, tau: float | np.ndarray) -> float | np.ndarray:
        """Convert dimensionless τ to physical time [ms]."""
        return tau * self.T_scale

    def voltage_to_physical(self, V_norm: np.ndarray | float) -> np.ndarray | float:
        """Convert normalized voltage [0, 1] to physical [mV]."""
        return self.V_rest + V_norm * self.V_range

    def voltage_to_normalized(self, V_mV: np.ndarray | float) -> np.ndarray | float:
        """Convert physical voltage [mV] to normalized [0, 1]."""
        return (V_mV - self.V_rest) / self.V_range

    # =========================================================================
    # Core Model Equations (Dimensionless τ)
    # =========================================================================

    def epsilon(self, V: np.ndarray | float, w: np.ndarray | float) -> np.ndarray | float:
        """
        Voltage-dependent time-scale function ε(V, w).

        This is the key governing equation that controls recovery dynamics.

        ε(V, w) = ε₀ + ε_rest * sigmoid_rest(V) + (μ₁ * w) / (V + μ₂)

        Components:
        -----------
        1. ε₀: Base time scale (constant)
        2. ε_rest * sigmoid_rest(V): Voltage-gated recovery boost
           - High at rest (V≈0) for fast w decay
           - Low during AP (V≈1) for slow w dynamics
        3. (μ₁ * w) / (V + μ₂): w-dependent modulation (standard AP)

        The sigmoid provides fast recovery at rest while preserving AP shape.

        Parameters:
        -----------
        V : dimensionless voltage [0, 1]
        w : dimensionless recovery variable

        Returns:
        --------
        ε : time scale for recovery dynamics [1/τ]
        """
        # Component 1: Base time scale
        eps_base = self.ionic.epsilon0

        # Component 2: Voltage-gated recovery boost
        # sigmoid_rest(V) = 1/(1 + exp(k*(V - V_thresh)))
        # High when V < V_thresh (at rest), low when V > V_thresh (during AP)
        sigmoid_rest = 1.0 / (1.0 + np.exp(
            self.ionic.k_sigmoid * (V - self.ionic.V_threshold)
        ))
        eps_rest_boost = self.ionic.epsilon_rest * sigmoid_rest

        # Component 3: w-dependent modulation (standard Aliev-Panfilov)
        eps_w_modulation = (self.ionic.mu1 * w) / (V + self.ionic.mu2)

        # Total epsilon: sum of all components
        return eps_base + eps_rest_boost + eps_w_modulation

    def I_ion(self, V: np.ndarray | float, w: np.ndarray | float) -> np.ndarray | float:
        """
        Ionic current (dimensionless).

        I_ion = k * V * (V - a) * (V - 1) + V * w
        """
        k = self.ionic.k
        a = self.ionic.a
        return k * V * (V - a) * (V - 1.0) + V * w

    def dV_dtau(
        self,
        V: np.ndarray | float,
        w: np.ndarray | float,
        I_stim: np.ndarray | float = 0.0
    ) -> np.ndarray | float:
        """
        Time derivative of voltage in dimensionless time [1/τ].

        dV/dτ = -I_ion(V, w) + I_stim
        """
        return -self.I_ion(V, w) + I_stim

    def dw_dtau(
        self,
        V: np.ndarray | float,
        w: np.ndarray | float
    ) -> np.ndarray | float:
        """
        Time derivative of recovery variable in dimensionless time [1/τ].

        dw/dτ = ε(V, w) * [-k * V * (V - a - 1) - w]
        """
        eps = self.epsilon(V, w)
        k = self.ionic.k
        a = self.ionic.a
        return eps * (-k * V * (V - a - 1.0) - w)

    # =========================================================================
    # Time Integration
    # =========================================================================

    def step_explicit_euler(
        self,
        V: np.ndarray | float,
        w: np.ndarray | float,
        dtau: float,
        I_stim: np.ndarray | float = 0.0
    ) -> Tuple[np.ndarray | float, np.ndarray | float]:
        """
        Single explicit Euler step in dimensionless time.

        Parameters
        ----------
        V : dimensionless voltage [0, 1]
        w : recovery variable (dimensionless)
        dtau : time step in dimensionless units
        I_stim : dimensionless stimulus

        Returns
        -------
        V_new, w_new : updated state
        """
        dV = self.dV_dtau(V, w, I_stim)
        dw = self.dw_dtau(V, w)

        V_new = V + dtau * dV
        w_new = w + dtau * dw

        return V_new, w_new

    def integrate_ode(
        self,
        t_span: Tuple[float, float],
        dt: float,
        I_stim_func: Callable[[float], float] = None,
        V0_mV: float = None,
        w0: float = 0.0,
        return_physical: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Integrate ODE system over physical time.

        User provides physical time (ms) and physical stimulus.
        Internally converts to dimensionless time τ for computation.

        Parameters
        ----------
        t_span : (t_start, t_end) in ms (physical time)
        dt : time step [ms] (physical time)
        I_stim_func : callable or None
            Function I_stim(t_ms) returning stimulus in dimensionless units
            Input t_ms is in physical time (ms)
        V0_mV : initial voltage [mV] (if None, uses V_rest)
        w0 : initial recovery variable (dimensionless)
        return_physical : if True, convert V to mV; if False, return normalized

        Returns
        -------
        times : time array [ms] (physical time)
        V_trace : voltage (mV if return_physical=True, else dimensionless)
        w_trace : recovery variable (dimensionless)
        """
        t_start, t_end = t_span

        # Convert to dimensionless time
        tau_start = self.time_to_tau(t_start)
        tau_end = self.time_to_tau(t_end)
        dtau = self.time_to_tau(dt)

        n_steps = int(np.ceil((tau_end - tau_start) / dtau))

        # Initialize
        if V0_mV is None:
            V0_mV = self.V_rest

        V0_norm = self.voltage_to_normalized(V0_mV)

        tau_array = np.zeros(n_steps + 1)
        V_array = np.zeros(n_steps + 1)
        w_array = np.zeros(n_steps + 1)

        V_array[0] = V0_norm
        w_array[0] = w0
        tau_array[0] = tau_start

        V_current = V0_norm
        w_current = w0

        # Time-stepping loop in dimensionless time
        for step in range(n_steps):
            tau = tau_start + step * dtau
            tau_array[step + 1] = tau + dtau

            # Convert tau to physical time for stimulus function
            t_physical = self.tau_to_time(tau)

            # Get stimulus (provided in dimensionless units)
            if I_stim_func is not None:
                I_stim = I_stim_func(t_physical)
            else:
                I_stim = 0.0

            # Step forward in dimensionless time
            V_current, w_current = self.step_explicit_euler(
                V_current, w_current, dtau, I_stim
            )

            V_array[step + 1] = V_current
            w_array[step + 1] = w_current

        # Convert tau array to physical time
        times = self.tau_to_time(tau_array)

        # Convert voltage to physical units if requested
        if return_physical:
            V_trace = self.voltage_to_physical(V_array)
        else:
            V_trace = V_array

        return times, V_trace, w_array


def create_stimulus_train(
    amplitude: float = 2.0,  # Dimensionless amplitude (typical 1.5-2.5)
    duration: float = 2.0,  # ms (physical time)
    period: float = 300.0,  # ms (physical time - BCL)
    n_pulses: int = 5,
    start_time: float = 10.0  # ms (physical time)
):
    """
    Create dimensionless stimulus train.

    User specifies times in physical units (ms).
    Returns function that accepts physical time and returns dimensionless stimulus.

    Parameters
    ----------
    amplitude : float
        Dimensionless stimulus amplitude (1.5-2.5 typical for AP initiation)
    duration : pulse width [ms] (physical time)
    period : basic cycle length [ms] (physical time)
    n_pulses : number of pulses
    start_time : time of first pulse [ms] (physical time)

    Returns
    -------
    stim_func : callable
        Function I_stim(t_ms) returning dimensionless stimulus
        Input t_ms is in physical time (ms)
    """
    def stim_func(t_ms):
        if t_ms < start_time:
            return 0.0

        elapsed = t_ms - start_time
        pulse_index = int(elapsed / period)

        if pulse_index >= n_pulses:
            return 0.0

        phase = elapsed - pulse_index * period
        if phase < duration:
            return amplitude
        else:
            return 0.0

    return stim_func


if __name__ == "__main__":
    from parameters import create_default_parameters

    print("Testing Time-Scaled Aliev-Panfilov Model...\\n")

    # Create parameters with standard values
    params = create_default_parameters()
    params.ionic.k = 8.0
    params.ionic.a = 0.15
    params.ionic.epsilon0 = 0.002
    params.ionic.mu1 = 0.2  # Standard value
    params.ionic.mu2 = 0.3

    # Create model with T_scale = 10.0 ms/τ
    model = AlievPanfilovModel(params.ionic, params.physical, T_scale=10.0)

    print(f"Model Configuration:")
    print(f"  Time scaling: T_scale = {model.T_scale} ms/τ")
    print(f"  Expected APD90 ≈ {25 * model.T_scale:.0f} ms")
    print(f"  Expected ERP ≈ {20 * model.T_scale:.0f} ms")

    # Quick test
    print(f"\\nQuick single-cell test (100 ms, with stimulus)...")
    stim = create_stimulus_train(amplitude=2.0, duration=2.0, period=300.0, n_pulses=1, start_time=10.0)
    times, V_trace, w_trace = model.integrate_ode(
        t_span=(0.0, 100.0),
        dt=0.01,
        I_stim_func=stim,
        V0_mV=-85.0,
        w0=0.0,
        return_physical=True
    )

    print(f"  Initial V = {V_trace[0]:.2f} mV, w = {w_trace[0]:.4f}")
    print(f"  Peak V = {np.max(V_trace):.2f} mV")
    print(f"  Final V = {V_trace[-1]:.2f} mV, w = {w_trace[-1]:.4f}")
    print(f"  Stability check: {np.all(np.isfinite(V_trace)) and np.all(np.isfinite(w_trace))}")
