"""
Fenton-Karma Ionic Model and Simulation
========================================

3-variable cardiac ionic model with Numba-accelerated computation.

State Variables:
- u: Normalized transmembrane voltage [0, 1]
- v: Fast inward current inactivation gate [0, 1]
- w: Slow inward current inactivation gate [0, 1]

Ionic Currents:
- J_fi: Fast inward (Na-like) - rapid depolarization
- J_so: Slow outward (K-like) - repolarization
- J_si: Slow inward (Ca-like) - plateau phase

The model supports operator splitting:
1. Diffusion step (spatial coupling)
2. Ionic step (local reaction)

References:
- Fenton & Karma 1998, Chaos 8:20-47
- Fenton et al. 2002, Chaos 12:852-892

Author: Generated with Claude Code
Date: 2025-12-10
"""

from __future__ import annotations
import numpy as np
import numba
from typing import Callable, Optional, Tuple, Dict
from dataclasses import dataclass

from parameters import (
    FKParams, SpatialParams, PhysicalConstants,
    default_fk_params, default_spatial_params, default_physical_constants
)


# =============================================================================
# Numba-Accelerated Ionic Kernel
# =============================================================================

@numba.jit(nopython=True, cache=True)
def fk_ionic_step_kernel(
    u_arr: np.ndarray,
    v_arr: np.ndarray,
    w_arr: np.ndarray,
    I_stim: np.ndarray,
    # Pre-scaled time constants (dt / tau)
    dt_tau_d: float,
    dt_tau_v_plus: float,
    dt_tau_v1_minus: float,
    dt_tau_v2_minus: float,
    dt_tau_0: float,
    dt_tau_r: float,
    dt_tau_si: float,
    dt_tau_w_plus: float,
    dt_tau_w_minus: float,
    # Other parameters
    k: float,
    u_csi: float,
    u_c: float,
    u_v: float,
    dt: float,
) -> None:
    """
    Numba kernel for FK ionic step (in-place update).

    Fenton-Karma 3-variable model:
        du/dt = -J_fi - J_so - J_si + I_stim
        dv/dt = (1-p)(1-v)/τ_v⁻ - pv/τ_v⁺
        dw/dt = (1-p)(1-w)/τ_w⁻ - pw/τ_w⁺

    Where:
        J_fi = -v · p · (1-u) · (u-u_c) / τ_d     (fast inward)
        J_so = u · (1-p) / τ_0 + p / τ_r          (slow outward)
        J_si = -w · (1 + tanh(k(u-u_csi))) / 2τ_si (slow inward)
        p = Θ(u - u_c)  (Heaviside step function)

    All tau parameters are PRE-SCALED as dt/tau for efficiency.
    """
    ny, nx = u_arr.shape

    for i in range(ny):
        for j in range(nx):
            # Local copies of state variables
            u = u_arr[i, j]
            v = v_arr[i, j]
            w = w_arr[i, j]
            stim = I_stim[i, j]

            # Heaviside step functions
            p = 1.0 if u > u_c else 0.0      # Θ(u - u_c)
            q = 1.0 if u > u_v else 0.0      # Θ(u - u_v)

            # Voltage-dependent tau_v_minus (switches at u_v threshold)
            dt_tau_v_minus = q * dt_tau_v1_minus + (1.0 - q) * dt_tau_v2_minus

            # =====================
            # Ionic Currents
            # =====================

            # Fast inward current: J_fi = -v · p · (1-u) · (u-u_c) / τ_d
            J_fi = -v * p * (1.0 - u) * (u - u_c) / dt * dt_tau_d

            # Slow outward current: J_so = u · (1-p) / τ_0 + p / τ_r
            J_so = u * (1.0 - p) / dt * dt_tau_0 + p / dt * dt_tau_r

            # Slow inward current: J_si = -w · (1 + tanh(k(u-u_csi))) / (2τ_si)
            J_si = -w * (1.0 + np.tanh(k * (u - u_csi))) * 0.5 / dt * dt_tau_si

            # =====================
            # Gate Dynamics (pre-scaled by dt)
            # =====================

            # v-gate: dv = (1-p)(1-v) · dt/τ_v⁻ - p·v · dt/τ_v⁺
            dv = (1.0 - p) * (1.0 - v) * dt_tau_v_minus - p * v * dt_tau_v_plus

            # w-gate: dw = (1-p)(1-w) · dt/τ_w⁻ - p·w · dt/τ_w⁺
            dw = (1.0 - p) * (1.0 - w) * dt_tau_w_minus - p * w * dt_tau_w_plus

            # =====================
            # Voltage Update
            # =====================

            # du/dt = -J_fi - J_so - J_si + I_stim
            # du = dt · (-J_fi - J_so - J_si + stim)
            du = dt * (-(J_fi + J_so + J_si) + stim)

            # =====================
            # Euler Update with Clamping to [0, 1]
            # =====================

            u_arr[i, j] = max(0.0, min(u + du, 1.0))
            v_arr[i, j] = max(0.0, min(v + dv, 1.0))
            w_arr[i, j] = max(0.0, min(w + dw, 1.0))


# =============================================================================
# Fenton-Karma Model Class
# =============================================================================

class FentonKarmaModel:
    """
    Fenton-Karma 3-variable ionic model.

    Handles:
    - Parameter storage and pre-scaling
    - State initialization
    - Ionic step computation (via Numba kernel)
    - Voltage conversion (dimensionless <-> mV)
    """

    def __init__(
        self,
        params: FKParams = None,
        physical: PhysicalConstants = None,
        dt: float = 0.02,
    ):
        """
        Initialize FK model.

        Parameters
        ----------
        params : FKParams
            FK model parameters (default: paramset_3)
        physical : PhysicalConstants
            Voltage conversion constants
        dt : float
            Time step [ms] - used for pre-scaling
        """
        self.params = params or default_fk_params()
        self.physical = physical or default_physical_constants()
        self.dt = dt

        # Validate parameters
        self.params.validate()

        # Pre-scale time constants: dt / tau
        self._prescale_params()

        print(f"FentonKarmaModel initialized:")
        print(f"  dt = {self.dt} ms")
        print(f"  Pre-scaled parameters computed")

    def _prescale_params(self) -> None:
        """Pre-compute dt/tau for all time constants."""
        p = self.params
        dt = self.dt

        self.dt_tau_d = dt / p.tau_d
        self.dt_tau_v_plus = dt / p.tau_v_plus
        self.dt_tau_v1_minus = dt / p.tau_v1_minus
        self.dt_tau_v2_minus = dt / p.tau_v2_minus
        self.dt_tau_0 = dt / p.tau_0
        self.dt_tau_r = dt / p.tau_r
        self.dt_tau_si = dt / p.tau_si
        self.dt_tau_w_plus = dt / p.tau_w_plus
        self.dt_tau_w_minus = dt / p.tau_w_minus

    def set_dt(self, dt: float) -> None:
        """Update time step and re-prescale parameters."""
        self.dt = dt
        self._prescale_params()

    def initialize_state(self, shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """
        Create initial state arrays at rest.

        Parameters
        ----------
        shape : tuple
            (ny, nx) grid dimensions

        Returns
        -------
        state : dict
            {'u': array, 'v': array, 'w': array}
        """
        return {
            'u': np.zeros(shape, dtype=np.float64),   # Voltage at rest
            'v': np.ones(shape, dtype=np.float64),    # v-gate open at rest
            'w': np.ones(shape, dtype=np.float64),    # w-gate open at rest
        }

    def ionic_step(
        self,
        state: Dict[str, np.ndarray],
        I_stim: np.ndarray = None,
    ) -> None:
        """
        Perform ionic model step (in-place).

        Parameters
        ----------
        state : dict
            State variables {'u', 'v', 'w'}
        I_stim : np.ndarray, optional
            Stimulus current (same shape as u)
        """
        u, v, w = state['u'], state['v'], state['w']

        if I_stim is None:
            I_stim = np.zeros_like(u)

        # Call Numba kernel
        fk_ionic_step_kernel(
            u, v, w, I_stim,
            self.dt_tau_d,
            self.dt_tau_v_plus,
            self.dt_tau_v1_minus,
            self.dt_tau_v2_minus,
            self.dt_tau_0,
            self.dt_tau_r,
            self.dt_tau_si,
            self.dt_tau_w_plus,
            self.dt_tau_w_minus,
            self.params.k,
            self.params.u_csi,
            self.params.u_c,
            self.params.u_v,
            self.dt,
        )

    def voltage_to_physical(self, u: np.ndarray) -> np.ndarray:
        """Convert dimensionless voltage to mV."""
        return self.physical.to_physical(u)

    def physical_to_voltage(self, V_mV: np.ndarray) -> np.ndarray:
        """Convert mV to dimensionless voltage."""
        return self.physical.to_dimensionless(V_mV)


# =============================================================================
# Single-Cell Simulation (No Diffusion)
# =============================================================================

def run_single_cell(
    model: FentonKarmaModel,
    t_end: float = 500.0,
    stim_amplitude: float = 0.5,
    stim_start: float = 10.0,
    stim_duration: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run single-cell (0D) simulation.

    Useful for testing AP shape without diffusion.

    Parameters
    ----------
    model : FentonKarmaModel
        Configured model
    t_end : float
        Simulation duration [ms]
    stim_amplitude : float
        Stimulus amplitude (dimensionless)
    stim_start : float
        Stimulus start time [ms]
    stim_duration : float
        Stimulus duration [ms]

    Returns
    -------
    t : np.ndarray
        Time array [ms]
    u : np.ndarray
        Voltage trace (dimensionless)
    v : np.ndarray
        v-gate trace
    w : np.ndarray
        w-gate trace
    """
    dt = model.dt
    n_steps = int(np.ceil(t_end / dt))

    # Single cell = 1x1 grid
    state = model.initialize_state((1, 1))

    # Storage
    t_history = np.zeros(n_steps)
    u_history = np.zeros(n_steps)
    v_history = np.zeros(n_steps)
    w_history = np.zeros(n_steps)

    I_stim = np.zeros((1, 1))

    for step in range(n_steps):
        t = step * dt
        t_history[step] = t

        # Record state
        u_history[step] = state['u'][0, 0]
        v_history[step] = state['v'][0, 0]
        w_history[step] = state['w'][0, 0]

        # Apply stimulus
        if stim_start <= t < stim_start + stim_duration:
            I_stim[0, 0] = stim_amplitude
        else:
            I_stim[0, 0] = 0.0

        # Ionic step
        model.ionic_step(state, I_stim)

    return t_history, u_history, v_history, w_history


def measure_apd(
    t: np.ndarray,
    u: np.ndarray,
    threshold: float = 0.9,
) -> float:
    """
    Measure APD at given repolarization threshold.

    Parameters
    ----------
    t : np.ndarray
        Time array [ms]
    u : np.ndarray
        Voltage trace (dimensionless)
    threshold : float
        Repolarization fraction (0.9 for APD90)

    Returns
    -------
    apd : float
        Action potential duration [ms]
    """
    # Find max voltage and its time
    i_max = np.argmax(u)
    u_max = u[i_max]
    t_max = t[i_max]

    if u_max < 0.5:
        return np.nan  # No AP

    # Repolarization level
    u_thresh = u_max * (1 - threshold)

    # Find when voltage drops below threshold after peak
    for i in range(i_max, len(u)):
        if u[i] < u_thresh:
            return t[i] - t_max

    return np.nan  # Didn't repolarize


# =============================================================================
# Test: Single Cell AP
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("FENTON-KARMA SINGLE CELL TEST")
    print("=" * 60)

    # Create model
    model = FentonKarmaModel(dt=0.02)

    # Run single cell simulation
    print("\nRunning single cell simulation...")
    t, u, v, w = run_single_cell(
        model,
        t_end=500.0,
        stim_amplitude=0.5,
        stim_start=10.0,
        stim_duration=2.0,
    )

    # Convert to physical voltage
    V_mV = model.voltage_to_physical(u)

    # Measure APD90
    apd90 = measure_apd(t, u, threshold=0.9)
    print(f"\nResults:")
    print(f"  u_max = {np.max(u):.4f} ({np.max(V_mV):.1f} mV)")
    print(f"  u_rest = {u[-1]:.4f} ({V_mV[-1]:.1f} mV)")
    print(f"  APD90 = {apd90:.1f} ms")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Voltage
    axes[0].plot(t, V_mV, 'b-', linewidth=1.5)
    axes[0].axhline(y=-85, color='gray', linestyle='--', alpha=0.5, label='Rest')
    axes[0].axhline(y=40, color='gray', linestyle='--', alpha=0.5, label='Peak')
    axes[0].set_ylabel('Voltage [mV]', fontsize=12)
    axes[0].set_title(f'Fenton-Karma Single Cell AP (APD90 = {apd90:.1f} ms)', fontsize=14)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-100, 50)

    # v-gate
    axes[1].plot(t, v, 'g-', linewidth=1.5)
    axes[1].set_ylabel('v-gate', fontsize=12)
    axes[1].set_title('Fast Inward Gate (v)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.1, 1.1)

    # w-gate
    axes[2].plot(t, w, 'r-', linewidth=1.5)
    axes[2].set_ylabel('w-gate', fontsize=12)
    axes[2].set_xlabel('Time [ms]', fontsize=12)
    axes[2].set_title('Slow Inward Gate (w)', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.savefig('fk_single_cell_ap.png', dpi=150)
    print(f"\nPlot saved: fk_single_cell_ap.png")

    plt.show()
