"""
Optimizer V1 — Tissue Runner (1D Cable CV Measurement)

Runs 1D monodomain cable simulations to measure conduction velocity.
Uses PHAS13 ionic model with configurable diffusion coefficients.
"""

import sys
import os
import torch
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', '..', '..', 'Monodomain', 'Engine_V5.4'))

from cardiac_sim.ionic.phas13 import PHAS13Model, PHAS13Parameters
from .config import TuningConfig, apply_scaling, theta_to_dict


@dataclass
class CVResult:
    """Result from a CV measurement."""
    cv: Optional[float] = None      # cm/s
    converged: bool = True
    activation_times: Optional[np.ndarray] = None


def run_cv_measurement(theta_ionic: torch.Tensor,
                       D: float,
                       config: TuningConfig,
                       cable_length_cm: float = 1.0,
                       dx_cm: float = 0.01,
                       n_beats: int = 5) -> CVResult:
    """
    Measure conduction velocity in a 1D cable.

    Parameters
    ----------
    theta_ionic : ionic parameter scaling factors
    D : diffusion coefficient in cm^2/ms
    config : TuningConfig
    cable_length_cm : cable length in cm
    dx_cm : spatial resolution in cm
    n_beats : number of pacing beats

    Returns
    -------
    CVResult with measured CV.
    """
    dt = config.dt
    Nx = int(cable_length_cm / dx_cm) + 1
    dx = cable_length_cm / (Nx - 1)

    # Create model with scaled parameters
    model = PHAS13Model(device=config.device)
    theta_dict = theta_to_dict(theta_ionic, config.tier)
    apply_scaling(model.params, theta_dict)

    # Initialize
    V = torch.full((Nx,), model.V_rest, dtype=config.dtype, device=config.device)
    states = model.get_initial_state(n_cells=Nx)
    if states.device != torch.device(config.device):
        states = states.to(config.device)

    # Diffusion coefficient
    r = D * dt / (dx ** 2)  # Stability parameter

    # Probe points for CV: 25% and 75% along cable
    probe1 = Nx // 4
    probe2 = 3 * Nx // 4
    distance_cm = (probe2 - probe1) * dx

    # Track activation using dV/dt threshold (more robust than voltage threshold
    # for spontaneously-beating models like PHAS13)
    activation_threshold = -30.0  # mV
    activated1 = False
    activated2 = False
    t_act1 = None
    t_act2 = None
    V_prev_probe1 = model.V_rest
    V_prev_probe2 = model.V_rest

    total_time = config.pacing_cl * n_beats
    n_steps = int(total_time / dt)
    stim_region = max(1, Nx // 20)  # Stimulus at left 5%

    # For multi-beat: only track last beat
    last_beat_start = config.pacing_cl * (n_beats - 1)

    t_current = 0.0

    for step_i in range(n_steps):
        # Stimulus at left end
        I_stim = torch.zeros(Nx, dtype=config.dtype, device=config.device)
        t_in_beat = t_current % config.pacing_cl
        if t_in_beat < config.stim_duration:
            I_stim[:stim_region] = config.stim_amplitude

        # Ionic step
        V_new, states = model.step(V, states, dt, I_stim)

        # Diffusion (explicit Euler, Neumann BC)
        laplacian = torch.zeros_like(V_new)
        laplacian[1:-1] = V_new[:-2] - 2.0 * V_new[1:-1] + V_new[2:]
        # Neumann (zero flux) at boundaries
        laplacian[0] = V_new[1] - V_new[0]
        laplacian[-1] = V_new[-2] - V_new[-1]

        V = V_new + r * laplacian
        t_current += dt

        # Check for divergence
        if not torch.isfinite(V).all():
            return CVResult(converged=False)

        # Only track activation during last beat
        if t_current >= last_beat_start:
            # Reset at start of last beat
            if t_current - last_beat_start < dt * 2:
                activated1 = False
                activated2 = False
                t_act1 = None
                t_act2 = None

            # Detect upstroke crossing at probe points
            v1 = V[probe1].item()
            v2 = V[probe2].item()

            if (not activated1 and v1 > activation_threshold and
                    V_prev_probe1 <= activation_threshold):
                activated1 = True
                t_act1 = t_current

            if (not activated2 and v2 > activation_threshold and
                    V_prev_probe2 <= activation_threshold):
                activated2 = True
                t_act2 = t_current

            V_prev_probe1 = v1
            V_prev_probe2 = v2

    # Calculate CV
    if t_act1 is not None and t_act2 is not None and t_act2 > t_act1:
        cv = distance_cm / (t_act2 - t_act1) * 1000.0  # cm/ms -> cm/s
        return CVResult(cv=cv, converged=True)

    return CVResult(converged=False)
