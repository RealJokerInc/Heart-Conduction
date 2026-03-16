"""
Optimizer V1 — Tissue Runner (1D Cable CV Measurement)

Runs 1D monodomain cable simulations to measure conduction velocity.
Uses ionic subcycling: diffusion at dt (CFL-limited), ionic at dt_cell.
"""

import sys
import os
import torch
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', '..', '..', 'Monodomain', 'Engine_V5.4'))

from cardiac_sim.ionic.phas13 import PHAS13Model
from cardiac_sim.ionic.mhas13 import MHAS13Model
from .config import TuningConfig, apply_scaling, theta_to_dict


def _create_model(config: TuningConfig):
    """Create the appropriate ionic model based on config."""
    if config.ionic_model == 'mhas13':
        return MHAS13Model(device=config.device)
    return PHAS13Model(device=config.device)


@dataclass
class CVResult:
    """Result from a CV measurement."""
    cv: Optional[float] = None
    converged: bool = True
    activation_times: Optional[np.ndarray] = None


def run_cv_measurement(theta_ionic: torch.Tensor,
                       D: float,
                       config: TuningConfig,
                       cable_length_cm: float = None,
                       dx_cm: float = None,
                       n_beats: int = 3) -> CVResult:
    """
    Measure conduction velocity in a 1D cable.

    Uses ionic subcycling: multiple diffusion steps per ionic step
    when dt < dt_cell, reducing ionic model evaluations.
    """
    if cable_length_cm is None:
        cable_length_cm = config.cable_length_cm
    if dx_cm is None:
        dx_cm = config.dx_cm

    dt = config.dt
    dt_cell = config.dt_cell
    Nx = int(cable_length_cm / dx_cm) + 1
    dx = cable_length_cm / (Nx - 1)

    # Ionic subcycling: how many diffusion steps per ionic step
    ionic_substeps = max(1, int(round(dt_cell / dt)))
    dt_ionic = dt * ionic_substeps  # Actual ionic dt (close to dt_cell)

    # Create model
    model = _create_model(config)
    theta_dict = theta_to_dict(theta_ionic, config.tier)
    apply_scaling(model.params, theta_dict)

    # Initialize
    device = config.device
    dtype = config.dtype
    V = torch.full((Nx,), model.V_rest, dtype=dtype, device=device)
    states = model.get_initial_state(n_cells=Nx)
    if states.device != torch.device(device):
        states = states.to(device)

    # Diffusion parameter
    r = D * dt / (dx ** 2)

    # Probe points: 25% and 75%
    probe1 = Nx // 4
    probe2 = 3 * Nx // 4
    distance_cm = (probe2 - probe1) * dx

    # Activation tracking
    activation_threshold = -30.0
    activated1 = False
    activated2 = False
    t_act1 = None
    t_act2 = None
    V_prev_probe1 = model.V_rest
    V_prev_probe2 = model.V_rest

    total_time = config.pacing_cl * n_beats
    last_beat_start = config.pacing_cl * (n_beats - 1)
    stim_region = max(1, Nx // 20)

    # Main loop: step diffusion at dt, ionic at dt_ionic
    n_diffusion_steps = int(total_time / dt)
    t_current = 0.0
    diffusion_count = 0

    for step_i in range(n_diffusion_steps):
        # Diffusion step (always at dt)
        laplacian = torch.zeros_like(V)
        laplacian[1:-1] = V[:-2] - 2.0 * V[1:-1] + V[2:]
        laplacian[0] = V[1] - V[0]
        laplacian[-1] = V[-2] - V[-1]
        V = V + r * laplacian

        diffusion_count += 1
        t_current += dt

        # Ionic step (every ionic_substeps diffusion steps)
        if diffusion_count >= ionic_substeps:
            diffusion_count = 0

            # Stimulus
            I_stim = torch.zeros(Nx, dtype=dtype, device=device)
            t_in_beat = t_current % config.pacing_cl
            if t_in_beat < config.stim_duration:
                I_stim[:stim_region] = config.stim_amplitude

            V, states = model.step(V, states, dt_ionic, I_stim)

        # Divergence check (every 1000 steps)
        if step_i % 1000 == 0 and not torch.isfinite(V).all():
            return CVResult(converged=False)

        # Activation tracking (last beat only)
        if t_current >= last_beat_start:
            if t_current - last_beat_start < dt * 2:
                activated1 = False
                activated2 = False
                t_act1 = None
                t_act2 = None

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

    if t_act1 is not None and t_act2 is not None and t_act2 > t_act1:
        cv = distance_cm / (t_act2 - t_act1) * 1000.0
        return CVResult(cv=cv, converged=True)

    return CVResult(converged=False)
