"""
Optimizer V1 — Single-Cell Simulation Runner

Wraps PHAS13 ionic model for paced single-cell simulations.
Extracts AP biomarkers for optimization objectives.
"""

import sys
import os
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from copy import deepcopy

# Add Engine path
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', '..', '..', 'Monodomain', 'Engine_V5.4'))

from cardiac_sim.ionic.phas13 import PHAS13Model, PHAS13Parameters
from .config import TuningConfig, TuningTargets, apply_scaling, theta_to_dict
from .metrics import detect_aps, measure_apd, measure_dvdt_max, measure_v_rest, measure_peak, measure_cl


@dataclass
class CellResult:
    """Results from a single-cell simulation."""
    apd90: Optional[float] = None       # ms
    dvdt_max: Optional[float] = None    # V/s
    v_rest: float = 0.0                 # mV
    v_peak: float = 0.0                 # mV
    cl: Optional[float] = None          # ms (spontaneous CL)
    V_trace: Optional[np.ndarray] = None
    t_trace: Optional[np.ndarray] = None
    restitution: Optional[List[Tuple[float, float]]] = None
    converged: bool = True


def _run_sim(model: PHAS13Model, config: TuningConfig,
             n_beats: int, cl: float, save_last_n_beats: int = 2,
             spontaneous: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Core simulation loop. Returns (t, V) arrays.

    Parameters
    ----------
    model : PHAS13Model with (possibly scaled) parameters
    config : TuningConfig
    n_beats : number of beats to pace
    cl : cycle length in ms
    save_last_n_beats : how many beats to record
    spontaneous : if True, no stimulus applied
    """
    dt = config.dt
    total_time = cl * n_beats if not spontaneous else cl * n_beats
    save_start = cl * max(0, n_beats - save_last_n_beats) if not spontaneous else 0.0

    V = torch.tensor(model.V_rest, dtype=config.dtype, device=config.device)
    states = model.get_initial_state(n_cells=1)
    if states.device != torch.device(config.device):
        states = states.to(config.device)

    t_list = []
    v_list = []

    n_steps = int(total_time / dt)
    t_current = 0.0

    for step_i in range(n_steps):
        # Stimulus
        I_stim = None
        if not spontaneous:
            t_in_beat = t_current % cl
            if t_in_beat < config.stim_duration:
                I_stim = torch.tensor(config.stim_amplitude, dtype=config.dtype,
                                      device=config.device)

        V, states = model.step(V, states, dt, I_stim)
        t_current += dt

        # Save
        if t_current >= save_start:
            t_list.append(t_current)
            v_list.append(V.item())

        # Early termination on divergence
        if not torch.isfinite(V):
            return np.array(t_list), np.array(v_list)

    return np.array(t_list), np.array(v_list)


def run_single_cell(theta: torch.Tensor, config: TuningConfig,
                    return_trace: bool = False) -> CellResult:
    """
    Pace a single cell and measure biomarkers.

    Parameters
    ----------
    theta : (n_params,) scaling factor tensor
    config : TuningConfig
    return_trace : whether to include V trace in result

    Returns
    -------
    CellResult with measured biomarkers.
    """
    model = PHAS13Model(device=config.device)
    theta_dict = theta_to_dict(theta, config.tier)
    apply_scaling(model.params, theta_dict)

    t, V = _run_sim(model, config, n_beats=config.n_beats,
                    cl=config.pacing_cl, save_last_n_beats=2)

    if len(V) == 0 or not np.isfinite(V).all():
        return CellResult(converged=False)

    result = CellResult(
        apd90=measure_apd(V, t),
        dvdt_max=measure_dvdt_max(V, t),
        v_rest=measure_v_rest(V, t),
        v_peak=measure_peak(V),
        converged=True,
    )

    if return_trace:
        result.V_trace = V
        result.t_trace = t

    return result


def run_spontaneous(theta: torch.Tensor, config: TuningConfig,
                    duration_ms: float = 5000.0,
                    return_trace: bool = False) -> CellResult:
    """
    Run without stimulus, measure spontaneous beating properties.

    Parameters
    ----------
    theta : (n_params,) scaling factor tensor
    config : TuningConfig
    duration_ms : total simulation time in ms

    Returns
    -------
    CellResult with CL and AP metrics.
    """
    model = PHAS13Model(device=config.device)
    theta_dict = theta_to_dict(theta, config.tier)
    apply_scaling(model.params, theta_dict)

    # Fake CL just to set total duration
    fake_cl = duration_ms / 5  # ~5 beats expected
    t, V = _run_sim(model, config, n_beats=5, cl=fake_cl,
                    save_last_n_beats=5, spontaneous=True)

    if len(V) == 0 or not np.isfinite(V).all():
        return CellResult(converged=False)

    result = CellResult(
        apd90=measure_apd(V, t),
        dvdt_max=measure_dvdt_max(V, t),
        v_rest=measure_v_rest(V, t),
        v_peak=measure_peak(V),
        cl=measure_cl(V, t),
        converged=True,
    )

    if return_trace:
        result.V_trace = V
        result.t_trace = t

    return result


def run_s1s2(theta: torch.Tensor, config: TuningConfig,
             di_values: List[float] = None) -> CellResult:
    """
    S1-S2 restitution protocol.

    Paces at S1 CL, then delivers S2 stimulus at varying DI after last S1.

    Parameters
    ----------
    theta : scaling factors
    config : TuningConfig
    di_values : list of diastolic intervals in ms

    Returns
    -------
    CellResult with restitution curve.
    """
    if di_values is None:
        di_values = [50.0, 100.0, 150.0, 200.0, 300.0, 500.0]

    model = PHAS13Model(device=config.device)
    theta_dict = theta_to_dict(theta, config.tier)
    apply_scaling(model.params, theta_dict)

    dt = config.dt
    s1_cl = config.pacing_cl
    n_s1 = config.n_beats

    # Phase 1: S1 pacing to steady state
    V = torch.tensor(model.V_rest, dtype=config.dtype, device=config.device)
    states = model.get_initial_state(n_cells=1)
    if states.device != torch.device(config.device):
        states = states.to(config.device)

    # Pace S1
    s1_time = s1_cl * n_s1
    n_steps = int(s1_time / dt)
    t_current = 0.0
    for _ in range(n_steps):
        I_stim = None
        t_in_beat = t_current % s1_cl
        if t_in_beat < config.stim_duration:
            I_stim = torch.tensor(config.stim_amplitude, dtype=config.dtype,
                                  device=config.device)
        V, states = model.step(V, states, dt, I_stim)
        t_current += dt
        if not torch.isfinite(V):
            return CellResult(converged=False)

    # Save state at end of last S1 AP
    V_s1_end = V.clone()
    states_s1_end = states.clone()

    # Phase 2: For each DI, deliver S2 and measure APD
    restitution = []
    for di in di_values:
        V_run = V_s1_end.clone()
        states_run = states_s1_end.clone()

        # Wait DI
        n_wait = int(di / dt)
        for _ in range(n_wait):
            V_run, states_run = model.step(V_run, states_run, dt)
            if not torch.isfinite(V_run):
                break

        if not torch.isfinite(V_run):
            continue

        # Apply S2 stimulus
        t_list = [0.0]
        v_list = [V_run.item()]

        # S2 stimulus + recording
        s2_record_time = 800.0  # ms, enough for one full AP
        n_s2 = int(s2_record_time / dt)
        t_s2 = 0.0
        for step_i in range(n_s2):
            I_stim = None
            if t_s2 < config.stim_duration:
                I_stim = torch.tensor(config.stim_amplitude, dtype=config.dtype,
                                      device=config.device)
            V_run, states_run = model.step(V_run, states_run, dt, I_stim)
            t_s2 += dt
            t_list.append(t_s2)
            v_list.append(V_run.item())
            if not torch.isfinite(V_run):
                break

        t_arr = np.array(t_list)
        v_arr = np.array(v_list)

        if np.isfinite(v_arr).all():
            apd = measure_apd(v_arr, t_arr)
            if apd is not None:
                restitution.append((di, apd))

    return CellResult(
        restitution=restitution,
        converged=len(restitution) > 0,
    )
