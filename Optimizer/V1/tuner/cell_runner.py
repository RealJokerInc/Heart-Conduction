"""
Optimizer V1 — Single-Cell Simulation Runner

Supports batched simulation of M parameter sets simultaneously via
batch_ionic.batch_step(). GPU-viable with batching.
"""

import sys
import os
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', '..', '..', 'Monodomain', 'Engine_V5.4'))

from cardiac_sim.ionic.phas13.parameters import V_REST, get_initial_state as _get_initial_state
from .config import TuningConfig, TuningTargets, get_param_names
from .batch_ionic import batch_step, build_conductance_tensor
from .metrics import detect_aps, measure_apd, measure_dvdt_max, measure_v_rest, measure_peak, measure_cl


@dataclass
class CellResult:
    """Results from a single-cell simulation."""
    apd90: Optional[float] = None
    dvdt_max: Optional[float] = None
    v_rest: float = 0.0
    v_peak: float = 0.0
    cl: Optional[float] = None
    V_trace: Optional[np.ndarray] = None
    t_trace: Optional[np.ndarray] = None
    restitution: Optional[List[Tuple[float, float]]] = None
    converged: bool = True


def run_single_cell_batch(theta_batch: torch.Tensor,
                          config: TuningConfig,
                          n_beats: int = None,
                          cl: float = None,
                          save_last_n_beats: int = 2,
                          spontaneous: bool = False,
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate M parameter sets simultaneously.

    Parameters
    ----------
    theta_batch : (M, n_params) scaling factor tensor
    config : TuningConfig
    n_beats : beats to pace (default: config.n_beats)
    cl : cycle length in ms (default: config.pacing_cl)
    save_last_n_beats : how many beats to record
    spontaneous : if True, no stimulus

    Returns
    -------
    t : (T_save,) time array
    V_all : (M, T_save) voltage traces for all M cells
    """
    if n_beats is None:
        n_beats = config.n_beats
    if cl is None:
        cl = config.pacing_cl

    M = theta_batch.shape[0]
    dt = config.dt_cell
    device = config.device
    dtype = config.dtype

    # Build conductance tensor: (M, 14)
    cond = build_conductance_tensor(theta_batch, config.tier, dtype, device)

    # Initialize: all cells start from same initial state
    V = torch.full((M,), V_REST, dtype=dtype, device=device)
    init_state = _get_initial_state(device=torch.device(device), dtype=dtype)
    states = init_state.unsqueeze(0).expand(M, -1).clone()

    total_time = cl * n_beats
    save_start = cl * max(0, n_beats - save_last_n_beats) if not spontaneous else 0.0
    n_steps = int(total_time / dt)

    # Pre-allocate save buffer
    n_save = int((total_time - save_start) / dt) + 1
    V_saved = torch.zeros(M, n_save, dtype=dtype, device=device)
    save_idx = 0
    t_current = 0.0

    for step_i in range(n_steps):
        # Stimulus: same for all cells
        I_stim = None
        if not spontaneous:
            t_in_beat = t_current % cl
            if t_in_beat < config.stim_duration:
                I_stim = torch.full((M,), config.stim_amplitude,
                                    dtype=dtype, device=device)

        V, states = batch_step(V, states, dt, cond, I_stim)
        t_current += dt

        # Save
        if t_current >= save_start and save_idx < n_save:
            V_saved[:, save_idx] = V
            save_idx += 1

        # Early termination on divergence
        if not torch.isfinite(V).all():
            break

    # Build time array
    t_arr = np.arange(save_idx) * dt + save_start
    V_arr = V_saved[:, :save_idx].cpu().numpy()  # (M, T_save)

    return t_arr, V_arr


def run_single_cell(theta: torch.Tensor, config: TuningConfig,
                    return_trace: bool = False) -> CellResult:
    """Convenience wrapper: simulate one parameter set."""
    theta_batch = theta.unsqueeze(0) if theta.dim() == 1 else theta[:1]
    t, V_all = run_single_cell_batch(theta_batch, config)

    V = V_all[0]
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
    """Run without stimulus, measure spontaneous beating."""
    theta_batch = theta.unsqueeze(0) if theta.dim() == 1 else theta[:1]
    fake_cl = duration_ms / 5
    t, V_all = run_single_cell_batch(
        theta_batch, config, n_beats=5, cl=fake_cl,
        save_last_n_beats=5, spontaneous=True)

    V = V_all[0]
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


def extract_biomarkers_batch(t: np.ndarray, V_all: np.ndarray,
                             config: TuningConfig,
                             targets: TuningTargets) -> List[CellResult]:
    """Extract biomarkers from M voltage traces."""
    M = V_all.shape[0]
    results = []
    for i in range(M):
        V = V_all[i]
        if not np.isfinite(V).all():
            results.append(CellResult(converged=False))
            continue
        results.append(CellResult(
            apd90=measure_apd(V, t),
            dvdt_max=measure_dvdt_max(V, t),
            v_rest=measure_v_rest(V, t),
            v_peak=measure_peak(V),
            converged=True,
        ))
    return results


def run_s1s2(theta: torch.Tensor, config: TuningConfig,
             di_values: List[float] = None) -> CellResult:
    """S1-S2 restitution protocol (single theta, not batched)."""
    if di_values is None:
        di_values = [50.0, 100.0, 150.0, 200.0, 300.0, 500.0]

    theta_batch = theta.unsqueeze(0) if theta.dim() == 1 else theta[:1]
    dt = config.dt_cell
    device = config.device
    dtype = config.dtype

    cond = build_conductance_tensor(theta_batch, config.tier, dtype, device)

    V = torch.full((1,), V_REST, dtype=dtype, device=device)
    init_state = _get_initial_state(device=torch.device(device), dtype=dtype)
    states = init_state.unsqueeze(0)

    # S1 pacing
    s1_cl = config.pacing_cl
    n_steps = int(s1_cl * config.n_beats / dt)
    t_current = 0.0
    for _ in range(n_steps):
        I_stim = None
        if (t_current % s1_cl) < config.stim_duration:
            I_stim = torch.full((1,), config.stim_amplitude, dtype=dtype, device=device)
        V, states = batch_step(V, states, dt, cond, I_stim)
        t_current += dt
        if not torch.isfinite(V).all():
            return CellResult(converged=False)

    V_s1, states_s1 = V.clone(), states.clone()

    # S2 at each DI
    restitution = []
    for di in di_values:
        V_run, states_run = V_s1.clone(), states_s1.clone()
        for _ in range(int(di / dt)):
            V_run, states_run = batch_step(V_run, states_run, dt, cond)
            if not torch.isfinite(V_run).all():
                break
        if not torch.isfinite(V_run).all():
            continue

        t_list, v_list = [0.0], [V_run[0].item()]
        t_s2 = 0.0
        for _ in range(int(800.0 / dt)):
            I_stim = None
            if t_s2 < config.stim_duration:
                I_stim = torch.full((1,), config.stim_amplitude, dtype=dtype, device=device)
            V_run, states_run = batch_step(V_run, states_run, dt, cond, I_stim)
            t_s2 += dt
            t_list.append(t_s2)
            v_list.append(V_run[0].item())
            if not torch.isfinite(V_run).all():
                break

        t_arr, v_arr = np.array(t_list), np.array(v_list)
        if np.isfinite(v_arr).all():
            apd = measure_apd(v_arr, t_arr)
            if apd is not None:
                restitution.append((di, apd))

    return CellResult(restitution=restitution, converged=len(restitution) > 0)
