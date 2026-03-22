"""Batched single-cell generator — runs N protocols in parallel as N "cells".

Uses torch.compile for GPU kernel fusion. At n=10,000 on Blackwell GPU:
77.5M steps/s throughput.

Key optimization: pre-compute I_stim schedule for all protocols as a tensor
BEFORE the simulation loop. This eliminates per-step Python function calls.
"""

import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
import numpy as np

_BIDOMAIN_ROOT = Path(__file__).resolve().parents[3] / 'Bidomain' / 'Engine_V1'
if str(_BIDOMAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_BIDOMAIN_ROOT))

from cardiac_sim.ionic.ttp06.model import TTP06Model
from cardiac_sim.ionic.ttp06.parameters import V_REST
from cardiac_sim.ionic.base import CellType

from .single_cell_generator import TraceData
from .protocols import Protocol


class BatchGenerator:
    """Run multiple protocols in parallel using batched TTP06.

    Pre-computes I_stim schedules as tensors. Uses torch.compile on GPU.
    """

    _CELLTYPE_MAP = {
        'EPI': CellType.EPI,
        'ENDO': CellType.ENDO,
        'M_CELL': CellType.M_CELL,
    }

    def __init__(self, cell_type: str = 'EPI', device: str = 'cuda',
                 use_compile: bool = True):
        self.cell_type_str = cell_type
        self.cell_type_enum = self._CELLTYPE_MAP[cell_type]
        self.device = torch.device(device)
        self.model = TTP06Model(cell_type=self.cell_type_enum, device=self.device)

        if use_compile and device == 'cuda':
            self._step_fn = torch.compile(self.model.step)
            # Warmup
            V = torch.full((2,), V_REST, dtype=torch.float64, device=self.device)
            s = self.model.get_initial_state(n_cells=2).to(self.device)
            self._step_fn(V, s, 0.01, I_stim=torch.zeros(2, dtype=torch.float64,
                                                           device=self.device))
            if device == 'cuda':
                torch.cuda.synchronize()
        else:
            self._step_fn = self.model.step

    def run_batch(self, protocols: List[Protocol],
                  progress_interval: float = 10.0,
                  record_every: int = 1) -> List[TraceData]:
        """Run multiple protocols in parallel.

        Args:
            protocols: list of Protocol objects (must share same dt_default)
            progress_interval: print progress every N seconds
            record_every: record state every N steps (1 = every step)

        Returns:
            List of TraceData, one per protocol
        """
        n = len(protocols)
        if n == 0:
            return []

        dt = protocols[0].dt_default
        max_duration = max(p.duration_ms for p in protocols)
        n_steps = int(max_duration / dt)
        durations_ms = [p.duration_ms for p in protocols]

        print(f'  Pre-computing I_stim schedules for {n} protocols × {n_steps} steps...')
        t0 = time.time()

        # Pre-compute I_stim schedule: (n_steps, n) on CPU then move to GPU
        stim_schedule = self._precompute_stim(protocols, n_steps, dt)
        ext_schedule = self._precompute_ext(protocols, n_steps, dt)
        clamp_schedule = self._precompute_clamp_mask(protocols, n_steps, dt)
        clamp_v_schedule = None
        has_clamp = clamp_schedule.any()
        if has_clamp:
            clamp_v_schedule = self._precompute_clamp_voltage(protocols, n_steps, dt)

        # Move schedules to device
        stim_gpu = stim_schedule.to(self.device)
        ext_gpu = ext_schedule.to(self.device)
        clamp_mask_gpu = clamp_schedule.to(self.device)
        if clamp_v_schedule is not None:
            clamp_v_gpu = clamp_v_schedule.to(self.device)

        # Active mask: which protocols are still running at each step
        active_steps_gpu = torch.tensor([int(d / dt) for d in durations_ms],
                                         dtype=torch.long, device=self.device)

        print(f'  Schedules ready in {time.time()-t0:.1f}s. Running simulation...')

        # Initialize state
        V = torch.full((n,), V_REST, dtype=torch.float64, device=self.device)
        states = self.model.get_initial_state(n_cells=n).to(self.device)

        # Pre-allocate recording ON GPU (transfer to CPU once at end)
        n_records = (n_steps + record_every - 1) // record_every
        all_Vm = torch.zeros((n_records, n), dtype=torch.float64, device=self.device)
        all_stim = torch.zeros((n_records, n), dtype=torch.float64, device=self.device)
        all_states = torch.zeros((n_records, n, 18), dtype=torch.float64, device=self.device)
        all_Iion = torch.zeros((n_records, n), dtype=torch.float64, device=self.device)
        all_clamp = torch.zeros((n_records, n), dtype=torch.float64, device=self.device)
        all_gate_inf = torch.zeros((n_records, n, 12), dtype=torch.float64, device=self.device)
        all_gate_tau = torch.zeros((n_records, n, 12), dtype=torch.float64, device=self.device)

        t_start = time.time()
        last_progress = t_start
        record_idx = 0

        for step in range(n_steps):
            # Get pre-computed values for this step (already on GPU)
            I_stim = stim_gpu[step]       # (n,)
            I_ext = ext_gpu[step]         # (n,)

            if step % record_every == 0:
                I_ion = self.model.compute_Iion(V, states)
                gate_inf = self.model.compute_gate_steady_states(V, states)  # (n, 12)
                gate_tau = self.model.compute_gate_time_constants(V, states)  # (n, 12)
                recorded_stim = -(I_stim + I_ext)

                all_Vm[record_idx] = V
                all_stim[record_idx] = recorded_stim
                all_states[record_idx] = states
                all_Iion[record_idx] = I_ion
                all_clamp[record_idx] = clamp_mask_gpu[step]
                all_gate_inf[record_idx] = gate_inf
                all_gate_tau[record_idx] = gate_tau
                record_idx += 1

            # Advance state
            total_stim = I_stim + I_ext
            V_new, states_new = self._step_fn(V, states, dt, I_stim=total_stim)

            # Handle voltage clamp
            if has_clamp:
                mask = clamp_mask_gpu[step] > 0.5
                if mask.any():
                    V_new = torch.where(mask, clamp_v_gpu[step], V_new)

            # Mask finished protocols
            active = (step < active_steps_gpu)
            V = torch.where(active, V_new, V)
            states = torch.where(active.unsqueeze(1), states_new, states)

            # Progress
            now = time.time()
            if now - last_progress > progress_interval:
                pct = (step + 1) / n_steps * 100
                rate = (step + 1) * n / (now - t_start)
                print(f'    {pct:.0f}% ({step+1}/{n_steps}, '
                      f'{rate/1e6:.1f}M cell-steps/s)')
                last_progress = now

        elapsed = time.time() - t_start
        print(f'  Done: {n}×{n_steps} steps in {elapsed:.1f}s '
              f'({n_steps*n/elapsed/1e6:.1f}M cell-steps/s)')

        # Transfer all recordings to CPU once
        all_Vm = all_Vm.cpu()
        all_stim = all_stim.cpu()
        all_states = all_states.cpu()
        all_Iion = all_Iion.cpu()
        all_clamp = all_clamp.cpu()
        all_gate_inf = all_gate_inf.cpu()
        all_gate_tau = all_gate_tau.cpu()

        # Assemble per-protocol TraceData
        traces = []
        for i, proto in enumerate(protocols):
            proto_steps = int(proto.duration_ms / dt)
            proto_records = (proto_steps + record_every - 1) // record_every

            # Build (T, 47) tensor
            data = torch.cat([
                all_Vm[:proto_records, i].unsqueeze(1),        # col 0
                all_stim[:proto_records, i].unsqueeze(1),      # col 1
                torch.full((proto_records, 1), dt, dtype=torch.float64),  # col 2
                all_states[:proto_records, i],                  # cols 3-20
                all_Iion[:proto_records, i].unsqueeze(1),      # col 21
                all_clamp[:proto_records, i].unsqueeze(1),     # col 22
                all_gate_inf[:proto_records, i],                # cols 23-34
                all_gate_tau[:proto_records, i],                # cols 35-46
            ], dim=1)  # (T, 47)

            metadata = {
                'protocol_name': proto.name,
                'protocol_tier': proto.tier,
                'cell_type': self.cell_type_str,
                'duration_ms': proto.duration_ms,
                'dt_default': dt,
                'n_timesteps': proto_records,
            }
            traces.append(TraceData(data=data, metadata=metadata))

        return traces

    def _precompute_stim(self, protocols, n_steps, dt):
        """Pre-compute I_stim for all protocols at all timesteps.

        Uses vectorized tensor ops for standard pacing protocols (BCL-based).
        Falls back to per-step Python calls only for complex protocols.
        """
        from .protocols import (SteadyStatePacing, S1S2Restitution,
                                AlternansProtocol, RandomIntervalPacing)

        n = len(protocols)
        schedule = torch.zeros((n_steps, n), dtype=torch.float64)
        t_all = torch.arange(n_steps, dtype=torch.float64) * dt  # (n_steps,)

        for i, p in enumerate(protocols):
            p_steps = min(n_steps, int(p.duration_ms / dt))
            t = t_all[:p_steps]

            if isinstance(p, (SteadyStatePacing, AlternansProtocol)):
                # Vectorized: stim when beat_phase < stim_duration
                beat_phase = t % p.bcl
                mask = beat_phase < p.stim_duration
                schedule[:p_steps, i] = torch.where(mask, p.stim_amplitude, 0.0)

            elif isinstance(p, S1S2Restitution):
                # S1 train
                s1_mask = (t < p._s1_end)
                s1_phase = t % p.s1_bcl
                s1_stim = s1_mask & (s1_phase < p.stim_duration)
                # S2 beat
                s2_stim = (t >= p._s2_onset) & (t < p._s2_onset + p.stim_duration)
                schedule[:p_steps, i] = torch.where(
                    s1_stim | s2_stim, p.stim_amplitude, 0.0)

            elif isinstance(p, RandomIntervalPacing):
                # Use cumulative sums for beat onsets
                for beat_idx in range(p.n_beats):
                    onset = 0.0 if beat_idx == 0 else float(p.cumulative[beat_idx - 1])
                    mask = (t >= onset) & (t < onset + p.stim_duration)
                    schedule[:p_steps, i] = torch.where(
                        mask, p.stim_amplitude, schedule[:p_steps, i])

            else:
                # Fallback: per-step Python call
                for step in range(p_steps):
                    schedule[step, i] = p.get_I_stim(step * dt)

        return schedule

    def _precompute_ext(self, protocols, n_steps, dt):
        """Pre-compute I_ext for all protocols at all timesteps."""
        n = len(protocols)
        # Check if any protocol has non-zero I_ext
        has_ext = any(hasattr(p, 'injection') or
                      (hasattr(p, 'get_I_ext') and
                       p.__class__.get_I_ext is not Protocol.get_I_ext)
                      for p in protocols)
        if not has_ext:
            return torch.zeros((n_steps, n), dtype=torch.float64)

        schedule = torch.zeros((n_steps, n), dtype=torch.float64)
        for i, p in enumerate(protocols):
            if p.__class__.get_I_ext is Protocol.get_I_ext:
                continue  # default returns 0, skip
            p_steps = min(n_steps, int(p.duration_ms / dt))
            for step in range(p_steps):
                schedule[step, i] = p.get_I_ext(step * dt)
        return schedule

    def _precompute_clamp_mask(self, protocols, n_steps, dt):
        """Pre-compute clamp mask for all protocols at all timesteps."""
        n = len(protocols)
        has_clamp = any(p.is_clamped(0) or
                        p.__class__.is_clamped is not Protocol.is_clamped
                        for p in protocols)
        if not has_clamp:
            return torch.zeros((n_steps, n), dtype=torch.float64)

        schedule = torch.zeros((n_steps, n), dtype=torch.float64)
        for i, p in enumerate(protocols):
            if p.__class__.is_clamped is Protocol.is_clamped:
                continue
            p_steps = min(n_steps, int(p.duration_ms / dt))
            # Most clamp protocols are always clamped
            if p.is_clamped(0):
                schedule[:p_steps, i] = 1.0
            else:
                for step in range(p_steps):
                    if p.is_clamped(step * dt):
                        schedule[step, i] = 1.0
        return schedule

    def _precompute_clamp_voltage(self, protocols, n_steps, dt):
        """Pre-compute clamp voltage for all protocols at all timesteps."""
        n = len(protocols)
        schedule = torch.full((n_steps, n), V_REST, dtype=torch.float64)
        for i, p in enumerate(protocols):
            if p.__class__.is_clamped is Protocol.is_clamped:
                continue
            p_steps = min(n_steps, int(p.duration_ms / dt))
            for step in range(p_steps):
                t = step * dt
                if p.is_clamped(t):
                    schedule[step, i] = p.get_clamp_voltage(t)
        return schedule
