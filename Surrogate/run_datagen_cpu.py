#!/usr/bin/env python
"""CPU batch data generation. Simple, reliable, ~6 hours for full dataset.

Uses TTP06 model.step() with batched tensors on CPU. No torch.compile,
no VRAM issues. At n=200 batch: 270K steps/s total throughput.

gate_inf and gate_tau computed post-hoc (vectorized, 0% overhead).
"""

import sys, time, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'Bidomain' / 'Engine_V1'))

import torch
import numpy as np

from cardiac_sim.ionic.ttp06.model import TTP06Model
from cardiac_sim.ionic.ttp06.parameters import V_REST, StateIndex
from cardiac_sim.ionic.base import CellType
from surrogate.data.single_cell_generator import TraceData
from surrogate.data.storage import TraceStorage
from surrogate.data.protocols import (
    Protocol, ProtocolLibrary, SteadyStatePacing, S1S2Restitution,
    QuiescentProtocol, ConcentrationPerturbation, CorruptionRecovery,
    RandomIntervalPacing,
)
from surrogate.data.injection import (
    OUNoiseInjection, RampInjection, SubThresholdBlips,
    SustainedOffset, BiphasicPulse, RandomTelegraph, InjectedPacing,
)
from surrogate.data.clamp import StepClamp, RampClamp, StaircaseClamp
from surrogate.data.augmentation import StitchedProtocol

RAW_DIR = '/media/norepinephrine/Elements-ext4/surrogate_data/raw'


def build_tier(tier, dt=0.01):
    """Build protocols for a tier. Returns list of Protocol objects."""
    protos = []
    if tier == 1:
        protos = ProtocolLibrary.tier1()
    elif tier == 2:
        protos = ProtocolLibrary.tier2()
    elif tier == 3:
        protos = ProtocolLibrary.tier3()
    elif tier == 4:
        protos = ProtocolLibrary.tier4(n_protocols=200)
    elif tier == 5:
        base = SteadyStatePacing(bcl=1000, n_beats=5)
        for name, inj in [
            ('ou_t5_s10', OUNoiseInjection(5, 10, base.duration_ms, seed=0)),
            ('ou_t10_s20', OUNoiseInjection(10, 20, base.duration_ms, seed=1)),
            ('ou_t2_s5', OUNoiseInjection(2, 5, base.duration_ms, seed=2)),
            ('ramp_fast', RampInjection(-30, 2.0, 500)),
            ('ramp_slow', RampInjection(-20, 5.0, 500)),
            ('blips', SubThresholdBlips(-15, 2, 0.02, base.duration_ms, 0)),
            ('sustained_neg5', SustainedOffset(-5.0)),
            ('sustained_pos3', SustainedOffset(3.0)),
            ('biphasic', BiphasicPulse(-20, 10, 3, 500)),
            ('telegraph', RandomTelegraph(-15, 3, base.duration_ms, 0)),
        ]:
            protos.append(InjectedPacing(copy.deepcopy(base), inj, f'_{name}'))
    elif tier == 6:
        protos = [StepClamp(v_test=v) for v in [-60, -40, -20, 0, 20, 40]]
        protos += [RampClamp(), StaircaseClamp()]
    elif tier == 7:
        combos = [
            {'Ko': 4.0}, {'Ko': 6.0}, {'Ko': 8.0}, {'Ko': 10.0},
            {'Nai_init': 6.0}, {'Nai_init': 12.0}, {'Nai_init': 15.0},
            {'Cai_scale': 0.5}, {'Cai_scale': 1.5}, {'Cai_scale': 2.0},
            {'Ko': 8.0, 'Nai_init': 12.0}, {'Ko': 6.0, 'Cai_scale': 1.5},
            {'Ko': 4.0, 'Nai_init': 6.0}, {'Ko': 10.0, 'Nai_init': 15.0, 'Cai_scale': 2.0},
        ]
        for combo in combos:
            for bcl in [500, 1000]:
                protos.append(ConcentrationPerturbation(SteadyStatePacing(bcl=bcl, n_beats=5), **combo))
    elif tier == 8:
        for bcl in [500, 1000]:
            protos.append(SteadyStatePacing(bcl=bcl, n_beats=50))
        for rest_s in [2, 5, 10]:
            protos.append(QuiescentProtocol(duration_ms=rest_s * 1000))
    elif tier == 9:
        base = QuiescentProtocol(duration_ms=100)
        for ct in ['random_gates', 'extreme_ca']:
            for sev in [0.3, 0.5, 0.8]:
                protos.append(CorruptionRecovery(base, ct, sev))
    elif tier == 10:
        base = SteadyStatePacing(bcl=1000, n_beats=5)
        for name, inj in [('boundary', SustainedOffset(1.0)), ('infarct', SustainedOffset(2.0)),
                           ('inert_sink', SustainedOffset(3.0))]:
            protos.append(InjectedPacing(copy.deepcopy(base), inj, f'_{name}'))
        for di in [40, 60, 80]:
            protos.append(S1S2Restitution(s2_di=di, s1_bcl=500, s1_beats=5))
    elif tier == 11:
        rng = np.random.RandomState(42)
        pool = [SteadyStatePacing(bcl=500, n_beats=3), SteadyStatePacing(bcl=1000, n_beats=3),
                S1S2Restitution(s2_di=200, s1_beats=3, s1_bcl=500)]
        for i in range(50):
            n_sub = rng.randint(2, 5)
            subs = [copy.deepcopy(pool[rng.randint(len(pool))]) for _ in range(n_sub)]
            rests = [float(np.exp(rng.uniform(np.log(500), np.log(5000)))) for _ in range(n_sub - 1)]
            protos.append(StitchedProtocol(protocols=subs, rest_durations=rests))

    for p in protos:
        if isinstance(p, StitchedProtocol):
            for sp in p.protocols: sp.dt_default = dt
        elif hasattr(p, 'dt_default'): p.dt_default = dt
        if isinstance(p, ConcentrationPerturbation): p.base_protocol.dt_default = dt
        if isinstance(p, CorruptionRecovery): p.base_protocol.dt_default = dt
    return protos


def run_batch_cpu(protos, model, dt=0.01, storage=None, tier=None,
                  chunk_steps=500000, cell_type='EPI'):
    """Run batchable protocols as parallel cells on CPU.

    Processes in time chunks to avoid OOM on long protocols.
    Each chunk: simulate → record → compute gate_inf/tau → write to HDF5.
    Writes per-protocol data to HDF5 immediately per chunk (no accumulation).

    Args:
        protos: list of Protocol objects (not ConcentrationPerturbation/CorruptionRecovery/Stitched)
        model: TTP06Model instance
        dt: timestep in ms
        storage: TraceStorage for incremental HDF5 writes (if None, returns TraceData list)
        tier: tier number for storage
        chunk_steps: max steps per chunk
        cell_type: cell type string for metadata
    """
    n = len(protos)
    max_dur = max(p.duration_ms for p in protos)
    n_steps = int(max_dur / dt)

    n_chunks = max(1, (n_steps + chunk_steps - 1) // chunk_steps)
    sched_gb = n_steps * n * 8 / 1e9
    chunk_buf_gb = min(n_steps, chunk_steps) * n * (18 + 4) * 8 / 1e9
    print(f'    {n} protos × {n_steps} steps, {n_chunks} chunks')
    print(f'    Schedule: {sched_gb:.1f}GB, chunk buffers: ~{chunk_buf_gb:.1f}GB')

    # Vectorized I_stim schedule: (n_steps, n) — ~26GB for 16M×200
    t_all = torch.arange(n_steps, dtype=torch.float64) * dt
    stim = torch.zeros((n_steps, n), dtype=torch.float64)
    for i, p in enumerate(protos):
        p_steps = min(n_steps, int(p.duration_ms / dt))
        t = t_all[:p_steps]
        if hasattr(p, 'bcl'):
            phase = t % p.bcl
            mask = phase < p.stim_duration
            stim[:p_steps, i] = torch.where(mask, p.stim_amplitude, 0.0)
        elif isinstance(p, S1S2Restitution):
            s1_mask = t < p._s1_end
            s1_stim = s1_mask & ((t % p.s1_bcl) < p.stim_duration)
            s2_stim = (t >= p._s2_onset) & (t < p._s2_onset + p.stim_duration)
            stim[:p_steps, i] = torch.where(s1_stim | s2_stim, p.stim_amplitude, 0.0)
        elif isinstance(p, RandomIntervalPacing):
            # Vectorized via searchsorted: O(n_steps * log(n_beats)) not O(n_steps * n_beats)
            onsets = torch.zeros(p.n_beats, dtype=torch.float64)
            if p.n_beats > 1:
                onsets[1:] = torch.tensor(p.cumulative[:p.n_beats-1].copy(), dtype=torch.float64)
            beat_idx = torch.searchsorted(onsets, t, right=True) - 1
            beat_idx = beat_idx.clamp(0, p.n_beats - 1)
            time_in_beat = t - onsets[beat_idx]
            mask = time_in_beat < p.stim_duration
            stim[:p_steps, i] = torch.where(mask, p.stim_amplitude, 0.0)
        else:
            for step in range(p_steps):
                stim[step, i] = p.get_I_stim(step * dt)

    # I_ext schedule — only allocate if needed
    has_ext = any(p.__class__.get_I_ext is not Protocol.get_I_ext for p in protos)
    ext = None
    if has_ext:
        ext = torch.zeros((n_steps, n), dtype=torch.float64)
        for i, p in enumerate(protos):
            if p.__class__.get_I_ext is Protocol.get_I_ext: continue
            p_steps = min(n_steps, int(p.duration_ms / dt))
            for step in range(p_steps):
                ext[step, i] = p.get_I_ext(step * dt)

    # Clamp — only allocate if needed
    has_clamp = any(p.__class__.is_clamped is not Protocol.is_clamped for p in protos)
    clamp_mask = None
    clamp_v = None
    if has_clamp:
        clamp_mask = torch.zeros((n_steps, n), dtype=torch.float64)
        clamp_v = torch.full((n_steps, n), V_REST, dtype=torch.float64)
        for i, p in enumerate(protos):
            if p.__class__.is_clamped is Protocol.is_clamped: continue
            p_steps = min(n_steps, int(p.duration_ms / dt))
            for step in range(p_steps):
                if p.is_clamped(step * dt):
                    clamp_mask[step, i] = 1.0
                    clamp_v[step, i] = p.get_clamp_voltage(step * dt)

    active_steps = torch.tensor([int(p.duration_ms / dt) for p in protos], dtype=torch.long)

    # For non-storage mode, accumulate per-protocol chunks in memory
    proto_chunks = {i: [] for i in range(n)} if storage is None else None

    # Track per-protocol total steps written (for HDF5 incremental writes)
    proto_steps_written = [0] * n

    V = torch.full((n,), V_REST, dtype=torch.float64)
    states = model.get_initial_state(n_cells=n)

    t0 = time.time()
    last_prog = t0

    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * chunk_steps
        chunk_end = min(chunk_start + chunk_steps, n_steps)
        cs = chunk_end - chunk_start

        # Allocate chunk buffers
        c_Vm = torch.zeros((cs, n), dtype=torch.float64)
        c_stim_rec = torch.zeros((cs, n), dtype=torch.float64)
        c_states = torch.zeros((cs, n, 18), dtype=torch.float64)
        c_Iion = torch.zeros((cs, n), dtype=torch.float64)
        c_clamp_rec = torch.zeros((cs, n), dtype=torch.float64)

        # Zero tensors for when ext/clamp not used
        zero_n = torch.zeros(n, dtype=torch.float64)

        for local_step in range(cs):
            step = chunk_start + local_step
            I_s = stim[step]
            I_e = ext[step] if has_ext else zero_n

            I_ion = model.compute_Iion(V, states)

            c_Vm[local_step] = V
            c_stim_rec[local_step] = -(I_s + I_e)
            c_states[local_step] = states
            c_Iion[local_step] = I_ion
            if has_clamp:
                c_clamp_rec[local_step] = clamp_mask[step]

            total = I_s + I_e
            V_new, states_new = model.step(V, states, dt, I_stim=total)

            if has_clamp:
                cm = clamp_mask[step] > 0.5
                if cm.any():
                    V_new = torch.where(cm, clamp_v[step], V_new)

            active = step < active_steps
            V = torch.where(active, V_new, V)
            states = torch.where(active.unsqueeze(1), states_new, states)

            now = time.time()
            if now - last_prog > 15:
                pct = (step+1)/n_steps*100
                rate = (step+1)*n/(now-t0)
                print(f'    {pct:.0f}% ({step+1}/{n_steps}, {rate/1e3:.0f}K cell-steps/s)')
                last_prog = now

        # Post-hoc gate_inf/tau for this chunk
        Vm_flat = c_Vm.reshape(-1)
        states_flat = c_states.reshape(-1, 18)
        gate_inf = model.compute_gate_steady_states(Vm_flat, states_flat).reshape(cs, n, 12)
        gate_tau = model.compute_gate_time_constants(Vm_flat, states_flat).reshape(cs, n, 12)

        # Build per-protocol data and write/accumulate immediately
        for i in range(n):
            p_steps_total = int(protos[i].duration_ms / dt)
            p_end = min(chunk_end, p_steps_total)
            if chunk_start >= p_end:
                continue
            local_start = 0
            local_end = p_end - chunk_start
            sl = slice(local_start, local_end)
            chunk_data = torch.cat([
                c_Vm[sl, i].unsqueeze(1),
                c_stim_rec[sl, i].unsqueeze(1),
                torch.full((local_end - local_start, 1), dt, dtype=torch.float64),
                c_states[sl, i],
                c_Iion[sl, i].unsqueeze(1),
                c_clamp_rec[sl, i].unsqueeze(1),
                gate_inf[sl, i],
                gate_tau[sl, i],
            ], dim=1)  # (chunk_len, 47)

            if storage is not None and tier is not None:
                # Write immediately to HDF5, append to existing dataset
                _append_to_h5(storage, tier, protos[i], chunk_data, dt, cell_type,
                              proto_steps_written[i])
                proto_steps_written[i] += chunk_data.shape[0]
            else:
                proto_chunks[i].append(chunk_data)

        # Free chunk buffers
        del c_Vm, c_stim_rec, c_states, c_Iion, c_clamp_rec, gate_inf, gate_tau

        print(f'    Chunk {chunk_idx+1}/{n_chunks} done')

    elapsed = time.time() - t0
    print(f'    Sim done: {n}×{n_steps} in {elapsed:.0f}s ({n_steps*n/elapsed/1e3:.0f}K cell-steps/s)')

    if storage is not None and tier is not None:
        print(f'    Saved {n} traces to tier {tier}')
        return []

    # Non-storage mode: assemble and return
    traces = []
    for i, p in enumerate(protos):
        if not proto_chunks[i]:
            continue
        data = torch.cat(proto_chunks[i], dim=0)
        meta = {
            'protocol_name': p.name, 'protocol_tier': p.tier,
            'cell_type': cell_type, 'duration_ms': p.duration_ms,
            'dt_default': dt, 'n_timesteps': data.shape[0],
        }
        traces.append(TraceData(data=data, metadata=meta))
    return traces


def _append_to_h5(storage, tier, proto, chunk_data, dt, cell_type, offset):
    """Append chunk data to HDF5 dataset, creating or resizing as needed."""
    import h5py
    path = Path(storage.base_dir) / f'tier{tier:02d}.h5'
    name = f'{proto.name}_dt{dt}'
    chunk_len = chunk_data.shape[0]

    with h5py.File(path, 'a') as f:
        if name not in f:
            # First chunk: create group + resizable dataset
            grp = f.create_group(name)
            maxshape = (None, 47)  # unlimited first dim
            grp.create_dataset('data', data=chunk_data.numpy(),
                               dtype='float64', maxshape=maxshape,
                               chunks=(min(chunk_len, 10000), 47))
            # Write metadata
            grp.attrs['protocol_name'] = proto.name
            grp.attrs['protocol_tier'] = proto.tier
            grp.attrs['cell_type'] = cell_type
            grp.attrs['duration_ms'] = proto.duration_ms
            grp.attrs['dt_default'] = dt
        else:
            # Subsequent chunks: resize and append
            ds = f[name]['data']
            old_len = ds.shape[0]
            ds.resize(old_len + chunk_len, axis=0)
            ds[old_len:old_len + chunk_len] = chunk_data.numpy()

        # Update n_timesteps
        f[name].attrs['n_timesteps'] = offset + chunk_len


def run_sequential_cpu(proto, model, cell_type='EPI', dt=0.01):
    """Run a single protocol sequentially (for ConcentrationPerturbation, Corruption, Stitched)."""
    from surrogate.data.single_cell_generator import SingleCellGenerator
    gen = SingleCellGenerator(cell_type=cell_type, device='cpu')
    return gen.run_protocol(proto)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiers', nargs='+', type=int, default=list(range(1, 13)))
    parser.add_argument('--celltypes', nargs='+', default=['EPI'])
    parser.add_argument('--raw-dir', default=RAW_DIR)
    parser.add_argument('--skip-existing', action='store_true', help='Skip tiers with existing .h5')
    args = parser.parse_args()

    storage = TraceStorage(args.raw_dir)
    t_total = time.time()

    for ct in args.celltypes:
        print(f'\n=== Celltype: {ct} ===')
        model = TTP06Model(cell_type=CellType[ct], device=torch.device('cpu'))

        for tier in args.tiers:
            if tier == 12: continue

            h5_path = Path(args.raw_dir) / f'tier{tier:02d}.h5'
            if args.skip_existing and h5_path.exists():
                print(f'  Tier {tier}: SKIP (exists)')
                continue

            protos = build_tier(tier)
            if not protos:
                continue

            # Split batchable vs sequential
            batchable = [p for p in protos
                         if not isinstance(p, (ConcentrationPerturbation, CorruptionRecovery, StitchedProtocol))]
            sequential = [p for p in protos
                          if isinstance(p, (ConcentrationPerturbation, CorruptionRecovery, StitchedProtocol))]

            print(f'  Tier {tier}: {len(batchable)} batch + {len(sequential)} seq')
            t0 = time.time()

            if batchable:
                run_batch_cpu(batchable, model, storage=storage, tier=tier, cell_type=ct)

            for i, p in enumerate(sequential):
                trace = run_sequential_cpu(p, model, cell_type=ct)
                storage.save_trace(trace, tier, f'{trace.metadata["protocol_name"]}_{i:03d}_dt0.01')
                if (i+1) % 5 == 0:
                    print(f'    Sequential: {i+1}/{len(sequential)}')
            if sequential:
                print(f'    Saved {len(sequential)} seq traces')

            print(f'  Tier {tier} done in {time.time()-t0:.0f}s')

        # Tier 12: other celltypes — all written to tier12.h5
        if 12 in args.tiers and ct == args.celltypes[0]:
            for ct2 in ['ENDO', 'M_CELL']:
                print(f'\n=== Tier 12: {ct2} ===')
                model2 = TTP06Model(cell_type=CellType[ct2], device=torch.device('cpu'))
                for sub_tier in [1, 2, 3]:
                    protos = build_tier(sub_tier)
                    run_batch_cpu(protos, model2, storage=storage, tier=12, cell_type=ct2)
                    print(f'  T12/{ct2}/T{sub_tier}: done')

    elapsed = time.time() - t_total
    total_size = sum(f.stat().st_size for f in Path(args.raw_dir).glob('*.h5'))
    print(f'\n{"="*60}')
    print(f'TOTAL: {elapsed/60:.0f} min ({elapsed/3600:.1f} hours)')
    print(f'Output: {total_size/1e9:.1f} GB')


if __name__ == '__main__':
    main()
