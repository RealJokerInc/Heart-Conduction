#!/usr/bin/env python
"""Production data generation script.

Uses GPU + torch.compile + batched protocols for ~78M cell-steps/s throughput.
Groups protocols by similar duration, pads to batch size, runs on GPU,
saves to HDF5 on external HDD.

Usage:
    python run_datagen.py                          # all tiers, single dt
    python run_datagen.py --tiers 1 2 3            # specific tiers
    python run_datagen.py --multi-dt               # 5 dt values for T1-4
    python run_datagen.py --dry-run                # show plan only
    python run_datagen.py --batch-size 5000        # adjust GPU batch
"""

import sys, os, time, copy, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'Bidomain' / 'Engine_V1'))

import torch
import numpy as np

from surrogate.data.single_cell_generator import SingleCellGenerator, TraceData
from surrogate.data.batch_generator import BatchGenerator
from surrogate.data.protocols import (
    Protocol, ProtocolLibrary, SteadyStatePacing, S1S2Restitution,
    QuiescentProtocol, ConcentrationPerturbation, CorruptionRecovery,
    RandomIntervalPacing, BCLRamp, BurstPacing, AlternansProtocol,
)
from surrogate.data.injection import (
    OUNoiseInjection, RampInjection, SubThresholdBlips,
    SustainedOffset, BiphasicPulse, RandomTelegraph, InjectedPacing,
)
from surrogate.data.clamp import StepClamp, RampClamp, StaircaseClamp
from surrogate.data.augmentation import StitchedProtocol
from surrogate.data.storage import TraceStorage

# ── Config ──────────────────────────────────────────────────
RAW_DIR = '/media/norepinephrine/Elements-ext4/surrogate_data/raw'
DEVICE = 'cuda'
BATCH_SIZE = 5000  # pad to this for GPU saturation
DT_VALUES = [0.005, 0.01, 0.02, 0.05, 0.1]


def build_tier_protocols(tier: int, dt: float = 0.01) -> list:
    """Build all protocols for a tier at given dt."""
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
        protos = [
            StepClamp(v_test=v) for v in [-60, -40, -20, 0, 20, 40]
        ] + [RampClamp(), StaircaseClamp()]
    elif tier == 7:
        combos = [
            {'Ko': 4.0}, {'Ko': 6.0}, {'Ko': 8.0}, {'Ko': 10.0},
            {'Nai_init': 6.0}, {'Nai_init': 12.0}, {'Nai_init': 15.0},
            {'Cai_scale': 0.5}, {'Cai_scale': 1.5}, {'Cai_scale': 2.0},
            {'Ko': 8.0, 'Nai_init': 12.0},
            {'Ko': 6.0, 'Cai_scale': 1.5},
            {'Ko': 4.0, 'Nai_init': 6.0},
            {'Ko': 10.0, 'Nai_init': 15.0, 'Cai_scale': 2.0},
        ]
        for combo in combos:
            for bcl in [500, 1000]:
                base = SteadyStatePacing(bcl=bcl, n_beats=5)
                protos.append(ConcentrationPerturbation(base, **combo))
    elif tier == 8:
        # Long pacing (fewer beats at coarser dt for speed)
        for bcl in [500, 1000]:
            protos.append(SteadyStatePacing(bcl=bcl, n_beats=50))
        for rest_s in [2, 5, 10]:
            protos.append(QuiescentProtocol(duration_ms=rest_s * 1000))
    elif tier == 9:
        base = QuiescentProtocol(duration_ms=100)
        for ctype in ['random_gates', 'extreme_ca']:
            for sev in [0.3, 0.5, 0.8]:
                protos.append(CorruptionRecovery(base, ctype, sev))
    elif tier == 10:
        base = SteadyStatePacing(bcl=1000, n_beats=5)
        for name, inj in [
            ('boundary', SustainedOffset(1.0)),
            ('infarct', SustainedOffset(2.0)),
            ('inert_sink', SustainedOffset(3.0)),
        ]:
            protos.append(InjectedPacing(copy.deepcopy(base), inj, f'_{name}'))
        for di in [40, 60, 80]:
            protos.append(S1S2Restitution(s2_di=di, s1_bcl=500, s1_beats=5))
    elif tier == 11:
        rng = np.random.RandomState(42)
        pool = [
            SteadyStatePacing(bcl=500, n_beats=3),
            SteadyStatePacing(bcl=1000, n_beats=3),
            S1S2Restitution(s2_di=200, s1_beats=3, s1_bcl=500),
        ]
        for i in range(50):
            n_sub = rng.randint(2, 5)
            subs = [copy.deepcopy(pool[rng.randint(len(pool))]) for _ in range(n_sub)]
            rests = [float(np.exp(rng.uniform(np.log(500), np.log(5000))))
                     for _ in range(n_sub - 1)]
            protos.append(StitchedProtocol(protocols=subs, rest_durations=rests))

    # Set dt on all non-stitched protocols
    for p in protos:
        if isinstance(p, StitchedProtocol):
            for sp in p.protocols:
                sp.dt_default = dt
        elif hasattr(p, 'dt_default'):
            p.dt_default = dt
        if isinstance(p, ConcentrationPerturbation):
            p.base_protocol.dt_default = dt
        if isinstance(p, CorruptionRecovery):
            p.base_protocol.dt_default = dt

    return protos


def run_batchable(protos, gen_batch, storage, tier, dt):
    """Run protocols that can be batched (no special handling needed)."""
    if not protos:
        return

    traces = gen_batch.run_batch(protos, progress_interval=10.0)
    for trace in traces:
        name = f'{trace.metadata["protocol_name"]}_dt{dt}'
        storage.save_trace(trace, tier=tier, protocol_name=name)

    print(f'    Saved {len(traces)} traces to tier {tier}')


def run_sequential(protos, gen_seq, storage, tier, dt):
    """Run protocols that need sequential execution (concentration, corruption, stitched)."""
    for i, p in enumerate(protos):
        trace = gen_seq.run_protocol(p)
        name = f'{trace.metadata["protocol_name"]}_dt{dt}'
        storage.save_trace(trace, tier=tier, protocol_name=name)
        if (i + 1) % 10 == 0:
            print(f'    Sequential: {i+1}/{len(protos)}')

    print(f'    Saved {len(protos)} traces to tier {tier}')


def run_tier(tier, gen_batch, gen_seq, storage, dt=0.01, dry_run=False):
    """Run a single tier."""
    protos = build_tier_protocols(tier, dt)
    if not protos:
        print(f'  Tier {tier}: no protocols')
        return

    # Separate batchable vs sequential
    batchable = []
    sequential = []
    for p in protos:
        if isinstance(p, (ConcentrationPerturbation, CorruptionRecovery, StitchedProtocol)):
            sequential.append(p)
        else:
            batchable.append(p)

    total = len(batchable) + len(sequential)
    print(f'  Tier {tier}: {total} protocols ({len(batchable)} batch, {len(sequential)} seq), dt={dt}')

    if dry_run:
        return

    t0 = time.time()

    if batchable:
        run_batchable(batchable, gen_batch, storage, tier, dt)

    if sequential:
        run_sequential(sequential, gen_seq, storage, tier, dt)

    elapsed = time.time() - t0
    print(f'  Tier {tier} done in {elapsed:.1f}s')


def main():
    parser = argparse.ArgumentParser(description='Generate surrogate training data')
    parser.add_argument('--tiers', nargs='+', type=int, default=list(range(1, 13)))
    parser.add_argument('--multi-dt', action='store_true',
                        help='Run Tiers 1-4 at 5 dt values')
    parser.add_argument('--celltypes', nargs='+', default=['EPI'])
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--raw-dir', default=RAW_DIR)
    args = parser.parse_args()

    print('=' * 60)
    print('Surrogate Data Generation')
    print(f'  Tiers: {args.tiers}')
    print(f'  Celltypes: {args.celltypes}')
    print(f'  Multi-dt: {args.multi_dt}')
    print(f'  Batch size: {args.batch_size}')
    print(f'  Output: {args.raw_dir}')
    print(f'  Device: {DEVICE}')
    print('=' * 60)

    if not args.dry_run:
        storage = TraceStorage(args.raw_dir)

    t_total = time.time()

    for ct in args.celltypes:
        print(f'\n=== Celltype: {ct} ===')

        gen_batch = None if args.dry_run else BatchGenerator(
            cell_type=ct, device=DEVICE, use_compile=True)
        gen_seq = None if args.dry_run else SingleCellGenerator(
            cell_type=ct, device='cpu')  # sequential on CPU

        for tier in args.tiers:
            if tier == 12:
                continue  # handled below

            dts = DT_VALUES if (args.multi_dt and tier <= 4) else [0.01]

            for dt in dts:
                run_tier(tier, gen_batch, gen_seq,
                         storage if not args.dry_run else None,
                         dt=dt, dry_run=args.dry_run)

    # Tier 12: other celltypes
    if 12 in args.tiers:
        for ct in ['ENDO', 'M_CELL']:
            print(f'\n=== Tier 12: {ct} ===')
            gen_batch = None if args.dry_run else BatchGenerator(
                cell_type=ct, device=DEVICE, use_compile=True)
            gen_seq = None if args.dry_run else SingleCellGenerator(
                cell_type=ct, device='cpu')

            for tier in [1, 2, 3]:
                run_tier(tier, gen_batch, gen_seq,
                         storage if not args.dry_run else None,
                         dt=0.01, dry_run=args.dry_run)

    elapsed = time.time() - t_total
    print(f'\n{"=" * 60}')
    print(f'TOTAL: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)')

    if not args.dry_run:
        # Show output size
        raw_path = Path(args.raw_dir)
        total_size = sum(f.stat().st_size for f in raw_path.glob('*.h5'))
        print(f'Output size: {total_size/1e9:.1f} GB')


if __name__ == '__main__':
    main()
