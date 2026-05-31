#!/usr/bin/env python
"""Generate all training data for the ionic surrogate model.

Runs all 12 protocol tiers through TTP06, stores to HDF5 on external HDD,
then converts to .pt training shards.

Usage:
    # Generate all tiers (full dataset, ~1.1TB, several hours):
    python generate_all.py

    # Generate specific tiers only:
    python generate_all.py --tiers 1 2 3

    # Generate with specific cell type:
    python generate_all.py --tiers 1 --celltypes EPI

    # Skip shard conversion (HDF5 only):
    python generate_all.py --no-shards

    # Dry run (show what would be generated):
    python generate_all.py --dry-run

Estimated time: ~4-6 hours on GPU, ~20+ hours on CPU.
Estimated storage: ~1.1TB HDF5 + ~1.1TB shards = ~2.2TB total.
"""

import sys
import time
import argparse
import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np

from surrogate.data.single_cell_generator import SingleCellGenerator
from surrogate.data.protocols import (
    ProtocolLibrary, SteadyStatePacing, S1S2Restitution,
    ConcentrationPerturbation, CorruptionRecovery, QuiescentProtocol,
    RandomIntervalPacing,
)
from surrogate.data.injection import (
    OUNoiseInjection, RampInjection, SubThresholdBlips,
    SustainedOffset, BiphasicPulse, RandomTelegraph, InjectedPacing,
)
from surrogate.data.clamp import (
    StepClamp, RampClamp, StaircaseClamp, APClamp, PartialClamp,
)
from surrogate.data.augmentation import StitchedProtocol
from surrogate.data.storage import TraceStorage, ShardProcessor


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

RAW_DIR = '/media/norepinephrine/Elements-ext4/surrogate_data/raw'
SHARD_DIR = '/media/norepinephrine/Elements-ext4/surrogate_data/train'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DT_VALUES = [0.005, 0.01, 0.02, 0.05, 0.1]  # ms
CELLTYPES = ['EPI', 'ENDO', 'M_CELL']
CONCENTRATION_COMBOS = [
    {'Ko': 4.0}, {'Ko': 5.0}, {'Ko': 6.0}, {'Ko': 8.0}, {'Ko': 10.0},
    {'Nai_init': 6.0}, {'Nai_init': 12.0}, {'Nai_init': 15.0},
    {'Cai_scale': 0.5}, {'Cai_scale': 1.5}, {'Cai_scale': 2.0},
    {'Ko': 8.0, 'Nai_init': 12.0},
    {'Ko': 6.0, 'Cai_scale': 1.5},
    {'Ko': 8.0, 'Cai_scale': 2.0},
    {'Ko': 4.0, 'Nai_init': 6.0},
    {'Ko': 10.0, 'Nai_init': 15.0, 'Cai_scale': 2.0},
    {'Ko': 6.0, 'Nai_init': 8.0, 'Cai_scale': 0.5},
    {'Ko': 5.0, 'Cai_scale': 1.5},
    {'Nai_init': 10.0, 'Cai_scale': 2.0},
    {'Ko': 8.0, 'Nai_init': 15.0},
]


# ──────────────────────────────────────────────────────────────
# Tier generators
# ──────────────────────────────────────────────────────────────

def generate_tier1(gen, storage, dt=0.01):
    """Tier 1: Steady-state pacing — 9 BCLs × 20 beats."""
    protocols = ProtocolLibrary.tier1()
    for p in protocols:
        p.dt_default = dt
        trace = gen.run_protocol(p)
        name = f'{p.name}_dt{dt}'
        storage.save_trace(trace, tier=1, protocol_name=name)
        print(f'  T1: {name} ({trace.data.shape[0]} steps)')


def generate_tier2(gen, storage, dt=0.01):
    """Tier 2: S1-S2 restitution — 8 DI values."""
    protocols = ProtocolLibrary.tier2()
    for p in protocols:
        p.dt_default = dt
        trace = gen.run_protocol(p)
        name = f'{p.name}_dt{dt}'
        storage.save_trace(trace, tier=2, protocol_name=name)
        print(f'  T2: {name} ({trace.data.shape[0]} steps)')


def generate_tier3(gen, storage, dt=0.01):
    """Tier 3: Dynamic protocols — ramp, burst, alternans."""
    protocols = ProtocolLibrary.tier3()
    for p in protocols:
        p.dt_default = dt
        trace = gen.run_protocol(p)
        name = f'{p.name}_dt{dt}'
        storage.save_trace(trace, tier=3, protocol_name=name)
        print(f'  T3: {name} ({trace.data.shape[0]} steps)')


def generate_tier4(gen, storage, n_protocols=200, dt=0.01):
    """Tier 4: Random intervals — 200 protocols, 5-200 beats each."""
    protocols = ProtocolLibrary.tier4(n_protocols=n_protocols)
    for i, p in enumerate(protocols):
        p.dt_default = dt
        trace = gen.run_protocol(p)
        name = f'{p.name}_dt{dt}'
        storage.save_trace(trace, tier=4, protocol_name=name)
        if i % 20 == 0:
            print(f'  T4: {i}/{n_protocols} ({trace.data.shape[0]} steps)')


def generate_tier5(gen, storage, dt=0.01):
    """Tier 5: Tissue-mimicking current injection."""
    base = SteadyStatePacing(bcl=1000, n_beats=5, dt_default=dt)

    injections = [
        ('ou_t5_s10', OUNoiseInjection(tau=5, sigma=10, duration_ms=base.duration_ms, seed=0)),
        ('ou_t10_s20', OUNoiseInjection(tau=10, sigma=20, duration_ms=base.duration_ms, seed=1)),
        ('ou_t2_s5', OUNoiseInjection(tau=2, sigma=5, duration_ms=base.duration_ms, seed=2)),
        ('ramp_fast', RampInjection(peak=-30, ramp_time=2.0, onset=500)),
        ('ramp_slow', RampInjection(peak=-20, ramp_time=5.0, onset=500)),
        ('blips', SubThresholdBlips(amplitude=-15, duration=2, rate=0.02,
                                    total_duration=base.duration_ms, seed=0)),
        ('sustained_neg5', SustainedOffset(amplitude=-5.0)),
        ('sustained_pos3', SustainedOffset(amplitude=3.0)),
        ('biphasic', BiphasicPulse(depol_amp=-20, hyperpol_amp=10, onset=500)),
        ('telegraph', RandomTelegraph(I_max=-15, rate=3, duration_ms=base.duration_ms, seed=0)),
    ]

    for name, inj in injections:
        proto = InjectedPacing(base, inj, name_suffix=f'_{name}')
        trace = gen.run_protocol(proto)
        storage.save_trace(trace, tier=5, protocol_name=f'{name}_dt{dt}')
        print(f'  T5: {name} ({trace.data.shape[0]} steps)')


def generate_tier6(gen, storage, dt=0.01):
    """Tier 6: Voltage clamp protocols."""
    protocols = [
        StepClamp(v_test=-60, dt_default=dt),
        StepClamp(v_test=-40, dt_default=dt),
        StepClamp(v_test=-20, dt_default=dt),
        StepClamp(v_test=0, dt_default=dt),
        StepClamp(v_test=20, dt_default=dt),
        StepClamp(v_test=40, dt_default=dt),
        RampClamp(ramp_duration=300, dt_default=dt),
        StaircaseClamp(step_duration=100, dt_default=dt),
    ]

    for p in protocols:
        trace = gen.run_protocol(p)
        storage.save_trace(trace, tier=6, protocol_name=f'{p.name}_dt{dt}')
        print(f'  T6: {p.name} ({trace.data.shape[0]} steps)')

    # AP clamp: use a previously generated AP as waveform
    ref = gen.run_pacing(bcl=1000, n_beats=2, dt=dt)
    n_beat = int(1000 / dt)
    vm_waveform = ref.data[n_beat:n_beat + int(500 / dt), 0]
    ap_clamp = APClamp(vm_waveform=vm_waveform, dt_waveform=dt, dt_default=dt)
    trace = gen.run_protocol(ap_clamp)
    storage.save_trace(trace, tier=6, protocol_name=f'ap_clamp_dt{dt}')
    print(f'  T6: ap_clamp ({trace.data.shape[0]} steps)')


def generate_tier7(gen, storage, dt=0.01):
    """Tier 7: Concentration perturbation — 20 combos × Tier 1 subset."""
    bcls = [500, 1000, 2000]  # subset of Tier 1
    for combo in CONCENTRATION_COMBOS:
        for bcl in bcls:
            base = SteadyStatePacing(bcl=bcl, n_beats=5, dt_default=dt)
            proto = ConcentrationPerturbation(base_protocol=base, **combo)
            trace = gen.run_protocol(proto)
            combo_str = '_'.join(f'{k}{v}' for k, v in combo.items())
            name = f'conc_{combo_str}_bcl{bcl}_dt{dt}'
            storage.save_trace(trace, tier=7, protocol_name=name)
    print(f'  T7: {len(CONCENTRATION_COMBOS) * len(bcls)} protocols')


def generate_tier8(gen, storage, dt=0.01):
    """Tier 8: Long-duration stability."""
    # Long pacing
    for bcl in [500, 1000]:
        proto = SteadyStatePacing(bcl=bcl, n_beats=50, dt_default=max(dt, 0.1))
        trace = gen.run_protocol(proto)
        storage.save_trace(trace, tier=8, protocol_name=f'long_bcl{bcl}_dt{dt}')
        print(f'  T8: long_bcl{bcl} ({trace.data.shape[0]} steps)')

    # Long quiescence
    for rest_s in [2, 5, 10]:
        proto = QuiescentProtocol(duration_ms=rest_s * 1000, dt_default=max(dt, 1.0))
        trace = gen.run_protocol(proto)
        storage.save_trace(trace, tier=8, protocol_name=f'quiescent_{rest_s}s')
        print(f'  T8: quiescent_{rest_s}s ({trace.data.shape[0]} steps)')


def generate_tier9(gen, storage, dt=0.01):
    """Tier 9: Recovery from corruption."""
    base = QuiescentProtocol(duration_ms=100, dt_default=dt)
    for ctype in ['random_gates', 'extreme_ca']:
        for severity in [0.3, 0.5, 0.8]:
            proto = CorruptionRecovery(base, corruption_type=ctype, severity=severity)
            trace = gen.run_protocol(proto)
            name = f'corrupt_{ctype}_s{severity}_dt{dt}'
            storage.save_trace(trace, tier=9, protocol_name=name)
    print(f'  T9: 6 corruption recovery protocols')


def generate_tier10(gen, storage, dt=0.01):
    """Tier 10: Tissue-specific scenarios."""
    base = SteadyStatePacing(bcl=1000, n_beats=5, dt_default=dt)

    scenarios = [
        ('boundary', SustainedOffset(amplitude=1.0)),   # reduced loading
        ('infarct', SustainedOffset(amplitude=2.0)),     # asymmetric
        ('inert_sink', SustainedOffset(amplitude=3.0)),  # repolarizing sink
    ]

    for name, inj in scenarios:
        proto = InjectedPacing(base, inj, name_suffix=f'_{name}')
        trace = gen.run_protocol(proto)
        storage.save_trace(trace, tier=10, protocol_name=f'{name}_dt{dt}')
        print(f'  T10: {name} ({trace.data.shape[0]} steps)')

    # Spiral tip: very short DI pacing
    for di in [40, 60, 80]:
        proto = S1S2Restitution(s2_di=di, s1_bcl=500, s1_beats=5, dt_default=dt)
        trace = gen.run_protocol(proto)
        storage.save_trace(trace, tier=10, protocol_name=f'spiral_di{di}_dt{dt}')
    print(f'  T10: 3 spiral tip protocols')


def generate_tier11(gen, storage, n_stitched=50, dt=0.01):
    """Tier 11: Combined stressors + stitched protocols."""
    rng = np.random.RandomState(42)
    pool = [
        SteadyStatePacing(bcl=500, n_beats=3, dt_default=dt),
        SteadyStatePacing(bcl=1000, n_beats=3, dt_default=dt),
        S1S2Restitution(s2_di=200, s1_beats=3, s1_bcl=500, dt_default=dt),
    ]

    for i in range(n_stitched):
        n_sub = rng.randint(2, 5)
        protos = [copy.deepcopy(pool[rng.randint(len(pool))]) for _ in range(n_sub)]
        rests = [float(np.exp(rng.uniform(np.log(500), np.log(5000))))
                 for _ in range(n_sub - 1)]
        stitched = StitchedProtocol(protocols=protos, rest_durations=rests)
        trace = gen.run_protocol(stitched)
        storage.save_trace(trace, tier=11, protocol_name=f'stitched_{i}')
        if i % 10 == 0:
            print(f'  T11: {i}/{n_stitched} ({trace.data.shape[0]} steps)')


def generate_tier12(storage, dt=0.01):
    """Tier 12: Celltype variants — run Tiers 1-3 for ENDO and M_CELL."""
    for ct in ['ENDO', 'M_CELL']:
        gen = SingleCellGenerator(cell_type=ct, device=DEVICE)
        print(f'  T12: Generating {ct}...')
        generate_tier1(gen, storage, dt=dt)
        generate_tier2(gen, storage, dt=dt)
        generate_tier3(gen, storage, dt=dt)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

TIER_FUNCS = {
    1: lambda gen, s, dt: generate_tier1(gen, s, dt),
    2: lambda gen, s, dt: generate_tier2(gen, s, dt),
    3: lambda gen, s, dt: generate_tier3(gen, s, dt),
    4: lambda gen, s, dt: generate_tier4(gen, s, dt=dt),
    5: lambda gen, s, dt: generate_tier5(gen, s, dt),
    6: lambda gen, s, dt: generate_tier6(gen, s, dt),
    7: lambda gen, s, dt: generate_tier7(gen, s, dt),
    8: lambda gen, s, dt: generate_tier8(gen, s, dt),
    9: lambda gen, s, dt: generate_tier9(gen, s, dt),
    10: lambda gen, s, dt: generate_tier10(gen, s, dt),
    11: lambda gen, s, dt: generate_tier11(gen, s, dt=dt),
    12: lambda _, s, dt: generate_tier12(s, dt),
}


def main():
    parser = argparse.ArgumentParser(description='Generate surrogate training data')
    parser.add_argument('--tiers', nargs='+', type=int, default=list(range(1, 13)),
                        help='Which tiers to generate (default: all 1-12)')
    parser.add_argument('--celltypes', nargs='+', default=['EPI'],
                        help='Cell types for Tiers 1-11 (default: EPI). Tier 12 adds ENDO/M_CELL.')
    parser.add_argument('--dt-values', nargs='+', type=float, default=[0.01],
                        help='dt values to sweep (default: [0.01]). Use --multi-dt for full sweep.')
    parser.add_argument('--multi-dt', action='store_true',
                        help='Run Tiers 1-4 at all 5 dt values')
    parser.add_argument('--no-shards', action='store_true',
                        help='Skip .pt shard conversion')
    parser.add_argument('--segment-length', type=int, default=1000,
                        help='Segment length for shards (default: 1000)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be generated without running')
    parser.add_argument('--raw-dir', default=RAW_DIR)
    parser.add_argument('--shard-dir', default=SHARD_DIR)
    args = parser.parse_args()

    if args.multi_dt:
        dt_sweep = DT_VALUES
    else:
        dt_sweep = args.dt_values

    if args.dry_run:
        print(f'Would generate tiers: {args.tiers}')
        print(f'Cell types: {args.celltypes}')
        print(f'dt values: {dt_sweep}')
        print(f'Raw dir: {args.raw_dir}')
        print(f'Shard dir: {args.shard_dir}')
        print(f'Shards: {"skip" if args.no_shards else "yes"}')
        return

    storage = TraceStorage(args.raw_dir)
    t_start = time.time()

    for ct in args.celltypes:
        gen = SingleCellGenerator(cell_type=ct, device=DEVICE)
        print(f'\n=== Cell type: {ct}, device: {DEVICE} ===')

        for tier in args.tiers:
            if tier == 12:
                continue  # Tier 12 handles its own celltypes
            if tier not in TIER_FUNCS:
                print(f'  Unknown tier {tier}, skipping')
                continue

            # dt sweep for Tiers 1-4, single dt for others
            dts = dt_sweep if (tier <= 4 and args.multi_dt) else [dt_sweep[0]]

            for dt in dts:
                print(f'\n--- Tier {tier}, dt={dt}ms ---')
                t0 = time.time()
                TIER_FUNCS[tier](gen, storage, dt)
                elapsed = time.time() - t0
                print(f'  Done in {elapsed:.1f}s')

    # Tier 12: celltypes
    if 12 in args.tiers:
        print(f'\n--- Tier 12: Celltype variants ---')
        t0 = time.time()
        generate_tier12(storage, dt=dt_sweep[0])
        print(f'  Done in {time.time() - t0:.1f}s')

    total = time.time() - t_start
    print(f'\n=== Data generation complete in {total/3600:.1f} hours ===')

    # Shard conversion
    if not args.no_shards:
        print(f'\n=== Converting to .pt shards (segment_length={args.segment_length}) ===')
        t0 = time.time()
        processor = ShardProcessor(
            args.raw_dir, args.shard_dir,
            segment_length=args.segment_length,
        )
        processor.process_all(tiers=args.tiers)
        print(f'Shard conversion done in {(time.time()-t0)/60:.1f} minutes')


if __name__ == '__main__':
    main()
