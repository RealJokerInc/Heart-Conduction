#!/usr/bin/env python
"""ORd data generation script — CPU single-cell, all tiers.

Usage:
    python run_ord_datagen.py --tiers 1 2 3 --celltypes EPI ENDO M_CELL
    python run_ord_datagen.py --tiers 1 --celltypes EPI --warmup 5  # quick test
    python run_ord_datagen.py --all  # generate everything

Generates 101-column ORd traces and saves to HDD.
CaMKII warmup: 100 beats by default (configurable).
"""

import sys
import time
import argparse
from pathlib import Path

import torch

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'Bidomain' / 'Engine_V1'))

from surrogate.data.ord_single_cell_generator import ORdSingleCellGenerator, ORdConcentrationPerturbation
from surrogate.data.ord_storage import ORdTraceStorage
from surrogate.data.ord_trace_data import ORdTraceData
from surrogate.data.protocols import (
    SteadyStatePacing, S1S2Restitution, BCLRamp, BurstPacing, AlternansProtocol
)

# === Config ===
RAW_DIR = '/media/HDD/surrogate_data/raw_ord'
DT_VALUES = [0.01]  # Start with just 0.01ms, add more later
CELLTYPES = ['EPI', 'ENDO', 'M_CELL']
WARMUP_BEATS = 100

# === Tier definitions ===

def make_tier1_protocols(dt=0.01):
    """Tier 1: Steady-state pacing at multiple BCLs."""
    protocols = []
    for bcl in [300, 400, 500, 600, 700, 800, 1000, 1500, 2000]:
        proto = SteadyStatePacing(bcl=bcl, n_beats=20, dt_default=dt)
        proto.name = f'steady_bcl{bcl}_dt{dt}'
        proto.tier = 1
        protocols.append(proto)
    return protocols


def make_tier2_protocols(dt=0.01):
    """Tier 2: S1-S2 restitution."""
    protocols = []
    for s2_di in [50, 75, 100, 150, 200, 300, 500, 800]:
        proto = S1S2Restitution(s1_bcl=1000, s1_beats=10, s2_di=s2_di, dt_default=dt)
        proto.name = f's1s2_di{s2_di}_dt{dt}'
        proto.tier = 2
        protocols.append(proto)
    return protocols


def make_tier3_protocols(dt=0.01):
    """Tier 3: Dynamic protocols (ramp, burst, alternans)."""
    protocols = []

    # BCL ramp down: 1000 → 300 over 30 beats
    proto = BCLRamp(bcl_start=1000, bcl_end=300, n_beats=30, dt_default=dt)
    proto.name = f'ramp_down_dt{dt}'
    proto.tier = 3
    protocols.append(proto)

    # BCL ramp up: 300 → 1000 over 30 beats
    proto = BCLRamp(bcl_start=300, bcl_end=1000, n_beats=30, dt_default=dt)
    proto.name = f'ramp_up_dt{dt}'
    proto.tier = 3
    protocols.append(proto)

    # Burst: 5 beats at BCL=300, then 2s pause, repeat 5x
    proto = BurstPacing(burst_bcl=300, burst_beats=5, pause_ms=2000, n_bursts=5, dt_default=dt)
    proto.name = f'burst_dt{dt}'
    proto.tier = 3
    protocols.append(proto)

    # Alternans: BCL=330, 20 beats
    proto = AlternansProtocol(bcl=330, n_beats=20, dt_default=dt)
    proto.name = f'alternans_bcl330_dt{dt}'
    proto.tier = 3
    protocols.append(proto)

    return protocols


def make_tier12_protocols(dt=0.01):
    """Tier 12: Celltype-specific (run for each celltype)."""
    # Same as Tier 1 but specifically for celltype comparison
    return make_tier1_protocols(dt=dt)


def make_tier13_protocols(dt=0.01):
    """Tier 13: CaMKII buildup dynamics (ORd-exclusive).

    Long pacing from zero CaMKII state to observe full buildup.
    """
    protocols = []
    for bcl in [500, 1000, 2000]:
        # 200 beats to capture CaMKII equilibration
        proto = SteadyStatePacing(bcl=bcl, n_beats=200, dt_default=dt)
        proto.name = f'camkii_buildup_bcl{bcl}_dt{dt}'
        proto.tier = 13
        protocols.append(proto)
    return protocols


def generate_tier(tier: int, cell_type: str, dt: float, warmup_beats: int,
                  storage: ORdTraceStorage):
    """Generate all protocols for a tier/celltype/dt combination."""
    # Select protocols
    if tier == 1:
        protocols = make_tier1_protocols(dt)
    elif tier == 2:
        protocols = make_tier2_protocols(dt)
    elif tier == 3:
        protocols = make_tier3_protocols(dt)
    elif tier == 12:
        protocols = make_tier12_protocols(dt)
    elif tier == 13:
        # T13: CaMKII buildup — run with warmup_beats=0 to capture full buildup
        protocols = make_tier13_protocols(dt)
        warmup_beats = 0  # Override: no warmup, we WANT to see the buildup
    else:
        print(f"  Tier {tier} not yet implemented for ORd, skipping")
        return 0

    gen = ORdSingleCellGenerator(
        cell_type=cell_type,
        device='cpu',
        warmup_beats=warmup_beats,
    )

    count = 0
    for proto in protocols:
        name = f'{cell_type}_{proto.name}'
        t0 = time.time()
        try:
            trace = gen.run_protocol(proto)
            storage.save_trace(trace, tier=tier, protocol_name=name)
            elapsed = time.time() - t0
            print(f"    {name}: {trace.data.shape[0]} steps, {elapsed:.1f}s")
            count += 1
        except Exception as e:
            print(f"    {name}: FAILED — {e}")

    return count


def main():
    parser = argparse.ArgumentParser(description='ORd data generation')
    parser.add_argument('--tiers', nargs='+', type=int, default=[1])
    parser.add_argument('--celltypes', nargs='+', default=['EPI'])
    parser.add_argument('--dt', nargs='+', type=float, default=[0.01])
    parser.add_argument('--warmup', type=int, default=WARMUP_BEATS)
    parser.add_argument('--all', action='store_true',
                        help='Generate T1-T3, T12, T13 for all celltypes')
    parser.add_argument('--raw-dir', default=RAW_DIR)
    args = parser.parse_args()

    if args.all:
        args.tiers = [1, 2, 3, 12, 13]
        args.celltypes = CELLTYPES
        args.dt = [0.01]

    storage = ORdTraceStorage(base_dir=args.raw_dir)
    print(f"ORd data generation")
    print(f"  Raw dir: {args.raw_dir}")
    print(f"  Tiers: {args.tiers}")
    print(f"  Celltypes: {args.celltypes}")
    print(f"  dt values: {args.dt}")
    print(f"  Warmup beats: {args.warmup}")
    print()

    total_count = 0
    total_t0 = time.time()

    for tier in args.tiers:
        for ct in args.celltypes:
            for dt in args.dt:
                print(f"Tier {tier}, {ct}, dt={dt}:")
                count = generate_tier(tier, ct, dt, args.warmup, storage)
                total_count += count

    total_elapsed = time.time() - total_t0
    print(f"\nDone: {total_count} traces generated in {total_elapsed:.0f}s")


if __name__ == '__main__':
    main()
