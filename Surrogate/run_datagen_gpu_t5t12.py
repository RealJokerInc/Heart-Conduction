#!/usr/bin/env python
"""GPU data generation for T5-T12 (small batches padded for GPU efficiency).

Runs concurrently with CPU T4 generation. Pads small batches to MIN_BATCH
by replicating protocols, then keeps only unique results.
GPU torch.compile needs n>=1000 to amortize kernel launch overhead.

Sequential protocols (ConcentrationPerturbation, CorruptionRecovery,
StitchedProtocol) run on CPU single-cell.
"""

import sys, time, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'Bidomain' / 'Engine_V1'))

import torch
from run_datagen_cpu import build_tier
from surrogate.data.batch_generator import BatchGenerator
from surrogate.data.single_cell_generator import SingleCellGenerator, TraceData
from surrogate.data.storage import TraceStorage
from surrogate.data.protocols import ConcentrationPerturbation, CorruptionRecovery
from surrogate.data.augmentation import StitchedProtocol

RAW_DIR = '/media/norepinephrine/Elements-ext4/surrogate_data/raw'
VRAM_BUDGET_GB = 28  # leave ~4GB headroom for model + schedules + overhead
BYTES_PER_CELL_PER_STEP = (4 + 18) * 8  # 22 float64 recording arrays


def compute_max_batch(protos, dt=0.01):
    """Compute max batch size that fits in GPU VRAM."""
    max_dur = max(p.duration_ms for p in protos)
    n_steps = int(max_dur / dt)
    vram_per_cell = BYTES_PER_CELL_PER_STEP * n_steps
    max_n = int(VRAM_BUDGET_GB * 1e9 / vram_per_cell)
    return max(len(protos), min(max_n, 2000))  # cap at 2000, floor at n_protos


def pad_protocols(protos, target_n):
    """Replicate protocols to reach target_n. Returns (padded_list, n_unique)."""
    n = len(protos)
    if n >= target_n:
        return protos, n
    padded = []
    for i in range(target_n):
        padded.append(copy.deepcopy(protos[i % n]))
    return padded, n


def run_gpu_batch(gpu_gen, protos, storage, tier):
    """Run batchable protocols on GPU with VRAM-aware padding."""
    max_n = compute_max_batch(protos)
    padded, n_unique = pad_protocols(protos, max_n)
    max_dur = max(p.duration_ms for p in protos)
    n_steps = int(max_dur / 0.01)
    print(f'    {len(protos)} protos → padded to {len(padded)} (max {max_n} for {n_steps/1e6:.1f}M steps)')

    traces = gpu_gen.run_batch(padded, progress_interval=10.0)

    # Save only unique traces (first n_unique)
    saved = 0
    for trace in traces[:n_unique]:
        name = f'{trace.metadata["protocol_name"]}_dt{trace.metadata["dt_default"]}'
        storage.save_trace(trace, tier, name)
        saved += 1

    del traces
    torch.cuda.empty_cache()
    print(f'    Saved {saved} unique traces')
    return saved


def main():
    storage = TraceStorage(RAW_DIR)
    t_total = time.time()

    # GPU batch generator (with torch.compile warmup)
    print('Initializing GPU BatchGenerator with torch.compile...')
    gpu_gen = BatchGenerator(cell_type='EPI', device='cuda', use_compile=True)
    # CPU single-cell for sequential protocols
    cpu_gen = SingleCellGenerator(cell_type='EPI', device='cpu')

    for tier in [5, 6, 7, 8, 9, 10, 11]:
        h5_path = Path(RAW_DIR) / f'tier{tier:02d}.h5'
        if h5_path.exists():
            print(f'  Tier {tier}: SKIP (exists)')
            continue

        protos = build_tier(tier)
        if not protos:
            continue

        batchable = [p for p in protos
                     if not isinstance(p, (ConcentrationPerturbation, CorruptionRecovery, StitchedProtocol))]
        sequential = [p for p in protos
                      if isinstance(p, (ConcentrationPerturbation, CorruptionRecovery, StitchedProtocol))]

        print(f'\n  Tier {tier}: {len(batchable)} GPU batch + {len(sequential)} CPU seq')
        t0 = time.time()

        # GPU batch with padding
        if batchable:
            run_gpu_batch(gpu_gen, batchable, storage, tier)

        # CPU sequential
        for i, p in enumerate(sequential):
            trace = cpu_gen.run_protocol(p)
            name = f'{trace.metadata["protocol_name"]}_dt{trace.metadata["dt_default"]}'
            storage.save_trace(trace, tier, name)
            if (i + 1) % 10 == 0:
                print(f'    Sequential: {i+1}/{len(sequential)}')
        if sequential:
            print(f'    CPU seq: {len(sequential)} traces saved')

        print(f'  Tier {tier} done in {time.time()-t0:.0f}s')

    # Tier 12: other celltypes (GPU batch)
    print('\n=== Tier 12: Celltype variants ===')
    for ct in ['ENDO', 'M_CELL']:
        print(f'  Celltype: {ct}')
        ct_gen = BatchGenerator(cell_type=ct, device='cuda', use_compile=True)
        for sub_tier in [1, 2, 3]:
            protos = build_tier(sub_tier)
            batchable = [p for p in protos
                         if not isinstance(p, (ConcentrationPerturbation, CorruptionRecovery, StitchedProtocol))]
            if not batchable:
                continue
            max_n = compute_max_batch(batchable)
            padded, n_unique = pad_protocols(batchable, max_n)
            print(f'    T12/{ct}/T{sub_tier}: {len(batchable)} → {len(padded)} padded')
            traces = ct_gen.run_batch(padded, progress_interval=10.0)
            for trace in traces[:n_unique]:
                name = f'{ct}_{trace.metadata["protocol_name"]}_dt{trace.metadata["dt_default"]}'
                storage.save_trace(trace, 12, name)
            print(f'    T12/{ct}/T{sub_tier}: {n_unique} traces saved')
            del traces
            torch.cuda.empty_cache()

    elapsed = time.time() - t_total
    total_size = sum(f.stat().st_size for f in Path(RAW_DIR).glob('*.h5'))
    print(f'\n{"="*60}')
    print(f'TOTAL: {elapsed/60:.0f} min ({elapsed/3600:.1f} hours)')
    print(f'Output: {total_size/1e9:.1f} GB')


if __name__ == '__main__':
    main()
