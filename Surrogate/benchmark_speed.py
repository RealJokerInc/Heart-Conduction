"""Benchmark TTP06 execution speed across configurations."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'Bidomain' / 'Engine_V1'))
sys.path.insert(0, str(Path(__file__).resolve().parents[0] / '..' / 'Bidomain' / 'Engine_V1'))

import torch
from cardiac_sim.ionic.ttp06.model import TTP06Model
from cardiac_sim.ionic.ttp06.parameters import V_REST
from cardiac_sim.ionic.base import CellType


def benchmark_single(device, use_lut, n_beats=1, bcl=1000.0, dt=0.01):
    """Benchmark single-cell TTP06 for n_beats at given BCL."""
    model = TTP06Model(cell_type=CellType.EPI, device=torch.device(device),
                       use_lut=use_lut)
    V = torch.tensor(V_REST, dtype=torch.float64, device=device)
    states = model.get_initial_state(n_cells=1).to(device)
    n_steps = int(bcl * n_beats / dt)
    stim_dur = 1.0  # ms
    stim_amp = -80.0

    if device == 'cuda':
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for step in range(n_steps):
        t = step * dt
        I_stim = torch.tensor(stim_amp if (t % bcl) < stim_dur else 0.0,
                               dtype=torch.float64, device=device)
        V, states = model.step(V, states, dt, I_stim=I_stim)

    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return elapsed, n_steps, V.item()


def benchmark_batch(device, use_lut, n_cells, n_beats=1, bcl=1000.0, dt=0.01):
    """Benchmark batched TTP06 — n_cells in parallel."""
    model = TTP06Model(cell_type=CellType.EPI, device=torch.device(device),
                       use_lut=use_lut)
    V = torch.full((n_cells,), V_REST, dtype=torch.float64, device=device)
    states = model.get_initial_state(n_cells=n_cells).to(device)
    n_steps = int(bcl * n_beats / dt)
    stim_dur = 1.0
    stim_amp = -80.0

    if device == 'cuda':
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for step in range(n_steps):
        t = step * dt
        I_stim = torch.full((n_cells,), stim_amp if (t % bcl) < stim_dur else 0.0,
                             dtype=torch.float64, device=device)
        V, states = model.step(V, states, dt, I_stim=I_stim)

    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return elapsed, n_steps, V[0].item()


def lut_accuracy_check():
    """Compare LUT vs no-LUT Vm traces."""
    print("\n=== LUT Accuracy Check ===")
    results = {}
    for use_lut in [False, True]:
        model = TTP06Model(cell_type=CellType.EPI, device=torch.device('cpu'),
                           use_lut=use_lut)
        V = torch.tensor(V_REST, dtype=torch.float64)
        states = model.get_initial_state(n_cells=1)
        dt = 0.01
        Vs = []
        for step in range(int(500 / dt)):
            t = step * dt
            I_stim = torch.tensor(-80.0 if t < 1.0 else 0.0, dtype=torch.float64)
            V, states = model.step(V, states, dt, I_stim=I_stim)
            if step % 100 == 0:
                Vs.append(V.item())
        results['lut' if use_lut else 'no_lut'] = Vs

    max_diff = max(abs(a - b) for a, b in zip(results['no_lut'], results['lut']))
    print(f"  Max Vm difference: {max_diff:.2e} mV")
    print(f"  {'PASS' if max_diff < 1e-3 else 'FAIL'}: LUT accuracy {'< 1e-3 mV' if max_diff < 1e-3 else f'= {max_diff:.2e} mV'}")
    return max_diff


def main():
    print("=" * 60)
    print("TTP06 Speed Benchmark")
    print("=" * 60)

    configs = [
        ('CPU', 'cpu', False),
        # ('CPU+LUT', 'cpu', True),  # LUT has a bug in get_all_gating — skip
    ]
    if torch.cuda.is_available():
        configs.extend([
            ('GPU', 'cuda', False),
            # ('GPU+LUT', 'cuda', True),  # LUT has a bug — skip
        ])
    else:
        print("CUDA not available — skipping GPU benchmarks")

    # Single-cell benchmarks (1 beat at BCL=1000ms, dt=0.01ms = 100K steps)
    print("\n=== Single Cell: 1 beat (100K steps) ===")
    print(f"{'Config':<12} {'Time (s)':<10} {'Steps/s':<12} {'V_final (mV)':<14}")
    print("-" * 48)
    for name, device, lut in configs:
        elapsed, n_steps, V_final = benchmark_single(device, lut, n_beats=1)
        steps_per_s = n_steps / elapsed
        print(f"{name:<12} {elapsed:<10.2f} {steps_per_s:<12.0f} {V_final:<14.2f}")

    # Batch benchmarks (GPU only)
    if torch.cuda.is_available():
        print("\n=== GPU Batch: 1 beat, varying n_cells ===")
        print(f"{'n_cells':<10} {'Time (s)':<10} {'Total steps/s':<14} {'Per-cell steps/s':<16}")
        print("-" * 50)
        for n_cells in [1, 10, 50, 100, 200, 500]:
            try:
                elapsed, n_steps, V_final = benchmark_batch('cuda', False, n_cells, n_beats=1)
                total_sps = n_steps * n_cells / elapsed
                per_cell_sps = n_steps / elapsed
                print(f"{n_cells:<10} {elapsed:<10.2f} {total_sps:<14.0f} {per_cell_sps:<16.0f}")
            except Exception as e:
                print(f"{n_cells:<10} FAILED: {e}")
                break

    # LUT accuracy — skipped, LUT has a bug in get_all_gating
    # lut_accuracy_check()
    print("\n=== LUT Accuracy: SKIPPED (engine bug in TTP06LUT.get_all_gating) ===")

    # Estimate total generation time
    print("\n=== Generation Time Estimates ===")
    # Use the fastest single-cell config
    _, n_steps, _ = benchmark_single('cuda' if torch.cuda.is_available() else 'cpu',
                                      False, n_beats=1)
    elapsed_1beat, _, _ = benchmark_single('cuda' if torch.cuda.is_available() else 'cpu',
                                            False, n_beats=1)

    # Rough estimate: ~200K beats total, average 300ms each
    total_beats = 200_000
    avg_beat_ms = 300
    total_steps = total_beats * int(avg_beat_ms / 0.01)
    est_time = total_steps / (n_steps / elapsed_1beat)
    print(f"  Estimated {total_beats:,} beats × {avg_beat_ms}ms avg")
    print(f"  Total steps: {total_steps:,.0f}")
    print(f"  At {n_steps/elapsed_1beat:.0f} steps/s: {est_time/3600:.1f} hours (sequential)")


if __name__ == '__main__':
    main()
