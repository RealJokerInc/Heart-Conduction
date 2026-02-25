#!/usr/bin/env python3
"""
Stage 6 Validation: LUT (Lookup Table) Optimization

Tests:
6.1: LUT interpolation accuracy
6.2: LUT vs direct: single cell AP
6.3: LUT vs direct: tissue simulation
6.4: LUT speedup factor
6.5: LUT memory overhead
6.6: Edge case handling (V at bounds)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time


def print_test_header(test_id: str, description: str):
    """Print test header."""
    print(f"\n[Test {test_id}] {description}")
    print("-" * 60)


def print_result(passed: bool, message: str = ""):
    """Print test result."""
    status = "PASS" if passed else "FAIL"
    symbol = "✓" if passed else "✗"
    if message:
        print(f"  {symbol} {status}: {message}")
    else:
        print(f"  {symbol} {status}")
    return passed


def test_lut_interpolation_accuracy():
    """
    Test 6.1: LUT interpolation matches direct computation within tolerance.
    """
    print_test_header("6.1", "LUT interpolation accuracy")

    from ionic.lut import TTP06LUT
    from ionic.ttp06.gating import (
        INa_m_inf, INa_m_tau, INa_h_inf, INa_h_tau,
        ICaL_d_inf, ICaL_d_tau, IKr_Xr1_inf, IKs_Xs_tau
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lut = TTP06LUT(device=device)

    # Test over full voltage range with random samples
    V_test = torch.linspace(-100, 60, 10000, device=device, dtype=torch.float64)

    # Test key functions
    test_cases = [
        ('m_inf', INa_m_inf),
        ('m_tau', INa_m_tau),
        ('h_inf', INa_h_inf),
        ('h_tau', INa_h_tau),
        ('d_inf', ICaL_d_inf),
        ('d_tau', ICaL_d_tau),
        ('Xr1_inf', IKr_Xr1_inf),
        ('Xs_tau', IKs_Xs_tau),
    ]

    max_errors = {}
    all_passed = True

    for name, func in test_cases:
        lut_vals = lut.lookup(name, V_test)
        direct_vals = func(V_test)

        # Relative error (with epsilon to avoid division by zero)
        rel_error = (lut_vals - direct_vals).abs() / (direct_vals.abs() + 1e-10)
        max_rel_error = rel_error.max().item()
        max_errors[name] = max_rel_error

        # Allow up to 1% relative error (h_tau has discontinuity at V=-40)
        if max_rel_error > 0.01:
            all_passed = False

    print(f"  Max relative errors:")
    for name, err in max_errors.items():
        status = "✓" if err < 0.01 else "✗"
        print(f"    {status} {name}: {err*100:.4f}%")

    return print_result(all_passed, f"Max error < 1% for all functions")


def test_lut_single_cell():
    """
    Test 6.2: LUT vs direct single cell AP produces similar results.
    """
    print_test_header("6.2", "LUT vs direct: single cell AP")

    from ionic import TTP06Model, CellType

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create models with and without LUT
    model_direct = TTP06Model(celltype=CellType.EPI, device=device, use_lut=False)
    model_lut = TTP06Model(celltype=CellType.EPI, device=device, use_lut=True)

    # Get initial states
    state_direct = model_direct.get_initial_state()
    state_lut = model_lut.get_initial_state()

    dt = 0.02
    V_direct = []
    V_lut = []

    # Run 500 ms simulation with stimulus at 10 ms
    for t_idx in range(25000):
        t = t_idx * dt
        I_stim = torch.tensor(-52.0, device=device) if 10.0 <= t < 11.0 else None

        state_direct = model_direct.step(state_direct, dt, I_stim)
        state_lut = model_lut.step(state_lut, dt, I_stim)

        if t_idx % 50 == 0:  # Save every 1 ms
            V_direct.append(model_direct.get_voltage(state_direct).item())
            V_lut.append(model_lut.get_voltage(state_lut).item())

    V_direct = np.array(V_direct)
    V_lut = np.array(V_lut)

    # Compare
    max_diff = np.max(np.abs(V_direct - V_lut))
    V_peak_direct = V_direct.max()
    V_peak_lut = V_lut.max()

    passed = max_diff < 0.5  # Less than 0.5 mV difference

    print(f"  Direct: V_peak = {V_peak_direct:.2f} mV")
    print(f"  LUT:    V_peak = {V_peak_lut:.2f} mV")
    print(f"  Max difference: {max_diff:.4f} mV")

    return print_result(passed, f"Max diff = {max_diff:.4f} mV")


def test_lut_tissue():
    """
    Test 6.3: LUT vs direct tissue simulation gives similar CV.
    """
    print_test_header("6.3", "LUT vs direct: tissue CV")

    from fem import TriangularMesh
    from ionic import TTP06Model, CellType
    from tissue import MonodomainSimulation
    from tissue.simulation import SimulationConfig

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Small 1D cable for speed
    mesh = TriangularMesh.create_rectangle(1.0, 0.1, 51, 6, device=device)

    # Run with direct model
    model_direct = TTP06Model(celltype=CellType.EPI, device=device, use_lut=False)
    config = SimulationConfig(D=1.5, chi=1400.0, dt=0.01, save_interval=0.5)
    sim_direct = MonodomainSimulation(mesh, model_direct, config)
    sim_direct.add_stimulus(lambda x, y: x < 0.1, start_time=1.0, duration=1.0)
    times_direct, V_direct = sim_direct.run(15.0)

    # Run with LUT model
    model_lut = TTP06Model(celltype=CellType.EPI, device=device, use_lut=True)
    sim_lut = MonodomainSimulation(mesh, model_lut, config)
    sim_lut.add_stimulus(lambda x, y: x < 0.1, start_time=1.0, duration=1.0)
    times_lut, V_lut = sim_lut.run(15.0)

    # Compare peak voltages
    V_peak_direct = V_direct.max()
    V_peak_lut = V_lut.max()

    # Compare overall traces
    max_diff = np.max(np.abs(V_direct - V_lut))

    passed = max_diff < 2.0 and V_peak_lut > 0  # Both produce APs

    print(f"  Direct V_peak: {V_peak_direct:.1f} mV")
    print(f"  LUT V_peak: {V_peak_lut:.1f} mV")
    print(f"  Max difference: {max_diff:.2f} mV")

    return print_result(passed, f"Max diff = {max_diff:.2f} mV")


def test_lut_speedup():
    """
    Test 6.4: LUT provides speedup over direct computation.
    """
    print_test_header("6.4", "LUT speedup factor")

    from ionic import TTP06Model, CellType

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Large batch to see speedup
    n_cells = 100000

    model_direct = TTP06Model(celltype=CellType.EPI, device=device, use_lut=False)
    model_lut = TTP06Model(celltype=CellType.EPI, device=device, use_lut=True)

    state_direct = model_direct.get_initial_state(n_cells)
    state_lut = model_lut.get_initial_state(n_cells)

    dt = 0.02
    n_steps = 100

    # Warmup
    for _ in range(10):
        state_direct = model_direct.step(state_direct, dt)
        state_lut = model_lut.step(state_lut, dt)

    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark direct
    t0 = time.perf_counter()
    for _ in range(n_steps):
        state_direct = model_direct.step(state_direct, dt)
    if device == 'cuda':
        torch.cuda.synchronize()
    time_direct = time.perf_counter() - t0

    # Benchmark LUT
    t0 = time.perf_counter()
    for _ in range(n_steps):
        state_lut = model_lut.step(state_lut, dt)
    if device == 'cuda':
        torch.cuda.synchronize()
    time_lut = time.perf_counter() - t0

    speedup = time_direct / time_lut if time_lut > 0 else 0

    # LUT should provide some speedup (at least 1.5x for GPU, may vary for CPU)
    passed = speedup >= 1.2

    print(f"  Direct: {time_direct*1000:.1f} ms for {n_steps} steps")
    print(f"  LUT:    {time_lut*1000:.1f} ms for {n_steps} steps")
    print(f"  Speedup: {speedup:.2f}x")

    return print_result(passed, f"Speedup = {speedup:.2f}x")


def test_lut_memory():
    """
    Test 6.5: LUT memory overhead is reasonable.
    """
    print_test_header("6.5", "LUT memory overhead")

    from ionic.lut import TTP06LUT

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lut = TTP06LUT(device=device)

    memory_mb = lut.memory_mb
    n_tables = len(lut.tables)

    # Should be less than 50 MB
    passed = memory_mb < 50.0

    print(f"  Number of tables: {n_tables}")
    print(f"  Total memory: {memory_mb:.2f} MB")

    return print_result(passed, f"Memory = {memory_mb:.2f} MB")


def test_lut_edge_cases():
    """
    Test 6.6: LUT handles edge cases (voltage at/beyond bounds).
    """
    print_test_header("6.6", "Edge case handling")

    from ionic.lut import TTP06LUT
    from ionic.ttp06.gating import INa_m_inf

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lut = TTP06LUT(device=device)

    # Test voltages at and beyond bounds
    V_tests = torch.tensor([-150.0, -100.0, -50.0, 0.0, 50.0, 80.0, 150.0],
                           device=device, dtype=torch.float64)

    # Should not crash and should give reasonable values
    try:
        lut_vals = lut.lookup('m_inf', V_tests)
        no_crash = True
        no_nan = not torch.any(torch.isnan(lut_vals)).item()
        no_inf = not torch.any(torch.isinf(lut_vals)).item()
        in_range = torch.all((lut_vals >= 0) & (lut_vals <= 1)).item()
    except Exception as e:
        no_crash = False
        no_nan = False
        no_inf = False
        in_range = False
        print(f"  Error: {e}")

    passed = no_crash and no_nan and no_inf and in_range

    print(f"  No crash: {no_crash}")
    print(f"  No NaN: {no_nan}")
    print(f"  No Inf: {no_inf}")
    print(f"  Values in [0, 1]: {in_range}")

    return print_result(passed, "Edge cases handled correctly")


def main():
    print("=" * 70)
    print("Stage 6 Validation: LUT (Lookup Table) Optimization")
    print("=" * 70)

    results = []

    results.append(("6.1", "LUT interpolation accuracy", test_lut_interpolation_accuracy()))
    results.append(("6.2", "LUT vs direct: single cell", test_lut_single_cell()))
    results.append(("6.3", "LUT vs direct: tissue", test_lut_tissue()))
    results.append(("6.4", "LUT speedup", test_lut_speedup()))
    results.append(("6.5", "LUT memory overhead", test_lut_memory()))
    results.append(("6.6", "Edge cases", test_lut_edge_cases()))

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    n_passed = sum(1 for _, _, p in results if p)
    n_total = len(results)

    for test_id, name, passed in results:
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {test_id}: {name} - {status}")

    print("-" * 70)
    print(f"Passed: {n_passed}/{n_total}")
    print("=" * 70)

    return 0 if n_passed == n_total else 1


if __name__ == '__main__':
    sys.exit(main())
