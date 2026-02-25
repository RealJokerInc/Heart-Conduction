#!/usr/bin/env python3
"""
Comprehensive LUT (Lookup Table) Testing Script

Tests the lookup table acceleration system for the TTP06 ionic model.

Test Categories:
1. Basic LUT construction and table integrity
2. Interpolation accuracy vs direct functions
3. Batch lookup functionality
4. Edge case handling (bounds, NaN, Inf)
5. Performance benchmarks (speedup factor)
6. Single-cell AP validation (LUT vs Direct)
7. Memory usage analysis
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time
from typing import Tuple, Dict

# Test result tracking
test_results = []


def print_header(title: str):
    """Print section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subtest(name: str):
    """Print subtest name."""
    print(f"\n  [{name}]")
    print("  " + "-" * 50)


def record_result(test_name: str, passed: bool, message: str = ""):
    """Record test result."""
    status = "PASS" if passed else "FAIL"
    symbol = "✓" if passed else "✗"
    test_results.append((test_name, passed, message))
    if message:
        print(f"    {symbol} {status}: {message}")
    else:
        print(f"    {symbol} {status}")
    return passed


def get_device():
    """Get compute device."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# Test 1: LUT Construction and Table Integrity
# =============================================================================

def test_lut_construction():
    """Test that LUT builds correctly with expected properties."""
    print_header("Test 1: LUT Construction and Table Integrity")

    from ionic.lut import TTP06LUT, LUTConfig

    device = get_device()
    print(f"  Device: {device}")

    # Test default construction
    print_subtest("1.1 Default LUT construction")
    try:
        lut = TTP06LUT(device=device)
        record_result("1.1a", True, "LUT created successfully")
    except Exception as e:
        record_result("1.1a", False, f"LUT creation failed: {e}")
        return

    # Verify config
    passed = (lut.V_min == -100.0 and lut.V_max == 80.0 and lut.n_points == 2001)
    record_result("1.1b", passed, f"Config: V=[{lut.V_min}, {lut.V_max}], n={lut.n_points}")

    # Verify dV
    expected_dV = 180.0 / 2000.0  # 0.09 mV
    passed = abs(lut.dV - expected_dV) < 1e-10
    record_result("1.1c", passed, f"dV = {lut.dV:.6f} mV (expected {expected_dV:.6f})")

    # Test table contents
    print_subtest("1.2 Table contents validation")

    expected_tables = [
        'm_inf', 'm_tau', 'h_inf', 'h_tau', 'j_inf', 'j_tau',
        'r_inf', 'r_tau', 's_inf_endo', 's_inf_epi', 's_tau_endo', 's_tau_epi',
        'd_inf', 'd_tau', 'f_inf', 'f_tau', 'f2_inf', 'f2_tau',
        'Xr1_inf', 'Xr1_tau', 'Xr2_inf', 'Xr2_tau',
        'Xs_inf', 'Xs_tau'
    ]

    missing = [t for t in expected_tables if t not in lut.tables]
    passed = len(missing) == 0
    record_result("1.2a", passed, f"All {len(expected_tables)} tables present" if passed
                  else f"Missing tables: {missing}")

    # Check table shapes
    all_correct_shape = all(t.shape == (2001,) for t in lut.tables.values())
    record_result("1.2b", all_correct_shape, "All tables have shape (2001,)")

    # Check for NaN/Inf in tables
    has_nan = any(torch.isnan(t).any() for t in lut.tables.values())
    has_inf = any(torch.isinf(t).any() for t in lut.tables.values())
    record_result("1.2c", not has_nan, "No NaN values in tables")
    record_result("1.2d", not has_inf, "No Inf values in tables")

    # Test steady-state value ranges (should be [0, 1] for _inf tables)
    print_subtest("1.3 Steady-state value ranges")

    inf_tables = [n for n in lut.tables if '_inf' in n]
    for name in inf_tables:
        table = lut.tables[name]
        in_range = (table >= 0).all() and (table <= 1).all()
        record_result(f"1.3.{name}", in_range,
                     f"{name}: [{table.min():.4f}, {table.max():.4f}]")


# =============================================================================
# Test 2: Interpolation Accuracy
# =============================================================================

def test_interpolation_accuracy():
    """Test LUT interpolation accuracy against direct function evaluation."""
    print_header("Test 2: Interpolation Accuracy")

    from ionic.lut import TTP06LUT
    from ionic.ttp06.gating import (
        INa_m_inf, INa_m_tau, INa_h_inf, INa_h_tau, INa_j_inf, INa_j_tau,
        Ito_r_inf, Ito_r_tau, Ito_s_inf_endo, Ito_s_tau_epi,
        ICaL_d_inf, ICaL_d_tau, ICaL_f_inf, ICaL_f_tau,
        IKr_Xr1_inf, IKr_Xr1_tau, IKs_Xs_inf, IKs_Xs_tau
    )

    device = get_device()
    lut = TTP06LUT(device=device)

    # Test voltages at various points (grid and non-grid)
    print_subtest("2.1 Grid-aligned voltages")
    V_grid = torch.tensor([-100.0, -80.0, -60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0, 80.0],
                          device=device, dtype=torch.float64)

    test_funcs = [
        ('m_inf', INa_m_inf),
        ('h_inf', INa_h_inf),
        ('d_inf', ICaL_d_inf),
        ('Xr1_inf', IKr_Xr1_inf),
    ]

    for name, func in test_funcs:
        lut_vals = lut.lookup(name, V_grid)
        direct_vals = func(V_grid)
        max_err = (lut_vals - direct_vals).abs().max().item()
        # Allow small interpolation error since test points may not align exactly with grid
        passed = max_err < 1e-4
        record_result(f"2.1.{name}", passed, f"{name}: max error = {max_err:.2e}")

    # Test non-grid voltages (interpolation required)
    print_subtest("2.2 Interpolated voltages (10000 random points)")
    torch.manual_seed(42)
    V_random = torch.rand(10000, device=device, dtype=torch.float64) * 180.0 - 100.0

    full_test_funcs = [
        ('m_inf', INa_m_inf),
        ('m_tau', INa_m_tau),
        ('h_inf', INa_h_inf),
        ('h_tau', INa_h_tau),
        ('j_inf', INa_j_inf),
        ('j_tau', INa_j_tau),
        ('d_inf', ICaL_d_inf),
        ('d_tau', ICaL_d_tau),
        ('f_inf', ICaL_f_inf),
        ('f_tau', ICaL_f_tau),
        ('Xr1_inf', IKr_Xr1_inf),
        ('Xr1_tau', IKr_Xr1_tau),
        ('Xs_inf', IKs_Xs_inf),
        ('Xs_tau', IKs_Xs_tau),
    ]

    max_errors = {}
    for name, func in full_test_funcs:
        lut_vals = lut.lookup(name, V_random)
        direct_vals = func(V_random)

        # Relative error with small epsilon
        rel_error = (lut_vals - direct_vals).abs() / (direct_vals.abs() + 1e-10)
        max_rel = rel_error.max().item()
        max_errors[name] = max_rel

    # Report all errors
    for name, err in sorted(max_errors.items(), key=lambda x: -x[1]):
        # Allow 1.5% for time constants (discontinuities at V=-40mV), 1% for steady-states
        tol = 0.015 if '_tau' in name else 0.01
        passed = err < tol
        record_result(f"2.2.{name}", passed, f"{name}: max relative error = {err*100:.4f}%")

    # Test physiological voltage range specifically
    print_subtest("2.3 Physiological range [-90, +50 mV]")
    V_physio = torch.linspace(-90, 50, 5000, device=device, dtype=torch.float64)

    physio_errors = {}
    for name, func in [('m_inf', INa_m_inf), ('h_inf', INa_h_inf), ('d_inf', ICaL_d_inf)]:
        lut_vals = lut.lookup(name, V_physio)
        direct_vals = func(V_physio)
        max_err = (lut_vals - direct_vals).abs().max().item()
        physio_errors[name] = max_err
        passed = max_err < 1e-5
        record_result(f"2.3.{name}", passed, f"{name}: max abs error = {max_err:.2e}")


# =============================================================================
# Test 3: Batch Lookup
# =============================================================================

def test_batch_lookup():
    """Test batch lookup functionality."""
    print_header("Test 3: Batch Lookup Functionality")

    from ionic.lut import TTP06LUT

    device = get_device()
    lut = TTP06LUT(device=device)

    # Create test voltages
    V = torch.linspace(-80, 40, 1000, device=device, dtype=torch.float64)

    print_subtest("3.1 Batch vs individual lookup")
    names = ['m_inf', 'm_tau', 'h_inf', 'h_tau', 'd_inf', 'd_tau']

    # Batch lookup
    batch_result = lut.lookup_batch(names, V)

    # Individual lookups
    individual_results = {name: lut.lookup(name, V) for name in names}

    # Compare
    all_match = True
    for name in names:
        diff = (batch_result[name] - individual_results[name]).abs().max().item()
        if diff > 1e-14:
            all_match = False
            record_result(f"3.1.{name}", False, f"Mismatch: {diff:.2e}")

    if all_match:
        record_result("3.1", True, "Batch lookup matches individual lookups exactly")

    print_subtest("3.2 get_all_gating method")

    # Test ENDO cell type
    gating_endo = lut.get_all_gating(V, celltype_is_endo=True)
    expected_keys = ['m_inf', 'm_tau', 'h_inf', 'h_tau', 'j_inf', 'j_tau',
                     'r_inf', 'r_tau', 's_inf', 's_tau',
                     'd_inf', 'd_tau', 'f_inf', 'f_tau', 'f2_inf', 'f2_tau',
                     'Xr1_inf', 'Xr1_tau', 'Xr2_inf', 'Xr2_tau', 'Xs_inf', 'Xs_tau']

    all_present = all(k in gating_endo for k in expected_keys)
    record_result("3.2a", all_present, f"ENDO: all {len(expected_keys)} gating vars returned")

    # Verify ENDO uses endo-specific tables
    endo_s_inf = lut.lookup('s_inf_endo', V)
    matches_endo = torch.allclose(gating_endo['s_inf'], endo_s_inf)
    record_result("3.2b", matches_endo, "ENDO s_inf uses s_inf_endo table")

    # Test EPI cell type
    gating_epi = lut.get_all_gating(V, celltype_is_endo=False)
    epi_s_inf = lut.lookup('s_inf_epi', V)
    matches_epi = torch.allclose(gating_epi['s_inf'], epi_s_inf)
    record_result("3.2c", matches_epi, "EPI s_inf uses s_inf_epi table")


# =============================================================================
# Test 4: Edge Cases
# =============================================================================

def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print_header("Test 4: Edge Cases and Boundary Handling")

    from ionic.lut import TTP06LUT

    device = get_device()
    lut = TTP06LUT(device=device)

    print_subtest("4.1 Voltage at exact bounds")
    V_bounds = torch.tensor([-100.0, 80.0], device=device, dtype=torch.float64)

    try:
        result = lut.lookup('m_inf', V_bounds)
        no_nan = not torch.isnan(result).any()
        no_inf = not torch.isinf(result).any()
        passed = no_nan and no_inf
        record_result("4.1", passed, f"Bounds lookup: [{result[0]:.6f}, {result[1]:.6f}]")
    except Exception as e:
        record_result("4.1", False, f"Exception: {e}")

    print_subtest("4.2 Voltage beyond bounds (clamping)")
    V_outside = torch.tensor([-200.0, -150.0, 100.0, 200.0], device=device, dtype=torch.float64)

    try:
        result = lut.lookup('m_inf', V_outside)
        no_nan = not torch.isnan(result).any()
        no_inf = not torch.isinf(result).any()
        in_range = (result >= 0).all() and (result <= 1).all()
        passed = no_nan and no_inf and in_range
        record_result("4.2", passed, f"Clamped values in [0,1]: {result.tolist()}")
    except Exception as e:
        record_result("4.2", False, f"Exception: {e}")

    print_subtest("4.3 Empty tensor")
    V_empty = torch.tensor([], device=device, dtype=torch.float64)
    try:
        result = lut.lookup('m_inf', V_empty)
        passed = result.numel() == 0
        record_result("4.3", passed, "Empty input returns empty output")
    except Exception as e:
        record_result("4.3", False, f"Exception: {e}")

    print_subtest("4.4 Single value")
    V_single = torch.tensor(-65.0, device=device, dtype=torch.float64)
    try:
        result = lut.lookup('m_inf', V_single)
        passed = result.numel() == 1
        record_result("4.4", passed, f"Single value: m_inf(-65mV) = {result.item():.6f}")
    except Exception as e:
        record_result("4.4", False, f"Exception: {e}")

    print_subtest("4.5 Multi-dimensional input")
    V_2d = torch.randn(10, 10, device=device, dtype=torch.float64) * 30 - 40
    try:
        result = lut.lookup('m_inf', V_2d)
        passed = result.shape == V_2d.shape
        record_result("4.5", passed, f"2D input shape {V_2d.shape} -> output {result.shape}")
    except Exception as e:
        record_result("4.5", False, f"Exception: {e}")


# =============================================================================
# Test 5: Performance Benchmarks
# =============================================================================

def test_performance():
    """Benchmark LUT performance vs direct computation."""
    print_header("Test 5: Performance Benchmarks")

    from ionic.lut import TTP06LUT
    from ionic.ttp06.gating import (
        INa_m_inf, INa_m_tau, INa_h_inf, INa_h_tau,
        ICaL_d_inf, ICaL_d_tau, IKr_Xr1_inf
    )

    device = get_device()
    lut = TTP06LUT(device=device)

    # Test with increasing batch sizes
    batch_sizes = [100, 1000, 10000, 100000]

    print_subtest("5.1 Single function lookup timing")

    for n in batch_sizes:
        V = torch.randn(n, device=device, dtype=torch.float64) * 30 - 40

        # Warmup
        for _ in range(5):
            _ = lut.lookup('m_inf', V)
            _ = INa_m_inf(V)

        if device == 'cuda':
            torch.cuda.synchronize()

        # Time LUT
        t0 = time.perf_counter()
        for _ in range(100):
            _ = lut.lookup('m_inf', V)
        if device == 'cuda':
            torch.cuda.synchronize()
        lut_time = (time.perf_counter() - t0) / 100

        # Time direct
        t0 = time.perf_counter()
        for _ in range(100):
            _ = INa_m_inf(V)
        if device == 'cuda':
            torch.cuda.synchronize()
        direct_time = (time.perf_counter() - t0) / 100

        speedup = direct_time / lut_time if lut_time > 0 else 0
        passed = True  # Just informational
        record_result(f"5.1.n{n}", passed,
                     f"n={n:6d}: LUT {lut_time*1e6:7.1f}µs, Direct {direct_time*1e6:7.1f}µs, "
                     f"Speedup {speedup:.2f}x")

    print_subtest("5.2 Batch gating lookup timing")

    for n in batch_sizes:
        V = torch.randn(n, device=device, dtype=torch.float64) * 30 - 40

        if device == 'cuda':
            torch.cuda.synchronize()

        # Time LUT batch
        t0 = time.perf_counter()
        for _ in range(100):
            _ = lut.get_all_gating(V, celltype_is_endo=True)
        if device == 'cuda':
            torch.cuda.synchronize()
        lut_time = (time.perf_counter() - t0) / 100

        # Time direct (all 11 voltage-dependent gating pairs)
        t0 = time.perf_counter()
        for _ in range(100):
            _ = INa_m_inf(V)
            _ = INa_m_tau(V)
            _ = INa_h_inf(V)
            _ = INa_h_tau(V)
            _ = ICaL_d_inf(V)
            _ = ICaL_d_tau(V)
            _ = IKr_Xr1_inf(V)
            # ... simplified for benchmark
        if device == 'cuda':
            torch.cuda.synchronize()
        direct_time = (time.perf_counter() - t0) / 100

        speedup = direct_time / lut_time if lut_time > 0 else 0
        record_result(f"5.2.n{n}", True,
                     f"n={n:6d}: LUT {lut_time*1e3:6.2f}ms, Direct {direct_time*1e3:6.2f}ms, "
                     f"Speedup {speedup:.2f}x")


# =============================================================================
# Test 6: Single-Cell AP Validation
# =============================================================================

def test_single_cell_ap():
    """Compare single-cell AP with and without LUT."""
    print_header("Test 6: Single-Cell AP Validation")

    from ionic import TTP06Model, CellType

    device = get_device()

    print_subtest("6.1 AP morphology comparison (EPI cell)")

    # Create models
    model_direct = TTP06Model(celltype=CellType.EPI, device=device, use_lut=False)
    model_lut = TTP06Model(celltype=CellType.EPI, device=device, use_lut=True)

    # Initialize states
    state_direct = model_direct.get_initial_state()
    state_lut = model_lut.get_initial_state()

    dt = 0.01  # 0.01 ms for accuracy
    t_end = 400  # 400 ms

    V_direct = []
    V_lut = []
    times = []

    for step in range(int(t_end / dt)):
        t = step * dt

        # Stimulus at 10 ms
        I_stim = torch.tensor(-52.0, device=device) if 10.0 <= t < 11.0 else None

        state_direct = model_direct.step(state_direct, dt, I_stim)
        state_lut = model_lut.step(state_lut, dt, I_stim)

        # Record every 1 ms
        if step % 100 == 0:
            times.append(t)
            V_direct.append(model_direct.get_voltage(state_direct).item())
            V_lut.append(model_lut.get_voltage(state_lut).item())

    V_direct = np.array(V_direct)
    V_lut = np.array(V_lut)

    # Compare metrics
    V_peak_direct = V_direct.max()
    V_peak_lut = V_lut.max()
    V_rest_direct = V_direct[-1]
    V_rest_lut = V_lut[-1]
    max_diff = np.max(np.abs(V_direct - V_lut))

    record_result("6.1a", abs(V_peak_direct - V_peak_lut) < 1.0,
                 f"Peak V: Direct={V_peak_direct:.1f}mV, LUT={V_peak_lut:.1f}mV")
    record_result("6.1b", abs(V_rest_direct - V_rest_lut) < 1.0,
                 f"Rest V: Direct={V_rest_direct:.1f}mV, LUT={V_rest_lut:.1f}mV")
    record_result("6.1c", max_diff < 2.0,
                 f"Max difference: {max_diff:.3f} mV")

    # APD90 comparison
    print_subtest("6.2 APD90 comparison")

    def compute_apd90(V, times, threshold_pct=0.9):
        """Compute APD90."""
        V_peak = V.max()
        V_rest = V[-1]
        V_90 = V_rest + (1 - threshold_pct) * (V_peak - V_rest)

        # Find upstroke (first crossing above V_90)
        above = V > V_90
        upstroke_idx = np.argmax(above)

        # Find repolarization (last crossing above V_90)
        repol_idx = len(V) - 1 - np.argmax(above[::-1])

        return times[repol_idx] - times[upstroke_idx]

    times_arr = np.array(times)
    apd90_direct = compute_apd90(V_direct, times_arr)
    apd90_lut = compute_apd90(V_lut, times_arr)

    apd_diff = abs(apd90_direct - apd90_lut)
    record_result("6.2", apd_diff < 5.0,
                 f"APD90: Direct={apd90_direct:.1f}ms, LUT={apd90_lut:.1f}ms, diff={apd_diff:.1f}ms")

    print_subtest("6.3 ENDO cell type")
    model_endo_lut = TTP06Model(celltype=CellType.ENDO, device=device, use_lut=True)
    state_endo = model_endo_lut.get_initial_state()

    # Quick simulation
    for step in range(20000):  # 200 ms
        t = step * dt
        I_stim = torch.tensor(-52.0, device=device) if 10.0 <= t < 11.0 else None
        state_endo = model_endo_lut.step(state_endo, dt, I_stim)

    V_peak_endo = model_endo_lut.get_voltage(state_endo).item()
    # Check it returns to resting
    record_result("6.3", V_peak_endo < 0,
                 f"ENDO cell runs without errors, V_final={V_peak_endo:.1f}mV")


# =============================================================================
# Test 7: Memory Usage
# =============================================================================

def test_memory_usage():
    """Analyze LUT memory footprint."""
    print_header("Test 7: Memory Usage Analysis")

    from ionic.lut import TTP06LUT, LUTConfig

    device = get_device()

    print_subtest("7.1 Default LUT memory")
    lut = TTP06LUT(device=device)

    n_tables = len(lut.tables)
    memory_mb = lut.memory_mb
    memory_bytes = lut.memory_bytes

    record_result("7.1a", True, f"Number of tables: {n_tables}")
    record_result("7.1b", True, f"Total memory: {memory_mb:.3f} MB ({memory_bytes:,} bytes)")
    record_result("7.1c", memory_mb < 50, f"Memory under 50MB limit: {memory_mb:.3f} MB")

    print_subtest("7.2 Per-table breakdown")

    table_sizes = {name: t.element_size() * t.numel() for name, t in lut.tables.items()}
    per_table_kb = table_sizes[list(table_sizes.keys())[0]] / 1024

    record_result("7.2", True, f"Each table: {per_table_kb:.2f} KB (2001 × float64)")

    print_subtest("7.3 Memory vs resolution tradeoff")

    for n_points in [501, 1001, 2001, 4001]:
        config = LUTConfig(n_points=n_points, device=torch.device(device))
        # Create minimal LUT to estimate
        bytes_per_table = n_points * 8  # float64
        estimated_mb = (n_tables * bytes_per_table) / (1024 * 1024)
        resolution = 180.0 / (n_points - 1)

        record_result(f"7.3.n{n_points}", True,
                     f"n={n_points}: ~{estimated_mb:.2f} MB, resolution={resolution:.3f} mV")


# =============================================================================
# Main
# =============================================================================

def print_summary():
    """Print test summary."""
    print("\n" + "=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)

    n_passed = sum(1 for _, passed, _ in test_results if passed)
    n_total = len(test_results)

    # Group by test category
    categories = {}
    for name, passed, msg in test_results:
        cat = name.split('.')[0]
        if cat not in categories:
            categories[cat] = {'passed': 0, 'total': 0}
        categories[cat]['total'] += 1
        if passed:
            categories[cat]['passed'] += 1

    for cat in sorted(categories.keys()):
        info = categories[cat]
        status = "✓" if info['passed'] == info['total'] else "✗"
        print(f"  {status} Category {cat}: {info['passed']}/{info['total']} passed")

    print("-" * 70)
    all_passed = n_passed == n_total
    print(f"  {'✓' if all_passed else '✗'} Total: {n_passed}/{n_total} tests passed")
    print("=" * 70)

    return 0 if all_passed else 1


def main():
    print("=" * 70)
    print("  Comprehensive LUT Testing Suite for TTP06 Model")
    print("  Engine V5.3")
    print("=" * 70)

    test_lut_construction()
    test_interpolation_accuracy()
    test_batch_lookup()
    test_edge_cases()
    test_performance()
    test_single_cell_ap()
    test_memory_usage()

    return print_summary()


if __name__ == '__main__':
    sys.exit(main())
