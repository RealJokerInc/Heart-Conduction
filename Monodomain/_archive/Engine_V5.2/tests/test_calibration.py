"""
Test calibration module components.

Tests:
1. Cable1D - Basic propagation and CV measurement
2. ERP measurement - S1-S2 protocol
3. Optimizer - Quick calibration test
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np


def test_cable_cv():
    """Test 1D cable CV measurement."""
    print("=" * 60)
    print("Test 1: Cable1D CV Measurement")
    print("=" * 60)

    from calibration.cable_1d import measure_cv_apd

    # Test with known D value
    D = 0.001  # cm²/ms

    print(f"\nRunning cable simulation with D = {D} cm²/ms...")
    result = measure_cv_apd(
        D=D,
        dx=0.02,        # Coarser for faster test
        dt=0.02,
        cable_length=2.0,
        duration=300.0,
        cell_type=0     # ENDO
    )

    print(f"\nResults:")
    print(f"  CV = {result.cv:.4f} cm/ms = {result.cv * 10:.1f} m/s")
    print(f"  APD90 = {result.apd90:.1f} ms")
    print(f"  APD50 = {result.apd50:.1f} ms")
    print(f"  V_max = {result.v_max:.1f} mV")
    print(f"  dV/dt_max = {result.dv_dt_max:.1f} mV/ms")
    print(f"  Success = {result.success}")

    # Basic validation
    assert result.success, "Propagation should succeed"
    assert result.cv > 0, "CV should be positive"
    assert result.apd90 > 200, "APD90 should be > 200 ms for ORd model"
    assert result.v_max > 20, "V_max should be > 20 mV"

    # CV should scale with sqrt(D)
    # For D=0.001, CV should be roughly 0.04-0.07 cm/ms
    assert 0.02 < result.cv < 0.1, f"CV = {result.cv} out of expected range"

    print("\n✓ Test passed!")
    return result


def test_cv_scaling():
    """Test that CV scales with sqrt(D)."""
    print("\n" + "=" * 60)
    print("Test 2: CV Scaling with D")
    print("=" * 60)

    from calibration.cable_1d import measure_cv_apd

    D_values = [0.0005, 0.001, 0.002]
    cv_values = []

    for D in D_values:
        print(f"\nD = {D} cm²/ms...")
        result = measure_cv_apd(D=D, dx=0.02, dt=0.02, duration=300.0)
        cv_values.append(result.cv)
        print(f"  CV = {result.cv:.4f} cm/ms")

    # Check scaling: CV_2 / CV_1 ≈ sqrt(D_2 / D_1)
    ratio_cv = cv_values[1] / cv_values[0]
    ratio_expected = np.sqrt(D_values[1] / D_values[0])

    print(f"\nCV scaling check:")
    print(f"  CV ratio (D2/D1): {ratio_cv:.3f}")
    print(f"  Expected (sqrt): {ratio_expected:.3f}")
    print(f"  Error: {abs(ratio_cv - ratio_expected) / ratio_expected * 100:.1f}%")

    # Allow 20% error due to discretization
    assert abs(ratio_cv - ratio_expected) / ratio_expected < 0.20, "CV scaling error too large"

    print("\n✓ Test passed!")


def test_single_cell_erp():
    """Test single-cell ERP measurement."""
    print("\n" + "=" * 60)
    print("Test 3: Single-Cell ERP")
    print("=" * 60)

    from calibration.erp_measurement import measure_erp_single_cell

    print("\nRunning S1-S2 protocol (single cell)...")
    print("(This may take a minute...)")

    result = measure_erp_single_cell(
        dt=0.02,
        cell_type=0,
        s1_bcl=1000.0,
        s1_count=5,     # Reduced for speed
        verbose=True
    )

    print(f"\nResults:")
    print(f"  ERP = {result.erp:.1f} ms")
    print(f"  APD90 = {result.apd90:.1f} ms")
    print(f"  ERP/APD = {result.erp / result.apd90:.2f}")

    # ERP should be slightly less than APD for single cell
    assert result.erp > 200, "ERP should be > 200 ms"
    assert result.erp < result.apd90 + 50, "ERP should be close to APD"

    print("\n✓ Test passed!")
    return result


def test_quick_calibration():
    """Quick test of the calibration optimizer."""
    print("\n" + "=" * 60)
    print("Test 4: Quick Calibration (Reduced Iterations)")
    print("=" * 60)

    from calibration.optimizer import (
        DiffusionCalibrator,
        CalibrationTargets,
        CalibrationWeights,
        CalibrationConfig
    )

    # Configure for quick test
    targets = CalibrationTargets(
        cv_longitudinal=0.05,     # Lower target for quick convergence
        anisotropy_ratio=2.0,     # Lower ratio
        erp_tissue_target=300.0,
        apd_min=250.0
    )

    weights = CalibrationWeights()

    config = CalibrationConfig(
        dx=0.02,          # Coarser mesh
        dt=0.02,
        cable_length=1.5, # Shorter cable
        de_maxiter=5,     # Very few iterations for quick test
        de_popsize=5,     # Small population
        s1_count=3,       # Fewer S1 beats
        cv_sim_duration=200.0  # Shorter simulation
    )

    calibrator = DiffusionCalibrator(targets, weights, config)

    print("\nRunning quick calibration (5 iterations max)...")
    print("(This tests the optimization loop, not accuracy)")

    result = calibrator.calibrate(verbose=True)

    print(f"\nOptimized Values:")
    print(f"  D_L = {result.D_longitudinal:.6f} cm²/ms")
    print(f"  D_T = {result.D_transverse:.6f} cm²/ms")
    print(f"  CV_L = {result.cv_longitudinal:.4f} cm/ms")

    # Just verify it ran without error
    assert result.D_longitudinal > 0, "D_L should be positive"
    assert result.D_transverse > 0, "D_T should be positive"
    assert result.n_evaluations > 0, "Should have function evaluations"

    print("\n✓ Test passed!")
    return result


def run_all_tests():
    """Run all calibration tests."""
    print("\n" + "#" * 60)
    print("# Engine V5.2 Calibration Module Tests")
    print("#" * 60)

    tests = [
        ("Cable CV", test_cable_cv),
        ("CV Scaling", test_cv_scaling),
        # ("Single-Cell ERP", test_single_cell_erp),  # Uncomment for full test
        # ("Quick Calibration", test_quick_calibration),  # Uncomment for full test
    ]

    results = []
    for name, test_fn in tests:
        try:
            test_fn()
            results.append((name, "PASSED"))
        except Exception as e:
            results.append((name, f"FAILED: {e}"))
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, status in results:
        print(f"  {name}: {status}")

    n_passed = sum(1 for _, s in results if s == "PASSED")
    print(f"\n{n_passed}/{len(results)} tests passed")


if __name__ == "__main__":
    run_all_tests()
