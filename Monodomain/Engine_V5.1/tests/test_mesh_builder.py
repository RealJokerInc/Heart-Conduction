"""
Test Mesh Builder and CV Tuning

Tests:
1. Default mesh configuration (15x15 cm, dx=0.02, dt=0.02)
2. 1D Cable CV tuning for longitudinal and transverse velocities
3. Full 2D mesh with tuned D values
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from tissue import (
    MeshBuilder, MeshConfig,
    CableMesh, CableConfig,
    create_default_mesh, create_cv_tuning_cable
)


def test_default_mesh():
    """Test default mesh configuration."""
    print("\n" + "=" * 60)
    print("TEST 1: Default Mesh Configuration")
    print("=" * 60)

    mesh = create_default_mesh()
    mesh.print_summary()

    cfg = mesh.get_config()

    # Verify defaults
    assert cfg.Lx == 15.0, f"Expected Lx=15.0, got {cfg.Lx}"
    assert cfg.Ly == 15.0, f"Expected Ly=15.0, got {cfg.Ly}"
    assert cfg.dx == 0.02, f"Expected dx=0.02, got {cfg.dx}"
    assert cfg.dy == 0.02, f"Expected dy=0.02, got {cfg.dy}"
    assert cfg.dt == 0.02, f"Expected dt=0.02, got {cfg.dt}"
    assert cfg.nx == 750, f"Expected nx=750, got {cfg.nx}"
    assert cfg.ny == 750, f"Expected ny=750, got {cfg.ny}"

    print("\n✓ Default mesh configuration verified!")
    print(f"  Domain: {cfg.Lx} x {cfg.Ly} cm")
    print(f"  Grid: {cfg.nx} x {cfg.ny} cells")
    print(f"  dx={cfg.dx}, dt={cfg.dt}")
    print(f"  D_L={cfg.D_L:.6f}, D_T={cfg.D_T:.6f}")


def test_cable_cv_measurement():
    """Test CV measurement on 1D cable."""
    print("\n" + "=" * 60)
    print("TEST 2: 1D Cable CV Measurement")
    print("=" * 60)

    # Create cable for CV = 0.06 cm/ms (0.6 m/s)
    target_cv = 0.06
    cable = create_cv_tuning_cable(target_cv=target_cv, dx=0.02, dt=0.02)
    cable.print_summary()

    print("\nRunning propagation simulation...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Measure CV
    cv_measured = cable.measure_cv(device=device)

    print(f"\nResults:")
    print(f"  Target CV: {target_cv*10:.3f} m/s ({target_cv:.5f} cm/ms)")
    print(f"  Measured CV: {cv_measured*10:.3f} m/s ({cv_measured:.5f} cm/ms)")
    print(f"  Error: {100*(cv_measured-target_cv)/target_cv:.1f}%")

    return cv_measured


def test_cv_tuning():
    """Test iterative CV tuning on 1D cable."""
    print("\n" + "=" * 60)
    print("TEST 3: CV Tuning (Longitudinal)")
    print("=" * 60)

    target_cv = 0.06  # 0.6 m/s
    cable = create_cv_tuning_cable(target_cv=target_cv, dx=0.02, dt=0.02)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    D_tuned = cable.tune_D_to_cv(target_cv=target_cv, device=device, verbose=True)

    # Verify tuned value
    cv_final = cable.measure_cv(D=D_tuned, device=device)
    error = abs(cv_final - target_cv) / target_cv

    print(f"\nTuning Results:")
    print(f"  Tuned D = {D_tuned:.6f} cm²/ms")
    print(f"  Final CV = {cv_final*10:.4f} m/s")
    print(f"  Error = {error*100:.2f}%")

    assert error < 0.05, f"CV tuning error {error*100:.1f}% exceeds 5% tolerance"
    print("\n✓ CV tuning successful!")

    return D_tuned


def test_cv_tuning_transverse():
    """Test CV tuning for transverse direction."""
    print("\n" + "=" * 60)
    print("TEST 4: CV Tuning (Transverse)")
    print("=" * 60)

    target_cv = 0.03  # 0.3 m/s
    cable = create_cv_tuning_cable(target_cv=target_cv, dx=0.02, dt=0.02)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    D_tuned = cable.tune_D_to_cv(target_cv=target_cv, device=device, verbose=True)

    # Verify tuned value
    cv_final = cable.measure_cv(D=D_tuned, device=device)
    error = abs(cv_final - target_cv) / target_cv

    print(f"\nTuning Results:")
    print(f"  Tuned D = {D_tuned:.6f} cm²/ms")
    print(f"  Final CV = {cv_final*10:.4f} m/s")
    print(f"  Error = {error*100:.2f}%")

    return D_tuned


def test_full_mesh_with_tuning():
    """Test full mesh creation with CV tuning."""
    print("\n" + "=" * 60)
    print("TEST 5: Full 2D Mesh with CV Tuning")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mesh = MeshBuilder.create_with_tuned_cv(
        cv_long=0.06,   # 0.6 m/s
        cv_trans=0.03,  # 0.3 m/s
        dx=0.02,
        dt=0.02,
        tune_verbose=True,
        device=device
    )

    mesh.print_summary()

    cfg = mesh.get_config()
    print(f"\nFinal tuned values:")
    print(f"  D_L = {cfg.D_L:.6f} cm²/ms (for CV = 0.6 m/s)")
    print(f"  D_T = {cfg.D_T:.6f} cm²/ms (for CV = 0.3 m/s)")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("MESH BUILDER AND CV TUNING TESTS")
    print("=" * 60)

    # Test 1: Default mesh
    test_default_mesh()

    # Test 2: Cable CV measurement
    test_cable_cv_measurement()

    # Test 3: CV tuning (longitudinal)
    D_L = test_cv_tuning()

    # Test 4: CV tuning (transverse)
    D_T = test_cv_tuning_transverse()

    # Summary
    print("\n" + "=" * 60)
    print("FINAL TUNED DIFFUSION COEFFICIENTS")
    print("=" * 60)
    print(f"  D_L = {D_L:.6f} cm²/ms (for CV_long = 0.6 m/s)")
    print(f"  D_T = {D_T:.6f} cm²/ms (for CV_trans = 0.3 m/s)")
    print(f"  Anisotropy ratio (D_L/D_T): {D_L/D_T:.2f}")
    print("=" * 60)

    print("\n✓ All tests passed!")


if __name__ == '__main__':
    main()
